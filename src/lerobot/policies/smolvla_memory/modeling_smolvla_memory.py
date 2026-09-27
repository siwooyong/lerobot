# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""SmolVLA with frame compression and RoPE-based causal memory."""

from pathlib import Path

import torch
from accelerate.utils import reduce
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_model
from torch import Tensor, nn
from torch.nn import functional as F

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package

from .configuration_smolvla_memory import SmolVLAMemoryConfig


def _transformer(hidden_size: int, num_heads: int, num_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        hidden_size,
        num_heads,
        dim_feedforward=hidden_size * 4,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers, norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False
    )


class FrameCompressor(nn.Module):
    """Self-attend over current VLM/state tokens and N learned frame tokens."""

    def __init__(self, input_size: int, config: SmolVLAMemoryConfig):
        super().__init__()
        self.input_proj = nn.Linear(input_size, config.compressor_hidden_size)
        self.frame_tokens = nn.Parameter(
            torch.randn(config.frame_num_tokens, config.compressor_hidden_size) * 0.02
        )
        self.transformer = _transformer(
            config.compressor_hidden_size, config.compressor_num_heads, config.compressor_num_layers
        )

    def forward(self, hidden: Tensor, token_mask: Tensor) -> Tensor:
        hidden = self.input_proj(hidden.to(self.input_proj.weight.dtype))
        frames = self.frame_tokens.to(hidden.dtype).unsqueeze(0).expand(hidden.shape[0], -1, -1)
        tokens = torch.cat([hidden, frames], dim=1)
        padding = torch.cat(
            [~token_mask.bool(), torch.zeros_like(frames[:, :, 0], dtype=torch.bool)], dim=1
        )
        return self.transformer(tokens, src_key_padding_mask=padding)[:, -frames.shape[1] :]


class TemporalMemory(nn.Module):
    """Frame-causal RoPE attention with an episode-wide streaming KV cache."""

    def __init__(self, config: SmolVLAMemoryConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.compressor_hidden_size, config.memory_hidden_size)
        self.transformer = _transformer(
            config.memory_hidden_size, config.memory_num_heads, config.memory_num_layers
        )

    def _run_layers(self, hidden: Tensor, positions: Tensor, mask: Tensor | None = None, cache=None):
        # Reuse the encoder's pre-norm, QKV, output projection and FFN weights.
        batch_size, length, hidden_size = hidden.shape
        next_cache = []
        for index, layer in enumerate(self.transformer.layers):
            attention = layer.self_attn
            shape = (batch_size, length, attention.num_heads, hidden_size // attention.num_heads)
            query, key, value = F.linear(
                layer.norm1(hidden), attention.in_proj_weight, attention.in_proj_bias
            ).chunk(3, dim=-1)
            query = apply_rope(query.reshape(shape), positions).transpose(1, 2)
            key = apply_rope(key.reshape(shape), positions).transpose(1, 2)
            value = value.reshape(shape).transpose(1, 2)
            if cache is not None:
                key = torch.cat([cache[index][0], key], dim=2)
                value = torch.cat([cache[index][1], value], dim=2)
            next_cache.append((key, value))
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
            attended = attended.transpose(1, 2).reshape(batch_size, length, hidden_size)
            hidden = hidden + layer.dropout1(attention.out_proj(attended))
            hidden = hidden + layer.dropout2(
                layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(hidden)))))
            )
        return self.transformer.norm(hidden), next_cache

    def forward(self, frames: Tensor, frame_padding: Tensor | None = None) -> Tensor:
        batch_size, length, num_tokens, _ = frames.shape
        hidden = self.input_proj(frames.to(self.input_proj.weight.dtype)).flatten(1, 2)
        frame_ids = torch.arange(length, device=frames.device).repeat_interleave(num_tokens)
        positions = frame_ids[None, :].expand(batch_size, -1)
        # SDPA's boolean mask uses True for allowed keys, including the entire current frame.
        mask = (frame_ids[:, None] >= frame_ids[None, :])[None, None, :, :]
        if frame_padding is not None:
            padding = frame_padding.to(frames.device).bool().repeat_interleave(num_tokens, dim=1)
            mask = mask & ~padding[:, None, None, :]
        hidden, _ = self._run_layers(hidden, positions, mask)
        return hidden

    @torch.no_grad()
    def step(self, frames: Tensor, position: int, cache=None):
        """Process one frame [B,N,D]; retain all past KV until policy.reset()."""
        hidden = self.input_proj(frames.to(self.input_proj.weight.dtype))
        positions = torch.full(frames.shape[:2], position, dtype=torch.long, device=frames.device)
        # Every available key is past or current; no triangular mask is needed here.
        return self._run_layers(hidden, positions, cache=cache)


class MemoryFlowMatching(VLAFlowMatching):
    """Preserve SmolVLA's layer K/V and append current causal-memory outputs."""

    def __init__(self, config: SmolVLAMemoryConfig):
        super().__init__(config)
        hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.frame_compressor = FrameCompressor(hidden_size, config)
        self.temporal_memory = TemporalMemory(config)
        self.memory_proj = nn.Linear(config.memory_hidden_size, hidden_size)
        if config.compile_model:
            # The parent already compiles this subclass's forward lazily.
            self.encode_observation = torch.compile(self.encode_observation, mode=config.compile_mode)
            self.sample_with_memory = torch.compile(
                self.sample_with_memory, mode=config.compile_mode, dynamic=True
            )

    @torch.no_grad()
    def _encode_vlm(self, images, img_masks, lang_tokens, lang_masks, state):
        # Reuse SmolVLA's image/language/state prefix and layer-wise KV prefill.
        hidden, padding, attention = super().embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        (hidden, _), cache = self.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(padding, attention),
            position_ids=padding.long().cumsum(dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[hidden, None],
            use_cache=self.config.use_cache,
        )
        return hidden, padding, cache

    def encode_observation(self, images, img_masks, lang_tokens, lang_masks, state):
        hidden, padding, cache = self._encode_vlm(images, img_masks, lang_tokens, lang_masks, state)
        return cache, padding, self.frame_compressor(hidden, padding)

    def condition_cache(self, cache, memory: Tensor, padding: Tensor):
        """Append N memory K/V per layer without changing the original prefix K/V."""
        positions = padding.long().cumsum(dim=1)[:, -memory.shape[1] :] - 1
        for index, layer in enumerate(self.vlm_with_expert.get_vlm_model().text_model.layers):
            attention = layer.self_attn
            # Keep gradients to memory despite frozen normalization/projection weights.
            normalized = layer.input_layernorm(memory).to(attention.k_proj.weight.dtype)
            shape = (*memory.shape[:2], -1, attention.head_dim)
            keys = apply_rope(attention.k_proj(normalized).view(shape), positions)
            values = attention.v_proj(normalized).view(shape)
            cache.update(keys.transpose(1, 2), values.transpose(1, 2), index)
        return cache

    def training_condition(self, frames: Tensor, sequence_padding: Tensor):
        batch_size, length = sequence_padding.shape
        frames = frames.reshape(batch_size, length, self.config.frame_num_tokens, -1)
        memory = self.memory_proj(self.temporal_memory(frames, sequence_padding))
        # Each expert receives only its own N outputs, already conditioned on the past.
        memory = memory.reshape(batch_size * length, self.config.frame_num_tokens, -1)
        memory_mask = (~sequence_padding).flatten()[:, None].expand(-1, self.config.frame_num_tokens)
        return memory, memory_mask

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions,
        noise=None, time=None, sequence_padding=None,
    ) -> Tensor:
        if sequence_padding is None:
            sequence_padding = torch.zeros(state.shape[0], 1, dtype=torch.bool, device=state.device)
        cache, padding, frames = self.encode_observation(images, img_masks, lang_tokens, lang_masks, state)
        memory, memory_mask = self.training_condition(frames, sequence_padding)
        padding = torch.cat([padding, memory_mask], dim=1)
        cache = self.condition_cache(cache, memory, padding)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prediction = self.denoise_step(padding, cache, x_t, time)
        return (prediction - (noise - actions)).square()

    def sample_with_memory(self, cache, padding: Tensor, memory: Tensor, noise=None) -> Tensor:
        memory = self.memory_proj(memory)
        padding = torch.cat([padding, torch.ones_like(memory[:, :, 0], dtype=torch.bool)], dim=1)
        cache = self.condition_cache(cache, memory, padding)
        if noise is None:
            noise = self.sample_noise(
                (padding.shape[0], self.config.chunk_size, self.config.max_action_dim), padding.device
            )
        # Current memory outputs and conditioning stay fixed throughout denoising.
        return euler_integrate(
            lambda x_t, time: self.denoise_step(padding, cache, x_t, time), noise, self.config.num_steps
        )


class SmolVLAMemoryPolicy(SmolVLAPolicy):
    config_class = SmolVLAMemoryConfig
    name = "smolvla_memory"

    def __init__(self, config: SmolVLAMemoryConfig, initialize_from_base: bool = True, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.rtc_processor = None
        self.model = MemoryFlowMatching(config)
        self.reset()
        if initialize_from_base and config.base_pretrained_path and not config.pretrained_path:
            self._load_base_weights(config.base_pretrained_path)

    def _load_base_weights(self, path: str):
        model_file = Path(path) / SAFETENSORS_SINGLE_FILE
        if not Path(path).is_dir():
            model_file = hf_hub_download(path, SAFETENSORS_SINGLE_FILE, revision=self.config.pretrained_revision)
        missing, unexpected = load_model(self, str(model_file), strict=False, device="cpu")
        added_modules = ("model.frame_compressor.", "model.temporal_memory.", "model.memory_proj.")
        invalid_missing = [key for key in missing if not key.startswith(added_modules)]
        if invalid_missing or unexpected:
            raise ValueError(
                f"Base SmolVLA checkpoint is incompatible: missing={invalid_missing}, unexpected={unexpected}"
            )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, strict: bool = True, **kwargs):
        # Restoring a memory checkpoint never fetches the base initialization checkpoint.
        return super().from_pretrained(
            pretrained_name_or_path, strict=strict, initialize_from_base=False, **kwargs
        )

    def supports_rtc(self) -> bool:
        return False

    def reset(self):
        super().reset()
        self._memory_cache = None
        self._memory_position = 0

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        images, img_masks = self.prepare_images(batch)
        cache, padding, frames = self.model.encode_observation(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch),
        )
        if self._memory_cache is not None and self._memory_cache[0][0].shape[0] != frames.shape[0]:
            raise ValueError("Streaming environment batch changed; call policy.reset() first.")
        memory, self._memory_cache = self.model.temporal_memory.step(
            frames, self._memory_position, self._memory_cache
        )
        self._memory_position += 1
        actions = self.model.sample_with_memory(cache, padding, memory, noise)
        actions = actions[:, :, : self.config.action_feature.shape[0]]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        batch = self._prepare_batch(dict(batch))
        if self.config.adapt_to_pi_aloha:
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        state = self.prepare_state(batch)
        sequence_padding = batch.get("sequence_is_pad")
        if sequence_padding is None:
            sequence_padding = torch.zeros(state.shape[0], 1, dtype=torch.bool, device=state.device)
        sequence_padding = sequence_padding.to(state.device).bool()
        batch_size, length = sequence_padding.shape
        if batch_size * length != state.shape[0] or length > self.config.sequence_length:
            raise ValueError("Expected flattened B*T observations and sequence_is_pad[B,T] within sequence_length.")
        images, img_masks = self.prepare_images(batch)
        losses = self.model(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state,
            self.prepare_action(batch), noise, time, sequence_padding,
        )[:, :, : self.config.action_feature.shape[0]]
        valid = (~sequence_padding).flatten()[:, None].expand(-1, losses.shape[1])
        if "action_is_pad" in batch:
            valid = valid & ~batch["action_is_pad"].to(state.device).bool()
        losses = losses * valid[:, :, None]
        numerator = losses.reshape(batch_size, length, -1).sum(dim=(1, 2))
        denominator = valid.reshape(batch_size, -1).sum(dim=1) * losses.shape[-1]
        per_sequence = numerator / denominator.clamp_min(1)
        if reduction == "none":
            return per_sequence, {"loss": per_sequence.mean().item()}
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        # DDP averages gradients, so normalize by the mean valid count across ranks.
        mean_count = reduce(denominator.sum().float(), reduction="mean")
        loss = numerator.sum() / mean_count.clamp_min(1e-8)
        return loss, {"loss": loss.item()}
