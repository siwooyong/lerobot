"""Flow-Noise policy adapter for SmolVLA's action expert."""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def trainable_parameters(module: nn.Module):
    """Yield exactly the parameters enabled by LeRobot's policy configuration."""
    return (parameter for parameter in module.parameters() if parameter.requires_grad)


def configure_smolvla_trainability(policy: nn.Module) -> None:
    """Apply LeRobot's SmolVLA freeze configuration, including later unfreezing."""
    model = policy.model
    expert = model.vlm_with_expert
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    expert.train_expert_only = policy.config.train_expert_only
    expert.freeze_vision_encoder = policy.config.freeze_vision_encoder
    expert.set_requires_grad()
    model.set_requires_grad()


class FlowSample(NamedTuple):
    actions: Tensor
    path: Tensor
    log_prob: Tensor
    critic_features: Tensor
    noise_std: Tensor


def executed_log_prob(log_prob: Tensor, steps: Tensor) -> Tensor:
    """Sum each worker's joint log-probability over actions it actually executed."""
    steps = steps.to(log_prob.device)
    mask = torch.arange(log_prob.shape[1], device=log_prob.device) < steps[:, None]
    return (log_prob * mask).sum(1)


def executed_mean(values: Tensor, steps: Tensor) -> Tensor:
    """Average per-action values over the valid prefix of an action chunk."""
    steps = steps.to(values.device)
    mask = torch.arange(values.shape[1], device=values.device) < steps[:, None]
    denominator = steps.clamp_min(1).to(values.dtype)
    return (values * mask).sum(1) / denominator


class FlowNoiseHead(nn.Module):
    """RLinf Flow-Noise log-variance head, evaluated entirely in float32."""

    def __init__(self, feature_dim: int, action_dim: int, min_std: float = 0.08, max_std: float = 0.16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.register_buffer("logvar_min", torch.log(torch.tensor(min_std**2, dtype=torch.float32)))
        self.register_buffer("logvar_max", torch.log(torch.tensor(max_std**2, dtype=torch.float32)))

    def forward(self, suffix_out: Tensor) -> Tensor:
        noise_feature = suffix_out.float()
        # The action expert runs under bfloat16 autocast, but RLinf's noise head
        # consumes float32 suffix features and preserves float32 log-variance.
        with torch.autocast(device_type=noise_feature.device.type, enabled=False):
            noise_logvar = torch.tanh(self.net(noise_feature))
            noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2
            return torch.exp(0.5 * noise_logvar)


class FlowNoiseActor(nn.Module):
    """PPO policy over the executed prefix of a SmolVLA action chunk."""

    def __init__(self, policy: nn.Module, action_steps: int):
        super().__init__()
        self.policy, self.model = policy, policy.model
        self.action_steps = action_steps
        self.chunk_size = policy.config.chunk_size
        self.action_dim = policy.config.action_feature.shape[0]
        self.latent_dim = policy.config.max_action_dim
        self.num_steps = policy.config.num_steps
        self.noise = FlowNoiseHead(self.model.vlm_with_expert.expert_hidden_size, self.latent_dim)
        configure_smolvla_trainability(policy)

    def _prefix(self, batch: dict[str, Tensor]) -> tuple[Tensor, object, Tensor]:
        batch = self.policy._prepare_batch(dict(batch))
        images, image_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        tokens, token_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        embeds, masks, ar = self.model.embed_prefix(images, image_masks, tokens, token_masks, state=state)
        cumulative = torch.cumsum(ar, 1)
        attention = (cumulative[:, None, :] <= cumulative[:, :, None]) & (masks[:, None, :] * masks[:, :, None]).bool()
        positions = torch.cumsum(masks, 1) - 1
        output, cache = self.model.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeds, None],
            use_cache=self.model.config.use_cache,
            fill_kv_cache=True,
        )
        context = (output[0] * masks.unsqueeze(-1)).sum(1) / masks.sum(1, keepdim=True).clamp_min(1)
        return masks, cache, context

    @staticmethod
    def _log_prob(delta: Tensor, std: Tensor) -> Tensor:
        delta, std = delta.float(), std.float()
        return (-0.5 * ((delta / std).square() + 2 * std.log() + math.log(2 * math.pi))).sum(-1)

    @staticmethod
    def _aggregate_entropy(stds: Tensor) -> Tensor:
        """RLinf-style mean Gaussian entropy per action over the Flow-Noise chain."""
        entropy = 0.5 * torch.log(2 * math.pi * math.e * stds.float().square())
        return entropy.mean(dim=(1, 3))

    def _step(self, x: Tensor, masks: Tensor, cache: object, time: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        velocity, suffix_out = self.model.denoise_step(
            prefix_pad_masks=masks,
            past_key_values=cache,
            x_t=x,
            timestep=time,
            return_suffix_out=True,
        )
        mean = x + (-1 / self.num_steps) * velocity
        return mean, self.noise(suffix_out), suffix_out

    def _sample(self, masks: Tensor, cache: object, context: Tensor) -> FlowSample:
        x = torch.randn(len(context), self.chunk_size, self.latent_dim, device=context.device)
        path = [x]
        log_prob = self._log_prob(
            x[:, : self.action_steps, : self.action_dim],
            torch.ones_like(x[:, : self.action_steps, : self.action_dim]),
        )
        noise_std = torch.zeros(len(context), device=context.device)
        for step in range(self.num_steps):
            time = torch.full((len(context),), 1 - step / self.num_steps, device=context.device)
            mean, std, _ = self._step(x, masks, cache, time)
            noise_std += std[:, : self.action_steps, : self.action_dim].mean(dim=(1, 2))
            delta = std * torch.randn_like(std)
            x = mean + delta.to(mean.dtype)
            # Match PPO replay: evaluate the sampled transition as
            # log N(x_next | mean, std), rather than through the separately
            # retained random delta.
            log_prob += self._log_prob(
                x[:, : self.action_steps, : self.action_dim].float()
                - mean[:, : self.action_steps, : self.action_dim].float(),
                std[:, : self.action_steps, : self.action_dim],
            )
            path.append(x)
        return FlowSample(
            x[:, : self.action_steps, : self.action_dim],
            torch.stack(path, 1),
            log_prob / (self.num_steps + 1),
            context.detach(),
            noise_std / self.num_steps,
        )

    def _path_log_prob(
        self, masks: Tensor, cache: object, context: Tensor, path: Tensor, return_entropy: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if path.shape[1] != self.num_steps + 1:
            raise ValueError("Flow path must include its initial latent and every denoising transition.")
        total = self._log_prob(
            path[:, 0, : self.action_steps, : self.action_dim],
            torch.ones_like(path[:, 0, : self.action_steps, : self.action_dim]),
        )
        stds = [torch.ones_like(path[:, 0, : self.action_steps, : self.action_dim])]
        for step in range(self.num_steps):
            time = torch.full((path.shape[0],), 1 - step / self.num_steps, device=path.device)
            mean, std, _ = self._step(path[:, step], masks, cache, time)
            delta = (
                path[:, step + 1, : self.action_steps, : self.action_dim].float()
                - mean[:, : self.action_steps, : self.action_dim].float()
            )
            total += self._log_prob(delta, std[:, : self.action_steps, : self.action_dim])
            stds.append(std[:, : self.action_steps, : self.action_dim])
        total = total / (self.num_steps + 1)
        if return_entropy:
            return (
                total,
                self._aggregate_entropy(torch.stack(stds, 1)),
                context.detach(),
            )
        return total

    def forward(
        self,
        batch: dict[str, Tensor],
        path: Tensor | None = None,
        context_only: bool = False,
        return_entropy: bool = False,
    ):
        """Sample, score a path, or return the critic feature."""
        masks, cache, context = self._prefix(batch)
        if context_only:
            return context.detach()
        if path is None:
            return self._sample(masks, cache, context)
        return self._path_log_prob(masks, cache, context, path, return_entropy=return_entropy)
