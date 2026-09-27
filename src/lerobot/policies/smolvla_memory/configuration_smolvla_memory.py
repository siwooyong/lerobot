from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_memory")
@dataclass
class SmolVLAMemoryConfig(SmolVLAConfig):
    """SmolVLA with frame compression and RoPE-based causal memory."""

    chunk_size: int = 16
    n_action_steps: int = 8
    train_state_proj: bool = False
    frame_num_tokens: int = 1
    compressor_num_layers: int = 1
    compressor_hidden_size: int = 512
    compressor_num_heads: int = 8
    memory_num_layers: int = 2
    memory_hidden_size: int = 512
    memory_num_heads: int = 8
    sequence_length: int = 16
    base_pretrained_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "chunk_size",
            "n_action_steps",
            "frame_num_tokens",
            "compressor_num_layers",
            "compressor_hidden_size",
            "compressor_num_heads",
            "memory_num_layers",
            "memory_hidden_size",
            "memory_num_heads",
            "sequence_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for module in ("compressor", "memory"):
            if getattr(self, f"{module}_hidden_size") % getattr(self, f"{module}_num_heads"):
                raise ValueError(f"{module}_hidden_size must be divisible by {module}_num_heads.")
        if (self.memory_hidden_size // self.memory_num_heads) % 2:
            raise ValueError("Memory head dimension must be even for RoPE.")
        if self.n_obs_steps != 1:
            raise ValueError("smolvla_memory uses sequence_length for history; n_obs_steps must be 1.")
        if not self.train_expert_only:
            raise ValueError("smolvla_memory requires train_expert_only=true to freeze the VLM.")
        if self.train_state_proj:
            raise ValueError("smolvla_memory requires train_state_proj=false for frozen VLM prefill.")
        if not self.use_cache:
            raise ValueError("smolvla_memory requires use_cache=true for expert conditioning.")
        if self.rtc_config is not None:
            raise ValueError("smolvla_memory does not yet support RTC.")
