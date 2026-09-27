from typing import Any

import torch

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_smolvla_memory import SmolVLAMemoryConfig


def make_smolvla_memory_pre_post_processors(
    config: SmolVLAMemoryConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_smolvla_pre_post_processors(config, dataset_stats)
