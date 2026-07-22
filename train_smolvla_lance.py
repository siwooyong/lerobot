"""Run LeRobot training with a local Lance-backed dataset.

The launcher replaces only LeRobot's dataset-construction callback. Model
creation, preprocessing, sampling, Accelerate, optimization, W&B, evaluation,
and checkpointing remain in the official ``lerobot-train`` implementation.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any


def _load_dependencies() -> SimpleNamespace:
    import torch

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.transforms import ImageTransforms
    from lerobot.utils.constants import IMAGENET_STATS
    from lerobot_lancedb import LeRobotLanceDataset

    return SimpleNamespace(
        ImageTransforms=ImageTransforms,
        IMAGENET_STATS=IMAGENET_STATS,
        LeRobotDatasetMetadata=LeRobotDatasetMetadata,
        LeRobotLanceDataset=LeRobotLanceDataset,
        as_tensor=torch.tensor,
        resolve_delta_timestamps=resolve_delta_timestamps,
    )


def _validate_dataset_config(cfg: Any) -> None:
    dataset_cfg = cfg.dataset
    if not isinstance(dataset_cfg.repo_id, str):
        raise ValueError("Lance launcher currently supports one dataset.repo_id only.")
    if not dataset_cfg.root:
        raise ValueError(
            "Set --dataset.root to the converted directory containing meta/ and one *.lance table."
        )
    if getattr(dataset_cfg, "streaming", False):
        raise ValueError("Lance and --dataset.streaming=true cannot be enabled together.")
    if getattr(dataset_cfg, "eval_split", 0.0) != 0.0:
        raise ValueError("Lance launcher currently requires --dataset.eval_split=0.")


def make_lance_train_eval_datasets(cfg: Any, *, dependencies: Any | None = None):
    """Build the Lance training dataset expected by LeRobot's training loop."""
    _validate_dataset_config(cfg)
    deps = dependencies or _load_dependencies()
    dataset_cfg = cfg.dataset

    transforms_cfg = dataset_cfg.image_transforms
    image_transforms = deps.ImageTransforms(transforms_cfg) if transforms_cfg.enable else None
    metadata = deps.LeRobotDatasetMetadata(
        dataset_cfg.repo_id,
        root=dataset_cfg.root,
        revision=getattr(dataset_cfg, "revision", None),
    )
    trainable_config = getattr(cfg, "trainable_config", None) or cfg.policy
    delta_timestamps = deps.resolve_delta_timestamps(trainable_config, metadata)
    decode_device = os.environ.get("LEROBOT_LANCE_DECODE_DEVICE", "cpu").lower()
    if decode_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("LEROBOT_LANCE_DECODE_DEVICE must be cpu, cuda, or auto.")

    dataset = deps.LeRobotLanceDataset(
        root=dataset_cfg.root,
        repo_id=dataset_cfg.repo_id,
        episodes=getattr(dataset_cfg, "episodes", None),
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=getattr(dataset_cfg, "revision", None),
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        decode_device=decode_device,
    )

    if getattr(dataset_cfg, "use_imagenet_stats", True):
        for key in dataset.meta.camera_keys:
            if key in dataset.meta.depth_keys:
                continue
            for stats_type, stats in deps.IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = deps.as_tensor(stats)

    # Some LeRobot releases pass this optional mapping to EpisodeAwareSampler.
    if not hasattr(dataset, "absolute_to_relative_idx"):
        dataset.absolute_to_relative_idx = None

    return dataset, None


def run(*, trainer_module: Any | None = None, dependencies: Any | None = None) -> None:
    """Patch the dataset callback and delegate to LeRobot's official CLI."""
    if trainer_module is None:
        from lerobot.scripts import lerobot_train as trainer_module

    if hasattr(trainer_module, "make_train_eval_datasets"):
        if dependencies is None:
            trainer_module.make_train_eval_datasets = make_lance_train_eval_datasets
        else:
            trainer_module.make_train_eval_datasets = lambda cfg: make_lance_train_eval_datasets(
                cfg, dependencies=dependencies
            )
    elif hasattr(trainer_module, "make_dataset"):
        trainer_module.make_dataset = lambda cfg: make_lance_train_eval_datasets(
            cfg, dependencies=dependencies
        )[0]
    else:
        raise RuntimeError("Unsupported LeRobot version: no dataset factory found in lerobot_train.")

    trainer_module.main()


if __name__ == "__main__":
    run()