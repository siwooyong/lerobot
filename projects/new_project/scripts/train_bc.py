"""Launch LeRobot BC training on raw RoboCasa v2.1 datasets."""

from __future__ import annotations

from typing import Any

import torch

from projects.new_project.dataset.robocasa import RoboCasaDataset


def _validate_config(cfg: Any) -> None:
    if not cfg.dataset.root:
        raise ValueError("Set --dataset.root to the directory containing RoboCasa task datasets.")
    if not isinstance(cfg.dataset.repo_id, str):
        raise ValueError("Raw RoboCasa training accepts one logical --dataset.repo_id string.")
    if cfg.dataset.streaming:
        raise ValueError("Raw RoboCasa is a map-style dataset; set --dataset.streaming=false.")
    if cfg.dataset.eval_split != 0.0:
        raise ValueError("This launcher does not create an eval split; set --dataset.eval_split=0.")
    if cfg.dataset.episodes is not None:
        raise ValueError("Episode filtering is not part of this minimal multi-task launcher.")


def make_robocasa_train_eval_datasets(cfg: Any) -> tuple[RoboCasaDataset, None]:
    """Create the project-local raw reader used by LeRobot's standard trainer."""
    _validate_config(cfg)

    from lerobot.transforms import ImageTransforms
    from lerobot.utils.constants import IMAGENET_STATS

    transforms_cfg = cfg.dataset.image_transforms
    image_transforms = ImageTransforms(transforms_cfg) if transforms_cfg.enable else None
    trainable_config = cfg.trainable_config
    dataset = RoboCasaDataset(
        root=cfg.dataset.root,
        action_delta_indices=trainable_config.action_delta_indices,
        observation_delta_indices=trainable_config.observation_delta_indices,
        image_transforms=image_transforms,
        video_backend=cfg.dataset.video_backend or "torchcodec",
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.dataset.use_imagenet_stats:
        for camera_key in dataset.meta.camera_keys:
            dataset.meta.stats[camera_key] = {
                name: torch.tensor(value, dtype=torch.float32) for name, value in IMAGENET_STATS.items()
            }
    return dataset, None


def run() -> None:
    """Patch only the dataset factory, then delegate to LeRobot's official trainer."""
    from lerobot.scripts import lerobot_train

    lerobot_train.make_train_eval_datasets = make_robocasa_train_eval_datasets
    lerobot_train.main()


if __name__ == "__main__":
    run()
