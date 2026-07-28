#!/usr/bin/env python
"""Collect N successful SmolVLA rollouts per LIBERO task.

The environment, policy, processor, and rollout engines come from LeRobot.
Only successful episodes are committed to an image-based LeRobotDataset whose
user-defined features match ``HuggingFaceVLA/libero``.
"""

import json
import logging
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from projects.robot_learning.dataset.success_data import (
    IMAGE_SHAPE,
    SuccessfulEpisodeRecorder,
    collect_task,
    libero_dataset_features,
)


@dataclass
class SuccessDatasetConfig:
    repo_id: str = "siwooyong/libero-smolvla-success-rollouts"
    root: Path = Path("outputs/datasets/libero_success_rollouts")
    successes_per_task: int = 10
    max_attempts_per_task: int = 1000
    resume: bool = True
    push_to_hub: bool = False
    private: bool = False
    image_writer_threads: int = 4
    fps: int = 10


@dataclass
class SuccessCollectionPipelineConfig(EvalPipelineConfig):
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            n_episodes=1,
            batch_size=1,
            use_async_envs=False,
        )
    )
    collection: SuccessDatasetConfig = field(default_factory=SuccessDatasetConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.env.type != "libero":
            raise ValueError("Success collection currently supports only --env.type=libero.")
        if self.eval.batch_size != 1:
            raise ValueError("Use --eval.batch_size=1; one writer cannot interleave multiple episodes.")
        if self.eval.use_async_envs:
            raise ValueError("Use --eval.use_async_envs=false for deterministic sequential collection.")
        if self.collection.successes_per_task <= 0:
            raise ValueError("collection.successes_per_task must be positive.")
        if self.collection.max_attempts_per_task < self.collection.successes_per_task:
            raise ValueError(
                "collection.max_attempts_per_task must be at least collection.successes_per_task."
            )
        if self.collection.fps != 10:
            raise ValueError("HuggingFaceVLA/libero compatibility requires collection.fps=10.")
        if self.env.observation_height != IMAGE_SHAPE[0] or self.env.observation_width != IMAGE_SHAPE[1]:
            raise ValueError(
                "HuggingFaceVLA/libero compatibility requires "
                "--env.observation_height=256 --env.observation_width=256."
            )
        if self.policy is None:
            raise ValueError("--policy.path must point to a trained SmolVLA policy.")
        if self.policy.type != "smolvla":
            raise ValueError("Success collection currently supports only --policy.type=smolvla.")


class JsonlManifest:
    def __init__(self, path: Path, common: dict[str, Any]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.common = common

    def append(self, record: dict[str, Any]) -> None:
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **self.common,
            **record,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def attempt_counts(self) -> Counter[tuple[str, int]]:
        counts: Counter[tuple[str, int]] = Counter()
        if not self.path.exists():
            return counts
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                counts[(str(record["suite"]), int(record["task_id"]))] += 1
        return counts


def _existing_task_counts(dataset: LeRobotDataset) -> Counter[str]:
    counts: Counter[str] = Counter()
    episodes = dataset.meta.episodes
    if episodes is None:
        return counts
    for episode in episodes:
        for task in episode.get("tasks", []):
            counts[str(task)] += 1
    return counts


def _make_output_dataset(cfg: SuccessCollectionPipelineConfig) -> LeRobotDataset:
    root = cfg.collection.root
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        if not cfg.collection.resume:
            raise FileExistsError(
                f"{root} already contains a dataset. Set --collection.resume=true or choose another root."
            )
        dataset = LeRobotDataset.resume(
            repo_id=cfg.collection.repo_id,
            root=root,
            image_writer_threads=cfg.collection.image_writer_threads,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=cfg.collection.repo_id,
            root=root,
            fps=cfg.collection.fps,
            robot_type="panda",
            features=libero_dataset_features(),
            use_videos=False,
            image_writer_threads=cfg.collection.image_writer_threads,
            metadata_buffer_size=1,
        )

    expected = libero_dataset_features()
    for key, feature in expected.items():
        if key not in dataset.features:
            raise ValueError(f"Output dataset is missing feature {key}.")
        if dataset.features[key]["dtype"] != feature["dtype"]:
            raise ValueError(
                f"{key} dtype mismatch: {dataset.features[key]['dtype']} != {feature['dtype']}."
            )
        if tuple(dataset.features[key]["shape"]) != tuple(feature["shape"]):
            raise ValueError(
                f"{key} shape mismatch: {dataset.features[key]['shape']} != {feature['shape']}."
            )
    forbidden_features = {"next.reward", "next.success", "next.done"} & set(dataset.features)
    if forbidden_features:
        raise ValueError(
            "Output dataset contains evaluation-only features that are absent from "
            f"HuggingFaceVLA/libero: {sorted(forbidden_features)}."
        )
    if dataset.meta.fps != cfg.collection.fps:
        raise ValueError(f"Output dataset FPS is {dataset.meta.fps}, expected {cfg.collection.fps}.")
    return dataset


def _task_description(env: Any) -> str:
    descriptions = list(env.call("task_description"))
    if len(descriptions) != 1 or not descriptions[0]:
        raise RuntimeError(f"Expected one LIBERO task description, got {descriptions}.")
    return str(descriptions[0])


@parser.wrap()
def collect_main(cfg: SuccessCollectionPipelineConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = JsonlManifest(
        cfg.output_dir / "collection_manifest.jsonl",
        {
            "policy_path": str(cfg.policy.pretrained_path),
            "n_action_steps": int(cfg.policy.n_action_steps),
            "num_steps": int(cfg.policy.num_steps),
            "control_mode": cfg.env.control_mode,
            "dataset_repo_id": cfg.collection.repo_id,
        },
    )
    previous_attempts = manifest.attempt_counts()

    logging.info("Creating LIBERO task environments.")
    envs = make_env(
        cfg.env,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=cfg.trust_remote_code,
    )
    logging.info("Loading policy and processor pipelines.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env,
        policy_cfg=cfg.policy,
    )

    dataset = _make_output_dataset(cfg)
    existing_counts = _existing_task_counts(dataset)
    incomplete: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    task_ordinal = 0

    try:
        autocast = (
            torch.autocast(device_type=device.type)
            if cfg.policy.use_amp
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            for suite, task_envs in envs.items():
                for task_id, env in task_envs.items():
                    task = _task_description(env)
                    already_collected = existing_counts[task]
                    remaining = max(0, cfg.collection.successes_per_task - already_collected)
                    prior_attempt_count = previous_attempts[(suite, task_id)]
                    task_seed_base = int(cfg.seed or 0) + task_ordinal * 1_000_000 + prior_attempt_count
                    task_ordinal += 1

                    logging.info(
                        "Collecting suite=%s task=%d | existing=%d target=%d remaining=%d",
                        suite,
                        task_id,
                        already_collected,
                        cfg.collection.successes_per_task,
                        remaining,
                    )
                    recorder = SuccessfulEpisodeRecorder(dataset)
                    result = collect_task(
                        suite=suite,
                        task_id=task_id,
                        task_description=task,
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        recorder=recorder,
                        successes_required=remaining,
                        max_attempts=cfg.collection.max_attempts_per_task,
                        first_seed=task_seed_base,
                        rollout_fn=rollout,
                        attempt_callback=manifest.append,
                    )
                    total_successes = already_collected + result.successes
                    task_result = {
                        **asdict(result),
                        "existing_successes": already_collected,
                        "total_successes": total_successes,
                        "target_successes": cfg.collection.successes_per_task,
                    }
                    results.append(task_result)
                    if total_successes < cfg.collection.successes_per_task:
                        incomplete.append(task_result)
    finally:
        dataset.finalize()
        close_envs(envs)

    summary = {
        "dataset_root": str(cfg.collection.root),
        "dataset_repo_id": cfg.collection.repo_id,
        "target_successes_per_task": cfg.collection.successes_per_task,
        "tasks": results,
        "incomplete_tasks": incomplete,
    }
    with (cfg.output_dir / "collection_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    if cfg.collection.push_to_hub:
        dataset.push_to_hub(private=cfg.collection.private)

    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} task(s) did not reach the requested number of successes. "
            f"See {cfg.output_dir / 'collection_summary.json'}."
        )


def main() -> None:
    init_logging()
    register_third_party_plugins()
    collect_main()


if __name__ == "__main__":
    main()
