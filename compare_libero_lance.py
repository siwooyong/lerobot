"""Check that LIBERO's original LeRobot and Lance datasets are training-equivalent.

The script compares identical frame anchors from both datasets, including the
two camera images, robot state, 50-step action target, padding mask, language,
frame identity, episode boundaries, and effective normalization statistics.
It exits with status 1 when any checked value differs.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


IDENTITY_KEYS = {
    "episode_index",
    "frame_index",
    "index",
    "task",
    "task_index",
    "timestamp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument(
        "--original-root",
        type=Path,
        default=None,
        help="Original LeRobot dataset root. Omit to use the Hugging Face cache.",
    )
    parser.add_argument(
        "--lance-root",
        type=Path,
        default=Path("outputs/datasets/libero_lance"),
    )
    parser.add_argument("--table-name", default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--boundary-episodes",
        type=int,
        default=8,
        help="Also check the first and last anchor of this many episodes.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--report-limit", type=int, default=20)
    parser.add_argument("--decode-device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def select_training_keys(sample: Mapping[str, Any]) -> list[str]:
    """Return fields that can affect SmolVLA training or anchor alignment."""
    selected = []
    for key in sample:
        if (
            key == "action"
            or key.startswith("observation.")
            or "pad" in key.lower()
            or key in IDENTITY_KEYS
        ):
            selected.append(key)
    return sorted(selected)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def compare_values(
    original: Any,
    lance: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare scalar, string, tensor, ndarray, or list values."""
    if isinstance(original, str) or isinstance(lance, str):
        equal = type(original) is type(lance) and original == lance
        return {
            "equal": equal,
            "reason": None if equal else "value_mismatch",
            "original": original,
            "lance": lance,
        }

    if original is None or lance is None:
        equal = original is None and lance is None
        return {
            "equal": equal,
            "reason": None if equal else "value_mismatch",
            "original": original,
            "lance": lance,
        }

    original_array = _to_numpy(original)
    lance_array = _to_numpy(lance)
    result = {
        "shape_original": tuple(original_array.shape),
        "shape_lance": tuple(lance_array.shape),
        "dtype_original": str(original_array.dtype),
        "dtype_lance": str(lance_array.dtype),
    }

    if original_array.shape != lance_array.shape:
        return {**result, "equal": False, "reason": "shape_mismatch"}

    dtype_equal = original_array.dtype == lance_array.dtype
    result["dtype_equal"] = dtype_equal

    if original_array.dtype.kind in "OUS" or lance_array.dtype.kind in "OUS":
        values_equal = np.array_equal(original_array, lance_array)
        return {
            **result,
            "equal": dtype_equal and values_equal,
            "reason": None if dtype_equal and values_equal else "value_mismatch",
        }

    original_float = original_array.astype(np.float64, copy=False)
    lance_float = lance_array.astype(np.float64, copy=False)
    values_equal = np.allclose(
        original_float,
        lance_float,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
    finite = np.isfinite(original_float) & np.isfinite(lance_float)
    absolute_diff = np.abs(original_float[finite] - lance_float[finite])
    max_abs_diff = float(absolute_diff.max()) if absolute_diff.size else 0.0
    mean_abs_diff = float(absolute_diff.mean()) if absolute_diff.size else 0.0
    equal = dtype_equal and bool(values_equal)
    reason = None
    if not dtype_equal:
        reason = "dtype_mismatch"
    elif not values_equal:
        reason = "value_mismatch"

    return {
        **result,
        "equal": equal,
        "reason": reason,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def compare_samples(
    original: Mapping[str, Any],
    lance: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    original_keys = set(select_training_keys(original))
    lance_keys = set(select_training_keys(lance))
    missing_in_lance = sorted(original_keys - lance_keys)
    missing_in_original = sorted(lance_keys - original_keys)
    comparisons = {}
    mismatches = {}

    for key in sorted(original_keys & lance_keys):
        comparison = compare_values(original[key], lance[key], atol=atol, rtol=rtol)
        comparisons[key] = comparison
        if not comparison["equal"]:
            mismatches[key] = comparison

    equal = not missing_in_lance and not missing_in_original and not mismatches
    return {
        "equal": equal,
        "missing_in_lance": missing_in_lance,
        "missing_in_original": missing_in_original,
        "comparisons": comparisons,
        "mismatches": mismatches,
    }


def flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def choose_indices(
    dataset: Any,
    *,
    samples: int,
    boundary_episodes: int,
    seed: int,
) -> list[int]:
    if samples < 0 or boundary_episodes < 0:
        raise ValueError("samples and boundary_episodes must be non-negative")

    length = len(dataset)
    rng = random.Random(seed)
    random_count = min(samples, length)
    selected = set(rng.sample(range(length), random_count))

    episodes = dataset.meta.episodes
    starts = list(episodes["dataset_from_index"])
    stops = list(episodes["dataset_to_index"])
    episode_count = min(boundary_episodes, len(starts))
    if episode_count:
        episode_ids = rng.sample(range(len(starts)), episode_count)
        for episode_id in episode_ids:
            start = int(starts[episode_id])
            stop = int(stops[episode_id])
            if 0 <= start < length:
                selected.add(start)
            if 0 <= stop - 1 < length:
                selected.add(stop - 1)

    return sorted(selected)


def _make_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.utils.constants import IMAGENET_STATS
    from lerobot_lancedb import LeRobotLanceDataset

    policy_cfg = SmolVLAConfig(
        vlm_model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        load_vlm_weights=True,
        num_vlm_layers=16,
        expert_width_multiplier=0.75,
        train_expert_only=True,
        train_state_proj=True,
        freeze_vision_encoder=True,
        compile_model=False,
    )
    dataset_kwargs: dict[str, Any] = {"repo_id": args.repo_id}
    if args.original_root is not None:
        dataset_kwargs["root"] = args.original_root
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(**dataset_kwargs),
        policy=policy_cfg,
        batch_size=64,
        num_workers=0,
        seed=args.seed,
    )

    print("Creating original training-equivalent LeRobotDataset...")
    original = make_dataset(cfg)

    print("Creating training-equivalent LeRobotLanceDataset...")
    lance_meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.lance_root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, lance_meta)
    lance = LeRobotLanceDataset(
        root=args.lance_root,
        repo_id=args.repo_id,
        table_name=args.table_name,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        decode_device=args.decode_device,
    )

    # Match the effective camera statistics installed by LeRobot's dataset
    # factory and by train_smolvla_lance.py.
    for key in lance.meta.camera_keys:
        if key in lance.meta.depth_keys:
            continue
        for stats_type, stats in IMAGENET_STATS.items():
            lance.meta.stats[key][stats_type] = torch.tensor(stats)

    return original, lance


def _compare_metadata(original: Any, lance: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    scalar_fields = {
        "num_frames": (original.num_frames, lance.num_frames),
        "num_episodes": (original.num_episodes, lance.num_episodes),
    }
    if hasattr(original, "fps") and hasattr(lance, "fps"):
        scalar_fields["fps"] = (original.fps, lance.fps)

    comparisons = {
        name: compare_values(left, right, atol=atol, rtol=rtol)
        for name, (left, right) in scalar_fields.items()
    }

    for column in ("dataset_from_index", "dataset_to_index"):
        comparisons[f"episodes.{column}"] = compare_values(
            list(original.meta.episodes[column]),
            list(lance.meta.episodes[column]),
            atol=atol,
            rtol=rtol,
        )

    original_stats = flatten_mapping(original.meta.stats)
    lance_stats = flatten_mapping(lance.meta.stats)
    original_stat_keys = set(original_stats)
    lance_stat_keys = set(lance_stats)
    missing_stats_in_lance = sorted(original_stat_keys - lance_stat_keys)
    missing_stats_in_original = sorted(lance_stat_keys - original_stat_keys)
    stats_comparisons = {}
    for key in sorted(original_stat_keys & lance_stat_keys):
        stats_comparisons[key] = compare_values(
            original_stats[key],
            lance_stats[key],
            atol=atol,
            rtol=rtol,
        )

    mismatches = {key: value for key, value in comparisons.items() if not value["equal"]}
    stats_mismatches = {
        key: value for key, value in stats_comparisons.items() if not value["equal"]
    }
    equal = (
        not mismatches
        and not stats_mismatches
        and not missing_stats_in_lance
        and not missing_stats_in_original
    )
    return {
        "equal": equal,
        "comparisons": comparisons,
        "mismatches": mismatches,
        "stats_mismatches": stats_mismatches,
        "missing_stats_in_lance": missing_stats_in_lance,
        "missing_stats_in_original": missing_stats_in_original,
    }


def _format_mismatch(key: str, result: Mapping[str, Any]) -> str:
    details = [f"{key}: {result.get('reason', 'mismatch')}"]
    if "shape_original" in result:
        details.append(
            f"shape={result['shape_original']} vs {result['shape_lance']}"
        )
    if "dtype_original" in result:
        details.append(
            f"dtype={result['dtype_original']} vs {result['dtype_lance']}"
        )
    if "max_abs_diff" in result:
        details.append(f"max_abs_diff={result['max_abs_diff']:.6g}")
    if "original" in result:
        details.append(f"original={result['original']!r}")
        details.append(f"lance={result['lance']!r}")
    return " | ".join(details)


def main() -> None:
    args = parse_args()
    original, lance = _make_datasets(args)

    print(
        "\nDataset sizes | "
        f"original={len(original):,} frames/{original.num_episodes:,} episodes | "
        f"lance={len(lance):,} frames/{lance.num_episodes:,} episodes"
    )
    metadata_result = _compare_metadata(original, lance, atol=args.atol, rtol=args.rtol)

    if len(original) != len(lance):
        indices = []
    else:
        indices = choose_indices(
            original,
            samples=args.samples,
            boundary_episodes=args.boundary_episodes,
            seed=args.seed,
        )

    field_totals: dict[str, int] = defaultdict(int)
    field_failures: dict[str, int] = defaultdict(int)
    mismatch_lines = []

    if indices:
        from tqdm import tqdm

        for index in tqdm(indices, desc="Comparing anchors", unit="sample"):
            sample_result = compare_samples(
                original[index],
                lance[index],
                atol=args.atol,
                rtol=args.rtol,
            )
            for key in sample_result["comparisons"]:
                field_totals[key] += 1
            for key, mismatch in sample_result["mismatches"].items():
                field_failures[key] += 1
                mismatch_lines.append(f"index={index} | {_format_mismatch(key, mismatch)}")
            for key in sample_result["missing_in_lance"]:
                field_failures[key] += 1
                mismatch_lines.append(f"index={index} | {key}: missing_in_lance")
            for key in sample_result["missing_in_original"]:
                field_failures[key] += 1
                mismatch_lines.append(f"index={index} | {key}: missing_in_original")

    print("\nField parity")
    for key in sorted(field_totals | field_failures):
        failures = field_failures[key]
        total = field_totals[key]
        status = "PASS" if failures == 0 else "FAIL"
        print(f"  {status:4}  {key:40} mismatches={failures}/{total}")

    print("\nMetadata and effective normalization statistics")
    if metadata_result["equal"]:
        print("  PASS  frame/episode boundaries and normalization statistics match")
    else:
        for key, mismatch in metadata_result["mismatches"].items():
            print(f"  FAIL  {_format_mismatch(key, mismatch)}")
        for key, mismatch in metadata_result["stats_mismatches"].items():
            print(f"  FAIL  stats.{_format_mismatch(key, mismatch)}")
        for key in metadata_result["missing_stats_in_lance"]:
            print(f"  FAIL  stats.{key}: missing_in_lance")
        for key in metadata_result["missing_stats_in_original"]:
            print(f"  FAIL  stats.{key}: missing_in_original")

    if mismatch_lines:
        print(f"\nFirst {min(args.report_limit, len(mismatch_lines))} sample mismatches")
        for line in mismatch_lines[: args.report_limit]:
            print(f"  {line}")

    passed = (
        len(original) == len(lance)
        and metadata_result["equal"]
        and not mismatch_lines
        and bool(indices)
    )
    print(
        "\nRESULT: "
        + (
            f"PASS — {len(indices)} anchors are training-equivalent."
            if passed
            else "FAIL — original and Lance paths are not training-equivalent."
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()