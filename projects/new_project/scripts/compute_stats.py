#!/usr/bin/env python3
"""Compute dataset-wide low-dimensional statistics for a LeRobot v3 dataset.

This script follows the same practical idea as RLDX-1's stats generation:
scan the actual LeRobot parquet data and compute statistics over the complete
training dataset.  The output, however, is written with LeRobot v3's own
statistics implementation and serializer.

By default only ``observation.state`` and ``action`` are processed. Images and
videos are intentionally excluded. Low-dimensional values are accumulated in
``float64`` to avoid catastrophic cancellation in nearly constant dimensions.

Example:
    python projects/new_project/scripts/compute_stats.py \
        --dataset-root projects/new_project/data/atomic

The generated file is:
    <dataset-root>/meta/stats.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, RunningQuantileStats
from lerobot.datasets.io_utils import load_stats, write_stats

DEFAULT_FEATURES = ("observation.state", "action")
DEFAULT_BATCH_SIZE = 65_536
STATS_DTYPE = np.float64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-dataset low-dimensional statistics from LeRobot v3 "
            "parquet files and write meta/stats.json."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset root (the directory containing meta/ and data/).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help=(
            "Low-dimensional feature keys to process. "
            "Default: observation.state action"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Arrow record-batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up an existing meta/stats.json before replacing it.",
    )
    return parser.parse_args()


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    codebase_version = str(info.get("codebase_version", ""))
    major = codebase_version.removeprefix("v").split(".", maxsplit=1)[0]
    if major != "3":
        raise ValueError(
            f"This script targets LeRobot v3 datasets, but meta/info.json reports "
            f"codebase_version={codebase_version!r}."
        )

    return info


def validate_requested_features(info: dict, features: Iterable[str]) -> dict[str, int]:
    info_features = info.get("features")
    if not isinstance(info_features, dict):
        raise ValueError("meta/info.json does not contain a valid 'features' mapping.")

    feature_dims: dict[str, int] = {}
    for key in features:
        if key not in info_features:
            raise KeyError(f"Feature {key!r} is not present in meta/info.json.")

        spec = info_features[key]
        dtype = str(spec.get("dtype", ""))
        if dtype in {"image", "video", "string", "language"}:
            raise ValueError(
                f"Feature {key!r} has dtype={dtype!r}. This script is for "
                "low-dimensional numerical features only."
            )

        shape = spec.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 1:
            raise ValueError(
                f"Feature {key!r} must be a 1-D numerical vector in info.json; "
                f"got shape={shape!r}."
            )

        dim = int(shape[0])
        if dim <= 0:
            raise ValueError(f"Feature {key!r} has invalid dimension {dim}.")
        feature_dims[key] = dim

    return feature_dims


def find_data_parquets(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        # Keep a slightly more permissive fallback for valid v3 datasets whose
        # chunk naming differs while retaining parquet storage.
        parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return parquet_files


def arrow_vector_to_numpy(array: pa.Array, *, expected_dim: int, key: str) -> np.ndarray:
    """Convert one Arrow numerical vector column to float64 shape (N, D).

    ``float64`` is deliberate. LeRobot's ``RunningQuantileStats`` computes
    variance from ``E[x^2] - E[x]^2``. For nearly constant features with a
    non-zero offset, performing that calculation in float32 can suffer severe
    cancellation and produce an impossible standard deviation.
    """
    if array.null_count:
        raise ValueError(f"Feature {key!r} contains {array.null_count} null rows.")

    if pa.types.is_fixed_size_list(array.type):
        list_size = array.type.list_size
        if list_size != expected_dim:
            raise ValueError(
                f"Feature {key!r} Arrow list size is {list_size}, "
                f"but meta/info.json expects {expected_dim}."
            )
        values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=STATS_DTYPE)
        result = values.reshape(len(array), expected_dim)
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        # Variable-list columns are less common for LeRobot state/action, but
        # to_pylist is a robust fallback and validates width below.
        result = np.asarray(array.to_pylist(), dtype=STATS_DTYPE)
    else:
        result = np.asarray(array.to_numpy(zero_copy_only=False), dtype=STATS_DTYPE)
        if result.ndim == 1 and expected_dim == 1:
            result = result.reshape(-1, 1)

    if result.ndim != 2 or result.shape[1] != expected_dim:
        raise ValueError(
            f"Feature {key!r} decoded to shape {result.shape}; "
            f"expected (N, {expected_dim})."
        )

    if not np.isfinite(result).all():
        bad = np.argwhere(~np.isfinite(result))[0]
        raise ValueError(
            f"Feature {key!r} contains NaN/Inf at local batch index "
            f"row={int(bad[0])}, dim={int(bad[1])}."
        )

    return result


def compute_dataset_stats(
    parquet_files: list[Path],
    feature_dims: dict[str, int],
    *,
    batch_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Stream every requested feature through LeRobot's RunningQuantileStats."""
    trackers = {
        key: RunningQuantileStats(quantile_list=list(DEFAULT_QUANTILES))
        for key in feature_dims
    }

    total_rows = 0
    feature_keys = list(feature_dims)

    for file_idx, parquet_path in enumerate(parquet_files, start=1):
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema_arrow.names)
        missing = [key for key in feature_keys if key not in available]
        if missing:
            raise KeyError(
                f"{parquet_path} is missing required columns: {', '.join(missing)}"
            )

        file_rows = 0
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=feature_keys,
        ):
            rows = record_batch.num_rows
            if rows == 0:
                continue

            for key in feature_keys:
                column_idx = record_batch.schema.get_field_index(key)
                array = record_batch.column(column_idx)
                values = arrow_vector_to_numpy(
                    array,
                    expected_dim=feature_dims[key],
                    key=key,
                )
                if values.shape[0] != rows:
                    raise RuntimeError(
                        f"Row-count mismatch for {key!r} in {parquet_path}: "
                        f"batch={rows}, decoded={values.shape[0]}"
                    )
                trackers[key].update(values)

            file_rows += rows
            total_rows += rows

        print(
            f"[{file_idx:4d}/{len(parquet_files):4d}] "
            f"rows={file_rows:>9,}  total={total_rows:>12,}  {parquet_path}",
            flush=True,
        )

    if total_rows < 2:
        raise ValueError(f"Need at least 2 rows to compute statistics; found {total_rows}.")

    stats = {key: tracker.get_statistics() for key, tracker in trackers.items()}
    return stats, total_rows


def validate_stats(
    stats: dict[str, dict[str, np.ndarray]],
    feature_dims: dict[str, int],
    *,
    expected_count: int,
) -> None:
    required_stat_keys = (
        "min",
        "max",
        "mean",
        "std",
        "count",
        "q01",
        "q10",
        "q50",
        "q90",
        "q99",
    )

    for feature_key, dim in feature_dims.items():
        if feature_key not in stats:
            raise ValueError(f"Missing statistics for feature {feature_key!r}.")

        feature_stats = stats[feature_key]
        missing = [key for key in required_stat_keys if key not in feature_stats]
        if missing:
            raise ValueError(
                f"Statistics for {feature_key!r} are missing keys: {missing}"
            )

        count = np.asarray(feature_stats["count"])
        if count.shape != (1,) or int(count[0]) != expected_count:
            raise ValueError(
                f"Bad count for {feature_key!r}: {count}; expected [{expected_count}]"
            )

        for stat_key in required_stat_keys:
            array = np.asarray(feature_stats[stat_key])
            if stat_key == "count":
                continue
            if array.shape != (dim,):
                raise ValueError(
                    f"{feature_key}.{stat_key} has shape {array.shape}; expected {(dim,)}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{feature_key}.{stat_key} contains NaN/Inf.")

        std = np.asarray(feature_stats["std"], dtype=np.float64)
        if np.any(std < 0):
            raise ValueError(f"{feature_key}.std contains a negative value.")

        mins = np.asarray(feature_stats["min"], dtype=np.float64)
        maxs = np.asarray(feature_stats["max"], dtype=np.float64)
        if np.any(mins > maxs):
            raise ValueError(f"min > max for feature {feature_key!r}.")

        # Popoviciu's inequality: for any bounded variable X in [a, b],
        # Var(X) <= (b-a)^2 / 4, so std(X) <= (b-a)/2.
        # This catches catastrophic-cancellation failures such as a nearly
        # constant feature reporting a std larger than its entire value range.
        std_upper_bound = 0.5 * (maxs - mins)
        scale = np.maximum.reduce(
            [np.ones_like(mins), np.abs(mins), np.abs(maxs)]
        )
        numeric_tol = 1e-10 * scale
        bad_std = std > (std_upper_bound + numeric_tol)
        if np.any(bad_std):
            dims = np.flatnonzero(bad_std).tolist()
            details = "; ".join(
                f"dim {i}: min={mins[i]:.17g}, max={maxs[i]:.17g}, "
                f"std={std[i]:.17g}, max_possible_std={std_upper_bound[i]:.17g}"
                for i in dims
            )
            raise ValueError(
                f"Impossible standard deviation for feature {feature_key!r} "
                f"(Popoviciu bound violated): {details}"
            )

        quantiles = np.stack(
            [np.asarray(feature_stats[key]) for key in ("q01", "q10", "q50", "q90", "q99")]
        )
        if np.any(np.diff(quantiles, axis=0) < -1e-6):
            raise ValueError(f"Quantiles are not monotonic for feature {feature_key!r}.")



def backup_existing_stats(dataset_root: Path) -> Path | None:
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        return None

    preferred = stats_path.with_name("stats.json.backup")
    backup_path = preferred
    if backup_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = stats_path.with_name(f"stats.json.backup-{timestamp}")

    shutil.copy2(stats_path, backup_path)
    return backup_path


def print_summary(stats: dict[str, dict[str, np.ndarray]]) -> None:
    print("\n=== DATASET STATS SUMMARY ===")
    for feature_key, feature_stats in stats.items():
        count = int(np.asarray(feature_stats["count"])[0])
        mean = np.asarray(feature_stats["mean"])
        std = np.asarray(feature_stats["std"])
        zero_std = np.flatnonzero(std == 0)
        near_zero_std = np.flatnonzero((std > 0) & (std < 1e-6))

        print(f"\n{feature_key}")
        print(f"  count: {count:,}")
        print(f"  mean : {np.array2string(mean, precision=8, separator=', ')}")
        print(f"  std  : {np.array2string(std, precision=8, separator=', ')}")
        if len(zero_std):
            print(f"  zero-std dims     : {zero_std.tolist()}")
        if len(near_zero_std):
            print(f"  near-zero std dims: {near_zero_std.tolist()}")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")

    print(f"dataset_root: {dataset_root}")
    info = load_info(dataset_root)
    feature_dims = validate_requested_features(info, args.features)
    parquet_files = find_data_parquets(dataset_root)

    print(f"codebase_version: {info.get('codebase_version')}")
    print(f"info.total_frames: {info.get('total_frames')}")
    print(f"parquet files: {len(parquet_files)}")
    print(f"statistics dtype: {np.dtype(STATS_DTYPE).name}")
    print("features:")
    for key, dim in feature_dims.items():
        print(f"  - {key}: dim={dim}")
    print()

    stats, total_rows = compute_dataset_stats(
        parquet_files,
        feature_dims,
        batch_size=args.batch_size,
    )

    info_total_frames = info.get("total_frames")
    if info_total_frames is not None and int(info_total_frames) != total_rows:
        raise RuntimeError(
            f"Frame-count mismatch: parquet scan found {total_rows:,} rows, "
            f"but meta/info.json reports total_frames={int(info_total_frames):,}. "
            "Refusing to write stats.json."
        )

    validate_stats(stats, feature_dims, expected_count=total_rows)
    print_summary(stats)

    if not args.no_backup:
        backup_path = backup_existing_stats(dataset_root)
        if backup_path is not None:
            print(f"\nBacked up existing stats.json -> {backup_path}")

    write_stats(stats, dataset_root)
    stats_path = dataset_root / "meta" / "stats.json"
    print(f"Wrote LeRobot v3 stats -> {stats_path}")

    # Reload through LeRobot's own reader.  This verifies both JSON structure
    # and the exact loading path that training will use.
    reloaded = load_stats(dataset_root)
    if reloaded is None:
        raise RuntimeError("LeRobot load_stats() returned None after writing stats.json.")

    validate_stats(reloaded, feature_dims, expected_count=total_rows)
    print("Reload validation through lerobot.datasets.io_utils.load_stats(): OK")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; stats.json was not intentionally modified after interruption.", file=sys.stderr)
        raise
