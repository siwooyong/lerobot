"""Benchmark Lance DataLoader reads with SmolVLA's LIBERO sample shape."""

import argparse
import statistics
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--lance-root", type=Path, default=Path("outputs/datasets/libero_lance"))
    parser.add_argument("--table-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def make_loader_kwargs(num_workers, prefetch_factor, persistent_workers, pin_memory):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return kwargs


def _quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(timings, batch_size):
    if not timings:
        raise ValueError("timings must contain at least one measurement")
    mean_s = statistics.fmean(timings)
    return {
        "mean_s": mean_s,
        "median_s": statistics.median(timings),
        "p95_s": _quantile(timings, 0.95),
        "max_s": max(timings),
        "batch_per_s": 1 / mean_s,
        "sample_per_s": batch_size / mean_s,
    }


def main():
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from lerobot.datasets import EpisodeAwareSampler, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.utils.collate import lerobot_collate_fn
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

    print("Creating training-equivalent LIBERO Lance dataset...")
    started = time.perf_counter()
    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.lance_root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    dataset = LeRobotLanceDataset(
        root=args.lance_root,
        repo_id=args.repo_id,
        table_name=args.table_name,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
        decode_device=args.decode_device,
    )
    print(
        f"Dataset ready in {time.perf_counter() - started:.1f}s | "
        f"frames={dataset.num_frames:,}, episodes={dataset.num_episodes:,}, "
        f"decode_device={args.decode_device}"
    )

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(policy_cfg, "drop_n_last_frames", 0),
        shuffle=True,
        seed=args.seed,
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        **make_loader_kwargs(
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=torch.cuda.is_available() and args.decode_device == "cpu",
        ),
    )

    iterator = iter(loader)
    timings = []
    absolute_steps = []
    total = args.warmup + args.steps

    with tqdm(range(1, total + 1), desc="Lance DataLoader", unit="batch") as progress:
        for step in progress:
            started = time.perf_counter()
            try:
                _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                _ = next(iterator)
            fetch_s = time.perf_counter() - started

            if step > args.warmup:
                timings.append(fetch_s)
                absolute_steps.append(step)
                progress.set_postfix(
                    fetch=f"{fetch_s:.3f}s",
                    mean=f"{statistics.fmean(timings):.3f}s",
                    max=f"{max(timings):.3f}s",
                )
            else:
                progress.set_postfix(warmup=f"{step}/{args.warmup}", fetch=f"{fetch_s:.3f}s")

    result = summarize(timings, args.batch_size)
    print("\nMeasured Lance DataLoader latency (model/preprocessing excluded)")
    print(f"mean      : {result['mean_s']:.4f} s")
    print(f"median    : {result['median_s']:.4f} s")
    print(f"p95       : {result['p95_s']:.4f} s")
    print(f"max       : {result['max_s']:.4f} s")
    print(f"rate      : {result['batch_per_s']:.2f} batch/s")
    print(f"throughput: {result['sample_per_s']:.2f} samples/s")

    if args.num_workers > 0:
        periodic = [
            latency
            for step, latency in zip(absolute_steps, timings, strict=True)
            if step % args.num_workers == 0
        ]
        others = [
            latency
            for step, latency in zip(absolute_steps, timings, strict=True)
            if step % args.num_workers != 0
        ]
        if periodic and others:
            print(
                f"steps divisible by {args.num_workers}: "
                f"{statistics.fmean(periodic):.4f} s mean "
                f"(others: {statistics.fmean(others):.4f} s)"
            )


if __name__ == "__main__":
    main()