"""Benchmark the same LIBERO DataLoader path used by SmolVLA training."""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler
from lerobot.datasets.factory import make_dataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.collate import lerobot_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-backend", default=None, help="Example: torchcodec or pyav")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_kwargs = {"repo_id": args.repo_id}
    if args.video_backend:
        dataset_kwargs["video_backend"] = args.video_backend

    # This creates only the SmolVLA configuration. No model weights are loaded.
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
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(**dataset_kwargs),
        policy=policy_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        seed=args.seed,
    )

    print("Creating training-equivalent LIBERO dataset...")
    started = time.perf_counter()
    dataset = make_dataset(cfg)  # Resolves SmolVLA delta indices into temporal windows.
    print(
        f"Dataset ready in {time.perf_counter() - started:.1f}s | "
        f"frames={dataset.num_frames:,}, episodes={dataset.num_episodes:,}"
    )

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(policy_cfg, "drop_n_last_frames", 0),
        shuffle=True,
        seed=args.seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )

    iterator = iter(loader)
    timings = []
    absolute_steps = []
    total = args.warmup + args.steps

    with tqdm(range(1, total + 1), desc="DataLoader", unit="batch") as progress:
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
                    mean=f"{sum(timings) / len(timings):.3f}s",
                    max=f"{max(timings):.3f}s",
                )
            else:
                progress.set_postfix(warmup=f"{step}/{args.warmup}", fetch=f"{fetch_s:.3f}s")

    values = torch.tensor(timings, dtype=torch.float64)
    print("\nMeasured DataLoader latency (model/preprocessing excluded)")
    print(f"mean   : {values.mean().item():.4f} s")
    print(f"median : {values.median().item():.4f} s")
    print(f"p95    : {values.quantile(0.95).item():.4f} s")
    print(f"max    : {values.max().item():.4f} s")
    print(f"rate   : {1 / values.mean().item():.2f} batch/s")

    if args.num_workers > 0:
        periodic = [t for step, t in zip(absolute_steps, timings, strict=True) if step % args.num_workers == 0]
        others = [t for step, t in zip(absolute_steps, timings, strict=True) if step % args.num_workers != 0]
        if periodic and others:
            print(
                f"steps divisible by {args.num_workers}: {sum(periodic) / len(periodic):.4f} s mean "
                f"(others: {sum(others) / len(others):.4f} s)"
            )


if __name__ == "__main__":
    main()