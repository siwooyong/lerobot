#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

export MUJOCO_GL=egl

lerobot-eval \
  --policy.path="$repo_root/outputs/smolvla045b_robocasa365_atomic_quantiles_bsz192_steps250k_ck16_gpu2/checkpoints/060000/pretrained_model" \
  --policy.device=cuda \
  --env.type=robocasa \
  --env.task=atomic_seen \
  --env.split=pretrain \
  --env.obj_registries="[objaverse,lightwheel]" \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --seed=42 \
  --output_dir="$repo_root/outputs/eval/robocasa_atomic_quantiles_060000"