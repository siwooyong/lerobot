#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

export MUJOCO_GL=egl

for task in atomic_seen composite_seen composite_unseen; do
  lerobot-eval \
    --policy.path="$repo_root/outputs/groot_n17_robocasa365_atomic_bsz128_steps300k_ck16_gpu1/checkpoints/030000/pretrained_model" \
    --policy.device=cuda \
    --policy.chunk_size=16 \
    --policy.n_action_steps=8 \
    --env.type=robocasa \
    --env.task="$task" \
    --env.split=pretrain \
    --env.obj_registries="[objaverse,lightwheel]" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --eval.use_async_envs=true \
    --seed=42 \
    --output_dir="$repo_root/outputs/eval/groot_n17_robocasa365_atomic_bsz128_steps300k_ck16_gpu1/030000/$task"
done