#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

export MUJOCO_GL=egl

for task in atomic_seen composite_seen composite_unseen; do
  lerobot-eval \
    --policy.path="$repo_root/outputs/smolvla045b_robocasa365_human300_quantiles_prtsaug_bsz192_steps250k_ck16_gpu4/checkpoints/250000/pretrained_model" \
    --policy.device=cuda \
    --env.type=robocasa \
    --env.task="$task" \
    --env.split=pretrain \
    --env.obj_registries="[objaverse,lightwheel]" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=10 \
    --eval.n_episodes=50 \
    --eval.use_async_envs=true \
    --seed=42 \
    --output_dir="$repo_root/outputs/eval/smolvla045b_robocasa365_human300_quantiles_prtsaug_bsz192_steps250k_ck16_gpu4/$task"
done