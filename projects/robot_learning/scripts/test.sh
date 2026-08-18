#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl

lerobot-eval \
  --policy.path=outputs/baseline_100k/checkpoints/100000/pretrained_model \
  --policy.device=cuda \
  --policy.n_action_steps=1 \
  --policy.num_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --eval.batch_size=1 \
  --eval.n_episodes=25 \
  --seed=1000 \
  --output_dir=outputs/eval/baseline_100k_n1