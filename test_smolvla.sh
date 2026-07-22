#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl

lerobot-eval \
  --policy.path=outputs/smolvla_024b_libero_no_vla_pt_lance/checkpoints/010000/pretrained_model \
  --policy.device=cuda \
  --policy.n_action_steps=1 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=1000 \
  --output_dir=outputs/eval/smolvla_024b_libero_no_vla_pt_lance