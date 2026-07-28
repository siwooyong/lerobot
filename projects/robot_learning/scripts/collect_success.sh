#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl

python -m projects.robot_learning.scripts.collect_success \
  --policy.path=outputs/baseline_100k/checkpoints/100000/pretrained_model \
  --policy.device=cuda \
  --policy.n_action_steps=1 \
  --policy.num_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.control_mode=relative \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --collection.repo_id=siwooyong/libero-smolvla-success-rollouts \
  --collection.root=outputs/datasets/libero_success_rollouts \
  --collection.successes_per_task=10 \
  --collection.max_attempts_per_task=1000 \
  --collection.resume=true \
  --collection.push_to_hub=false \
  --collection.fps=10 \
  --seed=1000 \
  --output_dir=outputs/collection/baseline_100k_success
