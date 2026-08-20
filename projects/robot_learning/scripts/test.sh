#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl

for update in $(seq 210 10 300); do
  update_id=$(printf '%05d' "${update}")

  lerobot-eval \
    --policy.path=outputs/flow_noise_libero_10/update_${update_id}/pretrained_model \
    --policy.device=cuda \
    --policy.n_action_steps=10 \
    --policy.num_steps=5 \
    --env.type=libero \
    --env.task=libero_10 \
    --env.control_mode=relative \
    --env.max_parallel_tasks=1 \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --eval.batch_size=25 \
    --eval.n_episodes=100 \
    --seed=1000 \
    --output_dir=outputs/eval/flow_noise_libero_10_u${update_id}
done
