#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

python -m projects.robot_learning.rl.train \
  --checkpoint=outputs/baseline_100k/checkpoints/100000/pretrained_model \
  --output=outputs/flow_noise_libero_10 \
  --suites libero_10 \
  --suites-per-update 1 \
  --workers default=3 \
  --max-concurrent-envs 30 \
  --ppo-epochs 4 \
  --chunks-per-worker 840 \
  --max-episode-steps 480 \
  --minibatch 30 \
  --gradient-accumulation-steps 70 \
  --action-steps 10 \
  --denoise-steps 4 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --pad-language-to "max_length" \
  --tokenizer-max-length 48 \
  --updates 500 \
  --total-training-steps 500 \
  --lr-scheduler constant \
  --compile-model \
  --train-expert-only \
  --train-state-proj \
  --freeze-vision-encoder \
  "$@" 2>&1 | sed -E '/robosuite WARNING|WARNING:robosuite_logs|\[info\] using task orders|Local assets not found|Assets already downloaded/d; /^[[:space:]]*$/d'
