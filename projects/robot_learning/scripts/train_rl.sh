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
  --checkpoint=outputs/flow_noise_libero_spatial_5/update_00030/pretrained_model \
  --output=outputs/flow_noise_libero_spatial \
  --suites libero_spatial \
  --suites-per-update 1 \
  --workers default=16 \
  --max-concurrent-envs 32 \
  --ppo-epochs 8 \
  --minibatch 32 \
  --gradient-accumulation-steps 70 \
  --action-steps 10 \
  --denoise-steps 5 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --pad-language-to "max_length" \
  --tokenizer-max-length 48 \
  --train-expert-only \
  --train-state-proj \
  --freeze-vision-encoder \
  "$@" 2>&1 | sed -E '/robosuite WARNING|WARNING:robosuite_logs|\[info\] using task orders|Local assets not found|Assets already downloaded/d; /^[[:space:]]*$/d'
