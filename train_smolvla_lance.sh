#!/usr/bin/env bash
set -euo pipefail

# CPU JPEG decode avoids creating one CUDA context per DataLoader worker.
export LEROBOT_LANCE_DECODE_DEVICE=cpu

accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  train_smolvla_lance.py \
  --policy.type=smolvla \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=16 \
  --policy.expert_width_multiplier=0.75 \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.freeze_vision_encoder=true \
  --policy.compile_model=true \
  --policy.push_to_hub=false \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=outputs/datasets/libero_lance \
  --batch_size=64 \
  --num_workers=16 \
  --steps=100000 \
  --save_freq=10000 \
  --env_eval_freq=0 \
  --seed=1000 \
  --output_dir=outputs/baseline[lr_decay_100k] \
  --job_name=baseline[lr_decay_100k] \
  --wandb.enable=true