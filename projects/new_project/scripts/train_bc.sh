#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  --module lerobot.scripts.lerobot_train \
  --accelerator.gradient_accumulation.steps=1 \
  --policy.type=smolvla \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=16 \
  --policy.expert_width_multiplier=0.75 \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.freeze_vision_encoder=true \
  --policy.compile_model=true \
  --policy.compile_mode=max-autotune-no-cudagraphs \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_betas='[0.9, 0.999]' \
  --policy.optimizer_weight_decay=1e-5 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=12500 \
  --policy.scheduler_decay_steps=250000 \
  --policy.scheduler_decay_lr=0 \
  --dataset.repo_id=robocasa365/pretrain_human \
  --dataset.root="$repo_root/projects/new_project/data/pretrain_human" \
  --dataset.eval_split=0 \
  --dataset.video_backend=torchcodec \
  --batch_size=48 \
  --num_workers=24 \
  --steps=250000 \
  --log_freq=200 \
  --save_freq=12500 \
  --env_eval_freq=0 \
  --seed=42 \
  --output_dir=outputs/smolvla024b_robocasa365_human300_bsz192_steps250k_ck16_gpu4 \
  --job_name=smolvla024b_robocasa365_human300_bsz192_steps250k_ck16_gpu4 \
  --wandb.enable=true \
  --wandb.project=robocasa_bc \
  --wandb.mode=online \
  --wandb.disable_artifact=true
