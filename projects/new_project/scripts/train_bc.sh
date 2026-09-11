#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  --module projects.new_project.scripts.train_bc \
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
  --policy.n_obs_steps=1 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --dataset.repo_id=robocasa/pretrain_human \
  --dataset.root="$repo_root/projects/new_project/robocasa/datasets" \
  --dataset.eval_split=0 \
  --dataset.video_backend=torchcodec \
  --batch_size=64 \
  --num_workers=16 \
  --steps=100000 \
  --save_freq=10000 \
  --env_eval_freq=0 \
  --seed=1000 \
  --output_dir=outputs/robocasa_bc_100k \
  --job_name=robocasa_bc_100k \
  --wandb.enable=true \
  --wandb.project=robocasa_bc \
  --wandb.mode=online \
  --wandb.disable_artifact=true
