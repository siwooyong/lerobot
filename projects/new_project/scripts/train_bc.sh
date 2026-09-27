#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --module lerobot.scripts.lerobot_train \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps=1 \
  --policy.type=smolvla_memory \
  --policy.base_pretrained_path=lerobot/smolvla_base \
  --policy.frame_num_tokens=4 \
  --policy.compressor_num_layers=1 \
  --policy.compressor_hidden_size=512 \
  --policy.compressor_num_heads=8 \
  --policy.memory_num_layers=2 \
  --policy.memory_hidden_size=512 \
  --policy.memory_num_heads=8 \
  --policy.sequence_length=24 \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.load_vlm_weights=true \
  --policy.num_vlm_layers=16 \
  --policy.expert_width_multiplier=0.75 \
  --policy.train_expert_only=true \
  --policy.train_state_proj=false \
  --policy.freeze_vision_encoder=true \
  --policy.prefix_length=0 \
  --policy.pad_language_to=max_length \
  --policy.compile_model=true \
  --policy.compile_mode=max-autotune-no-cudagraphs \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.normalization_mapping="{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: QUANTILES}" \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_betas='[0.9, 0.95]' \
  --policy.optimizer_weight_decay=1e-8 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=12500 \
  --policy.scheduler_decay_steps=250000 \
  --policy.scheduler_decay_lr=1e-6 \
  --dataset.repo_id=robocasa365/atomic \
  --dataset.root="$repo_root/projects/new_project/data/atomic" \
  --dataset.eval_split=0 \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.random_order=true \
  --batch_size=4 \
  --num_workers=16 \
  --steps=250000 \
  --log_freq=10 \
  --save_freq=10000 \
  --env_eval_freq=0 \
  --seed=42 \
  --output_dir=outputs/smolvla045b_memory_robocasa365_atomic_bsz8_steps250k_ck16_gpu2 \
  --job_name=smolvla045b_memory_robocasa365_atomic_bsz8_steps250k_ck16_gpu2 \
  --wandb.enable=true \
  --wandb.project=robocasa_bc \
  --wandb.mode=online \
  --wandb.disable_artifact=true
