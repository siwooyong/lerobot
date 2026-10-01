#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

torchrun \
  --standalone \
  --nproc_per_node=2 \
  --module lerobot.scripts.lerobot_train \
  --parallelism.dp_replicate=1 \
  --parallelism.dp_shard=2 \
  --accelerator.fsdp.wrap_modules='["Qwen3VLVisionBlock", "Qwen3VLTextDecoderLayer", "BasicTransformerBlock"]' \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps=1 \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.use_relative_actions=false \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_top_llm_layers=0 \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.tune_vlln=true \
  --policy.use_bf16=true \
  --policy.use_flash_attention=true \
  --policy.model_params_fp32=true \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_betas='[0.9, 0.999]' \
  --policy.optimizer_eps=1e-8 \
  --policy.optimizer_weight_decay=1e-5 \
  --policy.max_steps=300000 \
  --policy.push_to_hub=false \
  --dataset.repo_id=robocasa365/pretrain_human \
  --dataset.root="$repo_root/projects/new_project/data/pretrain_human" \
  --dataset.eval_split=0 \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=false \
  --batch_size=64 \
  --num_workers=4 \
  --steps=300000 \
  --log_freq=10 \
  --save_freq=10000 \
  --env_eval_freq=0 \
  --seed=42 \
  --output_dir=outputs/groot_n17_robocasa365_human300_bsz128_steps300k_ck16_gpu2_fsdp2 \
  --job_name=groot_n17_robocasa365_human300_bsz128_steps300k_ck16_gpu2_fsdp2 \
  --wandb.enable=true \
  --wandb.project=robocasa_bc \
  --wandb.mode=online \
  --wandb.disable_artifact=true
