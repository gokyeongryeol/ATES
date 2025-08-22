#!/bin/bash

export WANDB_PROJECT="fisheye_gen_llama"

DATA_DIR="/mnt/nas-1/data/FishEyeChallenge/FishEye8K"
PREF_DATA_DIR="$DATA_DIR/train-R_preference_with_naive_v0"

accelerate launch --multi_gpu external/trl/trl/scripts/dpo.py \
    --dataset_name "$PREF_DATA_DIR" \
    --dataset_streaming \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --learning_rate 5.0e-6 \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --output_dir "ckpt/llama/llama_dpo_fisheye8k_with_naive_v0" \
    --run_name "llama_dpo_fisheye8k_with_naive_v0" \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --report_to wandb
