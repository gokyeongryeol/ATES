#!/bin/bash
set -e

GEN_MODEL="black-forest-labs/FLUX.1-dev"
# GEN_MODEL_CKPT="ckpt/flux/flux_fisheye8k/checkpoint-1000"  # TO BE UPDATED
DATA_DIR="/mnt/nas-1/data/FishEyeChallenge/FishEye8K"

BASE_D_DIR="$DATA_DIR/train-D"
CAP_D_JSON="$BASE_D_DIR/train-D_with_caption.json"
NAV_V0_D_DATA_DIR="$DATA_DIR/train-D_naive_v0-gen"

BASE_R_DIR="$DATA_DIR/train-R"
REP_R_JSON="$BASE_R_DIR/train-R_with_rephrased.json"
REP_R_EVAL_JSON="$BASE_R_DIR/train-R_with_rephrased-eval.json"
REP_R_DATA_DIR="$DATA_DIR/train-R_rephrased-gen"
REP_R_DATA_EVAL_DIR="$REP_R_DATA_DIR-eval"

# AUT_V1_D_JSON="$BASE_D_DIR/train-D_with_automatic_v1.json"
# AUT_V1_D_DATA_DIR="$DATA_DIR/train-D_automatic_v1-gen"


echo "[Generating images] Output: $NAV_V0_D_DATA_DIR"
accelerate launch --multi_gpu tools/synthesize_from_text.py \
    --model_name "$GEN_MODEL" \
    --json_path "$CAP_D_JSON" \
    --ckpt_dir "$GEN_MODEL_CKPT" \
    --output_dir "$NAV_V0_D_DATA_DIR" \
    --use_naive


echo "[Generating images] Output: $REP_R_DATA_DIR"
accelerate launch --multi_gpu tools/synthesize_from_text.py \
    --model_name "$GEN_MODEL" \
    --json_path "$REP_R_JSON" \
    --ckpt_dir "$GEN_MODEL_CKPT" \
    --output_dir "$REP_R_DATA_DIR"


echo "[Generating images] Output: $REP_R_DATA_EVAL_DIR"
accelerate launch --multi_gpu tools/synthesize_from_text.py \
    --model_name "$GEN_MODEL" \
    --json_path "$REP_R_EVAL_JSON" \
    --ckpt_dir "$GEN_MODEL_CKPT" \
    --output_dir "$REP_R_DATA_EVAL_DIR"


# echo "[Generating images] Output: $AUT_V1_D_DATA_DIR"
# accelerate launch --multi_gpu tools/synthesize_from_text.py \
#     --model_name "$GEN_MODEL" \
#     --json_path "$AUT_V1_D_JSON" \
#     --ckpt_dir "$GEN_MODEL_CKPT" \
#     --output_dir "$AUT_V1_D_DATA_DIR"
