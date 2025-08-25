#!/bin/bash
set -e

CAPTION_MODEL="OpenGVLab/InternVL3-38B"
REPHRASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
DATA_DIR="/mnt/nas-1/data/FishEyeChallenge/FishEye8K"

BASE_D_DIR="$DATA_DIR/train-D/"
GT_D_JSON="$BASE_D_DIR/train-D.json"
CAP_D_JSON="$BASE_D_DIR/train-D_with_caption.json"

BASE_R_DIR="$DATA_DIR/train-R/"
GT_R_JSON="$BASE_R_DIR/train-R.json"
CAP_R_JSON="$BASE_R_DIR/train-R_with_caption.json"
REP_R_JSON="$BASE_R_DIR/train-R_with_rephrased.json"
REP_R_EVAL_JSON="$BASE_R_DIR/train-R_with_rephrased-eval.json"

# AUT_V1_MODEL_CKPT="ckpt/llama/llama_dpo_fisheye8k_with_naive_v0/checkpoint-800"  # TO BE UPDATED
# AUT_V2_MODEL_CKPT="ckpt/llama/llama_dpo_fisheye8k_with_naive_v0+automatic_v1/checkpoint-700"  # TO BE UPDATED
# AUT_V1_D_JSON="$BASE_D_DIR/train-D_with_automatic_v1.json"
# AUT_V2_D_JSON="$BASE_D_DIR/train-D_with_automatic_v2.json"


echo "[Extracting captions] Model: $CAPTION_MODEL"
accelerate launch --multi_gpu tools/extract_caption.py \
    --model_name "$CAPTION_MODEL" \
    --base_dir "$BASE_D_DIR" \
    --json_path "$GT_D_JSON" \
    --output_path "$CAP_D_JSON"


echo "[Extracting captions] Model: $CAPTION_MODEL"
accelerate launch --multi_gpu tools/extract_caption.py \
    --model_name "$CAPTION_MODEL" \
    --base_dir "$BASE_R_DIR" \
    --json_path "$GT_R_JSON" \
    --output_path "$CAP_R_JSON"


echo "[Rephrasing captions] Model: $REPHRASE_MODEL. With DIVERSE_PROMPT. Split: train"
accelerate launch --multi_gpu tools/rephrase_caption.py \
    --model_name "$REPHRASE_MODEL" \
    --json_path "$CAP_R_JSON" \
    --output_path "$REP_R_JSON"


echo "[Rephrasing captions] Model: $REPHRASE_MODEL. With DIVERSE_PROMPT. Split: test"
accelerate launch --multi_gpu tools/rephrase_caption.py \
    --model_name "$REPHRASE_MODEL" \
    --json_path "$CAP_R_JSON" \
    --output_path "$REP_R_EVAL_JSON"


# echo "[Rephrasing captions] Model: $REPHRASE_MODEL. With AUTOMATIC_V1_PROMPT"
# accelerate launch --multi_gpu tools/rephrase_caption.py \
#     --model_name "$REPHRASE_MODEL" \
#     --json_path "$CAP_D_JSON" \
#     --output_path "$AUT_V1_D_JSON" \
#     --ckpt_dir "$AUT_V1_MODEL_CKPT"


# echo "[Rephrasing captions] Model: $REPHRASE_MODEL. With AUTOMATIC_V2_PROMPT"
# accelerate launch --multi_gpu tools/rephrase_caption.py \
#     --model_name "$REPHRASE_MODEL" \
#     --json_path "$CAP_D_JSON" \
#     --output_path "$AUT_V2_D_JSON" \
#     --ckpt_dir "$AUT_V2_MODEL_CKPT"
