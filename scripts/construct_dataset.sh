#!/bin/bash

CONFIG_FILE_NAME="fisheye8k_with_naive_v0"
DET_MODEL_CKPT="ckpt/yolo/yolo11s_$CONFIG_FILE_NAME"

DATA_DIR="/mnt/nas-1/data/FishEyeChallenge/FishEye8K"
BASE_VAL_DIR="$DATA_DIR/train-R"
PREF_DATA_DIR="$DATA_DIR/train-R_preference_with_naive_v0"
REP_VAL_DATA_DIR="$DATA_DIR/train-R_rephrased-gen"
REP_VAL_DATA_EVAL_DIR="$REP_VAL_DATA_DIR-eval"

REP_VAL_JSON="$BASE_VAL_DIR/train-R_with_rephrased.json"
REP_VAL_EVAL_JSON="$BASE_VAL_DIR/train-R_with_rephrased-eval.json"

echo "[Constructing preference dataset] Split: train"
python scripts/construct_dataset.py \
    --json_path "$REP_VAL_JSON" \
    --base_dir "$REP_VAL_DATA_DIR" \
    --ckpt_dir "$DET_MODEL_CKPT" \
    --output_dir "$PREF_DATA_DIR/train"


echo "[Constructing preference dataset] Split: test"
python scripts/construct_dataset.py \
    --json_path "$REP_VAL_EVAL_JSON" \
    --base_dir "$REP_VAL_DATA_EVAL_DIR" \
    --ckpt_dir "$DET_MODEL_CKPT" \
    --output_dir "$PREF_DATA_DIR/test"


echo "[Saving dataset dictionary]"
python scripts/create_dataset_dict.py \
    --json_path "$PREF_DATA_DIR/dataset_dict.json"
