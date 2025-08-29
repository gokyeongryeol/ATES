#!/bin/bash
set -e

DATA_DIR="/mnt/data/FishEye8K"

BASE_R_DIR="$DATA_DIR/train-R"
GT_R_JSON="$BASE_R_DIR/train-R.json"
TMP_PL_R_JSON="$BASE_R_DIR/train-R_with_tmp_pseudolabel.json"
OPT_CONF_JSON="$BASE_R_DIR/opt_conf_thr.json"


echo "[Estimating optimal threshold] Output: $OPT_CONF_JSON"
python tools/estimate_optimal_threshold.py \
    --gt_json "$GT_R_JSON" \
    --pred_json "$TMP_PL_R_JSON" \
    --save_json "$OPT_CONF_JSON"
