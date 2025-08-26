#!/bin/bash
set -e

# CHECKPOINT="ckpt/codetr/codetr_fisheye8k/best_coco_bbox_mAP_epoch_9.pth"  # TO BE UPDATED
DATA_DIR="/mnt/data/FishEye8K"

BASE_R_DIR="$DATA_DIR/train-R"
GT_R_JSON="$BASE_R_DIR/train-R.json"
TMP_PL_R_JSON="$BASE_R_DIR/train-R_with_tmp_pseudolabel.json"
OPT_CONF_JSON="$BASE_R_DIR/opt_conf_thr.json"


echo "[Obtaining temporary prediction] Output: $TMP_PL_R_JSON"
python tools/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_train-R.py \
    "$CHECKPOINT" \
    "" \
    "$TMP_PL_R_JSON"


echo "[Estimating optimal threshold] Output: $OPT_CONF_JSON"
python tools/estimate_optimal_threshold.py \
    --gt_json "$GT_R_JSON" \
    --pred_json "$TMP_PL_R_JSON" \
    --save_json "$OPT_CONF_JSON"
