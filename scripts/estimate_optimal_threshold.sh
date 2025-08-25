#!/bin/bash
set -e

# CHECKPOINT="ckpt/codetr/codetr_fisheye8k/best_coco_bbox_mAP_epoch_9.pth"  # TO BE UPDATED
DATA_DIR="/mnt/nas-1/data/FishEyeChallenge/FishEye8K"

BASE_R_DIR="$DATA_DIR/train-R"
GT_R_JSON="$BASE_R_DIR/train-R.json"
NAIVE_R_JSON="$BASE_R_DIR/train-R_with_naive_pseudolabel.json"
OPT_CONF_JSON="$BASE_R_DIR/opt_conf_thr.json"


echo "[Obtaining naive prediction] Output: $NAIVE_R_JSON"
python tools/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_gt.py \
    "$CHECKPOINT" \
    "" \
    "$NAIVE_R_JSON"


echo "[Estimating optimal threshold] Output: $OPT_CONF_JSON"
python tools/estimate_optimal_threshold.py \
    --gt_json "$GT_R_JSON" \
    --pred_json "$NAIVE_R_JSON" \
    --save_json "$OPT_CONF_JSON"
