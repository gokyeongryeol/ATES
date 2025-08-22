#!/bin/bash
set -e

CHECKPOINT="ckpt/codetr/codetr_fisheye8k/best_coco_bbox_mAP_epoch_9.pth"
DATA_DIR="/mnt/nas-1/data/FishEyeChallenge/FishEye8K"

BASE_R_DIR="$DATA_DIR/train-R"
OPT_CONF_JSON="$BASE_R_DIR/opt_conf_thr.json"

NAV_V0_D_DATA_DIR="$DATA_DIR/train-D_naive_v0-gen"
NAV_V0_D_PSEUDO_JSON="$NAV_V0_D_DATA_DIR/train-D_naive_v0-gen_with_pseudolabel.json"
NAV_V1_D_DATA_DIR="$DATA_DIR/train-D_naive_v1-gen"
NAV_V1_D_PSEUDO_JSON="$NAV_V1_D_DATA_DIR/train-D_naive_v1-gen_with_pseudolabel.json"
NAV_V2_D_DATA_DIR="$DATA_DIR/train-D_naive_v2-gen"
NAV_V2_D_PSEUDO_JSON="$NAV_V2_D_DATA_DIR/train-D_naive_v2-gen_with_pseudolabel.json"

MAN_V1_D_DATA_DIR="$DATA_DIR/train-D_manual_v1-gen"
MAN_V1_D_PSEUDO_JSON="$MAN_V1_D_DATA_DIR/train-D_manual_v1-gen_with_pseudolabel.json"
MAN_V2_D_DATA_DIR="$DATA_DIR/train-D_manual_v2-gen"
MAN_V2_D_PSEUDO_JSON="$MAN_V2_D_DATA_DIR/train-D_manual_v2-gen_with_pseudolabel.json"

AUT_V1_D_DATA_DIR="$DATA_DIR/train-D_automatic_v1-gen"
AUT_V1_D_PSEUDO_JSON="$AUT_V1_D_DATA_DIR/train-D_automatic_v1-gen_with_pseudolabel.json"
AUT_V2_D_DATA_DIR="$DATA_DIR/train-D_automatic_v2-gen"
AUT_V2_D_PSEUDO_JSON="$AUT_V2_D_DATA_DIR/train-D_automatic_v2-gen_with_pseudolabel.json"

REP_R_DATA_DIR="$DATA_DIR/train-R_rephrased-gen"
REP_R_PSEUDO_JSON="$REP_R_DATA_DIR/train-R_rephrased-gen_with_pseudolabel.json"
REP_R_DATA_EVAL_DIR="$REP_R_DATA_DIR-eval"
REP_R_PSEUDO_EVAL_JSON="$REP_R_DATA_EVAL_DIR/train-R_rephrased-gen-eval_with_pseudolabel.json"

echo "[Pseudo-labeling images] Data: naive_v0-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_naive_v0-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$NAV_V0_D_PSEUDO_JSON"


echo "[Converting annotations] Data: naive_v0-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$NAV_V0_D_DATA_DIR"


echo "[Pseudo-labeling images] Data: naive_v1-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_naive_v1-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$NAV_V1_D_PSEUDO_JSON"


echo "[Converting annotations] Data: naive_v1-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$NAV_V1_D_DATA_DIR"


echo "[Pseudo-labeling images] Data: naive_v2-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_naive_v2-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$NAV_V2_D_PSEUDO_JSON"


echo "[Converting annotations] Data: naive_v2-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$NAV_V2_D_DATA_DIR"


echo "[Pseudo-labeling images] Data: manual_v1-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_manual_v1-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$MAN_V1_D_PSEUDO_JSON"


echo "[Converting annotations] Data: manual_v1-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$MAN_V1_D_DATA_DIR"


echo "[Pseudo-labeling images] Data: manual_v2-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_manual_v2-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$MAN_V2_D_PSEUDO_JSON"


echo "[Converting annotations] Data: manual_v2-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$MAN_V2_D_DATA_DIR"


echo "[Pseudo-labeling images] Data: automatic_v1-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_automatic_v1-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$AUT_V1_D_PSEUDO_JSON"


echo "[Converting annotations] Data: automatic_v1-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$AUT_V1_D_DATA_DIR"


echo "[Pseudo-labeling images] Data: automatic_v2-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_automatic_v2-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$AUT_V2_D_PSEUDO_JSON"


echo "[Converting annotations] Data: automatic_v2-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$AUT_V2_D_DATA_DIR"


echo "[Pseudo-labeling images] Data: rephrased-gen"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_rephrased-gen.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$REP_R_PSEUDO_JSON"


echo "[Converting annotations] Data: rephrased-gen"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$REP_R_DATA_DIR"


echo "[Pseudo-labeling images] Data: rephrased-gen-eval"
python scripts/obtain_pseudo_label.py \
    config/mmdetection/fisheye8k_pl_for_rephrased-gen-eval.py \
    "$CHECKPOINT" \
    "$OPT_CONF_JSON" \
    "$REP_R_PSEUDO_EVAL_JSON"


echo "[Converting annotations] Data: rephrased-gen-eval"
python scripts/convert_coco_to_yolo.py \
    --base_dir "$REP_R_DATA_EVAL_DIR"
