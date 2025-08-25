#!/bin/bash
set -e


CONFIG_NAMES=(
    "fisheye8k_with_naive_v0+automatic_v1"
)

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    echo "Running evaluation for: $CONFIG_NAME"

    python tools/eval_map_wo_tp.py \
        --model_path "ckpt/yolo/yolo11s_$CONFIG_NAME/weights/best.pt" \
        --data_yaml "config/ultralytics/$CONFIG_NAME.yaml" \
        --save_json "ckpt/yolo/yolo11s_$CONFIG_NAME/result.json" \
        --imgsz 1280 \
        # --ref_model_path "ckpt/yolo/yolo11s_fisheye8k_with_naive_v0/weights/best.pt"  # UNCOMMENT for mAP w/o TP
done
