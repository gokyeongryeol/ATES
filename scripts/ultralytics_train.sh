#!/bin/bash
set -e

PORT=29500
DEVICES="0,1,2,3"

CONFIG_NAMES=(
    "fisheye8k"
    "fisheye8k_with_naive_v0"
    # "fisheye8k_with_naive_v0+automatic_v1"
    # "fisheye8k_with_naive_v0+automatic_v1+automatic_v2"
)

for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    echo "Running training for: $CONFIG_NAME on devices: $DEVICES"

    python -m torch.distributed.run --nproc_per_node=4 --master_port "$PORT" tools/train_base_detector.py \
        --project-name fisheye_od_yolo \
        --run-name "yolo11s_$CONFIG_NAME" \
        --yaml-file "config/ultralytics/$CONFIG_NAME.yaml" \
        --init-weight ckpt/yolo/yolo11s.pt \
        --device "$DEVICES"
done
