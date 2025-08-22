#!/bin/bash

bash external/mmdetection/tools/dist_train.sh \
    config/mmdetection/fisheye8k.py \
    4 \
    --work-dir ckpt/codetr/codetr_fisheye8k