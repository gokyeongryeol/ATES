from typing import Optional
import argparse
import sys
import os
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultralytics_custom.models.yolo.detect.val import CustomDetectionValidator
from ultralytics_custom.pycocotools_custom.coco import COCO
from ultralytics_custom.pycocotools_custom.cocoeval_modified import COCOeval

from ates.io import write_json


def main(
    model_path,
    data_yaml,
    save_json,
    imgsz,
    ref_model_path=None,
    save_confusion_matrix=False,
):
    # load ground-truth from json
    with open(data_yaml) as stream:
        gt_json = yaml.safe_load(stream)["val_json"]
    coco_gt = COCO(gt_json)

    # update image id to filename
    for img_dict in coco_gt.loadImgs(coco_gt.getImgIds()):
        stem = Path(img_dict["file_name"]).stem
        image_id = int(stem) if stem.isnumeric() else stem
        for ann_dict in coco_gt.loadAnns(coco_gt.getAnnIds([img_dict["id"]])):
            ann_dict["image_id"] = image_id
        img_dict["id"] = image_id
    coco_gt.createIndex()

    # load model
    model = YOLO(model_path)

    # load validator
    custom = {"rect": True}
    kwargs = {"data": data_yaml, "save_json": True}
    args = {**model.overrides, **custom, **kwargs, "mode": "val", "plots": True}

    if ref_model_path is None:
        validator = DetectionValidator(args=args, _callbacks=model.callbacks)
    else:
        validator = CustomDetectionValidator(
            ref_weight=ref_model_path, args=args, _callbacks=model.callbacks,
        )

    # run validation
    validator(model=model.model)

    if save_confusion_matrix:
        cm = validator.confusion_matrix
        cm.plot(normalize=True, save_dir=os.path.join(os.path.dirname(save_json)))
        cm.plot(normalize=False, save_dir=os.path.join(os.path.dirname(save_json)))

    # get optimal confidence threshold
    metrics = validator.metrics
    conf_ticks = metrics.curves_results[1][0]
    f1 = metrics.curves_results[1][1]
    conf_indices = np.argmax(f1, axis=1)
    opt_conf_thresh_dict = {}
    for i, c in enumerate(metrics.ap_class_index):
        opt_conf_thresh_dict[str(c)] = float(
            conf_ticks[conf_indices[i]]
        )

    # filter predictions
    predictions_coco = []
    for pred in validator.jdict:
        # category id in fisheye data starts from 0
        pred["category_id"] -= 1

        # thresholding with score
        # if no conf threshold value is provided, remove prediction for the class
        if pred["score"] < opt_conf_thresh_dict[str(pred["category_id"])]:
            continue

        predictions_coco.append(pred)

    # get f1 score
    coco_dt = coco_gt.loadRes(predictions_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    f1 = float(coco_eval.stats[20])
    f1_50 = float(coco_eval.stats[21])
    print(f"F1: {f1}")
    print(f"F1@0.5: {f1_50}")

    # save
    if save_json is not None:
        data = {
            **opt_conf_thresh_dict,
            "f1": f1,
            "f1@0.5": f1_50,
        }
        write_json(save_json, data, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_yaml", type=str, required=True)
    parser.add_argument("--save_json", type=str)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--ref_model_path", type=str)
    parser.add_argument("--save_confusion_matrix", action="store_true")
    args = parser.parse_args()

    main(
        args.model_path,
        args.data_yaml,
        args.save_json,
        args.imgsz,
        ref_model_path=args.ref_model_path,
        save_confusion_matrix=args.save_confusion_matrix,
    )
