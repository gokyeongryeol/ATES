from typing import List, Dict, Any
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data import converter

from ultralytics_custom.pycocotools_custom.coco import COCO


class DetectionValidatorFromCOCOPrediction(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)

    def xywh2xyxy(self, bbox):
        x1, y1, w, h = bbox
        return [x1, y1, x1 + w, y1 + h]

    def init_metrics(self, names, is_coco=False):
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        self.is_coco = is_coco
        self.class_map = (
            converter.coco80_to_coco91_class()
            if self.is_coco
            else list(range(1, len(names) + 1))
        )
        self.args.save_json |= (
            self.args.val and (self.is_coco or self.is_lvis) and not self.training
        )  # run final val
        self.names = names
        self.nc = len(names)
        self.seen = 0
        self.jdict = []
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(names=list(names.values()))

    def update_metrics(
        self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]
    ) -> None:
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            batch (Dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = batch
            predn = pred
            # pbatch = self._prepare_batch(si, batch)
            # predn = self._prepare_pred(pred, pbatch)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            if no_pred and len(cls) == 0:
                continue
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)

            if no_pred:
                continue

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )

    def eval_from_coco_predictions(self, pred_json, gt_json):
        coco = COCO(gt_json)

        # indexing
        tmp_pred_data = json.load(open(pred_json, "r"))
        if isinstance(tmp_pred_data, dict) and "annotations" in tmp_pred_data:
            tmp_pred_data = tmp_pred_data["annotations"]
        pred_data = {}
        for x in tmp_pred_data:
            if x["image_id"] not in pred_data:
                pred_data[x["image_id"]] = []
            pred_data[x["image_id"]].append(x)

        # init
        names = {x["id"]: x["name"] for x in coco.dataset["categories"]}
        self.init_metrics(names)

        # eval
        for image_id, pred_list in tqdm(pred_data.items()):
            pred_bboxes = [self.xywh2xyxy(x["bbox"]) for x in pred_list]
            pred_conf = [x["score"] for x in pred_list]
            pred_cls = [x["category_id"] for x in pred_list]
            preds = {
                "bboxes": torch.Tensor(pred_bboxes),
                "conf": torch.Tensor(pred_conf),
                "cls": torch.Tensor(pred_cls),
            }

            gt_ann_list = coco.loadAnns(coco.getAnnIds([image_id]))
            batch = {
                "bboxes": torch.Tensor(
                    [self.xywh2xyxy(x["bbox"]) for x in gt_ann_list]
                ),
                "cls": torch.Tensor([x["category_id"] for x in gt_ann_list]),
            }

            self.update_metrics([preds], batch)

        stats = self.get_stats()
        self.finalize_metrics()
        return stats
