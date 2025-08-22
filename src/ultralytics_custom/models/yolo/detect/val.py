import numpy as np
import torch

from copy import deepcopy
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import select_device


def pair_boxes(gt_boxes, pred_boxes, device, iou_threshold):
    iou_matrix = box_iou(gt_boxes, pred_boxes)
    if len(iou_matrix) > 0:
        matched_vals, match_idxs = iou_matrix.max(dim=0)
    else:
        matched_vals = torch.zeros([len(pred_boxes)], device=device)
        match_idxs = torch.zeros([len(pred_boxes)], device=device, dtype=torch.int64)

    is_matched = match_idxs.new_full(match_idxs.size(), 1, dtype=torch.int8)

    for l, low, high in zip([0,1], [-float("inf"), iou_threshold], [iou_threshold, float("inf")]):
        low_high = (matched_vals >= low) & (matched_vals < high)
        is_matched[low_high] = l

    is_matched = is_matched.bool()
    return match_idxs, is_matched


class CustomDetectionValidator(DetectionValidator):
    def __init__(self, *args, ref_weight, **kwargs):
        super().__init__(*args, **kwargs)

        device = select_device(self.args.device, self.args.batch)
        self.ref_model = YOLO(ref_weight).to(device)

    def _prepare_batch(self, si, batch):
        """Prepares a batch for training or inference by applying transformations."""
        device = batch["img"].device

        prediction = self.ref_model(batch["img"][si:si+1], verbose=False)[0]
        pred_boxes = prediction.boxes.xyxy
        pred_classes = prediction.boxes.cls

        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]

        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]

        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]

            gt_boxes = bbox
            gt_classes = cls

            match_idxs, is_matched = pair_boxes(gt_boxes, pred_boxes, device, iou_threshold=0.95)
            match_labels = gt_classes[match_idxs[is_matched]]
            is_fp_cls = pred_classes[is_matched] != match_labels

            gt_idxs = torch.arange(len(gt_classes), device=device)
            tp_idxs = torch.unique(gt_idxs[match_idxs[is_matched]][~is_fp_cls])

            is_non_tp = torch.ones(len(gt_classes), device=device, dtype=torch.bool)
            is_non_tp[tp_idxs] = False

            drop_cls, drop_bbox = cls[tp_idxs], bbox[tp_idxs]
            cls, bbox = cls[is_non_tp], bbox[is_non_tp]
            tot_cls = torch.cat([drop_cls, cls])
            tot_bbox = torch.cat([drop_bbox, bbox])
            num_tp = len(drop_cls)
        else:
            tot_cls = cls
            tot_bbox = bbox
            num_tp = 0

        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
            "tot_cls": tot_cls,
            "tot_bbox": tot_bbox,
            "num_tp": num_tp,
        }

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch with transformed bounding boxes and class labels."""
        device = pbatch["tot_cls"].device

        gt_boxes = pbatch["tot_bbox"]
        gt_classes = pbatch["tot_cls"]
        num_tp = pbatch["num_tp"]
        num_gt = len(gt_classes)

        predn = deepcopy(pred)
        pred_boxes = predn["bboxes"]
        pred_classes = predn["cls"]

        match_idxs, is_matched = pair_boxes(gt_boxes, pred_boxes, device, iou_threshold=0.5)

        is_filtered = torch.zeros(len(pred_classes), device=device, dtype=torch.bool)
        for idx in range(num_tp, num_gt):
            is_filtered = torch.logical_or(is_filtered, match_idxs == idx)
        is_filtered = torch.logical_or(
            is_filtered, torch.logical_and(~is_filtered, ~is_matched),
        )

        predn = {k:v[is_filtered].float() for k,v in predn.items()}

        return predn
