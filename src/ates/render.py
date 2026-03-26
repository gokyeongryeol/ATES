from __future__ import annotations

import os
from pathlib import Path

import yaml

from ates.experiment import ExperimentConfig

CLASS_NAMES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]
MMDET_BASE = "../../external/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py"


def render_all_configs(experiment: ExperimentConfig) -> None:
    render_ultralytics_configs(experiment)
    render_mmdetection_configs(experiment)


def render_ultralytics_configs(experiment: ExperimentConfig) -> None:
    output_dir = experiment.generated_ultralytics_config_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "fisheye8k": {
            "train": str(experiment.train_d_dir / "images"),
            "val": str(experiment.test_dir / "images"),
            "val_json": str(experiment.test_dir / "test.json"),
        },
        "fisheye8k_with_naive_v0": {
            "train": [
                str(experiment.train_d_dir / "images"),
                str(experiment.naive_v0_dir / "images"),
            ],
            "val": str(experiment.test_dir / "images"),
            "val_json": str(experiment.test_dir / "test.json"),
        },
        "fisheye8k_with_naive_v0+automatic_v1": {
            "train": [
                str(experiment.train_d_dir / "images"),
                str(experiment.naive_v0_dir / "images"),
                str(experiment.automatic_v1_dir / "images"),
            ],
            "val": str(experiment.test_dir / "images"),
            "val_json": str(experiment.test_dir / "test.json"),
        },
    }

    for name, config in configs.items():
        payload = {"names": CLASS_NAMES, "nc": len(CLASS_NAMES), **config}
        with (output_dir / f"{name}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)


def render_mmdetection_configs(experiment: ExperimentConfig) -> None:
    output_dir = experiment.generated_mmdet_config_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "fisheye8k_train": render_mmdet_training_config(
            data_root=experiment.data_root,
            pretrained_checkpoint=maybe_relpath(experiment.codetr_pretrained_checkpoint, experiment.root_dir),
        ),
        "fisheye8k_pl_train_r": render_mmdet_inference_config(
            data_root=experiment.data_root,
            pretrained_checkpoint=maybe_relpath(experiment.codetr_pretrained_checkpoint, experiment.root_dir),
            ann_file="train-R/train-R.json",
            img_prefix="train-R/images/",
        ),
        "fisheye8k_pl_naive_v0": render_mmdet_inference_config(
            data_root=experiment.data_root,
            pretrained_checkpoint=maybe_relpath(experiment.codetr_pretrained_checkpoint, experiment.root_dir),
            ann_file="train-D_naive_v0-gen/train-D_naive_v0-gen_with_dummy.json",
            img_prefix="train-D_naive_v0-gen/images/",
        ),
        "fisheye8k_pl_rephrased": render_mmdet_inference_config(
            data_root=experiment.data_root,
            pretrained_checkpoint=maybe_relpath(experiment.codetr_pretrained_checkpoint, experiment.root_dir),
            ann_file="train-R_rephrased-gen/train-R_rephrased-gen_with_dummy.json",
            img_prefix="train-R_rephrased-gen/images/",
        ),
        "fisheye8k_pl_rephrased_eval": render_mmdet_inference_config(
            data_root=experiment.data_root,
            pretrained_checkpoint=maybe_relpath(experiment.codetr_pretrained_checkpoint, experiment.root_dir),
            ann_file="train-R_rephrased-gen-eval/train-R_rephrased-gen-eval_with_dummy.json",
            img_prefix="train-R_rephrased-gen-eval/images/",
        ),
        "fisheye8k_pl_automatic_v1": render_mmdet_inference_config(
            data_root=experiment.data_root,
            pretrained_checkpoint=maybe_relpath(experiment.codetr_pretrained_checkpoint, experiment.root_dir),
            ann_file="train-D_automatic_v1-gen/train-D_automatic_v1-gen_with_dummy.json",
            img_prefix="train-D_automatic_v1-gen/images/",
        ),
    }

    for name, content in configs.items():
        (output_dir / f"{name}.py").write_text(content, encoding="utf-8")


def maybe_relpath(path: Path, root_dir: Path) -> str:
    try:
        return os.path.relpath(path, root_dir)
    except ValueError:
        return path.as_posix()


def render_mmdet_training_config(*, data_root: Path, pretrained_checkpoint: str) -> str:
    return f"""_base_ = ['{MMDET_BASE}']

load_from = {pretrained_checkpoint!r}

num_dec_layer = 6
loss_lambda = 2.0
num_classes = 5
model = dict(
    query_head=dict(num_classes=num_classes),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56,
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda,
                ),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda,
                ),
            ),
        )
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda,
            ),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda,
            ),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda,
            ),
        ),
    ],
)

data_root = {data_root.as_posix()!r}
metainfo = dict(classes={tuple(CLASS_NAMES)!r})

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train-D/train-D.json',
        data_prefix=dict(img='train-D/images/'),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train-R/train-R.json',
        data_prefix=dict(img='train-R/images/'),
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + '/train-R/train-R.json')
test_evaluator = dict(ann_file=data_root + '/train-R/train-R.json')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=-1,
    )
)
"""


def render_mmdet_inference_config(
    *,
    data_root: Path,
    pretrained_checkpoint: str,
    ann_file: str,
    img_prefix: str,
) -> str:
    return f"""_base_ = ['{MMDET_BASE}']

load_from = {pretrained_checkpoint!r}

num_dec_layer = 6
loss_lambda = 2.0
num_classes = 5
model = dict(
    query_head=dict(num_classes=num_classes),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56,
            ),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda,
                ),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda,
                ),
            ),
        )
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda,
            ),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda,
            ),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda,
            ),
        ),
    ],
)

data_root = {data_root.as_posix()!r}
metainfo = dict(classes={tuple(CLASS_NAMES)!r})

test_cfg = dict(type='InferenceLoop')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 2048), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file={ann_file!r},
        data_prefix=dict(img={img_prefix!r}),
        pipeline=test_pipeline,
    ),
)

test_evaluator = dict(ann_file=data_root + '/' + {ann_file!r})

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP',
        rule='greater',
    )
)
"""
