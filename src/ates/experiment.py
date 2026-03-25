from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    project_name: str
    optimizer: str
    lr: float
    weight_decay: float
    image_size: int
    seed: int


@dataclass(frozen=True)
class DistributedConfig:
    nproc_per_node: int
    devices: str
    master_port: int


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path
    config_path: Path
    data_root: Path
    work_root: Path
    codetr_work_dir: Path
    codetr_pretrained_checkpoint: Path
    codetr_finetuned_checkpoint: Path | None
    flux_checkpoint_dir: Path | None
    automatic_rephraser_checkpoint_dir: Path | None
    caption_model: str
    rephrase_model: str
    generator_model: str
    yolo_init_weights: Path
    codetr_distributed: DistributedConfig
    yolo_train_configs: list[str]
    yolo_eval_configs: list[str]
    yolo_training: TrainingConfig
    yolo_distributed: DistributedConfig
    dpo_output_dir: Path
    dpo_run_name: str
    dpo_wandb_project: str

    @classmethod
    def from_file(cls, config_path: str | Path) -> "ExperimentConfig":
        config_path = Path(config_path).resolve()
        root_dir = config_path.parents[2]
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        paths = payload["paths"]
        models = payload["models"]
        checkpoints = payload["checkpoints"]
        yolo = payload["yolo"]
        codetr = payload["codetr"]
        dpo = payload["dpo"]

        data_root = cls._resolve_path(root_dir, paths["data_root"])
        work_root = cls._resolve_path(root_dir, paths["work_root"])
        codetr_work_dir = cls._resolve_path(root_dir, checkpoints["codetr_work_dir"])
        codetr_pretrained_checkpoint = cls._resolve_path(root_dir, checkpoints["codetr_pretrained"])

        return cls(
            root_dir=root_dir,
            config_path=config_path,
            data_root=data_root,
            work_root=work_root,
            codetr_work_dir=codetr_work_dir,
            codetr_pretrained_checkpoint=codetr_pretrained_checkpoint,
            codetr_finetuned_checkpoint=cls._resolve_optional_path(root_dir, checkpoints.get("codetr_finetuned")),
            flux_checkpoint_dir=cls._resolve_optional_path(root_dir, checkpoints.get("flux_adapter")),
            automatic_rephraser_checkpoint_dir=cls._resolve_optional_path(root_dir, checkpoints.get("automatic_rephraser")),
            caption_model=models["caption_model"],
            rephrase_model=models["rephrase_model"],
            generator_model=models["generator_model"],
            yolo_init_weights=cls._resolve_path(root_dir, checkpoints["yolo_init_weights"]),
            codetr_distributed=DistributedConfig(
                nproc_per_node=int(codetr["distributed"]["nproc_per_node"]),
                devices=str(codetr["distributed"]["devices"]),
                master_port=int(codetr["distributed"]["master_port"]),
            ),
            yolo_train_configs=list(yolo["train_configs"]),
            yolo_eval_configs=list(yolo["eval_configs"]),
            yolo_training=TrainingConfig(
                project_name=yolo["project_name"],
                optimizer=yolo["optimizer"],
                lr=float(yolo["lr"]),
                weight_decay=float(yolo["weight_decay"]),
                image_size=int(yolo["image_size"]),
                seed=int(yolo["seed"]),
            ),
            yolo_distributed=DistributedConfig(
                nproc_per_node=int(yolo["distributed"]["nproc_per_node"]),
                devices=str(yolo["distributed"]["devices"]),
                master_port=int(yolo["distributed"]["master_port"]),
            ),
            dpo_output_dir=cls._resolve_path(root_dir, dpo["output_dir"]),
            dpo_run_name=dpo["run_name"],
            dpo_wandb_project=dpo["wandb_project"],
        )

    @staticmethod
    def _resolve_path(root_dir: Path, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (root_dir / candidate).resolve()

    @staticmethod
    def _resolve_optional_path(root_dir: Path, value: str | None) -> Path | None:
        if not value:
            return None
        return ExperimentConfig._resolve_path(root_dir, value)

    def split_dir(self, name: str) -> Path:
        return self.data_root / name

    def split_json(self, split_name: str) -> Path:
        return self.split_dir(split_name) / f"{split_name}.json"

    @property
    def train_d_dir(self) -> Path:
        return self.split_dir("train-D")

    @property
    def train_r_dir(self) -> Path:
        return self.split_dir("train-R")

    @property
    def test_dir(self) -> Path:
        return self.split_dir("test")

    @property
    def train_d_caption_json(self) -> Path:
        return self.train_d_dir / "train-D_with_caption.json"

    @property
    def train_r_caption_json(self) -> Path:
        return self.train_r_dir / "train-R_with_caption.json"

    @property
    def train_r_rephrased_json(self) -> Path:
        return self.train_r_dir / "train-R_with_rephrased.json"

    @property
    def train_r_rephrased_eval_json(self) -> Path:
        return self.train_r_dir / "train-R_with_rephrased-eval.json"

    @property
    def train_d_automatic_json(self) -> Path:
        return self.train_d_dir / "train-D_with_automatic_v1.json"

    @property
    def opt_conf_json(self) -> Path:
        return self.train_r_dir / "opt_conf_thr.json"

    @property
    def tmp_pseudo_json(self) -> Path:
        return self.train_r_dir / "train-R_with_tmp_pseudolabel.json"

    def generated_dir(self, name: str) -> Path:
        return self.data_root / name

    def generated_dummy_json(self, name: str) -> Path:
        return self.generated_dir(name) / f"{name}_with_dummy.json"

    def generated_pseudo_json(self, name: str) -> Path:
        return self.generated_dir(name) / f"{name}_with_pseudolabel.json"

    @property
    def naive_v0_dir(self) -> Path:
        return self.generated_dir("train-D_naive_v0-gen")

    @property
    def rephrased_dir(self) -> Path:
        return self.generated_dir("train-R_rephrased-gen")

    @property
    def rephrased_eval_dir(self) -> Path:
        return self.generated_dir("train-R_rephrased-gen-eval")

    @property
    def automatic_v1_dir(self) -> Path:
        return self.generated_dir("train-D_automatic_v1-gen")

    @property
    def preference_root(self) -> Path:
        return self.data_root / "train-R_preference_with_naive_v0"

    def yolo_run_dir(self, config_name: str) -> Path:
        return self.root_dir / "ckpt" / "yolo" / f"yolo11s_{config_name}"

    def yolo_result_json(self, config_name: str) -> Path:
        return self.yolo_run_dir(config_name) / "result.json"

    def generated_ultralytics_config_dir(self) -> Path:
        return self.root_dir / "config" / "ultralytics"

    def generated_mmdet_config_dir(self) -> Path:
        return self.root_dir / "config" / "mmdetection"

    def generated_ultralytics_config(self, name: str) -> Path:
        return self.generated_ultralytics_config_dir() / f"{name}.yaml"

    def generated_mmdet_config(self, name: str) -> Path:
        return self.generated_mmdet_config_dir() / f"{name}.py"

    def to_summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "data_root": str(self.data_root),
            "codetr_work_dir": str(self.codetr_work_dir),
            "codetr_pretrained_checkpoint": str(self.codetr_pretrained_checkpoint),
            "codetr_finetuned_checkpoint": str(self.codetr_finetuned_checkpoint) if self.codetr_finetuned_checkpoint else None,
            "flux_checkpoint_dir": str(self.flux_checkpoint_dir) if self.flux_checkpoint_dir else None,
            "automatic_rephraser_checkpoint_dir": str(self.automatic_rephraser_checkpoint_dir) if self.automatic_rephraser_checkpoint_dir else None,
            "caption_model": self.caption_model,
            "rephrase_model": self.rephrase_model,
            "generator_model": self.generator_model,
            "codetr_distributed": {
                "nproc_per_node": self.codetr_distributed.nproc_per_node,
                "devices": self.codetr_distributed.devices,
                "master_port": self.codetr_distributed.master_port,
            },
            "yolo_train_configs": self.yolo_train_configs,
            "yolo_eval_configs": self.yolo_eval_configs,
        }
