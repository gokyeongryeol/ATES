#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ates.experiment import ExperimentConfig
from ates.render import render_all_configs


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT_DIR, env=env)


def accelerate_launch(script_path: str, *script_args: str) -> list[str]:
    return ["accelerate", "launch", "--multi_gpu", script_path, *script_args]


def require_path(label: str, path: Path | None) -> Path:
    if path is None:
        raise SystemExit(f"{label} is not configured. Update config/ates/default.yaml first.")
    return path


def stage_render_configs(experiment: ExperimentConfig, _: argparse.Namespace) -> None:
    render_all_configs(experiment)


def stage_mmdet_train(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    stage_render_configs(experiment, args)
    env = os.environ.copy()
    env["ATES_RELAX_MMCV_VERSION"] = "1"
    distributed = experiment.codetr_distributed
    run_command(
        [
            "bash",
            "external/mmdetection/tools/dist_train.sh",
            str(experiment.generated_mmdet_config("fisheye8k_train")),
            str(args.gpus or distributed.nproc_per_node),
            "--work-dir",
            str(experiment.codetr_work_dir),
        ],
        env=env,
    )


def stage_obtain_tmp_pseudo(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    stage_render_configs(experiment, args)
    checkpoint = str(require_path("checkpoints.codetr_finetuned", experiment.codetr_finetuned_checkpoint))
    env = os.environ.copy()
    env["ATES_RELAX_MMCV_VERSION"] = "1"
    run_command(
        [
            "python",
            "tools/obtain_pseudo_label.py",
            str(experiment.generated_mmdet_config("fisheye8k_pl_train_r")),
            checkpoint,
            "",
            str(experiment.tmp_pseudo_json),
        ],
        env=env,
    )


def stage_estimate_threshold(experiment: ExperimentConfig, _: argparse.Namespace) -> None:
    run_command(
        [
            "python",
            "tools/estimate_optimal_threshold.py",
            "--gt_json",
            str(experiment.split_json("train-R")),
            "--pred_json",
            str(experiment.tmp_pseudo_json),
            "--save_json",
            str(experiment.opt_conf_json),
        ]
    )


def stage_extract_and_rephrase(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    run_command(
        accelerate_launch(
            "tools/extract_caption.py",
            "--model_name",
            experiment.caption_model,
            "--base_dir",
            str(experiment.train_d_dir),
            "--json_path",
            str(experiment.split_json("train-D")),
            "--output_path",
            str(experiment.train_d_caption_json),
        )
    )
    run_command(
        accelerate_launch(
            "tools/extract_caption.py",
            "--model_name",
            experiment.caption_model,
            "--base_dir",
            str(experiment.train_r_dir),
            "--json_path",
            str(experiment.split_json("train-R")),
            "--output_path",
            str(experiment.train_r_caption_json),
        )
    )
    run_command(
        accelerate_launch(
            "tools/rephrase_caption.py",
            "--model_name",
            experiment.rephrase_model,
            "--json_path",
            str(experiment.train_r_caption_json),
            "--output_path",
            str(experiment.train_r_rephrased_json),
        )
    )
    run_command(
        accelerate_launch(
            "tools/rephrase_caption.py",
            "--model_name",
            experiment.rephrase_model,
            "--json_path",
            str(experiment.train_r_caption_json),
            "--output_path",
            str(experiment.train_r_rephrased_eval_json),
        )
    )
    if args.include_automatic_v1:
        checkpoint = str(require_path("checkpoints.automatic_rephraser", experiment.automatic_rephraser_checkpoint_dir))
        run_command(
            accelerate_launch(
                "tools/rephrase_caption.py",
                "--model_name",
                experiment.rephrase_model,
                "--json_path",
                str(experiment.train_d_caption_json),
                "--output_path",
                str(experiment.train_d_automatic_json),
                "--ckpt_dir",
                checkpoint,
            )
        )


def stage_synthesize(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    checkpoint = str(require_path("checkpoints.flux_adapter", experiment.flux_checkpoint_dir))
    jobs = [
        (experiment.train_d_caption_json, experiment.naive_v0_dir, True),
        (experiment.train_r_rephrased_json, experiment.rephrased_dir, False),
        (experiment.train_r_rephrased_eval_json, experiment.rephrased_eval_dir, False),
    ]
    if args.include_automatic_v1:
        jobs.append((experiment.train_d_automatic_json, experiment.automatic_v1_dir, False))

    for json_path, output_dir, use_naive in jobs:
        command = accelerate_launch(
            "tools/synthesize_from_text.py",
            "--model_name",
            experiment.generator_model,
            "--json_path",
            str(json_path),
            "--ckpt_dir",
            checkpoint,
            "--output_dir",
            str(output_dir),
        )
        if use_naive:
            command.append("--use_naive")
        run_command(command)


def stage_obtain_pseudo(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    stage_render_configs(experiment, args)
    checkpoint = str(require_path("checkpoints.codetr_finetuned", experiment.codetr_finetuned_checkpoint))
    env = os.environ.copy()
    env["ATES_RELAX_MMCV_VERSION"] = "1"
    jobs = [
        ("fisheye8k_pl_naive_v0", experiment.naive_v0_dir.name),
        ("fisheye8k_pl_rephrased", experiment.rephrased_dir.name),
        ("fisheye8k_pl_rephrased_eval", experiment.rephrased_eval_dir.name),
    ]
    if args.include_automatic_v1:
        jobs.append(("fisheye8k_pl_automatic_v1", experiment.automatic_v1_dir.name))

    for config_name, dataset_name in jobs:
        pseudo_json = experiment.generated_pseudo_json(dataset_name)
        run_command(
            [
                "python",
                "tools/obtain_pseudo_label.py",
                str(experiment.generated_mmdet_config(config_name)),
                checkpoint,
                str(experiment.opt_conf_json),
                str(pseudo_json),
            ],
            env=env,
        )
        run_command(["python", "tools/convert_coco_to_yolo.py", "--base_dir", str(experiment.generated_dir(dataset_name))])


def stage_train_yolo(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    stage_render_configs(experiment, args)
    distributed = experiment.yolo_distributed
    selected = [args.stage_config] if args.stage_config else experiment.yolo_train_configs
    for config_name in selected:
        run_command(
            [
                "python",
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={distributed.nproc_per_node}",
                f"--master_port={distributed.master_port}",
                "tools/train_base_detector.py",
                "--run-name",
                f"yolo11s_{config_name}",
                "--yaml-file",
                str(experiment.generated_ultralytics_config(config_name)),
                "--init-weight",
                str(experiment.yolo_init_weights),
                "--optimizer",
                experiment.yolo_training.optimizer,
                "--lr",
                str(experiment.yolo_training.lr),
                "--weight-decay",
                str(experiment.yolo_training.weight_decay),
                "--img-sz",
                str(experiment.yolo_training.image_size),
                "--device",
                distributed.devices,
                "--seed",
                str(experiment.yolo_training.seed),
            ]
        )

        run_command([
            "mv",
            f"runs/detect/yolo11s_{config_name}",
            f"ckpt/yolo/yolo11s_{config_name}",
        ])


def stage_construct_preference(experiment: ExperimentConfig, _: argparse.Namespace) -> None:
    jobs = [
        (experiment.train_r_rephrased_json, experiment.rephrased_dir, experiment.preference_root / "train"),
        (experiment.train_r_rephrased_eval_json, experiment.rephrased_eval_dir, experiment.preference_root / "test"),
    ]
    for json_path, base_dir, output_dir in jobs:
        run_command(
            [
                "python",
                "tools/construct_dataset.py",
                "--json_path",
                str(json_path),
                "--base_dir",
                str(base_dir),
                "--ckpt_dir",
                str(experiment.yolo_run_dir("fisheye8k_with_naive_v0")),
                "--output_dir",
                str(output_dir),
            ]
        )
    run_command(
        [
            "python",
            "tools/create_dataset_dict.py",
            "--json_path",
            str(experiment.preference_root / "dataset_dict.json"),
        ]
    )


def stage_train_dpo(experiment: ExperimentConfig, _: argparse.Namespace) -> None:
    env = os.environ.copy()
    run_command(
        [
            "accelerate",
            "launch",
            "--multi_gpu",
            "tools/train_dpo.py",
            "--dataset_name",
            str(experiment.preference_root),
            "--dataset_streaming",
            "--model_name_or_path",
            experiment.rephrase_model,
            "--learning_rate",
            "5.0e-6",
            "--max_steps",
            "1000",
            "--per_device_train_batch_size",
            "2",
            "--gradient_accumulation_steps",
            "8",
            "--eval_strategy",
            "steps",
            "--eval_steps",
            "100",
            "--save_steps",
            "100",
            "--output_dir",
            str(experiment.dpo_output_dir),
            "--no_remove_unused_columns",
            "--use_peft",
            "--lora_r",
            "32",
            "--lora_alpha",
            "16",
        ],
        env=env,
    )


def stage_eval(experiment: ExperimentConfig, args: argparse.Namespace) -> None:
    stage_render_configs(experiment, args)
    selected = [args.stage_config] if args.stage_config else experiment.yolo_eval_configs
    for config_name in selected:
        command = [
            "python",
            "tools/eval_metrics.py",
            "--model_path",
            str(experiment.yolo_run_dir(config_name) / "weights" / "best.pt"),
            "--data_yaml",
            str(experiment.generated_ultralytics_config(config_name)),
            "--save_json",
            str(experiment.yolo_result_json(config_name)),
            "--imgsz",
            str(experiment.yolo_training.image_size),
        ]
        if args.ref_model_path:
            command.extend(["--ref_model_path", args.ref_model_path])
        run_command(command)


def stage_print_config(experiment: ExperimentConfig, _: argparse.Namespace) -> None:
    print(experiment.to_summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATES pipeline runner")
    parser.add_argument(
        "--experiment-config",
        dest="experiment_config",
        default=str(ROOT_DIR / "config" / "ates" / "default.yaml"),
        help="Path to the experiment config YAML.",
    )

    subparsers = parser.add_subparsers(dest="stage", required=True)

    subparsers.add_parser("render-configs")

    train_mmdet = subparsers.add_parser("mmdet-train")
    train_mmdet.add_argument("--gpus", type=int, default=None)

    subparsers.add_parser("obtain-tmp-pseudo")
    subparsers.add_parser("estimate-threshold")

    extract = subparsers.add_parser("extract-and-rephrase")
    extract.add_argument("--include-automatic-v1", action="store_true")

    synthesize = subparsers.add_parser("synthesize")
    synthesize.add_argument("--include-automatic-v1", action="store_true")

    pseudo = subparsers.add_parser("obtain-pseudo")
    pseudo.add_argument("--include-automatic-v1", action="store_true")

    train_yolo = subparsers.add_parser("train-yolo")
    train_yolo.add_argument("--config", dest="stage_config", default=None)

    subparsers.add_parser("construct-preference")
    subparsers.add_parser("train-llama")

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--config", dest="stage_config", default=None)
    evaluate.add_argument("--ref-model-path", default=None)

    subparsers.add_parser("print-config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    experiment = ExperimentConfig.from_file(args.experiment_config)
    stage_handlers = {
        "render-configs": stage_render_configs,
        "mmdet-train": stage_mmdet_train,
        "obtain-tmp-pseudo": stage_obtain_tmp_pseudo,
        "estimate-threshold": stage_estimate_threshold,
        "extract-and-rephrase": stage_extract_and_rephrase,
        "synthesize": stage_synthesize,
        "obtain-pseudo": stage_obtain_pseudo,
        "train-yolo": stage_train_yolo,
        "construct-preference": stage_construct_preference,
        "train-llama": stage_train_dpo,
        "eval": stage_eval,
        "print-config": stage_print_config,
    }
    stage_handlers[args.stage](experiment, args)


if __name__ == "__main__":
    main()
