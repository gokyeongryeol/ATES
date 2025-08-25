import argparse

from ultralytics import YOLO, SETTINGS
from ultralytics_custom.models.yolo.detect import CustomDetectionTrainer


def main(args):
    if args.report_to == "wandb":
        SETTINGS["wandb"] = True
    else:
        SETTINGS["wandb"] = False

    model = YOLO(args.init_weight)
    model.train(
        trainer=CustomDetectionTrainer,
        project=args.project_name,
        name=args.run_name,
        data=args.yaml_file,
        optimizer=args.optimizer,
        lr0=args.lr,
        weight_decay=args.weight_decay,
        imgsz=args.img_sz,
        device=args.device,
        seed=args.seed,
        save_json=True,
        save_dir=f"ckpt/yolo/{args.run_name}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # logging
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
    )

    # data
    parser.add_argument(
        "--yaml-file",
        type=str,
        required=True,
    )

    # hyper-parameters
    parser.add_argument(
        "--init-weight",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adamw"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--img-sz",
        type=int,
        nargs="+",
        default=[1280],
    )

    # others
    parser.add_argument(
        "--device",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2035,
    )
    args = parser.parse_args()

    main(args)
