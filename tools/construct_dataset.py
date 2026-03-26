import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ates.io import load_json
from ates.prompts import DPO_SYSTEM_PROMPT


def run_evaluation(json_path, base_dir, ckpt_dir, output_dir):
    device = "cuda"

    data = load_json(json_path)
    fn_to_captions = {
        img_dict['file_name'].split('.')[0]: (img_dict['caption'], img_dict['rephrased'])
        for img_dict in data['images']
    }

    dataset = YOLODataset(
        data=check_det_dataset("./config/ultralytics/fisheye8k.yaml"),
        img_path=os.path.join(base_dir, "images"),
        augment=False,
        imgsz=1280,
        task="detect",
    )

    model = YOLO(f"{ckpt_dir}/weights/best.pt")
    model.to(device)
    model.model.args = get_cfg(f"{ckpt_dir}/args.yaml")
    model.model.criterion = model.model.init_criterion()

    all_results = defaultdict(dict)

    for sample in tqdm(dataset):
        sample["img"] = sample["img"].to(device).unsqueeze(dim=0).float() / 255
        sample["img"] = sample["img"].contiguous()
        loss = model.model.loss(sample)[0].sum().item()

        file_name, ext = os.path.basename(sample['im_file']).split('.')
        file_name, i = file_name.split('-')
        caption, rephrased = fn_to_captions[file_name]
        all_results[caption][rephrased[int(i)]] = loss

    preference = defaultdict(list)
    for caption, v in all_results.items():
        rephrased = list(v.keys())
        loss = list(v.values())

        indices = sorted(range(len(loss)), key=lambda k: loss[k])
        min_idx, max_idx = indices[0], indices[-1]

        preference['prompt'].append([
            {"role": "system", "content": DPO_SYSTEM_PROMPT},
            {"role": "user", "content": caption},
        ])
        preference['rejected'].append([
            {"role": "assistant", "content": rephrased[min_idx]},
        ])
        preference['chosen'].append([
            {"role": "assistant", "content": rephrased[max_idx]},
        ])

    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset.from_dict(preference)
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', help="json annotation file path with rephrased captions")
    parser.add_argument('--base_dir', help="directory of which the synthesized dataset is located")
    parser.add_argument('--ckpt_dir', help="directory of which the checkpoint of the trained base detector is located")
    parser.add_argument('--output_dir', help="directory to which the preference dataset would be saved")
    args = parser.parse_args()

    run_evaluation(
        json_path=args.json_path,
        base_dir=args.base_dir,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
    )
