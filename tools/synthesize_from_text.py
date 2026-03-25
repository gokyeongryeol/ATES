import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from diffusers import FluxPipeline, FluxTransformer2DModel
from lycoris import create_lycoris_from_weights

from ates.io import dedupe_image_dicts, load_json, write_json


class PromptDataset(TorchDataset):
    def __init__(self, image_dicts, use_naive):
        self.image_dicts = image_dicts
        self.use_naive = use_naive

    def __len__(self):
        return len(self.image_dicts)

    def __getitem__(self, idx):
        img_dict = self.image_dicts[idx]
        img_id = img_dict['id']
        file_name = img_dict['file_name']
        if self.use_naive:
            prompt = [img_dict['caption']] * 5
        else:
            prompt = img_dict['rephrased']
        return img_id, file_name, prompt


class ImageSynthesizer:
    def __init__(self, model_name, json_path, ckpt_dir, output_dir, use_naive=False):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        transformer = FluxTransformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        if ckpt_dir:
            lycoris_config = load_json(os.path.join(ckpt_dir, "lycoris_config.json"))
            multiplier = int(lycoris_config.pop("multiplier"))
            lycoris_wrapped_network = create_lycoris_from_weights(
                multiplier,
                os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"),
                transformer,
                weights_sd=None,
                **lycoris_config,
            )[0]
            lycoris_wrapped_network.merge_to()

        self.pipe = FluxPipeline.from_pretrained(
            model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.data = load_json(json_path)
        self.image_dicts = self.data['images']
        self.dataset = PromptDataset(self.image_dicts, use_naive=use_naive)

        img_id_lst = [img_dict['id'] for img_dict in self.image_dicts]
        self.min_img_id = min(img_id_lst)
        self.max_img_id = max(img_id_lst)

        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

    def run_inference(self, batch_size=1):
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False
        )
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)

        output_list = []
        for batch in tqdm(dataloader, desc=f"Rank {self.accelerator.process_index} Generating"):
            img_id_lst, file_name_lst, prompt_lst = batch
            prompt_lst = [[prompt[i] for prompt in prompt_lst] for i in range(batch_size)]
            assert len(img_id_lst) == len(file_name_lst) == len(prompt_lst)

            for img_id, file_name, prompt in zip(img_id_lst, file_name_lst, prompt_lst):
                file_name, ext = os.path.basename(file_name).split('.')
                images = self.pipe(
                    prompt=list(prompt),
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=20,
                    max_sequence_length=512,
                ).images

                for i, image in enumerate(images):
                    save_path = os.path.join(self.output_dir, "images", f"{file_name}-{i}.{ext}")
                    image.save(save_path)

                    width, height = image.size
                    img_dict = {
                        'file_name': os.path.basename(save_path),
                        'height': height,
                        'width': width,
                        'id': img_id.item() + (i+1) * (self.max_img_id - self.min_img_id + 1),
                    }
                    output_list.append(img_dict)

        self.accelerator.wait_for_everyone()
        all_outputs = gather_object(output_list)

        if self.accelerator.is_main_process:
            self.data['images'] = dedupe_image_dicts(list(all_outputs))
            self.data['annotations'] = []
            output_file = os.path.basename(self.output_dir) + "_with_dummy.json"
            output_path = os.path.join(self.output_dir, output_file)
            write_json(output_path, self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help="DM to synthesize images")
    parser.add_argument('--json_path', help="json annotation file path")
    parser.add_argument('--ckpt_dir', help="directory of which the checkpoint of the trained t2i-generator is located")
    parser.add_argument('--output_dir', help="directory to which the synthesized dataset would be saved")
    parser.add_argument('--use_naive', action="store_true", help="whether to use the caption duplicates rather than the rephrased captions")
    args = parser.parse_args()

    synthesizer = ImageSynthesizer(
        model_name=args.model_name,
        json_path=args.json_path,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        use_naive=args.use_naive,
    )
    synthesizer.run_inference()
