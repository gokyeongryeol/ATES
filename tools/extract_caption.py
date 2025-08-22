import argparse
import os
import json
import torch
import torchvision.transforms as T
import transformers
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from PIL import Image

transformers.utils.logging.set_verbosity(40)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROMPT = """
You are an expert in generating high-quality image captions. Please analyze the provided fish-eye image in detail.
Please adhere to the following format for the caption:
- Start with ”A photo of”.
- Limit the total length to 40-50 words.
- Focus on bus, bike, car, pedestrian and truck.
- Describe time of day, weather and location.
- Focus on the scene inside the fish-eye lens.
- Use grammatically correct and clear sentences.
"""

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_internvl(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().to(device)
    return tokenizer, model


def predict_internvl(img_path, tokenizer, model, device):
    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).to(device)
    question = f"<image>\n{PROMPT}"

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    output_text = model.chat(tokenizer, pixel_values, question, generation_config)
    return output_text


class ImgPathDataset(TorchDataset):
    def __init__(self, image_dicts, base_dir):
        self.image_dicts = image_dicts
        self.base_dir = base_dir

    def __len__(self):
        return len(self.image_dicts)

    def __getitem__(self, idx):
        file_name = self.image_dicts[idx]['file_name']
        img_path = os.path.join(self.base_dir, 'images', file_name)
        return idx, img_path


class CaptionAssistent:
    def __init__(self, model_name, base_dir, json_path, output_path):
        self.accelerator = Accelerator()
        self.model_name = model_name
        self.base_dir = base_dir
        self.json_path = json_path
        self.output_path = output_path

        self.data = json.load(open(self.json_path, 'r'))
        self.image_dicts = self.data['images']
        self.dataset = ImgPathDataset(self.image_dicts, base_dir)

        self.device = self.accelerator.device
        self.tokenizer, self.model = load_internvl(model_name, self.device)

    def run_inference(self, batch_size=1):
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)

        output_list = []
        for batch in tqdm(dataloader, desc=f"Rank {self.accelerator.process_index}"):
            indices, img_paths = batch
            for idx, img_path in zip(indices, img_paths):
                caption = predict_internvl(img_path, self.tokenizer, self.model, self.device)
                img_dict = self.image_dicts[idx]
                img_dict['caption'] = caption
                output_list.append(img_dict)

        self.accelerator.wait_for_everyone()
        all_outputs = gather_object(output_list)

        if self.accelerator.is_main_process:
            self.data['images'] = list(all_outputs)
            with open(self.output_path, 'w') as f:
                json.dump(self.data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help="VLM to extract captions from an image")
    parser.add_argument('--base_dir', help="directory of which the source dataset is located")
    parser.add_argument('--json_path', help="json annotation file path without captions")
    parser.add_argument('--output_path', help="json annotation file path with captions")
    args = parser.parse_args()

    assistent = CaptionAssistent(
        model_name=args.model_name,
        base_dir=args.base_dir,
        json_path=args.json_path,
        output_path=args.output_path,
    )
    assistent.run_inference()
