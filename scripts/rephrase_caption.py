import argparse
import re
import json
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm

MANUAL_PROMPT = """
You are an expert at visually-grounded image caption rewriting. Your task is to rewrite image captions taken with a fisheye camera so that:
- The objects Bus, Bike, Car, Pedestrian, and Truck are small in scale and located near the edges of the image, where fisheye distortion is strong.
- Object placement and interaction should be plausible within real-world traffic scenes.

Please adhere to the following format for the caption:
- Start with ”A photo of”.
- Limit the total length to 40-50 words.
- Use grammatically correct and clear sentences.

Apply the following object descriptions:
- Bus: large public passenger vehicles
- Truck: heavy-duty vehicles like dump trucks or semi-trailers
- Car: compact vehicles such as sedans, SUVs, or vans
- Bike: bicycles, motorcycles, or scooters, either parked or with riders
- Pedestrian: visible people walking, standing, or crossing

Avoid visual ambiguity between:
- Bus vs Truck → contrast size, function, silhouette
- Car vs Truck → emphasize bulk and height differences
- Pedestrian vs Bike → distinguish by motion, vehicle presence, posture

Ensure variation across scene conditions:
- Camera angles: side-view or front-view
- Intersection types: T-junctions, Y-junctions, cross-intersections, mid-blocks, pedestrian crossings, or irregular layouts
- Lighting: morning, afternoon, evening, or night
- Traffic flow: free-flowing, steady, or busy

When choosing scene elements, slightly favor the following (but still maintain diversity overall):
- Categories prominently shown: Pedestrian and Truck
- Time of day: Day, Afternoon, Night, or general Daytime
- Weather: Clear
- Location type: Urban areas such as City streets

Preserve the core content of the original caption, but rewrite it to reflect the above constraints. If none of the specified categories are present, you may subtly introduce one or more at the distorted outer edges of the image. Always maintain natural, fluent language, and don’t make the added objects the main focus unless already emphasized.
"""

AUTOMATIC_PROMPT = """
You are an expert in generating diverse yet realistic image captions for road scenes captured by a fish-eye surveillance camera.
Your task is to rewrite the caption I give you into a realistic variant.

Please adhere to the following format for the caption:
- Start with ”A photo of”.
- Limit the total length to 40-50 words.
- Use grammatically correct and clear sentences.
- While preserving the core elements (bus, bike, car, pedestrian, truck) of the original caption, vary:
  - Camera angles: side-view or front-view
  - Intersection types: T-junctions, Y-junctions, cross-intersections, mid-blocks, or pedestrian crossings
  - Lighting: morning, afternoon, evening, or night
  - Traffic flow: free-flowing, steady, or busy
  - Scene content: object count/placement and background features (e.g., buildings, shops, trees, signs, utility poles)
"""

DIVERSE_PROMPT = """
You are an expert in generating diverse yet realistic image captions for road scenes captured by a fish-eye surveillance camera.
Your task is to rewrite the caption I give you into 5 diverse and realistic variants.

Please adhere to the following format for the caption:
- Start with ”A photo of”.
- Limit the total length to 40-50 words.
- Use grammatically correct and clear sentences.
- While preserving the core elements (bus, bike, car, pedestrian, truck) of the original caption, vary:
  - Camera angles: side-view or front-view
  - Intersection types: T-junctions, Y-junctions, cross-intersections, mid-blocks, or pedestrian crossings
  - Lighting: morning, afternoon, evening, or night
  - Traffic flow: free-flowing, steady, or busy
  - Scene content: object count/placement and background features (e.g., buildings, shops, trees, signs, utility poles)
- Ensure each caption includes at least one distinct variation that differentiates it from the others.
- Output the 5 captions as a numbered list, using the format:
  [1] Caption 1
  [2] Caption 2
  [3] Caption 3
  [4] Caption 4
  [5] Caption 5
- Do not include any explanation or extra text outside of the list.
"""


def parse_captions(text):
    pattern = r'\[(\d+)]\s+(.*?)(?=(?:\[\d+\])|\Z)'
    captions = re.findall(pattern, text.strip(), flags=re.DOTALL)
    return [caption[1].strip() for caption in captions]


def remove_duplicate_image_dicts(image_dicts):
    seen_ids = set()
    unique_dicts = []
    for d in image_dicts:
        img_id = d["id"]
        if img_id not in seen_ids:
            seen_ids.add(img_id)
            unique_dicts.append(d)
    return unique_dicts


class CaptionDataset(TorchDataset):
    def __init__(self, image_dicts):
        self.image_dicts = image_dicts

    def __len__(self):
        return len(self.image_dicts)

    def __getitem__(self, idx):
        return idx, self.image_dicts[idx]['caption']


def load_llama(model_name, device, ckpt_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    if ckpt_dir is not None:
        model = PeftModel.from_pretrained(model, ckpt_dir)

    model = model.to(device)
    return tokenizer, model


def predict_llama(prompt, caption, tokenizer, model, device, temperature):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": caption},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    response = outputs[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(response, skip_special_tokens=True)
    return output_text


class CaptionRephraser:
    def __init__(self, model_name, json_path, output_path, use_manual_edge=False, ckpt_dir=None):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.use_manual_edge = use_manual_edge
        self.use_automatic_edge = ckpt_dir is not None
        assert not (self.use_manual_edge and self.use_automatic_edge)

        if self.use_automatic_edge:
            self.temperature = 1.0
        else:
            self.temperature = 0.6

        self.tokenizer, self.model = load_llama(model_name, self.device, ckpt_dir)
        self.json_path = json_path
        self.output_path = output_path

        self.data = json.load(open(self.json_path, 'r'))
        self.image_dicts = self.data['images']
        self.dataset = CaptionDataset(self.image_dicts)

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
            idx_lst, caption_lst = batch
            for idx, caption in zip(idx_lst, caption_lst):
                if self.use_manual_edge or self.use_automatic_edge:
                    if self.use_manual_edge:
                        prompt = MANUAL_PROMPT
                    elif self.use_automatic_edge:
                        prompt = AUTOMATIC_PROMPT
                    else:
                        raise NotImplementedError

                    rephrased = [predict_llama(
                        prompt, caption, self.tokenizer, self.model, self.device, self.temperature,
                    ) for _ in range(5)]
                else:
                    output_text = predict_llama(
                        DIVERSE_PROMPT, caption, self.tokenizer, self.model, self.device, self.temperature,
                    )
                    rephrased = parse_captions(output_text)

                image_dict = self.image_dicts[idx]
                image_dict['rephrased'] = rephrased
                output_list.append(image_dict)

        self.accelerator.wait_for_everyone()
        all_outputs = gather_object(output_list)

        if self.accelerator.is_main_process:
            self.data['images'] = remove_duplicate_image_dicts(list(all_outputs))
            with open(self.output_path, 'w') as f:
                json.dump(self.data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help="LLM to rephrase captions")
    parser.add_argument('--json_path', help="json annotation file path without rephrased captions")
    parser.add_argument('--output_path', help="json annotation file path with rephrased captions")
    parser.add_argument('--use_manual_edge', action="store_true", help="whether to use the manually defined edge prompt rather than the diverse prompt for rephrasing")
    parser.add_argument('--ckpt_dir', help="directory of which the checkpoint of the trained rephraser is located")
    args = parser.parse_args()

    rephraser = CaptionRephraser(
        model_name=args.model_name,
        json_path=args.json_path,
        output_path=args.output_path,
        use_manual_edge=args.use_manual_edge,
        ckpt_dir=args.ckpt_dir,
    )
    rephraser.run_inference()
