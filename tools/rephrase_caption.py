import argparse
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm

from ates.io import dedupe_image_dicts, load_json, write_json
from ates.prompts import (
    AUTOMATIC_REPHRASE_PROMPT,
    DIVERSE_REPHRASE_PROMPT,
    MANUAL_REPHRASE_PROMPT,
)


def parse_captions(text):
    pattern = r'\[(\d+)]\s+(.*?)(?=(?:\[\d+\])|\Z)'
    captions = re.findall(pattern, text.strip(), flags=re.DOTALL)
    return [caption[1].strip() for caption in captions]

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

        self.data = load_json(self.json_path)
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
                        prompt = MANUAL_REPHRASE_PROMPT
                    elif self.use_automatic_edge:
                        prompt = AUTOMATIC_REPHRASE_PROMPT
                    else:
                        raise NotImplementedError

                    rephrased = [predict_llama(
                        prompt, caption, self.tokenizer, self.model, self.device, self.temperature,
                    ) for _ in range(5)]
                else:
                    output_text = predict_llama(
                        DIVERSE_REPHRASE_PROMPT, caption, self.tokenizer, self.model, self.device, self.temperature,
                    )
                    rephrased = parse_captions(output_text)

                image_dict = self.image_dicts[idx]
                image_dict['rephrased'] = rephrased
                output_list.append(image_dict)

        self.accelerator.wait_for_everyone()
        all_outputs = gather_object(output_list)

        if self.accelerator.is_main_process:
            self.data['images'] = dedupe_image_dicts(list(all_outputs))
            write_json(self.output_path, self.data)


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
