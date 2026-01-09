# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import torch
from accelerate.logging import get_logger
import json
import os
import torch.distributed as dist

logger = get_logger(__name__)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm
import re
import ftfy
import html

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    """Clean prompt following WAN I2V pipeline approach"""
    text = whitespace_clean(basic_clean(text))
    return text

def contains_chinese(text):
    """检查字符串是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

class T5dataset(Dataset):
    def __init__(self, txt_path):
        self.txt_path = txt_path
        with open(self.txt_path, "r", encoding="utf-8") as f:
            self.train_dataset = [
                line for line in f.read().splitlines() if not contains_chinese(line)
            ]

    def __getitem__(self, idx):
        caption = self.train_dataset[idx]
        filename = str(idx)
        return dict(caption=caption, filename=filename)

    def __len__(self):
        return len(self.train_dataset)


def encode_prompt_wan_i2v(
    text_encoder,
    tokenizer,
    prompt,
    num_videos_per_prompt=1,
    max_sequence_length=512,
    device=None,
    dtype=None,
):
    """
    Encode prompts following WAN I2V pipeline logic exactly.
    This matches pipeline_wan_i2v.py _get_t5_prompt_embeds method.
    """
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    
    # Clean prompts like pipeline does
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)
    
    # Tokenize
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    
    # Encode
    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    
    # Truncate to actual sequence length and re-pad (pipeline behavior)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )
    
    # Duplicate text embeddings for each generation per prompt (pipeline behavior)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    
    # Also duplicate attention mask
    mask = mask.repeat(1, num_videos_per_prompt).view(batch_size * num_videos_per_prompt, -1)
    
    # Return embeddings and attention mask
    return prompt_embeds, mask


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Set random seed to {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"), exist_ok=True)

    # Load text encoder and tokenizer
    print(f"Loading text encoder from {args.model_path}")
    print(f"Model type: {args.model_type}")
    text_encoder = T5EncoderModel.from_pretrained(
        os.path.join(args.model_path, "text_encoder")
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_path, "tokenizer")
    )
    
    latents_txt_path = args.prompt_dir
    train_dataset = T5dataset(latents_txt_path)
    
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Use WAN I2V pipeline-compatible encoding
                prompt_embeds, prompt_attention_mask = encode_prompt_wan_i2v(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=data["caption"],
                    num_videos_per_prompt=args.num_videos_per_prompt,
                    max_sequence_length=args.max_sequence_length,
                    device=device,
                    dtype=text_encoder.dtype,
                )
                
                for idx, video_name in enumerate(data["filename"]):
                    prompt_embed_path = os.path.join(
                        args.output_dir, "prompt_embed", video_name + ".pt"
                    )
                    prompt_attention_mask_path = os.path.join(
                        args.output_dir, "prompt_attention_mask", video_name + ".pt"
                    )
                    
                    # Save embeddings
                    torch.save(prompt_embeds[idx].cpu(), prompt_embed_path)
                    torch.save(prompt_attention_mask[idx].cpu(), prompt_attention_mask_path)
                    
                    item = {}
                    item["prompt_embed_path"] = video_name + ".pt"
                    item["prompt_attention_mask"] = video_name + ".pt"
                    item["caption"] = data["caption"][idx]
                    json_data.append(item)
                    
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)
        print(f"Saved {len(all_json_data)} prompt embeddings to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="wan22_ti2v", 
        help="The type of model to use for preprocessing."
    )
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to WAN model (should contain text_encoder and tokenizer subdirs)")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for text encoding (should match pipeline default)",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos per prompt (for embedding duplication)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where embeddings will be saved.",
    )
    parser.add_argument(
        "--prompt_dir", 
        type=str, 
        required=True,
        help="Path to text file containing prompts (one per line)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible preprocessing"
    )
    args = parser.parse_args()
    main(args)
