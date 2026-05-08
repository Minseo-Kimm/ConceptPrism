#!/usr/bin/env python
"""ConceptPrism inference for Stable Diffusion v2.1.

Loads the learned token embeddings and the LoRA weights produced by
`train_sd21_stage1.py` and `train_sd21_stage2.py`, then renders `--prompt` (with
the placeholder `*` or `{V}` replaced by the target token).

Following the paper, prompts use only the target token — no class noun is appended.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from conceptprism.data import list_image_paths
from conceptprism.encode_prompt import patch_encode_prompt_no_textual_inversion
from conceptprism.tokens import load_learned_embeddings
from conceptprism.utils import slugify_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="ConceptPrism inference for SD v2.1")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--learned_embeddings_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True,
                        help="Either a directory containing pytorch_lora_weights.safetensors, "
                             "or the .safetensors file itself.")
    parser.add_argument("--instance_data_dir", type=str, required=True,
                        help="The reference image directory (needed to recover the per-image "
                             "residual token slots that exist in the saved embedding tensor).")
    parser.add_argument("--num_target_tokens", type=int, default=1)
    parser.add_argument("--num_residual_vectors", type=int, default=8)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--prompt", type=str, required=True,
                        help='A prompt template. The placeholder `*` (or "{V}") is '
                             'replaced with the target token. Example: "A * in the jungle".')

    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    return parser.parse_args()


def _load_pipeline(args):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    )

    num_images = len(list_image_paths(args.instance_data_dir))
    target_tokens = load_learned_embeddings(
        text_encoder, tokenizer, args.learned_embeddings_path,
        num_target_tokens=args.num_target_tokens,
        num_residual_vectors=args.num_residual_vectors,
        num_images=num_images,
    )

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=dtype,
    )
    patch_encode_prompt_no_textual_inversion(pipeline)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(args.lora_weights_path)
    pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    target_token_str = " ".join(target_tokens)
    return pipeline, target_token_str


def _expand_prompt(template: str, target_token_str: str) -> str:
    if "*" in template:
        return template.replace("*", target_token_str)
    if "{V}" in template:
        return template.replace("{V}", target_token_str)
    # If no placeholder is provided, prepend the token.
    return f"{target_token_str} {template}".strip()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline, target_token_str = _load_pipeline(args)

    full_prompt = _expand_prompt(args.prompt, target_token_str)
    slug = slugify_prompt(args.prompt)
    for i in tqdm(range(args.num_validation_images), desc="Generating"):
        generator = torch.Generator(device=args.device).manual_seed(args.seed + i)
        image = pipeline(
            prompt=full_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.resolution,
            width=args.resolution,
            generator=generator,
        ).images[0]
        image.save(Path(args.output_dir) / f"{slug}_{i}.png")


if __name__ == "__main__":
    main()
