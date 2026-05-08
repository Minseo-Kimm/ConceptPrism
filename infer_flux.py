#!/usr/bin/env python
"""ConceptPrism inference for FLUX.1.

Loads the learned T5 token embeddings + LoRA weights and renders `--prompt` (with
the placeholder `*` or `{V}` replaced by the target token). Supports both standard
bf16 and 4-bit quantized FLUX. Following the paper, prompts use only the target
token; no class noun is appended.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from tqdm.auto import tqdm
from transformers import (
    BitsAndBytesConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from conceptprism.data import list_image_paths
from conceptprism.tokens import load_learned_embeddings
from conceptprism.utils import slugify_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="ConceptPrism inference for FLUX.1")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--learned_embeddings_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--num_target_tokens", type=int, default=1)
    parser.add_argument("--num_residual_vectors", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--prompt", type=str, required=True,
                        help='A prompt template. The placeholder `*` (or "{V}") is '
                             'replaced with the target token. Example: "A * in the jungle".')

    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_4bit_transformer", action="store_true")
    parser.add_argument("--enable_cpu_offload", action="store_true",
                        help="Use FluxPipeline.enable_model_cpu_offload() (recommended on <40GB GPUs).")

    return parser.parse_args()


def _load_pipeline(args):
    dtype = torch.bfloat16
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=dtype
    )

    num_images = len(list_image_paths(args.instance_data_dir))
    target_tokens = load_learned_embeddings(
        text_encoder_two, tokenizer_two, args.learned_embeddings_path,
        num_target_tokens=args.num_target_tokens,
        num_residual_vectors=args.num_residual_vectors,
        num_images=num_images,
    )
    # Mirror token names into CLIP so the pipeline can tokenize the concept-prefixed prompts.
    new_token_strs = list(target_tokens)
    for i in range(num_images):
        new_token_strs.append(f"<|res{i}|>")
        for j in range(1, args.num_residual_vectors):
            new_token_strs.append(f"<|res{i}_{j}|>")
    tokenizer_one.add_tokens(new_token_strs)
    text_encoder_one.resize_token_embeddings(len(tokenizer_one))

    transformer_kwargs = dict(
        subfolder="transformer", torch_dtype=dtype,
    )
    if args.use_4bit_transformer:
        transformer_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, **transformer_kwargs,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    pipeline = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        transformer=transformer,
        vae=vae,
    )
    pipeline.load_lora_weights(args.lora_weights_path)

    if args.enable_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(args.device)

    pipeline.set_progress_bar_config(disable=True)
    return pipeline, " ".join(target_tokens)


def _expand_prompt(template: str, target_token_str: str) -> str:
    if "*" in template:
        return template.replace("*", target_token_str)
    if "{V}" in template:
        return template.replace("{V}", target_token_str)
    return f"{target_token_str} {template}".strip()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline, target_token_str = _load_pipeline(args)

    full_prompt = _expand_prompt(args.prompt, target_token_str)
    slug = slugify_prompt(args.prompt)
    for i in tqdm(range(args.num_validation_images), desc="Generating"):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + i)
        image = pipeline(
            prompt=full_prompt,
            height=args.resolution,
            width=args.resolution,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
        image.save(Path(args.output_dir) / f"{slug}_{i}.png")


if __name__ == "__main__":
    main()
