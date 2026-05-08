#!/usr/bin/env python
"""ConceptPrism Stage 2 — Concept Disentangled Fine-Tuning on Stable Diffusion v2.1.

Tokens learned in Stage 1 are loaded into the text encoder and frozen. A LoRA adapter
is inserted into the U-Net's attention layers and trained with the standard diffusion
reconstruction loss (paper Algorithm 2). Default reaches optimal quality at ~120 steps.

Usage:
    accelerate launch train_sd21_stage2.py \\
        --concept_name dog \\
        --instance_data_dir datasets/dog \\
        --learned_embeddings_path runs/sd21_stage1/dog/learned_embeddings.pt \\
        --pretrained_model_name_or_path Manojb/stable-diffusion-2-1-base \\
        --output_dir runs/sd21_stage2/dog \\
        --max_train_steps 200 --checkpointing_steps 40
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

from conceptprism.data import (
    ConceptPrismStage2Dataset,
    list_image_paths,
    stage2_collate,
)
from conceptprism.tokens import build_target_tokens, load_learned_embeddings

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConceptPrism Stage 2 (Concept Disentangled Fine-Tuning) for SD v2.1"
    )
    parser.add_argument("--concept_name", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--learned_embeddings_path", type=str, required=True,
                        help="Path to the .pt produced by train_sd21_stage1.py.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_target_tokens", type=int, default=1)
    parser.add_argument("--num_residual_vectors", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=40)

    return parser.parse_args()


def main():
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model components -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        revision=args.revision, variant=args.variant,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # --- Reload Stage 1 tokens -------------------------------------------------------
    image_paths = list_image_paths(args.instance_data_dir)
    num_images = len(image_paths)

    target_tokens = load_learned_embeddings(
        text_encoder, tokenizer, args.learned_embeddings_path,
        num_target_tokens=args.num_target_tokens,
        num_residual_vectors=args.num_residual_vectors,
        num_images=num_images,
    )
    logger.info(
        f"Loaded {args.num_target_tokens} target + "
        f"{num_images}×{args.num_residual_vectors} residual tokens from "
        f"{args.learned_embeddings_path}"
    )

    # --- Freeze base, attach LoRA adapter to UNet's attention layers -----------------
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    # LoRA params must be fp32 for training stability under mixed precision.
    if weight_dtype != torch.float32:
        for p in unet.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, unet.parameters())),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = ConceptPrismStage2Dataset(
        instance_data_root=args.instance_data_dir,
        target_tokens=target_tokens,
        num_residual_vectors=args.num_residual_vectors,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=stage2_collate,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    def encode_prompt_list(prompts):
        enc = tokenizer(
            prompts, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        return text_encoder(enc.input_ids.to(accelerator.device))[0]

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Stage 2",
    )
    global_step = 0
    num_train_epochs = math.ceil(args.max_train_steps / max(1, len(train_dataloader)))

    for _ in range(num_train_epochs):
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                hidden = encode_prompt_list(batch["prompts"]).to(dtype=weight_dtype)
                model_pred = unet(noisy_latents, timesteps, hidden, return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(loss=loss.detach().item())

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_unet)
                    )
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=save_path,
                        unet_lora_layers=lora_state_dict,
                    )
                    logger.info(f"Saved LoRA weights to {save_path}")

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet)
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=lora_state_dict,
        )
        logger.info(f"Stage 2 finished. LoRA saved to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
