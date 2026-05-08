#!/usr/bin/env python
"""ConceptPrism Stage 1 — Token Optimization on Stable Diffusion v2.1.

Optimizes a `<my_concept>` target token and one residual token group per reference
image via L_rec + β·L_excl (paper Algorithm 1). Only the new token embeddings in the
text encoder are updated; the rest of the model stays frozen.

Usage (single concept):
    accelerate launch train_sd21_stage1.py \\
        --concept_name dog \\
        --instance_data_dir datasets/dog \\
        --init_prompts_path datasets/init_prompts.json \\
        --pretrained_model_name_or_path Manojb/stable-diffusion-2-1-base \\
        --output_dir runs/sd21_stage1/dog \\
        --max_train_steps 200

Outputs `learned_embeddings.pt` (and one per checkpoint), suitable for Stage 2.
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
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from conceptprism.data import (
    ConceptPrismStage1Dataset,
    list_image_paths,
    stage1_collate,
)
from conceptprism.tokens import (
    add_tokens_to_tokenizer,
    build_residual_tokens,
    build_target_tokens,
    initialize_token_embeddings,
    save_learned_embeddings,
)
from conceptprism.utils import load_init_prompts

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConceptPrism Stage 1 (Token Optimization) for SD v2.1"
    )
    parser.add_argument("--concept_name", type=str, required=True,
                        help="Subdirectory name inside --instance_data_dir (purely informational; "
                             "used for logging and to look up captions in --init_prompts_path).")
    parser.add_argument("--instance_data_dir", type=str, required=True,
                        help="Directory containing the reference images for this concept.")
    parser.add_argument("--init_prompts_path", type=str, default="datasets/init_prompts.json",
                        help="JSON mapping concept_name -> list[caption], one caption per image.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    # ConceptPrism hyperparameters (paper defaults).
    parser.add_argument("--num_target_tokens", type=int, default=1)
    parser.add_argument("--num_residual_vectors", type=int, default=8,
                        help="Number of residual-token sub-tokens per image (n in the paper).")
    parser.add_argument("--exclusion_loss_weight", type=float, default=0.05,
                        help="β in the paper. Set to 0 to disable L_excl entirely.")
    parser.add_argument("--num_anchor_samples", type=int, default=3,
                        help="Number of anchor images j ≠ i sampled per step for L_excl.")

    # Standard training knobs.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
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
    parser.add_argument("--checkpointing_steps", type=int, default=200)

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

    # --- Load pretrained components --------------------------------------------------
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

    # --- Add ConceptPrism tokens and initialize --------------------------------------
    image_paths = list_image_paths(args.instance_data_dir)
    num_images = len(image_paths)
    logger.info(f"Loaded {num_images} reference images from {args.instance_data_dir}")

    target_tokens = build_target_tokens(args.num_target_tokens)
    residual_token_groups = build_residual_tokens(num_images, args.num_residual_vectors)
    flat_residual = [t for group in residual_token_groups for t in group]

    target_token_ids = add_tokens_to_tokenizer(tokenizer, text_encoder, target_tokens)
    residual_token_ids = []
    for group in residual_token_groups:
        ids = tokenizer.convert_tokens_to_ids(group)
        residual_token_ids.append(ids)
    # Ensure residual tokens were really added (some might have collided with vocab).
    if flat_residual:
        add_tokens_to_tokenizer(tokenizer, text_encoder, flat_residual)
        residual_token_ids = [tokenizer.convert_tokens_to_ids(g) for g in residual_token_groups]

    text_encoder.to(accelerator.device)

    if args.num_residual_vectors > 0:
        init_prompts = load_init_prompts(args.init_prompts_path, args.concept_name, num_images)
    else:
        init_prompts = None

    initialize_token_embeddings(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        target_token_ids=target_token_ids,
        residual_token_ids=residual_token_ids,
        residual_init_prompts=init_prompts,
        device=accelerator.device,
    )

    all_added_token_ids = list(target_token_ids) + [tid for group in residual_token_ids for tid in group]
    tok_id_min = min(all_added_token_ids)
    tok_id_max = max(all_added_token_ids)

    # --- Freeze everything except the input embeddings -------------------------------
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder.get_input_embeddings().requires_grad_(True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # Token embedding row stays in fp32 even under mixed precision (it's the only thing we train).
    text_encoder.to(accelerator.device, dtype=torch.float32 if weight_dtype != torch.float32 else weight_dtype)

    # --- Optimizer + dataloader ------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [text_encoder.get_input_embeddings().weight],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = ConceptPrismStage1Dataset(
        instance_data_root=args.instance_data_dir,
        target_tokens=target_tokens,
        num_residual_vectors=args.num_residual_vectors,
        size=args.resolution,
        center_crop=args.center_crop,
        num_anchor_samples=args.num_anchor_samples,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=stage1_collate,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Snapshot the embedding weight so we can re-clamp non-trained rows after each step.
    orig_embed_weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    def encode_prompt_list(prompts):
        enc = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = enc.input_ids.to(accelerator.device)
        return text_encoder(ids)[0]

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Stage 1",
    )
    global_step = 0
    num_train_epochs = math.ceil(args.max_train_steps / max(1, len(train_dataloader)))

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for batch in train_dataloader:
            with accelerator.accumulate(text_encoder):
                # ---- L_rec on image i with prompt "target + residual_i" ------------
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

                recon_hidden = encode_prompt_list(batch["recon_prompts"]).to(dtype=weight_dtype)
                model_pred = unet(noisy_latents, timesteps, recon_hidden, return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss_rec = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # ---- L_excl on anchor images j ≠ i ---------------------------------
                if args.exclusion_loss_weight > 0 and args.num_residual_vectors > 0:
                    anchor_pixels = batch["anchor_pixel_values"]  # (B, K, C, H, W)
                    bsz_anchor, k, ch, h, w = anchor_pixels.shape
                    anchor_pixels = anchor_pixels.view(bsz_anchor * k, ch, h, w).to(dtype=weight_dtype)

                    with torch.no_grad():
                        anchor_latents = vae.encode(anchor_pixels).latent_dist.sample() * vae.config.scaling_factor

                    anchor_noise = torch.randn_like(anchor_latents)
                    anchor_timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (anchor_latents.shape[0],),
                        device=anchor_latents.device,
                    ).long()
                    anchor_noisy = noise_scheduler.add_noise(anchor_latents, anchor_noise, anchor_timesteps)

                    # Each item in batch has its own residual_prompt; expand to k copies.
                    residual_prompts_expanded = [p for p in batch["residual_prompts"] for _ in range(k)]
                    null_prompts_expanded = [p for p in batch["null_prompts"] for _ in range(k)]

                    residual_hidden = encode_prompt_list(residual_prompts_expanded).to(dtype=weight_dtype)
                    pred_residual = unet(anchor_noisy, anchor_timesteps, residual_hidden, return_dict=False)[0]
                    with torch.no_grad():
                        null_hidden = encode_prompt_list(null_prompts_expanded).to(dtype=weight_dtype)
                        pred_null = unet(anchor_noisy, anchor_timesteps, null_hidden, return_dict=False)[0]
                    loss_excl = F.mse_loss(pred_residual.float(), pred_null.float(), reduction="mean")
                else:
                    loss_excl = torch.tensor(0.0, device=accelerator.device)

                loss = loss_rec + args.exclusion_loss_weight * loss_excl
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [text_encoder.get_input_embeddings().weight], args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Re-clamp every non-ConceptPrism row to its original value.
                with torch.no_grad():
                    weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
                    keep_mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
                    keep_mask[tok_id_min : tok_id_max + 1] = False
                    weight.data[keep_mask] = orig_embed_weight[keep_mask].to(weight.dtype)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(
                    loss=loss.detach().item(),
                    rec=loss_rec.detach().item(),
                    excl=loss_excl.detach().item(),
                )

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    save_learned_embeddings(
                        accelerator.unwrap_model(text_encoder),
                        target_token_ids,
                        residual_token_ids,
                        save_path / "learned_embeddings.pt",
                    )
                    logger.info(f"Saved learned embeddings to {save_path}")

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_learned_embeddings(
            accelerator.unwrap_model(text_encoder),
            target_token_ids,
            residual_token_ids,
            Path(args.output_dir) / "learned_embeddings.pt",
        )
        logger.info(f"Stage 1 finished. Final embeddings saved to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
