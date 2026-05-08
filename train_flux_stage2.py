#!/usr/bin/env python
"""ConceptPrism Stage 2 — Concept Disentangled Fine-Tuning on FLUX.1.

Tokens learned in Stage 1 are loaded into the T5 (and CLIP, by name) tokenizer; their
embeddings are frozen. A LoRA adapter is attached to the FLUX transformer's attention
+ FF layers and trained with the standard Flow Matching reconstruction loss.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import (
    BitsAndBytesConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from conceptprism.data import (
    ConceptPrismStage2Dataset,
    list_image_paths,
    stage2_collate,
)
from conceptprism.tokens import build_target_tokens, load_learned_embeddings

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConceptPrism Stage 2 (Fine-Tuning) for FLUX.1"
    )
    parser.add_argument("--concept_name", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--learned_embeddings_path", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_target_tokens", type=int, default=1)
    parser.add_argument("--num_residual_vectors", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--use_4bit_transformer", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
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

    weight_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else (
        torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    )

    # --- Load tokenizers and frozen models -------------------------------------------
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant, torch_dtype=weight_dtype,
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant, torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant, torch_dtype=weight_dtype,
    )

    transformer_kwargs = dict(
        subfolder="transformer", revision=args.revision, variant=args.variant,
        torch_dtype=weight_dtype,
    )
    if args.use_4bit_transformer:
        transformer_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=weight_dtype,
        )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, **transformer_kwargs,
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # --- Reload Stage 1 tokens -------------------------------------------------------
    image_paths = list_image_paths(args.instance_data_dir)
    num_images = len(image_paths)

    target_tokens = load_learned_embeddings(
        text_encoder_two, tokenizer_two, args.learned_embeddings_path,
        num_target_tokens=args.num_target_tokens,
        num_residual_vectors=args.num_residual_vectors,
        num_images=num_images,
    )
    # CLIP must accept the same token strings as plain ids (no embedding load needed
    # since CLIP only sees them as part of the prompt text and the FLUX prompt encoder
    # uses CLIP's pooled embedding rather than token-by-token routing).
    new_token_strs = list(target_tokens)
    for i in range(num_images):
        new_token_strs.append(f"<|res{i}|>")
        for j in range(1, args.num_residual_vectors):
            new_token_strs.append(f"<|res{i}_{j}|>")
    tokenizer_one.add_tokens(new_token_strs)
    text_encoder_one.resize_token_embeddings(len(tokenizer_one))

    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    vae.to(accelerator.device)
    if not args.use_4bit_transformer:
        transformer.to(accelerator.device)

    # --- LoRA on the FLUX transformer ------------------------------------------------
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    target_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
    ]
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(
        transformer_lora_parameters,
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

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def encode_prompts(prompts):
        clip_inputs = tokenizer_one(
            prompts, padding="max_length", max_length=77, truncation=True,
            return_overflowing_tokens=False, return_length=False, return_tensors="pt",
        )
        clip_out = text_encoder_one(clip_inputs.input_ids.to(accelerator.device), output_hidden_states=False)
        pooled = clip_out.pooler_output.to(weight_dtype)

        t5_inputs = tokenizer_two(
            prompts, padding="max_length", max_length=args.max_sequence_length, truncation=True,
            return_length=False, return_overflowing_tokens=False, return_tensors="pt",
        )
        t5_out = text_encoder_two(t5_inputs.input_ids.to(accelerator.device))[0].to(weight_dtype)
        text_ids = torch.zeros(t5_out.shape[1], 3, device=accelerator.device, dtype=weight_dtype)
        return t5_out, pooled, text_ids

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="FLUX Stage 2",
    )
    global_step = 0
    num_train_epochs = math.ceil(args.max_train_steps / max(1, len(train_dataloader)))
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    use_guidance = transformer.config.guidance_embeds

    for _ in range(num_train_epochs):
        transformer.train()
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    prompt_embeds, pooled_embeds, text_ids = encode_prompts(batch["prompts"])

                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                packed = FluxPipeline._pack_latents(
                    noisy_latents, bsz, latents.shape[1], latents.shape[2], latents.shape[3]
                )
                img_ids = FluxPipeline._prepare_latent_image_ids(
                    bsz, latents.shape[2] // 2, latents.shape[3] // 2, accelerator.device, weight_dtype
                )
                guidance = (
                    torch.tensor([args.guidance_scale], device=accelerator.device).expand(bsz)
                    if use_guidance else None
                )
                pred = transformer(
                    hidden_states=packed,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                pred = FluxPipeline._unpack_latents(
                    pred, latents.shape[2] * vae_scale_factor, latents.shape[3] * vae_scale_factor, vae_scale_factor
                )

                target = noise - latents
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                loss = (
                    (weighting.float() * (pred.float() - target.float()) ** 2)
                    .reshape(target.shape[0], -1)
                    .mean(1)
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
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
                    unwrapped_transformer = accelerator.unwrap_model(transformer)
                    transformer_lora_state_dict = get_peft_model_state_dict(unwrapped_transformer)
                    FluxPipeline.save_lora_weights(
                        save_directory=save_path,
                        transformer_lora_layers=transformer_lora_state_dict,
                    )
                    logger.info(f"Saved LoRA weights to {save_path}")

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        transformer_lora_state_dict = get_peft_model_state_dict(unwrapped_transformer)
        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_state_dict,
        )
        logger.info(f"Stage 2 finished. LoRA saved to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
