#!/usr/bin/env python
"""ConceptPrism Stage 1 — Token Optimization on FLUX.1.

Optimizes the target / residual token embeddings in the **T5** text encoder. The CLIP
text encoder, the VAE, and the FLUX transformer are frozen. Supports an optional
4-bit quantized FLUX transformer to fit on smaller GPUs.

Usage:
    accelerate launch train_flux_stage1.py \\
        --concept_name dog \\
        --instance_data_dir datasets/dog \\
        --init_prompts_path datasets/init_prompts.json \\
        --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \\
        --output_dir runs/flux_stage1/dog \\
        --max_train_steps 200 \\
        --use_4bit_transformer
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
        description="ConceptPrism Stage 1 (Token Optimization) for FLUX.1"
    )
    parser.add_argument("--concept_name", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--init_prompts_path", type=str, default="datasets/init_prompts.json")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_target_tokens", type=int, default=1)
    parser.add_argument("--num_residual_vectors", type=int, default=1,
                        help="FLUX defaults to 1 residual vector per image (T5 dim is large).")
    parser.add_argument("--exclusion_loss_weight", type=float, default=0.5,
                        help="β for L_excl. FLUX uses a higher β than SD2.1 by default.")
    parser.add_argument("--num_anchor_samples", type=int, default=3,
                        help="Number of anchor images j ≠ i sampled per step for L_excl.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Used only for the FLUX-dev variant (guidance_embeds=True).")
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--use_4bit_transformer", action="store_true",
                        help="Load the FLUX transformer in 4-bit (NF4) to fit on smaller GPUs.")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


def _load_t5_in_dtype(name_or_path, revision, variant, dtype):
    return T5EncoderModel.from_pretrained(
        name_or_path, subfolder="text_encoder_2",
        revision=revision, variant=variant, torch_dtype=dtype,
    )


def _encode_for_flux(text_encoders, tokenizers, prompts, max_sequence_length, device):
    """Return (T5 prompt_embeds, CLIP pooled_prompt_embeds, text_ids)."""
    if isinstance(prompts, str):
        prompts = [prompts]

    # CLIP (frozen) → pooled embedding
    clip_inputs = tokenizers[0](
        prompts, padding="max_length", max_length=77, truncation=True,
        return_overflowing_tokens=False, return_length=False, return_tensors="pt",
    )
    clip_out = text_encoders[0](clip_inputs.input_ids.to(device), output_hidden_states=False)
    pooled = clip_out.pooler_output

    # T5 (the only thing we train via its input embeddings)
    t5_inputs = tokenizers[1](
        prompts, padding="max_length", max_length=max_sequence_length, truncation=True,
        return_length=False, return_overflowing_tokens=False, return_tensors="pt",
    )
    t5_out = text_encoders[1](t5_inputs.input_ids.to(device))[0]

    text_ids = torch.zeros(t5_out.shape[1], 3, device=device, dtype=t5_out.dtype)
    return t5_out, pooled.to(t5_out.dtype), text_ids


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

    # --- Load tokenizers and frozen models -------------------------------------------
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    weight_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else (
        torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant, torch_dtype=weight_dtype,
    )
    text_encoder_two = _load_t5_in_dtype(
        args.pretrained_model_name_or_path, args.revision, args.variant, weight_dtype
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

    # --- Add ConceptPrism tokens and initialize on T5 --------------------------------
    image_paths = list_image_paths(args.instance_data_dir)
    num_images = len(image_paths)
    logger.info(f"Loaded {num_images} reference images from {args.instance_data_dir}")

    target_tokens = build_target_tokens(args.num_target_tokens)
    residual_token_groups = build_residual_tokens(num_images, args.num_residual_vectors)
    flat_residual = [t for group in residual_token_groups for t in group]
    all_new_tokens = list(target_tokens) + flat_residual

    # Add tokens to *both* tokenizers (CLIP must accept the prompt strings even though
    # we only train T5 embeddings).
    add_tokens_to_tokenizer(tokenizer_one, text_encoder_one, all_new_tokens)
    target_token_ids_t5 = add_tokens_to_tokenizer(tokenizer_two, text_encoder_two, target_tokens)
    if flat_residual:
        add_tokens_to_tokenizer(tokenizer_two, text_encoder_two, flat_residual)
    residual_token_ids_t5 = [
        tokenizer_two.convert_tokens_to_ids(group) for group in residual_token_groups
    ]

    text_encoder_two.to(accelerator.device)

    if args.num_residual_vectors > 0:
        init_prompts = load_init_prompts(args.init_prompts_path, args.concept_name, num_images)
    else:
        init_prompts = None

    initialize_token_embeddings(
        text_encoder=text_encoder_two,
        tokenizer=tokenizer_two,
        target_token_ids=target_token_ids_t5,
        residual_token_ids=residual_token_ids_t5,
        residual_init_prompts=init_prompts,
        device=accelerator.device,
    )

    all_added_token_ids = list(target_token_ids_t5) + [tid for g in residual_token_ids_t5 for tid in g]
    tok_id_min = min(all_added_token_ids)
    tok_id_max = max(all_added_token_ids)

    # T5 embedding row trains; everything else stays frozen.
    text_encoder_two.encoder.requires_grad_(False)
    text_encoder_two.get_input_embeddings().requires_grad_(True)

    # Move other models to device.
    text_encoder_one.to(accelerator.device)
    vae.to(accelerator.device)
    # transformer is already on device when bitsandbytes is involved; otherwise:
    if not args.use_4bit_transformer:
        transformer.to(accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        text_encoder_two.gradient_checkpointing_enable()

    # --- Optimizer / dataset / scheduler ---------------------------------------------
    optimizer = torch.optim.AdamW(
        [text_encoder_two.get_input_embeddings().weight],
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

    text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder_two, optimizer, train_dataloader, lr_scheduler
    )

    orig_t5_embed = accelerator.unwrap_model(text_encoder_two).get_input_embeddings().weight.data.clone()

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="FLUX Stage 1",
    )
    global_step = 0
    num_train_epochs = math.ceil(args.max_train_steps / max(1, len(train_dataloader)))

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    use_guidance = transformer.config.guidance_embeds

    def predict(latents, prompt_embeds, pooled_embeds, text_ids):
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
        return pred, noise, latents, sigmas

    for _ in range(num_train_epochs):
        text_encoder_two.train()
        for batch in train_dataloader:
            with accelerator.accumulate(text_encoder_two):
                # ---- L_rec on image i ---------------------------------------------
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                recon_embeds, recon_pooled, recon_ids = _encode_for_flux(
                    [text_encoder_one, text_encoder_two],
                    [tokenizer_one, tokenizer_two],
                    batch["recon_prompts"], args.max_sequence_length, accelerator.device,
                )
                pred, noise, lats, sigmas = predict(latents, recon_embeds, recon_pooled, recon_ids)
                target = noise - lats
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                loss_rec = (
                    (weighting.float() * (pred.float() - target.float()) ** 2)
                    .reshape(target.shape[0], -1)
                    .mean(1)
                    .mean()
                )

                # ---- L_excl on anchor images j ≠ i --------------------------------
                if args.exclusion_loss_weight > 0 and args.num_residual_vectors > 0:
                    anchor_pixels = batch["anchor_pixel_values"]  # (B, K, C, H, W)
                    bsz_anchor, k, ch, h, w = anchor_pixels.shape
                    anchor_pixels = anchor_pixels.view(bsz_anchor * k, ch, h, w).to(dtype=vae.dtype)
                    with torch.no_grad():
                        anchor_latents = vae.encode(anchor_pixels).latent_dist.sample()
                        anchor_latents = (anchor_latents - vae.config.shift_factor) * vae.config.scaling_factor

                    residual_prompts_expanded = [p for p in batch["residual_prompts"] for _ in range(k)]
                    null_prompts_expanded = [p for p in batch["null_prompts"] for _ in range(k)]

                    residual_embeds, residual_pooled, residual_ids = _encode_for_flux(
                        [text_encoder_one, text_encoder_two],
                        [tokenizer_one, tokenizer_two],
                        residual_prompts_expanded, args.max_sequence_length, accelerator.device,
                    )
                    pred_residual, _, _, _ = predict(anchor_latents, residual_embeds, residual_pooled, residual_ids)
                    with torch.no_grad():
                        null_embeds, null_pooled, null_ids = _encode_for_flux(
                            [text_encoder_one, text_encoder_two],
                            [tokenizer_one, tokenizer_two],
                            null_prompts_expanded, args.max_sequence_length, accelerator.device,
                        )
                        pred_null, _, _, _ = predict(anchor_latents, null_embeds, null_pooled, null_ids)
                    loss_excl = F.mse_loss(pred_residual.float(), pred_null.float())
                else:
                    loss_excl = torch.tensor(0.0, device=accelerator.device)

                loss = loss_rec + args.exclusion_loss_weight * loss_excl
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [text_encoder_two.get_input_embeddings().weight], args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    weight = accelerator.unwrap_model(text_encoder_two).get_input_embeddings().weight
                    keep_mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
                    keep_mask[tok_id_min : tok_id_max + 1] = False
                    weight.data[keep_mask] = orig_t5_embed[keep_mask].to(weight.dtype)

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
                        accelerator.unwrap_model(text_encoder_two),
                        target_token_ids_t5,
                        residual_token_ids_t5,
                        save_path / "learned_embeddings.pt",
                    )
                    logger.info(f"Saved learned T5 embeddings to {save_path}")

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_learned_embeddings(
            accelerator.unwrap_model(text_encoder_two),
            target_token_ids_t5,
            residual_token_ids_t5,
            Path(args.output_dir) / "learned_embeddings.pt",
        )
        logger.info(f"Stage 1 finished. Final embeddings saved to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
