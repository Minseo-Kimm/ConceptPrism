"""Patch a Stable Diffusion pipeline so it does not call `maybe_convert_prompt`.

Why: ConceptPrism adds tokens like `<my_concept>`, `<my_concept_1>`, ... directly to
the tokenizer as full vocab entries. They are *not* multi-vector textual inversion
tokens. `StableDiffusionPipeline.encode_prompt` calls `self.maybe_convert_prompt(...)`
which would try to expand suffix-numbered tokens, producing nonsense like
`"<my_concept_1>" → "<my_concept_1>_0 <my_concept_1>_1"`. The simplest fix is to
monkey-patch `encode_prompt` with a copy that omits that line.

Only needed for SD pipelines. FLUX's `FluxPipeline` does not call
`maybe_convert_prompt`, so this patch is a no-op there.
"""

from __future__ import annotations

from types import MethodType
from typing import List, Optional

import torch
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)


def _encode_prompt_no_textual_inversion(
    self,
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
):
    if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
        self._lora_scale = lora_scale
        if not USE_PEFT_BACKEND:
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
        else:
            scale_lora_layers(self.text_encoder, lora_scale)

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        attention_mask = None
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)

        if clip_skip is None:
            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
            )
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

    if self.text_encoder is not None:
        prompt_embeds_dtype = self.text_encoder.dtype
    elif self.unet is not None:
        prompt_embeds_dtype = self.unet.dtype
    else:
        prompt_embeds_dtype = prompt_embeds.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        if negative_prompt is None:
            uncond_tokens: List[str] = [""] * batch_size
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt] * batch_size
        else:
            uncond_tokens = list(negative_prompt)

        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        attention_mask = None
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)

        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(device), attention_mask=attention_mask
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    if self.text_encoder is not None and isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
        unscale_lora_layers(self.text_encoder, lora_scale)

    return prompt_embeds, negative_prompt_embeds


def patch_encode_prompt_no_textual_inversion(pipeline) -> None:
    """Monkey-patch `pipeline.encode_prompt` so it skips multi-vector TI conversion."""
    pipeline.encode_prompt = MethodType(_encode_prompt_no_textual_inversion, pipeline)
