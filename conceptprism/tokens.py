"""Add and initialize ConceptPrism's target / residual tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import torch

TARGET_TOKEN_BASE = "<my_concept>"


def build_target_tokens(num_target_tokens: int = 1, base: str = TARGET_TOKEN_BASE) -> List[str]:
    """Return [base, base_1, ..., base_{n-1}] for an n-token target."""
    if num_target_tokens < 1:
        raise ValueError(f"num_target_tokens must be >= 1, got {num_target_tokens}")
    tokens = [base]
    for j in range(1, num_target_tokens):
        tokens.append(f"{base}_{j}")
    return tokens


def build_residual_tokens(num_images: int, num_vectors: int) -> List[List[str]]:
    """Return [[<|res0|>, <|res0_1|>, ...], [<|res1|>, ...], ...].

    `num_vectors == 0` disables residual tokens entirely (returns []).
    """
    if num_vectors == 0:
        return []
    groups: List[List[str]] = []
    for i in range(num_images):
        group = [f"<|res{i}|>"]
        for j in range(1, num_vectors):
            group.append(f"<|res{i}_{j}|>")
        groups.append(group)
    return groups


def add_tokens_to_tokenizer(tokenizer, text_encoder, new_tokens: Sequence[str]) -> List[int]:
    """Add tokens, resize embeddings, and return the new ids in input order."""
    if not new_tokens:
        return []
    tokenizer.add_tokens(list(new_tokens))
    text_encoder.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(list(new_tokens))


def _mean_embedding_for_text(text: str, tokenizer, text_encoder, device) -> torch.Tensor:
    """Return a 1-D tensor: mean of valid (non-pad) text-encoder hidden states for `text`.

    Works for both CLIP-style and T5-style encoders. CLIP uses BOS/EOS markers; we use
    the attention mask to drop padding regardless of encoder family.
    """
    enc = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device) if "attention_mask" in enc else None

    with torch.no_grad():
        out = text_encoder(input_ids)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state  # [1, L, D]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            # For CLIP, also mask BOS (index 0) and any token after the first EOS
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                eos = tokenizer.eos_token_id
                bos = getattr(tokenizer, "bos_token_id", None)
                ids = input_ids[0].tolist()
                start = 1 if bos is not None and ids[0] == bos else 0
                end = next((i for i, t in enumerate(ids) if t == eos and i >= start), len(ids))
                m = torch.zeros_like(mask)
                m[:, start:end, :] = 1.0
                mask = mask * m
            denom = mask.sum(dim=1).clamp(min=1.0)
            mean = (hidden * mask).sum(dim=1) / denom
        else:
            mean = hidden.mean(dim=1)

    return mean[0]  # [D]


def initialize_token_embeddings(
    text_encoder,
    tokenizer,
    target_token_ids: Sequence[int],
    residual_token_ids: Sequence[Sequence[int]],
    residual_init_prompts: Sequence[str] | None,
    device,
) -> None:
    """In-place initialize target + residual token embeddings.

    - **Target tokens**: random init (paper Algorithm 1, "Initialize t_target randomly").
      We sample from the existing-vocabulary embedding distribution to keep the magnitude
      sane (mean and per-dim std of the encoder's pretrained input embeddings).
    - **Residual tokens**: set each image-i group to the mean (across valid positions) of
      the encoder's hidden states for `residual_init_prompts[i]`. If `residual_init_prompts`
      is None, fall back to the same random init as target.

    Edits `text_encoder.get_input_embeddings().weight` directly under `torch.no_grad()`.
    """
    embedding_layer = text_encoder.get_input_embeddings()
    weight = embedding_layer.weight  # [V, D]

    with torch.no_grad():
        existing = weight.data
        mean_vec = existing.mean(dim=0)
        std_vec = existing.std(dim=0)

        # Target: random Gaussian matching vocab statistics.
        for tid in target_token_ids:
            weight[tid] = torch.randn_like(mean_vec) * std_vec + mean_vec

        # Residual: per-image initialization from descriptive captions.
        if residual_token_ids:
            if residual_init_prompts is None:
                for group in residual_token_ids:
                    for tid in group:
                        weight[tid] = torch.randn_like(mean_vec) * std_vec + mean_vec
            else:
                if len(residual_init_prompts) != len(residual_token_ids):
                    raise ValueError(
                        f"Got {len(residual_init_prompts)} init prompts for "
                        f"{len(residual_token_ids)} residual token groups."
                    )
                for prompt, group in zip(residual_init_prompts, residual_token_ids):
                    init_vec = _mean_embedding_for_text(prompt, tokenizer, text_encoder, device)
                    init_vec = init_vec.to(dtype=weight.dtype, device=weight.device)
                    for tid in group:
                        weight[tid] = init_vec.clone()


def save_learned_embeddings(
    text_encoder,
    target_token_ids: Sequence[int],
    residual_token_ids: Sequence[Sequence[int]],
    save_path: str | Path,
) -> None:
    """Save the learned token embeddings as a single tensor in the order
    [target_0, target_1, ..., aux0_0, aux0_1, ..., aux1_0, ...].
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    weight = text_encoder.get_input_embeddings().weight.data
    flat_ids: List[int] = list(target_token_ids) + [
        tid for group in residual_token_ids for tid in group
    ]
    embeds = weight[flat_ids].detach().cpu().clone()
    torch.save(embeds, save_path)


def load_learned_embeddings(
    text_encoder,
    tokenizer,
    embeds_path: str | Path,
    *,
    num_target_tokens: int = 1,
    num_residual_vectors: int = 0,
    num_images: int = 0,
) -> List[str]:
    """Add tokens to `tokenizer`, resize `text_encoder`, write the saved embeddings
    into the new slots, and return the list of *target* token names (in order).

    The saved tensor must have layout
    `[num_target_tokens] + [num_images * num_residual_vectors]` rows, matching what
    `save_learned_embeddings` writes.

    For inference, residual tokens are loaded back so that any prompt can use them
    if desired; the standard inference path only references the target tokens.
    """
    embeds_path = Path(embeds_path)
    if not embeds_path.exists():
        raise FileNotFoundError(f"Learned embeddings not found at {embeds_path}")
    embeds = torch.load(embeds_path, map_location="cpu")

    target_tokens = build_target_tokens(num_target_tokens)
    residual_token_groups = build_residual_tokens(num_images, num_residual_vectors)

    flat_residual = [t for group in residual_token_groups for t in group]
    all_new_tokens = target_tokens + flat_residual

    expected = num_target_tokens + num_images * num_residual_vectors
    if embeds.shape[0] != expected:
        raise ValueError(
            f"Embedding tensor at {embeds_path} has {embeds.shape[0]} rows, "
            f"expected {expected} (num_target_tokens={num_target_tokens}, "
            f"num_images={num_images}, num_residual_vectors={num_residual_vectors})."
        )

    new_ids = add_tokens_to_tokenizer(tokenizer, text_encoder, all_new_tokens)
    weight = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        for tid, row in zip(new_ids, embeds):
            weight[tid] = row.to(dtype=weight.dtype, device=weight.device)

    return target_tokens
