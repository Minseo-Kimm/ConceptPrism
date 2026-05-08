"""Datasets used by ConceptPrism's two training stages.

Both stages iterate over `instance_data_root`, a directory of reference images. They
share an `index → image, residual_token_group_i` mapping that is consistent across
stages (so the residual tokens learned in stage 1 still correspond to the right
images in stage 2). To keep that mapping stable, we **sort** the image paths.

Variable names follow the paper. Reference image i is `x^(i)`; the K extra images
sampled per step (j ≠ i) are the **anchor images** `x^(j)` from Figure 2. The text
condition `t_target with t_residual^(i)` used by L_rec is `recon_prompt`; the
`t_residual^(i)`-only condition used by the L_excl injected branch is
`residual_prompt`; the L_excl reference branch ∅ is `null_prompt`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def list_image_paths(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise ValueError(f"Instance image root does not exist: {root}")
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images with suffixes {IMAGE_SUFFIXES} found in {root}")
    return paths


def _build_image_transforms(size: int, center_crop: bool):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def _residual_token_str(image_index: int, num_residual_vectors: int) -> str:
    """Return e.g. '<|res3|> <|res3_1|> <|res3_2|>' for image_index=3, n=3."""
    if num_residual_vectors == 0:
        return ""
    parts = [f"<|res{image_index}|>"]
    for j in range(1, num_residual_vectors):
        parts.append(f"<|res{image_index}_{j}|>")
    return " ".join(parts)


class ConceptPrismStage1Dataset(Dataset):
    """Stage 1 (Token Optimization) dataset.

    Each item supplies what we need for L_rec on image *i* and L_excl on K anchor
    images *j ≠ i*:

      - `pixel_values`: image *i*  (= x^(i)).
      - `anchor_pixel_values`: K stacked anchor-image tensors (j ≠ i; sampled per call).
      - `recon_prompt`: `"target ... target_{n-1} <|res_i|> ... <|res_i_{m-1}|>"` —
        c^(i) in the paper. Used for L_rec.
      - `residual_prompt`: `"<|res_i|> ... <|res_i_{m-1}|>"` — c_residual^(i) in the
        paper. The injected branch of L_excl.
      - `null_prompt`: `""` — ∅ in the paper. The reference branch of L_excl.
    """

    def __init__(
        self,
        instance_data_root: str | Path,
        target_tokens: Sequence[str],
        num_residual_vectors: int,
        size: int = 512,
        center_crop: bool = False,
        num_anchor_samples: int = 3,
    ) -> None:
        self.image_paths = list_image_paths(instance_data_root)
        self.num_images = len(self.image_paths)
        self.num_residual_vectors = num_residual_vectors
        self.target_token_str = " ".join(target_tokens)
        self.num_anchor_samples = min(num_anchor_samples, max(self.num_images - 1, 1))
        self.image_transforms = _build_image_transforms(size, center_crop)

    def __len__(self) -> int:
        return self.num_images

    def _load(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx])
        img = exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.image_transforms(img)

    def __getitem__(self, index: int) -> dict:
        index = index % self.num_images
        residual_str = _residual_token_str(index, self.num_residual_vectors)

        if residual_str:
            recon_prompt = f"{self.target_token_str} {residual_str}"
        else:
            recon_prompt = self.target_token_str

        # Sample anchor indices j ≠ index.
        anchor_indices: List[int] = []
        if self.num_images > 1:
            pool = [k for k in range(self.num_images) if k != index]
            anchor_indices = random.sample(pool, k=min(self.num_anchor_samples, len(pool)))
            while len(anchor_indices) < self.num_anchor_samples:
                # Fall back to sampling with replacement if not enough distinct images.
                anchor_indices.append(random.choice(pool))
        else:
            anchor_indices = [index] * self.num_anchor_samples

        return {
            "pixel_values": self._load(index),
            "anchor_pixel_values": torch.stack([self._load(j) for j in anchor_indices]),
            "recon_prompt": recon_prompt,        # c^(i)         — for L_rec
            "residual_prompt": residual_str,      # c_residual^(i) — for L_excl injected branch
            "null_prompt": "",                    # ∅             — for L_excl reference branch
        }


class ConceptPrismStage2Dataset(Dataset):
    """Stage 2 (Concept Disentangled Fine-Tuning) dataset.

    Returns the same `pixel_values` and `recon_prompt` as Stage 1 — for the standard
    diffusion reconstruction loss, conditioned on `target_token + residual_i` so the
    fine-tuner sees exactly the prompt format it will see during deployment with
    a target token only (the residual contribution is absorbed into the LoRA layers).
    """

    def __init__(
        self,
        instance_data_root: str | Path,
        target_tokens: Sequence[str],
        num_residual_vectors: int,
        size: int = 512,
        center_crop: bool = False,
    ) -> None:
        self.image_paths = list_image_paths(instance_data_root)
        self.num_images = len(self.image_paths)
        self.num_residual_vectors = num_residual_vectors
        self.target_token_str = " ".join(target_tokens)
        self.image_transforms = _build_image_transforms(size, center_crop)

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, index: int) -> dict:
        index = index % self.num_images
        img = Image.open(self.image_paths[index])
        img = exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        residual_str = _residual_token_str(index, self.num_residual_vectors)
        if residual_str:
            prompt = f"{self.target_token_str} {residual_str}"
        else:
            prompt = self.target_token_str

        return {
            "pixel_values": self.image_transforms(img),
            "prompt": prompt,
        }


def stage1_collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "anchor_pixel_values": torch.stack([b["anchor_pixel_values"] for b in batch]),
        "recon_prompts": [b["recon_prompt"] for b in batch],
        "residual_prompts": [b["residual_prompt"] for b in batch],
        "null_prompts": [b["null_prompt"] for b in batch],
    }


def stage2_collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
    }
