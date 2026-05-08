"""Small utilities shared across the training/inference scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_init_prompts(init_prompts_path: str | Path, concept_name: str, num_images: int) -> List[str]:
    """Load the per-image residual-token init captions for a given concept.

    `init_prompts.json` maps `concept_name -> list[str]`. Length must equal the number
    of reference images for the concept; if not, raise — the user almost certainly
    wants to know rather than silently use a wrong-size init.
    """
    init_prompts_path = Path(init_prompts_path)
    with init_prompts_path.open() as f:
        data = json.load(f)
    if concept_name not in data:
        raise KeyError(
            f"Concept '{concept_name}' not in {init_prompts_path}. "
            "Add an entry — see datasets/README.md §'Adding a new concept'."
        )
    prompts = data[concept_name]
    if len(prompts) != num_images:
        raise ValueError(
            f"init_prompts.json has {len(prompts)} captions for '{concept_name}' "
            f"but the reference set has {num_images} images."
        )
    return prompts


def slugify_prompt(prompt: str) -> str:
    """Filesystem-safe filename derived from a prompt template."""
    return (
        prompt.replace(" ", "_")
        .replace(",", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
