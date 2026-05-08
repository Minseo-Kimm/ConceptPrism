#!/usr/bin/env python
"""Generate residual-token initialization captions for a concept.

ConceptPrism's residual tokens are initialized from the mean text-encoder embedding of
an 8–32-word descriptive caption per reference image. This script asks Gemini 2.5 Flash
to produce one such caption for every image in `<dataset_root>/<concept_name>/` and
appends an entry to `init_prompts.json`.

Usage:
    export GEMINI_API_KEY=...
    python scripts/generate_init_prompts.py \\
        --concept_name my_new_concept \\
        --dataset_root datasets \\
        --init_prompts_path datasets/init_prompts.json

You only need to run this once per new concept. Existing entries in `init_prompts.json`
are left untouched unless you pass `--overwrite`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import google.generativeai as genai
from PIL import Image

PROMPT_TEMPLATE = """\
You are an AI that describes visual scenes in detail. Your mission is to describe a given image using 8 to 32 words of text.

Instructions:
1. Purpose of Description: Describe the image in sufficient detail that the original scene can be visually reconstructed based solely on the text description.
2. Elements to Describe: You must include not only the main subject but also the core visual elements that constitute the scene, such as its key attributes, background, composition, lighting, and style.
3. All descriptions must be written in English and be between 8 and 32 words.

For each image provided, generate a single description that follows the above rules.
Return the descriptions as a JSON array of strings, with one description per image,
in the same order as the images were provided. Output the JSON array only — no
prose, no Markdown fences.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Generate residual-token init captions.")
    parser.add_argument("--concept_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Directory holding the <concept_name>/ image folder.")
    parser.add_argument("--init_prompts_path", type=str, default="datasets/init_prompts.json")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Gemini API key. Defaults to env GEMINI_API_KEY.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing entry for this concept.")
    return parser.parse_args()


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def list_images(directory: Path):
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"No images with suffixes {IMAGE_SUFFIXES} in {directory}")
    return paths


def parse_json_array(text: str):
    """The model is asked for a bare JSON array, but sometimes wraps it in fences."""
    text = text.strip()
    if text.startswith("```"):
        # strip leading/trailing fences (with or without ```json language tag)
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return json.loads(text)


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or pass --api_key.")
    genai.configure(api_key=api_key)

    init_path = Path(args.init_prompts_path)
    init_path.parent.mkdir(parents=True, exist_ok=True)
    if init_path.exists():
        with init_path.open() as f:
            init_data = json.load(f)
    else:
        init_data = {}

    if args.concept_name in init_data and not args.overwrite:
        print(f"[!] Concept '{args.concept_name}' already exists in {init_path}. "
              f"Pass --overwrite to regenerate.", file=sys.stderr)
        return

    image_dir = Path(args.dataset_root) / args.concept_name
    image_paths = list_images(image_dir)
    print(f"[+] Captioning {len(image_paths)} images in {image_dir}")

    images = []
    for p in image_paths:
        img = Image.open(p)
        if img.format == "MPO":
            img = img.convert("RGB")
        images.append(img)

    model = genai.GenerativeModel(args.model)
    response = model.generate_content([PROMPT_TEMPLATE] + images)

    captions = parse_json_array(response.text)
    if len(captions) != len(image_paths):
        raise SystemExit(
            f"Model returned {len(captions)} captions for {len(image_paths)} images. "
            f"Re-run; if this keeps happening, captions per call may need to be split."
        )

    init_data[args.concept_name] = captions
    with init_path.open("w") as f:
        json.dump(init_data, f, indent=2)

    print(f"[+] Wrote {len(captions)} captions for '{args.concept_name}' to {init_path}")
    for img_path, caption in zip(image_paths, captions):
        print(f"    {img_path.name} → {caption}")


if __name__ == "__main__":
    main()
