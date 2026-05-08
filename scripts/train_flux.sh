#!/usr/bin/env bash
# Stage 1 + Stage 2 training for one concept on FLUX.1.
# Usage: bash scripts/train_flux.sh <concept_name>
# Example: bash scripts/train_flux.sh dog
#
# Default: bf16 transformer (needs ~40GB VRAM at 512x512 with grad checkpointing).
# To run on a 24GB card, set USE_4BIT=1 to load the FLUX transformer in 4-bit.
#
# After training, render images with infer_flux.py — see README.md.
set -euo pipefail

CONCEPT_NAME="${1:?usage: bash scripts/train_flux.sh <concept_name>}"
DATA_ROOT="${DATA_ROOT:-datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/flux/${CONCEPT_NAME}}"
USE_4BIT="${USE_4BIT:-0}"

EXTRA=()
if [[ "${USE_4BIT}" == "1" ]]; then
    EXTRA+=( "--use_4bit_transformer" )
fi

python pipeline.py \
    --backbone flux \
    --concept_name "${CONCEPT_NAME}" \
    --instance_data_dir "${DATA_ROOT}/${CONCEPT_NAME}" \
    --output_root "${OUTPUT_ROOT}" \
    --stage1_steps 200 \
    --stage2_steps 500 \
    "${EXTRA[@]}"
