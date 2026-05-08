#!/usr/bin/env bash
# Stage 1 + Stage 2 training for one concept on Stable Diffusion v2.1.
# Usage: bash scripts/train_sd21.sh <concept_name>
# Example: bash scripts/train_sd21.sh dog
#
# After training, render images with infer_sd21.py — see README.md.
set -euo pipefail

CONCEPT_NAME="${1:?usage: bash scripts/train_sd21.sh <concept_name>}"
DATA_ROOT="${DATA_ROOT:-datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/sd21/${CONCEPT_NAME}}"

python pipeline.py \
    --backbone sd21 \
    --concept_name "${CONCEPT_NAME}" \
    --instance_data_dir "${DATA_ROOT}/${CONCEPT_NAME}" \
    --output_root "${OUTPUT_ROOT}" \
    --stage1_steps 200 \
    --stage2_steps 200
