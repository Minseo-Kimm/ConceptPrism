#!/usr/bin/env python
"""End-to-end ConceptPrism training orchestrator.

Runs Stage 1 → Stage 2 for one concept on one backbone (sd21 or flux).

Example:
    python pipeline.py \\
        --backbone sd21 \\
        --concept_name dog \\
        --instance_data_dir datasets/dog \\
        --output_root runs/sd21/dog

After training, render images with `infer_sd21.py` / `infer_flux.py` directly:

    python infer_sd21.py \\
        --learned_embeddings_path runs/sd21/dog/stage1/learned_embeddings.pt \\
        --lora_weights_path runs/sd21/dog/stage2 \\
        --instance_data_dir datasets/dog \\
        --prompt "A * in the snow" \\
        --output_dir runs/sd21/dog/single

If you want to skip a stage, run the corresponding `train_*` script directly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd, description):
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    print("  $ " + " ".join(str(c) for c in cmd))
    print()
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_REPO_TREE_FETCH", "1")
    subprocess.run(cmd, check=True, env=env)


def parse_args():
    parser = argparse.ArgumentParser(description="ConceptPrism Stage 1 + Stage 2 orchestrator")
    parser.add_argument("--backbone", choices=["sd21", "flux"], required=True)
    parser.add_argument("--concept_name", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--init_prompts_path", type=str, default="datasets/init_prompts.json")
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                        help="Defaults to Manojb/stable-diffusion-2-1-base for sd21, "
                             "black-forest-labs/FLUX.1-dev for flux.")
    parser.add_argument("--num_target_tokens", type=int, default=1)
    parser.add_argument("--num_residual_vectors", type=int, default=None,
                        help="Defaults: 8 for sd21, 1 for flux.")
    parser.add_argument("--exclusion_loss_weight", type=float, default=None,
                        help="Defaults: 0.05 for sd21, 0.5 for flux.")

    parser.add_argument("--stage1_steps", type=int, default=200)
    parser.add_argument("--stage2_steps", type=int, default=None,
                        help="Defaults: 200 (checkpoint every 40) for sd21, 500 for flux.")
    parser.add_argument("--checkpointing_steps_stage2", type=int, default=None,
                        help="Defaults: 40 for sd21, 100 for flux.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit_transformer", action="store_true",
                        help="(FLUX only) Load the transformer in 4-bit.")

    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.backbone == "sd21":
        defaults = dict(
            pretrained="Manojb/stable-diffusion-2-1-base",
            num_residual=8,
            beta=0.05,
            stage2_steps=200,
            ckpt_stage2=40,
        )
    else:
        defaults = dict(
            pretrained="black-forest-labs/FLUX.1-dev",
            num_residual=1,
            beta=0.5,
            stage2_steps=500,
            ckpt_stage2=100,
        )
    pretrained = args.pretrained_model_name_or_path or defaults["pretrained"]
    num_residual = args.num_residual_vectors if args.num_residual_vectors is not None else defaults["num_residual"]
    beta = args.exclusion_loss_weight if args.exclusion_loss_weight is not None else defaults["beta"]
    stage2_steps = args.stage2_steps if args.stage2_steps is not None else defaults["stage2_steps"]
    ckpt_stage2 = (
        args.checkpointing_steps_stage2 if args.checkpointing_steps_stage2 is not None else defaults["ckpt_stage2"]
    )

    stage1_dir = output_root / "stage1"
    stage2_dir = output_root / "stage2"

    stage1_script = "train_sd21_stage1.py" if args.backbone == "sd21" else "train_flux_stage1.py"
    stage2_script = "train_sd21_stage2.py" if args.backbone == "sd21" else "train_flux_stage2.py"

    common_train = [
        f"--concept_name={args.concept_name}",
        f"--instance_data_dir={args.instance_data_dir}",
        f"--pretrained_model_name_or_path={pretrained}",
        f"--num_target_tokens={args.num_target_tokens}",
        f"--num_residual_vectors={num_residual}",
        f"--seed={args.seed}",
    ]

    if not args.skip_stage1:
        cmd = [
            "accelerate", "launch", stage1_script,
            *common_train,
            f"--init_prompts_path={args.init_prompts_path}",
            f"--exclusion_loss_weight={beta}",
            f"--max_train_steps={args.stage1_steps}",
            f"--output_dir={stage1_dir}",
        ]
        if args.backbone == "flux" and args.use_4bit_transformer:
            cmd.append("--use_4bit_transformer")
        run(cmd, f"Stage 1 (Token Optimization) for '{args.concept_name}' on {args.backbone}")

    if not args.skip_stage2:
        cmd = [
            "accelerate", "launch", stage2_script,
            *common_train,
            f"--learned_embeddings_path={stage1_dir / 'learned_embeddings.pt'}",
            f"--max_train_steps={stage2_steps}",
            f"--checkpointing_steps={ckpt_stage2}",
            f"--output_dir={stage2_dir}",
        ]
        if args.backbone == "flux" and args.use_4bit_transformer:
            cmd.append("--use_4bit_transformer")
        run(cmd, f"Stage 2 (Concept Disentangled Fine-Tuning) for '{args.concept_name}'")


if __name__ == "__main__":
    main()
