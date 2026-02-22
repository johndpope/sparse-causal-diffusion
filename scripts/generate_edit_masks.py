#!/usr/bin/env python3
"""Pre-generate synthetic edit masks for EditCtrl training.

OPTIONAL — masks can be generated on-the-fly during training.
This script is useful for reproducible experiments and debugging.

Generates random rectangle and ellipse masks for each sample in a dataset,
saving them alongside the pre-encoded latents.

Usage:
    python scripts/generate_edit_masks.py \
        --data_root /media/2TB/isometric_i2v_training \
        --output_dir /media/2TB/isometric_i2v_training/edit_masks \
        --num_variants 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scd.utils.mask_utils import generate_random_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic edit masks for EditCtrl training")

    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to preprocessed training data (contains latents/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for masks. Defaults to <data_root>/edit_masks/"
    )
    parser.add_argument(
        "--num_variants", type=int, default=3,
        help="Number of mask variants per sample"
    )
    parser.add_argument(
        "--min_area", type=float, default=0.05,
        help="Minimum mask area fraction"
    )
    parser.add_argument(
        "--max_area", type=float, default=0.6,
        help="Maximum mask area fraction"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    latents_dir = data_root / "latents"

    if not latents_dir.exists():
        print(f"ERROR: Latents directory not found: {latents_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_root / "edit_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all latent files
    latent_files = sorted(latents_dir.glob("*.pt"))
    print(f"Found {len(latent_files)} latent files in {latents_dir}")

    torch.manual_seed(args.seed)

    total_generated = 0

    for idx, latent_path in enumerate(latent_files):
        # Load latent to get dimensions
        latent_data = torch.load(latent_path, map_location="cpu", weights_only=True)
        latents = latent_data["latents"]  # [C, F, H, W] or [1, C, F, H, W]

        if latents.dim() == 5:
            _, C, F, H, W = latents.shape
        else:
            C, F, H, W = latents.shape

        # Approximate pixel dimensions (for mask generation)
        # VAE: temporal=8x, spatial=32x
        pixel_F = F * 8
        pixel_H = H * 32
        pixel_W = W * 32

        sample_name = latent_path.stem

        for variant in range(args.num_variants):
            mask = generate_random_masks(
                batch_size=1,
                num_frames=pixel_F,
                height=pixel_H,
                width=pixel_W,
                min_area=args.min_area,
                max_area=args.max_area,
            )  # [1, 1, F, H, W]

            mask_path = output_dir / f"{sample_name}_mask_{variant:02d}.pt"
            torch.save({
                "mask": mask.squeeze(0),  # [1, F, H, W]
                "pixel_dims": (pixel_F, pixel_H, pixel_W),
                "latent_dims": (F, H, W),
                "area_fraction": mask.mean().item(),
            }, mask_path)

            total_generated += 1

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(latent_files)} samples...")

    print(f"\nGenerated {total_generated} masks ({args.num_variants} per sample)")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
