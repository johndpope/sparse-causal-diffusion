#!/usr/bin/env python3
"""
Generate synthetic training data for DDiT (Denoising Diffusion Distillation Transformer).

The DDiT paper observes that distillation does NOT need real video content --
we just need diverse latent patterns so the teacher and student can be compared
on a wide distribution of inputs.  This script creates 5000 (default) synthetic
samples from 128 real isometric latents using three strategies:

    1. Random Gaussian  (30%)  -- pure torch.randn, matched mean/std
    2. Augmented real    (40%)  -- spatial flips, channel permutations,
                                   additive Gaussian noise (SNR 5-20 dB),
                                   random crop+resize
    3. Interpolated      (30%)  -- lerp between two random real samples

Text embeddings are cycled from the 128 existing condition files (the model
only sees these during forward passes; diversity in latents is what matters).

Usage:
    python scripts/generate_synthetic_latents.py                    # defaults
    python scripts/generate_synthetic_latents.py --count 10000
    python scripts/generate_synthetic_latents.py --output_dir /tmp/synth --count 500
"""

import argparse
import math
import os
import shutil
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_COUNT = 5000
DEFAULT_OUTPUT_DIR = "/media/2TB/ddit_synthetic_data"
LATENT_SRC_DIR = "/media/2TB/isometric_i2v_training/latents"
COND_SRC_DIR = "/media/2TB/isometric_i2v_training/conditions_final"

# Latent shape: [C=128, F=4, H=36, W=24], dtype=bfloat16
C, F, H, W = 128, 4, 36, 24

# Distribution stats from real data (mean ~ -0.015, std ~ 1.10)
REAL_MEAN = -0.015
REAL_STD = 1.10

# Strategy proportions
FRAC_GAUSSIAN = 0.30
FRAC_AUGMENTED = 0.40
# FRAC_INTERP = 0.30 (remainder)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_real_latents(src_dir: str) -> list[torch.Tensor]:
    """Load all *.pt latent tensors from *src_dir* into a list (CPU, bfloat16)."""
    src = Path(src_dir)
    files = sorted(src.glob("*.pt"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt files found in {src_dir}")
    latents = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        latents.append(d["latents"])  # [C, F, H, W] bfloat16
    print(f"Loaded {len(latents)} real latent samples from {src_dir}")
    return latents


def random_gaussian(rng: torch.Generator) -> torch.Tensor:
    """Pure random Gaussian latent matched to real distribution."""
    z = torch.randn(C, F, H, W, generator=rng, dtype=torch.float32)
    z = z * REAL_STD + REAL_MEAN
    return z.to(torch.bfloat16)


def augment_real(latent: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Apply random augmentations to a real latent sample.

    Augmentations (applied independently with p=0.5 each):
        - Horizontal flip  (dim=-1, W axis)
        - Vertical flip    (dim=-2, H axis)
        - Temporal flip    (dim=-3, F axis)
        - Channel permutation (random permute of C=128 channels)
        - Additive Gaussian noise at SNR uniformly drawn from [5, 20] dB
        - Random spatial crop + bilinear resize back to original dims
    """
    x = latent.clone().float()  # work in fp32 for numerical stability

    # --- Spatial flips ---
    if torch.rand(1, generator=rng).item() > 0.5:
        x = x.flip(-1)  # horizontal
    if torch.rand(1, generator=rng).item() > 0.5:
        x = x.flip(-2)  # vertical
    if torch.rand(1, generator=rng).item() > 0.5:
        x = x.flip(-3)  # temporal

    # --- Channel permutation ---
    if torch.rand(1, generator=rng).item() > 0.5:
        perm = torch.randperm(C, generator=rng)
        x = x[perm]

    # --- Additive noise at random SNR ---
    if torch.rand(1, generator=rng).item() > 0.5:
        snr_db = 5.0 + 15.0 * torch.rand(1, generator=rng).item()  # [5, 20]
        signal_power = x.pow(2).mean()
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = noise_power.sqrt()
        noise = torch.randn_like(x) * noise_std
        x = x + noise

    # --- Random crop + bilinear resize back to original size ---
    if torch.rand(1, generator=rng).item() > 0.5:
        crop_frac_h = 0.75 + 0.25 * torch.rand(1, generator=rng).item()
        crop_frac_w = 0.75 + 0.25 * torch.rand(1, generator=rng).item()
        ch = max(1, int(H * crop_frac_h))
        cw = max(1, int(W * crop_frac_w))
        # random crop origin
        oh = int((H - ch) * torch.rand(1, generator=rng).item())
        ow = int((W - cw) * torch.rand(1, generator=rng).item())
        cropped = x[:, :, oh : oh + ch, ow : ow + cw]
        # Resize each frame back to [H, W] via bilinear interpolation
        frames = []
        for fi in range(F):
            frame = cropped[:, fi, :, :]  # [C, ch, cw]
            frame = frame.unsqueeze(0)    # [1, C, ch, cw]
            frame = torch.nn.functional.interpolate(
                frame, size=(H, W), mode="bilinear", align_corners=False
            )
            frames.append(frame.squeeze(0))  # [C, H, W]
        x = torch.stack(frames, dim=1)  # [C, F, H, W]

    return x.to(torch.bfloat16)


def interpolate_latents(
    a: torch.Tensor, b: torch.Tensor, rng: torch.Generator
) -> torch.Tensor:
    """Lerp between two real latent samples with random alpha in [0.1, 0.9]."""
    alpha = 0.1 + 0.8 * torch.rand(1, generator=rng).item()
    result = a.float() * alpha + b.float() * (1.0 - alpha)
    return result.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic latents for DDiT distillation training."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of synthetic samples to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--latent_src",
        type=str,
        default=LATENT_SRC_DIR,
        help=f"Source directory for real latents (default: {LATENT_SRC_DIR})",
    )
    parser.add_argument(
        "--cond_src",
        type=str,
        default=COND_SRC_DIR,
        help=f"Source directory for condition embeddings (default: {COND_SRC_DIR})",
    )
    args = parser.parse_args()

    # --- Setup ---
    rng = torch.Generator()
    rng.manual_seed(args.seed)

    out_latent_dir = Path(args.output_dir) / "latents"
    out_cond_dir = Path(args.output_dir) / "conditions_final"
    out_latent_dir.mkdir(parents=True, exist_ok=True)
    out_cond_dir.mkdir(parents=True, exist_ok=True)

    # --- Load real data ---
    real_latents = load_real_latents(args.latent_src)
    N_real = len(real_latents)

    cond_files = sorted(Path(args.cond_src).glob("*.pt"))
    N_cond = len(cond_files)
    if N_cond == 0:
        raise FileNotFoundError(f"No condition files in {args.cond_src}")
    print(f"Found {N_cond} condition files in {args.cond_src}")

    # --- Compute strategy counts ---
    n_gaussian = int(args.count * FRAC_GAUSSIAN)
    n_augmented = int(args.count * FRAC_AUGMENTED)
    n_interp = args.count - n_gaussian - n_augmented
    print(f"\nGenerating {args.count} synthetic samples:")
    print(f"  Gaussian:     {n_gaussian} ({FRAC_GAUSSIAN*100:.0f}%)")
    print(f"  Augmented:    {n_augmented} ({FRAC_AUGMENTED*100:.0f}%)")
    print(f"  Interpolated: {n_interp} ({(1-FRAC_GAUSSIAN-FRAC_AUGMENTED)*100:.0f}%)")
    print()

    # --- Build a shuffled schedule so strategies are interleaved ---
    # 0 = gaussian, 1 = augmented, 2 = interpolated
    schedule = (
        [0] * n_gaussian + [1] * n_augmented + [2] * n_interp
    )
    # Shuffle deterministically
    indices = torch.randperm(len(schedule), generator=rng).tolist()
    schedule = [schedule[i] for i in indices]

    # Metadata dict template
    meta = {
        "num_frames": torch.tensor([F], dtype=torch.int64),
        "height": torch.tensor([H], dtype=torch.int64),
        "width": torch.tensor([W], dtype=torch.int64),
    }

    # --- Pre-load condition data to avoid repeated disk reads ---
    cond_cache: dict[int, dict] = {}

    def get_cond(idx: int) -> dict:
        cond_idx = idx % N_cond
        if cond_idx not in cond_cache:
            cond_cache[cond_idx] = torch.load(
                cond_files[cond_idx], map_location="cpu", weights_only=True
            )
        return cond_cache[cond_idx]

    # --- Generate ---
    counts = {0: 0, 1: 0, 2: 0}
    strategy_names = {0: "gaussian", 1: "augmented", 2: "interpolated"}
    log_interval = max(1, args.count // 20)  # ~5% progress ticks

    for idx, strategy in enumerate(schedule):
        # Generate latent
        if strategy == 0:
            latent = random_gaussian(rng)
        elif strategy == 1:
            src_idx = int(torch.randint(0, N_real, (1,), generator=rng).item())
            latent = augment_real(real_latents[src_idx], rng)
        else:  # strategy == 2
            a_idx = int(torch.randint(0, N_real, (1,), generator=rng).item())
            b_idx = int(torch.randint(0, N_real, (1,), generator=rng).item())
            # Ensure different samples
            while b_idx == a_idx and N_real > 1:
                b_idx = int(torch.randint(0, N_real, (1,), generator=rng).item())
            latent = interpolate_latents(real_latents[a_idx], real_latents[b_idx], rng)

        counts[strategy] += 1

        # Save latent
        save_dict = {"latents": latent, **meta}
        torch.save(save_dict, out_latent_dir / f"{idx:05d}.pt")

        # Save condition (copy from source, cycling)
        cond_data = get_cond(idx)
        torch.save(cond_data, out_cond_dir / f"{idx:05d}.pt")

        # Progress
        if (idx + 1) % log_interval == 0 or idx == 0 or idx == len(schedule) - 1:
            pct = 100.0 * (idx + 1) / args.count
            print(
                f"  [{idx+1:>5d}/{args.count}] {pct:5.1f}%  "
                f"(g={counts[0]:>4d} a={counts[1]:>4d} i={counts[2]:>4d})"
            )

    # --- Summary ---
    print(f"\nDone! Saved {args.count} samples to {args.output_dir}")
    print(f"  Latents:    {out_latent_dir}/  ({args.count} files)")
    print(f"  Conditions: {out_cond_dir}/  ({args.count} files)")
    for s in [0, 1, 2]:
        print(f"  {strategy_names[s]:>12s}: {counts[s]}")

    # --- Quick sanity check ---
    print("\nSanity check on a few samples:")
    check_indices = [0, 1, args.count // 2, args.count - 1]
    for i in check_indices:
        d = torch.load(out_latent_dir / f"{i:05d}.pt", map_location="cpu", weights_only=True)
        lat = d["latents"].float()
        print(
            f"  {i:05d}: shape={tuple(d['latents'].shape)}, "
            f"dtype={d['latents'].dtype}, "
            f"mean={lat.mean():.4f}, std={lat.std():.4f}, "
            f"nf={d['num_frames'].item()}, h={d['height'].item()}, w={d['width'].item()}"
        )
    c = torch.load(out_cond_dir / "00000.pt", map_location="cpu", weights_only=True)
    print(
        f"  cond 00000: keys={list(c.keys())}, "
        f"embeds={tuple(c['video_prompt_embeds'].shape)}, "
        f"mask={tuple(c['prompt_attention_mask'].shape)}"
    )


if __name__ == "__main__":
    main()
