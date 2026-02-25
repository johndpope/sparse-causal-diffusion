#!/usr/bin/env python3
"""
Encode a subset of Ditto-1M videos into VAE latents for DDiT distillation training.

Videos are loaded from /media/12TB/Ditto-1M/videos_extracted/local/{0000..XXXX}/*.mp4,
randomly sampled, resized, and encoded through the LTX-2 VAE encoder.

Text embeddings are cycled from existing isometric conditions_final (128 files).

Usage:
    python scripts/encode_ditto_for_ddit.py \
        --count 5000 \
        --output_dir /media/2TB/ddit_training_data \
        --device cuda:1 \
        --width 768 --height 512 --num_frames 25 \
        --batch_size 1

Output structure:
    output_dir/
        latents/{00000..04999}.pt       # VAE latents
        conditions_final/{00000..04999}.pt  # text embeddings (cycled from isometric)
        manifest.json                    # mapping from idx -> source video path
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_decord(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Load video with decord, resize, return [1, 3, F, H, W] in [-1, 1]."""
    from decord import VideoReader, cpu as decord_cpu

    vr = VideoReader(path, ctx=decord_cpu(0))
    total = len(vr)
    if total < num_frames:
        raise ValueError(f"Video has only {total} frames, need {num_frames}")

    # Take the first num_frames frames
    indices = list(range(num_frames))
    frames = vr.get_batch(indices).asnumpy()  # [F, H_orig, W_orig, 3] uint8

    # Convert to torch [F, 3, H, W] float
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [F, 3, H, W]

    # Resize with bilinear interpolation
    frames_t = F.interpolate(frames_t, size=(height, width), mode="bilinear", align_corners=False)

    # Normalize from [0, 255] to [-1, 1]
    frames_t = frames_t / 127.5 - 1.0

    # Reshape to [1, 3, F, H, W]
    frames_t = frames_t.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, F, H, W]
    return frames_t


def load_video_torchvision(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Fallback: load video with torchvision, resize, return [1, 3, F, H, W] in [-1, 1]."""
    import torchvision.io as tvio

    # Read video: returns [F, H, W, 3] uint8
    video, _, info = tvio.read_video(path, pts_unit="sec")
    total = video.shape[0]
    if total < num_frames:
        raise ValueError(f"Video has only {total} frames, need {num_frames}")

    frames = video[:num_frames]  # [F, H, W, 3]
    frames_t = frames.permute(0, 3, 1, 2).float()  # [F, 3, H, W]

    # Resize
    frames_t = F.interpolate(frames_t, size=(height, width), mode="bilinear", align_corners=False)

    # Normalize [0, 255] -> [-1, 1]
    frames_t = frames_t / 127.5 - 1.0

    # Reshape to [1, 3, F, H, W]
    frames_t = frames_t.permute(1, 0, 2, 3).unsqueeze(0)
    return frames_t


# Choose video loader
try:
    from decord import VideoReader  # noqa: F401
    _load_video = load_video_decord
    logger.info("Using decord for video loading")
except ImportError:
    _load_video = load_video_torchvision
    logger.info("Decord not available, falling back to torchvision")


# ---------------------------------------------------------------------------
# Discover all videos
# ---------------------------------------------------------------------------

def discover_videos(root_dir: str) -> list:
    """Recursively find all .mp4 files under root_dir."""
    videos = []
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Video root directory not found: {root_dir}")

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        for mp4 in sorted(subdir.glob("*.mp4")):
            videos.append(str(mp4))

    logger.info(f"Discovered {len(videos)} videos in {root_dir}")
    return videos


# ---------------------------------------------------------------------------
# Load VAE encoder
# ---------------------------------------------------------------------------

def load_vae_encoder(device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
    """Load LTX-2 VAE encoder."""
    sys.path.insert(0, "/home/johndpope/Documents/GitHub/ltx2-omnitransfer/ltx-trainer/src")
    from ltx_trainer.model_loader import load_video_vae_encoder

    checkpoint = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
    logger.info(f"Loading VAE encoder from {checkpoint} ...")
    encoder = load_video_vae_encoder(checkpoint, device=device, dtype=dtype)
    logger.info("VAE encoder loaded successfully")
    return encoder


# ---------------------------------------------------------------------------
# Main encoding loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Encode Ditto-1M videos for DDiT training")
    parser.add_argument("--video_dir", type=str,
                        default="/media/12TB/Ditto-1M/videos_extracted/local/",
                        help="Root directory containing video subdirs")
    parser.add_argument("--count", type=int, default=5000,
                        help="Number of videos to randomly sample and encode")
    parser.add_argument("--output_dir", type=str,
                        default="/media/2TB/ddit_training_data",
                        help="Output directory for latents and conditions")
    parser.add_argument("--conditions_src", type=str,
                        default="/media/2TB/isometric_i2v_training/conditions_final/",
                        help="Source directory for text embeddings to cycle from")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device for VAE encoding (cuda:1 = Blackwell 25GB)")
    parser.add_argument("--width", type=int, default=768,
                        help="Target video width in pixels (must be divisible by 32)")
    parser.add_argument("--height", type=int, default=512,
                        help="Target video height in pixels (must be divisible by 32)")
    parser.add_argument("--num_frames", type=int, default=25,
                        help="Number of frames to take (should be 1 + 8*k, e.g. 25)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for VAE encoding (1 recommended for memory)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible video selection")
    parser.add_argument("--resume", action="store_true",
                        help="Resume encoding: skip indices that already have latent files")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Data type for VAE encoder")
    args = parser.parse_args()

    # Validate frame count
    if (args.num_frames - 1) % 8 != 0:
        valid = [1 + 8 * k for k in range(20)]
        raise ValueError(
            f"num_frames={args.num_frames} is not valid (must be 1 + 8*k). "
            f"Valid options: {valid}"
        )

    # Validate resolution
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError(f"Width ({args.width}) and height ({args.height}) must be divisible by 32")

    # Compute expected latent shape
    latent_f = 1 + (args.num_frames - 1) // 8
    latent_h = args.height // 32
    latent_w = args.width // 32
    logger.info(
        f"Target: {args.width}x{args.height}, {args.num_frames} frames -> "
        f"latent shape [128, {latent_f}, {latent_h}, {latent_w}]"
    )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Create output directories
    latents_dir = Path(args.output_dir) / "latents"
    conditions_dir = Path(args.output_dir) / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # Discover and sample videos
    all_videos = discover_videos(args.video_dir)
    if len(all_videos) < args.count:
        logger.warning(
            f"Requested {args.count} videos but only {len(all_videos)} available. "
            f"Using all {len(all_videos)} videos."
        )
        count = len(all_videos)
    else:
        count = args.count

    random.seed(args.seed)
    selected_videos = sorted(random.sample(all_videos, count))
    logger.info(f"Selected {count} videos for encoding")

    # Find available condition files to cycle through
    cond_src = Path(args.conditions_src)
    cond_files = sorted(cond_src.glob("*.pt"))
    if not cond_files:
        raise FileNotFoundError(f"No .pt files found in {args.conditions_src}")
    num_cond = len(cond_files)
    logger.info(f"Found {num_cond} condition files to cycle through")

    # Determine which indices to process (resume support)
    indices_to_process = list(range(count))
    if args.resume:
        existing = set()
        for f in latents_dir.glob("*.pt"):
            try:
                existing.add(int(f.stem))
            except ValueError:
                pass
        indices_to_process = [i for i in indices_to_process if i not in existing]
        logger.info(f"Resume mode: {count - len(indices_to_process)} already done, "
                     f"{len(indices_to_process)} remaining")

    if not indices_to_process:
        logger.info("All videos already encoded. Nothing to do.")
        return

    # Load VAE encoder
    vae_encoder = load_vae_encoder(device=args.device, dtype=dtype)
    vae_encoder.eval()

    # Build manifest (full, not just remaining)
    manifest = {}
    for idx, video_path in enumerate(selected_videos):
        manifest[f"{idx:05d}"] = video_path

    # Encoding loop
    t_start = time.time()
    success_count = 0
    fail_count = 0
    skipped_count = count - len(indices_to_process)

    pbar = tqdm(indices_to_process, desc="Encoding videos", unit="vid")

    for idx in pbar:
        video_path = selected_videos[idx]
        latent_path = latents_dir / f"{idx:05d}.pt"
        cond_path = conditions_dir / f"{idx:05d}.pt"

        try:
            # Load and preprocess video
            video_tensor = _load_video(
                video_path, args.num_frames, args.height, args.width
            )  # [1, 3, F, H, W] float32 in [-1, 1]

            # Move to device and cast to encoder dtype
            video_tensor = video_tensor.to(device=args.device, dtype=dtype)

            # Encode through VAE
            with torch.no_grad():
                latents = vae_encoder(video_tensor)  # [1, 128, F', H', W']

            # Remove batch dim and move to CPU
            latents = latents.squeeze(0).cpu()  # [128, F', H', W']

            # Save latent in the same format as isometric data
            torch.save(
                {
                    "latents": latents,
                    "num_frames": torch.tensor([latents.shape[1]], dtype=torch.int64),
                    "height": torch.tensor([latents.shape[2]], dtype=torch.int64),
                    "width": torch.tensor([latents.shape[3]], dtype=torch.int64),
                },
                latent_path,
            )

            # Copy cycled condition file
            src_cond = cond_files[idx % num_cond]
            shutil.copy2(str(src_cond), str(cond_path))

            success_count += 1

        except Exception as e:
            fail_count += 1
            logger.warning(f"[{idx:05d}] FAILED: {video_path} -> {e}")
            # Record failure in manifest
            manifest[f"{idx:05d}"] = f"FAILED: {video_path} ({e})"
            continue

        # Update progress bar
        elapsed = time.time() - t_start
        rate = (success_count + fail_count) / max(elapsed, 1)
        remaining = (len(indices_to_process) - success_count - fail_count) / max(rate, 0.01)
        pbar.set_postfix(
            ok=success_count,
            fail=fail_count,
            rate=f"{rate:.1f}v/s",
            eta=f"{remaining / 60:.0f}m",
        )

        # Periodic VRAM cleanup
        if (success_count + fail_count) % 100 == 0:
            torch.cuda.empty_cache()

    # Save manifest
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest["__meta__"] = {
        "total_selected": count,
        "encoded": success_count + skipped_count,
        "failed": fail_count,
        "video_dir": args.video_dir,
        "resolution": f"{args.width}x{args.height}",
        "num_frames": args.num_frames,
        "latent_shape": f"[128, {latent_f}, {latent_h}, {latent_w}]",
        "seed": args.seed,
        "dtype": args.dtype,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Encoding complete!")
    logger.info(f"  Total selected:  {count}")
    logger.info(f"  Previously done:  {skipped_count}")
    logger.info(f"  Newly encoded:    {success_count}")
    logger.info(f"  Failed:           {fail_count}")
    logger.info(f"  Time:             {elapsed / 60:.1f} minutes")
    logger.info(f"  Rate:             {(success_count) / max(elapsed, 1):.2f} videos/sec")
    logger.info(f"  Latent shape:     [128, {latent_f}, {latent_h}, {latent_w}]")
    logger.info(f"  Output:           {args.output_dir}")
    logger.info(f"  Manifest:         {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
