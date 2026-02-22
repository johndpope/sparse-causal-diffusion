#!/usr/bin/env python3
"""Extract instance masks from MuLAn dataset for training data augmentation.

MuLAn (CVPR 2024) provides 44K+ images with multi-layer RGBA instance
decompositions. This script extracts instance alpha masks and saves them as
compact tensor files for use as diverse edit masks during EditCtrl training.

Supports two input modes:

1. ANNOTATION MODE (--mode annotations, recommended):
   Reads .p.zl annotation files directly. Each contains instance_alpha
   masks. Does NOT require the original COCO/LAION base images.

   Input: directory of .p.zl files (extracted from MuLAn RAR archive)
   Usage:
       unrar x -e /media/12TB/mulan_raw/mulan.part001.rar /media/12TB/mulan_annotations/
       python scripts/extract_mulan_masks.py \
           --mode annotations \
           --input_dir /media/12TB/mulan_annotations \
           --output_dir /media/12TB/mulan_masks \
           --target_size 64 --workers 8

2. DECOMPOSED MODE (--mode decomposed):
   Reads decomposed PNG layers from dataset_decomposition.py output.
   Requires running the full decomposition pipeline first (needs COCO images).

   Input: directory of {image_id}/ subdirs with layer_*.png files
   Usage:
       python scripts/extract_mulan_masks.py \
           --mode decomposed \
           --input_dir /media/2TB/mulan_layers/images \
           --output_dir /media/2TB/mulan_masks \
           --target_size 64 --workers 8

Output: Compact mask library directory
    masks/
        {image_id}.pt  — dict with:
            masks: [N, H, W] float32 alpha masks (N = number of instances)
            areas: [N] float32 area fractions for each mask
            image_size: (H, W) original image dimensions
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pickle
import zlib

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Mode 1: Extract from .p.zl annotation files (no COCO images needed)
# --------------------------------------------------------------------------- #

def extract_masks_from_annotation(
    pzl_path: Path,
    min_area: float = 0.01,
    max_area: float = 0.8,
    max_instances: int = 10,
    target_size: int | None = None,
) -> dict | None:
    """Extract instance masks from a MuLAn .p.zl annotation file.

    The annotation pickle contains instance_alpha (PIL Image alpha channel)
    and original_image_mask (numpy array) for each instance. We extract
    the instance_alpha directly — no base image needed.

    Args:
        pzl_path: Path to .p.zl annotation file
        min_area: Minimum mask area fraction
        max_area: Maximum mask area fraction
        max_instances: Maximum instances per image
        target_size: Resize masks to this square size

    Returns:
        Dict with masks, areas, image_size, or None if no valid masks
    """
    try:
        with open(pzl_path, "rb") as f:
            annotation = pickle.loads(zlib.decompress(f.read()))
    except Exception:
        return None

    instances = annotation.get("instances", {})
    if not instances:
        return None

    masks = []
    areas = []
    img_size = None

    for inst_key, inst_data in instances.items():
        if len(masks) >= max_instances:
            break

        # Try instance_alpha first (PIL Image), then original_image_mask (numpy)
        alpha = None

        if "instance_alpha" in inst_data:
            alpha_img = inst_data["instance_alpha"]
            if isinstance(alpha_img, Image.Image):
                alpha = np.array(alpha_img, dtype=np.float32) / 255.0
            elif isinstance(alpha_img, np.ndarray):
                alpha = alpha_img.astype(np.float32)
                if alpha.max() > 1.0:
                    alpha = alpha / 255.0

        if alpha is None and "original_image_mask" in inst_data:
            mask_arr = inst_data["original_image_mask"]
            if isinstance(mask_arr, np.ndarray):
                alpha = mask_arr.astype(np.float32)
                if alpha.max() > 1.0:
                    alpha = alpha / 255.0
                # Handle multi-channel masks
                if alpha.ndim == 3:
                    alpha = alpha.mean(axis=-1)

        if alpha is None:
            continue

        # Handle 2D vs squeezable
        if alpha.ndim != 2:
            alpha = alpha.squeeze()
        if alpha.ndim != 2:
            continue

        if img_size is None:
            img_size = alpha.shape

        # Binarize
        binary = (alpha > 0.5).astype(np.float32)
        area = binary.mean()

        if area < min_area or area > max_area:
            continue

        masks.append(binary)
        areas.append(area)

    if not masks:
        return None

    H, W = masks[0].shape
    mask_tensor = torch.from_numpy(np.stack(masks))  # [N, H, W]

    if target_size is not None and (H != target_size or W != target_size):
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(1),
            size=(target_size, target_size),
            mode="nearest",
        ).squeeze(1)

    return {
        "masks": mask_tensor,
        "areas": torch.tensor(areas, dtype=torch.float32),
        "image_size": (H, W),
    }


def process_one_annotation(args_tuple):
    """Worker function for annotation mode."""
    pzl_path, min_area, max_area, max_instances, target_size, output_dir = args_tuple
    image_id = pzl_path.stem.replace(".p", "")  # Remove .p from {id}.p.zl
    output_path = output_dir / f"{image_id}.pt"

    if output_path.exists():
        return image_id, True, "skipped (exists)"

    result = extract_masks_from_annotation(
        pzl_path, min_area, max_area, max_instances, target_size
    )

    if result is None:
        return image_id, False, "no valid masks"

    torch.save(result, output_path)
    n = result["masks"].shape[0]
    return image_id, True, f"{n} masks"


# --------------------------------------------------------------------------- #
#  Mode 2: Extract from decomposed PNG layers
# --------------------------------------------------------------------------- #

def extract_masks_from_image(
    image_dir: Path,
    min_area: float = 0.01,
    max_area: float = 0.8,
    max_instances: int = 10,
    target_size: int | None = None,
) -> dict | None:
    """Extract instance masks from a single MuLAn decomposed image.

    Args:
        image_dir: Directory containing layer_0.png, layer_1.png, etc.
        min_area: Minimum mask area fraction to include
        max_area: Maximum mask area fraction to include
        max_instances: Maximum number of instance masks per image
        target_size: If set, resize masks to (target_size, target_size)

    Returns:
        Dict with masks, areas, image_size, or None if no valid masks found
    """
    layer_files = sorted(image_dir.glob("layer_*.png"))
    if len(layer_files) < 2:
        return None  # Need at least background + 1 instance

    masks = []
    areas = []

    for layer_file in layer_files:
        # Skip layer_0 (background) — we want instance masks only
        layer_name = layer_file.stem
        if layer_name == "layer_0":
            continue

        try:
            img = Image.open(layer_file)
        except Exception:
            continue

        # Extract alpha channel
        if img.mode == "RGBA":
            alpha = np.array(img.split()[-1], dtype=np.float32) / 255.0
        elif img.mode == "LA":
            alpha = np.array(img.split()[-1], dtype=np.float32) / 255.0
        elif img.mode == "L":
            alpha = np.array(img, dtype=np.float32) / 255.0
        else:
            # RGB without alpha — use luminance as mask
            gray = img.convert("L")
            alpha = np.array(gray, dtype=np.float32) / 255.0

        # Compute area fraction
        area = alpha.mean()
        if area < min_area or area > max_area:
            continue

        # Binarize with threshold (MuLAn alphas are mostly clean but can have anti-aliasing)
        binary_alpha = (alpha > 0.5).astype(np.float32)
        binary_area = binary_alpha.mean()
        if binary_area < min_area or binary_area > max_area:
            continue

        masks.append(binary_alpha)
        areas.append(binary_area)

        if len(masks) >= max_instances:
            break

    if not masks:
        return None

    # Stack to tensor
    H, W = masks[0].shape
    mask_tensor = torch.from_numpy(np.stack(masks))  # [N, H, W]

    # Optionally resize to target resolution
    if target_size is not None and (H != target_size or W != target_size):
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(1),  # [N, 1, H, W]
            size=(target_size, target_size),
            mode="nearest",
        ).squeeze(1)  # [N, target_size, target_size]

    return {
        "masks": mask_tensor,
        "areas": torch.tensor(areas, dtype=torch.float32),
        "image_size": (H, W),
    }


def process_one_image(args_tuple):
    """Worker function for parallel processing."""
    image_dir, min_area, max_area, max_instances, target_size, output_dir = args_tuple
    image_id = image_dir.name
    output_path = output_dir / f"{image_id}.pt"

    if output_path.exists():
        return image_id, True, "skipped (exists)"

    result = extract_masks_from_image(
        image_dir, min_area, max_area, max_instances, target_size
    )

    if result is None:
        return image_id, False, "no valid masks"

    torch.save(result, output_path)
    n = result["masks"].shape[0]
    return image_id, True, f"{n} masks"


def build_index(output_dir: Path) -> None:
    """Build an index file mapping mask files to their properties."""
    mask_files = sorted(output_dir.glob("*.pt"))
    index = {}
    total_masks = 0

    for f in tqdm(mask_files, desc="Building index"):
        data = torch.load(f, map_location="cpu", weights_only=True)
        n = data["masks"].shape[0]
        total_masks += n
        index[f.stem] = {
            "num_masks": n,
            "areas": data["areas"].tolist(),
            "image_size": data["image_size"],
        }

    index_path = output_dir / "index.pt"
    torch.save(index, index_path)
    print(f"Index saved: {len(index)} images, {total_masks} total masks → {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract instance masks from MuLAn dataset")
    parser.add_argument("--mode", type=str, default="annotations",
                        choices=["annotations", "decomposed"],
                        help="Input mode: 'annotations' reads .p.zl files directly "
                             "(no COCO images needed), 'decomposed' reads PNG layers")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory (annotations: dir of .p.zl files; "
                             "decomposed: dir of image subdirectories)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for mask library")
    parser.add_argument("--min_area", type=float, default=0.01,
                        help="Minimum mask area fraction (default 0.01)")
    parser.add_argument("--max_area", type=float, default=0.8,
                        help="Maximum mask area fraction (default 0.8)")
    parser.add_argument("--max_instances", type=int, default=10,
                        help="Max instances per image (default 10)")
    parser.add_argument("--target_size", type=int, default=64,
                        help="Resize masks to this square size (default 64). "
                             "Masks are resized to latent-like resolution for efficiency. "
                             "Set 0 to keep original size.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default 8)")
    parser.add_argument("--build_index_only", action="store_true",
                        help="Only rebuild the index file")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = args.target_size if args.target_size > 0 else None

    if args.build_index_only:
        build_index(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "annotations":
        # Mode 1: Read .p.zl annotation files directly
        pzl_files = sorted(input_dir.glob("*.p.zl"))
        if not pzl_files:
            # Also check for nested structure
            pzl_files = sorted(input_dir.rglob("*.p.zl"))
        if not pzl_files:
            print(f"No .p.zl annotation files found in {input_dir}")
            return

        print(f"Mode: annotations (direct .p.zl extraction, no COCO images needed)")
        print(f"Found {len(pzl_files)} annotation files in {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Filter: area [{args.min_area:.2f}, {args.max_area:.2f}], "
              f"max {args.max_instances} instances/image")
        if target_size:
            print(f"Resize: {target_size}x{target_size}")

        work_items = [
            (f, args.min_area, args.max_area, args.max_instances, target_size, output_dir)
            for f in pzl_files
        ]
        worker_fn = process_one_annotation

    else:
        # Mode 2: Read decomposed PNG layers
        image_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        if not image_dirs:
            print(f"No image directories found in {input_dir}")
            return

        print(f"Mode: decomposed (PNG layers)")
        print(f"Found {len(image_dirs)} image directories in {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Filter: area [{args.min_area:.2f}, {args.max_area:.2f}], "
              f"max {args.max_instances} instances/image")
        if target_size:
            print(f"Resize: {target_size}x{target_size}")

        work_items = [
            (d, args.min_area, args.max_area, args.max_instances, target_size, output_dir)
            for d in image_dirs
        ]
        worker_fn = process_one_image

    # Process in parallel
    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_fn, item): item[0] for item in work_items}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting masks"):
            image_id, ok, msg = future.result()
            if ok:
                success += 1
            else:
                failed += 1

    print(f"\nDone! Processed {success + failed} items: {success} success, {failed} no valid masks")

    # Build index
    build_index(output_dir)


if __name__ == "__main__":
    main()
