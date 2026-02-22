#!/usr/bin/env python3
"""Pre-compute semantic object masks for EditCtrl training using Qwen2.5-VL grounding.

For each of the 8 unique source videos, detects objects with bounding boxes,
then saves per-sample mask data that the training dataloader can use instead of
random rectangles.

Output: /media/2TB/isometric_i2v_training/semantic_masks/{idx:03d}.pt
  Each file contains:
    - objects: list[dict] with {name, bbox_norm [x1,y1,x2,y2] in 0-1 range}
    - masks: Tensor [N_objects, H, W] binary masks at latent spatial resolution
    - frame_size: (W, H) original pixel size

Usage:
    python scripts/compute_semantic_masks.py [--device cuda:1] [--viz]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def load_qwen(device="cuda:1"):
    """Load Qwen2.5-VL in 8-bit on the specified GPU."""
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    print(f"Loading Qwen2.5-VL-7B-Instruct on {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        ),
        device_map={"": device},
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return model, processor


def detect_objects(model, processor, image_path: str) -> str:
    """Ask Qwen VL to identify objects with bounding boxes."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text": (
                        "List every distinct object and person visible in this isometric 3D room scene. "
                        "For each, give the bounding box as pixel coordinates.\n"
                        "Use EXACTLY this format, one per line:\n"
                        "OBJECT: <name>, BOX: (<x1>,<y1>),(<x2>,<y2>)\n"
                        "where x1,y1 is top-left and x2,y2 is bottom-right.\n"
                        "Include people, furniture, decorations, architectural elements."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[Image.open(image_path)],
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.decode(generated, skip_special_tokens=True)


def parse_objects(response: str, img_w: int, img_h: int) -> list[dict]:
    """Parse bounding boxes from Qwen VL response.

    Handles multiple output formats from Qwen VL:
    - OBJECT: name, BOX: (x1,y1,x2,y2)         — single group, parens
    - OBJECT: name, BOX: [x1,y1,x2,y2]          — single group, brackets
    - OBJECT: name, BOX: (x1,y1),(x2,y2)        — two groups
    - NAME: <name>, BOX: (x1,y1,x2,y2)          — name label format
    - **name**: (x1,y1),(x2,y2)                  — markdown bold
    - N. **name**: [x1,y1,x2,y2]                 — numbered markdown
    """
    objects = []
    seen_names = set()

    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try to extract name
        name = "unknown"
        # Pattern 1: WORD: name, BOX:  (e.g. "OBJECT: Shelf, BOX:" or "WOMAN: <name>, BOX:")
        m = re.search(r'^([A-Z][\w\s]*?):\s*(?:<[^>]*>|[\w\s]+?)\s*,\s*BOX:', line)
        if m:
            # Check if there's a proper name between the label and BOX
            m2 = re.search(r':\s*(.+?),\s*BOX:', line)
            if m2:
                candidate = m2.group(1).strip().strip('<>').strip()
                if candidate and candidate != 'name':
                    name = candidate
                else:
                    # Use the leading label as name
                    name = m.group(1).strip()
        if name == "unknown":
            # Pattern 1b: OBJECT: name, BOX:
            m = re.search(r'OBJECT:\s*(.+?),\s*BOX:', line)
            if m:
                name = m.group(1).strip()
        if name == "unknown":
            # Pattern 2: **name**: coords
            m = re.search(r'\*\*(.+?)\*\*', line)
            if m:
                name = m.group(1).strip()
        if name == "unknown":
            # Pattern 3: NAME: ... BOX: or NAME: (coords
            m = re.search(r'^(?:\d+\.\s*)?([A-Za-z][\w\s]+?):\s*[\(\[\{<]', line)
            if m:
                name = m.group(1).strip()
            else:
                m = re.search(r'^(?:\d+\.\s*)?([A-Za-z][\w\s]+?),\s*BOX:', line)
                if m:
                    name = m.group(1).strip()

        # Extract coordinates — try multiple patterns
        coord_patterns = [
            # 4 numbers in one group: (x1,y1,x2,y2) or [x1,y1,x2,y2]
            r'[\(\[](\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)[\)\]]',
            # Two separate groups: (x1,y1),(x2,y2)
            r'\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)',
            # Mixed brackets/parens
            r'[\(\[](\d+)\s*,\s*(\d+)[\)\]]\s*,?\s*[\(\[](\d+)\s*,\s*(\d+)[\)\]]',
            # Just 4 consecutive numbers after BOX:
            r'BOX:\s*\D*(\d+)\D+(\d+)\D+(\d+)\D+(\d+)',
        ]

        for pattern in coord_patterns:
            m = re.search(pattern, line)
            if m:
                x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))

                # Qwen sometimes outputs in 0-1000 normalized coords
                if max(x1, y1, x2, y2) > max(img_w, img_h) * 1.2:
                    x1 = int(x1 * img_w / 1000)
                    y1 = int(y1 * img_h / 1000)
                    x2 = int(x2 * img_w / 1000)
                    y2 = int(y2 * img_h / 1000)

                # Clamp to image bounds
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))

                # Ensure valid box
                if x2 <= x1 or y2 <= y1:
                    break

                # Normalize to 0-1 range
                bbox_norm = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

                # Deduplicate
                key = f"{name}_{x1}_{y1}"
                if key not in seen_names:
                    seen_names.add(key)
                    objects.append({
                        "name": name,
                        "bbox_pixel": [x1, y1, x2, y2],
                        "bbox_norm": bbox_norm,
                    })
                break

    return objects


def objects_to_masks(
    objects: list[dict],
    latent_h: int = 36,
    latent_w: int = 24,
) -> torch.Tensor:
    """Convert bounding boxes to binary masks at latent spatial resolution.

    LTX-2 latent shape for 768×1152 input: [C, F, 36, 24] where
    H_latent=36 corresponds to pixel H=1152 and W_latent=24 corresponds to pixel W=768.

    Returns: [N_objects, latent_h, latent_w] binary float tensor
    """
    masks = torch.zeros(len(objects), latent_h, latent_w)

    for i, obj in enumerate(objects):
        x1_n, y1_n, x2_n, y2_n = obj["bbox_norm"]
        # Map normalized coords to latent grid
        # Note: latent_h corresponds to image height, latent_w to image width
        lx1 = int(x1_n * latent_w)
        ly1 = int(y1_n * latent_h)
        lx2 = max(lx1 + 1, int(x2_n * latent_w + 0.5))
        ly2 = max(ly1 + 1, int(y2_n * latent_h + 0.5))
        # Clamp
        lx1 = max(0, min(lx1, latent_w - 1))
        ly1 = max(0, min(ly1, latent_h - 1))
        lx2 = max(1, min(lx2, latent_w))
        ly2 = max(1, min(ly2, latent_h))
        masks[i, ly1:ly2, lx1:lx2] = 1.0

    return masks


def visualize_detections(
    image_path: str,
    objects: list[dict],
    save_path: str,
):
    """Draw detected bounding boxes on the image for visual inspection."""
    img = Image.open(image_path).copy()
    draw = ImageDraw.Draw(img)
    colors = ["red", "lime", "blue", "yellow", "cyan", "magenta", "orange", "purple",
              "pink", "turquoise", "gold", "white", "coral", "khaki", "violet", "tomato"]

    for i, obj in enumerate(objects):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = obj["bbox_pixel"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Draw name label
        label = obj["name"]
        draw.rectangle([x1, max(0, y1 - 16), x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill="black")

    img.save(save_path)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Compute semantic masks using Qwen VL grounding")
    parser.add_argument("--device", default="cuda:1", help="GPU device for Qwen VL")
    parser.add_argument("--viz", action="store_true", help="Save visualization images")
    parser.add_argument("--data-root", default="/media/2TB/isometric_i2v_training")
    parser.add_argument("--frames-root", default="/media/12TB/isometric_3d/r2_native_dataset/new_grok_frames")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Load metadata
    meta_path = os.path.join(args.data_root, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    pairs = meta["pairs"]
    print(f"Dataset: {len(pairs)} pairs from {meta['num_source_videos']} source videos")

    # Group pairs by video_id (only need to run grounding once per scene)
    video_groups = {}
    for p in pairs:
        vid = p["video_id"]
        if vid not in video_groups:
            video_groups[vid] = []
        video_groups[vid].append(p)

    print(f"Unique scenes: {len(video_groups)}")

    # Output directories
    mask_dir = args.output_dir or os.path.join(args.data_root, "semantic_masks")
    os.makedirs(mask_dir, exist_ok=True)
    if args.viz:
        viz_dir = os.path.join(mask_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

    # Load Qwen VL
    model, processor = load_qwen(args.device)

    # Process each unique scene
    all_results = {}
    for vid_idx, (video_id, group) in enumerate(video_groups.items()):
        print(f"\n{'='*60}")
        print(f"Scene {vid_idx+1}/{len(video_groups)}: {video_id[:12]}...")
        print(f"  {len(group)} clips from this scene")

        # Use the first frame of the first clip as representative
        # (isometric scenes are mostly static — same objects across all clips)
        first_pair = group[0]
        start_frame = first_pair["start_frame"]
        frame_path = os.path.join(
            args.frames_root,
            f"{video_id}_{start_frame:03d}.jpg",
        )

        if not os.path.exists(frame_path):
            # Try alternate frame numbering
            frame_path = os.path.join(args.frames_root, f"{video_id}_012.jpg")
            if not os.path.exists(frame_path):
                print(f"  WARNING: No frame found for {video_id}, skipping")
                continue

        img = Image.open(frame_path)
        print(f"  Frame: {os.path.basename(frame_path)} ({img.size[0]}x{img.size[1]})")

        # Run grounding
        response = detect_objects(model, processor, frame_path)
        objects = parse_objects(response, img.width, img.height)
        print(f"  Detected {len(objects)} objects:")
        for obj in objects:
            area_pct = (obj["bbox_norm"][2] - obj["bbox_norm"][0]) * (obj["bbox_norm"][3] - obj["bbox_norm"][1]) * 100
            print(f"    {obj['name']}: {obj['bbox_pixel']} ({area_pct:.1f}% area)")

        if len(objects) == 0:
            print(f"  WARNING: No objects detected, raw response:\n{response[:200]}")

        # Convert to latent-resolution masks
        # LTX-2 latent: [C, F, H_lat, W_lat] where for 768x1152: H_lat=36, W_lat=24
        latent_h, latent_w = 36, 24
        masks = objects_to_masks(objects, latent_h, latent_w)
        print(f"  Masks shape: {masks.shape} (latent {latent_h}x{latent_w})")

        # Visualization
        if args.viz:
            viz_path = os.path.join(viz_dir, f"{video_id[:12]}.png")
            visualize_detections(frame_path, objects, viz_path)
            print(f"  Viz saved: {viz_path}")

        # Store results for this scene
        all_results[video_id] = {
            "objects": objects,
            "masks": masks,
            "response": response,
            "frame_path": frame_path,
            "frame_size": img.size,
        }

        # Save per-sample mask files
        for pair in group:
            idx = pair["id"]
            save_data = {
                "video_id": video_id,
                "objects": objects,
                "masks": masks,  # [N_objects, H_lat, W_lat]
                "frame_size": img.size,
                "num_objects": len(objects),
            }
            save_path = os.path.join(mask_dir, f"{idx:03d}.pt")
            torch.save(save_data, save_path)

        print(f"  Saved masks for {len(group)} samples")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_saved = len([f for f in os.listdir(mask_dir) if f.endswith(".pt")])
    print(f"Total mask files saved: {total_saved}")
    print(f"Output directory: {mask_dir}")

    # Object statistics across all scenes
    all_obj_names = []
    for vid, res in all_results.items():
        all_obj_names.extend([o["name"] for o in res["objects"]])
    from collections import Counter
    name_counts = Counter(all_obj_names)
    print(f"\nObject types across all scenes:")
    for name, count in name_counts.most_common(20):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
