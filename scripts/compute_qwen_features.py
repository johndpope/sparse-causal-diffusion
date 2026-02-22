#!/usr/bin/env python3
"""Pre-compute Qwen2.5-VL features for TMA training.

Reads source video frames (pre-extracted JPGs) and captions from the isometric
dataset, runs Qwen2.5-VL-7B-Instruct to extract hidden state features, and
saves them as .pt files aligned with the training dataset indices.

Each training sample maps to a (video_id, start_frame) pair. We feed Qwen:
  - 4 video frames (the clip the sample covers)
  - The sample's caption text
  - An editing task template

Output: qwen_vl_features/{idx:03d}.pt with keys:
  - qwen_features: [seq_len, hidden_dim]  (hidden_dim=3584 for 7B)
  - caption: str
  - video_id: str
  - hidden_dim: int

Usage:
    python scripts/compute_qwen_features.py \
        --frames_dir /media/12TB/isometric_3d/r2_native_dataset/new_grok_frames \
        --metadata /media/2TB/isometric_i2v_training/metadata.json \
        --output_dir /media/2TB/isometric_i2v_training/qwen_vl_features \
        --qwen_model Qwen/Qwen2.5-VL-7B-Instruct \
        --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute Qwen VL features for TMA")
    parser.add_argument(
        "--frames_dir", type=str,
        default="/media/12TB/isometric_3d/r2_native_dataset/new_grok_frames",
        help="Directory with pre-extracted frames ({video_id}_{frame:03d}.jpg)",
    )
    parser.add_argument(
        "--metadata", type=str,
        default="/media/2TB/isometric_i2v_training/metadata.json",
        help="Training dataset metadata.json",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/media/2TB/isometric_i2v_training/qwen_vl_features",
        help="Output directory for .pt feature files",
    )
    parser.add_argument(
        "--qwen_model", type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen VL model name or path",
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for Qwen VL")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 recommended for VLMs)")
    parser.add_argument("--max_frames_per_sample", type=int, default=4, help="Max frames to feed Qwen per sample")
    parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load Qwen in 8-bit")
    parser.add_argument("--start_idx", type=int, default=0, help="Start sample index (for resuming)")
    parser.add_argument("--end_idx", type=int, default=-1, help="End sample index (-1 for all)")
    return parser.parse_args()


EDITING_TASK_TEMPLATE = (
    "You are analyzing a video clip for a video editing task. "
    "Describe the visual content, scene layout, objects, style, and any motion or actions visible. "
    "Focus on details that would help an AI model understand what to preserve and what could be edited."
)


def get_frame_paths(
    frames_dir: str,
    video_id: str,
    start_frame: int,
    num_frames: int = 4,
    total_source_frames: int = 24,
) -> list[str]:
    """Get evenly-spaced frame paths for a training sample.

    The training dataset uses clip_stride=8 with 25 clip frames from source videos.
    Each source video has 24 extracted frames (1-indexed JPGs).
    We pick `num_frames` evenly spaced from the available range.
    """
    # Frame files are 1-indexed: {video_id}_{001..024}.jpg
    # Pick evenly spaced frames across the full video
    indices = []
    for i in range(num_frames):
        # Spread across the full 24 frames
        frame_idx = 1 + int(i * (total_source_frames - 1) / max(num_frames - 1, 1))
        frame_idx = min(frame_idx, total_source_frames)
        indices.append(frame_idx)

    paths = []
    for idx in indices:
        path = os.path.join(frames_dir, f"{video_id}_{idx:03d}.jpg")
        if os.path.exists(path):
            paths.append(path)
        else:
            # Try without zero-padding
            alt_path = os.path.join(frames_dir, f"{video_id}_{idx}.jpg")
            if os.path.exists(alt_path):
                paths.append(alt_path)

    return paths


def load_qwen_model(model_name: str, device: str, load_in_8bit: bool = True):
    """Load Qwen2.5-VL model and processor."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
    }
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        # device_map must be "auto" for quantized models
        model_kwargs["device_map"] = "auto"

    print(f"Loading Qwen VL model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, **model_kwargs
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name)

    # Qwen2.5-VL nests language config under text_config
    hidden_dim = model.config.text_config.hidden_size  # 3584 for 7B
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Device: {next(model.parameters()).device}")

    return model, processor, hidden_dim


def build_qwen_messages(frame_paths: list[str], caption: str) -> list[dict]:
    """Build Qwen VL conversation messages with frames + caption."""
    # Build image content list
    content = []

    # Add frames as images
    for path in frame_paths:
        content.append({"type": "image", "image": f"file://{path}"})

    # Add the task text
    content.append({
        "type": "text",
        "text": f"{EDITING_TASK_TEMPLATE}\n\nCaption: {caption}",
    })

    messages = [
        {"role": "user", "content": content},
    ]
    return messages


@torch.inference_mode()
def extract_features(
    model,
    processor,
    messages: list[dict],
    device: str,
) -> torch.Tensor:
    """Run Qwen VL and extract last hidden state features.

    Returns: [seq_len, hidden_dim] tensor
    """
    # Process the conversation
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[Image.open(c["image"].replace("file://", ""))
                for msg in messages
                for c in msg["content"]
                if c.get("type") == "image"],
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to model device
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Forward pass to get hidden states
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

    # Get the last hidden state (before the LM head)
    # Shape: [1, seq_len, hidden_dim]
    last_hidden = outputs.hidden_states[-1]

    # Remove batch dim → [seq_len, hidden_dim]
    features = last_hidden[0].float().cpu()

    return features


def main():
    args = parse_args()

    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)
    pairs = metadata["pairs"]

    # Resolve index range
    end_idx = args.end_idx if args.end_idx > 0 else len(pairs)
    pairs_to_process = pairs[args.start_idx:end_idx]
    print(f"Processing samples {args.start_idx} to {end_idx} ({len(pairs_to_process)} samples)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check which are already done
    already_done = set()
    for i in range(args.start_idx, end_idx):
        out_path = os.path.join(args.output_dir, f"{i:03d}.pt")
        if os.path.exists(out_path):
            already_done.add(i)
    if already_done:
        print(f"  {len(already_done)} samples already computed, skipping")

    # Load Qwen VL
    model, processor, hidden_dim = load_qwen_model(
        args.qwen_model, args.device, args.load_in_8bit
    )

    # Cache: since 16 samples share the same video_id, we can cache features per video
    # But each sample has a different caption, so we still need per-sample extraction
    # However, frame_paths repeat across samples with the same video_id, so we only
    # load images once per video_id

    errors = []
    for i, pair in enumerate(tqdm(pairs_to_process, desc="Computing Qwen features")):
        idx = args.start_idx + i
        out_path = os.path.join(args.output_dir, f"{idx:03d}.pt")

        if idx in already_done:
            continue

        video_id = pair["video_id"]
        caption = pair["caption"]
        start_frame = pair.get("start_frame", 0)

        # Get frame paths
        frame_paths = get_frame_paths(
            args.frames_dir, video_id, start_frame,
            num_frames=args.max_frames_per_sample,
        )

        if not frame_paths:
            print(f"  WARNING: No frames found for sample {idx} (video={video_id})")
            errors.append(idx)
            continue

        try:
            # Build messages
            messages = build_qwen_messages(frame_paths, caption)

            # Extract features
            features = extract_features(model, processor, messages, args.device)

            # Save
            torch.save({
                "qwen_features": features,  # [seq_len, hidden_dim]
                "caption": caption,
                "video_id": video_id,
                "hidden_dim": hidden_dim,
                "num_frames_used": len(frame_paths),
                "frame_paths": frame_paths,
            }, out_path)

            if idx % 16 == 0:
                tqdm.write(f"  Sample {idx}: features shape {features.shape}, video={video_id}")

        except Exception as e:
            print(f"  ERROR on sample {idx}: {e}")
            errors.append(idx)
            continue

    print(f"\nDone! Processed {len(pairs_to_process) - len(already_done) - len(errors)} samples")
    if errors:
        print(f"Errors on samples: {errors}")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
