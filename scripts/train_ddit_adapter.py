#!/usr/bin/env python3
"""Train DDiT adapter via knowledge distillation.

The DDiT adapter learns to produce similar outputs at coarser patch sizes
as the base model produces at the native resolution. This enables dynamic
patch scheduling at inference time for 1.6-3.5x speedup.

Training procedure:
1. Generate synthetic videos using the base model (or use existing latents)
2. For each training sample:
   a. Run base model at native resolution → teacher output
   b. Run base model + DDiT adapter at coarse resolution → student output
   c. L2 distillation loss between teacher and student (in latent space)
3. Only DDiT adapter parameters are trained (LoRA + merge layers)

The adapter adds ~2-5M parameters per scale level.

Usage:
    python scripts/train_ddit_adapter.py \
        --model_path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --data_root /media/2TB/isometric_i2v_training \
        --output_dir outputs/ddit_adapter \
        --device cuda:0 \
        --scales 2 4 \
        --steps 500 \
        --lr 1e-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_base_model(model_path: str, device: str = "cuda:0"):
    """Load the base LTX-2 model (quantized)."""
    from ltx_core.model.transformer import LTXModel

    print(f"Loading base model from {model_path}...")
    # Use the standard loading pipeline
    from safetensors.torch import load_file
    state_dict = load_file(model_path)

    # Detect model configuration from state dict
    # LTX-2 19B: inner_dim=4096, 48 layers, in_channels=128
    model = LTXModel(
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
    )

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (expected for audio components)")
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"  Model loaded. inner_dim={model.inner_dim}")
    return model


def create_ddit_adapter(base_model, scales=(2, 4), lora_rank=32):
    """Create DDiT adapter initialized from base model weights."""
    from ltx_core.model.transformer.ddit import DDiTAdapter, DDiTConfig

    config = DDiTConfig(
        enabled=True,
        supported_scales=(1, *scales),
        lora_rank=lora_rank,
    )

    adapter = DDiTAdapter(
        inner_dim=base_model.inner_dim,
        in_channels=128,  # LTX-2 VAE latent channels
        config=config,
    )

    # Initialize from base model weights
    adapter.init_from_base_model(base_model)

    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"DDiT adapter: {total_params:,} total params, {trainable_params:,} trainable")

    return adapter


def load_training_data(data_root: str, max_samples: int = 128):
    """Load pre-encoded latents for distillation training."""
    latent_dir = Path(data_root) / "latents"
    if not latent_dir.exists():
        raise FileNotFoundError(f"No latents directory at {latent_dir}")

    samples = []
    for pt_file in sorted(latent_dir.glob("*.pt"))[:max_samples]:
        data = torch.load(pt_file, weights_only=False, map_location="cpu")
        latents = data["latents"]  # [C, F, H, W]
        samples.append({
            "latents": latents,
            "num_frames": data.get("num_frames", latents.shape[1]),
            "height": data.get("height", latents.shape[2]),
            "width": data.get("width", latents.shape[3]),
        })

    print(f"Loaded {len(samples)} training samples from {latent_dir}")
    return samples


def get_teacher_output(
    base_model: nn.Module,
    latent: torch.Tensor,
    positions: torch.Tensor,
    text_context: torch.Tensor,
    text_mask: torch.Tensor,
    sigma: float,
    device: str,
) -> torch.Tensor:
    """Run teacher (base model at native resolution) and get output."""
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    B = latent.shape[0]
    seq_len = latent.shape[1]

    # Create noisy input
    noise = torch.randn_like(latent)
    noisy = (1 - sigma) * latent + sigma * noise
    timesteps = torch.full((B, seq_len), sigma, device=device, dtype=latent.dtype)

    modality = Modality(
        enabled=True,
        latent=noisy,
        timesteps=timesteps,
        positions=positions,
        context=text_context,
        context_mask=text_mask,
    )

    perturbations = BatchedPerturbationConfig.empty(B)

    with torch.no_grad():
        output, _ = base_model(video=modality, audio=None, perturbations=perturbations)

    return output  # [B, seq_len, C]


def get_student_output(
    base_model: nn.Module,
    ddit_adapter,
    latent: torch.Tensor,
    positions: torch.Tensor,
    text_context: torch.Tensor,
    text_mask: torch.Tensor,
    sigma: float,
    scale: int,
    num_frames: int,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    """Run student (base model with DDiT coarse patches) and get output."""
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    B = latent.shape[0]
    seq_len = latent.shape[1]

    # Create noisy input (same noise as teacher)
    noise = torch.randn_like(latent)
    noisy = (1 - sigma) * latent + sigma * noise

    # Merge spatial tokens using DDiT adapter
    merged_latent = ddit_adapter.merge_layers[str(scale)].forward_patchify(
        noisy, num_frames, height, width
    )
    merged_positions = ddit_adapter.adjust_positions(
        positions, scale, num_frames, height, width
    )

    new_h = height // scale
    new_w = width // scale
    new_seq_len = num_frames * new_h * new_w

    timesteps = torch.full((B, new_seq_len), sigma, device=device, dtype=latent.dtype)

    modality = Modality(
        enabled=True,
        latent=merged_latent,  # Already projected to inner_dim
        timesteps=timesteps,
        positions=merged_positions,
        context=text_context,
        context_mask=text_mask,
    )

    perturbations = BatchedPerturbationConfig.empty(B)

    # Run through transformer blocks (skip patchify_proj since we already projected)
    video_args = base_model.video_args_preprocessor.prepare(
        Modality(
            enabled=True,
            latent=torch.zeros(B, new_seq_len, 128, device=device, dtype=latent.dtype),
            timesteps=timesteps,
            positions=merged_positions,
            context=text_context,
            context_mask=text_mask,
        )
    )
    # Replace x with our merged+projected tokens
    from dataclasses import replace
    video_args = replace(video_args, x=merged_latent)

    # Run transformer blocks
    video_out, _ = base_model._process_transformer_blocks(
        video=video_args, audio=None, perturbations=perturbations
    )

    # Unpatchify using DDiT adapter
    output = ddit_adapter.merge_layers[str(scale)].forward_unpatchify(
        video_out.x, noisy, num_frames, height, width,
        residual_weight=ddit_adapter.config.residual_weight,
    )

    return output  # [B, seq_len, C]


def train(args):
    """Main training loop."""
    device = args.device
    dtype = torch.bfloat16

    # Load base model
    base_model = load_base_model(args.model_path, device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    # Create DDiT adapter
    scales = tuple(int(s) for s in args.scales.split(","))
    adapter = create_ddit_adapter(base_model, scales=scales, lora_rank=args.lora_rank)
    adapter = adapter.to(device=device, dtype=dtype)

    # Load training data
    samples = load_training_data(args.data_root, max_samples=args.max_samples)

    # Create dummy text context (distillation doesn't need real captions)
    dummy_text = torch.randn(1, 1024, 3840, device=device, dtype=dtype)
    dummy_mask = torch.ones(1, 1024, device=device, dtype=dtype)

    # Optimizer — only adapter parameters
    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining DDiT adapter for {args.steps} steps...")
    print(f"  Scales: {scales}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")

    global_step = 0
    losses = []
    start_time = time.time()

    while global_step < args.steps:
        for sample in samples:
            if global_step >= args.steps:
                break

            # Prepare batch
            latent = sample["latents"].unsqueeze(0).to(device=device, dtype=dtype)
            num_frames = sample["num_frames"]
            height = sample["height"]
            width = sample["width"]
            seq_len = num_frames * height * width

            # Reshape latent from [1, C, F, H, W] to [1, F*H*W, C]
            if latent.ndim == 5:
                latent = latent.permute(0, 2, 3, 4, 1).reshape(1, seq_len, -1)
            elif latent.ndim == 4:
                # Already [C, F, H, W] → [1, F*H*W, C]
                latent = latent.unsqueeze(0) if latent.ndim == 3 else latent
                latent = latent.permute(0, 2, 3, 4, 1).reshape(1, seq_len, -1) if latent.ndim == 5 else latent

            # Create positions [1, 3, seq_len]
            t_coords = torch.arange(num_frames, device=device).unsqueeze(1).unsqueeze(1)
            h_coords = torch.arange(height, device=device).unsqueeze(0).unsqueeze(2)
            w_coords = torch.arange(width, device=device).unsqueeze(0).unsqueeze(0)
            t_grid = t_coords.expand(num_frames, height, width).reshape(-1).float()
            h_grid = h_coords.expand(num_frames, height, width).reshape(-1).float()
            w_grid = w_coords.expand(num_frames, height, width).reshape(-1).float()
            positions = torch.stack([t_grid, h_grid, w_grid], dim=0).unsqueeze(0)  # [1, 3, seq_len]

            # Random sigma for this step
            sigma = torch.rand(1).item() * 0.9 + 0.05  # [0.05, 0.95]

            # Train on each scale
            total_loss = 0.0
            for scale in scales:
                if height % scale != 0 or width % scale != 0:
                    continue

                # Teacher output (base model, native resolution)
                teacher_out = get_teacher_output(
                    base_model, latent, positions, dummy_text, dummy_mask, sigma, device
                )

                # Student output (DDiT adapter, coarse resolution)
                student_out = get_student_output(
                    base_model, adapter, latent, positions, dummy_text, dummy_mask,
                    sigma, scale, num_frames, height, width, device
                )

                # Distillation loss
                loss = F.mse_loss(student_out, teacher_out.detach())
                total_loss = total_loss + loss

            if total_loss > 0:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                losses.append(total_loss.item())

            global_step += 1

            if global_step % 10 == 0:
                avg_loss = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - start_time
                print(f"  Step {global_step}/{args.steps} | Loss: {avg_loss:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"{elapsed:.0f}s elapsed")

            if global_step % args.save_interval == 0:
                save_path = out_dir / f"ddit_adapter_step_{global_step:05d}.safetensors"
                state = {k: v for k, v in adapter.state_dict().items()}
                save_file(state, str(save_path))
                print(f"  Saved checkpoint: {save_path}")

    # Final save
    save_path = out_dir / "ddit_adapter_final.safetensors"
    state = {k: v for k, v in adapter.state_dict().items()}
    save_file(state, str(save_path))
    print(f"\nTraining complete! Final checkpoint: {save_path}")

    # Save config
    config_path = out_dir / "ddit_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "scales": list(scales),
            "lora_rank": args.lora_rank,
            "inner_dim": base_model.inner_dim,
            "in_channels": 128,
            "threshold": 0.001,
            "percentile": 0.4,
            "warmup_steps": 3,
            "total_steps": args.steps,
            "final_loss": losses[-1] if losses else None,
        }, f, indent=2)
    print(f"Config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DDiT adapter for LTX-2")
    parser.add_argument("--model_path", required=True, help="Path to LTX-2 checkpoint")
    parser.add_argument("--data_root", required=True, help="Path to pre-encoded training data")
    parser.add_argument("--output_dir", default="outputs/ddit_adapter", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Training device")
    parser.add_argument("--scales", default="2,4", help="Comma-separated patch scales to train")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank for adapter")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=128, help="Max training samples")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint interval")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
