#!/usr/bin/env python3
"""Train DDiT adapter via knowledge distillation.

The DDiT adapter learns to produce similar outputs at coarser patch sizes
as the base model produces at the native resolution. This enables dynamic
patch scheduling at inference time for 1.6-3.5x speedup.

Two-phase training:
  Phase 1 (reconstruction): Train merge/unmerge layers to preserve latent info.
  Phase 2 (distillation):   Align adapter outputs with base model at native res.

Usage:
    python scripts/train_ddit_adapter.py \
        --model_path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --data_root /media/2TB/isometric_i2v_training \
        --output_dir outputs/ddit_adapter \
        --device cuda:1 \
        --scales 2,4 \
        --phase 1 \
        --steps 200 \
        --lr 5e-4

    # Phase 2 (after phase 1):
    python scripts/train_ddit_adapter.py \
        --model_path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --data_root /media/2TB/isometric_i2v_training \
        --output_dir outputs/ddit_adapter \
        --device cuda:1 \
        --scales 2,4 \
        --phase 2 \
        --resume outputs/ddit_adapter/ddit_adapter_phase1.safetensors \
        --steps 300 \
        --lr 1e-4
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_base_model(model_path: str, device: str = "cuda:1", quantize: bool = True):
    """Load the base LTX-2 model with int8-quanto quantization."""
    from ltx_core.model.transformer.model import LTXModel

    print(f"Loading base model from {model_path}...")
    raw_state_dict = load_file(model_path)

    # Strip 'model.diffusion_model.' prefix from checkpoint keys
    # (the checkpoint uses this prefix, but LTXModel expects bare keys)
    PREFIX = "model.diffusion_model."
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith(PREFIX):
            state_dict[k[len(PREFIX):]] = v
        else:
            state_dict[k] = v
    del raw_state_dict

    # LTX-2 19B architecture
    model = LTXModel(
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
    )

    # Load weights on CPU first
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: {len(state_dict) - len(missing)} keys, "
          f"missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        # Filter to show non-audio missing keys
        non_audio = [k for k in missing if 'audio' not in k]
        if non_audio:
            print(f"  WARNING: {len(non_audio)} non-audio missing keys: {non_audio[:5]}...")
    del state_dict
    model = model.to(dtype=torch.bfloat16)

    if quantize:
        print("  Quantizing to int8-quanto (block-by-block)...")
        from optimum.quanto import freeze, quantize as quanto_quantize, qint8

        # Modules to keep in float32/bf16
        EXCLUDE_PATTERNS = [
            "patchify_proj", "audio_patchify_proj",
            "proj_out", "audio_proj_out",
            "*adaln*", "time_proj", "timestep_embedder*",
            "caption_projection*", "audio_caption_projection*",
            "*norm*",
        ]
        SKIP_ROOT = {"patchify_proj", "audio_patchify_proj", "proj_out",
                      "audio_proj_out", "audio_caption_projection"}

        # Quantize transformer blocks one-by-one on GPU
        for i, block in enumerate(model.transformer_blocks):
            block.to(device, dtype=torch.bfloat16)
            quanto_quantize(block, weights=qint8, exclude=EXCLUDE_PATTERNS)
            freeze(block)
            block.to("cpu")
            torch.cuda.empty_cache()
            if (i + 1) % 12 == 0:
                print(f"    Quantized {i+1}/48 blocks")

        # Quantize remaining non-skip modules
        for name, module in model.named_children():
            if name == "transformer_blocks" or name in SKIP_ROOT:
                continue
            module.to(device, dtype=torch.bfloat16)
            quanto_quantize(module, weights=qint8, exclude=EXCLUDE_PATTERNS)
            freeze(module)
            module.to("cpu")
        torch.cuda.empty_cache()
        print("  Quantization complete.")

    # Move to target device
    model = model.to(device)
    model.eval()

    # Enable gradient checkpointing for memory efficiency
    model._enable_gradient_checkpointing = True

    # Freeze all base model parameters
    for p in model.parameters():
        p.requires_grad_(False)

    vram = torch.cuda.memory_allocated(device) / 1e9
    print(f"  Model on {device}: inner_dim={model.inner_dim}, VRAM={vram:.1f}GB")
    return model


def create_ddit_adapter(inner_dim=4096, in_channels=128, scales=(2, 4), lora_rank=32,
                        device="cuda:1", base_model=None):
    """Create DDiT adapter, optionally initialized from base model weights."""
    from ltx_core.model.transformer.ddit import DDiTAdapter, DDiTConfig

    config = DDiTConfig(
        enabled=True,
        supported_scales=(1, *scales),
        lora_rank=lora_rank,
    )

    adapter = DDiTAdapter(
        inner_dim=inner_dim,
        in_channels=in_channels,
        config=config,
    )

    # Initialize from base model weights if available
    if base_model is not None:
        adapter.init_from_base_model(base_model)

    adapter = adapter.to(device=device, dtype=torch.bfloat16)

    total = sum(p.numel() for p in adapter.parameters())
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"DDiT adapter: {total:,} total, {trainable:,} trainable")
    return adapter


def load_training_data(data_root: str, max_samples: int = 128):
    """Load pre-encoded latents and text embeddings for distillation training."""
    latent_dir = Path(data_root) / "latents"
    cond_dir = Path(data_root) / "conditions_final"
    if not latent_dir.exists():
        raise FileNotFoundError(f"No latents directory at {latent_dir}")

    samples = []
    for pt_file in sorted(latent_dir.glob("*.pt"))[:max_samples]:
        data = torch.load(pt_file, weights_only=False, map_location="cpu")
        latents = data["latents"]  # [C, F, H, W]

        # Extract dims — handle both tensor and int formats
        nf = data.get("num_frames", latents.shape[1])
        h = data.get("height", latents.shape[2])
        w = data.get("width", latents.shape[3])
        nf = nf.item() if isinstance(nf, torch.Tensor) else int(nf)
        h = h.item() if isinstance(h, torch.Tensor) else int(h)
        w = w.item() if isinstance(w, torch.Tensor) else int(w)

        # Load matching text embedding
        cond_file = cond_dir / pt_file.name
        text_embeds = None
        text_mask = None
        if cond_file.exists():
            cond = torch.load(cond_file, weights_only=False, map_location="cpu")
            text_embeds = cond["video_prompt_embeds"]  # [1024, 3840]
            text_mask = cond["prompt_attention_mask"]   # [1024]

        samples.append({
            "latents": latents, "num_frames": nf, "height": h, "width": w,
            "text_embeds": text_embeds, "text_mask": text_mask,
        })

    print(f"Loaded {len(samples)} training samples from {latent_dir}")
    if samples:
        s = samples[0]
        has_text = s["text_embeds"] is not None
        print(f"  Shape: {s['latents'].shape}, dims: F={s['num_frames']} H={s['height']} W={s['width']}")
        print(f"  Text embeddings: {'yes' if has_text else 'no'}")
    return samples


def make_positions(nf, h, w, device, dtype=torch.bfloat16):
    """Create positions in LTX-2 format: [1, 3, seq_len, 2] with pixel-space bounds.

    LTX-2 uses use_middle_indices_grid=True, so positions must be [B, 3, seq, 2]
    where the last dim is [start, end) in pixel space.
    VAE scale factors: time=8, height=32, width=32.
    """
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            frames=nf, height=h, width=w, batch=1, channels=128,
        ),
        device=device,
    )
    pixel_coords = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=SpatioTemporalScaleFactors.default(),
        causal_fix=True,
    ).to(dtype)

    # Scale temporal by 1/fps (24 fps default)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / 24.0

    return pixel_coords  # [1, 3, seq_len, 2]


def prepare_sample(sample, device, dtype=torch.bfloat16):
    """Prepare a training sample: reshape latent, create positions, get text embeddings."""
    latent = sample["latents"]  # [C, F, H, W]
    nf, h, w = sample["num_frames"], sample["height"], sample["width"]
    seq_len = nf * h * w

    # [C, F, H, W] → [1, F*H*W, C]
    latent = latent.to(device=device, dtype=dtype)
    latent = latent.permute(1, 2, 3, 0).reshape(1, seq_len, -1)  # [F,H,W,C] → [1, seq, C]

    # Positions [1, 3, seq_len, 2] — LTX-2 pixel-space bounds format
    positions = make_positions(nf, h, w, device, dtype)

    # Text embeddings — use real if available, zero otherwise
    # IMPORTANT: text_mask must be int (not float) so _prepare_attention_mask
    # converts it via (mask-1)*finfo.max → 0 for attend, -inf for ignore
    if sample["text_embeds"] is not None:
        text_ctx = sample["text_embeds"].unsqueeze(0).to(device=device, dtype=dtype)  # [1, 1024, 3840]
        text_mask = sample["text_mask"].unsqueeze(0).to(device=device)                # [1, 1024] int64
    else:
        text_ctx = torch.zeros(1, 256, 3840, device=device, dtype=dtype)
        text_mask = torch.ones(1, 256, device=device, dtype=torch.int64)

    return latent, positions, nf, h, w, text_ctx, text_mask


# ---------------------------------------------------------------------------
# Phase 1: Reconstruction — train merge/unmerge to preserve information
# ---------------------------------------------------------------------------
def train_phase1(adapter, samples, args):
    """Train merge/unmerge reconstruction (no base model needed)."""
    device = args.device
    dtype = torch.bfloat16
    scales = tuple(int(s) for s in args.scales.split(","))

    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 1: Reconstruction training ({args.steps} steps) ===")
    print(f"  Scales: {scales}, LR: {args.lr}")

    losses = []
    step = 0
    t0 = time.time()

    while step < args.steps:
        for sample in samples:
            if step >= args.steps:
                break

            latent, positions, nf, h, w = prepare_sample(sample, device, dtype)

            total_loss = torch.tensor(0.0, device=device)
            for scale in scales:
                if h % scale != 0 or w % scale != 0:
                    continue

                ml = adapter.merge_layers[str(scale)]

                # Forward: merge → patchify_proj → proj_out → unmerge
                merged = ml.merge(latent, nf, h, w)           # [1, reduced, C*s*s]
                projected = ml.patchify_proj(merged)           # [1, reduced, inner_dim]
                projected = projected + ml.patch_id
                unprojected = ml.proj_out(projected)           # [1, reduced, C*s*s]
                reconstructed = ml.unmerge(unprojected, nf, h, w)  # [1, seq, C]

                # Reconstruction + residual
                if adapter.config.residual_weight > 0:
                    residual = ml.residual_block(latent)
                    reconstructed = reconstructed + adapter.config.residual_weight * residual

                loss = F.mse_loss(reconstructed, latent)
                total_loss = total_loss + loss

            if total_loss.item() > 0:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(total_loss.item())

            step += 1
            if step % 10 == 0:
                avg = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - t0
                print(f"  Step {step}/{args.steps} | Loss: {avg:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

            if step % args.save_interval == 0:
                path = out_dir / f"ddit_adapter_p1_step{step:05d}.safetensors"
                save_file(dict(adapter.state_dict()), str(path))
                print(f"  Saved: {path}")

    # Final save
    path = out_dir / "ddit_adapter_phase1.safetensors"
    save_file(dict(adapter.state_dict()), str(path))
    print(f"\nPhase 1 complete! {path}")
    print(f"  Final loss: {losses[-1]:.6f}, avg last 10: {sum(losses[-10:])/min(10,len(losses)):.6f}")
    return adapter


# ---------------------------------------------------------------------------
# Phase 2: Full distillation through base model
# ---------------------------------------------------------------------------
def get_teacher_output(base_model, latent, positions, dummy_ctx, dummy_mask, sigma, device):
    """Teacher: base model forward at native resolution (no_grad)."""
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    B, seq_len, C = latent.shape
    noise = torch.randn_like(latent)
    noisy = (1 - sigma) * latent + sigma * noise
    timesteps = torch.full((B, seq_len), sigma, device=device, dtype=latent.dtype)

    modality = Modality(
        enabled=True, latent=noisy, timesteps=timesteps, positions=positions,
        context=dummy_ctx, context_mask=dummy_mask,
    )
    perturbations = BatchedPerturbationConfig.empty(B)

    with torch.no_grad():
        output, _ = base_model(video=modality, audio=None, perturbations=perturbations)

    return output, noisy  # Return noisy latent so student uses same noise


def get_student_output(base_model, adapter, noisy_latent, positions,
                       dummy_ctx, dummy_mask, sigma, scale,
                       nf, h, w, device):
    """Student: merge → base model → unmerge at coarse resolution."""
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    B, seq_len, C = noisy_latent.shape
    ml = adapter.merge_layers[str(scale)]

    # 1. Merge spatial tokens
    merged = ml.merge(noisy_latent, nf, h, w)      # [B, reduced, C*s*s]
    projected = ml.patchify_proj(merged)             # [B, reduced, inner_dim]
    projected = projected + ml.patch_id

    # 2. Adjust positions for coarser grid
    merged_positions = adapter.adjust_positions(positions, scale, nf, h, w)
    new_h, new_w = h // scale, w // scale
    new_seq = nf * new_h * new_w

    # 3. Create dummy modality for preprocessor (gets text embedding, timestep, RoPE)
    timesteps = torch.full((B, new_seq), sigma, device=device, dtype=noisy_latent.dtype)
    dummy_latent = torch.zeros(B, new_seq, C, device=device, dtype=noisy_latent.dtype)

    dummy_mod = Modality(
        enabled=True, latent=dummy_latent, timesteps=timesteps,
        positions=merged_positions, context=dummy_ctx, context_mask=dummy_mask,
    )

    # Cast dtype to match patchify_proj (float32 when excluded from quantization)
    target_dtype = base_model.patchify_proj.weight.dtype
    if dummy_mod.latent.dtype != target_dtype:
        dummy_mod = Modality(
            enabled=dummy_mod.enabled, latent=dummy_mod.latent.to(target_dtype),
            timesteps=dummy_mod.timesteps, positions=dummy_mod.positions,
            context=dummy_mod.context, context_mask=dummy_mod.context_mask,
        )

    video_args = base_model.video_args_preprocessor.prepare(dummy_mod)

    # 4. Swap in our DDiT-projected tokens
    video_args = replace(video_args, x=projected.to(video_args.x.dtype))

    perturbations = BatchedPerturbationConfig.empty(B)

    # 5. Run through transformer blocks with gradient checkpointing
    # Temporarily set training=True to enable gradient checkpointing
    # (checkpointing is gated on `self.training` in _process_transformer_blocks)
    was_training = base_model.training
    base_model.train()
    video_out, _ = base_model._process_transformer_blocks(
        video=video_args, audio=None, perturbations=perturbations,
    )
    if not was_training:
        base_model.eval()

    # 6. Output processing: norm + scale_shift + proj_out
    x = video_out.x
    emb_ts = video_out.embedded_timestep
    scale_shift = base_model.scale_shift_table
    shift, scale_val = (
        scale_shift[None, None].to(device=x.device, dtype=x.dtype) + emb_ts[:, :, None]
    ).unbind(dim=2)
    x = base_model.norm_out(x)
    x = x * (1 + scale_val) + shift

    # 7. Use DDiT proj_out instead of base proj_out
    output_merged = ml.proj_out(x)                   # [B, reduced, C*s*s]

    # 8. Unmerge back to full resolution
    output = ml.unmerge(output_merged, nf, h, w)     # [B, seq, C]

    # 9. Residual refinement
    if adapter.config.residual_weight > 0:
        residual = ml.residual_block(noisy_latent)
        output = output + adapter.config.residual_weight * residual

    return output


def train_phase2(base_model, adapter, samples, args):
    """Distillation training through frozen base model."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = args.device
    dtype = torch.bfloat16
    scales = tuple(int(s) for s in args.scales.split(","))

    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 2: Distillation training ({args.steps} steps) ===")
    print(f"  Scales: {scales}, LR: {args.lr}")
    vram = torch.cuda.memory_allocated(device) / 1e9
    print(f"  VRAM before training: {vram:.1f}GB")

    losses = []
    step = 0
    t0 = time.time()

    while step < args.steps:
        for sample in samples:
            if step >= args.steps:
                break

            latent, positions, nf, h, w, text_ctx, text_mask = prepare_sample(sample, device, dtype)

            # Random sigma for this step
            sigma = torch.rand(1).item() * 0.9 + 0.05  # [0.05, 0.95]

            # Teacher: full-res forward (no_grad, low memory)
            teacher_out, noisy = get_teacher_output(
                base_model, latent, positions, text_ctx, text_mask, sigma, device
            )
            # Detach teacher to free its computation graph
            teacher_out = teacher_out.detach()
            del latent  # Free input latent
            torch.cuda.empty_cache()

            # Process one scale at a time to minimize peak VRAM
            step_losses = []
            for scale in scales:
                if h % scale != 0 or w % scale != 0:
                    continue

                # Student: coarse-res forward (gradients flow through adapter)
                student_out = get_student_output(
                    base_model, adapter, noisy, positions,
                    text_ctx, text_mask, sigma, scale, nf, h, w, device,
                )

                loss = F.mse_loss(student_out, teacher_out)

                # Backward per-scale to free activations immediately
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                step_losses.append(loss.item())

                del student_out, loss
                torch.cuda.empty_cache()

            if step_losses:
                scheduler.step()
                losses.append(sum(step_losses) / len(step_losses))

            # Free activation memory
            del teacher_out, noisy
            torch.cuda.empty_cache()

            step += 1
            if step % 5 == 0:
                avg = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - t0
                vram = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"  Step {step}/{args.steps} | Loss: {avg:.6f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Peak VRAM: {vram:.1f}GB | {elapsed:.0f}s")

            if step % args.save_interval == 0:
                path = out_dir / f"ddit_adapter_p2_step{step:05d}.safetensors"
                save_file(dict(adapter.state_dict()), str(path))
                print(f"  Saved: {path}")

    # Final save
    path = out_dir / "ddit_adapter_final.safetensors"
    save_file(dict(adapter.state_dict()), str(path))
    print(f"\nPhase 2 complete! {path}")
    if losses:
        print(f"  Final loss: {losses[-1]:.6f}")
    return adapter


def train(args):
    """Main training entry point."""
    device = args.device
    scales = tuple(int(s) for s in args.scales.split(","))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    samples = load_training_data(args.data_root, max_samples=args.max_samples)

    if args.phase == 1:
        # Phase 1: reconstruction (no base model needed)
        adapter = create_ddit_adapter(
            scales=scales, lora_rank=args.lora_rank, device=device,
        )
        adapter = train_phase1(adapter, samples, args)

    elif args.phase == 2:
        # Phase 2: distillation (needs base model)
        base_model = load_base_model(args.model_path, device, quantize=True)

        adapter = create_ddit_adapter(
            scales=scales, lora_rank=args.lora_rank, device=device,
            base_model=base_model,
        )

        # Load phase 1 weights if resuming
        if args.resume:
            print(f"Loading adapter weights from {args.resume}")
            state = load_file(args.resume)
            adapter.load_state_dict(state, strict=False)
            adapter = adapter.to(device=device)

        adapter = train_phase2(base_model, adapter, samples, args)

    elif args.phase == 0:
        # Phase 0: both phases sequentially
        print("Running Phase 1 + Phase 2 sequentially")

        # Phase 1
        adapter = create_ddit_adapter(
            scales=scales, lora_rank=args.lora_rank, device=device,
        )
        p1_steps = min(200, args.steps // 3)
        args_p1 = argparse.Namespace(**vars(args))
        args_p1.steps = p1_steps
        args_p1.lr = 5e-4
        adapter = train_phase1(adapter, samples, args_p1)

        # Phase 2
        base_model = load_base_model(args.model_path, device, quantize=True)
        adapter.init_from_base_model(base_model)
        p2_steps = args.steps - p1_steps
        args_p2 = argparse.Namespace(**vars(args))
        args_p2.steps = p2_steps
        adapter = train_phase2(base_model, adapter, samples, args_p2)

    # Save config
    config_path = out_dir / "ddit_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "scales": list(scales),
            "lora_rank": args.lora_rank,
            "inner_dim": 4096,
            "in_channels": 128,
            "threshold": 0.001,
            "percentile": 0.4,
            "warmup_steps": 3,
            "phase": args.phase,
            "total_steps": args.steps,
        }, f, indent=2)
    print(f"Config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DDiT adapter for LTX-2")
    parser.add_argument("--model_path", default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors",
                        help="Path to LTX-2 checkpoint")
    parser.add_argument("--data_root", default="/media/2TB/isometric_i2v_training",
                        help="Path to pre-encoded training data")
    parser.add_argument("--output_dir", default="outputs/ddit_adapter", help="Output directory")
    parser.add_argument("--device", default="cuda:1", help="Training device")
    parser.add_argument("--scales", default="2,4", help="Comma-separated patch scales")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="Training phase: 0=both, 1=reconstruction, 2=distillation")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=128, help="Max training samples")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--resume", type=str, default=None, help="Resume from adapter checkpoint")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
