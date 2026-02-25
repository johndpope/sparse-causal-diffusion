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

    # Phase 2 with LoRA (after phase 1):
    python scripts/train_ddit_adapter.py \
        --model_path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --data_root /media/2TB/isometric_i2v_training \
        --output_dir outputs/ddit_adapter \
        --device cuda:0 \
        --scales 2,4 \
        --phase 2 \
        --resume outputs/ddit_adapter/ddit_adapter_phase1.safetensors \
        --steps 500 \
        --lr 1e-4 \
        --use_lora \
        --lora_target both
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_base_model(model_path: str, device: str = "cuda:1", quantize: bool = True):
    """Load the base LTX-2 model with int8-quanto quantization."""
    from ltx_core.model.transformer.model import LTXModel

    print(f"Loading base model from {model_path}...")
    raw_state_dict = load_file(model_path)

    # Strip 'model.diffusion_model.' prefix and convert to bfloat16 in one pass
    # to avoid keeping both fp32 and bf16 copies in RAM simultaneously
    PREFIX = "model.diffusion_model."
    state_dict = {}
    for k, v in raw_state_dict.items():
        key = k[len(PREFIX):] if k.startswith(PREFIX) else k
        state_dict[key] = v.to(torch.bfloat16) if v.is_floating_point() else v
    del raw_state_dict
    import gc; gc.collect()
    print(f"  State dict: {len(state_dict)} keys (converted to bfloat16)")

    # LTX-2 19B architecture — initialize in bfloat16 to avoid fp32 peak
    with torch.device("meta"):
        model = LTXModel(
            num_attention_heads=32,
            attention_head_dim=128,
            in_channels=128,
            out_channels=128,
            num_layers=48,
            cross_attention_dim=4096,
            caption_channels=3840,
        )

    # Load weights directly (model is on meta device, state dict is bf16)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"  Loaded: {len(state_dict) - len(unexpected)} keys, "
          f"missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        # Filter to show non-audio missing keys
        non_audio = [k for k in missing if 'audio' not in k]
        if non_audio:
            print(f"  WARNING: {len(non_audio)} non-audio missing keys: {non_audio[:5]}...")
    del state_dict
    gc.collect()

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


def setup_lora(model, lora_rank=32, lora_alpha=32, target="both"):
    """Apply PEFT LoRA to transformer blocks for DDiT distillation.

    LoRA gives the frozen transformer capacity to adapt its attention patterns
    for merged (coarser) tokens. Without it, the model can't compensate for
    spatial information loss during token merging.

    The DDiT paper applies LoRA to FFN layers. We also support attention-only
    or both attention+FFN targets.

    Args:
        model: LTXModel (quantized, frozen)
        lora_rank: LoRA rank (default 32, matching paper)
        lora_alpha: LoRA alpha scaling (default 32 = scaling factor of 1.0)
        target: "ff" (paper default), "attn", or "both"

    Returns:
        PeftModel wrapping the original model with LoRA adapters
    """
    from peft import LoraConfig as PeftLoraConfig, get_peft_model

    if target == "ff":
        target_modules = ["net.0.proj", "net.2"]
    elif target == "attn":
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    else:  # both
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",  # attention projections
            "net.0.proj",  # FF up-projection (GEGLU)
            "net.2",       # FF down-projection
        ]

    lora_config = PeftLoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied ({target}): {trainable:,} trainable / {total:,} total "
          f"({100*trainable/total:.2f}%)")
    print(f"  LoRA targets: {target_modules}")
    return model


def get_lora_params(model):
    """Extract LoRA trainable parameters from a PEFT-wrapped model."""
    return [p for p in model.parameters() if p.requires_grad]


def save_lora_weights(model, path):
    """Save only LoRA weights from a PEFT-wrapped model."""
    lora_state = {}
    for k, v in model.state_dict().items():
        if 'lora_' in k:
            lora_state[k] = v.contiguous().to(torch.bfloat16)
    save_file(lora_state, str(path))
    print(f"  Saved LoRA weights: {path} ({len(lora_state)} tensors)")
    return lora_state


def load_lora_weights(model, path):
    """Load LoRA weights into a PEFT-wrapped model."""
    lora_state = load_file(str(path))
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    # missing = model keys not in lora_state (base weights — expected)
    # unexpected = lora_state keys not in model (should be 0)
    loaded = len(lora_state) - len(unexpected)
    print(f"  Loaded LoRA weights: {path} ({loaded}/{len(lora_state)} tensors)")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected LoRA keys (not in model)")
    return model


def load_vae_decoder(model_path, device="cuda:1"):
    """Load the LTX-2 VAE decoder for pixel-space reconstructions."""
    try:
        # Try adding ltx-trainer source to path if not installed
        trainer_src = Path("/home/johndpope/Documents/GitHub/ltx2-omnitransfer/ltx-trainer/src")
        if trainer_src.exists() and str(trainer_src) not in sys.path:
            sys.path.insert(0, str(trainer_src))

        from ltx_trainer.model_loader import load_model as load_ltx_model
        components = load_ltx_model(
            checkpoint_path=model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_decoder=True,
            with_video_vae_encoder=False,
            with_text_encoder=False,
        )
        decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        print(f"  VAE decoder loaded (will decode on {device})")
        return decoder
    except Exception as e:
        print(f"  WARNING: Could not load VAE decoder: {e}")
        print(f"  (Latent-space visualizations will still work)")
        return None


def latent_to_images(latent, nf, h, w, channels=(0, 1, 2)):
    """Convert latent [B, F*H*W, C] to wandb images for visualization.

    Takes first frame, selected channels → normalized RGB-like image.
    """
    B, S, C = latent.shape
    x = latent[0].detach().float().cpu()  # [seq, C]
    x = x.view(nf, h, w, C)  # [F, H, W, C]
    frame = x[0]  # [H, W, C] — first frame

    # Select 3 channels and normalize to [0, 1]
    img = frame[:, :, list(channels)]  # [H, W, 3]
    lo, hi = img.min(), img.max()
    if hi - lo > 1e-6:
        img = (img - lo) / (hi - lo)
    else:
        img = img * 0 + 0.5
    return img.numpy()


def decode_latent_to_video(vae_decoder, latent_seq, nf, h, w, in_channels=128,
                            device="cuda:1"):
    """Decode latent [B, F*H*W, C] → pixel frames via VAE decoder.

    Returns list of numpy images [H_pixel, W_pixel, 3] in [0, 255].
    """
    if vae_decoder is None:
        return None

    B, S, C = latent_seq.shape
    # Reshape: [B, F*H*W, C] → [B, C, F, H, W]
    x = latent_seq[0:1].detach().float()
    x = x.view(1, nf, h, w, C).permute(0, 4, 1, 2, 3)  # [1, C, F, H, W]
    x = x.to(device=device, dtype=torch.bfloat16)

    vae_decoder.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        pixels = vae_decoder(x)  # [1, 3, F_out, H_out, W_out]
    vae_decoder.to("cpu")
    torch.cuda.empty_cache()

    # Convert to numpy frames
    pixels = pixels[0].float().cpu().clamp(-1, 1)  # [3, F, H, W]
    pixels = (pixels + 1) / 2 * 255  # [0, 255]
    frames = []
    for f in range(pixels.shape[1]):
        frame = pixels[:, f].permute(1, 2, 0).numpy().astype(np.uint8)  # [H, W, 3]
        frames.append(frame)
    return frames


def log_reconstructions(step, teacher_out, student_outs, noisy, nf, h, w,
                        vae_decoder=None, vae_device="cuda:1"):
    """Log teacher vs student reconstructions to wandb.

    Args:
        student_outs: dict mapping scale → student output tensor
    """
    if not HAS_WANDB or wandb.run is None:
        return

    log_dict = {}

    # --- Latent-space visualizations (cheap, every log step) ---
    # Noisy input
    noisy_img = latent_to_images(noisy, nf, h, w, channels=(0, 1, 2))
    log_dict["latent/noisy_input"] = wandb.Image(noisy_img, caption="Noisy input (ch 0-2)")

    # Teacher
    teacher_img = latent_to_images(teacher_out, nf, h, w, channels=(0, 1, 2))
    log_dict["latent/teacher"] = wandb.Image(teacher_img, caption="Teacher (scale=1)")

    # Students per scale
    for scale, student_out in student_outs.items():
        student_img = latent_to_images(student_out, nf, h, w, channels=(0, 1, 2))
        log_dict[f"latent/student_scale{scale}"] = wandb.Image(
            student_img, caption=f"Student (scale={scale})")

        # Difference map (error visualization)
        diff = (teacher_out - student_out)[0].detach().float().cpu()
        diff = diff.view(nf, h, w, -1)[0]  # first frame [H, W, C]
        diff_mag = diff.norm(dim=-1)  # [H, W] magnitude
        lo, hi = diff_mag.min(), diff_mag.max()
        if hi - lo > 1e-6:
            diff_mag = (diff_mag - lo) / (hi - lo)
        log_dict[f"latent/error_scale{scale}"] = wandb.Image(
            diff_mag.numpy(), caption=f"Error magnitude (scale={scale})")

    # --- VAE pixel-space reconstructions (expensive, less frequent) ---
    if vae_decoder is not None:
        try:
            teacher_frames = decode_latent_to_video(
                vae_decoder, teacher_out, nf, h, w, device=vae_device)
            if teacher_frames:
                log_dict["pixel/teacher"] = wandb.Image(
                    teacher_frames[0], caption="Teacher frame 0")

            for scale, student_out in student_outs.items():
                student_frames = decode_latent_to_video(
                    vae_decoder, student_out, nf, h, w, device=vae_device)
                if student_frames:
                    log_dict[f"pixel/student_scale{scale}"] = wandb.Image(
                        student_frames[0], caption=f"Student scale={scale} frame 0")
        except Exception as e:
            print(f"  WARNING: VAE decode failed: {e}")

    wandb.log(log_dict, step=step)


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
    """Load pre-encoded latents and text embeddings for distillation training.

    Latents are loaded eagerly (small: ~1MB each).
    Conditions are loaded lazily with deduplication — synthetic data cycles
    conditions from N_cond originals, so we cache unique ones (~16MB each).
    """
    latent_dir = Path(data_root) / "latents"
    cond_dir = Path(data_root) / "conditions_final"
    if not latent_dir.exists():
        raise FileNotFoundError(f"No latents directory at {latent_dir}")

    # Phase 1: Load latents eagerly (they're small, ~4GB for 5000 samples)
    samples = []
    skipped = 0
    latent_files = sorted(latent_dir.glob("*.pt"))[:max_samples]
    print(f"Loading {len(latent_files)} latent files from {latent_dir}...")
    for i, pt_file in enumerate(latent_files):
        try:
            data = torch.load(pt_file, weights_only=False, map_location="cpu")
            latents = data["latents"]  # [C, F, H, W]

            # Extract dims — handle both tensor and int formats
            nf = data.get("num_frames", latents.shape[1])
            h = data.get("height", latents.shape[2])
            w = data.get("width", latents.shape[3])
            nf = nf.item() if isinstance(nf, torch.Tensor) else int(nf)
            h = h.item() if isinstance(h, torch.Tensor) else int(h)
            w = w.item() if isinstance(w, torch.Tensor) else int(w)

            samples.append({
                "latents": latents, "num_frames": nf, "height": h, "width": w,
                "text_embeds": None, "text_mask": None,
                "cond_file": cond_dir / pt_file.name,
            })
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skipped corrupt file: {pt_file.name}: {e}")
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i+1}/{len(latent_files)} latents...")

    # Phase 2: Load conditions with deduplication cache
    # Conditions cycle from N_cond originals — detect by file content hash or just
    # cache by file size+name pattern. For simplicity, cache all unique conditions.
    cond_cache = {}  # maps cond file path -> (text_embeds, text_mask)
    cond_loaded = 0
    cond_files = sorted(cond_dir.glob("*.pt")) if cond_dir.exists() else []
    n_cond_files = len(cond_files)

    # Detect cycle length: if conditions are cycled copies, loading only unique ones
    # saves massive RAM (128 * 16MB = 2GB vs 5000 * 16MB = 74GB)
    # Strategy: load first condition, then check if file at index N_cond matches
    cycle_len = n_cond_files  # default: all unique
    if n_cond_files > 128:
        # Try common cycle lengths
        for test_cycle in [128, 256, 512]:
            if test_cycle < n_cond_files:
                try:
                    c0 = torch.load(cond_files[0], weights_only=False, map_location="cpu")
                    cn = torch.load(cond_files[test_cycle], weights_only=False, map_location="cpu")
                    if torch.equal(c0["video_prompt_embeds"], cn["video_prompt_embeds"]):
                        cycle_len = test_cycle
                        print(f"  Detected condition cycle length: {cycle_len}")
                        break
                except Exception:
                    pass

    # Load only unique conditions
    print(f"Loading {min(cycle_len, n_cond_files)} unique conditions (of {n_cond_files} total)...")
    for i in range(min(cycle_len, n_cond_files)):
        try:
            cond = torch.load(cond_files[i], weights_only=False, map_location="cpu")
            cond_cache[i] = (cond["video_prompt_embeds"], cond["prompt_attention_mask"])
            cond_loaded += 1
        except Exception:
            pass

    # Assign conditions to samples (cycling through cache)
    for i, s in enumerate(samples):
        cache_idx = i % cycle_len if cycle_len <= n_cond_files else i
        if cache_idx in cond_cache:
            s["text_embeds"], s["text_mask"] = cond_cache[cache_idx]
        # Remove cond_file reference (not needed anymore)
        s.pop("cond_file", None)

    print(f"Loaded {len(samples)} training samples from {latent_dir}"
          + (f" (skipped {skipped} corrupt)" if skipped else "")
          + f", {cond_loaded} unique conditions cached")
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
def get_teacher_output(base_model, latent, positions, dummy_ctx, dummy_mask,
                       sigma, device, use_lora=False):
    """Teacher: base model forward at native resolution (no_grad).

    When use_lora=True, temporarily disables LoRA adapters so the teacher
    produces ground-truth base model outputs (without any adaptation).
    """
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

    # Disable LoRA for teacher pass — we want pure base model output
    if use_lora:
        base_model.disable_adapter_layers()

    with torch.no_grad():
        output, _ = base_model(video=modality, audio=None, perturbations=perturbations)

    # Re-enable LoRA for student pass
    if use_lora:
        base_model.enable_adapter_layers()

    return output, noisy  # Return noisy latent so student uses same noise


def get_student_output(base_model, adapter, noisy_latent, positions,
                       dummy_ctx, dummy_mask, sigma, scale,
                       nf, h, w, device):
    """Student: merge → base model (with LoRA) → unmerge at coarse resolution.

    When base_model is PEFT-wrapped, LoRA adapters are active during this pass,
    allowing the transformer to adapt attention/FF for merged tokens.
    PEFT's __getattr__ delegates attribute access (patchify_proj, norm_out, etc.)
    to the underlying LTXModel transparently.
    """
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
    # Works through PEFT wrapper via __getattr__ → base_model.model.patchify_proj
    real_model = getattr(base_model, 'base_model', base_model)
    real_model = getattr(real_model, 'model', real_model)
    target_dtype = real_model.patchify_proj.weight.dtype
    if dummy_mod.latent.dtype != target_dtype:
        dummy_mod = Modality(
            enabled=dummy_mod.enabled, latent=dummy_mod.latent.to(target_dtype),
            timesteps=dummy_mod.timesteps, positions=dummy_mod.positions,
            context=dummy_mod.context, context_mask=dummy_mod.context_mask,
        )

    video_args = real_model.video_args_preprocessor.prepare(dummy_mod)

    # 4. Swap in our DDiT-projected tokens
    video_args = replace(video_args, x=projected.to(video_args.x.dtype))

    perturbations = BatchedPerturbationConfig.empty(B)

    # 5. Run through transformer blocks with gradient checkpointing
    # Temporarily set training=True to enable gradient checkpointing
    # (checkpointing is gated on `self.training` in _process_transformer_blocks)
    # With PEFT: train() propagates to all submodules including LoRA layers
    was_training = base_model.training
    base_model.train()
    video_out, _ = real_model._process_transformer_blocks(
        video=video_args, audio=None, perturbations=perturbations,
    )
    if not was_training:
        base_model.eval()

    # 6. Output processing: norm + scale_shift + proj_out
    x = video_out.x
    emb_ts = video_out.embedded_timestep
    scale_shift = real_model.scale_shift_table
    shift, scale_val = (
        scale_shift[None, None].to(device=x.device, dtype=x.dtype) + emb_ts[:, :, None]
    ).unbind(dim=2)
    x = real_model.norm_out(x)
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


def train_phase2(base_model, adapter, samples, args, use_lora=False,
                  vae_decoder=None):
    """Distillation training through frozen base model.

    When use_lora=True, the base_model is a PeftModel with LoRA adapters.
    LoRA is disabled for teacher pass (ground truth) and enabled for student
    pass (adaptation). Both adapter and LoRA params are optimized jointly.
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = args.device
    dtype = torch.bfloat16
    scales = tuple(int(s) for s in args.scales.split(","))
    log_interval = getattr(args, 'log_interval', 50)
    vae_interval = getattr(args, 'vae_interval', 200)
    vae_device = "cuda:1" if torch.cuda.device_count() > 1 else device

    # Collect all trainable parameters: adapter + LoRA
    trainable_params = list(adapter.parameters())
    if use_lora:
        lora_params = get_lora_params(base_model)
        trainable_params.extend(lora_params)
        print(f"  Optimizer: {sum(p.numel() for p in adapter.parameters()):,} adapter + "
              f"{sum(p.numel() for p in lora_params):,} LoRA = "
              f"{sum(p.numel() for p in trainable_params):,} total trainable")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_tag = " + LoRA" if use_lora else ""
    print(f"\n=== Phase 2: Distillation training{lora_tag} ({args.steps} steps) ===")
    print(f"  Scales: {scales}, LR: {args.lr}")
    vram = torch.cuda.memory_allocated(device) / 1e9
    print(f"  VRAM before training: {vram:.1f}GB")
    print(f"  Logging: latent images every {log_interval} steps, "
          f"VAE decode every {vae_interval} steps")

    # Init wandb
    if HAS_WANDB and not getattr(args, 'no_wandb', False):
        wandb.init(
            project="ddit-distillation",
            name=f"p2_lora-{args.lora_target}_r{args.lora_rank}_lr{args.lr}",
            config={
                "phase": 2,
                "scales": list(scales),
                "lr": args.lr,
                "steps": args.steps,
                "use_lora": use_lora,
                "lora_target": getattr(args, 'lora_target', 'both'),
                "lora_rank": args.lora_rank,
                "adapter_params": sum(p.numel() for p in adapter.parameters()),
                "lora_params": sum(p.numel() for p in trainable_params) - sum(p.numel() for p in adapter.parameters()),
                "total_trainable": sum(p.numel() for p in trainable_params),
                "data_samples": len(samples),
                "device": str(device),
            },
        )
        print(f"  wandb: {wandb.run.url}")

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

            # Teacher: full-res forward (no_grad, LoRA disabled for base model output)
            teacher_out, noisy = get_teacher_output(
                base_model, latent, positions, text_ctx, text_mask, sigma, device,
                use_lora=use_lora,
            )
            # Detach teacher to free its computation graph
            teacher_out = teacher_out.detach()
            del latent  # Free input latent
            torch.cuda.empty_cache()

            # Accumulate gradients across all scales, then do one optimizer step.
            # This is critical when using LoRA: shared LoRA params would get
            # conflicting updates if we stepped per-scale.
            step_losses = {}
            valid_scales = [s for s in scales if h % s == 0 and w % s == 0]
            optimizer.zero_grad()

            # Should we log reconstructions this step?
            should_log_latent = (step + 1) % log_interval == 0
            should_log_vae = vae_decoder is not None and (step + 1) % vae_interval == 0
            student_outs_for_vis = {}  # Capture for visualization

            for scale in valid_scales:
                # Student: coarse-res forward (gradients through adapter + LoRA)
                student_out = get_student_output(
                    base_model, adapter, noisy, positions,
                    text_ctx, text_mask, sigma, scale, nf, h, w, device,
                )

                # Normalize loss by number of scales for balanced gradients
                loss = F.mse_loss(student_out, teacher_out) / len(valid_scales)

                # Backward accumulates gradients; frees activations per-scale
                loss.backward()
                step_losses[scale] = loss.item() * len(valid_scales)  # unnormalized

                # Keep detached copy for visualization before deleting
                if should_log_latent or should_log_vae:
                    student_outs_for_vis[scale] = student_out.detach()

                del student_out, loss
                torch.cuda.empty_cache()

            if step_losses:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                avg_loss = sum(step_losses.values()) / len(step_losses)
                losses.append(avg_loss)

                # wandb scalar logging
                if HAS_WANDB and wandb.run is not None:
                    log_scalars = {
                        "train/loss": avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/sigma": sigma,
                    }
                    for s, l in step_losses.items():
                        log_scalars[f"train/loss_scale{s}"] = l
                    vram_now = torch.cuda.max_memory_allocated(device) / 1e9
                    log_scalars["train/peak_vram_gb"] = vram_now
                    wandb.log(log_scalars, step=step + 1)

            # Log reconstruction images
            if (should_log_latent or should_log_vae) and student_outs_for_vis:
                use_vae = vae_decoder if should_log_vae else None
                log_reconstructions(
                    step + 1, teacher_out, student_outs_for_vis, noisy,
                    nf, h, w, vae_decoder=use_vae, vae_device=vae_device,
                )
                del student_outs_for_vis
            else:
                student_outs_for_vis.clear()

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
                # Save adapter
                path = out_dir / f"ddit_adapter_p2_step{step:05d}.safetensors"
                save_file(dict(adapter.state_dict()), str(path))
                print(f"  Saved adapter: {path}")
                # Save LoRA
                if use_lora:
                    lora_path = out_dir / f"ddit_lora_p2_step{step:05d}.safetensors"
                    save_lora_weights(base_model, lora_path)

    # Final save
    path = out_dir / "ddit_adapter_final.safetensors"
    save_file(dict(adapter.state_dict()), str(path))
    print(f"\nPhase 2 complete! {path}")
    if use_lora:
        lora_path = out_dir / "ddit_lora_final.safetensors"
        save_lora_weights(base_model, lora_path)
    if losses:
        print(f"  Final loss: {losses[-1]:.6f}")

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()

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

        # Initialize adapter from base model BEFORE applying LoRA
        adapter = create_ddit_adapter(
            scales=scales, lora_rank=args.lora_rank, device=device,
            base_model=base_model,
        )

        # Apply LoRA to transformer blocks for adaptation capacity
        use_lora = args.use_lora
        if use_lora:
            base_model = setup_lora(
                base_model, lora_rank=args.lora_rank,
                lora_alpha=args.lora_rank,  # scaling = alpha/rank = 1.0
                target=args.lora_target,
            )

        # Load phase 1 weights if resuming
        if args.resume:
            print(f"Loading adapter weights from {args.resume}")
            state = load_file(args.resume)
            adapter.load_state_dict(state, strict=False)
            adapter = adapter.to(device=device)

        # Load LoRA weights if resuming with LoRA
        if use_lora and args.resume_lora:
            load_lora_weights(base_model, args.resume_lora)

        # Load VAE decoder for pixel-space reconstructions
        vae_decoder = None
        if not getattr(args, 'no_wandb', False) and not getattr(args, 'no_vae', False):
            vae_decoder = load_vae_decoder(args.model_path)

        adapter = train_phase2(base_model, adapter, samples, args,
                                use_lora=use_lora, vae_decoder=vae_decoder)

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

        use_lora = args.use_lora
        if use_lora:
            base_model = setup_lora(
                base_model, lora_rank=args.lora_rank,
                lora_alpha=args.lora_rank, target=args.lora_target,
            )

        p2_steps = args.steps - p1_steps
        args_p2 = argparse.Namespace(**vars(args))
        args_p2.steps = p2_steps
        adapter = train_phase2(base_model, adapter, samples, args_p2, use_lora=use_lora)

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
            "use_lora": getattr(args, 'use_lora', False),
            "lora_target": getattr(args, 'lora_target', 'both'),
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
    parser.add_argument("--resume_lora", type=str, default=None, help="Resume from LoRA checkpoint")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Apply LoRA to transformer blocks (default: True for phase 2)")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (adapter-only training)")
    parser.add_argument("--lora_target", type=str, default="both",
                        choices=["ff", "attn", "both"],
                        help="LoRA target: ff (paper), attn, or both (default: both)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no_vae", action="store_true", help="Disable VAE decode for reconstructions")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log latent reconstruction images every N steps")
    parser.add_argument("--vae_interval", type=int, default=200,
                        help="Log VAE pixel reconstructions every N steps")
    args = parser.parse_args()

    # Handle --no_lora flag
    if args.no_lora:
        args.use_lora = False

    train(args)


if __name__ == "__main__":
    main()
