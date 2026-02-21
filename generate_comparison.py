#!/usr/bin/env python3
"""
Generate actual video frames: Baseline vs SCD comparison.

Produces side-by-side output so you can visually compare:
  - Baseline: Full 48-layer LTX-2 transformer, 20 denoising steps
  - SCD: 32-layer encoder (once) + 16-layer decoder (20 steps)

Uses presaved text embeddings from previous isometric runs (no Gemma needed).
Both runs use identical noise and embeddings for fair comparison.

Output: PNG frames in output/baseline/ and output/scd/
"""

import time
import gc
import torch
import numpy as np
from pathlib import Path
from dataclasses import replace
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
CHECKPOINT = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
CACHED_EMBEDDINGS = "/media/2TB/omnitransfer/inference/i2v_test/cached_embedding.pt"
DEVICE = "cuda:0"  # RTX 5090 (32GB) — PyTorch maps this as cuda:0
DTYPE = torch.bfloat16
SEED = 42
NUM_STEPS = 20  # Denoising steps (fewer = faster, 20 is reasonable)

# Video dimensions: small to fit in memory
PIXEL_H, PIXEL_W = 512, 512
NUM_PIXEL_FRAMES = 9  # 9 pixel frames → 2 latent frames

# Derived latent dimensions
LATENT_H = PIXEL_H // 32   # 16
LATENT_W = PIXEL_W // 32   # 16
LATENT_F = (NUM_PIXEL_FRAMES - 1) // 8 + 1  # 2
IN_CHANNELS = 128
TOKENS_PER_FRAME = LATENT_H * LATENT_W  # 256
SEQ_LEN = LATENT_F * TOKENS_PER_FRAME   # 512

OUTPUT_DIR = Path("/home/johndpope/Documents/GitHub/sparse-causal-diffusion/output")

print("=" * 70)
print("GENERATE COMPARISON: BASELINE vs SCD")
print("=" * 70)
print(f"  Resolution: {PIXEL_H}x{PIXEL_W} @ {NUM_PIXEL_FRAMES} frames")
print(f"  Latent: {LATENT_H}x{LATENT_W} x {LATENT_F} frames")
print(f"  Tokens: {SEQ_LEN} (video)")
print(f"  Steps: {NUM_STEPS}")
print(f"  Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
# Step 1: Load transformer
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LOADING TRANSFORMER")
print("=" * 70)

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel, build_frame_causal_mask, shift_encoder_features,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.utils import to_denoised

t0 = time.time()
state_dict = load_file(CHECKPOINT, device="cpu")
PREFIX = "model.diffusion_model."
renamed = {k[len(PREFIX):]: v for k, v in state_dict.items() if k.startswith(PREFIX)}
del state_dict
gc.collect()

model = LTXModel(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=32, attention_head_dim=128,
    in_channels=128, out_channels=128, num_layers=48,
    cross_attention_dim=4096, caption_channels=3840,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    timestep_scale_multiplier=1000, use_middle_indices_grid=True,
    apply_gated_attention=False,
)
model.load_state_dict(renamed, strict=False)
model = model.to(device=DEVICE, dtype=DTYPE).eval()
del renamed
gc.collect(); torch.cuda.empty_cache()
print(f"  Model loaded in {time.time()-t0:.1f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# ─────────────────────────────────────────────────────────────
# Step 2: Create inputs (identical for both runs)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("CREATING INPUTS")
print("=" * 70)

from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.types import VideoLatentShape, VideoPixelShape, SpatioTemporalScaleFactors

B = 1
FPS = 24.0

# Build proper positional embeddings (not random!)
pixel_shape = VideoPixelShape(batch=B, frames=NUM_PIXEL_FRAMES, height=PIXEL_H, width=PIXEL_W, fps=FPS)
scale_factors = SpatioTemporalScaleFactors.default()  # (8, 32, 32)
latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=IN_CHANNELS, scale_factors=scale_factors)
patchifier = VideoLatentPatchifier(patch_size=1)

# Get proper positions: [B, 3, seq_len, 2]
latent_coords = patchifier.get_patch_grid_bounds(output_shape=latent_shape, device=DEVICE)
positions = get_pixel_coords(latent_coords=latent_coords, scale_factors=scale_factors, causal_fix=True).float()
positions[:, 0, ...] = positions[:, 0, ...] / FPS  # Normalize time to seconds
positions = positions.to(DTYPE)
print(f"  positions: {positions.shape}")

# Load presaved text embeddings (from previous isometric runs — no Gemma needed)
print(f"  Loading presaved embeddings from {CACHED_EMBEDDINGS}...")
cached = torch.load(CACHED_EMBEDDINGS, map_location="cpu", weights_only=False)
caption = cached["video_context_positive"].to(device=DEVICE, dtype=DTYPE)  # [1, 1024, 3840]
caption_mask = torch.ones(B, caption.shape[1], device=DEVICE, dtype=DTYPE)
TEXT_SEQ = caption.shape[1]
prompt_text = cached.get("prompt", "N/A")
print(f"  Prompt: {prompt_text[:100]}...")
print(f"  Caption embedding: {caption.shape}")
del cached

# Create initial noise (same for both baseline and SCD)
generator = torch.Generator(device=DEVICE).manual_seed(SEED)
initial_noise = torch.randn(B, IN_CHANNELS, LATENT_F, LATENT_H, LATENT_W,
                            device=DEVICE, dtype=DTYPE, generator=generator)
# Patchify: [B, C, F, H, W] → [B, seq_len, C]
noise_patchified = initial_noise.permute(0, 2, 3, 4, 1).reshape(B, SEQ_LEN, IN_CHANNELS)
print(f"  noise (raw): {initial_noise.shape}")
print(f"  noise (patchified): {noise_patchified.shape}")
print(f"  caption: {caption.shape}")

# Denoise mask: all ones (denoise everything)
denoise_mask = torch.ones(B, SEQ_LEN, device=DEVICE, dtype=torch.float32)

perturbations = BatchedPerturbationConfig.empty(B)

# Sigma schedule (simple linear)
from ltx_core.components.schedulers import LTX2Scheduler
scheduler = LTX2Scheduler()
sigmas = scheduler.execute(
    steps=NUM_STEPS,
    latent=initial_noise,
).to(device=DEVICE, dtype=torch.float32)
print(f"  sigmas: {sigmas.shape} [{sigmas[0]:.4f} → {sigmas[-1]:.4f}]")

# ─────────────────────────────────────────────────────────────
# Step 3: BASELINE denoising (48 layers × N steps)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"BASELINE DENOISING ({NUM_STEPS} steps, 48 layers)")
print("=" * 70)

# Start from pure noise
x_baseline = noise_patchified.clone()

t0 = time.time()
with torch.no_grad():
    for step_idx in tqdm(range(NUM_STEPS), desc="Baseline"):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        # Build modality with current noisy latent
        timesteps = denoise_mask * sigma
        video_mod = Modality(
            enabled=True, latent=x_baseline,
            timesteps=timesteps, positions=positions,
            context=caption, context_mask=caption_mask,
        )

        # Forward pass → velocity prediction
        velocity, _ = model(video=video_mod, audio=None, perturbations=perturbations)

        # Velocity → denoised (x0 = x_noisy - v * sigma)
        denoised = to_denoised(x_baseline, velocity, sigma)

        # Post-process with mask
        denoised = (denoised * denoise_mask.unsqueeze(-1).float()
                    + noise_patchified.float() * (1 - denoise_mask.unsqueeze(-1).float())).to(DTYPE)

        # Euler step: x_{t+1} = x_t + velocity * dt
        dt = sigma_next - sigma
        velocity_f32 = (x_baseline.float() - denoised.float()) / sigma.float()
        x_baseline = (x_baseline.float() + velocity_f32 * dt.float()).to(DTYPE)

baseline_time = time.time() - t0
print(f"  Done in {baseline_time:.1f}s ({baseline_time/NUM_STEPS*1000:.0f}ms/step)")
print(f"  Output stats: mean={x_baseline.float().mean():.4f}, std={x_baseline.float().std():.4f}")

# Save baseline latent for later decoding
baseline_latent_patchified = x_baseline.clone()

# ─────────────────────────────────────────────────────────────
# Step 4: SCD denoising (encoder once + decoder × N steps)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SCD DENOISING ({NUM_STEPS} steps: 32L encoder once + 16L decoder × {NUM_STEPS})")
print("=" * 70)

scd = LTXSCDModel(base_model=model, encoder_layers=32, decoder_input_combine="token_concat")
print(f"  Encoder: {len(scd.encoder_blocks)} layers, Decoder: {len(scd.decoder_blocks)} layers")

# --- Encoder pass (runs ONCE with timestep=0) ---
print("  Running encoder (once)...")
t_enc_start = time.time()
encoder_modality = Modality(
    enabled=True, latent=noise_patchified,  # Use same starting noise
    timesteps=torch.zeros(B, SEQ_LEN, device=DEVICE, dtype=DTYPE),  # t=0 for encoder
    positions=positions, context=caption, context_mask=caption_mask,
)

with torch.no_grad():
    enc_args, _ = scd.forward_encoder(
        video=encoder_modality, audio=None,
        perturbations=perturbations, tokens_per_frame=TOKENS_PER_FRAME,
    )
torch.cuda.synchronize()
t_enc = time.time() - t_enc_start
print(f"  Encoder: {t_enc*1000:.0f}ms")

# Shift encoder features (frame t-1 → frame t)
shifted_enc = shift_encoder_features(enc_args.x, TOKENS_PER_FRAME, LATENT_F)
print(f"  Shifted encoder features: {shifted_enc.shape}")

# --- Decoder loop (runs N times) ---
x_scd = noise_patchified.clone()

t_dec_start = time.time()
with torch.no_grad():
    for step_idx in tqdm(range(NUM_STEPS), desc="SCD Decoder"):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        timesteps = denoise_mask * sigma
        video_mod = Modality(
            enabled=True, latent=x_scd,
            timesteps=timesteps, positions=positions,
            context=caption, context_mask=caption_mask,
        )

        # SCD decoder forward (with encoder features)
        velocity, _ = scd.forward_decoder(
            video=video_mod, encoder_features=shifted_enc,
            audio=None, perturbations=perturbations,
        )

        # Velocity → denoised
        denoised = to_denoised(x_scd, velocity, sigma)

        # Post-process with mask
        denoised = (denoised * denoise_mask.unsqueeze(-1).float()
                    + noise_patchified.float() * (1 - denoise_mask.unsqueeze(-1).float())).to(DTYPE)

        # Euler step
        dt = sigma_next - sigma
        velocity_f32 = (x_scd.float() - denoised.float()) / sigma.float()
        x_scd = (x_scd.float() + velocity_f32 * dt.float()).to(DTYPE)

t_dec = time.time() - t_dec_start
scd_total_time = t_enc + t_dec
print(f"  Decoder: {t_dec:.1f}s ({t_dec/NUM_STEPS*1000:.0f}ms/step)")
print(f"  Total SCD: {scd_total_time:.1f}s (encoder {t_enc:.1f}s + decoder {t_dec:.1f}s)")
print(f"  Output stats: mean={x_scd.float().mean():.4f}, std={x_scd.float().std():.4f}")

scd_latent_patchified = x_scd.clone()

# ─────────────────────────────────────────────────────────────
# Step 5: Compare denoised latents
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LATENT COMPARISON")
print("=" * 70)

with torch.no_grad():
    b_flat = baseline_latent_patchified.float().flatten()
    s_flat = scd_latent_patchified.float().flatten()
    mse = (b_flat - s_flat).pow(2).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        b_flat.unsqueeze(0), s_flat.unsqueeze(0)
    ).item()
    print(f"  MSE: {mse:.6f}")
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Baseline: mean={b_flat.mean():.4f}, std={b_flat.std():.4f}")
    print(f"  SCD:      mean={s_flat.mean():.4f}, std={s_flat.std():.4f}")

print(f"\n  Timing:")
print(f"    Baseline: {baseline_time:.1f}s total")
print(f"    SCD:      {scd_total_time:.1f}s total ({t_enc:.1f}s enc + {t_dec:.1f}s dec)")
print(f"    Speedup:  {baseline_time/scd_total_time:.2f}x")

# ─────────────────────────────────────────────────────────────
# Step 6: Free transformer, load VAE decoder
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LOADING VAE DECODER")
print("=" * 70)

# Unpatchify latents: [B, seq_len, C] → [B, C, F, H, W]
baseline_latent_5d = baseline_latent_patchified.reshape(B, LATENT_F, LATENT_H, LATENT_W, IN_CHANNELS)
baseline_latent_5d = baseline_latent_5d.permute(0, 4, 1, 2, 3).contiguous()
scd_latent_5d = scd_latent_patchified.reshape(B, LATENT_F, LATENT_H, LATENT_W, IN_CHANNELS)
scd_latent_5d = scd_latent_5d.permute(0, 4, 1, 2, 3).contiguous()

print(f"  Baseline latent: {baseline_latent_5d.shape}")
print(f"  SCD latent: {scd_latent_5d.shape}")

# Save latents to CPU before freeing GPU
baseline_latent_cpu = baseline_latent_5d.cpu()
scd_latent_cpu = scd_latent_5d.cpu()

# Also save latents to disk so we can skip denoising next time
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.save({"baseline": baseline_latent_cpu, "scd": scd_latent_cpu,
            "cos_sim": cos_sim, "baseline_time": baseline_time,
            "scd_total_time": scd_total_time, "t_enc": t_enc, "t_dec": t_dec},
           OUTPUT_DIR / "latents.pt")
print(f"  Latents saved to {OUTPUT_DIR / 'latents.pt'}")

# Free transformer from GPU
del model, scd, enc_args, shifted_enc
del x_baseline, x_scd, baseline_latent_patchified, scd_latent_patchified
del baseline_latent_5d, scd_latent_5d
del noise_patchified, initial_noise, positions, caption, caption_mask
gc.collect(); torch.cuda.empty_cache()

print("  Transformer freed from GPU")

# Load VAE decoder via ModelLedger
from ltx_pipelines.utils import ModelLedger

ledger = ModelLedger(
    dtype=DTYPE,
    device=torch.device(DEVICE),
    checkpoint_path=CHECKPOINT,
)
vae_decoder = ledger.video_decoder()
print(f"  VAE decoder loaded: {sum(p.numel() for p in vae_decoder.parameters())/1e6:.0f}M params")

# ─────────────────────────────────────────────────────────────
# Step 7: Decode to pixels
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("DECODING TO PIXELS")
print("=" * 70)

from ltx_core.model.video_vae import decode_video

# Decode baseline
print("  Decoding baseline...")
baseline_latent_gpu = baseline_latent_cpu.to(device=DEVICE, dtype=DTYPE)
gen_baseline = torch.Generator(device=DEVICE).manual_seed(SEED)
baseline_frames = []
for chunk in decode_video(baseline_latent_gpu[0], vae_decoder, generator=gen_baseline):
    baseline_frames.append(chunk.cpu())
baseline_frames = torch.cat(baseline_frames, dim=0)
print(f"  Baseline frames: {baseline_frames.shape}")
del baseline_latent_gpu
torch.cuda.empty_cache()

# Decode SCD
print("  Decoding SCD...")
scd_latent_gpu = scd_latent_cpu.to(device=DEVICE, dtype=DTYPE)
gen_scd = torch.Generator(device=DEVICE).manual_seed(SEED)
scd_frames = []
for chunk in decode_video(scd_latent_gpu[0], vae_decoder, generator=gen_scd):
    scd_frames.append(chunk.cpu())
scd_frames = torch.cat(scd_frames, dim=0)
print(f"  SCD frames: {scd_frames.shape}")
del scd_latent_gpu, vae_decoder, ledger
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────
# Step 8: Save frames
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SAVING FRAMES")
print("=" * 70)

# Create output directories
(OUTPUT_DIR / "baseline").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "scd").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "comparison").mkdir(parents=True, exist_ok=True)

num_frames = baseline_frames.shape[0]
print(f"  Saving {num_frames} frames...")

for i in range(num_frames):
    # Individual frames
    bl_img = Image.fromarray(baseline_frames[i].numpy(), mode='RGB')
    scd_img = Image.fromarray(scd_frames[i].numpy(), mode='RGB')
    bl_img.save(OUTPUT_DIR / "baseline" / f"frame_{i:03d}.png")
    scd_img.save(OUTPUT_DIR / "scd" / f"frame_{i:03d}.png")

    # Side-by-side comparison
    w, h = bl_img.size
    comparison = Image.new('RGB', (w * 2 + 10, h + 30), (40, 40, 40))
    comparison.paste(bl_img, (0, 30))
    comparison.paste(scd_img, (w + 10, 30))

    # Add labels
    from PIL import ImageDraw
    draw = ImageDraw.Draw(comparison)
    draw.text((w//2 - 30, 5), "BASELINE", fill=(255, 255, 255))
    draw.text((w + 10 + w//2 - 15, 5), "SCD", fill=(255, 255, 255))
    comparison.save(OUTPUT_DIR / "comparison" / f"frame_{i:03d}.png")

print(f"  Saved to {OUTPUT_DIR}/")
print(f"    baseline/   — {num_frames} frames")
print(f"    scd/        — {num_frames} frames")
print(f"    comparison/ — {num_frames} side-by-side frames")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"""
  Resolution:      {PIXEL_H}x{PIXEL_W} @ {NUM_PIXEL_FRAMES} frames
  Denoising steps: {NUM_STEPS}

  Baseline time:   {baseline_time:.1f}s ({NUM_STEPS} steps × 48 layers)
  SCD time:        {scd_total_time:.1f}s (1 encoder + {NUM_STEPS} decoder steps)
  Speedup:         {baseline_time/scd_total_time:.2f}x

  Latent cosine similarity: {cos_sim:.4f}
  (1.0 = identical, 0.0 = uncorrelated)

  Using presaved text embeddings from isometric runs.
  Prompt: {prompt_text[:80]}...
  Both runs use identical noise + embeddings for fair comparison.
""")
print("=" * 70)
