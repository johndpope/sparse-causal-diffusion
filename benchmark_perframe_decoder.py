"""
SCD Per-Frame Decoder Benchmark — Real 19B Weights

Tests the ACTUAL inference mode:
  - Encoder: runs ONCE on all frames (causal mask)
  - Decoder: runs N times on SINGLE FRAME tokens (256 tokens, not all 512)
  - Compare vs baseline: 48 layers × N steps on all frames

This is where the ~3x speedup comes from.
"""

import time
import torch
import gc
from safetensors.torch import load_file

CHECKPOINT = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# 512x512 @ 17 frames = 3 latent frames (more frames → better amortization)
PIXEL_H, PIXEL_W = 512, 512
NUM_PIXEL_FRAMES = 17
LATENT_H = PIXEL_H // 32   # 16
LATENT_W = PIXEL_W // 32   # 16
LATENT_F = (NUM_PIXEL_FRAMES - 1) // 8 + 1  # 3 latent frames
IN_CHANNELS = 128
TOKENS_PER_FRAME = LATENT_H * LATENT_W  # 256
TOTAL_SEQ = LATENT_F * TOKENS_PER_FRAME  # 768
TEXT_SEQ = 64

print("=" * 70)
print("SCD PER-FRAME DECODER BENCHMARK — LTX-2 19B")
print("=" * 70)
print(f"  Resolution: {PIXEL_H}x{PIXEL_W} @ {NUM_PIXEL_FRAMES} pixel frames")
print(f"  Latent: {LATENT_H}x{LATENT_W} × {LATENT_F} frames")
print(f"  Tokens per frame: {TOKENS_PER_FRAME}")
print(f"  Total sequence: {TOTAL_SEQ}")

# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LOADING MODEL")
print("=" * 70)

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel, build_frame_causal_mask, shift_encoder_features,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig

state_dict = load_file(CHECKPOINT, device="cpu")
PREFIX = "model.diffusion_model."
renamed = {k[len(PREFIX):]: v for k, v in state_dict.items() if k.startswith(PREFIX)}
del state_dict

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
print(f"  Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

scd = LTXSCDModel(base_model=model, encoder_layers=32, decoder_input_combine="token_concat")
print(f"  SCD: {len(scd.encoder_blocks)} encoder + {len(scd.decoder_blocks)} decoder")

# ─────────────────────────────────────────────────────────────
# Create inputs
# ─────────────────────────────────────────────────────────────
torch.manual_seed(42)
B = 1
perturbations = BatchedPerturbationConfig.empty(B)

# Full-sequence inputs (all frames)
latent_all = torch.randn(B, TOTAL_SEQ, IN_CHANNELS, device=DEVICE, dtype=DTYPE)
positions_all = torch.randn(B, 3, TOTAL_SEQ, 2, device=DEVICE, dtype=DTYPE)
caption = torch.randn(B, TEXT_SEQ, 3840, device=DEVICE, dtype=DTYPE)
caption_mask = torch.ones(B, TEXT_SEQ, device=DEVICE, dtype=DTYPE)

# Per-frame inputs (single frame = 256 tokens)
# For frame index `f`, extract its slice
def make_frame_modality(frame_idx, sigma=0.5):
    start = frame_idx * TOKENS_PER_FRAME
    end = start + TOKENS_PER_FRAME
    return Modality(
        enabled=True,
        latent=latent_all[:, start:end, :],
        timesteps=torch.full((B, TOKENS_PER_FRAME), sigma, device=DEVICE, dtype=DTYPE),
        positions=positions_all[:, :, start:end, :],
        context=caption,
        context_mask=caption_mask,
    )

# Full-sequence modality for baseline
full_modality = Modality(
    enabled=True, latent=latent_all,
    timesteps=torch.full((B, TOTAL_SEQ), 0.5, device=DEVICE, dtype=DTYPE),
    positions=positions_all, context=caption, context_mask=caption_mask,
)

# Encoder modality (timestep=0, clean signal)
encoder_modality = Modality(
    enabled=True, latent=latent_all,
    timesteps=torch.zeros(B, TOTAL_SEQ, device=DEVICE, dtype=DTYPE),
    positions=positions_all, context=caption, context_mask=caption_mask,
)

print(f"  Full sequence: {TOTAL_SEQ} tokens ({LATENT_F} frames × {TOKENS_PER_FRAME} tokens)")
print(f"  Per-frame: {TOKENS_PER_FRAME} tokens")

# ─────────────────────────────────────────────────────────────
# Benchmark 1: BASELINE (48 layers × N steps × all frames)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("BASELINE: 48 layers × all frames per step")
print("=" * 70)

# Warmup
with torch.no_grad():
    _ = model(video=full_modality, audio=None, perturbations=perturbations)
torch.cuda.synchronize()

times_baseline = []
with torch.no_grad():
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        out, _ = model(video=full_modality, audio=None, perturbations=perturbations)
        torch.cuda.synchronize()
        times_baseline.append(time.time() - t0)

baseline_ms = sum(times_baseline) / len(times_baseline) * 1000
print(f"  Single step (all {TOTAL_SEQ} tokens): {baseline_ms:.1f}ms")

# ─────────────────────────────────────────────────────────────
# Benchmark 2: SCD ENCODER (runs once)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SCD ENCODER: 32 layers × all frames × ONCE")
print("=" * 70)

# Warmup
with torch.no_grad():
    enc_args, _ = scd.forward_encoder(
        video=encoder_modality, audio=None,
        perturbations=perturbations, tokens_per_frame=TOKENS_PER_FRAME,
    )
torch.cuda.synchronize()

times_encoder = []
with torch.no_grad():
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        enc_args, _ = scd.forward_encoder(
            video=encoder_modality, audio=None,
            perturbations=perturbations, tokens_per_frame=TOKENS_PER_FRAME,
        )
        torch.cuda.synchronize()
        times_encoder.append(time.time() - t0)

encoder_ms = sum(times_encoder) / len(times_encoder) * 1000
print(f"  Encoder (all {TOTAL_SEQ} tokens, causal mask): {encoder_ms:.1f}ms")

# Shift encoder features
shifted_all = shift_encoder_features(enc_args.x, TOKENS_PER_FRAME, LATENT_F)

# ─────────────────────────────────────────────────────────────
# Benchmark 3: SCD DECODER — ALL FRAMES (current implementation)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SCD DECODER (all frames): 16 layers × all frames")
print("=" * 70)

# Warmup
with torch.no_grad():
    scd_out, _ = scd.forward_decoder(
        video=full_modality, encoder_features=shifted_all,
        audio=None, perturbations=perturbations,
    )
torch.cuda.synchronize()

times_dec_all = []
with torch.no_grad():
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        scd_out, _ = scd.forward_decoder(
            video=full_modality, encoder_features=shifted_all,
            audio=None, perturbations=perturbations,
        )
        torch.cuda.synchronize()
        times_dec_all.append(time.time() - t0)

dec_all_ms = sum(times_dec_all) / len(times_dec_all) * 1000
print(f"  Decoder all-frames ({TOTAL_SEQ} tokens → {TOTAL_SEQ*2} with concat): {dec_all_ms:.1f}ms")

# ─────────────────────────────────────────────────────────────
# Benchmark 4: SCD DECODER — PER FRAME (the real speedup)
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SCD DECODER (per-frame): 16 layers × single frame")
print("=" * 70)

# Use the last frame (frame_idx = LATENT_F - 1)
target_frame = LATENT_F - 1
frame_modality = make_frame_modality(target_frame, sigma=0.5)

# Extract per-frame encoder features (256 tokens)
frame_start = target_frame * TOKENS_PER_FRAME
frame_end = frame_start + TOKENS_PER_FRAME
frame_enc_features = shifted_all[:, frame_start:frame_end, :]

# Warmup
with torch.no_grad():
    frame_out, _ = scd.forward_decoder(
        video=frame_modality, encoder_features=frame_enc_features,
        audio=None, perturbations=perturbations,
    )
torch.cuda.synchronize()

times_dec_frame = []
with torch.no_grad():
    for i in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        frame_out, _ = scd.forward_decoder(
            video=frame_modality, encoder_features=frame_enc_features,
            audio=None, perturbations=perturbations,
        )
        torch.cuda.synchronize()
        times_dec_frame.append(time.time() - t0)

dec_frame_ms = sum(times_dec_frame) / len(times_dec_frame) * 1000
print(f"  Decoder per-frame ({TOKENS_PER_FRAME} tokens → {TOKENS_PER_FRAME*2} with concat): {dec_frame_ms:.1f}ms")
print(f"  Output shape: {frame_out.shape}")

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TIMING SUMMARY")
print("=" * 70)
print(f"  Baseline step (48L × {TOTAL_SEQ} tok):     {baseline_ms:7.1f}ms")
print(f"  SCD encoder   (32L × {TOTAL_SEQ} tok):     {encoder_ms:7.1f}ms  (runs ONCE)")
print(f"  SCD decoder all-frames (16L × {TOTAL_SEQ*2} tok): {dec_all_ms:7.1f}ms  (token_concat)")
print(f"  SCD decoder per-frame  (16L × {TOKENS_PER_FRAME*2} tok):  {dec_frame_ms:7.1f}ms  (token_concat)")

print(f"\n{'='*70}")
print("SPEEDUP: N-STEP DENOISING")
print("=" * 70)
print(f"\n  Mode 1: SCD with ALL-FRAME decoder (current batch training mode)")
for N in [8, 20, 50]:
    bl = baseline_ms * N
    scd_t = encoder_ms + dec_all_ms * N
    print(f"    N={N:3d}: Baseline={bl:7.0f}ms  SCD={scd_t:7.0f}ms  Speedup={bl/scd_t:.2f}x")

print(f"\n  Mode 2: SCD with PER-FRAME decoder (autoregressive inference mode)")
for N in [8, 20, 50]:
    bl = baseline_ms * N
    # For F frames: encoder once + decoder N times per frame
    scd_t = encoder_ms + dec_frame_ms * N * LATENT_F
    print(f"    N={N:3d}: Baseline={bl:7.0f}ms  SCD={scd_t:7.0f}ms  Speedup={bl/scd_t:.2f}x")

print(f"\n  Mode 3: SCD per-frame with KV-cache encoder (amortized across frames)")
for N in [8, 20, 50]:
    bl = baseline_ms * N
    # Encoder amortized: full cost / num_frames (rough estimate for incremental frame)
    enc_amortized = encoder_ms / LATENT_F
    scd_t = enc_amortized + dec_frame_ms * N
    print(f"    N={N:3d}: Baseline={bl:7.0f}ms  SCD(1 frame)={scd_t:6.0f}ms  "
          f"Speedup={bl/scd_t:.2f}x  (per-frame cost)")

# For longer videos
print(f"\n{'='*70}")
print("PROJECTION: LONGER VIDEOS")
print("=" * 70)
for frames_label, lat_f in [("1 sec (33 frames)", 5), ("5 sec (121 frames)", 16), ("10 sec (241 frames)", 31)]:
    total_tok = lat_f * TOKENS_PER_FRAME
    # Baseline scales quadratically with attention
    # Rough: baseline_ms * (total_tok / TOTAL_SEQ)^2 (quadratic attention)
    scale_factor = (total_tok / TOTAL_SEQ) ** 2
    bl_step = baseline_ms * scale_factor
    # SCD encoder also scales quadratically (causal mask still full sequence)
    enc_cost = encoder_ms * scale_factor
    # SCD per-frame decoder: FIXED cost per frame (only 256 tokens!)
    dec_per_step_all_frames = dec_frame_ms * lat_f

    print(f"\n  {frames_label}: {lat_f} latent frames, {total_tok} total tokens")
    for N in [20, 50]:
        bl_total = bl_step * N
        scd_total = enc_cost + dec_per_step_all_frames * N
        if scd_total > 0:
            print(f"    N={N:3d}: Baseline≈{bl_total/1000:6.1f}s  SCD≈{scd_total/1000:6.1f}s  "
                  f"Speedup≈{bl_total/scd_total:.1f}x")

print(f"\n{'='*70}")
print("KEY TAKEAWAY")
print("=" * 70)
print(f"""
  The per-frame decoder ({TOKENS_PER_FRAME} tokens) costs {dec_frame_ms:.1f}ms
  vs all-frame decoder ({TOTAL_SEQ} tokens → {TOTAL_SEQ*2} concat) at {dec_all_ms:.1f}ms.

  Per-frame is {dec_all_ms/dec_frame_ms:.1f}x faster per decoder step.

  For N=50 denoising steps on {LATENT_F} frames:
    Baseline: {baseline_ms:.0f}ms × 50 = {baseline_ms*50:.0f}ms
    SCD:      {encoder_ms:.0f}ms + {dec_frame_ms:.0f}ms × 50 × {LATENT_F} = {encoder_ms + dec_frame_ms*50*LATENT_F:.0f}ms
    Speedup:  {(baseline_ms*50)/(encoder_ms + dec_frame_ms*50*LATENT_F):.2f}x
""")
print("=" * 70)
