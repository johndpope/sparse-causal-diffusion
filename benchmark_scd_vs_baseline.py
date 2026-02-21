"""
SCD vs Baseline: Real comparison with LTX-2 19B checkpoint.

Loads actual weights, runs both standard and SCD forward passes on
the same input, and compares:
  1. Output similarity (cosine sim, MSE)
  2. Wall-clock time
  3. Output statistics

This tells us: is SCD output in the same ballpark as baseline WITHOUT fine-tuning?
"""

import time
import torch
import gc
from safetensors.torch import load_file

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
CHECKPOINT = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
DEVICE = "cuda:0"  # RTX 5090 (32GB)
DTYPE = torch.bfloat16

# Small generation: 512x512, 9 frames (1 keyframe + 1 second at 8fps latent)
PIXEL_H, PIXEL_W = 512, 512
NUM_PIXEL_FRAMES = 9  # Keep small for memory
LATENT_H = PIXEL_H // 32   # 16
LATENT_W = PIXEL_W // 32   # 16
LATENT_F = (NUM_PIXEL_FRAMES - 1) // 8 + 1  # 2 latent frames
IN_CHANNELS = 128
TOKENS_PER_FRAME = LATENT_H * LATENT_W  # 256
SEQ_LEN = LATENT_F * TOKENS_PER_FRAME   # 512
TEXT_SEQ = 64  # Short text sequence

print("=" * 70)
print("SCD vs BASELINE COMPARISON — LTX-2 19B")
print("=" * 70)
print(f"  Checkpoint: {CHECKPOINT}")
print(f"  Device: {DEVICE}")
print(f"  Resolution: {PIXEL_H}x{PIXEL_W} @ {NUM_PIXEL_FRAMES} frames")
print(f"  Latent: {LATENT_H}x{LATENT_W} x {LATENT_F} frames")
print(f"  Video tokens: {SEQ_LEN}, Text tokens: {TEXT_SEQ}")

# ─────────────────────────────────────────────────────────────────
# Step 1: Load model
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("LOADING MODEL")
print("=" * 70)

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    build_frame_causal_mask,
    shift_encoder_features,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig

print("  Loading state dict from safetensors...")
t0 = time.time()
state_dict = load_file(CHECKPOINT, device="cpu")
print(f"  Loaded {len(state_dict)} keys in {time.time()-t0:.1f}s")

# Filter and rename transformer keys: "model.diffusion_model.X" → "X"
PREFIX = "model.diffusion_model."
renamed = {k[len(PREFIX):]: v for k, v in state_dict.items() if k.startswith(PREFIX)}
print(f"  Transformer keys: {len(renamed)}")

# Build VideoOnly model (no audio = less memory) WITHOUT gated attention
# (the dev checkpoint doesn't include gated attention weights)
print("  Building VideoOnly model...")
t0 = time.time()
model = LTXModel(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=32,
    attention_head_dim=128,
    in_channels=128,
    out_channels=128,
    num_layers=48,
    cross_attention_dim=4096,
    caption_channels=3840,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    apply_gated_attention=False,  # Checkpoint doesn't have these weights
)

# Load weights (strict=False because VideoOnly model doesn't have audio keys)
print("  Loading weights...")
result = model.load_state_dict(renamed, strict=False)
if result.missing_keys:
    print(f"  Missing keys: {len(result.missing_keys)} (expected for VideoOnly)")
if result.unexpected_keys:
    print(f"  Unexpected keys: {len(result.unexpected_keys)} (audio keys, ignored)")

model = model.to(device=DEVICE, dtype=DTYPE)
model.eval()
print(f"  Model loaded in {time.time()-t0:.1f}s")
print(f"  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# Free the state dict
del state_dict, renamed
gc.collect()
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────
# Step 2: Create inputs
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("CREATING INPUTS")
print("=" * 70)

torch.manual_seed(42)
B = 1

# Patchified video latent: [B, seq_len, C]
latent = torch.randn(B, SEQ_LEN, IN_CHANNELS, device=DEVICE, dtype=DTYPE)
# Positions: [B, 3, seq_len, 2]
positions = torch.randn(B, 3, SEQ_LEN, 2, device=DEVICE, dtype=DTYPE)
# Text embeddings (random, since we're comparing relative quality)
caption = torch.randn(B, TEXT_SEQ, 3840, device=DEVICE, dtype=DTYPE)
caption_mask = torch.ones(B, TEXT_SEQ, device=DEVICE, dtype=DTYPE)
# Timestep: sigma=0.5 for a mid-denoising step
timesteps = torch.full((B, SEQ_LEN), 0.5, device=DEVICE, dtype=DTYPE)

modality = Modality(
    enabled=True,
    latent=latent,
    timesteps=timesteps,
    positions=positions,
    context=caption,
    context_mask=caption_mask,
)

perturbations = BatchedPerturbationConfig.empty(B)
print(f"  latent: {latent.shape}, positions: {positions.shape}")
print(f"  caption: {caption.shape}, timesteps: {timesteps.shape}")

# ─────────────────────────────────────────────────────────────────
# Step 3: Baseline forward pass
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("BASELINE FORWARD PASS (all 48 layers)")
print("=" * 70)

# Warmup
with torch.no_grad():
    _ = model(video=modality, audio=None, perturbations=perturbations)
torch.cuda.synchronize()

# Timed run
timings_baseline = []
with torch.no_grad():
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        baseline_out, _ = model(video=modality, audio=None, perturbations=perturbations)
        torch.cuda.synchronize()
        dt = time.time() - t0
        timings_baseline.append(dt)
        print(f"  Run {i+1}: {dt*1000:.1f}ms")

baseline_mean_ms = sum(timings_baseline) / len(timings_baseline) * 1000
print(f"  Output shape: {baseline_out.shape}")
print(f"  Mean: {baseline_out.float().mean():.6f}, Std: {baseline_out.float().std():.6f}")
print(f"  Avg time: {baseline_mean_ms:.1f}ms")

# ─────────────────────────────────────────────────────────────────
# Step 4: SCD forward pass
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SCD FORWARD PASS (32 encoder + 16 decoder, token_concat)")
print("=" * 70)

scd = LTXSCDModel(base_model=model, encoder_layers=32, decoder_input_combine="token_concat")

# Encoder modality: timestep=0 (clean signal)
encoder_timesteps = torch.zeros(B, SEQ_LEN, device=DEVICE, dtype=DTYPE)
encoder_modality = Modality(
    enabled=True,
    latent=latent,
    timesteps=encoder_timesteps,
    positions=positions,
    context=caption,
    context_mask=caption_mask,
)

# Warmup
with torch.no_grad():
    enc_args, _ = scd.forward_encoder(
        video=encoder_modality, audio=None,
        perturbations=perturbations, tokens_per_frame=TOKENS_PER_FRAME,
    )
    shifted = shift_encoder_features(enc_args.x, TOKENS_PER_FRAME, LATENT_F)
    scd_out, _ = scd.forward_decoder(
        video=modality, encoder_features=shifted,
        audio=None, perturbations=perturbations,
    )
torch.cuda.synchronize()

# Timed runs
timings_encoder = []
timings_decoder = []
timings_scd_total = []

with torch.no_grad():
    for i in range(3):
        torch.cuda.synchronize()

        # Encoder (runs once per frame in inference)
        t0 = time.time()
        enc_args, _ = scd.forward_encoder(
            video=encoder_modality, audio=None,
            perturbations=perturbations, tokens_per_frame=TOKENS_PER_FRAME,
        )
        shifted = shift_encoder_features(enc_args.x, TOKENS_PER_FRAME, LATENT_F)
        torch.cuda.synchronize()
        t_enc = time.time() - t0

        # Decoder (runs N times per frame in inference)
        t0 = time.time()
        scd_out, _ = scd.forward_decoder(
            video=modality, encoder_features=shifted,
            audio=None, perturbations=perturbations,
        )
        torch.cuda.synchronize()
        t_dec = time.time() - t0

        timings_encoder.append(t_enc)
        timings_decoder.append(t_dec)
        timings_scd_total.append(t_enc + t_dec)
        print(f"  Run {i+1}: encoder={t_enc*1000:.1f}ms, decoder={t_dec*1000:.1f}ms, "
              f"total={( t_enc+t_dec)*1000:.1f}ms")

enc_mean_ms = sum(timings_encoder) / len(timings_encoder) * 1000
dec_mean_ms = sum(timings_decoder) / len(timings_decoder) * 1000
scd_total_ms = enc_mean_ms + dec_mean_ms
print(f"  Output shape: {scd_out.shape}")
print(f"  Mean: {scd_out.float().mean():.6f}, Std: {scd_out.float().std():.6f}")
print(f"  Avg: encoder={enc_mean_ms:.1f}ms, decoder={dec_mean_ms:.1f}ms, total={scd_total_ms:.1f}ms")

# ─────────────────────────────────────────────────────────────────
# Step 5: Compare outputs
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("OUTPUT COMPARISON")
print("=" * 70)

with torch.no_grad():
    # Flatten for comparison
    b_flat = baseline_out.float().flatten()
    s_flat = scd_out.float().flatten()

    # MSE
    mse = (b_flat - s_flat).pow(2).mean().item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        b_flat.unsqueeze(0), s_flat.unsqueeze(0)
    ).item()

    # Correlation
    b_centered = b_flat - b_flat.mean()
    s_centered = s_flat - s_flat.mean()
    correlation = (b_centered * s_centered).sum() / (b_centered.norm() * s_centered.norm())

    # Per-element comparison
    abs_diff = (b_flat - s_flat).abs()

    print(f"  MSE:                 {mse:.6f}")
    print(f"  Cosine similarity:   {cos_sim:.6f}")
    print(f"  Pearson correlation: {correlation.item():.6f}")
    print(f"  Max absolute diff:   {abs_diff.max().item():.6f}")
    print(f"  Mean absolute diff:  {abs_diff.mean().item():.6f}")
    print()
    print(f"  Baseline stats: mean={baseline_out.float().mean():.4f}, std={baseline_out.float().std():.4f}, "
          f"min={baseline_out.float().min():.4f}, max={baseline_out.float().max():.4f}")
    print(f"  SCD stats:      mean={scd_out.float().mean():.4f}, std={scd_out.float().std():.4f}, "
          f"min={scd_out.float().min():.4f}, max={scd_out.float().max():.4f}")

# ─────────────────────────────────────────────────────────────────
# Step 6: Speedup analysis
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SPEEDUP ANALYSIS")
print("=" * 70)

print(f"\n  Single forward pass:")
print(f"    Baseline (48 layers): {baseline_mean_ms:.1f}ms")
print(f"    SCD encoder (32 layers + causal mask): {enc_mean_ms:.1f}ms")
print(f"    SCD decoder (16 layers, token_concat): {dec_mean_ms:.1f}ms")
print(f"    SCD total (enc+dec): {scd_total_ms:.1f}ms")

print(f"\n  N-step denoising (encoder runs once, decoder runs N times):")
for N in [8, 20, 50]:
    baseline_total = baseline_mean_ms * N
    scd_inference = enc_mean_ms + dec_mean_ms * N
    speedup = baseline_total / scd_inference
    print(f"    N={N:3d}: Baseline={baseline_total:.0f}ms, SCD={scd_inference:.0f}ms, "
          f"Speedup={speedup:.2f}x")

print(f"\n  Per-frame autoregressive (encoder has KV-cache, only new tokens):")
print(f"    After first frame, encoder cost drops to ~1/{LATENT_F} of full pass")
print(f"    (Only new frame's {TOKENS_PER_FRAME} tokens computed, rest from KV-cache)")
enc_per_frame_est = enc_mean_ms / LATENT_F  # Rough estimate with KV-cache
for N in [8, 20, 50]:
    baseline_total = baseline_mean_ms * N
    scd_per_frame = enc_per_frame_est + dec_mean_ms * N
    speedup = baseline_total / scd_per_frame
    print(f"    N={N:3d}: Baseline={baseline_total:.0f}ms, SCD≈{scd_per_frame:.0f}ms, "
          f"Speedup≈{speedup:.2f}x")

# ─────────────────────────────────────────────────────────────────
# Step 7: Quality assessment
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("QUALITY ASSESSMENT (without fine-tuning)")
print("=" * 70)

if cos_sim > 0.9:
    quality = "VERY SIMILAR — outputs strongly correlated"
elif cos_sim > 0.5:
    quality = "SOMEWHAT SIMILAR — outputs partially correlated"
elif cos_sim > 0.0:
    quality = "WEAKLY CORRELATED — significant divergence"
else:
    quality = "UNCORRELATED/OPPOSED — output is likely garbage"

print(f"""
  Cosine similarity: {cos_sim:.4f} → {quality}

  What this means:
  - cos_sim > 0.9: SCD is nearly equivalent to baseline, fine-tuning optional
  - cos_sim 0.5-0.9: SCD captures some structure, fine-tuning recommended
  - cos_sim < 0.5: Outputs diverge significantly, fine-tuning REQUIRED
  - cos_sim < 0.0: Completely broken, architecture issues

  Expected: Without fine-tuning, cosine sim around 0.3-0.6.
  The encoder has never seen a causal mask, and the decoder has
  never received token_concat features. The shared weights provide
  a starting point but need adaptation.

  The SCD paper reports that fine-tuning for 10K-50K steps brings
  quality to within ~2% of baseline FVD scores.
""")

print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
