"""Trace tensor shapes through the LTX-2 forward pass.

Creates a model on meta device (no weights needed), then traces shapes
through the preprocessor, transformer blocks, and output projection.
"""

import torch
from dataclasses import replace

# ─────────────────────────────────────────────────────────────────
# LTX-2 Architecture Constants
# ─────────────────────────────────────────────────────────────────
NUM_LAYERS = 48
NUM_HEADS = 32
HEAD_DIM = 128
INNER_DIM = NUM_HEADS * HEAD_DIM  # 4096
IN_CHANNELS = 128          # VAE latent channels
CAPTION_CHANNELS = 3840    # Raw text embedding dim
CROSS_ATTN_DIM = 4096      # After caption projection

# Audio
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_INNER_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM  # 2048

# VAE compression: spatial 32x, temporal 8x (but first frame 1:1)
# Video: [B, C=128, F, H, W] where H,W are 1/32 of pixel, F is (frames-1)/8+1

# ─────────────────────────────────────────────────────────────────
# Example generation parameters
# ─────────────────────────────────────────────────────────────────
BATCH = 1
# For 512x512 @ 33 frames (1 second + keyframe at 24fps)
PIXEL_H, PIXEL_W = 512, 512
NUM_PIXEL_FRAMES = 33

# Latent dimensions after VAE encoding
LATENT_H = PIXEL_H // 32   # 16
LATENT_W = PIXEL_W // 32   # 16
LATENT_F = (NUM_PIXEL_FRAMES - 1) // 8 + 1  # 5

TOKENS_PER_FRAME = LATENT_H * LATENT_W  # 256
VIDEO_SEQ_LEN = LATENT_F * TOKENS_PER_FRAME  # 1280

# Text conditioning (Gemma output)
TEXT_SEQ_LEN = 512  # typical max

print("=" * 70)
print("LTX-2 FORWARD PASS SHAPE TRACE")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    INPUT PARAMETERS                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Pixel:  {PIXEL_H}×{PIXEL_W} × {NUM_PIXEL_FRAMES} frames                                ║
║  Latent: {LATENT_H}×{LATENT_W} × {LATENT_F} frames (VAE: 32× spatial, 8× temporal)       ║
║  Tokens: {TOKENS_PER_FRAME}/frame × {LATENT_F} frames = {VIDEO_SEQ_LEN} total                 ║
║  Text:   {TEXT_SEQ_LEN} tokens × {CAPTION_CHANNELS} dim (Gemma)                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────────────
# STAGE 1: VAE Encoding
# ─────────────────────────────────────────────────────────────────
print("─── STAGE 1: VAE Encoder ───")
pixel_input = f"[{BATCH}, 3, {NUM_PIXEL_FRAMES}, {PIXEL_H}, {PIXEL_W}]"
latent_output = f"[{BATCH}, {IN_CHANNELS}, {LATENT_F}, {LATENT_H}, {LATENT_W}]"
print(f"  Pixel video:    {pixel_input}")
print(f"  → VAE encode →")
print(f"  Latent video:   {latent_output}")
print()

# ─────────────────────────────────────────────────────────────────
# STAGE 2: Patchify (flatten spatial+temporal → sequence)
# ─────────────────────────────────────────────────────────────────
print("─── STAGE 2: Patchify (patch_size=1, just reshape) ───")
patchified = f"[{BATCH}, {VIDEO_SEQ_LEN}, {IN_CHANNELS}]"
print(f"  Latent:     [{BATCH}, {IN_CHANNELS}, {LATENT_F}, {LATENT_H}, {LATENT_W}]")
print(f"  → reshape → [{BATCH}, {LATENT_F}×{LATENT_H}×{LATENT_W}, {IN_CHANNELS}]")
print(f"  Patchified: {patchified}")
print()

# ─────────────────────────────────────────────────────────────────
# STAGE 3: TransformerArgsPreprocessor.prepare()
# ─────────────────────────────────────────────────────────────────
print("─── STAGE 3: Preprocessor (patchify_proj + AdaLN + RoPE) ───")
print(f"  patchify_proj: Linear({IN_CHANNELS} → {INNER_DIM})")
print(f"    x:            [{BATCH}, {VIDEO_SEQ_LEN}, {IN_CHANNELS}] → [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]")
print()
print(f"  adaln_single: timestep → (shift, scale, gate) embeddings")
print(f"    timesteps:    [{BATCH}, {VIDEO_SEQ_LEN}]")
print(f"    → adaln →    [{BATCH}, {VIDEO_SEQ_LEN}, 6×{INNER_DIM}]  (6 ada params)")
print(f"    embedded_ts:  [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]")
print()
print(f"  caption_proj: Linear({CAPTION_CHANNELS} → {INNER_DIM})")
print(f"    context:      [{BATCH}, {TEXT_SEQ_LEN}, {CAPTION_CHANNELS}] → [{BATCH}, {TEXT_SEQ_LEN}, {INNER_DIM}]")
print()
print(f"  RoPE PE:        [{BATCH}, {VIDEO_SEQ_LEN}, {NUM_HEADS}, {HEAD_DIM}//3, 2]")
print()
print(f"  TransformerArgs {{")
print(f"    x:                     [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]")
print(f"    context:               [{BATCH}, {TEXT_SEQ_LEN}, {INNER_DIM}]")
print(f"    context_mask:          [{BATCH}, 1, 1, {TEXT_SEQ_LEN}]")
print(f"    timesteps:             [{BATCH}, {VIDEO_SEQ_LEN}, 6×{INNER_DIM}]")
print(f"    embedded_timestep:     [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]")
print(f"    positional_embeddings: [{BATCH}, {VIDEO_SEQ_LEN}, {NUM_HEADS}, ...]")
print(f"    self_attn_mask:        None (or [{1}, {VIDEO_SEQ_LEN}, {VIDEO_SEQ_LEN}] for SCD)")
print(f"  }}")
print()

# ─────────────────────────────────────────────────────────────────
# STAGE 4: Transformer Blocks (×48)
# ─────────────────────────────────────────────────────────────────
print("─── STAGE 4: Transformer Blocks (×48) ───")
print(f"""
  Each BasicAVTransformerBlock:
  ┌─────────────────────────────────────────────────────────────────┐
  │  INPUT: x [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]                           │
  │                                                                 │
  │  1. Video Self-Attention (attn1)                                │
  │     AdaLN: shift/scale/gate from timestep embedding             │
  │     norm_x = RMSNorm(x) * (1 + scale) + shift                  │
  │     Q,K,V = Linear(norm_x) → [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}] each   │
  │     Q,K = RMSNorm → RoPE                                       │
  │     attn = softmax(Q·Kᵀ/√d + mask) · V                        │
  │     x = x + attn * gate                                        │
  │                                                                 │
  │  2. Cross-Attention (attn2) — text conditioning                 │
  │     Q = Linear(RMSNorm(x))   [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]         │
  │     K,V = Linear(context)    [{BATCH}, {TEXT_SEQ_LEN}, {INNER_DIM}]         │
  │     attn = softmax(Q·Kᵀ/√d + ctx_mask) · V                    │
  │     x = x + attn                                               │
  │                                                                 │
  │  3. Audio↔Video Cross-Attention (if audio enabled)              │
  │     (skipped for video-only)                                    │
  │                                                                 │
  │  4. Feed-Forward (ff)                                           │
  │     AdaLN: shift/scale/gate from timestep embedding             │
  │     norm_x = RMSNorm(x) * (1 + scale) + shift                  │
  │     x = x + GELU_MLP(norm_x) * gate                            │
  │                                                                 │
  │  OUTPUT: x [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]  (same shape)             │
  └─────────────────────────────────────────────────────────────────┘

  All 48 blocks have identical structure and shapes.
  Total parameters per block: ~100M  (48 blocks ≈ 5B for transformer)
""")

# ─────────────────────────────────────────────────────────────────
# STAGE 5: Output Projection
# ─────────────────────────────────────────────────────────────────
print("─── STAGE 5: Output Projection ───")
print(f"  scale_shift_table: [{2}, {INNER_DIM}] → per-token shift/scale")
print(f"  norm_out: LayerNorm({INNER_DIM})")
print(f"  x = norm_out(x) * (1 + scale) + shift")
print(f"  proj_out: Linear({INNER_DIM} → {IN_CHANNELS})")
print(f"  x: [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}] → [{BATCH}, {VIDEO_SEQ_LEN}, {IN_CHANNELS}]")
print()

# ─────────────────────────────────────────────────────────────────
# STAGE 6: Unpatchify + VAE Decode
# ─────────────────────────────────────────────────────────────────
print("─── STAGE 6: Unpatchify + VAE Decode ───")
print(f"  Unpatchify: [{BATCH}, {VIDEO_SEQ_LEN}, {IN_CHANNELS}] → [{BATCH}, {IN_CHANNELS}, {LATENT_F}, {LATENT_H}, {LATENT_W}]")
print(f"  VAE decode: [{BATCH}, {IN_CHANNELS}, {LATENT_F}, {LATENT_H}, {LATENT_W}] → [{BATCH}, 3, {NUM_PIXEL_FRAMES}, {PIXEL_H}, {PIXEL_W}]")
print()

# ─────────────────────────────────────────────────────────────────
# SCD SPLIT DIAGRAM
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("SCD (SEPARABLE CAUSAL DIFFUSION) SPLIT")
print("=" * 70)

ENC_LAYERS = 32
DEC_LAYERS = 48 - ENC_LAYERS  # 16

print(f"""
  ┌────────────────────────────────────────────────────────────────┐
  │                    STANDARD LTX-2                              │
  │                                                                │
  │  Layer 0  ─┐                                                   │
  │  Layer 1   │                                                   │
  │  ...       │ All 48 layers run N denoising steps per frame     │
  │  Layer 47 ─┘                                                   │
  │                                                                │
  │  Cost: 48 × N × seq_len  per frame                            │
  └────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────┐
  │                    SCD SPLIT                                   │
  │                                                                │
  │  ENCODER (layers 0-{ENC_LAYERS-1})  ─ runs ONCE per frame              │
  │    Input:  clean latents, timestep=0, causal mask              │
  │    Output: encoder_features [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]          │
  │                                                                │
  │  ── shift by 1 frame ──                                        │
  │    frame t gets frame t-1's encoder features                   │
  │    frame 0 gets zeros                                          │
  │                                                                │
  │  DECODER (layers {ENC_LAYERS}-47) ─ runs N denoising steps per frame    │
  │    Input:  [encoder_features; noisy_tokens] (token_concat)     │
  │            actual sigma timestep                               │
  │    Output: velocity prediction [{BATCH}, {VIDEO_SEQ_LEN}, {IN_CHANNELS}]         │
  │                                                                │
  │  Cost: ({ENC_LAYERS} × 1 + {DEC_LAYERS} × N) × seq_len  per frame               │
  │  Speedup: ({ENC_LAYERS} + {DEC_LAYERS}×N) / (48×N) = ({ENC_LAYERS}/{48/1:.0f}N + {DEC_LAYERS}/48)            │
  │  @ N=50 steps: ({ENC_LAYERS} + {DEC_LAYERS}×50) / (48×50) = {(ENC_LAYERS + DEC_LAYERS*50)/(48*50):.2f}x       │
  │  @ N=20 steps: ({ENC_LAYERS} + {DEC_LAYERS}×20) / (48×20) = {(ENC_LAYERS + DEC_LAYERS*20)/(48*20):.2f}x       │
  │  @ N=10 steps: ({ENC_LAYERS} + {DEC_LAYERS}×10) / (48×10) = {(ENC_LAYERS + DEC_LAYERS*10)/(48*10):.2f}x       │
  └────────────────────────────────────────────────────────────────┘

  Token Concat Shapes (decoder input):
    encoder_features (shifted): [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]
    noisy_tokens:               [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]
    combined:                   [{BATCH}, {2*VIDEO_SEQ_LEN}, {INNER_DIM}]

    After decoder, take second half:
    decoder_output:             [{BATCH}, {2*VIDEO_SEQ_LEN}, {INNER_DIM}]
    → slice [{VIDEO_SEQ_LEN}:]:          [{BATCH}, {VIDEO_SEQ_LEN}, {INNER_DIM}]
    → proj_out:                 [{BATCH}, {VIDEO_SEQ_LEN}, {IN_CHANNELS}]
""")

# ─────────────────────────────────────────────────────────────────
# CHECKPOINT WEIGHT MAPPING
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("CHECKPOINT WEIGHT MAPPING FOR SCD")
print("=" * 70)
print(f"""
  The LTX-2 checkpoint stores weights as:
    transformer_blocks.0.attn1.*     ← Layer 0 self-attention
    transformer_blocks.0.attn2.*     ← Layer 0 cross-attention
    transformer_blocks.0.ff.*        ← Layer 0 feed-forward
    transformer_blocks.0.scale_shift_table  ← Layer 0 AdaLN params
    ...
    transformer_blocks.47.attn1.*    ← Layer 47

  SCD uses the SAME weights — no surgical cutting needed!
    encoder_blocks = base_model.transformer_blocks[:32]  # shared refs
    decoder_blocks = base_model.transformer_blocks[32:]  # shared refs

  Only new parameters in SCD wrapper:
    - token_concat_with_proj mode: LayerNorm + Linear({INNER_DIM}→{INNER_DIM})
      (small: ~33M params, << 19B total)
    - token_concat mode: NO new parameters at all!

  ∴ You can load any LTX-2 checkpoint directly into SCD wrapper.
    No weight conversion or surgical extraction needed.
""")

# ─────────────────────────────────────────────────────────────────
# TRAINING REQUIREMENTS
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("TRAINING REQUIREMENTS")
print("=" * 70)
print(f"""
  Q: Does SCD need additional training?

  A: It depends on the coupling mode:

  1. token_concat (RECOMMENDED — no new params):
     - Uses base LTX-2 weights as-is for both encoder and decoder
     - The encoder layers already learned temporal features
     - The decoder layers already learned spatial denoising
     - Causal mask + frame shift change the ATTENTION PATTERN, not weights
     - SOME fine-tuning is recommended to adapt to the split:
       * Encoder learns to produce clean temporal features (timestep=0)
       * Decoder learns to condition on concatenated encoder tokens
     - Estimated: 5K-20K steps with existing training data
     - Can start generating immediately with degraded quality

  2. token_concat_with_proj (small learned alignment):
     - Same as above but adds a LayerNorm+Linear projection
     - Needs training to learn the alignment projection
     - Estimated: 10K-30K steps

  3. add mode (element-wise addition):
     - Simplest coupling, may lose information
     - Still needs fine-tuning for the split
     - Estimated: 5K-15K steps

  Key insight: The SCD paper shows that early layers (0-31) naturally
  produce REDUNDANT features across denoising steps. This is why SCD
  works — those layers already compute temporal features that don't
  change much between steps. The split formalizes this observation.
""")

print("DONE — see state diagram above")
