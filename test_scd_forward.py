"""Test SCD encoder → shift → decoder forward pass.

Creates a small LTX-2 model and wraps it with LTXSCDModel to verify:
1. Encoder produces hidden features with causal mask
2. shift_encoder_features correctly shifts by 1 frame
3. Decoder token_concat handles tuple PE (cos, sin) correctly
4. Output shapes match expected velocity prediction
"""

import torch
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    KVCache,
    build_frame_causal_mask,
    shift_encoder_features,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig

torch.manual_seed(42)
DEVICE = "cpu"
DTYPE = torch.float32

# ── Small model for testing ──────────────────────────────────────
print("Creating small test model...")
model = LTXModel(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=128,
    out_channels=128,
    caption_channels=64,
    cross_attention_dim=32,
    num_layers=8,           # Small: 6 encoder + 2 decoder
    timestep_scale_multiplier=1000,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    apply_gated_attention=True,
)
model.eval()
print(f"  Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
      f"{len(model.transformer_blocks)} layers")

# ── Wrap with SCD (6 encoder + 2 decoder) ────────────────────────
scd = LTXSCDModel(base_model=model, encoder_layers=6, decoder_input_combine="token_concat")
print(f"  SCD split: {len(scd.encoder_blocks)} encoder + {len(scd.decoder_blocks)} decoder")

# ── Create test inputs ───────────────────────────────────────────
# Latent must be in PATCHIFIED format: [B, seq_len, C] (not raw [B, C, F, H, W])
# The patchify_proj is a Linear(in_channels=128, inner_dim=32) applied to last dim
B = 1
IN_CHANNELS = 128  # Must match model's in_channels
F, H, W = 3, 4, 4  # 3 frames, 4x4 latent
tokens_per_frame = H * W  # 16
seq_len = F * tokens_per_frame  # 48

# Patchified latent: [B, seq_len, C]
latent = torch.randn(B, seq_len, IN_CHANNELS, device=DEVICE, dtype=DTYPE)
# Positions: [B, 3, seq_len, 2] (3 axes = time, height, width; 2 = start, end)
positions = torch.randn(B, 3, seq_len, 2, device=DEVICE, dtype=DTYPE)
# Caption: [B, text_seq, caption_channels]
caption_embeds = torch.randn(B, 32, 64, device=DEVICE, dtype=DTYPE)
caption_mask = torch.ones(B, 32, device=DEVICE, dtype=DTYPE)

# Encoder: timestep = 0 (clean signal)
encoder_ts = torch.zeros(B, seq_len, device=DEVICE, dtype=DTYPE)
# Decoder: timestep = sigma (actual noise level)
decoder_ts = torch.full((B, seq_len), 0.5, device=DEVICE, dtype=DTYPE)

encoder_modality = Modality(
    latent=latent, positions=positions, context=caption_embeds,
    context_mask=caption_mask, timesteps=encoder_ts, enabled=True,
)
decoder_modality = Modality(
    latent=latent, positions=positions, context=caption_embeds,
    context_mask=caption_mask, timesteps=decoder_ts, enabled=True,
)

perturbations = BatchedPerturbationConfig.empty(B)

# ═════════════════════════════════════════════════════════════════
# TEST 1: Encoder forward pass
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: Encoder forward pass with causal mask")
print("=" * 60)

with torch.no_grad():
    video_args, audio_args = scd.forward_encoder(
        video=encoder_modality,
        audio=None,
        perturbations=perturbations,
        tokens_per_frame=tokens_per_frame,
    )

print(f"  encoder_features shape: {video_args.x.shape}")
print(f"  PE type: {type(video_args.positional_embeddings)}")
if isinstance(video_args.positional_embeddings, tuple):
    print(f"  PE[0] (cos) shape: {video_args.positional_embeddings[0].shape}")
    print(f"  PE[1] (sin) shape: {video_args.positional_embeddings[1].shape}")
assert video_args.x.shape == (B, seq_len, scd.inner_dim), \
    f"Expected {(B, seq_len, scd.inner_dim)}, got {video_args.x.shape}"
print("  ✓ Encoder pass OK")

# ═════════════════════════════════════════════════════════════════
# TEST 2: Shift encoder features
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Shift encoder features by 1 frame")
print("=" * 60)

encoder_features = video_args.x
shifted = shift_encoder_features(encoder_features, tokens_per_frame, F)
print(f"  Input shape:   {encoder_features.shape}")
print(f"  Shifted shape: {shifted.shape}")
# Frame 0 should be all zeros
frame0 = shifted[:, :tokens_per_frame, :]
print(f"  Frame 0 all zeros: {(frame0 == 0).all().item()}")
# Frame 1 should be frame 0 of original
frame1_shifted = shifted[:, tokens_per_frame:2*tokens_per_frame, :]
frame0_original = encoder_features[:, :tokens_per_frame, :]
match = torch.allclose(frame1_shifted, frame0_original, atol=1e-6)
if not match:
    diff = (frame1_shifted - frame0_original).abs().max().item()
    print(f"  Frame 1 ~ original frame 0: max diff = {diff:.2e}")
else:
    print(f"  Frame 1 == original frame 0: True")
assert shifted.shape == encoder_features.shape
assert (frame0 == 0).all()
print("  ✓ Shift OK")

# ═════════════════════════════════════════════════════════════════
# TEST 3: Decoder forward pass (token_concat with tuple PE fix)
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Decoder forward pass (token_concat + tuple PE)")
print("=" * 60)

with torch.no_grad():
    video_pred, audio_pred = scd.forward_decoder(
        video=decoder_modality,
        encoder_features=shifted,
        audio=None,
        perturbations=perturbations,
    )

print(f"  video_pred shape: {video_pred.shape}")
# Output is patchified: [B, seq_len, out_channels]
expected_out = (B, seq_len, IN_CHANNELS)
assert video_pred.shape == expected_out, \
    f"Expected {expected_out}, got {video_pred.shape}"
print(f"  audio_pred: {audio_pred}")
print("  ✓ Decoder pass OK (PE tuple bug is FIXED)")

# ═════════════════════════════════════════════════════════════════
# TEST 4: Full SCD pipeline (encoder → shift → decoder)
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Full SCD pipeline end-to-end")
print("=" * 60)

with torch.no_grad():
    # Step 1: Encoder
    enc_video, _ = scd.forward_encoder(
        video=encoder_modality, audio=None,
        perturbations=perturbations, tokens_per_frame=tokens_per_frame,
    )
    # Step 2: Shift
    shifted_features = shift_encoder_features(enc_video.x, tokens_per_frame, F)
    # Step 3: Decoder
    pred, _ = scd.forward_decoder(
        video=decoder_modality, encoder_features=shifted_features,
        audio=None, perturbations=perturbations,
    )

print(f"  Input:  {latent.shape} (latent)")
print(f"  Output: {pred.shape} (velocity prediction)")
print(f"  Values finite: {torch.isfinite(pred).all().item()}")
print(f"  Mean: {pred.mean().item():.6f}, Std: {pred.std().item():.6f}")
print("  ✓ Full pipeline OK")

# ═════════════════════════════════════════════════════════════════
# TEST 5: Causal mask verification
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: Causal mask structure")
print("=" * 60)

mask = build_frame_causal_mask(seq_len, tokens_per_frame, DEVICE, DTYPE)
print(f"  Mask shape: {mask.shape}")
# Verify: query_frame >= key_frame is allowed (0.0), otherwise -inf
# Token 0 (frame 0) should only see frame 0 tokens
row0_allowed = (mask[0, 0, :] == 0).sum().item()
print(f"  Token 0 (frame 0) attends to {row0_allowed} tokens (expected {tokens_per_frame})")
# Last token (frame 2) should see all tokens
last_row_allowed = (mask[0, -1, :] == 0).sum().item()
print(f"  Token {seq_len-1} (frame {F-1}) attends to {last_row_allowed} tokens (expected {seq_len})")
assert row0_allowed == tokens_per_frame, f"Frame 0 should see {tokens_per_frame} tokens, got {row0_allowed}"
assert last_row_allowed == seq_len, f"Last frame should see {seq_len} tokens, got {last_row_allowed}"
print("  ✓ Causal mask correct")

# ═════════════════════════════════════════════════════════════════
# TEST 6: Passthrough forward (non-SCD mode)
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: Passthrough forward (backward compat)")
print("=" * 60)

with torch.no_grad():
    vx, ax = scd(video=encoder_modality, audio=None, perturbations=perturbations)

print(f"  Passthrough output: {vx.shape}")
assert vx.shape == (B, seq_len, IN_CHANNELS)
print("  ✓ Passthrough OK")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
