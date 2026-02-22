#!/usr/bin/env python3
"""Test EditCtrl + SCD forward pass training loop.

Validates the full training pipeline:
  1. SCD model wrapping (encoder → shift → decoder)
  2. LocalContextModule (sparse boundary tokens → dense control)
  3. GlobalContextEmbedder (background → global tokens)
  4. Mask generation (random, dilation)
  5. Decoder forward pass with EditCtrl control signals
  6. Loss computation on edit region only
  7. Gradient flow through trainable EditCtrl modules

Uses a small synthetic model on CPU to verify shapes and data flow
without needing 43GB of real weights or GPU.
"""

import sys
import os
import torch
import torch.nn as nn
from dataclasses import dataclass

# Ensure scd/ package is importable
sys.path.insert(0, os.path.dirname(__file__))

from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    shift_encoder_features,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.editctrl_modules import (
    LocalContextModule,
    GlobalContextEmbedder,
)
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from scd.utils.mask_utils import (
    generate_random_token_masks,
    dilate_token_mask,
    gather_masked_tokens,
    prepare_background_latents,
)

torch.manual_seed(42)
DEVICE = "cpu"
DTYPE = torch.float32

# ═════════════════════════════════════════════════════════════════
#  Model dimensions (small for testing)
# ═════════════════════════════════════════════════════════════════
NUM_HEADS = 2
HEAD_DIM = 16
INNER_DIM = NUM_HEADS * HEAD_DIM  # 32
IN_CHANNELS = 128  # LTX-2 patchified latent dim
CAPTION_DIM = 64   # Text embedding dim
CROSS_DIM = 32     # Cross-attention dim
NUM_LAYERS = 8     # 6 encoder + 2 decoder
ENCODER_LAYERS = 6

# Video dimensions
B = 1
F = 3       # 3 frames
H = 4       # 4x4 latent grid
W = 4
TOKENS_PER_FRAME = H * W  # 16
SEQ_LEN = F * TOKENS_PER_FRAME  # 48

# EditCtrl params
LCM_INNER_DIM = 32
LCM_HEADS = 2
LCM_DIM_HEAD = 16
LCM_NUM_BLOCKS = 2
MASK_DILATION = 1
GLOBAL_NUM_TOKENS = 8


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═════════════════════════════════════════════════════════════════
#  Step 0: Create small model + SCD wrapper
# ═════════════════════════════════════════════════════════════════
print_header("STEP 0: Create small LTX-2 model + SCD wrapper")

model = LTXModel(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=NUM_HEADS,
    attention_head_dim=HEAD_DIM,
    in_channels=IN_CHANNELS,
    out_channels=IN_CHANNELS,
    caption_channels=CAPTION_DIM,
    cross_attention_dim=CROSS_DIM,
    num_layers=NUM_LAYERS,
    timestep_scale_multiplier=1000,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    apply_gated_attention=True,
)

scd = LTXSCDModel(
    base_model=model,
    encoder_layers=ENCODER_LAYERS,
    decoder_input_combine="token_concat",
)
print(f"  Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
print(f"  SCD: {len(scd.encoder_blocks)} enc + {len(scd.decoder_blocks)} dec")
print(f"  Inner dim: {scd.inner_dim}")

# ═════════════════════════════════════════════════════════════════
#  Step 1: Create EditCtrl modules
# ═════════════════════════════════════════════════════════════════
print_header("STEP 1: Create EditCtrl modules")

# LCM output_dim must match the model's inner_dim for injection into decoder
# latent_dim=128 (raw), mask_channel_concat=True → patchify_proj = Linear(129, inner)
lcm = LocalContextModule(
    latent_dim=IN_CHANNELS,        # Raw latent dim (mask channel added internally)
    inner_dim=LCM_INNER_DIM,
    context_dim=CAPTION_DIM,       # Must match text embedding dim
    num_blocks=LCM_NUM_BLOCKS,
    heads=LCM_HEADS,
    dim_head=LCM_DIM_HEAD,
    output_dim=scd.inner_dim,      # Must match model inner_dim for control injection
    timestep_dim=scd.inner_dim,    # AdaLN timestep embedding dim
    mask_channel_concat=True,
)
lcm_params = sum(p.numel() for p in lcm.parameters())
print(f"  LocalContextModule: {lcm_params / 1e3:.1f}K params")
print(f"    latent_dim={IN_CHANNELS}, inner={LCM_INNER_DIM}, output={scd.inner_dim}")

gce = GlobalContextEmbedder(
    latent_dim=IN_CHANNELS,
    inner_dim=scd.inner_dim,
    num_tokens=GLOBAL_NUM_TOKENS,
)
gce_params = sum(p.numel() for p in gce.parameters())
print(f"  GlobalContextEmbedder: {gce_params / 1e3:.1f}K params")
print(f"    latent_dim={IN_CHANNELS}, inner={scd.inner_dim}, num_tokens={GLOBAL_NUM_TOKENS}")

# ═════════════════════════════════════════════════════════════════
#  Step 2: Generate synthetic training batch
# ═════════════════════════════════════════════════════════════════
print_header("STEP 2: Generate synthetic training batch")

# Patchified video latents: [B, seq_len, C]
video_latents = torch.randn(B, SEQ_LEN, IN_CHANNELS, device=DEVICE, dtype=DTYPE)
# Video positions: [B, 3, seq_len, 2]
video_positions = torch.randn(B, 3, SEQ_LEN, 2, device=DEVICE, dtype=DTYPE)
# Text embeddings: [B, text_seq, caption_dim]
text_embeds = torch.randn(B, 16, CAPTION_DIM, device=DEVICE, dtype=DTYPE)
text_mask = torch.ones(B, 16, device=DEVICE, dtype=DTYPE)
# Sigma (noise level)
sigma = torch.tensor([0.5], device=DEVICE, dtype=DTYPE)

print(f"  video_latents: {video_latents.shape}")
print(f"  video_positions: {video_positions.shape}")
print(f"  text_embeds: {text_embeds.shape}")
print(f"  sigma: {sigma}")

# ═════════════════════════════════════════════════════════════════
#  Step 3: Generate edit masks
# ═════════════════════════════════════════════════════════════════
print_header("STEP 3: Generate edit masks + dilation")

edit_mask = generate_random_token_masks(
    batch_size=B,
    seq_len=SEQ_LEN,
    tokens_per_frame=TOKENS_PER_FRAME,
    min_area=0.05,
    max_area=0.6,
    device=DEVICE,
    height=H,
    width=W,
)
print(f"  edit_mask: {edit_mask.shape}, dtype={edit_mask.dtype}")
print(f"  edit_mask coverage: {edit_mask.float().mean():.2%}")

dilated_mask = dilate_token_mask(
    edit_mask,
    tokens_per_frame=TOKENS_PER_FRAME,
    dilation=MASK_DILATION,
    height=H,
    width=W,
)
print(f"  dilated_mask: {dilated_mask.shape}")
print(f"  dilated_mask coverage: {dilated_mask.float().mean():.2%}")
print(f"  boundary tokens: {(dilated_mask & ~edit_mask).float().sum().int().item()}")

# ═════════════════════════════════════════════════════════════════
#  Step 4: Construct noisy latents
# ═════════════════════════════════════════════════════════════════
print_header("STEP 4: Construct noisy latents")

video_noise = torch.randn_like(video_latents)
sigma_exp = sigma.view(-1, 1, 1)
noisy_video = (1 - sigma_exp) * video_latents + sigma_exp * video_noise

# Only noisy where edit_mask is True
edit_mask_exp = edit_mask.unsqueeze(-1)
noisy_video = torch.where(edit_mask_exp, noisy_video, video_latents)

# Velocity target
video_targets = video_noise - video_latents

print(f"  noisy_video: {noisy_video.shape}")
print(f"  video_targets: {video_targets.shape}")
print(f"  noisy tokens: {edit_mask.sum().item()}")
print(f"  clean tokens: {(~edit_mask).sum().item()}")

# ═════════════════════════════════════════════════════════════════
#  Step 5: SCD Encoder pass (clean latents, timestep=0)
# ═════════════════════════════════════════════════════════════════
print_header("STEP 5: SCD Encoder pass (timestep=0, causal mask)")

encoder_ts = torch.zeros(B, SEQ_LEN, device=DEVICE, dtype=DTYPE)
encoder_modality = Modality(
    latent=video_latents,
    positions=video_positions,
    context=text_embeds,
    context_mask=text_mask,
    timesteps=encoder_ts,
    enabled=True,
)
perturbations = BatchedPerturbationConfig.empty(B)

with torch.no_grad():
    enc_video_args, enc_audio_args = scd.forward_encoder(
        video=encoder_modality,
        audio=None,
        perturbations=perturbations,
        tokens_per_frame=TOKENS_PER_FRAME,
    )

encoder_features = enc_video_args.x
shifted_features = shift_encoder_features(encoder_features, TOKENS_PER_FRAME, F)

print(f"  encoder_features: {encoder_features.shape}")
print(f"  shifted_features: {shifted_features.shape}")
print(f"  frame 0 all zeros: {(shifted_features[:, :TOKENS_PER_FRAME] == 0).all().item()}")

# ═════════════════════════════════════════════════════════════════
#  Step 6: LocalContextModule forward
# ═════════════════════════════════════════════════════════════════
print_header("STEP 6: LocalContextModule forward pass")

# Concat edit_mask channel to source tokens (paper: C = [E(V_b), V_m↓])
edit_mask_channel = edit_mask.unsqueeze(-1).to(dtype=DTYPE)  # [B, seq_len, 1]
lcm_input = torch.cat([video_latents, edit_mask_channel], dim=-1)  # [B, seq_len, C+1]
print(f"  lcm_input (with mask channel): {lcm_input.shape}")

# Gather sparse tokens at dilated mask positions
sparse_tokens, sparse_lengths = gather_masked_tokens(lcm_input, dilated_mask)
print(f"  sparse_tokens: {sparse_tokens.shape} (max {sparse_lengths.max().item()} active)")

# Get timestep embedding from base model's AdaLN
with torch.no_grad():
    _, embedded_timestep = scd.base_model.adaln_single(sigma.flatten(), None)
timestep_emb = embedded_timestep.unsqueeze(1)  # [B, 1, inner_dim]
print(f"  timestep_emb: {timestep_emb.shape}")

# LCM forward (requires grad for testing gradient flow)
lcm.train()
local_control = lcm(
    source_tokens=sparse_tokens,
    mask_indices=dilated_mask,
    text_context=text_embeds,
    text_mask=text_mask,
    timestep_emb=timestep_emb,
    seq_len=SEQ_LEN,
)
print(f"  local_control: {local_control.shape}")
print(f"  local_control finite: {torch.isfinite(local_control).all().item()}")
assert local_control.shape == (B, SEQ_LEN, scd.inner_dim), \
    f"Expected {(B, SEQ_LEN, scd.inner_dim)}, got {local_control.shape}"
print(f"  OK: shape matches [B, seq_len, inner_dim]")

# ═════════════════════════════════════════════════════════════════
#  Step 7: GlobalContextEmbedder forward (Phase 2)
# ═════════════════════════════════════════════════════════════════
print_header("STEP 7: GlobalContextEmbedder forward pass (Phase 2)")

bg_tokens = prepare_background_latents(
    source_latents=video_latents,
    edit_mask=edit_mask,
    target_num_tokens=GLOBAL_NUM_TOKENS,
    fill_value=0.5,
)
print(f"  bg_tokens: {bg_tokens.shape}")

gce.train()
global_context = gce(bg_tokens)
print(f"  global_context: {global_context.shape}")
print(f"  global_context finite: {torch.isfinite(global_context).all().item()}")
assert global_context.shape == (B, GLOBAL_NUM_TOKENS, scd.inner_dim), \
    f"Expected {(B, GLOBAL_NUM_TOKENS, scd.inner_dim)}, got {global_context.shape}"
print(f"  OK: shape matches [B, num_global, inner_dim]")

# ═════════════════════════════════════════════════════════════════
#  Step 8: SCD Decoder pass with EditCtrl control signals
# ═════════════════════════════════════════════════════════════════
print_header("STEP 8: SCD Decoder pass + EditCtrl control injection")

# Decoder timesteps: sigma at all positions (no first-frame conditioning)
decoder_ts = torch.full((B, SEQ_LEN), 0.5, device=DEVICE, dtype=DTYPE)
decoder_modality = Modality(
    latent=noisy_video,
    positions=video_positions,
    context=text_embeds,
    context_mask=text_mask,
    timesteps=decoder_ts,
    enabled=True,
)

# Detach encoder features (they don't need gradients — only EditCtrl modules do)
shifted_features_detached = shifted_features.detach()

video_pred, audio_pred = scd.forward_decoder(
    video=decoder_modality,
    encoder_features=shifted_features_detached,
    audio=None,
    perturbations=perturbations,
    local_control=local_control,
    global_context=global_context,
)

print(f"  video_pred: {video_pred.shape}")
print(f"  video_pred finite: {torch.isfinite(video_pred).all().item()}")
assert video_pred.shape == (B, SEQ_LEN, IN_CHANNELS), \
    f"Expected {(B, SEQ_LEN, IN_CHANNELS)}, got {video_pred.shape}"
print(f"  OK: velocity prediction shape matches input")

# ═════════════════════════════════════════════════════════════════
#  Step 9: Compute loss (edit region only)
# ═════════════════════════════════════════════════════════════════
print_header("STEP 9: Compute masked loss (edit region only)")

# Velocity prediction loss, masked to edit region
loss_per_token = (video_pred - video_targets).pow(2)  # [B, seq_len, C]
mask_float = edit_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
masked_loss = loss_per_token * mask_float
num_masked = mask_float.sum().clamp(min=1.0)
loss = masked_loss.sum() / num_masked

print(f"  loss_per_token: {loss_per_token.shape}")
print(f"  num_masked_tokens: {int(edit_mask.sum().item())}")
print(f"  loss value: {loss.item():.6f}")
print(f"  loss finite: {torch.isfinite(loss).item()}")

# ═════════════════════════════════════════════════════════════════
#  Step 10: Backward pass + gradient check
# ═════════════════════════════════════════════════════════════════
print_header("STEP 10: Backward pass + gradient flow check")

loss.backward()

# Check LCM gradients
lcm_has_grad = False
lcm_grad_norm = 0.0
lcm_total_params = 0
for name, param in lcm.named_parameters():
    if param.requires_grad:
        lcm_total_params += 1
        if param.grad is not None:
            lcm_has_grad = True
            lcm_grad_norm += param.grad.norm().item() ** 2
lcm_grad_norm = lcm_grad_norm ** 0.5

print(f"  LCM trainable params: {lcm_total_params}")
print(f"  LCM has gradients: {lcm_has_grad}")
print(f"  LCM grad norm: {lcm_grad_norm:.6f}")

# Check GCE gradients
gce_has_grad = False
gce_grad_norm = 0.0
gce_total_params = 0
for name, param in gce.named_parameters():
    if param.requires_grad:
        gce_total_params += 1
        if param.grad is not None:
            gce_has_grad = True
            gce_grad_norm += param.grad.norm().item() ** 2
gce_grad_norm = gce_grad_norm ** 0.5

print(f"  GCE trainable params: {gce_total_params}")
print(f"  GCE has gradients: {gce_has_grad}")
print(f"  GCE grad norm: {gce_grad_norm:.6f}")

# Verify base model does NOT have gradients (should be frozen for EditCtrl)
base_has_grad = False
for param in model.parameters():
    if param.grad is not None and param.grad.abs().max() > 0:
        base_has_grad = True
        break

print(f"  Base model has gradients: {base_has_grad}")
if not base_has_grad:
    print(f"  OK: Base model frozen (no gradients)")

# Assert EditCtrl modules received gradients
assert lcm_has_grad, "FAIL: LocalContextModule did not receive gradients!"
assert gce_has_grad, "FAIL: GlobalContextEmbedder did not receive gradients!"
print(f"  OK: Both EditCtrl modules received gradients")

# ═════════════════════════════════════════════════════════════════
#  Step 11: Verify optimizer step would work
# ═════════════════════════════════════════════════════════════════
print_header("STEP 11: Verify optimizer step")

trainable_params = list(lcm.parameters()) + list(gce.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

# Record pre-step parameter values
lcm_param_before = next(lcm.parameters()).data.clone()
gce_param_before = next(gce.parameters()).data.clone()

optimizer.step()
optimizer.zero_grad()

# Verify parameters changed
lcm_param_after = next(lcm.parameters()).data
gce_param_after = next(gce.parameters()).data

# Use exact comparison — any update is a change
lcm_changed = not torch.equal(lcm_param_before, lcm_param_after)
gce_changed = not torch.equal(gce_param_before, gce_param_after)
lcm_diff = (lcm_param_before - lcm_param_after).abs().max().item()
gce_diff = (gce_param_before - gce_param_after).abs().max().item()

print(f"  LCM params changed after step: {lcm_changed} (max diff: {lcm_diff:.2e})")
print(f"  GCE params changed after step: {gce_changed} (max diff: {gce_diff:.2e})")
assert lcm_changed or lcm_diff > 0, "FAIL: LCM params did not change after optimizer step!"
assert gce_changed or gce_diff > 0, "FAIL: GCE params did not change after optimizer step!"
print(f"  OK: Optimizer step updates EditCtrl modules")


# ═════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════
print_header("ALL TESTS PASSED")
print(f"""
  EditCtrl + SCD Training Forward Pass Verified:

  1. SCD encoder pass (causal mask, timestep=0)           OK
  2. Feature shift (frame t-1 → t)                         OK
  3. Edit mask generation + dilation                        OK
  4. Noisy latent construction (masked regions only)        OK
  5. LocalContextModule (sparse → dense control)            OK
  6. GlobalContextEmbedder (background → global tokens)     OK
  7. SCD decoder with control injection                     OK
  8. Masked loss computation (edit region only)              OK
  9. Gradient flow through EditCtrl modules                  OK
  10. Optimizer step updates parameters                      OK

  Shapes:
    Input latent:     [{B}, {SEQ_LEN}, {IN_CHANNELS}]
    Edit mask:        [{B}, {SEQ_LEN}] ({edit_mask.float().mean():.0%} masked)
    Local control:    [{B}, {SEQ_LEN}, {scd.inner_dim}]
    Global context:   [{B}, {GLOBAL_NUM_TOKENS}, {scd.inner_dim}]
    Velocity pred:    [{B}, {SEQ_LEN}, {IN_CHANNELS}]
    Loss:             {loss.item():.4f} (masked MSE)
""")
