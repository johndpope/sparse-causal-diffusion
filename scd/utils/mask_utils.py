"""Mask utilities for EditCtrl video editing.

Provides conversion between pixel-space masks and token-space masks,
sparse token gathering/scattering, background preparation, and
synthetic mask generation for training.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def pixel_mask_to_token_mask(
    pixel_mask: Tensor,
    vae_temporal_factor: int = 8,
    vae_spatial_factor: int = 32,
    patch_size: int = 1,
    dilation_pixels: int = 0,
) -> Tensor:
    """Convert pixel-space binary mask to token-space boolean mask.

    The LTX-2 pipeline: pixel [B,1,F,H,W] → VAE downsample → latent [B,C,f,h,w]
    → patchify (patch_size=1) → tokens [B, f*h*w]. A token is True (masked/edit)
    if ANY pixel in its receptive field is masked.

    Args:
        pixel_mask: Binary mask [B, 1, F, H, W] in pixel space (1 = edit region)
        vae_temporal_factor: VAE temporal downsample (8 for LTX-2)
        vae_spatial_factor: VAE spatial downsample (32 for LTX-2)
        patch_size: Patchifier patch size (1 for LTX-2)
        dilation_pixels: Dilate pixel mask by this many pixels before downsampling

    Returns:
        Boolean token mask [B, seq_len] where True = edit token
    """
    B, _, F_pix, H_pix, W_pix = pixel_mask.shape

    # Optional dilation in pixel space
    if dilation_pixels > 0:
        kernel_size = 2 * dilation_pixels + 1
        pixel_mask = F.max_pool3d(
            pixel_mask.float(),
            kernel_size=(1, kernel_size, kernel_size),
            stride=1,
            padding=(0, dilation_pixels, dilation_pixels),
        )

    # Downsample to latent resolution using max-pool
    # If any pixel in the receptive field is masked, the latent token is masked
    f_lat = math.ceil(F_pix / vae_temporal_factor)
    h_lat = math.ceil(H_pix / vae_spatial_factor)
    w_lat = math.ceil(W_pix / vae_spatial_factor)

    # Use adaptive max pool for clean downsampling to exact latent dims
    token_mask_3d = F.adaptive_max_pool3d(
        pixel_mask.float(), output_size=(f_lat, h_lat, w_lat)
    )  # [B, 1, f, h, w]

    # Further downsample by patch_size if needed
    if patch_size > 1:
        token_mask_3d = F.adaptive_max_pool3d(
            token_mask_3d,
            output_size=(
                f_lat // patch_size,
                h_lat // patch_size,
                w_lat // patch_size,
            ),
        )

    # Flatten to token sequence [B, seq_len]
    token_mask = token_mask_3d.view(B, -1) > 0.5
    return token_mask


def dilate_token_mask(
    token_mask: Tensor,
    tokens_per_frame: int,
    dilation: int = 2,
) -> Tensor:
    """Dilate token mask by N latent positions for boundary context.

    Expands masked regions so the LocalContextModule has context around
    edit boundaries (important for seamless blending).

    Args:
        token_mask: Boolean mask [B, seq_len]
        tokens_per_frame: Number of tokens per frame (h_lat * w_lat)
        dilation: Number of latent positions to dilate

    Returns:
        Dilated boolean mask [B, seq_len]
    """
    if dilation <= 0:
        return token_mask

    B, seq_len = token_mask.shape
    num_frames = seq_len // tokens_per_frame
    h_lat = w_lat = int(math.sqrt(tokens_per_frame))

    # Reshape to spatial grid per frame
    mask_3d = token_mask.float().view(B, num_frames, h_lat, w_lat)

    # Dilate with max pool per frame
    kernel = 2 * dilation + 1
    mask_3d = mask_3d.view(B * num_frames, 1, h_lat, w_lat)
    mask_3d = F.max_pool2d(mask_3d, kernel_size=kernel, stride=1, padding=dilation)
    mask_3d = mask_3d.view(B, num_frames, h_lat, w_lat)

    return mask_3d.view(B, seq_len) > 0.5


def gather_masked_tokens(tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Gather tokens at masked (True) positions into dense tensor.

    Args:
        tokens: [B, seq_len, D]
        mask: Boolean [B, seq_len]

    Returns:
        Tuple of:
        - sparse_tokens: [B, max_masked, D] (padded with zeros)
        - lengths: [B] number of actual masked tokens per batch item
    """
    B, S, D = tokens.shape
    # Count masked tokens per batch element
    lengths = mask.sum(dim=1)  # [B]
    max_masked = lengths.max().item()

    if max_masked == 0:
        return tokens.new_zeros(B, 1, D), lengths

    sparse = tokens.new_zeros(B, max_masked, D)
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False).squeeze(-1)  # [num_masked]
        n = idx.shape[0]
        if n > 0:
            sparse[b, :n] = tokens[b, idx]

    return sparse, lengths


def scatter_masked_tokens(
    sparse: Tensor,
    mask: Tensor,
    seq_len: int,
) -> Tensor:
    """Scatter sparse tokens back to full sequence (zeros elsewhere).

    Args:
        sparse: [B, max_masked, D] dense masked tokens
        mask: Boolean [B, seq_len]
        seq_len: Full sequence length

    Returns:
        Full sequence [B, seq_len, D] with sparse values at masked positions
    """
    B, _, D = sparse.shape
    full = sparse.new_zeros(B, seq_len, D)
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
        n = idx.shape[0]
        if n > 0:
            full[b, idx] = sparse[b, :n]
    return full


def prepare_background_latents(
    source_latents: Tensor,
    edit_mask: Tensor,
    target_num_tokens: int = 256,
) -> Tensor:
    """Prepare background context latents for GlobalContextEmbedder.

    Zeros out the masked (edit) region in source latents, then downsamples
    to a fixed number of tokens for the global context pathway.

    Args:
        source_latents: Patchified source latents [B, seq_len, C]
        edit_mask: Boolean token mask [B, seq_len] (True = edit region)
        target_num_tokens: Target number of background tokens

    Returns:
        Background tokens [B, target_num_tokens, C]
    """
    B, S, C = source_latents.shape

    # Zero out masked region — keep only background
    bg_mask = (~edit_mask).unsqueeze(-1).float()  # [B, S, 1]
    bg_latents = source_latents * bg_mask  # [B, S, C]

    # Downsample to fixed number of tokens via adaptive average pool
    # Reshape: [B, S, C] → [B, C, S] for 1D pooling
    bg_transposed = bg_latents.permute(0, 2, 1)  # [B, C, S]
    bg_pooled = F.adaptive_avg_pool1d(bg_transposed, target_num_tokens)  # [B, C, target]
    bg_tokens = bg_pooled.permute(0, 2, 1)  # [B, target, C]

    return bg_tokens


def generate_random_masks(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    min_area: float = 0.05,
    max_area: float = 0.6,
    device: torch.device | str = "cpu",
    mask_types: Sequence[str] | None = None,
) -> Tensor:
    """Generate synthetic edit masks for training.

    Creates random rectangle or ellipse masks that cover between min_area
    and max_area of the frame. Masks are consistent across all frames
    (static masks, not moving).

    Args:
        batch_size: Number of masks to generate
        num_frames: Number of video frames
        height: Pixel height
        width: Pixel width
        min_area: Minimum mask area fraction
        max_area: Maximum mask area fraction
        device: Target device
        mask_types: List of mask types to sample from. Default: ["rectangle", "ellipse"]

    Returns:
        Binary mask [B, 1, F, H, W] where 1 = edit region
    """
    if mask_types is None:
        mask_types = ["rectangle", "ellipse"]

    masks = torch.zeros(batch_size, 1, num_frames, height, width, device=device)

    for b in range(batch_size):
        mask_type = random.choice(mask_types)
        area_frac = random.uniform(min_area, max_area)

        if mask_type == "rectangle":
            # Random rectangle with target area
            aspect = random.uniform(0.5, 2.0)
            mask_h = int(math.sqrt(area_frac * height * width / aspect))
            mask_w = int(mask_h * aspect)
            mask_h = min(mask_h, height)
            mask_w = min(mask_w, width)

            y0 = random.randint(0, max(0, height - mask_h))
            x0 = random.randint(0, max(0, width - mask_w))
            masks[b, 0, :, y0 : y0 + mask_h, x0 : x0 + mask_w] = 1.0

        elif mask_type == "ellipse":
            # Random ellipse
            cy = random.uniform(0.2, 0.8) * height
            cx = random.uniform(0.2, 0.8) * width
            ry = math.sqrt(area_frac * height * width / math.pi) * random.uniform(0.7, 1.3)
            rx = (area_frac * height * width) / (math.pi * ry) if ry > 0 else ry

            yy = torch.arange(height, device=device).float()
            xx = torch.arange(width, device=device).float()
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")

            ellipse = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2
            masks[b, 0, :] = (ellipse <= 1.0).float()

    return masks


def generate_random_token_masks(
    batch_size: int,
    seq_len: int,
    tokens_per_frame: int,
    min_area: float = 0.05,
    max_area: float = 0.6,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generate random masks directly in token space (faster for training).

    Operates at latent resolution without pixel-space conversion.

    Args:
        batch_size: Number of masks
        seq_len: Total token sequence length
        tokens_per_frame: Tokens per frame (h_lat * w_lat)
        min_area: Minimum area fraction
        max_area: Maximum area fraction
        device: Target device

    Returns:
        Boolean token mask [B, seq_len] where True = edit region
    """
    num_frames = seq_len // tokens_per_frame
    h = w = int(math.sqrt(tokens_per_frame))

    masks = torch.zeros(batch_size, num_frames, h, w, device=device, dtype=torch.bool)

    for b in range(batch_size):
        area_frac = random.uniform(min_area, max_area)
        aspect = random.uniform(0.5, 2.0)
        mask_h = int(math.sqrt(area_frac * h * w / aspect))
        mask_w = int(mask_h * aspect)
        mask_h = min(max(mask_h, 1), h)
        mask_w = min(max(mask_w, 1), w)

        y0 = random.randint(0, max(0, h - mask_h))
        x0 = random.randint(0, max(0, w - mask_w))
        masks[b, :, y0 : y0 + mask_h, x0 : x0 + mask_w] = True

    return masks.view(batch_size, seq_len)
