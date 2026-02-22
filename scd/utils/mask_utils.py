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
    height: int | None = None,
    width: int | None = None,
) -> Tensor:
    """Dilate token mask by N latent positions for boundary context.

    Expands masked regions so the LocalContextModule has context around
    edit boundaries (important for seamless blending).

    Args:
        token_mask: Boolean mask [B, seq_len]
        tokens_per_frame: Number of tokens per frame (h_lat * w_lat)
        dilation: Number of latent positions to dilate
        height: Latent spatial height (if None, assumes square)
        width: Latent spatial width (if None, assumes square)

    Returns:
        Dilated boolean mask [B, seq_len]
    """
    if dilation <= 0:
        return token_mask

    B, seq_len = token_mask.shape
    num_frames = seq_len // tokens_per_frame
    if height is not None and width is not None:
        h_lat, w_lat = height, width
    else:
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
    fill_value: float = 0.5,
) -> Tensor:
    """Prepare background context latents for GlobalContextEmbedder.

    Fills the masked (edit) region with a neutral value, then downsamples
    to a fixed number of tokens for the global context pathway.

    Args:
        source_latents: Patchified source latents [B, seq_len, C]
        edit_mask: Boolean token mask [B, seq_len] (True = edit region)
        target_num_tokens: Target number of background tokens
        fill_value: Value to fill masked (edit) region with (paper: 0.5)

    Returns:
        Background tokens [B, target_num_tokens, C]
    """
    B, S, C = source_latents.shape

    # Keep background, fill masked (edit) region with 0.5 (mid-gray neutral value).
    # Paper: "set masked area to 0.5 in source video" — avoids hard zeros that
    # create artifacts in VAE latent space and bias pooling.
    bg_mask = (~edit_mask).unsqueeze(-1).float()  # [B, S, 1]
    bg_latents = source_latents * bg_mask + fill_value * (1 - bg_mask)  # [B, S, C]

    # Downsample to fixed number of tokens via adaptive average pool
    # Reshape: [B, S, C] → [B, C, S] for 1D pooling
    bg_transposed = bg_latents.permute(0, 2, 1)  # [B, C, S]
    bg_pooled = F.adaptive_avg_pool1d(bg_transposed, target_num_tokens)  # [B, C, target]
    bg_tokens = bg_pooled.permute(0, 2, 1)  # [B, target, C]

    return bg_tokens


def compute_boundary_blend_mask(
    token_mask: Tensor,
    tokens_per_frame: int,
    height: int,
    width: int,
    falloff_tokens: int = 3,
    sharpness: float = 10.0,
) -> Tensor:
    """Compute a soft boundary blend mask using signed distance from mask edges.

    For each token, computes its distance to the nearest mask boundary. Tokens
    deep inside the mask get alpha ~1.0, tokens far outside get ~0.0, and
    boundary tokens get a smooth sigmoid transition.

    This replaces hard binary torch.where() masking with smooth blending at
    edit boundaries (LayerFusion Phase 1: distance-based, no model changes).

    The distance field is computed by iterative erosion: at each iteration,
    boundary pixels are peeled off. A token's distance = the iteration at
    which it was peeled. Then a sigmoid maps distance → alpha.

    Args:
        token_mask: Boolean token mask [B, seq_len] where True = edit region
        tokens_per_frame: Number of tokens per frame (h * w)
        height: Latent spatial height
        width: Latent spatial width
        falloff_tokens: Number of tokens over which the transition occurs.
            Higher = softer boundary. 3 tokens ≈ 96 pixels of transition.
        sharpness: Sigmoid steepness. Higher = sharper transition within
            the falloff zone. 10.0 gives a clean S-curve.

    Returns:
        Float blend mask [B, seq_len] in [0, 1] where 1.0 = fully denoised
        edit content, 0.0 = fully source content.
    """
    B, seq_len = token_mask.shape
    num_frames = seq_len // tokens_per_frame
    device = token_mask.device

    # Reshape to spatial grid: [B*F, 1, H, W]
    mask_spatial = token_mask.float().view(B, num_frames, height, width)
    mask_spatial = mask_spatial.view(B * num_frames, 1, height, width)

    # Compute signed distance field via iterative erosion/dilation
    # Positive distance = inside mask, negative = outside
    # We compute distance for both interior (erode mask) and exterior (erode ~mask)

    # Interior distance: how far inside the mask boundary
    interior_dist = torch.zeros_like(mask_spatial)
    eroded = mask_spatial.clone()
    for i in range(1, falloff_tokens + 1):
        # Erode by 1 pixel: min-pool with 3x3 kernel
        eroded = -F.max_pool2d(
            -eroded, kernel_size=3, stride=1, padding=1
        )
        interior_dist += eroded  # Still-interior pixels accumulate distance

    # Normalize: interior_dist is in [0, falloff_tokens] for interior pixels
    # Boundary pixels have low interior_dist, deep interior has high

    # For exterior pixels (mask=0), compute exterior distance similarly
    inv_mask = 1.0 - mask_spatial
    exterior_dist = torch.zeros_like(mask_spatial)
    eroded_inv = inv_mask.clone()
    for i in range(1, falloff_tokens + 1):
        eroded_inv = -F.max_pool2d(
            -eroded_inv, kernel_size=3, stride=1, padding=1
        )
        exterior_dist += eroded_inv

    # Signed distance: positive inside, negative outside
    # At boundary: both distances are 0 → signed_dist = 0
    # Deep inside: interior_dist = falloff_tokens → signed_dist = falloff_tokens
    # Deep outside: exterior_dist = falloff_tokens → signed_dist = -falloff_tokens
    signed_dist = interior_dist - exterior_dist  # [B*F, 1, H, W]

    # Apply sigmoid for smooth transition
    # sigmoid(sharpness * signed_dist / falloff_tokens) maps:
    #   deep inside → 1.0, boundary → 0.5, deep outside → 0.0
    if falloff_tokens > 0:
        alpha = torch.sigmoid(sharpness * signed_dist / falloff_tokens)
    else:
        alpha = mask_spatial  # No falloff → hard mask

    # Reshape back to token space: [B, seq_len]
    alpha = alpha.view(B, num_frames, height, width).view(B, seq_len)
    return alpha


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
    height: int | None = None,
    width: int | None = None,
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
        height: Latent spatial height (if None, assumes square)
        width: Latent spatial width (if None, assumes square)

    Returns:
        Boolean token mask [B, seq_len] where True = edit region
    """
    num_frames = seq_len // tokens_per_frame
    if height is not None and width is not None:
        h, w = height, width
    else:
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


# --------------------------------------------------------------------------- #
#  Semantic mask library (MuLAn dataset integration)
# --------------------------------------------------------------------------- #

class SemanticMaskLibrary:
    """Loads and serves pre-extracted instance masks from MuLAn dataset.

    Lazily loads mask files on first use and caches them in memory.
    Supports random selection with area filtering and spatial augmentation.

    Usage:
        lib = SemanticMaskLibrary("/media/2TB/mulan_masks")
        masks = lib.sample_masks(batch_size=4, height=9, width=16, device="cuda")
    """

    def __init__(self, mask_dir: str | None, max_cached: int = 5000):
        self._mask_dir = mask_dir
        self._max_cached = max_cached
        self._masks: list[Tensor] | None = None  # Lazy loaded
        self._areas: list[float] | None = None

    def _load(self) -> None:
        """Load mask library from disk (called once on first sample)."""
        import os
        from pathlib import Path

        if self._masks is not None:
            return
        if self._mask_dir is None:
            self._masks = []
            self._areas = []
            return

        mask_path = Path(self._mask_dir)
        if not mask_path.exists():
            print(f"WARNING: Semantic mask dir not found: {self._mask_dir}")
            self._masks = []
            self._areas = []
            return

        all_masks = []
        all_areas = []

        # Load all .pt files
        pt_files = sorted(mask_path.glob("*.pt"))
        if not pt_files:
            print(f"WARNING: No .pt mask files in {self._mask_dir}")
            self._masks = []
            self._areas = []
            return

        for pt_file in pt_files:
            if pt_file.name == "index.pt":
                continue
            try:
                data = torch.load(pt_file, map_location="cpu", weights_only=True)
                masks = data["masks"]  # [N, H, W]
                areas = data["areas"]  # [N]
                for i in range(masks.shape[0]):
                    all_masks.append(masks[i])
                    all_areas.append(areas[i].item())
                    if len(all_masks) >= self._max_cached:
                        break
            except Exception as e:
                continue
            if len(all_masks) >= self._max_cached:
                break

        self._masks = all_masks
        self._areas = all_areas
        print(f"Loaded {len(self._masks)} semantic masks from {self._mask_dir}")

    @property
    def available(self) -> bool:
        """Whether the mask library has masks available."""
        self._load()
        return len(self._masks) > 0

    def sample_masks(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        min_area: float = 0.05,
        max_area: float = 0.6,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Sample random semantic masks from the library.

        Masks are resized to target resolution, randomly augmented (flip, rotate),
        and broadcast across frames (static masks).

        Args:
            batch_size: Number of masks to generate
            num_frames: Number of video frames
            height: Target spatial height (pixel or latent)
            width: Target spatial width
            min_area: Minimum area fraction filter
            max_area: Maximum area fraction filter
            device: Target device

        Returns:
            Binary mask [B, 1, F, H, W] where 1 = edit region
        """
        self._load()

        # Filter by area range
        valid_indices = [
            i for i, a in enumerate(self._areas)
            if min_area <= a <= max_area
        ]

        if not valid_indices:
            # Fallback: use any mask
            valid_indices = list(range(len(self._masks)))

        if not valid_indices:
            # No masks at all — return random rectangles as fallback
            return generate_random_masks(
                batch_size, num_frames, height, width,
                min_area=min_area, max_area=max_area, device=device,
            )

        masks = torch.zeros(batch_size, 1, num_frames, height, width, device=device)

        for b in range(batch_size):
            idx = random.choice(valid_indices)
            mask_2d = self._masks[idx].float()  # [H_orig, W_orig]

            # Resize to target spatial dims
            mask_2d = F.interpolate(
                mask_2d.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                size=(height, width),
                mode="nearest",
            ).squeeze(0).squeeze(0)  # [H, W]

            # Random augmentation
            if random.random() > 0.5:
                mask_2d = mask_2d.flip(-1)  # Horizontal flip
            if random.random() > 0.5:
                mask_2d = mask_2d.flip(-2)  # Vertical flip
            if random.random() > 0.3:
                # Random 90° rotation
                k = random.choice([1, 2, 3])
                mask_2d = torch.rot90(mask_2d, k, dims=(-2, -1))
                # Re-crop/pad to target size after rotation
                if mask_2d.shape != (height, width):
                    mask_2d = F.interpolate(
                        mask_2d.unsqueeze(0).unsqueeze(0),
                        size=(height, width),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)

            # Binarize after augmentation
            mask_2d = (mask_2d > 0.5).float()

            # Broadcast across frames (static mask)
            masks[b, 0, :] = mask_2d.to(device)

        return masks


def generate_semantic_masks(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    mask_library: SemanticMaskLibrary | None = None,
    min_area: float = 0.05,
    max_area: float = 0.6,
    device: torch.device | str = "cpu",
    mode: str = "mixed",
    semantic_ratio: float = 0.5,
) -> Tensor:
    """Generate edit masks using semantic masks, random masks, or a mix.

    This is the unified mask generation entry point for training. It supports
    three modes:
    - "random": Only random rectangles/ellipses (original behavior)
    - "semantic": Only MuLAn instance masks (requires mask library)
    - "mixed": Random selection of semantic vs random per sample

    Args:
        batch_size: Number of masks to generate
        num_frames: Number of video frames
        height: Pixel height
        width: Pixel width
        mask_library: Pre-loaded SemanticMaskLibrary (None for random-only)
        min_area: Minimum mask area fraction
        max_area: Maximum mask area fraction
        device: Target device
        mode: "random", "semantic", or "mixed"
        semantic_ratio: Fraction of samples using semantic masks in "mixed" mode

    Returns:
        Binary mask [B, 1, F, H, W] where 1 = edit region
    """
    if mode == "random" or mask_library is None or not mask_library.available:
        return generate_random_masks(
            batch_size, num_frames, height, width,
            min_area=min_area, max_area=max_area, device=device,
        )

    if mode == "semantic":
        return mask_library.sample_masks(
            batch_size, num_frames, height, width,
            min_area=min_area, max_area=max_area, device=device,
        )

    if mode == "mixed":
        masks = torch.zeros(batch_size, 1, num_frames, height, width, device=device)
        for b in range(batch_size):
            if random.random() < semantic_ratio:
                # Semantic mask for this sample
                m = mask_library.sample_masks(
                    1, num_frames, height, width,
                    min_area=min_area, max_area=max_area, device=device,
                )
            else:
                # Random mask for this sample
                m = generate_random_masks(
                    1, num_frames, height, width,
                    min_area=min_area, max_area=max_area, device=device,
                )
            masks[b] = m[0]
        return masks

    raise ValueError(f"Unknown mask mode: {mode!r}. Expected 'random', 'semantic', or 'mixed'.")
