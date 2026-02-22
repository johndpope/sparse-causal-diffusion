"""Boundary blend module for smooth edit transitions (LayerFusion-inspired).

Implements attention-derived soft masks for seamless boundary blending during
EditCtrl + SCD inference. Based on LayerFusion (arXiv:2412.04460) which uses
self-attention sparsity and cross-attention content confidence to create
data-dependent soft alpha masks.

Three blending modes:
- "distance": Pre-computed signed-distance field (Phase 1, no model changes)
- "attention": Attention-derived soft mask (Phase 2, requires weight capture)
- "hybrid": Distance base + attention refinement (Phase 3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BlendConfig:
    """Configuration for boundary blending.

    Attributes:
        mode: Blending mode ("hard", "distance", "attention", "hybrid")
        falloff_tokens: Tokens over which distance-based transition occurs
        sharpness: Sigmoid steepness for distance-based blending
        step_threshold: Fraction of denoising steps before soft blend kicks in
        attn_layers: Which decoder layer indices to capture attention from.
            Default: last 3 decoder layers (good balance of computation/quality).
        eos_token_idx: Index of the EOS token in cross-attention context.
            -1 = last token (typical for Gemma-style text encoders).
        hybrid_distance_weight: Weight of distance mask in hybrid mode (0-1).
    """
    mode: str = "distance"
    falloff_tokens: int = 3
    sharpness: float = 10.0
    step_threshold: float = 0.5
    attn_layers: set[int] | None = None
    eos_token_idx: int = -1
    hybrid_distance_weight: float = 0.3

    def __post_init__(self):
        if self.attn_layers is None and self.mode in ("attention", "hybrid"):
            # Default: capture from last 3 decoder layers
            self.attn_layers = {13, 14, 15}  # 16 decoder layers (32-47), last 3


class BoundaryBlendModule:
    """Manages soft boundary masks for edit region transitions.

    Accumulates attention weights during denoising and computes data-dependent
    soft masks using LayerFusion's sparsity + content formula.

    Usage:
        blend = BoundaryBlendModule(config)
        blend.set_binary_mask(token_mask, tpf, h, w)

        for step in denoising_loop:
            # After decoder forward with captured attention:
            blend.update_attention_maps(self_attn_weights, cross_attn_weights)
            soft_mask = blend.get_blend_mask(step, total_steps)
            # Use soft_mask for blending instead of hard torch.where()

    VRAM usage: ~600MB for 3 layers at 1440-token sequence (fits on both GPUs).
    """

    def __init__(self, config: BlendConfig):
        self.config = config
        self._binary_mask: Tensor | None = None
        self._distance_mask: Tensor | None = None
        self._tokens_per_frame: int = 0
        self._height: int = 0
        self._width: int = 0

        # Accumulated attention weights per denoising step
        self._self_attn_weights: list[Tensor] = []
        self._cross_attn_weights: list[Tensor] = []
        self._attention_mask: Tensor | None = None

    def set_binary_mask(
        self,
        token_mask: Tensor,
        tokens_per_frame: int,
        height: int,
        width: int,
    ) -> None:
        """Initialize with the binary edit mask.

        Pre-computes the distance-based soft mask (used in distance/hybrid modes).
        Call once before the denoising loop.

        Args:
            token_mask: Boolean mask [B, seq_len] (True = edit region)
            tokens_per_frame: Tokens per frame (h * w)
            height: Latent spatial height
            width: Latent spatial width
        """
        self._binary_mask = token_mask
        self._tokens_per_frame = tokens_per_frame
        self._height = height
        self._width = width

        # Pre-compute distance mask for distance/hybrid modes
        if self.config.mode in ("distance", "hybrid"):
            from scd.utils.mask_utils import compute_boundary_blend_mask
            self._distance_mask = compute_boundary_blend_mask(
                token_mask,
                tokens_per_frame=tokens_per_frame,
                height=height,
                width=width,
                falloff_tokens=self.config.falloff_tokens,
                sharpness=self.config.sharpness,
            )

    def update_attention_maps(
        self,
        self_attn_weights: dict[int, Tensor],
        cross_attn_weights: dict[int, Tensor],
    ) -> None:
        """Accumulate attention weights from the current denoising step.

        Called after each decoder forward pass when attention capture is enabled.

        Args:
            self_attn_weights: {layer_idx: attn_weights [B, H, S, S]}
            cross_attn_weights: {layer_idx: attn_weights [B, H, S, ctx_len]}
        """
        if self.config.mode not in ("attention", "hybrid"):
            return

        # Average across captured layers for this step
        if self_attn_weights:
            sa_avg = torch.stack(list(self_attn_weights.values())).mean(0)
            self._self_attn_weights.append(sa_avg)
        if cross_attn_weights:
            ca_avg = torch.stack(list(cross_attn_weights.values())).mean(0)
            self._cross_attn_weights.append(ca_avg)

    def _compute_attention_mask(self) -> Tensor | None:
        """Compute LayerFusion-style soft mask from accumulated attention weights.

        Uses self-attention sparsity (structural edges) multiplied by
        cross-attention content confidence (semantic boundaries) to produce
        a data-dependent soft mask.

        Returns:
            Soft mask [B, seq_len] in [0, 1] or None if not enough data.
        """
        if not self._self_attn_weights or not self._cross_attn_weights:
            return None

        # Average over denoising steps (use later steps which have more signal)
        # Take last 50% of accumulated steps for cleaner signal
        n_steps = len(self._self_attn_weights)
        start = max(0, n_steps // 2)
        sa_mean = torch.stack(self._self_attn_weights[start:]).mean(0)  # [B, H, S, S]
        ca_mean = torch.stack(self._cross_attn_weights[start:]).mean(0)  # [B, H, S, ctx_len]

        # Structure component: self-attention sparsity
        # s_i = 1 / Σ_j(m²_i,j)  — higher = more sparse = less boundary
        # s'_i = 1 - normalize(s)  — invert so boundary = high value
        sa_sq = sa_mean ** 2  # [B, H, S, S]
        sparsity = 1.0 / (sa_sq.sum(dim=-1) + 1e-8)  # [B, H, S]
        sparsity = sparsity.mean(dim=1)  # [B, S] — average across heads

        # Normalize to [0, 1]
        s_min = sparsity.min(dim=-1, keepdim=True).values
        s_max = sparsity.max(dim=-1, keepdim=True).values
        s_norm = (sparsity - s_min) / (s_max - s_min + 1e-8)
        structure = 1.0 - s_norm  # Invert: boundary/dense = high

        # Content component: cross-attention to EOS token
        # c = mean over heads of cross_attn[:, :, :, eos_idx]
        eos_idx = self.config.eos_token_idx  # -1 = last token
        content = ca_mean[:, :, :, eos_idx].mean(dim=1)  # [B, S]

        # Normalize content to [0, 1]
        c_min = content.min(dim=-1, keepdim=True).values
        c_max = content.max(dim=-1, keepdim=True).values
        content_norm = (content - c_min) / (c_max - c_min + 1e-8)

        # Soft mask = structure * content (both high at content-rich areas)
        soft_mask = structure * content_norm  # [B, S]

        # Normalize final mask
        m_min = soft_mask.min(dim=-1, keepdim=True).values
        m_max = soft_mask.max(dim=-1, keepdim=True).values
        soft_mask = (soft_mask - m_min) / (m_max - m_min + 1e-8)

        # Multiply with binary mask to ensure we don't bleed into definitely-unmasked
        # regions — only soften at the boundary
        if self._binary_mask is not None:
            # Dilate binary mask slightly to include boundary region
            from scd.utils.mask_utils import dilate_token_mask
            dilated = dilate_token_mask(
                self._binary_mask,
                self._tokens_per_frame,
                dilation=self.config.falloff_tokens,
                height=self._height,
                width=self._width,
            ).float()
            soft_mask = soft_mask * dilated

        self._attention_mask = soft_mask
        return soft_mask

    def get_blend_mask(self, step: int, total_steps: int) -> Tensor:
        """Get the effective blend mask for the current denoising step.

        Returns a float mask [B, seq_len] in [0, 1] based on the configured mode.
        Hard masking is handled by the pipeline — this only returns the soft component.

        Args:
            step: Current denoising step index
            total_steps: Total number of denoising steps

        Returns:
            Float blend mask [B, seq_len] in [0, 1]
        """
        if self.config.mode == "distance":
            if self._distance_mask is not None:
                return self._distance_mask
            # Fallback to binary if distance not computed
            return self._binary_mask.float() if self._binary_mask is not None else None

        elif self.config.mode == "attention":
            attn_mask = self._compute_attention_mask()
            if attn_mask is not None:
                return attn_mask
            # Fallback to distance if attention not available
            if self._distance_mask is not None:
                return self._distance_mask
            return self._binary_mask.float() if self._binary_mask is not None else None

        elif self.config.mode == "hybrid":
            # Combine distance (base shape) with attention (boundary refinement)
            dist_mask = self._distance_mask
            attn_mask = self._compute_attention_mask()

            if dist_mask is not None and attn_mask is not None:
                w = self.config.hybrid_distance_weight
                return w * dist_mask + (1 - w) * attn_mask
            elif dist_mask is not None:
                return dist_mask
            elif attn_mask is not None:
                return attn_mask
            return self._binary_mask.float() if self._binary_mask is not None else None

        else:
            # "hard" mode — should not reach here, but return binary mask
            return self._binary_mask.float() if self._binary_mask is not None else None

    def reset(self) -> None:
        """Reset accumulated attention weights between runs."""
        self._self_attn_weights.clear()
        self._cross_attn_weights.clear()
        self._attention_mask = None
