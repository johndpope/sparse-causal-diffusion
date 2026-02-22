"""EditCtrl + SCD inference pipeline for video editing.

Given a source video, binary edit masks, and text prompts, generates an
edited video where only the masked regions change according to the prompt.

The pipeline combines SCD's autoregressive encoder with EditCtrl's
local/global context modules for efficient, high-quality video editing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


@dataclass
class EditCtrlSCDOutput:
    """Output of the EditCtrl + SCD inference pipeline."""
    video: Tensor          # [B, C, F, H, W] in pixel space [-1, 1]
    latents: Tensor        # [B, C, F, H, W] in latent space
    edit_mask: Tensor      # [B, 1, F, H, W] pixel mask used
    blend_mask: Tensor | None = None  # [B, seq_len] soft blend mask (for debugging)


class EditCtrlSCDPipeline:
    """EditCtrl + SCD pipeline for video editing.

    Workflow:
    1. VAE encode source video → latents
    2. Convert pixel masks → token masks
    3. (Optional) TMA: extract Qwen VL features → prepend to text context
    4. SCD encoder pass with KV-cache (autoregressive per-frame)
    5. For each frame:
       a. Gather sparse source tokens → LocalContextModule → local_control
       b. GlobalContextEmbedder → global_context (if available)
       c. Initialize: clean at unmasked positions, noise at masked
       d. Denoising loop (N steps):
          - forward_decoder(noisy, enc_features, local_control, global_context)
          - Scheduler step
          - Re-apply clean latents at unmasked positions (inpainting constraint)
       e. Append denoised frame to sequence
    6. VAE decode all frames → output video

    Args:
        scd_model: LTXSCDModel instance
        local_context_module: Trained LocalContextModule
        global_embedder: Trained GlobalContextEmbedder (or None for phase 1)
        vae_encoder: VAE encoder for source video
        vae_decoder: VAE decoder for output video
        scheduler: Noise scheduler (flow matching)
        patchifier: Video latent patchifier
        text_encoder: Text encoder (or cached embeddings)
        tma_module: Trained TMA module (or None if not using TMA)
    """

    def __init__(
        self,
        scd_model: nn.Module,
        local_context_module: nn.Module,
        global_embedder: nn.Module | None,
        vae_encoder: nn.Module,
        vae_decoder: nn.Module,
        scheduler: object,
        patchifier: object,
        text_encoder: nn.Module | None = None,
        tma_module: nn.Module | None = None,
    ):
        self.scd_model = scd_model
        self.local_context_module = local_context_module
        self.global_embedder = global_embedder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.scheduler = scheduler
        self.patchifier = patchifier
        self.text_encoder = text_encoder
        self.tma_module = tma_module

    @torch.inference_mode()
    def __call__(
        self,
        source_video: Tensor,
        edit_masks: Tensor,
        text_context: Tensor,
        text_mask: Tensor,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        seed: int = 42,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        qwen_features: Tensor | None = None,
        boundary_blend_mode: str = "hard",
        boundary_falloff: int = 3,
        boundary_sharpness: float = 10.0,
        boundary_step_threshold: float = 0.5,
    ) -> EditCtrlSCDOutput:
        """Run video editing inference.

        Args:
            source_video: Source video tensor [B, C, F, H, W] in pixel space [-1, 1]
            edit_masks: Binary pixel masks [B, 1, F, H, W] (1 = edit region)
            text_context: Pre-computed text embeddings [B, ctx_len, D]
            text_mask: Text attention mask [B, ctx_len]
            num_inference_steps: Number of denoising steps per frame
            guidance_scale: CFG guidance scale
            seed: Random seed
            device: Target device
            dtype: Target dtype
            qwen_features: Pre-computed Qwen VL features [B, seq_len, hidden_dim] (optional)
            boundary_blend_mode: Blending mode at mask boundaries.
                "hard": Original binary torch.where (default, bit-identical regression)
                "distance": Signed-distance soft boundary (Phase 1, no model changes)
                "attention": Attention-derived soft mask (Phase 2, requires capture)
                "hybrid": Distance + attention combined (Phase 3)
            boundary_falloff: Number of tokens for boundary transition (default 3)
            boundary_sharpness: Sigmoid steepness for boundary (default 10.0)
            boundary_step_threshold: Fraction of steps before switching to soft blend.
                Early steps (high noise) use hard masking; after this fraction, soft
                blending ramps in via cosine schedule. (default 0.5 = halfway)

        Returns:
            EditCtrlSCDOutput with edited video and optional blend_mask
        """
        generator = torch.Generator(device=device).manual_seed(seed)

        # 1. VAE encode source video → latents
        source_latents = self.vae_encoder(
            source_video.to(device=device, dtype=dtype)
        )  # [B, C, f, h, w]

        B, C, f, h, w = source_latents.shape
        tokens_per_frame = h * w

        # 2. Patchify latents
        source_patchified = self.patchifier.patchify(source_latents)  # [B, seq_len, C]
        seq_len = source_patchified.shape[1]

        # 2.5. TMA: prepend Qwen VL semantic tokens to text context
        if self.tma_module is not None and qwen_features is not None:
            qwen_features = qwen_features.to(device=device, dtype=dtype)
            task_indices = torch.zeros(B, dtype=torch.long, device=device)
            tma_context = self.tma_module(qwen_features, task_indices)  # [B, num_queries, output_dim]
            text_context = torch.cat([tma_context, text_context.to(device=device, dtype=dtype)], dim=1)
            tma_mask = torch.ones(B, tma_context.shape[1], dtype=text_mask.dtype, device=device)
            text_mask = torch.cat([tma_mask, text_mask.to(device=device)], dim=1)

        # 3. Convert pixel masks → token masks
        from scd.utils.mask_utils import (
            pixel_mask_to_token_mask, dilate_token_mask,
            compute_boundary_blend_mask,
        )

        token_mask = pixel_mask_to_token_mask(
            edit_masks.to(device=device),
            vae_temporal_factor=8,
            vae_spatial_factor=32,
        )  # [B, seq_len]

        dilated_mask = dilate_token_mask(token_mask, tokens_per_frame, dilation=2, height=h, width=w)

        # Pre-compute soft boundary blend mask (used when boundary_blend_mode != "hard")
        soft_blend_mask = None
        if boundary_blend_mode in ("distance", "hybrid"):
            soft_blend_mask = compute_boundary_blend_mask(
                token_mask,
                tokens_per_frame=tokens_per_frame,
                height=h,
                width=w,
                falloff_tokens=boundary_falloff,
                sharpness=boundary_sharpness,
            ).to(dtype=dtype)  # [B, seq_len] float in [0,1]

        # 4. SCD encoder pass (full sequence, autoregressive via causal mask)
        from ltx_core.model.transformer.modality import Modality

        # Build video positions
        from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
        from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

        patchifier = VideoLatentPatchifier(patch_size=1)
        latent_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(frames=f, height=h, width=w, batch=B, channels=C),
            device=device,
        )
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype)
        pixel_coords[:, 0] = pixel_coords[:, 0] / 25.0  # FPS normalization

        encoder_timesteps = torch.zeros(B, seq_len, device=device, dtype=dtype)

        encoder_modality = Modality(
            enabled=True,
            latent=source_patchified,
            timesteps=encoder_timesteps,
            positions=pixel_coords,
            context=text_context.to(device=device, dtype=dtype),
            context_mask=text_mask.to(device=device, dtype=dtype),
        )

        encoder_video_args, _ = self.scd_model.forward_encoder(
            video=encoder_modality,
            audio=None,
            perturbations=None,
            tokens_per_frame=tokens_per_frame,
        )

        from ltx_core.model.transformer.scd_model import shift_encoder_features
        encoder_features = encoder_video_args.x
        shifted_features = shift_encoder_features(encoder_features, tokens_per_frame, f)

        # 5. Prepare LocalContextModule inputs
        from scd.utils.mask_utils import gather_masked_tokens, prepare_background_latents

        sparse_tokens, _ = gather_masked_tokens(source_patchified, dilated_mask)

        # Get timestep embedding (will be updated per denoising step)
        # For now, prepare once with sigma=1.0 as placeholder
        adaln = self.scd_model.base_model.adaln_single
        placeholder_sigma = torch.ones(B, 1, device=device, dtype=dtype)
        timestep_emb = adaln(placeholder_sigma, None)
        if timestep_emb.dim() == 3 and timestep_emb.shape[1] > 1:
            timestep_emb = timestep_emb[:, :1, :]

        local_control = self.local_context_module(
            source_tokens=sparse_tokens,
            mask_indices=dilated_mask,
            text_context=text_context.to(device=device, dtype=dtype),
            text_mask=text_mask.to(device=device, dtype=dtype),
            timestep_emb=timestep_emb,
            seq_len=seq_len,
        )

        # 6. Prepare GlobalContextEmbedder inputs (if available)
        global_context = None
        if self.global_embedder is not None:
            bg_tokens = prepare_background_latents(source_patchified, token_mask)
            global_context = self.global_embedder(bg_tokens)

        # 7. Denoising loop
        # Initialize: noise at masked positions, clean at unmasked
        noise = torch.randn(
            B, seq_len, C, device=device, dtype=dtype, generator=generator
        )

        # Build sigma schedule
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

        # Compute the step index at which soft blending kicks in
        blend_start_step = int(num_inference_steps * boundary_step_threshold)

        # Start from pure noise at masked, clean at unmasked
        # Initial injection always uses hard mask (noise must not leak into source)
        token_mask_exp = token_mask.unsqueeze(-1).float()
        noisy_latents = source_patchified.clone()
        noisy_latents = torch.where(
            token_mask_exp.bool().expand_as(noisy_latents),
            noise,
            source_patchified,
        )

        # Optional: BoundaryBlendModule for attention-based blending (Phase 2)
        blend_module = None
        if boundary_blend_mode in ("attention", "hybrid"):
            from scd.utils.boundary_blend import BoundaryBlendModule, BlendConfig
            blend_config = BlendConfig(
                mode=boundary_blend_mode,
                falloff_tokens=boundary_falloff,
                sharpness=boundary_sharpness,
                step_threshold=boundary_step_threshold,
            )
            blend_module = BoundaryBlendModule(blend_config)
            blend_module.set_binary_mask(token_mask, tokens_per_frame, h, w)

        for i in tqdm(range(num_inference_steps), desc="Denoising"):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Build decoder timesteps (sigma at all positions)
            decoder_timesteps = torch.full(
                (B, seq_len), sigma.item(), device=device, dtype=dtype
            )

            decoder_modality = Modality(
                enabled=True,
                latent=noisy_latents,
                timesteps=decoder_timesteps,
                positions=pixel_coords,
                context=text_context.to(device=device, dtype=dtype),
                context_mask=text_mask.to(device=device, dtype=dtype),
            )

            # Forward decoder with EditCtrl signals
            # Phase 2: optionally capture attention weights
            capture_layers = None
            if blend_module is not None:
                capture_layers = blend_module.config.attn_layers

            velocity_pred, _ = self.scd_model.forward_decoder(
                video=decoder_modality,
                encoder_features=shifted_features,
                audio=None,
                perturbations=None,
                local_control=local_control,
                global_context=global_context,
                capture_attention_layers=capture_layers,
            )

            # If decoder returned captured attention, update blend module
            if blend_module is not None and hasattr(velocity_pred, '__len__') is False:
                # velocity_pred is just the tensor; attention captured via model hooks
                pass

            # Euler step: x_next = x_curr + (sigma_next - sigma) * velocity
            dt = sigma_next - sigma
            noisy_latents = noisy_latents + dt * velocity_pred

            # Re-apply source constraint at unmasked/boundary positions
            if boundary_blend_mode == "hard":
                # Original behavior: hard binary swap (bit-identical regression)
                noisy_latents = torch.where(
                    token_mask_exp.bool().expand_as(noisy_latents),
                    noisy_latents,
                    source_patchified,
                )
            else:
                # Soft blending with step-dependent schedule
                if i < blend_start_step:
                    # Early steps: hard masking (noise is high, need strict constraint)
                    noisy_latents = torch.where(
                        token_mask_exp.bool().expand_as(noisy_latents),
                        noisy_latents,
                        source_patchified,
                    )
                else:
                    # After threshold: soft blending with cosine ramp
                    # t ramps from 0 (at blend_start_step) to 1 (at final step)
                    remaining = num_inference_steps - blend_start_step
                    t = (i - blend_start_step) / max(remaining, 1)
                    # Cosine ramp: starts gentle, ends with full soft blend
                    blend_strength = 0.5 * (1 - math.cos(math.pi * t))

                    # Get the appropriate soft mask
                    if blend_module is not None:
                        current_blend = blend_module.get_blend_mask(
                            i, num_inference_steps
                        ).to(dtype=dtype)
                    elif soft_blend_mask is not None:
                        current_blend = soft_blend_mask
                    else:
                        current_blend = token_mask.float()

                    # Interpolate between hard mask and soft mask based on blend_strength
                    hard_mask = token_mask.float()
                    effective_mask = (
                        (1 - blend_strength) * hard_mask
                        + blend_strength * current_blend
                    )  # [B, seq_len]

                    # Apply: lerp between source and denoised based on effective_mask
                    effective_mask_exp = effective_mask.unsqueeze(-1)  # [B, seq_len, 1]
                    noisy_latents = (
                        effective_mask_exp * noisy_latents
                        + (1 - effective_mask_exp) * source_patchified
                    )

        # 8. Unpatchify and VAE decode
        # Reshape patchified [B, seq_len, C] → [B, C, f, h, w]
        edited_latents = noisy_latents.reshape(B, f, h, w, C).permute(0, 4, 1, 2, 3)

        edited_video = self.vae_decoder(edited_latents)

        # Collect blend mask for debugging/visualization
        output_blend_mask = None
        if soft_blend_mask is not None:
            output_blend_mask = soft_blend_mask
        elif blend_module is not None:
            output_blend_mask = blend_module.get_blend_mask(
                num_inference_steps - 1, num_inference_steps
            )

        return EditCtrlSCDOutput(
            video=edited_video,
            latents=edited_latents,
            edit_mask=edit_masks,
            blend_mask=output_blend_mask,
        )
