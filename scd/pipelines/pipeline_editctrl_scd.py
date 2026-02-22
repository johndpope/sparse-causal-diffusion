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


class EditCtrlSCDPipeline:
    """EditCtrl + SCD pipeline for video editing.

    Workflow:
    1. VAE encode source video → latents
    2. Convert pixel masks → token masks
    3. SCD encoder pass with KV-cache (autoregressive per-frame)
    4. For each frame:
       a. Gather sparse source tokens → LocalContextModule → local_control
       b. GlobalContextEmbedder → global_context (if available)
       c. Initialize: clean at unmasked positions, noise at masked
       d. Denoising loop (N steps):
          - forward_decoder(noisy, enc_features, local_control, global_context)
          - Scheduler step
          - Re-apply clean latents at unmasked positions (inpainting constraint)
       e. Append denoised frame to sequence
    5. VAE decode all frames → output video

    Args:
        scd_model: LTXSCDModel instance
        local_context_module: Trained LocalContextModule
        global_embedder: Trained GlobalContextEmbedder (or None for phase 1)
        vae_encoder: VAE encoder for source video
        vae_decoder: VAE decoder for output video
        scheduler: Noise scheduler (flow matching)
        patchifier: Video latent patchifier
        text_encoder: Text encoder (or cached embeddings)
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
    ):
        self.scd_model = scd_model
        self.local_context_module = local_context_module
        self.global_embedder = global_embedder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.scheduler = scheduler
        self.patchifier = patchifier
        self.text_encoder = text_encoder

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

        Returns:
            EditCtrlSCDOutput with edited video
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

        # 3. Convert pixel masks → token masks
        from scd.utils.mask_utils import pixel_mask_to_token_mask, dilate_token_mask

        token_mask = pixel_mask_to_token_mask(
            edit_masks.to(device=device),
            vae_temporal_factor=8,
            vae_spatial_factor=32,
        )  # [B, seq_len]

        dilated_mask = dilate_token_mask(token_mask, tokens_per_frame, dilation=2)

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

        # Start from pure noise at masked, clean at unmasked
        token_mask_exp = token_mask.unsqueeze(-1).float()
        noisy_latents = source_patchified.clone()
        noisy_latents = torch.where(
            token_mask_exp.bool().expand_as(noisy_latents),
            noise,
            source_patchified,
        )

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
            velocity_pred, _ = self.scd_model.forward_decoder(
                video=decoder_modality,
                encoder_features=shifted_features,
                audio=None,
                perturbations=None,
                local_control=local_control,
                global_context=global_context,
            )

            # Euler step: x_next = x_curr + (sigma_next - sigma) * velocity
            dt = sigma_next - sigma
            noisy_latents = noisy_latents + dt * velocity_pred

            # Re-apply clean latents at unmasked positions (inpainting constraint)
            noisy_latents = torch.where(
                token_mask_exp.bool().expand_as(noisy_latents),
                noisy_latents,
                source_patchified,
            )

        # 8. Unpatchify and VAE decode
        # Reshape patchified [B, seq_len, C] → [B, C, f, h, w]
        edited_latents = noisy_latents.reshape(B, f, h, w, C).permute(0, 4, 1, 2, 3)

        edited_video = self.vae_decoder(edited_latents)

        return EditCtrlSCDOutput(
            video=edited_video,
            latents=edited_latents,
            edit_mask=edit_masks,
        )
