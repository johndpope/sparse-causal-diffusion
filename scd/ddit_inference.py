"""DDiT inference integration for SCD pipeline.

Provides the DDiTInferenceWrapper that plugs into the EditCtrl+SCD pipeline
to enable dynamic patch scheduling during the denoising loop.

The wrapper intercepts the decoder forward pass:
1. Before each step: scheduler determines optimal patch scale
2. If scale > 1: merge tokens, adjust positions/masks
3. Run decoder at reduced resolution (fewer tokens = faster attention)
4. After step: unmerge tokens back to full resolution
5. Record latent for next step's scheduling decision

The encoder ALWAYS runs at native resolution (runs once, needs full detail).
Only the decoder benefits from DDiT (runs N times per frame).

Loading LoRA (optional, improves quality):
    DDiT training jointly optimizes adapter weights AND LoRA on transformer blocks.
    At inference, loading the LoRA weights lets the transformer better handle
    merged tokens. Without LoRA, the adapter still works but quality is lower.

Usage in pipeline:
    ddit_wrapper = DDiTInferenceWrapper(scd_model, ddit_adapter)
    ddit_wrapper.load_lora("outputs/ddit_adapter/ddit_lora_final.safetensors")
    ddit_wrapper.reset()

    for step_idx, sigma in enumerate(sigmas):
        scale = ddit_wrapper.get_scale(noisy_latents, step_idx, ...)
        output = ddit_wrapper.decode_with_ddit(
            noisy_latents, encoder_features, sigma, scale, ...
        )
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from safetensors.torch import load_file

from ltx_core.model.transformer.ddit import DDiTAdapter, DDiTConfig
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.scd_model import LTXSCDModel


class DDiTInferenceWrapper:
    """Wraps SCD model with DDiT dynamic patch scheduling for inference."""

    def __init__(
        self,
        scd_model: LTXSCDModel,
        ddit_adapter: DDiTAdapter,
        verbose: bool = True,
    ):
        self.scd_model = scd_model
        self.adapter = ddit_adapter
        self.verbose = verbose
        self._step_scales: list[int] = []  # Track scale per step for logging
        self._lora_loaded = False

    @classmethod
    def from_checkpoint(
        cls,
        scd_model: LTXSCDModel,
        adapter_path: str,
        lora_path: str | None = None,
        config_path: str | None = None,
        verbose: bool = True,
    ) -> "DDiTInferenceWrapper":
        """Create wrapper from saved checkpoint files.

        Args:
            scd_model: SCD model instance
            adapter_path: Path to ddit_adapter_final.safetensors
            lora_path: Optional path to ddit_lora_final.safetensors
            config_path: Optional path to ddit_config.json (auto-detected if None)
            verbose: Print loading info
        """
        import json
        from pathlib import Path

        adapter_dir = Path(adapter_path).parent

        # Load config
        if config_path is None:
            config_path = adapter_dir / "ddit_config.json"
        if Path(config_path).exists():
            with open(config_path) as f:
                cfg = json.load(f)
            scales = tuple(cfg.get("scales", [2, 4]))
            lora_rank = cfg.get("lora_rank", 32)
        else:
            scales = (2, 4)
            lora_rank = 32

        # Create adapter
        config = DDiTConfig(
            enabled=True,
            supported_scales=(1, *scales),
            lora_rank=lora_rank,
        )
        adapter = DDiTAdapter(
            inner_dim=scd_model.base_model.inner_dim,
            in_channels=128,
            config=config,
        )

        # Load adapter weights
        state = load_file(adapter_path)
        adapter.load_state_dict(state)
        device = next(scd_model.base_model.parameters()).device
        adapter = adapter.to(device=device, dtype=torch.bfloat16)
        if verbose:
            print(f"DDiT adapter loaded from {adapter_path}")

        wrapper = cls(scd_model, adapter, verbose=verbose)

        # Load LoRA if available
        if lora_path is None:
            lora_path = adapter_dir / "ddit_lora_final.safetensors"
        if Path(lora_path).exists():
            wrapper.load_lora(str(lora_path))

        return wrapper

    def load_lora(self, lora_path: str) -> None:
        """Load LoRA weights into the SCD model's base transformer.

        Applies PEFT LoRA to the base model and loads trained weights.
        This improves DDiT quality by letting the transformer adapt its
        attention/FF patterns for merged tokens.
        """
        from peft import LoraConfig, get_peft_model

        base_model = self.scd_model.base_model
        lora_state = load_file(lora_path)

        # Determine LoRA config from saved weights
        # Find rank from lora_A shape
        rank = 32  # default
        for k, v in lora_state.items():
            if 'lora_A' in k:
                rank = v.shape[0]
                break

        # Determine targets from saved keys
        targets = set()
        for k in lora_state.keys():
            # Key format: base_model.model.transformer_blocks.0.attn1.to_q.lora_A.default.weight
            parts = k.split('.')
            for i, part in enumerate(parts):
                if part in ('to_q', 'to_k', 'to_v'):
                    targets.add(part)
                elif part == 'to_out' and i + 1 < len(parts) and parts[i + 1] == '0':
                    targets.add('to_out.0')
                elif part == 'net' and i + 1 < len(parts):
                    if parts[i + 1] == '0' and i + 2 < len(parts) and parts[i + 2] == 'proj':
                        targets.add('net.0.proj')
                    elif parts[i + 1] == '2':
                        targets.add('net.2')

        if not targets:
            targets = {"to_q", "to_k", "to_v", "to_out.0"}

        # Apply PEFT LoRA
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=list(targets),
            lora_dropout=0.0,
        )

        # Freeze base model first
        for p in base_model.parameters():
            p.requires_grad_(False)

        peft_model = get_peft_model(base_model, config)

        # Load trained LoRA weights
        missing, unexpected = peft_model.load_state_dict(lora_state, strict=False)
        loaded = len(lora_state) - len(missing)

        # Replace base model reference in SCD model
        self.scd_model.base_model = peft_model
        peft_model.eval()
        self._lora_loaded = True

        if self.verbose:
            print(f"DDiT LoRA loaded: {loaded} tensors, rank={rank}, targets={targets}")

    def reset(self) -> None:
        """Reset for a new generation. Call before each video."""
        self.adapter.scheduler.reset()
        self._step_scales = []

    def get_scale(
        self,
        latent: Tensor,
        step_idx: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> int:
        """Determine optimal patch scale for this denoising step.

        Args:
            latent: Current noisy latent [B, F*H*W, C]
            step_idx: Denoising step index (0 = noisiest)
            num_frames, height, width: spatial dims

        Returns:
            Patch scale (1, 2, or 4)
        """
        # Record latent for trajectory analysis
        self.adapter.scheduler.record(latent)

        # Compute schedule
        scale = self.adapter.scheduler.compute_schedule(
            latent, step_idx, num_frames, height, width
        )

        # Validate dimensions
        if height % scale != 0 or width % scale != 0:
            scale = 1  # Fallback

        self._step_scales.append(scale)

        if self.verbose and step_idx < 5 or scale != 1:
            print(f"  DDiT step {step_idx}: scale={scale}x "
                  f"(seq_len {num_frames*height*width} → {num_frames*(height//scale)*(width//scale)})")

        return scale

    def decode_with_ddit(
        self,
        video_modality: Modality,
        encoder_features: Tensor,
        scale: int,
        num_frames: int,
        height: int,
        width: int,
        audio: Modality | None = None,
        perturbations: Any = None,
        encoder_audio_args: Any = None,
        local_control: Tensor | None = None,
        global_context: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Run SCD decoder with DDiT dynamic patching.

        At scale=1: passes through to normal SCD decoder (no change).
        At scale>1: merges tokens before decoder, unmerges after.

        Returns:
            Same as scd_model.forward_decoder: (video_pred, audio_pred)
        """
        if scale == 1:
            # Native resolution — standard SCD decoder
            return self.scd_model.forward_decoder(
                video=video_modality,
                encoder_features=encoder_features,
                audio=audio,
                perturbations=perturbations,
                encoder_audio_args=encoder_audio_args,
                local_control=local_control,
                global_context=global_context,
            )

        # --- DDiT coarse resolution path ---
        merge_layer = self.adapter.merge_layers[str(scale)]
        input_latent = video_modality.latent  # [B, seq_len, C] original

        # 1. Merge spatial tokens in latent space
        merged_latent = merge_layer.merge(
            input_latent, num_frames, height, width
        )  # [B, new_seq_len, C*s*s]

        # We need to project merged_latent through the merge layer's patchify_proj
        # BUT the SCD model's forward_decoder calls base_model.video_args_preprocessor.prepare()
        # which applies patchify_proj internally. We need to bypass that.

        # 2. Project merged tokens through DDiT's patchify layer
        merged_projected = merge_layer.patchify_proj(merged_latent)  # [B, new_seq_len, inner_dim]
        merged_projected = merged_projected + merge_layer.patch_id  # Add scale identifier

        # 3. Adjust positions for coarser grid
        merged_positions = self.adapter.adjust_positions(
            video_modality.positions, scale, num_frames, height, width
        )

        # 4. Adjust timesteps
        new_seq_len = merged_projected.shape[1]
        B = merged_projected.shape[0]
        if video_modality.timesteps.shape[-1] == input_latent.shape[1]:
            # Per-token timesteps — need to reduce
            merged_timesteps = video_modality.timesteps[:, :new_seq_len]
        else:
            merged_timesteps = video_modality.timesteps

        # 5. Create merged modality for the preprocessor
        # We need a dummy modality just to get timestep/context processing
        # Then we'll swap in our merged tokens
        dummy_latent = torch.zeros(
            B, new_seq_len, self.scd_model.base_model.patchify_proj.in_features,
            device=input_latent.device, dtype=input_latent.dtype,
        )
        merged_modality = Modality(
            enabled=True,
            latent=dummy_latent,
            timesteps=merged_timesteps,
            positions=merged_positions,
            context=video_modality.context,
            context_mask=video_modality.context_mask,
        )

        # 6. Merge encoder features too (they'll be token-concatenated)
        if encoder_features is not None:
            merged_enc_features = merge_layer.merge(
                encoder_features, num_frames, height, width
            )
            merged_enc_features = merge_layer.patchify_proj(merged_enc_features)
        else:
            merged_enc_features = None

        # 7. Merge local control if present
        merged_local_control = None
        if local_control is not None:
            # local_control is [B, seq_len, D] in inner_dim space
            # Need to spatially pool it to the coarser grid
            lc = local_control.view(B, num_frames, height, width, -1)
            new_h, new_w = height // scale, width // scale
            lc = lc.permute(0, 1, 4, 2, 3)  # [B, F, D, H, W]
            lc = lc.reshape(B * num_frames, -1, height, width)
            lc_pooled = torch.nn.functional.adaptive_avg_pool2d(
                lc, output_size=(new_h, new_w)
            )
            lc_pooled = lc_pooled.reshape(B, num_frames, -1, new_h, new_w)
            lc_pooled = lc_pooled.permute(0, 1, 3, 4, 2)  # [B, F, h, w, D]
            merged_local_control = lc_pooled.reshape(B, new_seq_len, -1)

        # 8. Run decoder through preprocessor to get text embeddings etc.
        video_args = self.scd_model.base_model.video_args_preprocessor.prepare(
            self.scd_model._cast_modality_dtype(merged_modality)
        )

        # 9. Swap in our DDiT-projected tokens (bypass base patchify_proj)
        video_args = replace(video_args, x=merged_projected)

        # 10. Adjust causal mask for new sequence length
        new_tpf = (height // scale) * (width // scale)
        from ltx_core.model.transformer.scd_model import build_frame_causal_mask
        merged_mask = build_frame_causal_mask(
            seq_len=new_seq_len,
            tokens_per_frame=new_tpf,
            device=merged_projected.device,
            dtype=merged_projected.dtype,
        )
        video_args = replace(video_args, self_attn_mask=merged_mask)

        # 11. Combine encoder features with decoder tokens (token_concat)
        if merged_enc_features is not None:
            video_args = self.scd_model._combine_encoder_decoder(video_args, merged_enc_features)

        # 12. Inject local control and global context
        if merged_local_control is not None:
            if self.scd_model.local_control_injection == "pre_decoder":
                if merged_enc_features is not None:
                    enc_seq = merged_enc_features.shape[1]
                    pad = merged_local_control.new_zeros(B, enc_seq, merged_local_control.shape[2])
                    lc_padded = torch.cat([pad, merged_local_control], dim=1)
                else:
                    lc_padded = merged_local_control
                video_args = replace(video_args, x=video_args.x + lc_padded)

        if global_context is not None:
            new_context = torch.cat([global_context, video_args.context], dim=1)
            gc_mask = torch.ones(
                B, global_context.shape[1],
                device=video_args.context_mask.device,
                dtype=video_args.context_mask.dtype,
            )
            new_ctx_mask = torch.cat([gc_mask, video_args.context_mask], dim=1)
            video_args = replace(video_args, context=new_context, context_mask=new_ctx_mask)

        # 13. Run decoder blocks
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(B)

        audio_args = encoder_audio_args

        # Determine per-layer injection
        inject_layers = set()
        if (self.scd_model.local_control_injection == "per_layer"
                and merged_local_control is not None):
            if self.scd_model.local_control_layers is not None:
                inject_layers = set(self.scd_model.local_control_layers)
            else:
                inject_layers = set(range(len(self.scd_model.decoder_blocks)))

        lc_padded_for_inject = None
        if inject_layers and merged_local_control is not None:
            if merged_enc_features is not None:
                enc_seq = merged_enc_features.shape[1]
                pad = merged_local_control.new_zeros(B, enc_seq, merged_local_control.shape[2])
                lc_padded_for_inject = torch.cat([pad, merged_local_control], dim=1)
            else:
                lc_padded_for_inject = merged_local_control

        for i, block in enumerate(self.scd_model.decoder_blocks):
            video_args, audio_args = block(
                video=video_args,
                audio=audio_args,
                perturbations=perturbations,
            )
            if i in inject_layers and lc_padded_for_inject is not None:
                video_args = replace(video_args, x=video_args.x + lc_padded_for_inject)

        # 14. Process output
        decoder_x = video_args.x
        decoder_emb_ts = video_args.embedded_timestep
        if self.scd_model.decoder_input_combine in ("token_concat", "token_concat_with_proj"):
            if merged_enc_features is not None:
                enc_seq_len = merged_enc_features.shape[1]
                decoder_x = decoder_x[:, enc_seq_len:]
                decoder_emb_ts = decoder_emb_ts[:, enc_seq_len:]

        # 15. Use DDiT's proj_out instead of base
        # Apply scale-shift modulation (same as base)
        scale_shift = self.scd_model.base_model.scale_shift_table
        shift, scale_val = (
            scale_shift[None, None].to(device=decoder_x.device, dtype=decoder_x.dtype)
            + decoder_emb_ts[:, :, None]
        ).unbind(dim=2)
        decoder_x = self.scd_model.base_model.norm_out(decoder_x)
        decoder_x = decoder_x * (1 + scale_val) + shift

        # Project through DDiT's merge layer proj_out
        merged_output = merge_layer.proj_out(decoder_x)  # [B, new_seq_len, C*s*s]

        # 16. Unmerge back to full resolution
        vx = merge_layer.unmerge(merged_output, num_frames, height, width)  # [B, F*H*W, C]

        # Residual refinement
        if self.adapter.config.residual_weight > 0:
            residual = merge_layer.residual_block(input_latent)
            vx = vx + self.adapter.config.residual_weight * residual

        return vx, None

    def get_speedup_stats(self) -> dict:
        """Return statistics about DDiT scheduling decisions."""
        if not self._step_scales:
            return {}

        total_steps = len(self._step_scales)
        scale_counts = {}
        for s in self._step_scales:
            scale_counts[s] = scale_counts.get(s, 0) + 1

        # Estimate speedup (attention is O(N^2))
        # scale=1: 1.0x, scale=2: ~3x (N/4 tokens), scale=4: ~4x (N/16 tokens)
        speedup_map = {1: 1.0, 2: 3.0, 4: 4.5}
        total_compute = sum(1 / speedup_map.get(s, 1.0) for s in self._step_scales)
        effective_speedup = total_steps / total_compute

        return {
            "total_steps": total_steps,
            "scale_distribution": scale_counts,
            "estimated_speedup": f"{effective_speedup:.2f}x",
            "coarse_ratio": f"{(total_steps - scale_counts.get(1, 0)) / total_steps:.1%}",
        }
