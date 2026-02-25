#!/usr/bin/env python3
"""Benchmark vanilla vs SCD vs SCD+DDiT on a 6-second video clip.

Measures actual wall-clock time for the transformer forward passes
(where >95% of compute lives). VAE encode/decode excluded.

6s @ 25fps = 150 frames → 18 latent frames (VAE temporal=8)
At 512×768: H=16, W=24 → seq_len = 18×16×24 = 6,912 tokens
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_positions(nf, h, w, device, dtype=torch.bfloat16):
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(frames=nf, height=h, width=w, batch=1, channels=128),
        device=device,
    )
    pixel_coords = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=SpatioTemporalScaleFactors.default(),
        causal_fix=True,
    ).to(dtype)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / 25.0
    return pixel_coords


def benchmark():
    from dataclasses import replace
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.scd_model import LTXSCDModel, shift_encoder_features
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig
    from train_ddit_adapter import load_base_model

    device = "cuda:0"
    dtype = torch.bfloat16
    K = 20  # denoising steps

    # 10-second clip at 720×1280 (720p)
    nf, h, w = 31, 22, 40
    seq_len = nf * h * w
    tpf = h * w
    C = 128

    print(f"{'='*65}")
    print(f"6-SECOND VIDEO BENCHMARK")
    print(f"{'='*65}")
    print(f"Resolution: 720×1280, 248 frames @ 25fps")
    print(f"Latent: F={nf} H={h} W={w}, seq_len={seq_len:,}")
    print(f"Denoising steps: {K}")
    print()

    # Load model
    print("Loading LTX-2 19B (int8-quanto)...")
    base_model = load_base_model(
        "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors",
        device=device, quantize=True,
    )

    # Create dummy inputs
    latent = torch.randn(1, seq_len, C, device=device, dtype=dtype)
    positions = make_positions(nf, h, w, device, dtype)
    text_ctx = torch.randn(1, 256, 3840, device=device, dtype=dtype)
    text_mask = torch.ones(1, 256, device=device, dtype=torch.int64)
    perturbations = BatchedPerturbationConfig.empty(1)
    sigmas = torch.linspace(1.0, 0.0, K + 1, device=device)

    # =========================================================
    # 1. VANILLA LTX-2 (all 48 layers, every step)
    # =========================================================
    print(f"\n--- VANILLA LTX-2 (48 layers × {K} steps) ---")
    noise = torch.randn_like(latent)
    x = noise.clone()

    # Warmup
    with torch.no_grad():
        ts = torch.full((1, seq_len), 0.5, device=device, dtype=dtype)
        mod = Modality(enabled=True, latent=x, timesteps=ts,
                       positions=positions, context=text_ctx, context_mask=text_mask)
        _ = base_model(video=mod, audio=None, perturbations=perturbations)
    torch.cuda.synchronize()

    x = noise.clone()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
            mod = Modality(enabled=True, latent=x, timesteps=ts,
                           positions=positions, context=text_ctx, context_mask=text_mask)
            vel, _ = base_model(video=mod, audio=None, perturbations=perturbations)
            dt = sigma_next - sigma
            x = x + dt * vel
    torch.cuda.synchronize()
    vanilla_time = time.perf_counter() - t0
    vanilla_per_step = vanilla_time / K
    print(f"  Total: {vanilla_time:.2f}s ({vanilla_per_step:.3f}s/step)")
    vram = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Peak VRAM: {vram:.1f}GB")

    # =========================================================
    # 2. SCD (encoder once + decoder × K)
    # =========================================================
    print(f"\n--- SCD (32 enc × 1 + 16 dec × {K}) ---")
    scd_model = LTXSCDModel(
        base_model=base_model, encoder_layers=32,
        decoder_input_combine="token_concat",
    )

    x = noise.clone()
    torch.cuda.synchronize()
    t0_total = time.perf_counter()

    # Encoder pass (once)
    enc_ts = torch.zeros(1, seq_len, device=device, dtype=dtype)
    enc_mod = Modality(enabled=True, latent=latent, timesteps=enc_ts,
                       positions=positions, context=text_ctx, context_mask=text_mask)
    torch.cuda.synchronize()
    t_enc_start = time.perf_counter()
    with torch.no_grad():
        enc_args, _ = scd_model.forward_encoder(
            video=scd_model._cast_modality_dtype(enc_mod),
            audio=None, perturbations=perturbations,
            tokens_per_frame=tpf,
        )
    torch.cuda.synchronize()
    enc_time = time.perf_counter() - t_enc_start
    shifted = shift_encoder_features(enc_args.x, tpf, nf)

    # Decoder loop (K steps)
    torch.cuda.synchronize()
    t_dec_start = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
            mod = Modality(enabled=True, latent=x, timesteps=ts,
                           positions=positions, context=text_ctx, context_mask=text_mask)
            vel, _ = scd_model.forward_decoder(
                video=scd_model._cast_modality_dtype(mod),
                encoder_features=shifted,
                audio=None, perturbations=perturbations,
                encoder_audio_args=None,
            )
            dt = sigma_next - sigma
            x = x + dt * vel
    torch.cuda.synchronize()
    dec_time = time.perf_counter() - t_dec_start
    scd_time = time.perf_counter() - t0_total
    dec_per_step = dec_time / K
    print(f"  Encoder: {enc_time:.2f}s (once)")
    print(f"  Decoder: {dec_time:.2f}s ({dec_per_step:.3f}s/step)")
    print(f"  Total: {scd_time:.2f}s")
    print(f"  Decoder seq_len with token_concat: {seq_len * 2:,}")
    vram = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Peak VRAM: {vram:.1f}GB")

    # =========================================================
    # 3. SCD with "add" coupling (no sequence doubling)
    # =========================================================
    print(f"\n--- SCD with 'add' coupling (no token_concat) ---")
    scd_add = LTXSCDModel(
        base_model=base_model, encoder_layers=32,
        decoder_input_combine="add",
    )

    x = noise.clone()
    # Encoder (reuse)
    torch.cuda.synchronize()
    t0_total = time.perf_counter()
    with torch.no_grad():
        enc_args_add, _ = scd_add.forward_encoder(
            video=scd_add._cast_modality_dtype(enc_mod),
            audio=None, perturbations=perturbations,
            tokens_per_frame=tpf,
        )
    torch.cuda.synchronize()
    enc_time_add = time.perf_counter() - t0_total
    shifted_add = shift_encoder_features(enc_args_add.x, tpf, nf)

    torch.cuda.synchronize()
    t_dec_start = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
            mod = Modality(enabled=True, latent=x, timesteps=ts,
                           positions=positions, context=text_ctx, context_mask=text_mask)
            vel, _ = scd_add.forward_decoder(
                video=scd_add._cast_modality_dtype(mod),
                encoder_features=shifted_add,
                audio=None, perturbations=perturbations,
                encoder_audio_args=None,
            )
            dt = sigma_next - sigma
            x = x + dt * vel
    torch.cuda.synchronize()
    dec_time_add = time.perf_counter() - t_dec_start
    scd_add_time = enc_time_add + dec_time_add
    dec_per_step_add = dec_time_add / K
    print(f"  Encoder: {enc_time_add:.2f}s (once)")
    print(f"  Decoder: {dec_time_add:.2f}s ({dec_per_step_add:.3f}s/step)")
    print(f"  Total: {scd_add_time:.2f}s")
    print(f"  Decoder seq_len (no doubling): {seq_len:,}")

    # =========================================================
    # 4. SCD add + DDiT (scale=2 and scale=4)
    # =========================================================
    from ltx_core.model.transformer.ddit import DDiTAdapter, DDiTConfig
    from safetensors.torch import load_file

    adapter_path = "outputs/ddit_adapter/ddit_adapter_final.safetensors"
    ddit_config = DDiTConfig(enabled=True, supported_scales=(1, 2, 4))
    adapter = DDiTAdapter(inner_dim=base_model.inner_dim, in_channels=C, config=ddit_config)
    adapter_state = load_file(adapter_path)
    adapter.load_state_dict(adapter_state)
    adapter = adapter.to(device=device, dtype=dtype)
    adapter.eval()
    print(f"\n  DDiT adapter loaded ({sum(p.numel() for p in adapter.parameters()):,} params)")

    ddit_results = {}
    for scale in [2, 4]:
        if h % scale != 0 or w % scale != 0:
            print(f"  Skipping scale={scale}: dims {h}x{w} not divisible")
            continue

        ml = adapter.merge_layers[str(scale)]
        new_h, new_w = h // scale, w // scale
        new_seq = nf * new_h * new_w

        print(f"\n--- SCD add + DDiT scale={scale} (dec seq {seq_len:,}→{new_seq:,}) ---")

        x = noise.clone()
        # Encoder (reuse shifted_add from SCD add test)

        torch.cuda.synchronize()
        t0_total = time.perf_counter()

        # Encoder (same as SCD add)
        with torch.no_grad():
            enc_args_d, _ = scd_add.forward_encoder(
                video=scd_add._cast_modality_dtype(enc_mod),
                audio=None, perturbations=perturbations,
                tokens_per_frame=tpf,
            )
        torch.cuda.synchronize()
        t_enc_done = time.perf_counter()
        enc_t = t_enc_done - t0_total
        shifted_d = shift_encoder_features(enc_args_d.x, tpf, nf)

        # Pool encoder features to match DDiT resolution
        B_enc = shifted_d.shape[0]
        D = shifted_d.shape[-1]
        ef = shifted_d.view(B_enc, nf, h, w, D).permute(0, 1, 4, 2, 3)
        ef = ef.reshape(B_enc * nf, D, h, w)
        ef_pooled = torch.nn.functional.adaptive_avg_pool2d(ef, (new_h, new_w))
        ef_pooled = ef_pooled.reshape(B_enc, nf, D, new_h, new_w)
        ef_pooled = ef_pooled.permute(0, 1, 3, 4, 2).reshape(B_enc, new_seq, D)

        # Decoder loop with DDiT
        torch.cuda.synchronize()
        t_dec_start = time.perf_counter()
        with torch.no_grad():
            for i in range(K):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]

                # 1. Merge spatial tokens
                merged = ml.merge(x, nf, h, w)
                projected = ml.patchify_proj(merged)
                projected = projected + ml.patch_id

                # 2. Positions
                merged_positions = adapter.adjust_positions(positions, scale, nf, h, w)

                # 3. Preprocessor for timestep/RoPE
                ts = torch.full((1, new_seq), sigma.item(), device=device, dtype=dtype)
                dummy_lat = torch.zeros(1, new_seq, C, device=device, dtype=dtype)
                dummy_mod = Modality(enabled=True, latent=dummy_lat, timesteps=ts,
                                     positions=merged_positions, context=text_ctx,
                                     context_mask=text_mask)
                # Cast dtype
                real_model = getattr(base_model, 'base_model', base_model)
                real_model = getattr(real_model, 'model', real_model)
                target_dtype = real_model.patchify_proj.weight.dtype
                if dummy_mod.latent.dtype != target_dtype:
                    dummy_mod = Modality(
                        enabled=dummy_mod.enabled, latent=dummy_mod.latent.to(target_dtype),
                        timesteps=dummy_mod.timesteps, positions=dummy_mod.positions,
                        context=dummy_mod.context, context_mask=dummy_mod.context_mask,
                    )
                video_args = real_model.video_args_preprocessor.prepare(dummy_mod)

                # 4. Swap in projected + add pooled encoder features
                merged_x = projected.to(video_args.x.dtype) + ef_pooled.to(video_args.x.dtype)
                video_args = replace(video_args, x=merged_x)

                # 5. Run decoder blocks only (16 layers)
                for block in scd_add.decoder_blocks:
                    video_args, _ = block(video=video_args, audio=None,
                                          perturbations=perturbations)

                # 6. Output: norm + scale_shift + proj_out
                dec_x = video_args.x
                emb_ts = video_args.embedded_timestep
                ss = real_model.scale_shift_table
                shift, sv = (
                    ss[None, None].to(device=dec_x.device, dtype=dec_x.dtype)
                    + emb_ts[:, :, None]
                ).unbind(dim=2)
                dec_x = real_model.norm_out(dec_x)
                dec_x = dec_x * (1 + sv) + shift
                output_merged = ml.proj_out(dec_x)

                # 7. Unmerge + Euler step
                vel = ml.unmerge(output_merged, nf, h, w)
                if adapter.config.residual_weight > 0:
                    vel = vel + adapter.config.residual_weight * ml.residual_block(x)

                dt = sigma_next - sigma
                x = x + dt * vel

        torch.cuda.synchronize()
        dec_t = time.perf_counter() - t_dec_start
        total_t = enc_t + dec_t
        print(f"  Encoder: {enc_t:.2f}s (once)")
        print(f"  Decoder: {dec_t:.2f}s ({dec_t/K:.3f}s/step, seq={new_seq:,})")
        print(f"  Total: {total_t:.2f}s")
        ddit_results[scale] = {"enc": enc_t, "dec": dec_t, "total": total_t,
                                "dec_per_step": dec_t / K, "dec_seq": new_seq}

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'='*65}")
    print(f"SUMMARY — 10-second video, {K} denoising steps")
    print(f"{'='*65}")
    print(f"{'Config':<45} {'Time':>7} {'Speedup':>8}")
    print(f"{'-'*65}")
    print(f"{'Vanilla (48 layers × K)':<45} {vanilla_time:>6.2f}s {'1.00x':>8}")
    print(f"{'SCD token_concat (32×1 + 16×K)':<45} {scd_time:>6.2f}s {vanilla_time/scd_time:>7.2f}x")
    print(f"{'SCD add (32×1 + 16×K)':<45} {scd_add_time:>6.2f}s {vanilla_time/scd_add_time:>7.2f}x")
    for scale, r in ddit_results.items():
        label = f"SCD add + DDiT {scale}x (32×1 + 16×K @ {r['dec_seq']:,})"
        print(f"{label:<45} {r['total']:>6.2f}s {vanilla_time/r['total']:>7.2f}x")
    print()
    print(f"Per-step decoder cost:")
    print(f"  Vanilla:              {vanilla_per_step:.3f}s  (48 layers, seq={seq_len:,})")
    print(f"  SCD token_concat:     {dec_per_step:.3f}s  (16 layers, seq={seq_len*2:,})")
    print(f"  SCD add:              {dec_per_step_add:.3f}s  (16 layers, seq={seq_len:,})")
    for scale, r in ddit_results.items():
        print(f"  SCD add + DDiT {scale}x:   {r['dec_per_step']:.3f}s  (16 layers, seq={r['dec_seq']:,})")


if __name__ == "__main__":
    benchmark()
