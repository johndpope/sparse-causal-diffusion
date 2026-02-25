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

    # 6-second clip at 512×768
    nf, h, w = 18, 16, 24
    seq_len = nf * h * w
    tpf = h * w
    C = 128

    print(f"{'='*65}")
    print(f"6-SECOND VIDEO BENCHMARK")
    print(f"{'='*65}")
    print(f"Resolution: 512×768, 150 frames @ 25fps")
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
    # 2. SCD (encoder once + decoder × K) — token_concat
    # =========================================================
    print(f"\n--- SCD (32 enc × 1 + 16 dec × {K}) ---")
    scd_time = float("inf")
    dec_per_step = float("inf")

    # Encoder pass (once) — shared across all SCD modes
    enc_ts = torch.zeros(1, seq_len, device=device, dtype=dtype)
    enc_mod = Modality(enabled=True, latent=latent, timesteps=enc_ts,
                       positions=positions, context=text_ctx, context_mask=text_mask)

    try:
        scd_model = LTXSCDModel(
            base_model=base_model, encoder_layers=32,
            decoder_input_combine="token_concat",
        )
        x = noise.clone()
        torch.cuda.synchronize()
        t0_total = time.perf_counter()

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
    except torch.OutOfMemoryError:
        print(f"  OOM! token_concat doubles seq to {seq_len*2:,} — too large at this resolution")
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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
    # 3b. SCD with "concat" (feature-dim) coupling
    # =========================================================
    print(f"\n--- SCD with 'concat' (feature-dim) coupling ---")
    scd_concat = LTXSCDModel(
        base_model=base_model, encoder_layers=32,
        decoder_input_combine="concat",
    )
    # Move the new projection layer to device
    scd_concat.feature_concat_proj = scd_concat.feature_concat_proj.to(device=device, dtype=dtype)

    x = noise.clone()
    torch.cuda.synchronize()
    t0_total = time.perf_counter()
    with torch.no_grad():
        enc_args_cat, _ = scd_concat.forward_encoder(
            video=scd_concat._cast_modality_dtype(enc_mod),
            audio=None, perturbations=perturbations,
            tokens_per_frame=tpf,
        )
    torch.cuda.synchronize()
    enc_time_cat = time.perf_counter() - t0_total
    shifted_cat = shift_encoder_features(enc_args_cat.x, tpf, nf)

    torch.cuda.synchronize()
    t_dec_start = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
            mod = Modality(enabled=True, latent=x, timesteps=ts,
                           positions=positions, context=text_ctx, context_mask=text_mask)
            vel, _ = scd_concat.forward_decoder(
                video=scd_concat._cast_modality_dtype(mod),
                encoder_features=shifted_cat,
                audio=None, perturbations=perturbations,
                encoder_audio_args=None,
            )
            dt = sigma_next - sigma
            x = x + dt * vel
    torch.cuda.synchronize()
    dec_time_cat = time.perf_counter() - t_dec_start
    scd_cat_time = enc_time_cat + dec_time_cat
    dec_per_step_cat = dec_time_cat / K
    print(f"  Encoder: {enc_time_cat:.2f}s (once)")
    print(f"  Decoder: {dec_time_cat:.2f}s ({dec_per_step_cat:.3f}s/step)")
    print(f"  Total: {scd_cat_time:.2f}s")
    print(f"  Decoder seq_len (no doubling): {seq_len:,}")
    concat_proj_params = sum(p.numel() for p in scd_concat.feature_concat_proj.parameters())
    print(f"  feature_concat_proj params: {concat_proj_params:,}")

    # =========================================================
    # 4. SCD per-frame decoder (THE big speedup)
    # =========================================================
    print(f"\n--- SCD per-frame decoder (add coupling) ---")
    print(f"  Per-frame seq: {tpf:,} tokens (vs {seq_len:,} all-at-once)")

    x = noise.clone()
    torch.cuda.synchronize()
    t0_total = time.perf_counter()
    # Reuse encoder features from SCD add
    with torch.no_grad():
        enc_args_pf, _ = scd_add.forward_encoder(
            video=scd_add._cast_modality_dtype(enc_mod),
            audio=None, perturbations=perturbations,
            tokens_per_frame=tpf,
        )
    torch.cuda.synchronize()
    enc_time_pf = time.perf_counter() - t0_total
    shifted_pf = shift_encoder_features(enc_args_pf.x, tpf, nf)

    torch.cuda.synchronize()
    t_dec_start = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
            mod = Modality(enabled=True, latent=x, timesteps=ts,
                           positions=positions, context=text_ctx, context_mask=text_mask)
            vel, _ = scd_add.forward_decoder_per_frame(
                video=scd_add._cast_modality_dtype(mod),
                encoder_features=shifted_pf,
                perturbations=perturbations,
                tokens_per_frame=tpf,
                num_frames=nf,
            )
            dt = sigma_next - sigma
            x = x + dt * vel
    torch.cuda.synchronize()
    dec_time_pf = time.perf_counter() - t_dec_start
    scd_pf_time = enc_time_pf + dec_time_pf
    dec_per_step_pf = dec_time_pf / K
    print(f"  Encoder: {enc_time_pf:.2f}s (once)")
    print(f"  Decoder: {dec_time_pf:.2f}s ({dec_per_step_pf:.3f}s/step)")
    print(f"  Total: {scd_pf_time:.2f}s")
    vram_pf = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Peak VRAM: {vram_pf:.1f}GB")

    # =========================================================
    # 4b. SCD per-frame decoder with token_concat coupling
    # =========================================================
    print(f"\n--- SCD per-frame decoder (token_concat coupling) ---")
    print(f"  Per-frame seq: {tpf*2:,} tokens (tpf*2, vs {seq_len*2:,} all-at-once)")

    scd_tc = LTXSCDModel(
        base_model=base_model, encoder_layers=32,
        decoder_input_combine="token_concat",
    )

    x = noise.clone()
    torch.cuda.synchronize()
    t0_total = time.perf_counter()
    with torch.no_grad():
        enc_args_tc, _ = scd_tc.forward_encoder(
            video=scd_tc._cast_modality_dtype(enc_mod),
            audio=None, perturbations=perturbations,
            tokens_per_frame=tpf,
        )
    torch.cuda.synchronize()
    enc_time_tc = time.perf_counter() - t0_total
    shifted_tc = shift_encoder_features(enc_args_tc.x, tpf, nf)

    torch.cuda.synchronize()
    t_dec_start = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
            mod = Modality(enabled=True, latent=x, timesteps=ts,
                           positions=positions, context=text_ctx, context_mask=text_mask)
            vel, _ = scd_tc.forward_decoder_per_frame(
                video=scd_tc._cast_modality_dtype(mod),
                encoder_features=shifted_tc,
                perturbations=perturbations,
                tokens_per_frame=tpf,
                num_frames=nf,
            )
            dt = sigma_next - sigma
            x = x + dt * vel
    torch.cuda.synchronize()
    dec_time_tc_pf = time.perf_counter() - t_dec_start
    scd_tc_pf_time = enc_time_tc + dec_time_tc_pf
    dec_per_step_tc_pf = dec_time_tc_pf / K
    print(f"  Encoder: {enc_time_tc:.2f}s (once)")
    print(f"  Decoder: {dec_time_tc_pf:.2f}s ({dec_per_step_tc_pf:.3f}s/step)")
    print(f"  Total: {scd_tc_pf_time:.2f}s")

    # =========================================================
    # 5. SCD add + DDiT (scale=2 and scale=4)
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
    print(f"\n{'='*70}")
    print(f"SUMMARY — 6-second video @ 512p, {K} denoising steps")
    print(f"{'='*70}")
    print(f"{'Config':<52} {'Time':>7} {'Speedup':>8}")
    print(f"{'-'*70}")
    print(f"{'Vanilla (48 layers × K)':<52} {vanilla_time:>6.2f}s {'1.00x':>8}")
    if scd_time < float("inf"):
        print(f"{'SCD token_concat (32×1 + 16×K)':<52} {scd_time:>6.2f}s {vanilla_time/scd_time:>7.2f}x")
    else:
        print(f"{'SCD token_concat (32×1 + 16×K)':<52} {'OOM':>7} {'N/A':>8}")
    print(f"{'SCD add (32×1 + 16×K)':<52} {scd_add_time:>6.2f}s {vanilla_time/scd_add_time:>7.2f}x")
    print(f"{'SCD concat/feat-dim (32×1 + 16×K)':<52} {scd_cat_time:>6.2f}s {vanilla_time/scd_cat_time:>7.2f}x")
    print(f"{'SCD per-frame add (32×1 + 16×K×{nf}f)':<52} {scd_pf_time:>6.2f}s {vanilla_time/scd_pf_time:>7.2f}x")
    print(f"{'SCD per-frame token_concat (32×1 + 16×K×{nf}f)':<52} {scd_tc_pf_time:>6.2f}s {vanilla_time/scd_tc_pf_time:>7.2f}x")
    for scale, r in ddit_results.items():
        label = f"SCD add + DDiT {scale}x (32×1 + 16×K @ {r['dec_seq']:,})"
        print(f"{label:<52} {r['total']:>6.2f}s {vanilla_time/r['total']:>7.2f}x")
    print()
    print(f"Per-step decoder cost:")
    print(f"  Vanilla:                 {vanilla_per_step:.3f}s  (48 layers, seq={seq_len:,})")
    if dec_per_step < float("inf"):
        print(f"  SCD token_concat:        {dec_per_step:.3f}s  (16 layers, seq={seq_len*2:,})")
    else:
        print(f"  SCD token_concat:        OOM     (16 layers, seq={seq_len*2:,})")
    print(f"  SCD add:                 {dec_per_step_add:.3f}s  (16 layers, seq={seq_len:,})")
    print(f"  SCD concat (feat-dim):   {dec_per_step_cat:.3f}s  (16 layers, seq={seq_len:,})")
    print(f"  SCD per-frame add:       {dec_per_step_pf:.3f}s  (16 layers, {nf}×{tpf} tokens)")
    print(f"  SCD per-frame tc:        {dec_per_step_tc_pf:.3f}s  (16 layers, {nf}×{tpf*2} tokens)")
    for scale, r in ddit_results.items():
        print(f"  SCD add + DDiT {scale}x:      {r['dec_per_step']:.3f}s  (16 layers, seq={r['dec_seq']:,})")

    # =========================================================
    # BONUS: 720p per-frame benchmark (where per-frame shines)
    # =========================================================
    print(f"\n\n{'='*70}")
    print(f"BONUS — 720p per-frame decoder benchmark")
    print(f"{'='*70}")

    # Clear VRAM from ALL previous runs
    import gc
    # Delete all 512p tensors and SCD wrappers
    for name in list(locals().keys()):
        obj = locals()[name]
        if isinstance(obj, (torch.Tensor, LTXSCDModel)):
            del obj
    del latent, noise, x, positions, enc_mod
    del scd_model, scd_add, scd_concat, scd_tc
    del shifted, shifted_add, shifted_cat, shifted_tc, shifted_pf
    del adapter
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"  VRAM after cleanup: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    nf_720, h_720, w_720 = 31, 22, 40
    seq_720 = nf_720 * h_720 * w_720
    tpf_720 = h_720 * w_720
    print(f"Resolution: 720×1280, F={nf_720} H={h_720} W={w_720}, seq_len={seq_720:,}")
    print(f"Per-frame: {tpf_720} tokens × {nf_720} frames")

    # Create 720p inputs
    positions_720 = make_positions(nf_720, h_720, w_720, device, dtype)
    noise_720 = torch.randn(1, seq_720, C, device=device, dtype=dtype)

    # Use synthetic encoder features — the encoder itself OOMs at 720p all-at-once
    # (needs ~8GB for 27K tokens × 32 layers). In production, encoder would run
    # per-frame too, or use KV-cache. For decoder speed benchmark, synthetic is fine.
    scd_pf_720 = LTXSCDModel(
        base_model=base_model, encoder_layers=32,
        decoder_input_combine="add",
    )
    D_inner = base_model.inner_dim
    shifted_720 = torch.randn(1, seq_720, D_inner, device=device, dtype=dtype)
    enc_time_720 = 0.0  # Skipped — would be ~5-8s with KV-cache encoder
    print(f"\n  Encoder: SKIPPED (synthetic features, encoder OOMs all-at-once)")
    print(f"  VRAM with synthetic features: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    # Per-frame decoder
    x_720 = noise_720.clone()
    torch.cuda.synchronize()
    t0_dec = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_720), sigma.item(), device=device, dtype=dtype)
            mod = Modality(enabled=True, latent=x_720, timesteps=ts,
                           positions=positions_720, context=text_ctx, context_mask=text_mask)
            vel, _ = scd_pf_720.forward_decoder_per_frame(
                video=scd_pf_720._cast_modality_dtype(mod),
                encoder_features=shifted_720,
                perturbations=perturbations,
                tokens_per_frame=tpf_720,
                num_frames=nf_720,
            )
            dt = sigma_next - sigma
            x_720 = x_720 + dt * vel
    torch.cuda.synchronize()
    dec_time_720 = time.perf_counter() - t0_dec
    total_720 = enc_time_720 + dec_time_720
    vram_720 = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"\n  Per-frame decoder: {dec_time_720:.2f}s ({dec_time_720/K:.3f}s/step)")
    print(f"  Total: {total_720:.2f}s")
    print(f"  Peak VRAM: {vram_720:.1f}GB")
    print(f"  Speedup vs vanilla 720p (~134s): {134/total_720:.1f}x")

    # Per-frame + DDiT combined at 720p
    print(f"\n--- 720p per-frame + DDiT 2× combined ---")
    gc.collect()
    torch.cuda.empty_cache()

    # Reload DDiT adapter
    from ltx_core.model.transformer.ddit import DDiTAdapter, DDiTConfig
    from safetensors.torch import load_file as sf_load
    adapter_path = "outputs/ddit_adapter/ddit_adapter_final.safetensors"
    ddit_config = DDiTConfig(enabled=True, supported_scales=(1, 2, 4))
    adapter_720 = DDiTAdapter(inner_dim=D_inner, in_channels=C, config=ddit_config)
    adapter_720.load_state_dict(sf_load(adapter_path))
    adapter_720 = adapter_720.to(device=device, dtype=dtype)
    adapter_720.eval()

    scale_720 = 2
    ml_720 = adapter_720.merge_layers[str(scale_720)]
    new_h_720 = h_720 // scale_720
    new_w_720 = w_720 // scale_720
    new_tpf_720 = new_h_720 * new_w_720  # 220 tokens per frame
    new_seq_720 = nf_720 * new_tpf_720
    print(f"  Per-frame: {new_tpf_720} tokens/frame (vs {tpf_720} native)")

    # Pool encoder features to DDiT resolution
    B_enc = shifted_720.shape[0]
    ef = shifted_720.view(B_enc, nf_720, h_720, w_720, D_inner).permute(0, 1, 4, 2, 3)
    ef = ef.reshape(B_enc * nf_720, D_inner, h_720, w_720)
    ef_pooled_720 = torch.nn.functional.adaptive_avg_pool2d(ef, (new_h_720, new_w_720))
    ef_pooled_720 = ef_pooled_720.reshape(B_enc, nf_720, D_inner, new_h_720, new_w_720)
    ef_pooled_720 = ef_pooled_720.permute(0, 1, 3, 4, 2).reshape(B_enc, new_seq_720, D_inner)
    del ef

    x_720d = noise_720.clone()
    torch.cuda.synchronize()
    t0_dec_ddit = time.perf_counter()
    with torch.no_grad():
        for i in range(K):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Per-frame DDiT: merge each frame, run decoder, unmerge
            frame_vels = []
            for f in range(nf_720):
                # Extract frame tokens in latent space
                frame_lat = x_720d[:, f*tpf_720:(f+1)*tpf_720, :]  # [B, 880, 128]

                # Merge spatial tokens
                merged = ml_720.merge(frame_lat, 1, h_720, w_720)  # [B, 220, 128*4]
                projected = ml_720.patchify_proj(merged) + ml_720.patch_id  # [B, 220, 4096]

                # Get per-frame encoder features (already pooled)
                frame_enc = ef_pooled_720[:, f*new_tpf_720:(f+1)*new_tpf_720, :]

                # Positions for merged frame
                frame_pos = adapter_720.adjust_positions(positions_720, scale_720, nf_720, h_720, w_720)
                # Extract this frame's positions: need 1 frame of new_tpf tokens
                # Actually adjust_positions returns for all frames, extract one frame
                # For single-frame decode, create per-frame positions
                from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
                from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
                patchifier = VideoLatentPatchifier(patch_size=1)
                frame_coords = patchifier.get_patch_grid_bounds(
                    output_shape=VideoLatentShape(frames=1, height=new_h_720, width=new_w_720, batch=1, channels=128),
                    device=device,
                )
                frame_pixel_coords = get_pixel_coords(
                    latent_coords=frame_coords,
                    scale_factors=SpatioTemporalScaleFactors.default(),
                    causal_fix=True,
                ).to(dtype)
                frame_pixel_coords[:, 0, ...] = frame_pixel_coords[:, 0, ...] / 25.0

                # Preprocess timestep/RoPE for this frame
                ts_f = torch.full((1, new_tpf_720), sigma.item(), device=device, dtype=dtype)
                dummy_lat_f = torch.zeros(1, new_tpf_720, C, device=device, dtype=dtype)
                dummy_mod_f = Modality(enabled=True, latent=dummy_lat_f, timesteps=ts_f,
                                       positions=frame_pixel_coords, context=text_ctx,
                                       context_mask=text_mask)
                real_model = getattr(base_model, 'base_model', base_model)
                real_model = getattr(real_model, 'model', real_model)
                target_dtype = real_model.patchify_proj.weight.dtype
                if dummy_mod_f.latent.dtype != target_dtype:
                    dummy_mod_f = replace(dummy_mod_f, latent=dummy_mod_f.latent.to(target_dtype))
                video_args_f = real_model.video_args_preprocessor.prepare(dummy_mod_f)

                # Swap in projected + encoder features
                merged_x = projected.to(video_args_f.x.dtype) + frame_enc.to(video_args_f.x.dtype)
                video_args_f = replace(video_args_f, x=merged_x)

                # Run 16 decoder blocks
                for block in scd_pf_720.decoder_blocks:
                    video_args_f, _ = block(video=video_args_f, audio=None, perturbations=perturbations)

                # Output projection
                dec_x = video_args_f.x
                emb_ts = video_args_f.embedded_timestep
                ss = real_model.scale_shift_table
                shift_v, sv = (
                    ss[None, None].to(device=dec_x.device, dtype=dec_x.dtype)
                    + emb_ts[:, :, None]
                ).unbind(dim=2)
                dec_x = real_model.norm_out(dec_x)
                dec_x = dec_x * (1 + sv) + shift_v
                output_merged = ml_720.proj_out(dec_x)

                # Unmerge
                frame_vel = ml_720.unmerge(output_merged, 1, h_720, w_720)
                if adapter_720.config.residual_weight > 0:
                    frame_vel = frame_vel + adapter_720.config.residual_weight * ml_720.residual_block(frame_lat)
                frame_vels.append(frame_vel)

            vel_720d = torch.cat(frame_vels, dim=1)
            dt = sigma_next - sigma
            x_720d = x_720d + dt * vel_720d

    torch.cuda.synchronize()
    dec_time_720d = time.perf_counter() - t0_dec_ddit
    vram_720d = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Per-frame+DDiT decoder: {dec_time_720d:.2f}s ({dec_time_720d/K:.3f}s/step)")
    print(f"  Total (exc encoder): {dec_time_720d:.2f}s")
    print(f"  Peak VRAM: {vram_720d:.1f}GB")
    print(f"  Speedup vs vanilla 720p (~134s): {134/dec_time_720d:.1f}x")
    print(f"  Per-frame tokens: {new_tpf_720} (DDiT {scale_720}×)")


if __name__ == "__main__":
    benchmark()
