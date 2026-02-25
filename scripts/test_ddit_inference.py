#!/usr/bin/env python3
"""Test DDiT (Dynamic Patch Scheduling) inference quality and speed.

Compares native full-resolution decoding vs DDiT at scale 2 and 4:
  1. Loads SCD model (LTX-2 19B, int8-quanto quantized)
  2. Loads DDiT adapter + LoRA from training checkpoints
  3. Runs encoder once on a test sample
  4. Runs decoder at native, scale=2, scale=4
  5. Reports MSE, cosine similarity, and speed

Usage:
    python scripts/test_ddit_inference.py
    python scripts/test_ddit_inference.py --device cuda:1 --steps 10
    python scripts/test_ddit_inference.py --full_loop  # Full denoising, not single step
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_scd_model(base_model, encoder_layers=32):
    """Wrap a base LTXModel into an SCD model."""
    from ltx_core.model.transformer.scd_model import LTXSCDModel
    scd = LTXSCDModel(
        base_model=base_model,
        encoder_layers=encoder_layers,
        decoder_input_combine="token_concat",
    )
    return scd


def make_positions(nf, h, w, device, dtype=torch.bfloat16):
    """Create positions in LTX-2 format: [1, 3, seq_len, 2]."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            frames=nf, height=h, width=w, batch=1, channels=128,
        ),
        device=device,
    )
    pixel_coords = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=SpatioTemporalScaleFactors.default(),
        causal_fix=True,
    ).to(dtype)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / 24.0
    return pixel_coords


def run_standalone_comparison(base_model, adapter, sample, device, sigma=0.5):
    """Test DDiT matching training exactly — all 48 blocks, no SCD split.

    This verifies the adapter actually works before trying the SCD split.
    Uses the same forward pass as get_teacher_output/get_student_output from training.
    """
    from dataclasses import replace
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    dtype = torch.bfloat16
    latent = sample["latents"].to(device=device, dtype=dtype)
    nf, h, w = sample["num_frames"], sample["height"], sample["width"]
    seq_len = nf * h * w
    latent_seq = latent.permute(1, 2, 3, 0).reshape(1, seq_len, -1)
    positions = make_positions(nf, h, w, device, dtype)

    if sample["text_embeds"] is not None:
        text_ctx = sample["text_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
        text_mask = sample["text_mask"].unsqueeze(0).to(device=device)
    else:
        text_ctx = torch.zeros(1, 256, 3840, device=device, dtype=dtype)
        text_mask = torch.ones(1, 256, device=device, dtype=torch.int64)

    # Get the real underlying model (unwrap PEFT if present)
    real_model = getattr(base_model, 'base_model', base_model)
    real_model = getattr(real_model, 'model', real_model)

    # Add noise
    noise = torch.randn_like(latent_seq)
    noisy = (1 - sigma) * latent_seq + sigma * noise
    timesteps = torch.full((1, seq_len), sigma, device=device, dtype=dtype)

    results = {}

    # --- Teacher: native resolution through all 48 blocks (LoRA disabled) ---
    print(f"  Teacher (native, all 48 blocks, LoRA OFF)...")
    modality = Modality(
        enabled=True, latent=noisy, timesteps=timesteps,
        positions=positions, context=text_ctx, context_mask=text_mask,
    )
    perturbations = BatchedPerturbationConfig.empty(1)

    # Disable LoRA for teacher
    has_lora = hasattr(base_model, 'disable_adapter_layers')
    if has_lora:
        base_model.disable_adapter_layers()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        teacher_out, _ = base_model(video=modality, audio=None, perturbations=perturbations)
    torch.cuda.synchronize()
    teacher_time = time.perf_counter() - t0

    if has_lora:
        base_model.enable_adapter_layers()

    results["teacher"] = {"pred": teacher_out.float(), "time": teacher_time}
    print(f"    Time: {teacher_time:.3f}s, shape: {teacher_out.shape}")

    # --- Student at each scale (LoRA ON, DDiT merge/unmerge) ---
    for scale in [2, 4]:
        if h % scale != 0 or w % scale != 0:
            print(f"  Skipping scale={scale}: dims {h}x{w} not divisible")
            continue

        ml = adapter.merge_layers[str(scale)]
        new_h, new_w = h // scale, w // scale
        new_seq = nf * new_h * new_w

        print(f"  Student (scale={scale}, seq={seq_len}→{new_seq}, LoRA ON)...")

        # 1. Merge
        merged = ml.merge(noisy, nf, h, w)
        projected = ml.patchify_proj(merged)
        projected = projected + ml.patch_id

        # 2. Positions
        merged_positions = adapter.adjust_positions(positions, scale, nf, h, w)

        # 3. Dummy modality for preprocessor
        dummy_ts = torch.full((1, new_seq), sigma, device=device, dtype=dtype)
        dummy_latent = torch.zeros(1, new_seq, 128, device=device, dtype=dtype)
        dummy_mod = Modality(
            enabled=True, latent=dummy_latent, timesteps=dummy_ts,
            positions=merged_positions, context=text_ctx, context_mask=text_mask,
        )
        # Cast dtype
        target_dtype = real_model.patchify_proj.weight.dtype
        if dummy_mod.latent.dtype != target_dtype:
            dummy_mod = Modality(
                enabled=dummy_mod.enabled, latent=dummy_mod.latent.to(target_dtype),
                timesteps=dummy_mod.timesteps, positions=dummy_mod.positions,
                context=dummy_mod.context, context_mask=dummy_mod.context_mask,
            )
        video_args = real_model.video_args_preprocessor.prepare(dummy_mod)

        # 4. Swap in projected tokens
        video_args = replace(video_args, x=projected.to(video_args.x.dtype))

        # 5. Run through ALL 48 transformer blocks
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            video_out, _ = real_model._process_transformer_blocks(
                video=video_args, audio=None, perturbations=perturbations,
            )
        torch.cuda.synchronize()
        student_time = time.perf_counter() - t0

        # 6. Output: norm + scale_shift + proj_out
        x = video_out.x
        emb_ts = video_out.embedded_timestep
        scale_shift = real_model.scale_shift_table
        shift, scale_val = (
            scale_shift[None, None].to(device=x.device, dtype=x.dtype) + emb_ts[:, :, None]
        ).unbind(dim=2)
        x = real_model.norm_out(x)
        x = x * (1 + scale_val) + shift
        output_merged = ml.proj_out(x)

        # 7. Unmerge
        student_out = ml.unmerge(output_merged, nf, h, w)

        # Residual
        if adapter.config.residual_weight > 0:
            residual = ml.residual_block(noisy)
            student_out = student_out + adapter.config.residual_weight * residual

        # Metrics
        student_f = student_out.float()
        teacher_f = results["teacher"]["pred"]
        mse = (student_f - teacher_f).pow(2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            student_f.flatten(), teacher_f.flatten(), dim=0
        ).item()
        speedup = teacher_time / student_time if student_time > 0 else 0

        results[f"scale_{scale}"] = {
            "pred": student_f, "time": student_time,
            "mse": mse, "cosine_sim": cos_sim, "speedup": speedup,
        }
        print(f"    Time: {student_time:.3f}s (speedup: {speedup:.2f}x)")
        print(f"    MSE vs teacher: {mse:.6f}")
        print(f"    Cosine sim: {cos_sim:.6f}")

    return results


def run_single_step_comparison(scd_model, ddit_wrapper, sample, device, sigma=0.5):
    """Run a single decoder step at native vs DDiT scales and compare outputs."""
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.scd_model import shift_encoder_features
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    dtype = torch.bfloat16
    latent = sample["latents"].to(device=device, dtype=dtype)  # [C, F, H, W]
    nf, h, w = sample["num_frames"], sample["height"], sample["width"]
    seq_len = nf * h * w

    # Patchify: [C, F, H, W] → [1, F*H*W, C]
    latent_seq = latent.permute(1, 2, 3, 0).reshape(1, seq_len, -1)

    # Positions
    positions = make_positions(nf, h, w, device, dtype)

    # Text context
    if sample["text_embeds"] is not None:
        text_ctx = sample["text_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
        text_mask = sample["text_mask"].unsqueeze(0).to(device=device)
    else:
        text_ctx = torch.zeros(1, 256, 3840, device=device, dtype=dtype)
        text_mask = torch.ones(1, 256, device=device, dtype=torch.int64)

    # Encoder pass (once, sigma=0)
    encoder_ts = torch.zeros(1, seq_len, device=device, dtype=dtype)
    encoder_modality = Modality(
        enabled=True, latent=latent_seq, timesteps=encoder_ts,
        positions=positions, context=text_ctx, context_mask=text_mask,
    )
    perturbations = BatchedPerturbationConfig.empty(1)
    tpf = h * w  # tokens per frame

    print(f"  Encoder pass (seq_len={seq_len}, tpf={tpf})...")
    t0 = time.perf_counter()
    with torch.no_grad():
        enc_args, _ = scd_model.forward_encoder(
            video=scd_model._cast_modality_dtype(encoder_modality),
            audio=None, perturbations=perturbations,
            tokens_per_frame=tpf,
        )
    enc_time = time.perf_counter() - t0
    print(f"  Encoder: {enc_time:.2f}s")

    # Extract and shift encoder features
    encoder_features = enc_args.x  # [1, seq_len, D]
    shifted_features = shift_encoder_features(encoder_features, tpf, nf)

    # Add noise at sigma level
    noise = torch.randn_like(latent_seq)
    noisy = latent_seq + sigma * noise

    # Decoder timesteps
    decoder_ts = torch.full((1, seq_len), sigma, device=device, dtype=dtype)

    results = {}

    # --- Native resolution (scale=1) ---
    decoder_modality = Modality(
        enabled=True, latent=noisy, timesteps=decoder_ts,
        positions=positions, context=text_ctx, context_mask=text_mask,
    )
    print(f"  Native decoder (scale=1, seq={seq_len})...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        native_pred, _ = scd_model.forward_decoder(
            video=scd_model._cast_modality_dtype(decoder_modality),
            encoder_features=shifted_features,
            audio=None, perturbations=perturbations,
            encoder_audio_args=None,
        )
    torch.cuda.synchronize()
    native_time = time.perf_counter() - t0
    results["native"] = {"pred": native_pred.float(), "time": native_time}
    print(f"    Time: {native_time:.3f}s, shape: {native_pred.shape}")

    # --- DDiT at each scale ---
    for scale in [2, 4]:
        if h % scale != 0 or w % scale != 0:
            print(f"  Skipping scale={scale}: dims {h}x{w} not divisible")
            continue

        decoder_modality = Modality(
            enabled=True, latent=noisy, timesteps=decoder_ts,
            positions=positions, context=text_ctx, context_mask=text_mask,
        )
        new_seq = nf * (h // scale) * (w // scale)
        print(f"  DDiT decoder (scale={scale}, seq={seq_len}→{new_seq})...")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            ddit_pred, _ = ddit_wrapper.decode_with_ddit(
                video_modality=decoder_modality,
                encoder_features=shifted_features,
                scale=scale,
                num_frames=nf, height=h, width=w,
            )
        torch.cuda.synchronize()
        ddit_time = time.perf_counter() - t0

        # Compute metrics
        ddit_pred_f = ddit_pred.float()
        native_f = results["native"]["pred"]
        mse = (ddit_pred_f - native_f).pow(2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            ddit_pred_f.flatten(), native_f.flatten(), dim=0
        ).item()
        speedup = native_time / ddit_time if ddit_time > 0 else 0

        results[f"scale_{scale}"] = {
            "pred": ddit_pred_f, "time": ddit_time,
            "mse": mse, "cosine_sim": cos_sim, "speedup": speedup,
        }
        print(f"    Time: {ddit_time:.3f}s (speedup: {speedup:.2f}x)")
        print(f"    MSE vs native: {mse:.6f}")
        print(f"    Cosine sim: {cos_sim:.6f}")

    return results


def run_full_denoising(scd_model, ddit_wrapper, sample, device, num_steps=20):
    """Run full denoising loop at native vs DDiT, compare final outputs."""
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.scd_model import shift_encoder_features
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    dtype = torch.bfloat16
    latent = sample["latents"].to(device=device, dtype=dtype)
    nf, h, w = sample["num_frames"], sample["height"], sample["width"]
    seq_len = nf * h * w
    tpf = h * w

    # Patchify
    latent_seq = latent.permute(1, 2, 3, 0).reshape(1, seq_len, -1)
    positions = make_positions(nf, h, w, device, dtype)

    if sample["text_embeds"] is not None:
        text_ctx = sample["text_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
        text_mask = sample["text_mask"].unsqueeze(0).to(device=device)
    else:
        text_ctx = torch.zeros(1, 256, 3840, device=device, dtype=dtype)
        text_mask = torch.ones(1, 256, device=device, dtype=torch.int64)

    perturbations = BatchedPerturbationConfig.empty(1)

    # Encoder pass (shared)
    encoder_ts = torch.zeros(1, seq_len, device=device, dtype=dtype)
    encoder_modality = Modality(
        enabled=True, latent=latent_seq, timesteps=encoder_ts,
        positions=positions, context=text_ctx, context_mask=text_mask,
    )
    print(f"\n  Encoder pass...")
    with torch.no_grad():
        enc_args, _ = scd_model.forward_encoder(
            video=scd_model._cast_modality_dtype(encoder_modality),
            audio=None, perturbations=perturbations,
            tokens_per_frame=tpf,
        )
    shifted_features = shift_encoder_features(enc_args.x, tpf, nf)

    # Sigma schedule
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    results = {}

    # --- Native full loop ---
    print(f"\n  Native denoising ({num_steps} steps)...")
    noise = torch.randn_like(latent_seq)
    x_native = noise.clone()  # Start from pure noise
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
            mod = Modality(
                enabled=True, latent=x_native, timesteps=ts,
                positions=positions, context=text_ctx, context_mask=text_mask,
            )
            vel, _ = scd_model.forward_decoder(
                video=scd_model._cast_modality_dtype(mod),
                encoder_features=shifted_features,
                audio=None, perturbations=perturbations,
                encoder_audio_args=None,
            )
            dt = sigma_next - sigma
            x_native = x_native + dt * vel
    torch.cuda.synchronize()
    native_time = time.perf_counter() - t0
    results["native"] = {"output": x_native.float(), "time": native_time}
    print(f"    Total: {native_time:.2f}s ({native_time/num_steps:.3f}s/step)")

    # --- DDiT full loop (force scale=2 for all steps) ---
    for scale in [2, 4]:
        if h % scale != 0 or w % scale != 0:
            continue
        print(f"\n  DDiT denoising scale={scale} ({num_steps} steps)...")
        x_ddit = noise.clone()  # Same starting noise
        ddit_wrapper.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(num_steps):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                ts = torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype)
                mod = Modality(
                    enabled=True, latent=x_ddit, timesteps=ts,
                    positions=positions, context=text_ctx, context_mask=text_mask,
                )
                vel, _ = ddit_wrapper.decode_with_ddit(
                    video_modality=mod,
                    encoder_features=shifted_features,
                    scale=scale,
                    num_frames=nf, height=h, width=w,
                )
                dt = sigma_next - sigma
                x_ddit = x_ddit + dt * vel
        torch.cuda.synchronize()
        ddit_time = time.perf_counter() - t0

        ddit_f = x_ddit.float()
        native_f = results["native"]["output"]
        mse = (ddit_f - native_f).pow(2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            ddit_f.flatten(), native_f.flatten(), dim=0
        ).item()

        # Also compare to original clean latent
        mse_vs_clean = (ddit_f - latent_seq.float()).pow(2).mean().item()
        native_vs_clean = (native_f - latent_seq.float()).pow(2).mean().item()

        results[f"scale_{scale}"] = {
            "output": ddit_f, "time": ddit_time,
            "mse_vs_native": mse, "cosine_sim": cos_sim,
            "mse_vs_clean": mse_vs_clean,
            "speedup": native_time / ddit_time if ddit_time > 0 else 0,
        }
        print(f"    Total: {ddit_time:.2f}s ({ddit_time/num_steps:.3f}s/step)")
        print(f"    Speedup: {native_time/ddit_time:.2f}x")
        print(f"    MSE vs native: {mse:.6f}")
        print(f"    Cosine sim: {cos_sim:.6f}")
        print(f"    MSE vs clean — native: {native_vs_clean:.6f}, DDiT: {mse_vs_clean:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test DDiT inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_path", default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
    parser.add_argument("--adapter_path", default="outputs/ddit_adapter/ddit_adapter_final.safetensors")
    parser.add_argument("--lora_path", default="outputs/ddit_adapter/ddit_lora_final.safetensors")
    parser.add_argument("--data_root", default="/media/2TB/isometric_i2v_training",
                        help="Data root with latents/ and conditions_final/")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps for full loop")
    parser.add_argument("--full_loop", action="store_true", help="Run full denoising (not just single step)")
    parser.add_argument("--sigma", type=float, default=0.5, help="Sigma for single-step test")
    parser.add_argument("--no_lora", action="store_true", help="Skip loading LoRA")
    args = parser.parse_args()

    device = args.device
    print(f"=== DDiT Inference Test ===")
    print(f"Device: {device}")

    # 1. Load a test sample
    print(f"\n--- Loading test sample ---")
    latent_dir = Path(args.data_root) / "latents"
    cond_dir = Path(args.data_root) / "conditions_final"
    latent_files = sorted(latent_dir.glob("*.pt"))
    pt_file = latent_files[args.sample_idx]
    data = torch.load(pt_file, weights_only=False, map_location="cpu")
    lat = data["latents"]
    nf = data.get("num_frames", lat.shape[1])
    h = data.get("height", lat.shape[2])
    w = data.get("width", lat.shape[3])
    nf = nf.item() if isinstance(nf, torch.Tensor) else int(nf)
    h = h.item() if isinstance(h, torch.Tensor) else int(h)
    w = w.item() if isinstance(w, torch.Tensor) else int(w)

    cond_file = cond_dir / pt_file.name
    text_embeds = text_mask = None
    if cond_file.exists():
        cond = torch.load(cond_file, weights_only=False, map_location="cpu")
        text_embeds = cond["video_prompt_embeds"]
        text_mask = cond["prompt_attention_mask"]

    sample = {
        "latents": lat, "num_frames": nf, "height": h, "width": w,
        "text_embeds": text_embeds, "text_mask": text_mask,
    }
    print(f"  Sample: {pt_file.name}, shape={tuple(lat.shape)}, F={nf} H={h} W={w}")
    print(f"  Seq len: {nf*h*w}, text: {'yes' if text_embeds is not None else 'no'}")

    # 2. Load base model
    print(f"\n--- Loading LTX-2 model ---")
    # Reuse the memory-efficient loader from training script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_ddit_adapter import load_base_model
    base_model = load_base_model(args.model_path, device=device, quantize=True)

    # 3. Load DDiT adapter
    print(f"\n--- Loading DDiT adapter ---")
    from ltx_core.model.transformer.ddit import DDiTAdapter, DDiTConfig
    config = DDiTConfig(enabled=True, supported_scales=(1, 2, 4))
    adapter = DDiTAdapter(
        inner_dim=base_model.inner_dim,
        in_channels=128,
        config=config,
    )
    state = load_file(args.adapter_path)
    adapter.load_state_dict(state)
    adapter = adapter.to(device=device, dtype=torch.bfloat16)
    adapter.eval()
    print(f"  Loaded from {args.adapter_path}")
    print(f"  Params: {sum(p.numel() for p in adapter.parameters()):,}")

    # 4. Load LoRA (optional) — applied to base_model directly
    if not args.no_lora and Path(args.lora_path).exists():
        print(f"\n--- Loading LoRA ---")
        from peft import LoraConfig, get_peft_model
        lora_state = load_file(args.lora_path)
        rank = 32
        for k, v in lora_state.items():
            if 'lora_A' in k:
                rank = v.shape[0]
                break
        targets = set()
        for k in lora_state.keys():
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
        lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=list(targets), lora_dropout=0.0)
        for p in base_model.parameters():
            p.requires_grad_(False)
        base_model = get_peft_model(base_model, lora_config)
        missing, unexpected = base_model.load_state_dict(lora_state, strict=False)
        loaded = len(lora_state) - len(unexpected)
        base_model.eval()
        print(f"  LoRA: {loaded} tensors, rank={rank}, targets={targets}")
    elif args.no_lora:
        print(f"\n--- Skipping LoRA (--no_lora) ---")

    # 5. Run standalone test (matches training forward pass exactly)
    vram = torch.cuda.memory_allocated(device) / 1e9
    print(f"\n--- Ready (VRAM: {vram:.1f}GB) ---")

    print(f"\n{'='*60}")
    print(f"STANDALONE TEST (all 48 blocks, matches training)")
    print(f"sigma={args.sigma}")
    print(f"{'='*60}")
    results = run_standalone_comparison(base_model, adapter, sample, device, args.sigma)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    teacher_time = results["teacher"]["time"]
    print(f"  Teacher (native, all 48): {teacher_time:.3f}s")
    for key in ["scale_2", "scale_4"]:
        if key in results:
            r = results[key]
            print(f"  {key}: {r['time']:.3f}s "
                  f"(speedup {r['speedup']:.2f}x, "
                  f"MSE {r['mse']:.6f}, "
                  f"cos_sim {r['cosine_sim']:.6f})")

    # Verdict
    print(f"\n  Verdict:")
    for key in ["scale_2", "scale_4"]:
        if key not in results:
            continue
        r = results[key]
        mse = r["mse"]
        cos = r["cosine_sim"]
        spd = r["speedup"]
        if cos > 0.99 and spd > 1.5:
            grade = "EXCELLENT"
        elif cos > 0.95 and spd > 1.2:
            grade = "GOOD"
        elif cos > 0.90:
            grade = "FAIR"
        else:
            grade = "POOR"
        print(f"    {key}: {grade} — {spd:.1f}x faster, {cos:.4f} cos_sim, {mse:.6f} MSE")


if __name__ == "__main__":
    main()
