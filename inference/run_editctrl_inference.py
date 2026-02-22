#!/usr/bin/env python3
"""EditCtrl + SCD inference script for video editing.

Loads a source video, applies edit masks, and generates an edited video
where only the masked regions change according to the text prompt.

Usage:
    python inference/run_editctrl_inference.py \
        --source_video /path/to/video.mp4 \
        --mask_rect "0.2,0.2,0.6,0.6" \
        --prompt "a red sports car" \
        --checkpoint /path/to/editctrl_checkpoint.safetensors \
        --output /path/to/output.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EditCtrl + SCD video editing inference")

    parser.add_argument("--source_video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--masks", type=str, default=None,
        help="Path to mask video/images (grayscale, white=edit region)"
    )
    parser.add_argument(
        "--mask_rect", type=str, default=None,
        help="Rectangle mask as 'x1,y1,x2,y2' in normalized [0,1] coords"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Edit prompt")
    parser.add_argument(
        "--negative_prompt", type=str,
        default="worst quality, blurry, distorted",
        help="Negative prompt"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="EditCtrl checkpoint path")
    parser.add_argument("--scd_checkpoint", type=str, default=None, help="Base SCD LoRA checkpoint")
    parser.add_argument(
        "--model_path", type=str,
        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors",
        help="Base LTX-2 model path"
    )
    parser.add_argument(
        "--text_encoder_path", type=str,
        default="/media/2TB/ltx-models/gemma",
        help="Text encoder path"
    )
    parser.add_argument("--output", type=str, default="edited_output.mp4", help="Output video path")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--max_frames", type=int, default=33, help="Max frames to process")
    parser.add_argument("--height", type=int, default=288, help="Resize height")
    parser.add_argument("--width", type=int, default=512, help="Resize width")

    # HRR editing options
    parser.add_argument("--edit_prompt", type=str, default=None, help="Edit prompt (uses HRR to blend with source)")
    parser.add_argument("--hrr_checkpoint", type=str, default=None, help="HRR enhancer checkpoint path")
    parser.add_argument(
        "--hrr_mode", type=str, default="direct",
        choices=["hybrid", "interpolated", "direct"],
        help="HRR editing mode: hybrid (channel-swap), interpolated (freq-blend), direct (enhance only)"
    )
    parser.add_argument("--hrr_alpha", type=float, default=0.5, help="Blend weight for interpolated mode")
    parser.add_argument(
        "--hrr_freq_profile", type=str, default=None,
        choices=["structure_preserving", "texture_swap"],
        help="Frequency profile for interpolated mode"
    )

    return parser.parse_args()


def load_video(path: str, max_frames: int, height: int, width: int) -> tuple[torch.Tensor, float]:
    """Load and preprocess video. Returns [1, C, F, H, W] in [-1, 1] and fps."""
    video, _, info = tvio.read_video(path, pts_unit="sec")
    fps = info.get("video_fps", 25.0)

    # Truncate frames
    video = video[:max_frames]  # [F, H, W, C]

    # Resize and normalize
    video = video.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W] in [0,1]
    video = TF.resize(video, [height, width], antialias=True)
    video = video * 2.0 - 1.0  # [-1, 1]

    # [F, C, H, W] → [1, C, F, H, W]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)

    return video, fps


def create_rect_mask(
    rect_str: str, num_frames: int, height: int, width: int
) -> torch.Tensor:
    """Create rectangle mask from 'x1,y1,x2,y2' string. Returns [1,1,F,H,W]."""
    coords = [float(x) for x in rect_str.split(",")]
    x1, y1, x2, y2 = coords

    mask = torch.zeros(1, 1, num_frames, height, width)
    h1, h2 = int(y1 * height), int(y2 * height)
    w1, w2 = int(x1 * width), int(x2 * width)
    mask[:, :, :, h1:h2, w1:w2] = 1.0

    return mask


def load_mask(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Load mask from image/video file. Returns [1,1,F,H,W]."""
    p = Path(path)
    if p.suffix in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = Image.open(path).convert("L")
        mask_2d = TF.to_tensor(img)  # [1, H, W]
        mask_2d = TF.resize(mask_2d, [height, width], antialias=True)
        mask = (mask_2d > 0.5).float()
        # Repeat for all frames
        mask = mask.unsqueeze(0).unsqueeze(2).expand(-1, -1, num_frames, -1, -1)
    else:
        # Video mask
        mask_vid, _, _ = tvio.read_video(path, pts_unit="sec")
        mask_vid = mask_vid[:num_frames, :, :, 0:1].float() / 255.0  # [F, H, W, 1]
        mask_vid = mask_vid.permute(3, 0, 1, 2)  # [1, F, H, W]
        mask_vid = TF.resize(mask_vid, [height, width], antialias=True)
        mask = (mask_vid > 0.5).float().unsqueeze(0)  # [1, 1, F, H, W]

    return mask


def main():
    args = parse_args()

    print(f"Loading source video from {args.source_video}...")
    source_video, fps = load_video(
        args.source_video, args.max_frames, args.height, args.width
    )
    num_frames = source_video.shape[2]
    print(f"  Loaded: {num_frames} frames, {args.height}x{args.width}, {fps:.1f} fps")

    # Create/load edit mask
    if args.mask_rect:
        edit_mask = create_rect_mask(args.mask_rect, num_frames, args.height, args.width)
        mask_area = edit_mask.mean().item()
        print(f"  Rectangle mask: {mask_area:.1%} area")
    elif args.masks:
        edit_mask = load_mask(args.masks, num_frames, args.height, args.width)
        mask_area = edit_mask.mean().item()
        print(f"  Loaded mask: {mask_area:.1%} area")
    else:
        print("ERROR: Must specify either --mask_rect or --masks")
        sys.exit(1)

    # Load models
    print("Loading LTX-2 model components...")
    from ltx_trainer.model_loader import load_model as load_ltx_model, load_text_encoder

    components = load_ltx_model(
        checkpoint_path=args.model_path,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=True,
        with_video_vae_decoder=True,
        with_text_encoder=False,
    )

    transformer = components.transformer.to(dtype=torch.bfloat16)

    # Quantize for memory efficiency
    from ltx_trainer.quantization import quantize_model
    print("Quantizing transformer (int8)...")
    transformer = quantize_model(transformer, precision="int8-quanto", device=args.device)
    transformer = transformer.to(args.device)

    # Wrap with SCD
    from ltx_core.model.transformer.scd_model import LTXSCDModel
    scd_model = LTXSCDModel(
        base_model=transformer,
        encoder_layers=32,
        decoder_input_combine="token_concat",
    )

    # Load SCD LoRA if provided
    if args.scd_checkpoint:
        print(f"Loading SCD LoRA from {args.scd_checkpoint}...")
        from safetensors.torch import load_file
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        lora_config = LoraConfig(
            r=32, lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        scd_model = get_peft_model(scd_model, lora_config)
        scd_state = load_file(args.scd_checkpoint)
        scd_state = {k.replace("diffusion_model.", "", 1): v for k, v in scd_state.items()}
        set_peft_model_state_dict(scd_model.get_base_model(), scd_state)

    # Load EditCtrl modules
    from ltx_core.model.transformer.editctrl_modules import LocalContextModule, GlobalContextEmbedder

    print("Loading EditCtrl modules...")
    local_module = LocalContextModule(
        latent_dim=128,
        inner_dim=scd_model.inner_dim if hasattr(scd_model, 'inner_dim') else 4096,
        context_dim=4096,
        num_blocks=4,
    ).to(device=args.device, dtype=torch.bfloat16)

    global_embedder = GlobalContextEmbedder(
        latent_dim=128,
        inner_dim=4096,
        num_tokens=256,
    ).to(device=args.device, dtype=torch.bfloat16)

    # Load EditCtrl checkpoint
    from safetensors.torch import load_file
    ckpt = load_file(args.checkpoint)

    lcm_dict = {k.replace("strategy.local_context_module.", ""): v for k, v in ckpt.items() if "local_context_module" in k}
    ge_dict = {k.replace("strategy.global_embedder.", ""): v for k, v in ckpt.items() if "global_embedder" in k}

    if lcm_dict:
        local_module.load_state_dict(lcm_dict, strict=False)
        print(f"  Loaded LocalContextModule: {len(lcm_dict)} tensors")
    if ge_dict:
        global_embedder.load_state_dict(ge_dict, strict=False)
        print(f"  Loaded GlobalContextEmbedder: {len(ge_dict)} tensors")

    # Load text encoder and encode prompt(s)
    print("Encoding text prompt...")
    text_encoder = load_text_encoder(
        checkpoint_path=args.model_path,
        gemma_model_path=args.text_encoder_path,
        device=args.device,
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        text_context, _, attention_mask = text_encoder(args.prompt)

        # Encode edit prompt if provided
        edit_text_context = None
        edit_text_mask = None
        if args.edit_prompt:
            edit_text_context, _, edit_text_mask = text_encoder(args.edit_prompt)
            print(f"  Edit prompt encoded: '{args.edit_prompt}'")

    # Free text encoder
    del text_encoder
    torch.cuda.empty_cache()

    # Load HRR enhancer if checkpoint provided
    hrr_enhancer = None
    if args.hrr_checkpoint:
        print(f"Loading HRR enhancer from {args.hrr_checkpoint}...")
        from safetensors.torch import load_file as safe_load
        hrr_state = safe_load(args.hrr_checkpoint)

        # Extract HRR weights from checkpoint
        hrr_prefix = "strategy.hrr_enhancer."
        hrr_dict = {
            k[len(hrr_prefix):]: v for k, v in hrr_state.items()
            if k.startswith(hrr_prefix)
        }

        if hrr_dict:
            from ltx_trainer.hrr_text_enhancer import TokenAwareHRR
            hrr_enhancer = TokenAwareHRR(dim=3840, num_channels=16)
            hrr_enhancer.load_state_dict(hrr_dict, strict=False)
            hrr_enhancer = hrr_enhancer.to(device=args.device, dtype=torch.bfloat16)
            print(f"  Loaded HRR enhancer: {len(hrr_dict)} tensors")
        else:
            print("  Warning: No HRR weights found in checkpoint")

    # Move VAE components
    vae_encoder = components.video_vae_encoder.to(device=args.device, dtype=torch.bfloat16)
    vae_decoder = components.video_vae_decoder.to(device=args.device, dtype=torch.bfloat16)

    # Build pipeline
    from scd.pipelines.pipeline_editctrl_scd import EditCtrlSCDPipeline
    from ltx_core.components.patchifiers import VideoLatentPatchifier

    pipeline = EditCtrlSCDPipeline(
        scd_model=scd_model,
        local_context_module=local_module,
        global_embedder=global_embedder if ge_dict else None,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        scheduler=components.scheduler,
        patchifier=VideoLatentPatchifier(patch_size=1),
        hrr_enhancer=hrr_enhancer,
    )

    # Run inference
    mode_str = f", HRR mode={args.hrr_mode}" if hrr_enhancer else ""
    print(f"Running EditCtrl inference ({args.steps} steps{mode_str})...")
    output = pipeline(
        source_video=source_video,
        edit_masks=edit_mask,
        text_context=text_context,
        text_mask=attention_mask,
        edit_text_context=edit_text_context,
        edit_text_mask=edit_text_mask,
        hrr_edit_mode=args.hrr_mode,
        hrr_alpha=args.hrr_alpha,
        hrr_freq_profile=args.hrr_freq_profile,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
    )

    # Save output video
    print(f"Saving edited video to {args.output}...")
    edited = output.video[0]  # [C, F, H, W]
    edited = ((edited.float().clamp(-1, 1) + 1) / 2 * 255).byte()  # [0, 255]
    edited = edited.permute(1, 2, 3, 0).cpu()  # [F, H, W, C]

    tvio.write_video(args.output, edited, fps=fps)
    print(f"Done! Output saved to {args.output}")


if __name__ == "__main__":
    main()
