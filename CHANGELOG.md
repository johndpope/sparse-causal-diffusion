# Changelog

## [2026-02-23] Qwen VL Semantic Object Masks + Per-Sample Targeting

### Problem
Training used random rectangular masks (or MuLAn instance masks from unrelated COCO/LAION images).
The TMA module provided Qwen VL semantic tokens to the denoising context, but these tokens had
no relationship to the edit mask region — the model saw "bar scene" semantics while editing a
random rectangle.

### Solution: Qwen VL → Object Detection → Per-Sample Masks
1. **`compute_semantic_masks.py`** — Runs Qwen2.5-VL-7B-Instruct on each of the 8 unique
   training scene frames. Detects objects with bounding boxes (Person, Shelf, Bottle, Bar Counter,
   Wine Glass, Window, etc.). Saves per-sample `.pt` files with object names + bbox masks at
   latent resolution.

2. **`editctrl_scd_strategy.py`** — Updated to load per-sample masks at init. During training,
   randomly picks a detected object for each sample and uses its mask as the edit region.
   In `mixed` mode (70/30), 70% of steps use real object masks, 30% use random masks for diversity.

3. **Config** (`editctrl_scd_tma.yaml`) — Points `semantic_mask_dir` to per-sample Qwen VL
   detections instead of MuLAn library.

### Result
- Mask coverage varies naturally per object: 0.3% (wine glass) → 33% (bar counter)
- TMA semantic tokens now align with edit region (model learns "this region is a Person")
- 128 per-sample mask files, 4-34 objects detected per scene

### Debug Visualization
- `trainer.py` — Added 4th column to debug_decoded.png: red mask overlay showing targeted region
- `outputs/editctrl_scd_tma/debug_mask_overlay.png` — Standalone overlay visualization

### Training Restart Context
- Previous run reached step ~411 with MuLAn masks (random shapes from other images)
- Restarted from step 0 with per-sample Qwen VL object masks
- No checkpoint to resume (checkpoints save every 500 steps)
- WandB: https://wandb.ai/snoozie/editctrl-scd

### Known Issue: Masks Are Still Bounding Boxes
The current `objects_to_masks()` fills rectangular bounding boxes from Qwen VL detections.
**Next step**: Pipe Qwen VL bboxes through SAM (Segment Anything) for pixel-accurate
silhouette masks (person outline, bottle shape, etc. instead of rectangles).

---

## [2026-02-22] LayerFusion Boundary Blending + MuLAn Dataset Integration

### Added
- `scd/utils/boundary_blend.py` — Distance-based soft boundary blending for inference
- `scd/utils/mask_utils.py` — `SemanticMaskLibrary`, `generate_semantic_masks()`,
  `pixel_mask_to_token_mask()`, `compute_boundary_blend_mask()`
- `scripts/extract_mulan_masks.py` — Extract instance masks from MuLAn `.p.zl` annotations
- MuLAn RAR extraction: 44,860 annotations → 40,512 images, 62,779 instance masks
- `scd/pipelines/pipeline_editctrl_scd.py` — Soft blending in denoising loop

### Fixed
- `scd_model.py` `_duplicate_pe()` — Handle both 3D `[B,S,D]` and 4D `[B,H,S,D]` PE tensors
  (was doubling wrong dimension for 3D case)

---

## [2026-02-21] EditCtrl + SCD + TMA Phase 1

### Added
- EditCtrl video editing on top of SCD: LocalContextModule + GlobalContextEmbedder
- TMA (Task-adaptive Multimodal Alignment): Qwen VL 8-token semantic context
- `configs/editctrl_scd_tma.yaml` — Full training config
- `scripts/compute_qwen_features.py` — Pre-compute Qwen VL features per sample
- `inference/run_editctrl_inference.py` — Inference pipeline with mask-based editing
- `test_editctrl_forward.py` — End-to-end forward pass test (11 steps, all passing)

---

## [2026-02-20] SCD Base Training

### Added
- SCD LoRA training on LTX-2 19B (encoder/decoder split at layer 32)
- `configs/scd_finetune.yaml` — Base SCD training config
- Trained 2000 steps on 128 isometric room clips
- Checkpoint: `outputs/scd_finetune_v1/checkpoints/lora_weights_step_02000.safetensors`
