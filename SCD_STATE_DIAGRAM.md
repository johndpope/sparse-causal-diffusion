# Separable Causal Diffusion (SCD) — Complete State Diagram

## Porting Guide: SCD → LTX-2

---

## 1. HIGH-LEVEL ARCHITECTURE COMPARISON

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCD (Sparse Causal Diffusion)                       │
│                                                                        │
│  Key Insight: Causal reasoning is SEPARABLE from denoising             │
│  ─ Encoder runs ONCE per frame (causal temporal reasoning)             │
│  ─ Decoder runs N times per frame (iterative denoising)                │
│                                                                        │
│  ┌──────────────┐         ┌───────────────┐                           │
│  │  Causal       │  ──→   │  Lightweight   │                           │
│  │  Transformer  │ (1x)   │  Diffusion     │ (Nx denoising steps)     │
│  │  Encoder      │        │  Decoder       │                           │
│  └──────────────┘         └───────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    LTX-2 (Lightricks Video Transformer)                │
│                                                                        │
│  Standard approach: Full transformer at every denoising step           │
│                                                                        │
│  ┌──────────────────────────────────────────┐                          │
│  │  LTXModel (48-layer Transformer)         │                          │
│  │  ─ Patchify → AdaLN → Attention → FFN    │  (Nx denoising steps)   │
│  │  ─ Cross-attention to text embeddings     │                          │
│  │  ─ RoPE positional encoding               │                          │
│  └──────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. SCD COMPLETE STATE DIAGRAM

```
                    ╔══════════════════════════════════════╗
                    ║      TRAINING PIPELINE (train.py)     ║
                    ╚══════════════════════════════════════╝
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          ┌──────────────┐                 ┌──────────────────┐
          │ Config (YAML) │                 │ SCDTrainer.__init__│
          │ scd_minecraft │                 │ trainer_scd.py     │
          │   .yml        │                 └──────┬───────────┘
          └──────┬───────┘                         │
                 │                    ┌────────────┼────────────┐
                 ▼                    ▼            ▼            ▼
        ┌────────────┐     ┌────────────┐  ┌──────────┐  ┌───────────┐
        │ build_model │     │ Encoder     │  │ Decoder  │  │ VAE       │
        │ (registry)  │     │ SCD_M_enc   │  │ SCD_M_dec│  │ (DCAE or  │
        └────────────┘     │ 8 layers    │  │ 4 layers │  │  AE-KL)   │
                           │ causal mask │  │ no mask  │  │ frozen    │
                           └──────┬──────┘  └────┬─────┘  └─────┬─────┘
                                  │              │              │
                                  ▼              ▼              │
                           ┌─────────────────────────┐         │
                           │   SCDEncoderDecoder      │         │
                           │   (wraps both models)    │         │
                           └────────────┬────────────┘         │
                                        │                      │
                    ┌───────────────────┼──────────────────────┘
                    ▼                   ▼
          ┌──────────────┐    ┌──────────────────┐
          │ Scheduler     │    │ FlowMatchEuler   │
          │ (Flow Match)  │    │ DiscreteScheduler│
          └──────────────┘    └──────────────────┘
```

### 2A. TRAINING STEP STATE MACHINE

```
┌─────────────────────────────────────────────────────────────────────┐
│                   train_step(batch) — trainer_scd.py                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
    ┌─────────────────┐              ┌─────────────────────┐
    │ PIXEL SPACE PATH │              │ LATENT SPACE PATH   │
    │ video → [-1,1]   │              │ video → VAE.encode  │
    │ latents=pixels   │              │ latents → normalize │
    └────────┬────────┘              └──────────┬──────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
           ┌────────────────────────────────┐
           │  latents: [B, T, C, H, W]     │
           │  (B=batch, T=frames, C=32,    │
           │   H=8, W=8 for 128x128 input) │
           └───────────────┬────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Sample noise     │   │ Sample timesteps │
    │ noise ~ N(0,1)   │   │ u ~ logit_normal │
    │ same shape as    │   │ per-frame random  │
    │ latents          │   │ t ∈ [0,1]        │
    └────────┬─────────┘   └────────┬─────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
           ┌──────────────────────────────┐
           │ noisy_latents = scale_noise  │
           │   = (1-σ)·latents + σ·noise  │
           │ (flow matching formulation)  │
           └───────────────┬──────────────┘
                           │
                           ▼
           ┌──────────────────────────────────────────┐
           │ CONTEXT MASKING                           │
           │                                          │
           │ For "long_context" training:              │
           │ ┌─────────────────────────────────────┐  │
           │ │ Frames[0..T-ctx_win]: always clean  │  │
           │ │ Frames[T-ctx_win..T]: p=0.1 clean   │  │
           │ │ Clean frames get timestep = -1      │  │
           │ │ Clean frames get original latents   │  │
           │ └─────────────────────────────────────┘  │
           └───────────────┬──────────────────────────┘
                           │
                           ▼
        ╔══════════════════════════════════════════════╗
        ║     SCDEncoderDecoder.forward_train()        ║
        ╚══════════════════════════════════════════════╝
                           │
              ┌────────────┴────────────┐
              ▼                         │
   ┌─────────────────────┐              │
   │ ENCODER (runs once) │              │
   │                     │              │
   │ Input:              │              │
   │  hidden_states =    │              │
   │    noisy_latents    │              │
   │    [B,T,C,H,W]     │              │
   │                     │              │
   │  timestep = ones    │              │
   │    * (-1)           │              │
   │  (clean context     │              │
   │   signal)           │              │
   │                     │              │
   │ Processing:         │              │
   │  1. Pack patches    │              │
   │  2. x_embedder      │              │
   │  3. Build CAUSAL    │              │
   │     attention mask  │              │
   │  4. RoPE pos embed  │              │
   │  5. 8 transformer   │              │
   │     blocks with     │              │
   │     KV-cache        │              │
   │  6. AdaLN norm_out  │              │
   │                     │              │
   │ Output:             │              │
   │  encoder_output     │              │
   │  [B, T*H'*W', D]   │              │
   │  (token features)   │              │
   └─────────┬───────────┘              │
             │                          │
             ▼                          ▼
   ┌──────────────────────────────────────────────────┐
   │ RESHAPE + SHIFT                                  │
   │                                                  │
   │ encoder_output_shifted =                         │
   │   encoder_output[:, -ctx_win-1:-1, ...]          │
   │                                                  │
   │ (Use frame t's encoder output to decode          │
   │  frame t+1 — the causal shift!)                  │
   │                                                  │
   │ hidden_states_decoder =                          │
   │   noisy_latents[:, -ctx_win:, ...]               │
   │                                                  │
   │ Reshape: [B*ctx_win, 1, C, H, W]                │
   │ (Each frame decoded independently)               │
   └───────────────┬──────────────────────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────────────────────┐
   │ DECODER (runs once during training,              │
   │          N steps during inference)                │
   │                                                  │
   │ Input:                                           │
   │  hidden_states = noisy frame [B*W, 1, C, H, W]  │
   │  encoder_output = shifted features               │
   │  timestep = per-frame noise level                │
   │                                                  │
   │ Combine encoder + decoder input:                 │
   │  MODE "token_concat":                            │
   │    concat([encoder_tokens, noisy_tokens], dim=1) │
   │    sequence length doubled                       │
   │  MODE "concat":                                  │
   │    cat along feature dim → linear projection     │
   │  MODE "add":                                     │
   │    element-wise addition                         │
   │                                                  │
   │ Processing:                                      │
   │  1. x_embedder(noisy patches)                    │
   │  2. Combine with encoder features                │
   │  3. 4 transformer blocks                         │
   │     (NO causal mask — frame-wise only)           │
   │  4. norm_out → proj_out                          │
   │  5. Unpack to spatial                            │
   │                                                  │
   │ Output: predicted velocity [B, ctx_win, C, H, W] │
   └───────────────┬──────────────────────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────────────────────┐
   │ LOSS COMPUTATION                                 │
   │                                                  │
   │ target = noise - latents  (velocity target)      │
   │                                                  │
   │ loss = MSE(model_pred, target)                   │
   │       per-frame, masked (skip context frames)    │
   │                                                  │
   │ loss_mask = ~context_mask                        │
   │ loss = (loss * mask).sum() / mask.sum()          │
   └──────────────────────────────────────────────────┘
```

### 2B. INFERENCE STATE MACHINE

```
┌─────────────────────────────────────────────────────────────────────┐
│              pipeline_scd.py — SCDPipeline.generate()               │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ INITIALIZATION                         │
        │                                       │
        │ 1. Encode context frames via VAE      │
        │    context_sequence → vae_encode →     │
        │    latents [B, ctx_len, C, H, W]      │
        │                                       │
        │ 2. Sample init noise for all          │
        │    future frames                       │
        │    [B, unroll_len, C, H, W]           │
        │                                       │
        │ 3. Initialize KV cache                │
        │    context_cache = {                   │
        │      is_cache_step: True,             │
        │      kv_cache: {},                    │
        │      cached_seqlen: 0                 │
        │    }                                  │
        └──────────────────┬────────────────────┘
                           │
               ╔═══════════╧═══════════════╗
               ║ FOR EACH NEW FRAME (f):    ║
               ║ f = ctx_len → ctx_len+N    ║
               ╚═══════════╤═══════════════╝
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │ __call__() — DENOISING LOOP FOR FRAME f      │
        │                                              │
        │ Setup:                                       │
        │  ─ vision_context = all generated + context  │
        │  ─ latents = init_noise for frame f          │
        │  ─ CFG: duplicate batch (cond + uncond)      │
        │                                              │
        │ scheduler.set_timesteps(num_steps=50)        │
        └──────────────────┬───────────────────────────┘
                           │
                ╔══════════╧══════════════╗
                ║ FOR EACH TIMESTEP t:     ║
                ║ t = 1.0 → 0.0           ║
                ╚══════════╤══════════════╝
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐     ┌──────────────────────┐
   │ t == first step   │     │ t > first step       │
   │                   │     │                      │
   │ is_cache_step=T   │     │ is_cache_step=F      │
   │                   │     │                      │
   │ ENCODER RUNS:     │     │ ENCODER SKIPPED:     │
   │ Processes ALL     │     │ Uses cached          │
   │ context frames    │     │ encoder_output       │
   │ with KV-cache     │     │ (lazy_encoder_output)│
   │                   │     │                      │
   │ KV-cache updated  │     │ No KV-cache update   │
   └─────────┬────────┘     └──────────┬───────────┘
             │                         │
             └──────────┬──────────────┘
                        ▼
           ┌──────────────────────────┐
           │ DECODER ALWAYS RUNS:     │
           │                          │
           │ Takes encoder features   │
           │ + current noisy frame    │
           │ → predicts velocity      │
           │                          │
           │ noise_pred = decoder(    │
           │   noisy_frame,           │
           │   encoder_features[-1])  │
           └─────────┬────────────────┘
                     │
                     ▼
           ┌──────────────────────────┐
           │ CFG GUIDANCE             │
           │                          │
           │ pred = uncond +          │
           │  scale*(cond - uncond)   │
           └─────────┬────────────────┘
                     │
                     ▼
           ┌──────────────────────────┐
           │ EULER STEP               │
           │                          │
           │ latents = scheduler.step │
           │   (noise_pred, t,        │
           │    latents).prev_sample  │
           └─────────┬────────────────┘
                     │
                     ▼
              [back to timestep loop]

                     │ (after all timesteps)
                     ▼
           ┌──────────────────────────┐
           │ Frame f is denoised!     │
           │                          │
           │ Append to latents:       │
           │ latents = cat(latents,   │
           │              pred_frame) │
           │                          │
           │ Update KV cache for      │
           │ next frame               │
           └─────────┬────────────────┘
                     │
                     ▼
              [back to frame loop]

                     │ (after all frames)
                     ▼
           ┌──────────────────────────┐
           │ VAE DECODE               │
           │                          │
           │ latents → vae_decode →   │
           │ pixel video [B,T,3,H,W] │
           └──────────────────────────┘
```

---

## 3. COMPONENT-LEVEL MAPPING: SCD → LTX-2

```
┌─────────────────────────────┬───────────────────────────────────────┐
│       SCD Component         │        LTX-2 Equivalent               │
├─────────────────────────────┼───────────────────────────────────────┤
│ SCDTransformer              │ LTXModel                              │
│  (encoder OR decoder)       │  (single unified transformer)         │
│                             │                                       │
│ SCDTransformerBlock         │ BasicAVTransformerBlock                │
│  - AdaLNZeroSingle          │  - AdaLayerNormSingle                 │
│  - Attention (SDPA)         │  - AttentionFunction (SDPA)           │
│  - FeedForward (GELU)       │  - FeedForward                        │
│                             │                                       │
│ SCDAttnProcessor            │ Attention in transformer.py           │
│  - KV cache for encoder     │  (no built-in KV cache)              │
│  - RoPE via FluxPosEmbed    │  - RoPE via LTXRopeType              │
│  - Causal attention mask    │  (no built-in causal mask)            │
│                             │                                       │
│ SCDEncoderDecoder           │ *** DOES NOT EXIST — MUST CREATE ***  │
│  - Wraps encoder + decoder  │                                       │
│  - forward_train/eval       │                                       │
│                             │                                       │
│ FlowMatchEulerScheduler     │ LTX2Scheduler + EulerDiffusionStep   │
│                             │                                       │
│ SCDPipeline                 │ TI2VidOneStagePipeline                │
│                             │                                       │
│ MyAutoencoderDC (DCAE)      │ VideoVAE (3D causal VAE)             │
│  - 2D image VAE             │  - 3D video VAE with temporal         │
│  - per-frame encode/decode  │    compression                        │
│                             │                                       │
│ LabelEmbedding (actions)    │ Gemma text encoder                    │
│  - discrete action classes  │  - continuous text embeddings          │
│  - lookup table             │  - PixArtAlphaTextProjection          │
│                             │                                       │
│ AdaLayerNormContinuous      │ AdaLayerNormSingle (adaln.py)         │
│  (output norm with noise)   │  (timestep-conditioned norm)          │
│                             │                                       │
│ x_embedder (Linear)         │ patchify_proj (Linear)                │
│ patch_size=1                │ patchifier (3D patching)              │
│                             │                                       │
│ Timesteps + TimestepEmbed   │ AdaLayerNormSingle                    │
│  - sinusoidal → MLP         │  - combined timestep embedding        │
│                             │                                       │
│ Velocity prediction:        │ Velocity prediction:                  │
│  target = noise - latents   │  target = noise - latents             │
│  (identical formulation)    │  (identical formulation)              │
│                             │                                       │
│ Modality (implicit)         │ Modality (dataclass)                  │
│  - latent, timestep, pos    │  - latent, timesteps, positions,      │
│                             │    context, context_mask               │
└─────────────────────────────┴───────────────────────────────────────┘
```

---

## 4. SCD INTERNAL DATA FLOW (SHAPES)

```
Assuming: Minecraft dataset, 128x128 resolution, DCAE 16x compression
  → Latent: 8x8 spatial, 32 channels
  → Patch size: 1 → tokens_per_frame = 64

INPUT VIDEO: [B=2, T=300, C=3, H=128, W=128]
                    │
                    ▼ VAE encode
LATENTS: [B=2, T=300, C=32, H=8, W=8]
                    │
                    ▼ pack patches (patch_size=1)
TOKENS: [B=2, T*H*W=19200, C_patch=32]
                    │
                    ▼ x_embedder (Linear 32→768)
EMBEDDINGS: [B=2, 19200, D=768]
                    │
    ┌───────────────┤
    ▼               │
 ENCODER            │
 (8 layers)         │
                    │
 Causal Mask:       │
 ┌──────────────────────────┐
 │ Frame 0:  [1 0 0 0 ...]  │  (can only see itself)
 │ Frame 1:  [1 1 0 0 ...]  │  (sees frame 0 + self)
 │ Frame 2:  [1 1 1 0 ...]  │  (sees frames 0,1 + self)
 │ ...                       │
 │ Frame T:  [1 1 1 1 ... 1] │  (sees all previous)
 └──────────────────────────┘
 (block-diagonal at frame boundaries)
                    │
                    ▼
 ENCODER OUTPUT: [B=2, 19200, D=768]
                    │
                    ▼ reshape + causal shift
 Per-frame features shifted by 1:
   enc[frame_t] → used for decoding frame_t+1
                    │
                    ▼
 DECODER INPUT (per frame):
   noisy_tokens[frame]:  [B*ctx, 64, 32] → embed → [B*ctx, 64, 768]
   encoder_tokens[frame-1]: [B*ctx, 64, 768]
                    │
                    ▼ token_concat
 COMBINED: [B*ctx, 128, 768]  (doubled sequence)
                    │
                    ▼
 DECODER (4 layers, NO causal mask)
                    │
                    ▼ take second half of tokens
 OUTPUT: [B*ctx, 64, 768]
                    │
                    ▼ proj_out (768→32)
 PRED: [B*ctx, 64, 32]
                    │
                    ▼ unpack patches
 VELOCITY: [B, ctx_win, 32, 8, 8]
```

---

## 5. PORTING STRATEGY: SCD → LTX-2

```
╔══════════════════════════════════════════════════════════════════════╗
║                    WHAT NEEDS TO CHANGE FOR LTX-2                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║ 1. SPLIT THE LTX TRANSFORMER INTO ENCODER + DECODER                ║
║                                                                     ║
║    Current LTXModel: 48 layers (single pass)                        ║
║    Proposed SCD-LTX: 32-layer encoder + 16-layer decoder            ║
║                      (or 36+12, etc.)                               ║
║                                                                     ║
║    ┌───────────────────────────────────────────────┐                ║
║    │ class LTXSCDModel(nn.Module):                 │                ║
║    │     def __init__(self):                       │                ║
║    │         self.encoder = LTXEncoder(            │                ║
║    │             num_layers=32,                    │                ║
║    │             # reuse LTX transformer blocks    │                ║
║    │             # ADD causal temporal mask         │                ║
║    │             # ADD KV-cache support             │                ║
║    │         )                                     │                ║
║    │         self.decoder = LTXDecoder(            │                ║
║    │             num_layers=16,                    │                ║
║    │             # reuse LTX transformer blocks    │                ║
║    │             # ADD encoder feature injection   │                ║
║    │             # NO causal mask (per-frame)       │                ║
║    │         )                                     │                ║
║    └───────────────────────────────────────────────┘                ║
║                                                                     ║
║ 2. ADD CAUSAL ATTENTION MASK TO ENCODER                             ║
║                                                                     ║
║    SCD's mask is FRAME-LEVEL causal, not token-level:               ║
║    - All tokens in frame t can see all tokens in frames ≤ t         ║
║    - Implemented as block-diagonal mask                             ║
║                                                                     ║
║    In LTX-2 terms:                                                  ║
║    - Modality.self_attn_mask is currently UNUSED                    ║
║    - Must inject causal mask into BasicAVTransformerBlock            ║
║    - Or use Modality.rcl_split_point mechanism                      ║
║                                                                     ║
║ 3. ADD KV-CACHE TO ENCODER                                         ║
║                                                                     ║
║    SCD caches encoder KV for ALL prior frames:                      ║
║    ┌───────────────────────────────────────────────┐                ║
║    │ context_cache = {                             │                ║
║    │   'kv_cache': {                               │                ║
║    │     layer_0: {'key': [...], 'value': [...]},  │                ║
║    │     layer_1: {'key': [...], 'value': [...]},  │                ║
║    │     ...                                       │                ║
║    │   },                                          │                ║
║    │   'is_cache_step': True/False,                │                ║
║    │   'cached_seqlen': N,                         │                ║
║    │ }                                             │                ║
║    └───────────────────────────────────────────────┘                ║
║                                                                     ║
║    Must add KV-cache to LTX-2's attention mechanism                 ║
║                                                                     ║
║ 4. ENCODER-DECODER FEATURE INJECTION                                ║
║                                                                     ║
║    Three strategies (SCD implements all, "token_concat" is best):   ║
║                                                                     ║
║    a) "token_concat": Concatenate encoder tokens before decoder     ║
║       tokens along sequence dim. Decoder attends to both.           ║
║       → Doubled sequence length in decoder                          ║
║       → After processing, take only decoder half                    ║
║                                                                     ║
║    b) "concat": Concatenate along feature dim + linear projection   ║
║       → Same sequence length, richer features                       ║
║                                                                     ║
║    c) "add": Element-wise addition                                  ║
║       → Simplest, least expressive                                  ║
║                                                                     ║
║ 5. REPLACE CONDITIONING MECHANISM                                   ║
║                                                                     ║
║    SCD uses: discrete action classes → LabelEmbedding → AdaLN      ║
║    LTX-2 uses: text → Gemma encoder → cross-attention               ║
║                                                                     ║
║    For the port:                                                    ║
║    - Keep LTX-2's text conditioning for the decoder                 ║
║    - Encoder gets text via cross-attention (already in LTX blocks)  ║
║    - May add temporal position info to encoder                      ║
║                                                                     ║
║ 6. ADAPT TRAINING STRATEGY                                         ║
║                                                                     ║
║    Create new TrainingStrategy subclass:                             ║
║                                                                     ║
║    ┌───────────────────────────────────────────────┐                ║
║    │ class SCDTrainingStrategy(TrainingStrategy):   │                ║
║    │                                                │                ║
║    │   def prepare_training_inputs(self, batch):    │                ║
║    │     # 1. Get video latents (patchified)        │                ║
║    │     # 2. Create per-frame noise                │                ║
║    │     # 3. Apply context masking                 │                ║
║    │     #    (some frames stay clean, t=-1)        │                ║
║    │     # 4. Build Modality for encoder            │                ║
║    │     #    (with causal mask)                    │                ║
║    │     # 5. Build Modality for decoder            │                ║
║    │     #    (per-frame, with encoder features)    │                ║
║    │     return ModelInputs(                        │                ║
║    │       video_encoder=encoder_modality,          │                ║
║    │       video_decoder=decoder_modality,          │                ║
║    │       video_targets=velocity_targets,          │                ║
║    │     )                                          │                ║
║    │                                                │                ║
║    │   def compute_loss(self, pred, inputs):        │                ║
║    │     # Masked MSE on non-context frames         │                ║
║    └───────────────────────────────────────────────┘                ║
║                                                                     ║
║ 7. ADAPT INFERENCE PIPELINE                                         ║
║                                                                     ║
║    Modify TI2VidOneStagePipeline:                                   ║
║    - Generate frames autoregressively                               ║
║    - Encoder runs once per new frame (with KV cache)                ║
║    - Decoder runs N denoising steps per frame                       ║
║    - Significant speedup: encoder is the expensive part             ║
║                                                                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 6. KEY MATHEMATICAL DIFFERENCES

```
┌─────────────────────────────────────────────────────────────────────┐
│ DIFFUSION FORMULATION (IDENTICAL IN BOTH)                          │
│                                                                     │
│ Both use Flow Matching / Rectified Flow:                            │
│                                                                     │
│   Forward:   x_t = (1-σ)·x_0 + σ·ε     where ε ~ N(0,I)          │
│   Target:    v = ε - x_0                 (velocity)                │
│   Loss:      L = ||v_pred - v||²                                    │
│   Sampling:  x_{t-1} = x_t + v·dt       (Euler step)              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ TIMESTEP SAMPLING                                                   │
│                                                                     │
│ SCD: logit_normal (density-based, diffusers util)                  │
│ LTX-2: Custom TimestepSampler (configurable)                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ POSITIONAL ENCODING                                                 │
│                                                                     │
│ SCD: FluxPosEmbed (3D: frame, height, width)                       │
│      axes_dims_rope = (16, 24, 24) for 768-dim                     │
│                                                                     │
│ LTX-2: LTXRopeType.INTERLEAVED                                     │
│        3D: (temporal, height, width)                                │
│        max_pos = [20, 2048, 2048]                                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ KEY SCD-SPECIFIC: CONTEXT TIMESTEP = -1                             │
│                                                                     │
│ Clean/context frames use a special timestep of -1                   │
│ This signals "this frame is already clean, just encode it"          │
│ The encoder always receives timestep=-1 for all frames              │
│ The decoder receives the actual noise level per frame               │
│                                                                     │
│ In LTX-2: Use timestep=0 for conditioning frames (already exists   │
│ in first_frame_conditioning mechanism)                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. FILE-LEVEL PORTING MAP

```
SCD FILE                         → LTX-2 TARGET LOCATION
────────────────────────────────────────────────────────────────────

scd/models/scd_model.py          → ltx-core/src/ltx_core/model/transformer/scd_model.py
  SCDTransformer                     (new: LTXSCDEncoder, LTXSCDDecoder)
  SCDEncoderDecoder                  (new: LTXSCDEncoderDecoder)
  SCDAttnProcessor                   (modify: add KV-cache to LTX attention)
  _build_causal_mask                 (new: causal mask utility)

scd/pipelines/pipeline_scd.py    → ltx-pipelines/src/ltx_pipelines/scd_pipeline.py
  SCDPipeline.generate               (new: autoregressive generation)
  SCDPipeline.__call__               (adapt denoising loop)

scd/trainers/trainer_scd.py      → ltx-trainer/src/ltx_trainer/training_strategies/scd.py
  SCDTrainer.train_step              (new: SCDTrainingStrategy)
  context masking logic              (integrate into prepare_training_inputs)

scd/losses/lpips.py              → (optional, LTX-2 uses MSE)
scd/metrics/                     → (optional, evaluation metrics)
```

---

## 8. CRITICAL IMPLEMENTATION DETAILS

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. ENCODER ONLY RUNS WHEN is_cache_step=True                       │
│    (First denoising step of each new frame)                         │
│    Result is stored in lazy_encoder_output and reused               │
│                                                                     │
│ 2. CAUSAL SHIFT IS KEY                                              │
│    Encoder output for frame t-1 feeds decoder for frame t           │
│    This is the temporal causality mechanism                         │
│                                                                     │
│ 3. DECODER MULTI-BATCH (decoder_multi_batch: 2)                    │
│    Training runs decoder 2x per encoder pass with different noise   │
│    This amortizes the expensive encoder computation                 │
│                                                                     │
│ 4. ENCODER LR vs DECODER LR                                        │
│    encoder_lr_ratio: 1.0, decoder_lr_ratio: 2.0                    │
│    Decoder trains 2x faster (smaller, needs to catch up)            │
│                                                                     │
│ 5. NOISE INJECTION IN ENCODER NORM                                  │
│    norm_out has df_noise_strength=0.05 in encoder                   │
│    Adds small Gaussian noise after LayerNorm                        │
│    Prevents encoder from being too "clean" / overfitting            │
│                                                                     │
│ 6. CLEAN CONTEXT RATIO                                              │
│    10% of training frames are kept clean (no noise added)           │
│    These get timestep=-1, teaching model to handle real context     │
│                                                                     │
│ 7. SHORT_TERM_CTX_WINSIZE                                           │
│    Controls how many recent frames the decoder can see              │
│    Default: 299 (nearly all frames in 300-frame sequence)           │
│    During inference: can be reduced for memory savings              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. SPEEDUP ANALYSIS

```
Standard Causal Diffusion (e.g., original LTX-2):
  Cost per frame = N_steps × (Encoder_cost + Decoder_cost)
  With 48 layers, 50 steps: 48 × 50 = 2400 layer-passes per frame

SCD Architecture:
  Cost per frame = 1 × Encoder_cost + N_steps × Decoder_cost
  With 32-enc + 16-dec, 50 steps: 32 + (16 × 50) = 832 layer-passes

  SPEEDUP ≈ 2400/832 ≈ 2.9x theoretical

  With reduced decoder steps (e.g., 10):
  32 + (16 × 10) = 192 layer-passes → 12.5x speedup

  The encoder can also benefit from KV-cache across frames:
  After first frame, only new frame tokens need processing
```
