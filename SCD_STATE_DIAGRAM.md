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

---

## 10. EDITCTRL + SCD VIDEO EDITING PIPELINE

EditCtrl adds structured video editing to SCD: given a source video, binary edit
masks, and a text prompt, only the masked regions change while preserving the
unmasked source content. EditCtrl introduces LocalContextModule (boundary-aware
local features) and GlobalContextEmbedder (scene-level background context).

### 10A. EDITCTRL ARCHITECTURE OVERVIEW

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    EditCtrl + SCD + TMA Pipeline                             │
│                                                                              │
│  ┌────────────┐   ┌──────────────────────────────────────────────────────┐  │
│  │ Source Video│   │            SCD Backbone                              │  │
│  │ [B,C,F,H,W]│   │                                                      │  │
│  └─────┬──────┘   │  ┌──────────────┐  shift  ┌───────────────┐         │  │
│        │          │  │ Encoder      │ ──────→  │ Decoder       │         │  │
│        ▼          │  │ Layers 0-31  │  (1 frm) │ Layers 32-47  │         │  │
│   ┌─────────┐    │  │ Causal mask  │          │ + EditCtrl     │         │  │
│   │ VAE Enc │    │  │ t=0 (clean)  │          │   signals      │         │  │
│   └────┬────┘    │  └──────────────┘          └───────┬───────┘         │  │
│        │          │                                    │                  │  │
│        ▼          └────────────────────────────────────┼──────────────────┘  │
│   ┌──────────┐                                         │                     │
│   │ Patchify │─── source_patchified ──────────┐       │                     │
│   └──────────┘         [B, seq, C]            │       ▼                     │
│        │                                      │  ┌──────────┐               │
│        │          ┌──────────────────┐        │  │ Velocity │               │
│        ├────────→ │ LocalContextModule│        │  │ Predict  │               │
│        │          │ (sparse tokens +  │        │  └────┬─────┘               │
│        │          │  dilated mask)    │        │       │                     │
│        │          └────────┬─────────┘        │       ▼                     │
│        │                   │ local_control     │  ┌──────────┐               │
│        │                   │ [B, seq, D]       │  │ Euler    │               │
│        │                   │ injected at       │  │ Step     │               │
│        │                   │ decoder layers    │  └────┬─────┘               │
│        │                   │ {0,2,4,...,14}    │       │                     │
│        │                                       │       ▼                     │
│        ├────────→ ┌──────────────────┐         │  ┌───────────────┐          │
│        │          │ GlobalContext     │         │  │ Boundary      │          │
│        │          │ Embedder         │         │  │ Blend         │          │
│        │          │ (bg tokens,      │         │  │ (hard/soft)   │          │
│        │          │  256 pooled)     │         │  └───────┬───────┘          │
│        │          └────────┬─────────┘         │          │                  │
│        │                   │ global_context     │          ▼                  │
│        │                   │ prepended to       │  ┌──────────────┐          │
│        │                   │ cross-attn context  │  │ Re-apply     │          │
│        │                                        │  │ source at    │          │
│        │                                        └──│ unmasked pos │          │
│        │                                           └──────┬───────┘          │
│  ┌─────┴──────┐                                           │                  │
│  │ Edit Masks │                                           ▼                  │
│  │ [B,1,F,H,W]│                                   ┌────────────┐            │
│  └────────────┘                                    │ VAE Decode │            │
│                                                    └─────┬──────┘            │
│  ┌────────────┐                                          │                   │
│  │ Text Prompt│                                          ▼                   │
│  │ (Gemma enc)│                                   ┌────────────┐            │
│  └────────────┘                                   │ Edited     │            │
│                                                    │ Video      │            │
│  ┌────────────┐    ┌─────────┐                    │ [B,C,F,H,W]│            │
│  │ Qwen VL    │───→│ TMA     │──→ prepend to     └────────────┘            │
│  │ Features   │    │ Module  │    text context                               │
│  │ (optional) │    │ 8 tokens│                                               │
│  └────────────┘    └─────────┘                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 10B. EDITCTRL TRAINING STATE MACHINE

```
┌─────────────────────────────────────────────────────────────────────────┐
│        EditCtrl + SCD Training Step (editctrl_scd_strategy.py)          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │ Load pre-encoded     │          │ Load cached Gemma   │
    │ VAE latents          │          │ text embeddings     │
    │ [B, C, f, h, w]     │          │ [B, 1024, 3840]     │
    └─────────┬───────────┘          └──────────┬──────────┘
              │                                 │
              │        ┌────────────────────────┤
              │        │                        │
              │        ▼                        │
              │  ┌──────────────────┐           │
              │  │ Load Qwen VL     │ (if TMA)  │
              │  │ features (cached)│           │
              │  │ [B, seq, 3584]   │           │
              │  └────────┬─────────┘           │
              │           │                     │
              │           ▼                     │
              │  ┌──────────────────┐           │
              │  │ TMA Module       │           │
              │  │ cross-attn +     │           │
              │  │ 3-layer MLP      │           │
              │  │ → 8 tokens       │           │
              │  │ [B, 8, 3840]     │           │
              │  └────────┬─────────┘           │
              │           │                     │
              │           ▼                     │
              │  ┌──────────────────┐           │
              │  │ Prepend to text: │           │
              │  │ [B, 1032, 3840]  │           │
              │  └────────┬─────────┘           │
              │           │                     │
              ▼           ▼                     ▼
    ┌────────────────────────────────────────────────────────────┐
    │ GENERATE EDIT MASKS                                        │
    │                                                            │
    │  mask_source config:                                       │
    │  ┌────────────┬──────────────────┬──────────────────────┐ │
    │  │  "random"   │   "semantic"     │   "mixed"            │ │
    │  │ rectangles  │  MuLAn instance  │  50/50 random +      │ │
    │  │ + ellipses  │  masks (library) │  MuLAn semantic      │ │
    │  │ min 5%      │  real contours   │                      │ │
    │  │ max 60%     │  augmented       │                      │ │
    │  └────────────┴──────────────────┴──────────────────────┘ │
    │                                                            │
    │  generate_semantic_masks() → [B, 1, F, H, W]              │
    │  pixel_mask_to_token_mask() → [B, seq_len] bool           │
    │  dilate_token_mask(dilation=2) → [B, seq_len] bool        │
    └───────────────────────┬────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
    ┌──────────────────┐        ┌──────────────────────────┐
    │ Gather sparse    │        │ Prepare background       │
    │ source tokens    │        │ latents (fill mask=0.5)  │
    │ at dilated mask  │        │ adaptive pool → 256 tok  │
    │ positions        │        │                          │
    └────────┬─────────┘        └─────────────┬────────────┘
             │                                │
             ▼                                ▼
    ┌──────────────────┐        ┌──────────────────────────┐
    │ LocalContext      │        │ GlobalContext             │
    │ Module            │        │ Embedder                 │
    │                   │        │                          │
    │ 2 xformer blocks  │        │ Linear → norm → gate    │
    │ cross-attn to     │        │ [B, 256, D] → [B, n, D] │
    │ text context      │        │                          │
    │ gate_proj → D     │        │ Prepended to text ctx   │
    │                   │        │ for cross-attn in all   │
    │ → local_control   │        │ decoder blocks          │
    │ [B, seq_len, D]   │        │                          │
    └────────┬─────────┘        └─────────────┬────────────┘
             │                                │
             └────────────┬───────────────────┘
                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │ SCD ENCODER (layers 0-31, causal mask, t=0)             │
    │ → encoder_features [B, seq, D]                          │
    │ → shift by 1 frame (causal conditioning)                │
    └─────────────────────┬────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │ SCD DECODER (layers 32-47, actual sigma, no causal mask)│
    │                                                          │
    │ Input: token_concat([shifted_enc_feat, noisy_tokens])   │
    │ + local_control injected at layers {0,2,4,6,8,10,12,14} │
    │ + global_context prepended to cross-attn context         │
    │                                                          │
    │ → velocity_pred [B, seq_len, C]                         │
    └─────────────────────┬────────────────────────────────────┘
                          │
                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │ LOSS (MSE on velocity, edit-masked frames only)          │
    │ loss = MSE(velocity_pred, target) * frame_mask           │
    └──────────────────────────────────────────────────────────┘
```

### 10C. EDITCTRL INFERENCE STATE MACHINE (WITH LAYERFUSION BLENDING)

```
┌─────────────────────────────────────────────────────────────────────────┐
│     pipeline_editctrl_scd.py — EditCtrlSCDPipeline.__call__()           │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 1. VAE ENCODE SOURCE VIDEO            │
        │    source → [B, C, f, h, w]           │
        │    patchify → [B, seq_len, C]         │
        └──────────────────┬────────────────────┘
                           │
                           ▼
        ┌───────────────────────────────────────┐
        │ 2. (OPTIONAL) TMA: QWEN VL FEATURES  │
        │                                       │
        │    Live extraction (4 sampled frames): │
        │    source → Qwen2.5-VL-7B (8-bit)    │
        │    → hidden_states[-1]                │
        │    → TMA cross-attn + MLP             │
        │    → 8 semantic tokens [B, 8, 3840]   │
        │    → prepend to text context          │
        └──────────────────┬────────────────────┘
                           │
                           ▼
        ┌───────────────────────────────────────┐
        │ 3. CONVERT MASKS                      │
        │                                       │
        │    pixel [B,1,F,H,W]                  │
        │      → token [B, seq_len] bool        │
        │      → dilated [B, seq_len] bool      │
        │                                       │
        │  ┌─ if boundary_blend != "hard" ────┐ │
        │  │  Pre-compute soft blend mask:     │ │
        │  │  compute_boundary_blend_mask()    │ │
        │  │  → [B, seq_len] float [0,1]      │ │
        │  │                                   │ │
        │  │  Signed distance field:           │ │
        │  │  ┌──────────────────────┐         │ │
        │  │  │ Iterative erosion:   │         │ │
        │  │  │  interior_dist +=    │         │ │
        │  │  │    erode(mask)       │         │ │
        │  │  │  exterior_dist +=    │         │ │
        │  │  │    erode(~mask)      │         │ │
        │  │  │  signed = int - ext  │         │ │
        │  │  │  alpha = sigmoid(    │         │ │
        │  │  │    s * dist/falloff) │         │ │
        │  │  └──────────────────────┘         │ │
        │  │                                   │ │
        │  │  Result:                          │ │
        │  │   Interior: alpha ≈ 1.0           │ │
        │  │   Boundary: alpha = 0.5 (S-curve) │ │
        │  │   Exterior: alpha ≈ 0.0           │ │
        │  └───────────────────────────────────┘ │
        └──────────────────┬────────────────────┘
                           │
                           ▼
        ┌───────────────────────────────────────┐
        │ 4. SCD ENCODER PASS                   │
        │    (causal mask, t=0, runs once)      │
        │    → encoder_features                 │
        │    → shift_encoder_features(+1 frame) │
        └──────────────────┬────────────────────┘
                           │
                           ▼
        ┌───────────────────────────────────────┐
        │ 5. EDITCTRL CONTEXT MODULES           │
        │                                       │
        │  LocalContextModule:                  │
        │    sparse_tokens at dilated_mask      │
        │    + text cross-attn                  │
        │    + timestep embedding               │
        │    → local_control [B, seq, D]        │
        │                                       │
        │  GlobalContextEmbedder:               │
        │    bg_tokens (masked=0.5, pooled)     │
        │    → global_context [B, 256, D]       │
        └──────────────────┬────────────────────┘
                           │
                           ▼
        ┌───────────────────────────────────────┐
        │ 6. INITIALIZE LATENTS                 │
        │                                       │
        │    noise at masked positions           │
        │    source at unmasked positions        │
        │    (always hard mask for init)         │
        └──────────────────┬────────────────────┘
                           │
                ╔══════════╧══════════════╗
                ║ 7. DENOISING LOOP       ║
                ║ i = 0 → num_steps-1     ║
                ║ σ: 1.0 → 0.0           ║
                ╚══════════╤══════════════╝
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │  forward_decoder(noisy, shifted_enc_feat,    │
        │                  local_control, global_ctx)  │
        │                                              │
        │  ┌── if attention capture enabled ────────┐  │
        │  │  PytorchAttention.capture_weights=True │  │
        │  │  at decoder layers {13,14,15}          │  │
        │  │  → manual softmax(QK^T/√d)            │  │
        │  │  → store _captured_weights             │  │
        │  │  (~600MB for 3 layers)                 │  │
        │  └────────────────────────────────────────┘  │
        │                                              │
        │  → velocity_pred [B, seq_len, C]             │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │  EULER STEP                                   │
        │  dt = σ_next - σ                              │
        │  noisy_latents += dt * velocity_pred          │
        └──────────────────┬───────────────────────────┘
                           │
                           ▼
        ╔══════════════════════════════════════════════════════════════╗
        ║           BOUNDARY BLEND DECISION                           ║
        ║                                                             ║
        ║  boundary_blend_mode?                                       ║
        ║                                                             ║
        ║  ┌──────────┐  ┌──────────────────────────────────────────┐║
        ║  │  "hard"   │  │  "distance" / "attention" / "hybrid"    │║
        ║  │           │  │                                          │║
        ║  │ torch.    │  │  step < threshold?                       │║
        ║  │ where()   │  │    YES → hard mask (noise too high)      │║
        ║  │ bit-exact │  │    NO  → soft blending:                  │║
        ║  │ original  │  │                                          │║
        ║  │ behavior  │  │  ┌─────────────────────────────────────┐│║
        ║  │           │  │  │ COSINE RAMP SCHEDULE                ││║
        ║  │           │  │  │                                     ││║
        ║  │           │  │  │ t = (step - threshold) / remaining  ││║
        ║  │           │  │  │ blend_strength = ½(1 - cos(πt))     ││║
        ║  │           │  │  │                                     ││║
        ║  │           │  │  │   1.0 ─       ╭──────               ││║
        ║  │           │  │  │        │     ╱                      ││║
        ║  │           │  │  │ blend  │   ╱                        ││║
        ║  │           │  │  │        │ ╱                          ││║
        ║  │           │  │  │   0.0 ─╯─────┬───────              ││║
        ║  │           │  │  │        0   thresh   1.0             ││║
        ║  │           │  │  │             step fraction           ││║
        ║  │           │  │  └─────────────────────────────────────┘│║
        ║  │           │  │                                          │║
        ║  │           │  │  effective_mask =                        │║
        ║  │           │  │    (1-blend)*hard + blend*soft_mask      │║
        ║  │           │  │                                          │║
        ║  │           │  │  noisy = eff_mask * denoised             │║
        ║  │           │  │        + (1-eff_mask) * source           │║
        ║  └──────────┘  └──────────────────────────────────────────┘║
        ╚══════════════════════════════════════════════════════════════╝
                           │
                           ▼
                    [back to denoising loop]

                           │ (after all steps)
                           ▼
        ┌───────────────────────────────────────┐
        │ 8. VAE DECODE → edited video          │
        │    + return blend_mask for debug viz   │
        └───────────────────────────────────────┘
```

### 10D. LAYERFUSION BLENDING — THREE PHASES

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LAYERFUSION BLENDING PHASES                          │
│                     (arXiv:2412.04460 adaptation)                        │
│                                                                          │
│  Phase 1: DISTANCE-BASED (training-free, 0 VRAM overhead)               │
│  ─────────────────────────────────────────────────────────               │
│                                                                          │
│    Binary mask:     ░░░░████████░░░░   (hard boundary)                  │
│                          ↓                                               │
│    Erosion (×N):    compute signed distance from boundary               │
│                          ↓                                               │
│    Sigmoid:         ░░▒▒▓███████▓▒▒░   (smooth S-curve)                │
│                                                                          │
│    Spatial view (8×8 latent grid, edit region in center):               │
│    ┌────────────────────────────────┐                                   │
│    │ 0.01 0.01 0.01 0.01 0.01 0.01 │  Exterior: ~0                     │
│    │ 0.01 0.50 0.50 0.50 0.50 0.01 │  Boundary: 0.5                    │
│    │ 0.01 0.50 0.99 0.99 0.50 0.01 │  Interior: ~1                     │
│    │ 0.01 0.50 0.99 0.99 0.50 0.01 │                                   │
│    │ 0.01 0.50 0.50 0.50 0.50 0.01 │  falloff_tokens=2                 │
│    │ 0.01 0.01 0.01 0.01 0.01 0.01 │  sharpness=10                     │
│    └────────────────────────────────┘                                   │
│                                                                          │
│    File: scd/utils/mask_utils.py::compute_boundary_blend_mask()         │
│                                                                          │
│  Phase 2: ATTENTION-DERIVED (requires weight capture, +600MB VRAM)      │
│  ─────────────────────────────────────────────────────────────           │
│                                                                          │
│    Self-attention sparsity (structural edges):                          │
│    ┌────────────────────────────────────┐                               │
│    │  s_i = 1 / Σ_j(m²_i,j)           │  m = attention weights        │
│    │  s'_i = 1 - normalize(s)          │  high at boundaries           │
│    └────────────────────────────────────┘                               │
│         ×                                                                │
│    Cross-attention content confidence:                                  │
│    ┌────────────────────────────────────┐                               │
│    │  c = mean_heads(CA[:,:,:,eos_idx]) │  high at content areas       │
│    │  c' = normalize(c)                 │                               │
│    └────────────────────────────────────┘                               │
│         =                                                                │
│    Soft mask = normalize(s' × c')                                       │
│                                                                          │
│    Requires: PytorchAttention.capture_weights = True                    │
│    Capture at: decoder layers {13, 14, 15}                              │
│    Manual softmax (no SDPA fusion) — inference only                     │
│                                                                          │
│    File: scd/utils/boundary_blend.py::BoundaryBlendModule               │
│    File: ltx-core/.../attention.py::PytorchAttention                    │
│                                                                          │
│  Phase 3: HYBRID (distance base + attention refinement)                 │
│  ──────────────────────────────────────────────────────                  │
│                                                                          │
│    hybrid_mask = w * distance_mask + (1-w) * attention_mask             │
│    w = hybrid_distance_weight (default 0.3)                             │
│                                                                          │
│    Distance provides stable base shape (always available).              │
│    Attention refines boundaries with data-dependent detail.             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10E. TMA (TASK-ADAPTIVE MULTIMODAL ALIGNMENT)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TMA MODULE — Qwen VL Semantic Guidance                │
│                                                                          │
│  Problem: Gemma text embeddings lack visual-semantic understanding      │
│  Solution: Extract visual features from Qwen2.5-VL-7B and project      │
│            them into 8 learned tokens prepended to the text context     │
│                                                                          │
│  ┌───────────────┐    ┌────────────────────────────────────────────┐    │
│  │ Source Video   │    │                TMA Module                  │    │
│  │ (4 frames)    │    │                                            │    │
│  └──────┬────────┘    │  ┌──────────────┐   ┌───────────────────┐ │    │
│         │             │  │ MetaQueryBank │   │ Cross-Attention   │ │    │
│         ▼             │  │ [K, 8, D]     │   │ Q = meta_queries  │ │    │
│  ┌──────────────┐     │  │ K = num_tasks │   │ K,V = qwen_feat   │ │    │
│  │ Qwen2.5-VL   │     │  └──────┬───────┘   └─────────┬─────────┘ │    │
│  │ 7B-Instruct  │     │         │                      │           │    │
│  │ (8-bit quant) │     │         ▼                      │           │    │
│  │              │     │  ┌──────────────────────────────┘           │    │
│  │ hidden_dim   │     │  │                                          │    │
│  │ = 3584       │     │  ▼                                          │    │
│  └──────┬───────┘     │  ┌──────────────┐   ┌───────────────────┐ │    │
│         │             │  │ Attended      │──→│ 3-Layer MLP       │ │    │
│         ▼             │  │ Queries       │   │ Connector         │ │    │
│  ┌──────────────┐     │  │ [B, 8, 3584]  │   │ 3584 → 3840      │ │    │
│  │ Features     │     │  └──────────────┘   │ (LayerNorm + GELU)│ │    │
│  │ [B, seq,     │     │                     └─────────┬─────────┘ │    │
│  │  3584]       │─────│─────────────────────────────── │           │    │
│  └──────────────┘     │                               ▼           │    │
│                       │                     ┌───────────────────┐ │    │
│                       │                     │ 8 Semantic Tokens │ │    │
│                       │                     │ [B, 8, 3840]      │ │    │
│                       │                     └─────────┬─────────┘ │    │
│                       └───────────────────────────────┼────────────┘    │
│                                                       │                  │
│                                                       ▼                  │
│                                            ┌─────────────────────┐      │
│                                            │ Prepend to Gemma    │      │
│                                            │ text embeddings:    │      │
│                                            │ [B, 8+1024, 3840]  │      │
│                                            │ = [B, 1032, 3840]  │      │
│                                            └─────────────────────┘      │
│                                                                          │
│  Trainable params: ~91M (MetaQueryBank + cross-attn + MLP connector)    │
│  VRAM impact: ~78MB (negligible)                                         │
│                                                                          │
│  Training:                                                               │
│    Pre-computed: scripts/compute_qwen_features.py                       │
│    Saved to:     qwen_vl_features/{idx:03d}.pt                          │
│    Source:       /media/12TB/.../new_grok_frames/{video_id}_{frame}.jpg  │
│                                                                          │
│  Inference:                                                              │
│    Live extraction: _extract_qwen_features_live()                       │
│    Model loaded in 8-bit on cuda:1, freed after extraction              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. MULAN DATASET INTEGRATION

```
┌─────────────────────────────────────────────────────────────────────────┐
│              MuLAn — Multi-Layer Annotated Dataset (CVPR 2024)           │
│              https://huggingface.co/datasets/mulan-dataset/v1.0         │
│              https://mulan-dataset.github.io/                           │
│                                                                          │
│  44K+ images with instance-level RGBA layer decompositions              │
│  Sources: COCO 2017 + LAION Aesthetic v2 6.5+                           │
│                                                                          │
│  Format (decomposed):                                                   │
│    images/{image_id}/                                                   │
│      layer_0.png  ─── background (full alpha)                           │
│      layer_1.png  ─── instance 1 (RGBA, alpha = instance mask)          │
│      layer_2.png  ─── instance 2 (RGBA)                                 │
│      ...                                                                │
│                                                                          │
│  Pipeline:                                                              │
│                                                                          │
│  ┌─────────────┐     ┌──────────────────────┐     ┌──────────────────┐ │
│  │ MuLAn HF    │────→│ extract_mulan_masks  │────→│ Mask Library     │ │
│  │ dataset     │     │ .py                  │     │ {id}.pt files    │ │
│  │             │     │                      │     │ + index.pt       │ │
│  │ 44K images  │     │ - Walk layer dirs    │     │                  │ │
│  │ RGBA layers │     │ - Extract alphas     │     │ Per file:        │ │
│  │             │     │ - Filter by area     │     │  masks: [N,H,W]  │ │
│  │             │     │ - Resize to 64×64    │     │  areas: [N]      │ │
│  │             │     │ - Parallel (8 workers)│     │  image_size: HxW │ │
│  └─────────────┘     └──────────────────────┘     └────────┬─────────┘ │
│                                                            │            │
│                                                            ▼            │
│                                              ┌──────────────────────┐  │
│                                              │ SemanticMaskLibrary  │  │
│                                              │                      │  │
│                                              │ Lazy-loaded on first │  │
│                                              │ sample() call        │  │
│                                              │                      │  │
│                                              │ sample_masks():      │  │
│                                              │  1. Filter by area   │  │
│                                              │  2. Random select    │  │
│                                              │  3. Resize to target │  │
│                                              │  4. Augment (flip,   │  │
│                                              │     rotate, scale)   │  │
│                                              │  5. Broadcast frames │  │
│                                              │                      │  │
│                                              │ Falls back to random │  │
│                                              │ masks if unavailable │  │
│                                              └──────────────────────┘  │
│                                                                          │
│  Training config (editctrl_scd_tma.yaml):                               │
│    mask_source: mixed                                                   │
│    semantic_mask_dir: /media/2TB/mulan_masks                            │
│    semantic_mask_ratio: 0.5                                             │
│                                                                          │
│  VRAM: ~200MB (CPU RAM for mask library, not GPU)                       │
│                                                                          │
│  Download:                                                              │
│    huggingface-cli download mulan-dataset/v1.0 --local-dir mulan_raw   │
│    python scripts/extract_mulan_masks.py \                              │
│      --input_dir mulan_raw/images \                                     │
│      --output_dir /media/2TB/mulan_masks \                              │
│      --target_size 64 --workers 8                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. COMPLETE FILE MAP (CURRENT STATE)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    sparse-causal-diffusion/ (this repo)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  scd/                                                                    │
│  ├── pipelines/                                                          │
│  │   └── pipeline_editctrl_scd.py  ─── EditCtrl+SCD inference pipeline  │
│  │       EditCtrlSCDPipeline.__call__():                                │
│  │         VAE enc → masks → TMA → encoder → LCM/GCE → denoise loop   │
│  │         + LayerFusion boundary blending (hard/distance/attn/hybrid)  │
│  │                                                                      │
│  └── utils/                                                              │
│      ├── mask_utils.py  ──────────── Mask conversion, generation        │
│      │   pixel_mask_to_token_mask()    VAE→token space                  │
│      │   dilate_token_mask()           Boundary expansion               │
│      │   compute_boundary_blend_mask() Signed-distance soft mask        │
│      │   gather/scatter_masked_tokens()  Sparse token ops               │
│      │   prepare_background_latents()  GCE input prep                   │
│      │   generate_random_masks()       Rectangles + ellipses            │
│      │   generate_random_token_masks() Direct token-space masks         │
│      │   SemanticMaskLibrary           MuLAn mask loader                │
│      │   generate_semantic_masks()     Unified: random/semantic/mixed   │
│      │                                                                   │
│      └── boundary_blend.py  ────────── LayerFusion blending module      │
│          BlendConfig                   Mode, falloff, sharpness, etc.   │
│          BoundaryBlendModule           Attn-derived soft masks           │
│            .set_binary_mask()          Pre-compute distance              │
│            .update_attention_maps()    Accumulate per-step               │
│            .get_blend_mask()           Return soft mask                  │
│            ._compute_attention_mask()  Sparsity × content formula       │
│                                                                          │
│  configs/                                                                │
│  ├── scd_finetune.yaml  ──────────── SCD LoRA base training             │
│  ├── editctrl_scd_phase1.yaml  ───── EditCtrl Phase 1 (LCM only)       │
│  ├── editctrl_scd_phase2.yaml  ───── EditCtrl Phase 2 (LCM + GCE)      │
│  └── editctrl_scd_tma.yaml  ──────── EditCtrl + TMA + mask config       │
│      mask_source: random|semantic|mixed                                  │
│      semantic_mask_dir, semantic_mask_ratio                              │
│                                                                          │
│  inference/                                                              │
│  └── run_editctrl_inference.py  ───── CLI inference script               │
│      --boundary_blend hard|distance|attention|hybrid                     │
│      --boundary_falloff 3  --boundary_sharpness 10                      │
│      --boundary_step_threshold 0.5                                      │
│      --tma_checkpoint, --qwen_model_path                                │
│                                                                          │
│  scripts/                                                                │
│  ├── compute_qwen_features.py  ────── Pre-compute Qwen VL features     │
│  ├── compute_semantic_masks.py  ───── Semantic mask utilities           │
│  └── extract_mulan_masks.py  ──────── MuLAn → mask library             │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                    ltx-core/ (editable install)                           │
│                    /home/johndpope/.../LTX-2/packages/ltx-core           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  .../model/transformer/                                                  │
│  ├── scd_model.py  ───────────────── LTXSCDModel wrapper                │
│  │   KVCache                         Per-layer K/V for AR encoder       │
│  │   build_frame_causal_mask()       Frame-level causal mask            │
│  │   LTXSCDModel                     Splits 48→32 enc + 16 dec         │
│  │     .forward_encoder()            Causal, t=0, optional KV-cache    │
│  │     .forward_decoder()            Per-frame, actual sigma            │
│  │       + local_control injection   At layers {0,2,4,...,14}           │
│  │       + global_context prepend    To cross-attn context              │
│  │       + capture_attention_layers  For LayerFusion blending           │
│  │   shift_encoder_features()        Causal shift by 1 frame           │
│  │                                                                      │
│  ├── attention.py  ───────────────── Attention backends                 │
│  │   PytorchAttention               + capture_weights flag              │
│  │     .capture_weights: bool       Manual softmax when True            │
│  │     ._captured_weights: Tensor   Stored for BoundaryBlendModule     │
│  │   XFormersAttention                                                   │
│  │   FlashAttention3                                                     │
│  │   Attention                      + kv_cache param for SCD            │
│  │                                                                      │
│  ├── editctrl_modules.py  ────────── EditCtrl trainable modules         │
│  │   LocalContextModule             Sparse tokens → xformer → gate     │
│  │   GlobalContextEmbedder          Background pool → linear            │
│  │                                                                      │
│  └── transformer.py, modality.py, transformer_args.py                   │
│      (self_attn_mask field, kv_cache threading)                          │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                    ltx2-omnitransfer/ (trainer)                           │
│                    /home/johndpope/.../ltx2-omnitransfer                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  .../training_strategies/                                                │
│  ├── scd_strategy.py  ────────────── SCDTrainingStrategy                │
│  │   Encoder + decoder split forward pass                               │
│  │   Per-frame noise sampling + context masking                         │
│  │   decoder_multi_batch for amortized encoder cost                     │
│  │                                                                      │
│  └── editctrl_scd_strategy.py  ───── EditCtrl + SCD strategy            │
│      Phase 1: LCM + LoRA (freeze SCD base)                              │
│      Phase 2: + GCE + unfreeze LoRA                                     │
│      + TMA integration (Qwen VL features)                               │
│      + mask_source: random | semantic | mixed                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 13. VRAM BUDGET (RTX 5090 + Blackwell RTX PRO 4000)

```
                        TRAINING                    INFERENCE
Component              cuda:0 (5090)  cuda:1       cuda:0 (5090)
────────────────────── ────────────── ────────── ──────────────
LTX-2 (int8-quanto)    ~10 GB         —          ~10 GB
SCD LoRA (r=32)         ~0.3 GB        —          ~0.3 GB
EditCtrl LCM            ~0.1 GB        —          ~0.1 GB
EditCtrl GCE            ~0.05 GB       —          ~0.05 GB
TMA Module              ~0.08 GB       —          ~0.08 GB
VAE Encoder             ~2 GB          —          ~2 GB
VAE Decoder             —             ~2 GB       ~2 GB
Text Encoder            ~4 GB          —          (freed after)
Activations + KV        ~12 GB         —          ~4 GB
Optimizer states        ~4 GB          —          —
─────────────────────────────────────────────────────────────
Subtotal                ~33 GB         ~2 GB      ~19 GB

LayerFusion Overhead (inference only):
  Phase 1 (distance)    —              —          +0.1 MB
  Phase 2 (attention)   —              —          +600 MB
  Phase 3 (hybrid)      —              —          +600 MB

MuLAn mask library      +200 MB (CPU RAM, not GPU)
─────────────────────────────────────────────────────────────
Total                   ~33 GB         ~2 GB      ~19.6 GB
Headroom                ~0.7 GB        ~23 GB     ~14 GB
```

---

## 14. VACE ARCHITECTURE ANALYSIS (ControlNet-style Bypass Network)

VACE (All-in-One Video Creation and Editing, ICCV 2025, Alibaba) is the base model
EditCtrl is built on. It provides the actual editing capability through a ControlNet-style
parallel bypass network. Our EditCtrl+SCD uses a different approach — understanding the
differences explains why our paired edit training fails.

**Paper**: arXiv:2503.07598 — https://arxiv.org/abs/2503.07598
**Code**: https://github.com/ali-vilab/VACE

### 14A. VACE-LTX FORWARD PASS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              VaceTransformer3DModel.forward()                                │
│              (vace/models/ltx/models/transformers/transformer3d.py)          │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
   ┌──────────────────┐        ┌──────────────────────────┐
   │ hidden_states     │        │ source_latents            │
   │ (noisy video)     │        │ (source video + mask)     │
   │ [B, seq_len, C]   │        │ [B, seq_len, C_ctx]       │
   └────────┬─────────┘        └──────────┬───────────────┘
            │                             │
            ▼                             ▼
   ┌──────────────────┐        ┌──────────────────────────┐
   │ patchify_proj     │        │ patchify_context_proj     │
   │ Linear(C→D)       │        │ Linear(C_ctx→D)           │
   │ D = inner_dim     │        │ 384 → inner_dim           │
   └────────┬─────────┘        └──────────┬───────────────┘
            │                             │
            │ + RoPE pos embed            │
            │ + AdaLN timestep embed      │
            ▼                             ▼
   ┌──────────────────┐        ╔══════════════════════════╗
   │ hidden_states     │        ║ BYPASS BLOCKS (parallel) ║
   │ [B, seq, D]       │        ║ transformer_context_      ║
   │                   │        ║ blocks[0..N]              ║
   └────────┬─────────┘        ╚════════════╤═════════════╝
            │                               │
            │         ┌─────────────────────┘
            │         │
            │         ▼
            │  ┌─────────────────────────────────────────────────────┐
            │  │ BYPASS BLOCK 0 (BasicTransformerBypassBlock):       │
            │  │                                                      │
            │  │   context = before_proj(context) + hidden_states    │
            │  │                    ↑ ZERO-INIT             ↑ MAIN   │
            │  │                                                      │
            │  │   context = transformer_block(context)               │
            │  │     (self-attn + cross-attn to text + FFN)          │
            │  │                                                      │
            │  │   hint = after_proj(context)                        │
            │  │            ↑ ZERO-INIT                              │
            │  │                                                      │
            │  │   return (hint, context)                             │
            │  └─────────────────────┬──────────┬────────────────────┘
            │                        │ hint     │ context
            │                        ▼          ▼
            │  ┌─────────────────────────────────────────────────────┐
            │  │ BYPASS BLOCK k:                                     │
            │  │   context = transformer_block(context)              │
            │  │   hint_k = after_proj(context)                     │
            │  │   return (hint_k, context)                          │
            │  └─────────────────────┬──────────┬────────────────────┘
            │                        │          │
            │            ┌───────────┘          └──→ (continues...)
            │            │
            │            ▼
            │  ┌─────────────────────────────┐
            │  │ context_hints = [hint_0,     │
            │  │   hint_1, ..., hint_N]      │
            │  │ One per bypass block layer   │
            │  └──────────────┬──────────────┘
            │                 │
            ▼                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ MAIN BLOCKS (BasicTransformerMainBlock):                      │
   │                                                                │
   │  FOR block_idx in range(num_layers):                          │
   │    hidden_states = main_transformer_block(hidden_states)      │
   │                     (self-attn + cross-attn to text + FFN)    │
   │                                                                │
   │    if block_idx in context_num_layers:                        │
   │      hidden_states += context_hints[block_idx] * context_scale│
   │                       ↑                          ↑            │
   │                    bypass hint              inference knob     │
   │                    (zero at init)          (default 1.0)      │
   │                                                                │
   └────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
   ┌──────────────────────────────────┐
   │ OUTPUT                            │
   │ norm_out → scale_shift → proj_out │
   │ → velocity prediction [B, seq, C] │
   └──────────────────────────────────┘
```

### 14B. VACE SOURCE DECOMPOSITION (THE KEY "SMART")

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              VACE Source Encoding — Decomposed Background + Foreground       │
│                                                                              │
│  The most important architectural difference from our approach!              │
│                                                                              │
│  ┌───────────────┐     ┌───────────────┐                                    │
│  │ Source Video   │     │ Edit Mask     │                                    │
│  │ [C, F, H, W]  │     │ [1, F, H, W]  │                                    │
│  └──────┬────────┘     │ 0=keep 1=edit │                                    │
│         │               └──────┬────────┘                                    │
│         │                      │                                             │
│         └──────────┬───────────┘                                             │
│                    │                                                         │
│         ┌──────────┴──────────┐                                              │
│         ▼                     ▼                                              │
│  ┌──────────────┐     ┌──────────────┐                                      │
│  │ UNCHANGED     │     │ CHANGED       │                                      │
│  │ = src*(1-mask)│     │ = src*mask    │                                      │
│  │ (background)  │     │ (foreground)  │                                      │
│  │ edit region=0 │     │ keep region=0 │                                      │
│  └──────┬───────┘     └──────┬───────┘                                      │
│         │                    │                                               │
│         ▼                    ▼                                               │
│  ┌──────────────┐     ┌──────────────┐                                      │
│  │ VAE encode   │     │ VAE encode   │  ← TWO separate VAE passes!          │
│  │ [C', F', H', │     │ [C', F', H', │                                      │
│  │  W']         │     │  W']         │                                      │
│  └──────┬───────┘     └──────┬───────┘                                      │
│         │                    │                                               │
│         └────────┬───────────┘                                               │
│                  ▼                                                            │
│  ┌──────────────────────────────────┐                                        │
│  │ CONCAT along channel dim:        │                                        │
│  │ source_latents = cat(unchanged,  │                                        │
│  │                      changed)    │                                        │
│  │ [2*C', F', H', W']              │  ← 2x channel width!                  │
│  └──────────────┬───────────────────┘                                        │
│                  │                                                            │
│                  │  (optional: prepend reference image latents)               │
│                  ▼                                                            │
│  ┌──────────────────────────────────┐                                        │
│  │ Patchify → [B, seq_len, 2*C']    │                                        │
│  └──────────────┬───────────────────┘                                        │
│                  │                                                            │
│                  ▼                                                            │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │ MASK ENCODING (separate path):                            │                │
│  │                                                            │                │
│  │ mask [1, F, H, W]                                          │                │
│  │   → reshape to [stride_h*stride_w, F', H', W']            │                │
│  │   → interpolate to match latent dims                       │                │
│  │   → patchify → [B, seq_len, stride_h*stride_w]            │                │
│  │                                                            │                │
│  │ This gives per-token mask information at SUB-PATCH level! │                │
│  │ (8x8 = 64 mask values per token for LTX-2's stride)      │                │
│  └──────────────────────────┬─────────────────────────────────┘                │
│                              │                                                 │
│                              ▼                                                 │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │ COMBINE: source_latents = cat(source_latents,            │                │
│  │                               source_mask_latents, dim=-1)│                │
│  │ [B, seq_len, 2*C' + 64]                                   │                │
│  │                                                            │                │
│  │ This is the in_context_channels = 384:                    │                │
│  │   2*128 (unchanged+changed latents) + 64 (mask) + padding │                │
│  └──────────────────────────┬─────────────────────────────────┘                │
│                              │                                                 │
│                              ▼                                                 │
│  ┌──────────────────────────────────────────────────────────┐                │
│  │ patchify_context_proj: Linear(384 → inner_dim)            │                │
│  │ → context_hidden_states [B, seq_len, D]                   │                │
│  │ → fed into bypass transformer blocks                      │                │
│  └──────────────────────────────────────────────────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 14C. VACE-WAN ARCHITECTURE (Wan-2.1 14B variant)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              VaceWanModel — Wan-2.1 14B with VACE bypass                     │
│              (vace/models/wan/modules/model.py)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
   ┌──────────────────────┐     ┌──────────────────────────────────┐
   │ x (noisy latents)    │     │ vace_context                     │
   │ [C_in, F, H, W]      │     │ = cat(source_latents, mask)      │
   │                       │     │ [vace_in_dim, F', H', W']       │
   │ patch_embedding       │     │                                  │
   │ Conv3d(in_dim→dim)    │     │ vace_patch_embedding             │
   │                       │     │ Conv3d(vace_in_dim→dim)          │
   └────────┬─────────────┘     └──────────────┬───────────────────┘
            │                                  │
            │ + timestep embed                 │ pad to seq_len
            │ + text embedding                 │
            ▼                                  ▼
   ┌─────────────────┐             ┌──────────────────────────────┐
   │ x [B, seq, dim]  │             │ forward_vace():               │
   │                   │             │                               │
   │                   │             │ FOR each vace_block:          │
   │                   │             │   block_0:                    │
   │                   │      ┌──────│     c = before_proj(c) + x   │
   │                   │      │      │   all blocks:                 │
   │                   │      │      │     c = WanAttentionBlock(c)  │
   │                   │      │      │     hint = after_proj(c)      │
   │                   │      │      │                               │
   │                   │      │      │ hints = all collected hints   │
   └────────┬─────────┘      │      └──────────────┬───────────────┘
            │                 │                     │
            │      x passed──┘                      │
            │      to block 0                       │
            │      of bypass                        │
            ▼                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ MAIN BLOCKS with hint injection:                                  │
   │                                                                    │
   │ vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,       │
   │                24, 26, 28, 30]  (every other, for 32-layer Wan)   │
   │                                                                    │
   │ FOR i, block in enumerate(blocks):                                │
   │   x = WanAttentionBlock(x, text_context, timestep)                │
   │   if i in vace_layers:                                            │
   │     x = x + hints[vace_layers_mapping[i]] * context_scale         │
   │                                                                    │
   │ 32 main blocks, 16 bypass blocks (at even-numbered layers)        │
   └──────────────────────────────────────────────────────────────────┘
```

### 14D. CRITICAL DIFFS: VACE vs OUR EDITCTRL+SCD

```
┌──────────────────────┬───────────────────────────────┬─────────────────────────────┐
│ Feature              │ VACE                          │ Our EditCtrl+SCD            │
├──────────────────────┼───────────────────────────────┼─────────────────────────────┤
│ BASE MODEL           │ LTX-Video-2B or Wan-14B       │ LTX-2 19B                   │
│                      │ (full model, not LoRA)         │ (int8-quanto + LoRA r32)    │
│                      │                                │                             │
│ CONDITIONING         │ ControlNet-style BYPASS        │ LCM (sparse tokens) +       │
│ ARCHITECTURE         │ NETWORK running parallel       │ GCE (background pool) as    │
│                      │ transformer blocks             │ separate small modules      │
│                      │                                │                             │
│ SOURCE ENCODING      │ DECOMPOSED:                    │ DIRECT:                     │
│                      │ unchanged = src*(1-mask)        │ source_patchified = whole   │
│                      │ changed = src*mask              │ source video patchified     │
│                      │ → TWO VAE encodes              │ → ONE patchify pass         │
│                      │ → cat along channel (2x)       │                             │
│                      │                                │                             │
│ MASK INPUT           │ DIRECT as extra channels:      │ INDIRECT:                   │
│                      │ mask patchified → 64 channels  │ token-level bool mask       │
│                      │ fed alongside source latents   │ (binary, not per-subpixel)  │
│                      │ → model sees SPATIAL mask      │ used for sparse gathering   │
│                      │                                │                             │
│ TOTAL CONTEXT        │ 2*128 + 64 = 320 channels     │ sparse_tokens: ~1200 tokens │
│ CHANNELS             │ (+ padding = 384)              │ bg_latents: 256 pooled tok  │
│                      │ per token, DENSE               │ NOT per-token conditioning  │
│                      │                                │                             │
│ INJECTION METHOD     │ ADDITIVE HINTS:                │ LCM: additive at decoder    │
│                      │ x += hint[layer] * scale       │ layers {0,2,4,...,14}       │
│                      │ at every-other main layer      │ GCE: prepend to cross-attn  │
│                      │                                │ context (text concat)       │
│                      │                                │                             │
│ ZERO INIT            │ YES — before_proj + after_proj │ LCM has gate_proj           │
│                      │ initialized to ZERO            │ (not zero-init by default)  │
│                      │ → starts from identity         │                             │
│                      │                                │                             │
│ BYPASS BLOCK COUNT   │ Wan: 16 blocks (every other)   │ LCM: 2 transformer blocks   │
│                      │ LTX: configurable              │ GCE: 1 linear + norm        │
│                      │ (context_num_layers)            │                             │
│                      │                                │                             │
│ BYPASS CAPACITY      │ FULL transformer blocks:       │ SMALL modules:              │
│                      │ self-attn + cross-attn + FFN   │ ~22M (LCM) + ~91M (TMA)    │
│                      │ × 16 = ~7B params for Wan-14B  │ ~113M total trainable       │
│                      │ or proportional for LTX-2B     │                             │
│                      │                                │                             │
│ EDITING CAPABILITY   │ PRE-TRAINED on massive paired  │ NEVER TRAINED for editing   │
│                      │ edit data via VACE training     │ Only reconstruction via     │
│                      │ pipeline (multi-stage)          │ mask inpainting             │
│                      │                                │                             │
│ REFERENCE IMAGES     │ YES — ref images VAE-encoded   │ NO reference image support  │
│                      │ and prepended to context        │                             │
│                      │                                │                             │
│ INFERENCE CONTROL    │ context_scale: 0.0→2.0+        │ edit_strength: 0.0→1.0      │
│                      │ Smoothly controls bypass        │ (decoder text interpolation │
│                      │ contribution strength           │  source↔edit caption)       │
└──────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

### 14E. WHY VACE's APPROACH WORKS FOR EDITING (AND OURS DOESN'T)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                 ROOT CAUSE ANALYSIS                                          ║
║                                                                              ║
║  VACE's 3 key "smarts" that enable editing:                                  ║
║                                                                              ║
║  1. DECOMPOSED SOURCE ENCODING                                               ║
║     ─────────────────────────                                                ║
║     By splitting into unchanged (background) and changed (foreground),       ║
║     the model gets EXPLICIT information about what to preserve vs edit.      ║
║                                                                              ║
║     Our approach: source is fed as-is. The model must LEARN which parts      ║
║     are masked vs unmasked from a binary token mask. This is much harder     ║
║     for a small adapter to learn.                                            ║
║                                                                              ║
║  2. MASSIVE BYPASS CAPACITY                                                  ║
║     ───────────────────────                                                  ║
║     VACE's bypass network has FULL transformer blocks (self-attn +           ║
║     cross-attn + FFN) at every other main layer. For Wan-14B, that's         ║
║     ~7B parameters in the bypass alone.                                      ║
║                                                                              ║
║     Our EditCtrl: LCM has 2 small transformer blocks (~22M params).          ║
║     GCE is a single linear layer. Total ~113M trainable params.              ║
║     The bypass simply doesn't have enough capacity for editing.              ║
║                                                                              ║
║  3. MASK AS DIRECT INPUT CHANNELS                                            ║
║     ────────────────────────────                                             ║
║     VACE patchifies the mask into 64 sub-pixel channels per token and        ║
║     feeds it ALONGSIDE the source latents as input to the bypass.            ║
║     Every bypass block sees exactly where the edit boundary is at            ║
║     sub-patch precision.                                                     ║
║                                                                              ║
║     Our EditCtrl: mask is a binary token-level flag. LCM gathers sparse     ║
║     tokens at masked positions. Less precise, less information.              ║
║                                                                              ║
║  CONCLUSION:                                                                 ║
║  EditCtrl (from the paper) was designed as a CONTROL module on top of        ║
║  VACE, which already has editing capability. It adds LOCAL boundary          ║
║  refinement and GLOBAL context preservation — but the actual editing         ║
║  is done by VACE's large bypass network.                                     ║
║                                                                              ║
║  We tried to make EditCtrl do BOTH control AND editing with 113M params     ║
║  on a base model that doesn't understand editing. That's the mismatch.      ║
║                                                                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 14F. POSSIBLE PATHS FORWARD

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OPTIONS                                             │
│                                                                              │
│  OPTION A: Port VACE's bypass architecture to LTX-2                         │
│  ──────────────────────────────────────────────────                          │
│  Add ControlNet-style bypass blocks to our LTX-2 SCD model.                │
│  Train bypass blocks on paired data with zero-init.                         │
│  Keep existing SCD + LoRA frozen.                                           │
│                                                                              │
│  Pros: Proven architecture, separates control from generation               │
│  Cons: Large param count (~2-4B bypass for 48-layer LTX-2),                │
│         requires significant VRAM for training                               │
│                                                                              │
│  OPTION B: Use VACE-LTX directly as base                                    │
│  ────────────────────────────────────────                                    │
│  Download VACE-LTX checkpoint (pre-trained for editing).                    │
│  Port SCD encoder/decoder split onto VACE-LTX backbone.                    │
│  Use VACE's existing bypass for editing.                                    │
│                                                                              │
│  Pros: Gets editing "for free", proven editing quality                      │
│  Cons: VACE-LTX is based on LTX-Video-2B (older, smaller model),           │
│         not LTX-2 19B. Architecture mismatch.                               │
│                                                                              │
│  OPTION C: Teach editing through base LoRA first                            │
│  ─────────────────────────────────────────────────                           │
│  Unfreeze SCD LoRA, train on paired data with low LR.                       │
│  LoRA has access to ALL 48 layers — enough capacity.                        │
│  Then freeze LoRA, train EditCtrl for boundary control.                     │
│                                                                              │
│  Pros: Minimal arch changes, reuses existing training setup                 │
│  Cons: LoRA r32 may not have enough capacity for full editing               │
│                                                                              │
│  OPTION D: Add VACE-style decomposed encoding to our pipeline               │
│  ─────────────────────────────────────────────────────────────               │
│  Keep our EditCtrl modules but change the source encoding:                  │
│  1. Split source into unchanged+changed halves                              │
│  2. Double the input channels to bypass/LCM                                 │
│  3. Feed patchified mask as extra input channels                            │
│  4. Keep training with paired data                                           │
│                                                                              │
│  Pros: Targeted improvement to most impactful difference                    │
│  Cons: Still limited by LCM/GCE capacity (~113M params)                    │
│                                                                              │
│  OPTION E: Hybrid — VACE bypass (lightweight) + paired data                 │
│  ────────────────────────────────────────────────────────                    │
│  Add a SMALL bypass network (fewer layers, lower dim) to SCD decoder.       │
│  Feed decomposed source + mask (VACE-style).                                │
│  Train bypass on paired edit data with zero-init.                           │
│  Keep SCD base + LoRA frozen.                                               │
│                                                                              │
│  Pros: Best of both — VACE's proven conditioning + our SCD speedup         │
│  Cons: New architecture to implement and debug                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 15. DDiT: DYNAMIC PATCH SCHEDULING FOR EFFICIENT INFERENCE

Based on [DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers](https://arxiv.org/abs/2602.16968).

### 15A. Core Insight

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DDiT: Dynamic Patch Scheduling                          │
│                                                                            │
│  Key Insight: Not all denoising steps need the same spatial resolution     │
│                                                                            │
│  Early steps (t≈1.0):  Coarse global structure → LARGE patches (fewer     │
│                         tokens) → 4-16x fewer tokens → fast               │
│  Later steps (t≈0.0):  Fine detail refinement → SMALL patches (more       │
│                         tokens) → full resolution → quality                │
│                                                                            │
│  Since attention is O(N²), reducing tokens gives quadratic speedup:       │
│  ┌──────────┬─────────────┬──────────────┬───────────┐                    │
│  │ Scale    │ Tokens      │ Attention    │ Speedup   │                    │
│  ├──────────┼─────────────┼──────────────┼───────────┤                    │
│  │ 1x (p)  │ F×H×W       │ O(N²)       │ 1.0x      │                    │
│  │ 2x (2p) │ F×H/2×W/2   │ O(N²/16)   │ ~3.0x     │                    │
│  │ 4x (4p) │ F×H/4×W/4   │ O(N²/256)  │ ~4.5x     │                    │
│  └──────────┴─────────────┴──────────────┴───────────┘                    │
│                                                                            │
│  Combined with SCD:                                                        │
│  ┌──────────────┐       ┌────────────────────────────────┐                │
│  │  Encoder      │ 1x   │  Decoder (DDiT scheduling)     │                │
│  │  (layers 0-31)│ ──→  │  (layers 32-47)                │                │
│  │  runs ONCE    │       │  Step 1-5:  4x patches         │                │
│  │  full res     │       │  Step 6-12: 2x patches         │                │
│  │               │       │  Step 13-20: 1x patches (fine) │                │
│  └──────────────┘       └────────────────────────────────┘                │
│                                                                            │
│  SCD alone: 3x speedup (encoder reuse)                                    │
│  DDiT alone: 1.6-2.2x speedup (dynamic patches)                          │
│  SCD + DDiT: ~4-5x potential compound speedup                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15B. Architecture: Multi-Resolution Patchification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LTX-2 Native vs DDiT Token Flow                         │
│                                                                            │
│  NATIVE (scale=1):                                                         │
│  VAE latent [B,128,F,H,W]                                                 │
│    → patchify(p=1): [B, F×H×W, 128]                                       │
│    → patchify_proj:  Linear(128, 4096)  → [B, F×H×W, 4096]               │
│    → 48 transformer blocks                                                 │
│    → proj_out:       Linear(4096, 128)  → [B, F×H×W, 128]                │
│    → unpatchify: [B,128,F,H,W]                                            │
│                                                                            │
│  DDiT (scale=2):                                                           │
│  VAE latent [B,128,F,H,W]                                                 │
│    → patchify(p=1): [B, F×H×W, 128]                                       │
│    → merge_2x2:     [B, F×H/2×W/2, 512]  ← 4 tokens merged              │
│    → patchify_2x:   Linear(512, 4096)  → [B, F×H/2×W/2, 4096]           │
│    → patch_id:      + learned embedding  (tells model: "I'm 2x scale")   │
│    → 48 transformer blocks  ← 4x FEWER tokens!                            │
│    → proj_out_2x:   Linear(4096, 512)  → [B, F×H/2×W/2, 512]            │
│    → unmerge_2x2:   [B, F×H×W, 128]                                       │
│    → residual_block: + learned refinement from input                       │
│    → unpatchify: [B,128,F,H,W]                                            │
│                                                                            │
│  DDiT (scale=4):                                                           │
│  ... same but merge_4x4: [B, F×H/4×W/4, 2048]                            │
│  → patchify_4x: Linear(2048, 4096) → 16x fewer tokens!                   │
│                                                                            │
│  ┌─────────────────────────────────────────────┐                          │
│  │  DDiTMergeLayer (per scale)                  │                          │
│  │  ├── patchify_proj: Linear(C×s², D)          │ D=4096                  │
│  │  ├── proj_out:      Linear(D, C×s²)          │                          │
│  │  ├── patch_id:      Parameter(1, 1, D)       │ scale identifier        │
│  │  └── residual_block: LN → Linear → GELU → Linear                      │
│  │                                               │                          │
│  │  Init: tile base weights with 1/s² scaling    │ near-identity start    │
│  └─────────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15C. Scheduling: Third-Order Finite Difference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Patch Scale Scheduling Algorithm                         │
│                                                                            │
│  For each denoising step t:                                                │
│                                                                            │
│  1. Record latent z_t                                                      │
│  2. If have 3+ history points:                                             │
│     a. First-order:  Δz_t = z_t - z_{t+1}     (displacement)             │
│     b. Third-order:  Δ³z = acceleration of trajectory                      │
│     c. Reshape Δ³z to spatial grid [B, F, H, W, C]                       │
│     d. For each candidate scale s ∈ {4, 2, 1}:                           │
│        - Divide into s×s patches                                           │
│        - Compute std within each patch → σ                                │
│        - Take 40th percentile of patch stds                               │
│        - If percentile < threshold (0.001): USE THIS SCALE               │
│  3. Default: scale=1 (finest)                                              │
│                                                                            │
│  Intuition:                                                                │
│  ┌────────────────────────────────────────────────────────┐               │
│  │ Step:    1    2    3    4    5 ... 10 ... 15 ... 20    │               │
│  │ Scale:   1    1    4    4    4     2      1      1     │               │
│  │          ↑warmup   ↑coarse→→→→  ↑mid→→  ↑fine→→→→→    │               │
│  │                                                        │               │
│  │ Early: low variance = coarse global structure forming  │               │
│  │ Later: high variance = fine details being refined      │               │
│  └────────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15D. Training: Knowledge Distillation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DDiT Adapter Training (Two-Phase)                        │
│                                                                            │
│  Phase 1: RECONSTRUCTION (no base model needed)                            │
│  ─ Train merge/unmerge layers to preserve spatial information              │
│  ─ L = MSE(unmerge(merge(z)), z)                                          │
│  ─ 200 steps, 2 seconds, loss 1.35 → 0.12                                │
│                                                                            │
│  Phase 2: DISTILLATION (through quantized base model + LoRA)               │
│                                                                            │
│  ┌──────────────┐  same z_t  ┌──────────────────┐                        │
│  │ Teacher      │  ────────→ │ Student           │                        │
│  │ base model   │  same σ    │ DDiT adapter      │                        │
│  │ LoRA OFF     │            │ + LoRA ON          │                        │
│  │ scale=1      │            │ scale=s            │                        │
│  └──────┬───────┘            └──────┬─────────────┘                       │
│         │                           │                                      │
│         ▼                           ▼                                      │
│    teacher_out                 student_out                                 │
│    [B,N,128]                   [B,N,128]  (after unmerge)                 │
│         │                           │                                      │
│         └──────── L2 Loss ─────────┘                                      │
│                                                                            │
│  L = Σ_s ‖ε_θ+LoRA(z_t^{scale=s}, t) - ε_θ(z_t^{scale=1}, t)‖₂²     │
│                                                                            │
│  CRITICAL: Gradient accumulation across scales (NOT per-scale stepping)    │
│  ─ LoRA params are SHARED across scales 2 and 4                           │
│  ─ Per-scale stepping causes conflicting gradient updates (loss stuck ~1.0)│
│  ─ Accumulated gradients give coherent direction (loss drops to 0.67)      │
│                                                                            │
│  Trainable:   DDiT merge layers (21M) + LoRA on attn+FF (308M)           │
│               = 329M total trainable params                                │
│  Frozen:      Base model weights (int8-quanto quantized, ~12GB VRAM)       │
│  LoRA:        rank=32, alpha=32, targets: to_q/k/v/out, net.0.proj, net.2 │
│  Data:        Pre-encoded latents + Gemma text embeddings                  │
│  VRAM:        23.3GB peak (fits RTX 5090 33.7GB with room to spare)       │
│  Speed:       ~2s/step, 500 steps in ~16 minutes                          │
│                                                                            │
│  Results:     Loss 1.19 → 0.67 (21% better than adapter-only plateau)     │
│                                                                            │
│  Optimizer: AdamW, lr=1e-4, cosine schedule, weight_decay=1e-4            │
│  Loss: MSE in latent space between teacher and student predictions        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15E. Integration with SCD + EditCtrl

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Full Inference Pipeline with DDiT                        │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │  1. ENCODE (once per frame, always scale=1)              │              │
│  │     source_video → VAE → patchify → SCD encoder (32L)   │              │
│  │     → encoder_features (full resolution)                  │              │
│  └─────────────────────────────────────┬───────────────────┘              │
│                                         │                                  │
│  ┌─────────────────────────────────────▼───────────────────┐              │
│  │  2. DECODE (per denoising step, DDiT scheduling)         │              │
│  │                                                           │              │
│  │  ddit_wrapper.reset()                                     │              │
│  │  for step_idx, σ in enumerate(sigmas):                   │              │
│  │      # DDiT decides patch scale                           │              │
│  │      scale = ddit.get_scale(z_t, step_idx, F, H, W)     │              │
│  │                                                           │              │
│  │      if scale > 1:                                        │              │
│  │          # Coarse path: fewer tokens, faster              │              │
│  │          z_merged = ddit.merge(z_t, scale)                │              │
│  │          enc_merged = ddit.merge(enc_features, scale)     │              │
│  │          pos_merged = ddit.adjust_positions(pos, scale)   │              │
│  │          mask_merged = ddit.adjust_mask(mask, scale)      │              │
│  │                                                           │              │
│  │          output = scd.forward_decoder(                    │              │
│  │              z_merged, enc_merged, ... # fewer tokens     │              │
│  │          )                                                │              │
│  │          z_pred = ddit.unmerge(output, scale)             │              │
│  │      else:                                                │              │
│  │          # Fine path: native resolution                   │              │
│  │          z_pred = scd.forward_decoder(z_t, enc, ...)      │              │
│  │                                                           │              │
│  │      z_t = denoise_step(z_t, z_pred, σ)                 │              │
│  │      ddit.scheduler.record(z_t)                           │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                                                            │
│  Speed estimates (20 denoising steps, 768×576 video):                     │
│  ┌───────────────────────────────┬──────────────┬─────────┐              │
│  │ Configuration                 │ Tokens/step  │ Time    │              │
│  ├───────────────────────────────┼──────────────┼─────────┤              │
│  │ Vanilla LTX-2 (48L, 20 steps)│ 3456         │ ~160s   │              │
│  │ + SCD (32+16L, encoder 1x)   │ 3456         │ ~55s    │              │
│  │ + DDiT (mixed schedule)      │ 864-3456     │ ~30s    │              │
│  │ + DDiT aggressive (4x early) │ 216-3456     │ ~20s    │              │
│  └───────────────────────────────┴──────────────┴─────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15F. File Locations

```
DDiT Implementation:
├── ltx-core/
│   └── src/ltx_core/model/transformer/
│       └── ddit.py                    # DDiTAdapter, DDiTMergeLayer,
│                                      # DDiTPatchScheduler, DDiTConfig
├── sparse-causal-diffusion/
│   ├── scd/
│   │   └── ddit_inference.py          # DDiTInferenceWrapper for pipeline
│   │                                  # + load_lora(), from_checkpoint()
│   ├── scripts/
│   │   └── train_ddit_adapter.py      # Two-phase distillation training
│   │                                  # + PEFT LoRA integration
│   └── outputs/ddit_adapter/
│       ├── ddit_adapter_phase1.safetensors  # Phase 1 weights (21M params)
│       ├── ddit_adapter_final.safetensors   # Phase 2 adapter weights (21M)
│       ├── ddit_lora_final.safetensors      # Phase 2 LoRA weights (308M)
│       └── ddit_config.json                 # Config (scales, rank, targets)
```
