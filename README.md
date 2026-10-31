# Spark-X2.6-Recursion (`SparkX-Loop v0.2`)

**Full-Finetuning Looped Recursive Transformer (`60 Effective Layers` from `36 Physical Layers`) built on `XHToken/Spark-X2.6-4B` with Sublayer Dual-mHC (Manifold-Constrained Hyper-Connections), Pass-Conditioned Deliberation Adapters, Multi-Scale Causal Engram, and 1M-Context-Safe Fused SDPA.**

- **GitHub Repository**: [`BaiTian6641/Spark-X2.6-Recursion`](https://github.com/BaiTian6641/Spark-X2.6-Recursion)
- **Target Hugging Face Hub**: [`TassP0lar/Spark-X2.6-4B-Loop-mHC`](https://huggingface.co/TassP0lar)
- **Target Compute Environment**: Google Colab **G4 Instance** (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, **96 GB VRAM + 130 GB System RAM**, 23-hour run / 1000 compute units) with **Unsloth Studio** (`unsloth`, `unsloth_zoo`, `bitsandbytes` `PagedAdamW8bit`, `cut-cross-entropy`).

---

## 1. Architectural Highlights

### 1.1 Progressive Looped Depth Expansion (`36 -> 60 Effective Layers`)
`XHToken/Spark-X2.6-4B` consists of 36 physical layers (`hidden_size=2560`, `intermediate_size=6912`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=80`, `vocab_size=167168`) structured into nine 4-layer macro-blocks (`3 × sliding_attention (w=512, RoPE)` + `1 × full_attention (NoPE)`).

We split the 36 physical layers into:
- **Prelude (`[0, 4)`)**: 1 macro-block (`3 sliding + 1 full`) executed once.
- **Recurrent Core (`[4, 28)`)**: 6 macro-blocks (24 layers) unrolled progressively (`warmup -> R8 -> R12 -> R24`, `loop_repeats=2`) so the second pass re-traverses the middle 24 layers while preserving exact `3:1` sliding-to-full macro-block alignment.
- **Coda (`[28, 36)`)**: 2 macro-blocks (8 layers) executed once before the final norm and `lm_head`.

### 1.2 Aggressive Sublayer Dual-mHC + Pass Conditioning (`SublayerManifoldHyperConnection`)
Instead of a single static per-layer residual mixer, every recurrent decoder layer (`[4, 28)`) wraps **both** sublayers (`input_layernorm -> self_attn` and `post_attention_layernorm -> mlp`) with token-dependent **Sublayer Dual-mHC** (`mhc_attn` + `mhc_ffn`, 96 Sinkhorn routing steps across 2 passes):
- **Token-Dependent FP32 RMSNorm + Sinkhorn Routing**: Computes dynamic $2\times 2$ Birkhoff doubly-stochastic mixing matrices $P_t = \text{Sinkhorn}(\theta_{\text{mix}} + b_{\text{pass}} + \alpha_{\text{dyn}} W_{\text{dyn}}\,\text{RMSNorm}_{\text{fp32}}([h^{(0)}_t, h^{(1)}_t]))$ in `float32`.
- **Pass-Indexed Conditioning (`pass_routing_bias` + `pass_down`/`pass_up`)**: Each pass index $p \in \{0, 1, 2, 3\}$ has dedicated Sinkhorn routing logit biases and a **zero-initialized low-rank SwiGLU/SiLU deliberation adapter** (`rank=16`), allowing Pass 0 (initial extraction) and Pass 1+ (recursive latent deliberation) to specialize functionally while sharing physical backbone weights.
- **Exact Step-0 Zero-Gate Identity**: At initialization (`extra_gain=0`, `ffn_extra_gain=0`, `dynamic_proj.weight=0`, `ffn_dynamic_proj.weight=0`, `pass_up.weight=0`), Pass 2 is an exact mathematical identity on top of Pass 1 (`max_abs_diff == 0.0`), preventing initial shock to the pretrained `XHToken/Spark-X2.6-4B` representation.

### 1.3 Multi-Scale Causal N-Gram Engram (`HashedBigramEngram`)
Injected at layer 2 (Prelude) with zero-initialized projection (`proj.weight = 0`):
- Combines **2-gram** $(x_{t-1}, x_t)$, **3-gram** $(x_{t-2}, x_{t-1}, x_t)$, and **skip-2-gram** $(x_{t-2}, x_t)$ polynomial modular hash lookups into a low-rank embedding table (`buckets=65536, dim=192`).
- Applies **FP32 RMSNorm** across the multi-scale n-gram embedding and modulates injection via a **context-dependent sigmoid gate** $\sigma(W_{\text{ctx}} h_t)$ so the hidden state dynamically controls how much lexical memory enters each token.

### 1.4 Preserving `XHToken/Spark-X2.6-4B` 1M Token Context Window (`patch_spark_attention_for_long_context`)
Why does `XHToken/Spark-X2.6-4B` natively generalize to a **1M token context window**?
1. Its 27 sliding-window layers (`sliding_window=512`, `partial_rotary_factor=0.5`, `rope_theta=10000`) attend strictly to the local 512-token window, so relative RoPE distances never exceed 512 regardless of total sequence length.
2. Its 9 full-attention layers (`layers 3, 7, 11, 15, 19, 23, 27, 31, 35`) set `partial_rotary_factor=0.0` (**NoPE** — No Position Embedding), avoiding any global RoPE frequency extrapolation bottleneck up to 1M tokens.

**Critical Upstream Bottleneck Fixed**: Upstream `modeling_spark.py` hardcoded `eager_attention_forward` inside `Spark2_5Attention.forward`, calling `repeat_kv` (expanding 8 KV heads to 32) and materializing a dense `[B, 32, T, T]` attention score tensor in VRAM.
- [`patch_spark_attention_for_long_context`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py) patches every `Spark2_5Attention` module in-place to route through fused PyTorch `F.scaled_dot_product_attention(..., enable_gqa=True)` (FlashAttention-2 / cuDNN SDPA) whenever `output_attentions=False`, preserving exact `sliding_attention` + `full_attention` causal masks and `g_proj` sigmoid gating while reducing attention memory from $O(T^2)$ to $O(T)$.

### 1.5 Unsloth Studio Speedups + Token-Chunked 167,168-Vocab Cross-Entropy
- **Token-Chunked Masked CE (`fused_masked_cross_entropy`)**: Because `Spark-X2.6-4B` has a massive **167,168** vocabulary, materializing `[B, T, 167168]` FP32 logits at `T=4096` consumes ~2.74 GB per micro-batch. `fused_masked_cross_entropy` filters to supervised positions (`labels != -100`) and evaluates `lm_head` + cross-entropy in checkpointed 512-token chunks.
- **Full Finetuning on G4 (96 GB VRAM + 130 GB System RAM)**:
  - Unfreezes all 36 backbone layers + Dual-mHC + Pass Adapters + Multi-Scale Engram (`FULL_FINETUNE=True`) with discriminative learning rates (`backbone=2e-5`, `mHC/Engram/PassAdapters=1.5e-4`).
  - Uses `bitsandbytes.optim.PagedAdamW8bit` (via Unsloth Studio environment) to cut optimizer state VRAM by 75%, plus pinned CPU RAM (`offload_checkpoint_to_cpu`) for fast async checkpoint staging.

---

## 2. Repository Structure

| File | Description |
| :--- | :--- |
| [`SparkX_Loop_v02_OneShot_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v02_OneShot_Colab.ipynb) | One-shot Google Colab G4 notebook: clones/syncs repo, runs `setup_g4_env.sh`, streams 16 HF datasets + OpenCode teacher CoT traces into a 130 GB RAM packed pool, executes progressive 4-stage full finetuning (`warmup -> R8 -> R12 -> R24`), evaluates perplexity + GSM8K, and uploads to `https://huggingface.co/TassP0lar`. |
| [`sparkx_loop_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py) | Core implementation: `patch_spark_attention_for_long_context`, `SublayerManifoldHyperConnection` (`mhc_attn` + `mhc_ffn` + pass adapters), `HashedBigramEngram` (2-gram + 3-gram + skip-gram), `SparkXLoopWrapper`, `fused_masked_cross_entropy`, `build_optimizer`, `offload_checkpoint_to_cpu`, and `_VisitCache` fast KV-cached generation. |
| [`load_inference.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/load_inference.py) | Standalone inference loader (`load_exported_sparkx_loop`) that reloads `XHToken/Spark-X2.6-4B` (or a full-finetuned export directory), applies the 1M-context fused SDPA patch, attaches `loop_Overlay.pt`, and generates with `<think>...</think>` support. |
| [`test_loop_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/test_loop_core.py) | Self-contained smoke tests: step-0 exact identity (`max_abs_diff == 0.0`), Sublayer Dual-mHC + Pass Adapter + Multi-Scale Engram gradient flow, 1M-context SDPA equivalence, and cached vs. uncached greedy generation parity. |
| [`setup_g4_env.sh`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/setup_g4_env.sh) | Automated environment bootstrapper for Colab G4 (`unsloth`, `unsloth_zoo`, `bitsandbytes`, `cut-cross-entropy`, `transformers`, `datasets`, `accelerate`, `peft`, `openai`, `zstandard`). |
| [`requirements.txt`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/requirements.txt) | Python dependencies for full finetuning and inference. |
| [`START_HERE.md`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/START_HERE.md) | Quick-start guide for running the 23-hour Colab G4 full finetune. |

---

## 3. Quick Start on Google Colab G4 (96 GB VRAM + 130 GB System RAM)

### Step 1: Configure Colab Secrets
In Google Colab, open the **Secrets** sidebar (key icon) and add:
1. `HF_TOKEN` (or `HF_TOKEM`): Hugging Face write token for [`https://huggingface.co/TassP0lar`](https://huggingface.co/TassP0lar) (pushes to `TassP0lar/Spark-X2.6-4B-Loop-mHC`).
2. `OPENCODE_API_KEY` (or `OPENCODE_API`): OpenCode API key for supervised reasoning/CoT distillation from `deepseek-v4-flash`, `haiku5.5`, and `mimo-v2.6-pro`.

### Step 2: Run the One-Shot Notebook
Open [`SparkX_Loop_v02_OneShot_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v02_OneShot_Colab.ipynb) in Colab, select the **G4 GPU** runtime, and click **Runtime > Run all**.

Or initialize directly from a G4 terminal:
```bash
git clone git@github.com:BaiTian6641/Spark-X2.6-Recursion.git
cd Spark-X2.6-Recursion
bash setup_g4_env.sh
python3 test_loop_core.py
```

---

## 4. Curated Reasoning & CoT Training Mixture

The data pipeline streams, normalizes (`<think>...</think>` formatting), and packs sequences in the G4's 130 GB System RAM from:
- **Claude Fable**:
  - `kelexine/fable-5-sft-traces`
  - `HelioAI/Claude-Fable-5-5500x`
  - `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`
  - `saidutta69/fable-5-premium-v2`
  - `armand0e/Fable-5-Chat`
- **Claude Opus (4.6 / 4.7 / 4.8)**:
  - `Roman1111111/claude-opus-4.6-10000x`
  - `TeichAI/lordx64-claude-opus-4.7-max-cleaned`
  - `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`
  - `ronaldcmz/claude_opus_4.8_max_thinking_5k_v2`
  - `TeichAI/Claude-Opus-4.6-Reasoning-887x`
- **GPT Astra / GPT-6 / GPT-5**:
  - `FlaneAI/GPT-6Astra-scientific-research`
  - `Roman1111111/gpt-5.4-step-by-step-reasoning`
  - `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x`
  - `TeichAI/gpt-5.1-high-reasoning-1000x`
  - `Manusagents/Everything-you-will-really-need-for-train-good-llm` (`gpt55_thinking`)
- **Math Verification & OpenCode Teacher Augmentation**:
  - `openai/gsm8k`
  - Live OpenCode teacher distillation (`deepseek-v4-flash`, `haiku5.5`, `mimo-v2.6-pro`)
