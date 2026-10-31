# Spark-X2.6-Recursion (`SparkX-Loop v0.2`)

**Full-Finetuning Looped Recursive Transformer (`60 Effective Layers` from `36 Physical Layers`) built on `XHToken/Spark-X2.6-4B` with Sublayer Dual-mHC (Manifold-Constrained Hyper-Connections), Pass-Conditioned Deliberation Adapters, Memory-Tiered MoE-Engram Backbone Layers, and 1M-Context-Safe Fused SDPA.**

- **GitHub Repository**: [`BaiTian6641/Spark-X2.6-Recursion`](https://github.com/BaiTian6641/Spark-X2.6-Recursion)
- **Target Hugging Face Hub**: [`TassP0lar/Spark-X2.6-4B-Loop-mHC`](https://huggingface.co/TassP0lar)
- **Target Compute Environment**: Google Colab **G4 Instance** (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, **96 GB VRAM + 130 GB System RAM**, 23-hour run / 1000 compute units) with **Unsloth Studio** (`unsloth`, `unsloth_zoo`, `bitsandbytes` `PagedAdamW8bit`, `cut-cross-entropy`).

---

## 1. Architectural Highlights

### 1.1 Progressive Looped Depth Expansion (`36 -> 60 Effective Layers`)
`XHToken/Spark-X2.6-4B` consists of 36 physical layers (`hidden_size=2560`, `intermediate_size=6912`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=80`, `vocab_size=167168`) structured into nine 4-layer macro-blocks (`3 × sliding_attention (w=512, RoPE)` + `1 × full_attention (NoPE)`).

We split the 36 physical layers into:
- **Prelude (`[0, 4)`)**: 1 macro-block (`3 sliding + 1 full`) executed once, followed by the Prelude `TieredMoEEngram`.
- **Recurrent Core (`[4, 28)`)**: 6 macro-blocks (24 layers) unrolled progressively (`warmup -> R8 -> R12 -> R24`, `loop_repeats=2`) with a Backbone `TieredMoEEngram` anchored at the end of every 4-layer macro-block (`Layers 7, 11, 15, 19, 23, 27`).
- **Coda (`[28, 36)`)**: 2 macro-blocks (8 layers) executed once before the final norm and `lm_head`.

### 1.2 Aggressive Sublayer Dual-mHC + Pass Conditioning (`TwoStreamMHC`)
Every recurrent decoder layer (`[4, 28)`) wraps **both** sublayers (`input_layernorm -> self_attn` and `post_attention_layernorm -> mlp`) with token-dependent **Sublayer Dual-mHC** (`mhc_attn` + `mhc_ffn`, 96 Sinkhorn routing steps across 2 passes):
- **Token-Dependent FP32 RMSNorm + Sinkhorn Routing**: Computes dynamic $2\times 2$ Birkhoff doubly-stochastic mixing matrices in `float32`.
- **Pass-Indexed Conditioning (`pass_routing_bias` + `pass_down`/`pass_up`)**: Each pass index $p \in \{0, 1, 2, 3\}$ has dedicated Sinkhorn routing logit biases and a **zero-initialized low-rank SwiGLU/SiLU deliberation adapter** (`rank=16`), allowing Pass 0 (initial extraction) and Pass 1+ (recursive latent deliberation) to specialize functionally while sharing physical backbone weights.

### 1.3 Memory-Tiered MoE-Engram Backbone (`TieredMoEEngram` / `HashedBigramEngram`)
Inspired by **DeepSeek Engram ("Conditional Memory via Scalable Lookup")**, **DeepSeekMoE (Shared + Routed Experts)**, and **PEER / UltraMemV2 (Sparse Top-$K$ Value Expansion)**, we embed **7 `TieredMoEEngram` modules** across the backbone (`Prelude Layer 3` + `Recurrent Core Macro-Block Anchors 7, 11, 15, 19, 23, 27` — 13 total Engram executions across a 2-pass loop, 112 total MoE memory experts):
1. **Tier 1 — Fast Lexical N-Gram Lookup Tier ($O(1)$ Conditional Memory)**:
   - Multi-head prime modular hashing over causal **2-gram** $(x_{t-1}, x_t)$, **3-gram** $(x_{t-2}, x_{t-1}, x_t)$, and **skip-2-gram** $(x_{t-2}, x_t)$ token windows (`F.embedding`, zero matmul FLOPs). Precomputed once per forward call and reused across all backbone Engram layers and recurrent passes.
2. **Tier 2 — Hot Shared Knowledge Tier (Always-Active Compact Shared Expert)**:
   - Low-rank SwiGLU/SiLU bottleneck (`shared_down -> SiLU -> shared_up`) that captures high-frequency syntactic and cross-domain reasoning primitives so sparse experts do not duplicate common knowledge.
3. **Tier 3 — Cold/Domain Sparse Routed Memory Experts (MoE + PEER Top-$K$ Routing)**:
   - Each Engram layer contains $E=16$ specialized domain memory experts (`expert_down`, `expert_up`), activating only **Top-$K$ ($K=2$) experts per token** via a **pass-conditioned router** ($W_{\text{router}}\hat{h}_t + b_{\text{pass}}^{(p)} + b_{\text{balance}}$) with **DeepSeek-V3 auxiliary-loss-free load balancing** (`expert_load_ema`, `expert_bias`).
   - Query vector $q_t = W_q \hat{h}_t + e_{\text{lexical}, t}$ conditions expert value retrieval on both the semantic hidden state and local lexical n-gram identity.
4. **Context-Aware Gating & Gentle Engram Formation**:
   - Modulates the fused 3-tier memory via $2\sigma(W_{\text{ctx}} \hat{h}_t)$ and projects back to the residual stream via **zero-initialized** `project.weight = 0` (plus `tanh(ffn_extra_gain)` scaling on Pass 1+).
   - At **Step 0**, the model is 100% identical to stock `XHToken/Spark-X2.6-4B` (`max_abs_diff == 0.0`). During full finetuning, the MoE-Engram layers train at `LR_ADAPTER=1.5e-4` while the 36-layer backbone trains gently at `LR_FULL=1.5e-5`, acting as a high-capacity "knowledge sponge" that compacts new scientific and conversational knowledge without overwriting pretrained backbone reasoning circuits.

### 1.4 Preserving `XHToken/Spark-X2.6-4B` 1M Token Context Window (`patch_spark_attention_for_long_context`)
- [`patch_spark_attention_for_long_context`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py) patches every `Spark2_5Attention` module in-place to replace upstream's $O(T^2)$ `eager_attention_forward` with fused PyTorch `F.scaled_dot_product_attention(..., enable_gqa=True)` while preserving `3 × sliding_attention (w=512, RoPE) + 1 × full_attention (NoPE)` and `g_proj` sigmoid gating.
- The `TieredMoEEngram` modules add **zero KV-cache memory** and are strictly position-independent, keeping the 1M context window 100% intact.

---

## 2. Repository Structure

| File | Description |
| :--- | :--- |
| [`SparkX_Loop_v02_OneShot_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v02_OneShot_Colab.ipynb) | One-shot Google Colab G4 notebook: clones/syncs repo, runs `setup_g4_env.sh`, streams 26 HF datasets + OpenCode teacher CoT traces into a 130 GB RAM packed pool, executes progressive 4-stage full finetuning (`warmup -> R8 -> R12 -> R24`), evaluates perplexity + GSM8K, and uploads to `https://huggingface.co/TassP0lar`. |
| [`sparkx_loop_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py) | Core implementation: `patch_spark_attention_for_long_context`, `TwoStreamMHC` (`mhc_attn` + `mhc_ffn` + pass adapters), `TieredMoEEngram` / `HashedBigramEngram` (3-tier MoE-Engram), `SparkXLoop` (`engram` + `backbone_engrams`), `fused_masked_cross_entropy`, `build_optimizer`, and `_VisitCache` fast KV-cached generation. |
| [`load_inference.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/load_inference.py) | Standalone inference loader (`load_inference.py`) that reloads `XHToken/Spark-X2.6-4B`, applies the 1M-context fused SDPA patch, attaches `trainable.safetensors`, and generates with `<think>...</think>` support. |
| [`test_loop_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/test_loop_core.py) | Self-contained smoke tests: step-0 exact identity (`max_abs_diff == 0.0`), Sublayer Dual-mHC + Pass Adapter + 3-Tier Backbone MoE-Engram gradient flow, and `_VisitCache` verification. |
| [`setup_g4_env.sh`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/setup_g4_env.sh) | Automated environment bootstrapper for Colab G4 (`unsloth`, `unsloth_zoo`, `bitsandbytes`, `cut-cross-entropy`, `transformers==4.57.1`, `datasets`, `accelerate`, `peft`, `openai`, `zstandard`). |
| [`requirements.txt`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/requirements.txt) | Python dependencies for full finetuning and inference. |
| [`START_HERE.md`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/START_HERE.md) | Quick-start guide for running the 23-hour Colab G4 full finetune. |

---

## 3. Curated 26-Source Reasoning, Scientific & Multi-Turn Conversation Mixture

The data pipeline streams, normalizes (`<think>...</think>` formatting + multi-turn dialogue history), and packs sequences in the G4's 130 GB System RAM from:
- **Claude Fable**: `kelexine/fable-5-sft-traces`, `HelioAI/Claude-Fable-5-5500x`, `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`, `saidutta69/fable-5-premium-v2`, `armand0e/Fable-5-Chat`
- **Claude Opus (4.6 / 4.7 / 4.8)**: `Roman1111111/claude-opus-4.6-10000x`, `TeichAI/lordx64-claude-opus-4.7-max-cleaned`, `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`, `ronaldcmz/claude_opus_4.8_max_thinking_5k_v2`, `TeichAI/Claude-Opus-4.6-Reasoning-887x`
- **GPT Astra / GPT-6 / GPT-5**: `FlaneAI/GPT-6Astra-scientific-research`, `Roman1111111/gpt-5.4-step-by-step-reasoning`, `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x`, `TeichAI/gpt-5.1-high-reasoning-1000x`, `Manusagents/.../gpt55_thinking`
- **Scientific Reasoning & PhD/University STEM CoT**:
  - `nvidia/OpenScienceReasoning-2` (`<think>...</think>` + `\boxed{}` advanced STEM & scientific problems)
  - `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` (`PHD-Science` partition)
  - `MegaScience/MegaScience` & `MegaScience/TextbookReasoning` (peer-curated physics, chemistry, medicine, biology textbook derivations)
  - `OpenDataArena/Spark-234K` (graduate-level physics & quantitative scientific derivations)
  - `HuggingFaceTB/smoltalk2` (`Mid/OpenThoughts3_1.2M` STEM & code reasoning)
- **High-Quality Multi-Turn Conversation & Instruction Following**:
  - `nvidia/Nemotron-Instruction-Following-Chat-v1` (`chat_if` multi-turn dialogue with reasoning)
  - `NousResearch/Hermes-3-Dataset` (multi-turn system-prompted assistant dialogues)
  - `HuggingFaceH4/ultrachat_200k` (`train_sft` deep multi-turn conversations)
  - `DataCreatorAI/Multi-Turn-Conversational-SFT` (clean multi-turn everyday & technical dialogues)
- **Math Verification & OpenCode Teacher Augmentation**:
  - `openai/gsm8k` + live OpenCode teacher distillation (`deepseek-v4-flash`, `haiku5.5`, `mimo-v2.6-pro`)
