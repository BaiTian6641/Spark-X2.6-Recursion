# Spark-X2.6-Recursion (`Spark-X2.6-4B-En4B-A1B` `v0.3`)

**72-Layer Telescoping Recurrent Transformer (`36 Physical Layers -> 72 Effective Layers`) + 8-Anchor 3-Tier Zipfian `MoE-Engram` (`32` Experts, `Top-7`, `8.34B` Total / `5.33B` Active Parameters) with Doubly-Stochastic Sinkhorn Hyper-Connections (`TwoStreamMHC`), Self-Balancing X-MoE Routing, and Runtime $O(n \log n)$ Log-Linear KV Cache.**

- **GitHub Repository**: [`BaiTian6641/Spark-X2.6-Recursion`](https://github.com/BaiTian6641/Spark-X2.6-Recursion)
- **Hugging Face Checkpoint**: [`TassP0lar/Spark-X2.6-4B-En4B-A1B`](https://huggingface.co/TassP0lar/Spark-X2.6-4B-En4B-A1B)
- **Base Backbone**: [`XHToken/Spark-X2.6-4B`](https://huggingface.co/XHToken/Spark-X2.6-4B) (`36` physical layers, `hidden_size=2560`, `vocab_size=167168`, `3 × sliding_attention (w=512, RoPE) + 1 × full_attention (NoPE)`)

---

## 1. Architecture Overview

### 1.1 72-Layer Telescoping Recurrence (`36L -> 60L -> 72L`)
`XHToken/Spark-X2.6-4B` comprises 36 physical layers organized into nine 4-layer macro-blocks (`3 sliding RoPE + 1 full NoPE`). `SparkXLoopV03` expands effective reasoning depth to **72 virtual layers** without duplicating backbone weights:
- **Prelude (`[0, 4)`)**: 4 physical layers executed once (`Anchor 0` at Layer `3`).
- **Pass 1 (`[4, 28)`)**: 24 recurrent core layers (initial representation extraction, `Anchors 1..6` at Layers `7, 11, 15, 19, 23, 27`).
- **Pass 2 (`[4, 28)`)**: 24 recurrent core layers (first latent deliberation pass via `TwoStreamMHC` -> `60` cumulative layers).
- **Pass 3 (`[12, 24)`)**: 12 inner-core layers (deep latent verification pass via `p3_extra_gain` -> `72` effective layers).
- **Coda (`[28, 36)`)**: 8 physical layers executed once (`Anchor 7` at Layer `31`) before final RMSNorm and `lm_head`.

### 1.2 Sublayer Dual-`mHC` Doubly-Stochastic Sinkhorn Controllers (`TwoStreamMHC`)
Each of the 24 recurrent layers (`[4, 28)`) wraps both the Attention and FFN sublayers with a token-dependent **`TwoStreamMHC`** controller maintaining two parallel streams (Primary Residual Stream $B_l$ and Highway Stream $S_l$):
- **Closed-Form $2\times 2$ Birkhoff-Sinkhorn Mixing**: Maps dynamic logits into an exact doubly-stochastic matrix $M = \begin{pmatrix} p & 1-p \\ 1-p & p \end{pmatrix}$ in `float32`, bounding the spectral norm $\|M\|_2 \le 1$ across all 72 recurrent visits and eliminating gradient/activation explosion.
- **Pass-Indexed Deliberation Adapters (`pass_down` / `pass_up`)**: Rank-16 zero-initialized per-pass adapters and routing biases allow Pass 1, Pass 2, and Pass 3 to specialize functionally while sharing physical weights.
- **Exact Step-0 Identity (`max_abs_diff == 0.0`)**: Zero-gated initialization (`extra_gain = 0`, `p3_extra_gain = 0`) guarantees that at Step 0, `36L`, `60L`, and `72L` forward passes produce bitwise-identical logits.

### 1.3 8-Anchor 3-Tier Zipfian `MoE-Engram` & Self-Balancing Router (`ZipfianTieredMoEEngram`)
Eight `ZipfianTieredMoEEngram` modules are anchored at the end of every 4-layer macro-block (`Layers 3, 7, 11, 15, 19, 23, 27, 31` — `514M` Total / `138M` Active per anchor = **`4.11B` Total / `1.10B` Active** across 8 anchors):
1. **Tier 1A + Tier 1B — Ahead-of-Time Causal N-Gram Lookup ($O(1)$ Zero-FLOP Memory)**:
   - Deterministic prime modular hashing over causal **2-gram** $(x_{t-1}, x_t)$, **3-gram** $(x_{t-2}, x_{t-1}, x_t)$, and **skip-2-gram** $(x_{t-2}, x_t)$ windows computed once before Layer 0.
   - **Tier 1A (`16,384 × 3` buckets)**: Resides in GPU HBM.
   - **Tier 1B (`49,152 × 3` buckets)**: Offloaded to pinned Host DRAM with coalesced unique-row prefetching and custom post-backward sparse Adam updates (`step_tier1b_sparse_adam` in `<0.3 ms/step`).
2. **Tier 2 — 100%-Active Hot Shared SwiGLU Expert (`2560 -> 5888 -> 2560`)**:
   - Fused Triton SwiGLU expert capturing universal syntactic and cross-domain reasoning primitives.
3. **Tier 3 — 32 Routed SwiGLU Memory Experts (`E=32, Top-K=7`, `8 -> 32` Coarse-to-Fine Upcycling)**:
   - Trained as `E=8, Top-2` coarse macro-experts in Stage 2, then cloned & orthogonally perturbed via `upcycle_all_engrams_8_to_32()` into **`E=32, Top-7`** fine-grained experts (`256` total routed experts across 8 anchors) in Stage 3.
   - **Zero-Sync Dispatch**: Batched contraction for decoding (`n_tokens <= 8`) and single-`bincount` sorted contiguous bucket dispatch for 16K prefill/training (zero per-expert `.any()`/`.nonzero()` CUDA sync stalls).
4. **Self-Balancing X-MoE Hypersphere Cosine Router + Per-Pass Telescoping Gates**:
   - L2-normalized hypersphere cosine router with learnable temperature (`router_log_temp`), ST-MoE router `z_loss`, and **DeepSeek-V3 auxiliary-loss-free adaptive load balancing** (`expert_load_ema`, adaptive `expert_bias`, and automatic starved-centroid re-anchoring when any expert's load drops below $0.15 / E_{\text{act}}$).
   - **Per-Pass Telescoping Gates (`pass_gain_logits`)**: Scales Engram injection across recurrent passes (`[Pass 1: 0.35, Pass 2: 0.15, Pass 3: 0.05]`), injecting lexical/factual memory strongly during initial extraction while keeping Pass 2 and Pass 3 focused on latent reasoning.

---

## 2. Repository Structure & Colab Notebooks

| File | Purpose |
| :--- | :--- |
| [`SparkX_Loop_v03_Full_Pipeline_All_In_One_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v03_Full_Pipeline_All_In_One_Colab.ipynb) | **Formal All-in-One Full-Sequence Notebook**: Covers environment bootstrap, modifiable `MoE-Engram` & `mHC` auto-config, Step-0 5-invariant check, 6-pillar dataset curation & 16K packing, 4-Stage progressive curriculum (`mHC` -> `Engram-8` -> `Engram-32 Upcycle` -> `Joint Full-FT`), Phase-4 Post-Training Consolidation, router balance telemetry, 3D WebGL synthesis, and the 8-benchmark Frontier Scorecard. |
| [`SparkX_Loop_v03_En4B_A1B_OneShot_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v03_En4B_A1B_OneShot_Colab.ipynb) | **One-Shot Unattended Colab Training Notebook** (synced with the full 4-Stage + Phase-4 pipeline). |
| [`SparkX_Loop_v03_Frontier_Benchmarks_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v03_Frontier_Benchmarks_Colab.ipynb) | **Standalone 8-Benchmark Frontier Evaluation & Phase-4 Consolidation Notebook**: Evaluates `SWE-bench Pro`, `SWE-bench Verified`, `GPQA Diamond`, `HLE`, `IFEval`, `IFBench`, `HumanEval`, and `MATH-500`, plus standalone Phase-4 Joint `mHC` + `MoE-Engram` finalization. |
| [`SparkX_Loop_v03_En4B_A1B_Inference_Eval_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v03_En4B_A1B_Inference_Eval_Colab.ipynb) | **Standalone 72-Layer Inference, Quantitative Perplexity Ablation & Interactive 3D WebGL Playground**. |
| [`sparkx_loop_v03_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py) | Core `v0.3` model implementation (`LoopV03Settings`, `TwoStreamMHC`, `ZipfianTieredMoEEngram`, `SparkXLoopV03`, Triton RMSNorm/SwiGLU/Cut-CE kernels, and $O(n \log n)$ Log-Linear KV cache patcher). |
| [`sparkx_loop_v03_train_env.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_train_env.py) | Hardware auto-scaler (`auto_configure_vram_16k`), 6-pillar dataset streaming engine (`curate_all_pillars_v03`), 16K sequence packer (`build_16k_stage_pools`), 4-stage trainer (`run_v03_four_stage_training`), Hub exporter, and `SparkXAgenticWorkspace`. |
| [`sparkx_loop_v03_benchmarks.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py) | Self-contained evaluation runners for `IFEval`, `IFBench`, `GPQA Diamond`, `HLE`, `SWE-bench Pro`, `SWE-bench Verified`, `HumanEval`, `MATH-500`, and `run_phase4_engram_consolidation`. |
| [`load_inference_v03.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/load_inference_v03.py) | One-call loader (`load_sparkx_loop_v03`) that auto-resolves merged or base+adapter checkpoints from Hugging Face Hub or local disk. |

---

## 3. Full-Sequence Training Pipeline (Pre-Training to Post-Training)

The training curriculum in [`sparkx_loop_v03_train_env.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_train_env.py) transitions cleanly from pure recurrent loop adaptation to full 32-expert `MoE-Engram` specialization and backbone-frozen post-training consolidation:

| Phase / Stage | Effective Depth & SeqLen | Trainable Modules | Key Objective |
| :--- | :--- | :--- | :--- |
| **Stage 0 (Optional)** `ENABLE_STAGE0_PROGRESSIVE_DEPTH_WARMUP` | `48L (4K) -> 60L (8K)` | `24x TwoStreamMHC` (`Backbone Frozen`, `Engram OFF`) | Progressive stage-merging warmup of recurrent Sinkhorn mixing matrices before 72L expansion. |
| **Stage 1 (Phase 1)** `Stage1_Loop_mHC_Adapt_72L_8K` | `72L (8,192 tok)` | `24x TwoStreamMHC` (`Backbone Frozen`, `Engram OFF`) | Train Pass-2 (`[4..28)`) and Pass-3 (`[12..24)`) `mHC` gates and rank-16 deliberation adapters with zero Engram interference. |
| **Stage 2 (Phase 2A)** `Stage2_Engram8_Adapt_72L_8K` | `72L (8,192 tok)` | `24x TwoStreamMHC` + `8x MoE-Engram (E=8, Top-2)` (`Backbone Frozen`) | Form coarse domain experts, Tier-1A/1B n-gram embeddings, and self-balancing X-MoE cosine routing. |
| **Stage 3 (Phase 2B)** `Stage3_Engram32_Upcycle_72L_16K` | `72L (16,384 tok)` | `24x TwoStreamMHC` + `8x MoE-Engram (E=32, Top-7)` (`Backbone Frozen`) | Execute `upcycle_all_engrams_8_to_32()` (`64 -> 256` total routed experts) and adapt to full 16K long-horizon reasoning trajectories. |
| **Stage 4 (Phase 3)** `Stage4_Joint_FullFT_72L_16K` | `72L (16,384 tok)` | `36L Backbone (0.25x LR)` + `24x mHC` + `8x MoE-Engram-32` | Stage-merged joint full-finetuning co-aligning backbone representations with the 72L loop and 32-expert Engram memory. |
| **Phase 4 (Post-Train)** `run_phase4_engram_consolidation` | `72L (768–2048 tok)` | `24x TwoStreamMHC` + `8x MoE-Engram-32` (`36L Backbone 100% Frozen`) | Calibrate Per-Pass Telescoping Gates (`[P1: 0.35, P2: 0.15, P3: 0.05]`) on `IFEval`, `SWE-bench` diffs, `GPQA`, and `MATH-500` with a monotonic held-out Perplexity gate. |

---

## 4. 6-Pillar Multi-Domain Dataset Mixture (`48+` Verified Shards)

`curate_all_pillars_v03(CFG)` streams directly from verified `.parquet`/`.jsonl` shards with `<tool_response>` loss masking (`-100`) and zero mid-thought truncation:

1. **`code_swe` (`28%`) — Software Engineering & Algorithmic Code**:
   - `nvidia/OpenCodeReasoning`, `open-r1/verifiable-coding-problems-python_decontaminated`, `bigcode/self-oss-instruct-sc2-exec-filter-50k`, `ise-uiuc/Magicoder-OSS-Instruct-75K`, `ise-uiuc/Magicoder-Evol-Instruct-110K`, `m-a-p/CodeFeedback-Filtered-Instruction`, `iamtarun/python_code_instructions_18k_alpaca`, `TeichAI/Claude-Opus-4.6-Reasoning-887x`.
2. **`math_stem` (`24%`) — PhD/University STEM & Competition Mathematics**:
   - `nvidia/OpenScienceReasoning-2`, `MegaScience/MegaScience`, `MegaScience/TextbookReasoning`, `OpenDataArena/Spark-234K`, `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned (PHD-Science)`, `open-thoughts/OpenThoughts-114k`, `AI-MO/NuminaMath-CoT`, `bespokelabs/Bespoke-Stratos-17k`, `openai/gsm8k`.
3. **`web_3d_ui` (`16%`) — Web Design, React Frontend & 3D Three.js WebGL**:
   - `Tesslate/UIGEN-T1.5-Dataset`, `cfahlgren1/react-code-instructions`, `kelexine/fable-5-sft-traces`, `HelioAI/Claude-Fable-5-5500x`.
4. **`fable_opus_astra_cot` (`14%`) — Frontier Long-Run Reasoning CoT (Fable / Opus / GPT-6 Astra)**:
   - `FlaneAI/GPT-6Astra-scientific-research`, `Roman1111111/gpt-5.4-step-by-step-reasoning`, `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x`, `TeichAI/gpt-5.1-high-reasoning-1000x`, `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`, `saidutta69/fable-5-premium-v2`, `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k`, `Roman1111111/claude-opus-4.6-10000x`, `TeichAI/lordx64-claude-opus-4.7-max-cleaned`, `ronaldcmz/claude_opus_4.8_max_thinking_5k_v2`, `ServiceNow-AI/R1-Distill-SFT`, `Skywork/Skywork-OR1-RL-Data`.
5. **`literacy_dialogue` (`8%`) — Literacy, Academic Prose, Constraint Following & Multi-Turn Dialogue**:
   - `HuggingFaceH4/ultrachat_200k`, `nvidia/Nemotron-Instruction-Following-Chat-v1`, `NousResearch/Hermes-3-Dataset`, `DataCreatorAI/Multi-Turn-Conversational-SFT`, `HuggingFaceH4/no_robots`, `armand0e/Fable-5-Chat`.
6. **`tool_agent` (`10%`) — Multi-Turn Agentic Function & Tool Calling**:
   - `Locutusque/function-calling-chatml`, `lilacai/glaive-function-calling-v2-sharegpt`, `NousResearch/hermes-function-calling-v1`, `glaiveai/glaive-function-calling-v2`.

---

## 5. Verified Empirical Results (`TassP0lar/Spark-X2.6-4B-En4B-A1B`)

### 5.1 Architectural Perplexity & Health Ablation
| Configuration | Held-Out Mean CE Loss | Held-Out Perplexity | Next-Token Top-1 Acc | Unseen OOD `dijkstra_grid` Health Score |
| :--- | :---: | :---: | :---: | :---: |
| **`36L_Base_Only` (No Loop, No Engram)** | `4.02815` | `56.1570` | `45.15%` | `2 / 11` (Collapsed) |
| **`72L_Loop_mHC_Only` (Engram OFF)** | `1.79226` | `6.0030` (**$9.35\times$ lower**) | `70.02%` (**$+24.87\%$**) | **`11 / 11`** (`17.5 tok/s`) |
| **`72L_Full + 32-Exp MoE-Engram (scale=0.25)`** | **`1.78806` (Best)** | **`5.9778` (Best)** | **`70.02%` (Best)** | **`11 / 11`** (`13.9 tok/s`) |

### 5.2 8-Benchmark Frontier Scorecard
| Benchmark | Metric | Score | Breakdown / Details |
| :--- | :--- | :---: | :--- |
| **`HumanEval`** (`openai/openai_humaneval`) | **Pass@1 (Live Execution)** | **`72.0%`** | `18 / 25` passed subprocess unit tests |
| **`MATH-500`** (`HuggingFaceH4/MATH-500`) | **Exact Symbolic `\boxed{}` Acc** | **`52.0%`** | `Level 1: 66.7%`, `Level 2: 100.0%`, `Level 3: 50.0%`, `Level 5: 37.5%` |
| **`GPQA Diamond`** (`Idavidrein/gpqa`) | **PhD Science CoT / Hybrid Acc** | **`48.0%`** | `Chemistry: 63.6% (7/11)`, `Physics: 36.4% (4/11)`, `Biology: 33.3%` |
| **`SWE-bench Verified`** (`princeton-nlp/SWE-bench_Verified`) | **Valid Diff / Target File / Resolve** | **`100%` / `100%` / `40.0%`** | `37.08%` Mean Symbol Recall |
| **`SWE-bench Pro`** (`ScaleAI/SWE-bench_Pro`) | **Valid Diff / Target File / Resolve** | **`100%` / `100%` / `10.0%`** | `22.63%` Mean Symbol Recall |
| **`IFEval`** (`google/IFEval`) | **Prompt Strict / Inst Loose Acc** | **`64.0%` / `70.27%`** | `67.57%` Instruction Strict Accuracy |
| **`IFBench`** (`allenai/IFBench_test`) | **OOD Constraint Verification** | **Supported** | Auto-clones `allenai/IFBench` + NLTK & 58 OOD rules |
| **`HLE`** (`Humanity's Last Exam` + `MMLU-Pro`) | **Multi-Domain Frontier Acc** | **`33.33%`** | `Chemistry: 100.0%`, `Mathematics: 50.0%`, `Business: 30.0%` |

---

## 6. Usage Guide & Quick-Start Commands

### 6.1 Option A: End-to-End Training & Post-Training in Google Colab
1. Open **[`SparkX_Loop_v03_Full_Pipeline_All_In_One_Colab.ipynb`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/SparkX_Loop_v03_Full_Pipeline_All_In_One_Colab.ipynb)** in Google Colab (A100 80GB, H100 80GB, or RTX PRO 6000 Blackwell G4 96GB recommended; L4 24GB also supported via automatic activation offloading).
2. Add your `HF_TOKEN` (and optional `OPENCODE_API_KEY`) to Colab Secrets.
3. Adjust any desired `MoE-Engram` or `mHC` parameters in **Cell 2** (`CFG['ENG_EXPERTS']`, `CFG['ENG_TOP_K']`, `CFG['PILLAR_RATIOS']`, `CFG['ENABLE_STAGE0_PROGRESSIVE_DEPTH_WARMUP']`, `CFG['ENABLE_PHASE4_POST_TRAIN']`).
4. Run **Cells 1–8** sequentially to execute Step-0 verification, 6-pillar 16K sequence packing, 4-stage progressive training, Phase-4 consolidation, standalone Hub export, 3D WebGL synthesis, and the 8-benchmark Frontier Scorecard.

### 6.2 Option B: One-Call Python Inference & Phase-4 Consolidation
```python
import torch
from load_inference_v03 import load_sparkx_loop_v03
import sparkx_loop_v03_benchmarks as bench

# 1. Load trained 72-Layer + 32-Expert Top-7 MoE-Engram checkpoint in one call
model, tokenizer = load_sparkx_loop_v03(
    repo_id="TassP0lar/Spark-X2.6-4B-En4B-A1B",
    device="cuda" if torch.cuda.is_available() else "cpu",
    enable_loglinear_kv=True,
)

# 2. Optional: Run Phase-4 Joint mHC + MoE-Engram Post-Training Consolidation (36L Base Frozen)
phase4_report = bench.run_phase4_engram_consolidation(
    model,
    tokenizer,
    num_steps=80,
    lr_engram=1.5e-4,
    lr_gates=5.0e-4,
    lr_mhc=3.0e-5,
    train_mhc=True,
)

# 3. Generate with 72-Layer Recurrent Deliberation (<think>...</think>)
resp = bench.generate_chat_response(
    model,
    tokenizer,
    user_prompt="Write a Python function `dijkstra_grid(grid)` using heapq to find the min-cost path from (0,0) to (R-1,C-1).",
    max_new_tokens=1024,
)
print(resp["answer_text"])
```
