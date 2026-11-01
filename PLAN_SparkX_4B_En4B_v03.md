# Master Engineering Plan: `Spark-X2.6-4B-En4B-A1B` (`SparkX-Loop v0.3`)

**Target Architecture:** `8.2B` Total Parameters (`4.2B` Looped Backbone + `4.0B` Total / **`1.0B` Active** 3-Tier Zipfian MoE-Engram) | **`5.2B` Active Parameters (`4.2B` Backbone + `1.0B` Active Engram)** | **72 Effective Layers** (3-Pass Telescoping Recurrence) | **Runtime-Only $O(n \log n)$ Log-Linear KV Cache** | **Host DRAM / NVMe SSD Engram Offloading** | **23-Hour Colab G4 Fast-Convergence Training Protocol**

---

## 1. Executive Summary & Design Goals

Building directly on the empirical findings from `SparkX-Loop v0.2` (where a `26.3M`-parameter 7-layer MoE-Engram + 60-layer Dual-mHC loop reduced held-out proof perplexity by `-5.3%` in live ablation), `Spark-X2.6-4B-En4B-A1B` (`v0.3`) scales along four orthogonal axes while fitting strictly inside a **single 23-hour Google Colab G4 (96 GB VRAM + 130 GB System RAM)** training window:

1. **High-Throughput Conditional Memory (`4.0B` Total / `1.0B` Active Engram $\rightarrow$ `8.2B` Total / `5.2B` Active Model)**:
   - Expands the embedded MoE-Engram system across **8 macro-block anchors (`Layers 3, 7, 11, 15, 19, 23, 27, 31`)** to **`500M` total / `125M` active parameters per anchor** (`4.00B` total / `1.00B` active Engram parameters).
   - **Why `En4B-A1B` (`25%` Active Engram Ratio) Is Optimal for 23-Hour Training**:
     - Each 4-layer Spark macro-block has `~380M` active backbone parameters paired with `125M` active Engram parameters—matching DeepSeek Engram's **U-Shaped Scaling Law** optimal ratio (`~75%` backbone compute to `~25%` conditional memory) without overwhelming the `d_model = 2560` residual stream.
     - Activating `1.0B` out of `4.0B` Engram parameters (`25%` active per token via a `45M` shared expert + `Top-7 of 32` routed experts per anchor) gives every expert **~4× denser gradient flow per step**, ensuring full convergence inside a single 23-hour G4 run.
   - Uses **Host DRAM / NVMe SSD `mmap` offloading with deterministic ahead-of-time token-ID prefetching** (*DeepSeek Engram, Cheng et al., 2026*) so the `0.81B` Tier-1 N-gram bank can reside in Host DRAM/SSD with zero GPU VRAM overhead.
2. **Deep Telescoping Recurrence (`36 Physical Layers -> 72 Effective Layers`)**:
   - Unrolls the 36 physical layers in a **3-Pass Macro-Block Pyramid** ($36 + 24 + 12 = 72$ effective layers) governed by **Sublayer Dual-mHC (`TwoStreamMHC`)** with exact closed-form $2\times 2$ Sinkhorn projection and Pass-0 / Pass-1 / Pass-2 deliberation adapters.
3. **Runtime-Only, Zero-Collapse $O(n^2) \to O(n \log n)$ KV Cache Optimization**:
   - Applied purely at **architecture runtime (`patch_spark_attention_for_loglinear_kv`)** without modifying or retraining existing checkpoint weights—works immediately on both `v0.2` (`R24`) and `v0.3` (`En4B-A1B`) checkpoints.
   - Exploits the fact that `Spark-X2.6-4B` has **27 native $O(n)$ Sliding-Window layers (`w=512`, RoPE)** and **9 Full-Attention layers that use NoPE (`No Position Embedding`)**, enabling **Log-Mass-Conserved Dyadic Fenwick Pooling + HiP Top-$M$ Tree Search** with zero rotary phase distortion and zero softmax partition-function collapse.
4. **23-Hour Fast-Convergence Protocol & Balanced Cognitive + Natural-Prose Curriculum**:
   - Solves the "4B Engram in 23 hours" training bottleneck via **Coalesced Sparse Row Updates**, **8-to-32 Coarse-to-Fine Expert Upcycling**, and **Backbone-Protected Asymmetric Optimization**.
   - Pairs **Fable / Claude Opus / GPT Astra-5-6 cognitive planning & self-evaluation traces** with **verifiable STEM scratchpads** and **100%-human natural prose / anti-slop datasets** (`no_robots`, `oasst2`, `LitBench`, `Creative_Writing-ShareGPT`, `LongPage`, `LIMA`) so the model plans and evaluates systematically inside `<think>` while writing natural, non-robotic prose in its final response.

---

## 2. Detailed `4.0B-A1B` Engram Architecture & Hardware Offloading (`En4B-A1B`)

### 2.1 Per-Anchor Parameter Allocation (`500M` Total / `125M` Active × 8 Anchors = `4.00B` Total / `1.00B` Active)

We embed a `TieredMoEEngram` module at the end of the first 8 Spark 4-layer macro-blocks (`Layers 3, 7, 11, 15, 19, 23, 27, 31`), leaving the final Coda macro-block (`Layers 32..35`) dedicated to output decoding:

```mermaid
flowchart LR
    Tok["Input Token IDs x"] -->|"Ahead of Layer 0 (Deterministic)"| Hash["Multi-Scale Causal Hash (2/3/skip-gram)"]
    Hash -->|"Hot Top-16k Buckets"| T1A["Tier 1A: GPU HBM Lexical Cache (0.20B)"]
    Hash -->|"Async PCIe / mmap Prefetch"| T1B["Tier 1B: Host DRAM / NVMe SSD Bank (0.61B)"]
    H["Layer l Hidden State h_t"] --> Norm["FP32 RMSNorm"]
    Norm --> T2["Tier 2: Always-Active Shared SwiGLU Expert (0.36B Total / 0.36B Active)"]
    Norm --> Router["Hypersphere Cosine + Sigmoid Router (E=32, Top-K=7)"]
    T1A & T1B --> Query["Lexical-Conditioned Query q_t"]
    Query & Router --> T3["Tier 3: 32 Orthogonal SwiGLU Memory Experts (2.83B Total / 0.58B Active)"]
    T1A & T1B & T2 & T3 --> Gate["Context-Aware Gate 2*sigmoid(W_ctx h_t) + Zero-Init Project (0.06B Active)"]
```

| Tier | Architectural Specification (per Anchor Layer) | Total / Anchor | Total (8 Anchors) | Active / Anchor | Active (8 Anchors) | Memory Placement |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1A (Hot Lexical Cache)** | `16,384` buckets × `512` dim × 3 scales | `25.17M` | **`0.201B`** | `6.80M` *(with gate/proj)* | **`0.054B`** | **GPU HBM** |
| **Tier 1B (Long-Tail N-Gram Bank)** | `49,152` buckets × `512` dim × 3 scales (`2/3/skip-gram`), **zero-init** | `75.50M` | **`0.604B`** | `0.005M` ($O(1)$ lookup) | **`0.000B`** | **Host DRAM (130 GB G4) / NVMe `mmap`** |
| **Tier 2 (Hot Shared Expert)** | SwiGLU `2560 -> 5888 -> 2560` (`100%` active) | `45.22M` | **`0.362B`** | `45.22M` (`100%`) | **`0.362B`** | **GPU HBM** |
| **Tier 3 (Routed Memory Experts)** | `E = 32` SwiGLU experts (`Top-K = 7`), bottleneck `rank = 1440` (`11.06M`/expert) | `354.11M` | **`2.833B`** | `72.98M` (`21.9%`) | **`0.584B`** | **GPU HBM** |
| **Total `En4B-A1B` System** | **8 Macro-Block Anchors (`Layers 3, 7, 11, 15, 19, 23, 27, 31`)** | **`500.00M`** | **`4.000B`** | **`125.00M`** | **`1.000B` (`A1B`)** | **`3.40B` GPU HBM + `0.60B` DRAM/SSD** |

### 2.2 How We Train a `4.0B-A1B` Engram Inside the **23-Hour Colab G4 Cap**

Training 4.0B additional parameters in 23 hours (`~4,000 optimizer steps`) fails if done naively due to (a) dense optimizer sweeps over untouched hash rows, (b) slow convergence of cold-started experts, and (c) redundant backbone weight-gradient computation. `En4B-A1B` solves all three:

1. **Coalesced Sparse Row Updates for Tier 1B (`0.60B` Parameters in `<0.3 ms/step`)**:
   - In any micro-batch of `2,048` tokens, at most `3 × 2,048 = 6,144` bucket rows out of `147,456` per layer (`<4.2%`) receive non-zero gradients.
   - Instead of registering Tier 1B in a dense `AdamW` group that sweeps all `0.60B` parameters every step, Tier 1B resides in **pinned Host RAM (`130 GB` on Colab G4)** and executes **coalesced sparse Adam updates (`torch.unique` active bucket indices only)** asynchronously on CPU/CUDA stream during the next step's forward prelude.
   - **Zero-Init Cold Safety**: Because Tier 1A and Tier 1B tables are initialized to **exact `0.0`** (while `query_proj` and `shared_down` use Kaiming initialization to break symmetry), any long-tail N-gram bucket that is never visited during the 23-hour run outputs **exact `0.0`** at inference time—never injecting untrained random noise.

2. **8 $\to$ 32 Coarse-to-Fine Expert Upcycling (`21.9%` Active Coverage per Expert)**:
   - Because `En4B-A1B` uses `E = 32` SwiGLU experts with `Top-K = 7` (`21.9%` of tokens routed to every expert, plus a `45.2M` `100%`-active shared expert per anchor), every expert receives **3.5× denser gradients** than a 128-expert layout!
   - **Stage 1 (`warmup` + `R12`, first 600 steps)**: Each Engram layer trains **8 coarse macro-experts (`Top-K = 4`, `50%` token coverage per expert)**.
   - **Stage 2 Upcycling (`R24` transition at step 601)**: Each trained macro-expert $m \in \{0..7\}$ is **cloned into 4 fine-grained sub-experts ($8 \times 4 = 32$ experts)** with small mutually orthogonal perturbations ($\pm \epsilon Q_j$, $\epsilon = 10^{-3}$), so all 256 routed experts across the 8 layers converge rapidly within the 23-hour G4 run.

3. **Two-Speed Backbone-Protected Optimization**:
   - During the first half of each stage, backbone `self_attn` and `mlp` weights can optionally freeze (`requires_grad=False`) or run with low-frequency updates, cutting activation-checkpoint backward FLOPs by **~38%** while the `4B-A1B` Engram + Dual-mHC controllers train at full speed (`LR_ADAPTER = 2.0e-4`).

---

## 3. Anti-Collapse MoE-Engram Routing (4 Literature-Backed Mechanisms)

To guarantee that our `256` Tier-3 memory experts (`32 experts × 8 anchors`, `2.83B` total / `0.58B` active) never suffer from **Routing Collapse** (expert starvation) or **Representation Collapse** (hidden-state shrinkage), [`TieredMoEEngram`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_core.py#L216-L366) incorporates four proven defenses (cited in [`REFERENCES.md`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/REFERENCES.md#L305-L355)):

1. **$\ell_2$-Normalized Hypersphere Cosine Router (`X-MoE`, *Chi et al., NeurIPS 2022*, `arXiv:2204.09179`)**:
   - Projects normalized hidden state $\hat{h}_t \in \mathbb{R}^{2560}$ into a low-dimensional routing subspace $u_t = W_{\text{route}} \hat{h}_t \in \mathbb{R}^{64}$ and computes cosine similarity on the unit hypersphere against expert centroids $e_i \in \mathbb{R}^{64}$ with learnable log-temperature $\tau$:
     $$z_{t, i}^{(p)} = \frac{1}{\tau} \left\langle \frac{u_t}{\|u_t\|_2 + \epsilon}, \frac{e_i}{\|e_i\|_2 + \epsilon} \right\rangle + b_{\text{pass}}^{(p, i)}$$
2. **Sigmoid Affinity Gating + Auxiliary-Loss-Free Bias Balancing (*Wang et al., DeepSeek-V3, 2024*, `arXiv:2408.15664`)**:
   - Computes independent expert affinities $s_{t, i} = \sigma(z_{t, i}^{(p)})$, selects Top-$K$ ($K=7$) via $\operatorname{topk}(s_{t, i} + b_{\text{balance}, i})$, and normalizes gates across the selected $K$ experts:
     $$g_{t, i} = \frac{s_{t, i}}{\sum_{j \in \mathcal{T}_t} s_{t, j} + 10^{-6}}, \qquad b_{\text{balance}, i} \leftarrow \operatorname{clamp}\!\left(b_{\text{balance}, i} + \gamma \operatorname{sign}\!\left(\frac{K}{E} - \text{EMA}_i\right), -0.5, 0.5\right)$$
3. **Router Z-Loss (*Zoph et al., ST-MoE, 2022*, `arXiv:2202.08906`)**:
   - Adds a lightweight numerical stabilizer $\mathcal{L}_z = 10^{-4} \cdot \frac{1}{BT}\sum_{t} \left(\log \sum_{i=1}^{E} \exp(z_{t, i}^{(p)})\right)^2$ to prevent router logit explosion across 72 virtual layers.
4. **Block-Orthogonal Expert Initialization (`nn.init.orthogonal_`)**:
   - Ensures all $E = 32$ experts per anchor span orthogonal subspaces at initialization.

---

## 4. Expanding Recursive Depth to **72 Effective Layers** (Telescoping 3-Pass Pyramid)

`XHToken/Spark-X2.6-4B` has 36 physical layers organized into nine 4-layer macro-blocks (`3 × sliding_attention w=512 RoPE + 1 × full_attention NoPE`). To reach **exact 72 effective layers** while keeping every loop transition aligned to 4-layer macro-block boundaries (`% 4 == 0`), we extend `SparkXLoop` from a flat 2-pass loop to a **3-Pass Telescoping Pyramid**:

```mermaid
flowchart TD
    P["Prelude: Layers [0..4) (4 layers, Pass 1)"] --> C1["Core Pass 1: Layers [4..28) (24 layers, pass_idx=0)"]
    C1 --> C2["Core Pass 2: Layers [4..28) (24 layers, pass_idx=1)"]
    C2 --> C3["Deep Inner Core Pass 3: Layers [12..24) (12 layers, pass_idx=2)"]
    C3 --> Coda["Coda: Layers [28..36) (8 layers, Pass 1)"]
```

- **Total Virtual Depth**:
  $$\underbrace{4}_{\text{Prelude }[0,4)} + \underbrace{24}_{\text{Pass 1 }[4,28)} + \underbrace{24}_{\text{Pass 2 }[4,28)} + \underbrace{12}_{\text{Pass 3 }[12,24)} + \underbrace{8}_{\text{Coda }[28,36)} = \mathbf{72\text{ Effective Layers}}$$
- **Per-Pass Specialization**:
  - **Pass 1 (`pass_idx = 0`, 36 layers)**: Initial representation extraction & Tier-1/2/3 factual lookup.
  - **Pass 2 (`pass_idx = 1`, 24 middle layers `[4, 28)`)**: Broad multi-step latent reasoning & cross-clause synthesis.
  - **Pass 3 (`pass_idx = 2`, 12 deep-core layers `[12, 24)`)**: High-precision algebraic/logical self-verification using dedicated `pass_routing_bias[2]`, `pass_down[2]/pass_up[2]` adapters, and dedicated Pass-3 gates (`p3_attn_extra_gain`, `p3_ffn_extra_gain`, initialized to `0.0` so the `60L -> 72L` stage transition is 100% smooth!).

| Curriculum Stage | Active Loop Spans | Effective Layers | Optimizer Steps | Cumulative Steps | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stage 1: `warmup`** | `[0..36)` (1 pass) | **36 layers** | `200` | `200` | Warm-start 8 MoE-Engrams (8 macro-experts, `Top-K=4`) |
| **Stage 2: `R12`** | `[4..16)` × 2 passes | **48 layers** | `400` | `600` | Open lower-middle Dual-mHC gates |
| **Stage 3: `R24`** | `[4..28)` × 2 passes | **60 layers** | `1,400` | `2,000` | Upcycle `8 -> 32` experts (`Top-K=7`, `A1B`); full 2-pass loop |
| **Stage 4: `R72` (`R24+P3_12`)** | `[4..28)` × 2 + `[12..24)` × 3 | **72 layers** | `2,000` | `4,000` | Activate Pass-3 deep-core verification (`72L`) |

---

## 5. Runtime-Only, Zero-Collapse $O(n^2) \to O(n \log n)$ KV Cache Optimization

### 5.1 Design Requirement: 100% Runtime-Only (Zero Checkpoint Modification, Zero Collapse)
This optimization is implemented purely in the attention forward/cache runtime (`patch_spark_attention_for_loglinear_kv`) so it can be toggled on **any existing checkpoint (`v0.2 R24` or `v0.3 R72`) at inference or evaluation time without touching a single weight**.

### 5.2 Why `Spark-X2.6-4B` Can Be Compressed to $O(n \log n)$ Without Phase Distortion
- **27 of the 36 physical layers (`75%`) are already $O(n)$** (`sliding_attention` with window `w = 512` and RoPE): their KV cache is strictly bounded at `512` tokens per virtual visit.
- **Only the 9 Full-Attention layers (`Layers 3, 7, 11, 15, 19, 23, 27, 31, 35`) are $O(n^2)$—and critically, all 9 Full-Attention layers use NoPE (`No Position Embedding`, `partial_rotary_factor = 0.0`)!**
- In standard RoPE models, pooling adjacent Key vectors $k_{t_1}, k_{t_2}$ corrupts their rotary angles $\theta_{t_1}, \theta_{t_2}$. Because `Spark-X2.6-4B`'s Full-Attention layers are **NoPE**, their Key and Value vectors live in a **position-invariant vector space** where multiscale tree aggregation is mathematically exact.

### 5.3 How We Prevent Runtime Softmax Collapse: Log-Mass-Conserved Dyadic Fenwick Tree + HiP Top-$M$ Exact Recall

Naive KV pooling or eviction causes model collapse at long context for two reasons: (1) losing initial **Attention Sink** tokens, and (2) changing the **Softmax Partition Function (denominator mass)** when $m$ keys are merged into 1 summary key. Our runtime `LogLinearNoPECache` eliminates both failure modes:

```mermaid
flowchart LR
    KV["Full NoPE History [0 .. t]"] --> Sinks["1. Pinned Attention Sinks [0 .. 31] (Exact 1:1, m=1)"]
    KV --> Mid["2. Middle History [32 .. t - 1024]"]
    KV --> Local["3. Exact Local Window [t - 1024 .. t] (Exact 1:1, m=1)"]
    Mid --> Tree["Dyadic Fenwick Tree (Levels l = 0 .. log2(T/32))"]
    Tree --> HiP["HiP Top-M (M=8) Highest-Affinity Blocks (Exact 1:1 Tokens)"]
    Tree --> Summary["Remaining O(log T) Dyadic Summary Nodes (k_bar, v_bar) + Log-Mass Bias +ln(m)"]
```

For any sequence length $T$:
1. **Exact Short/Medium Context Bypass ($T \le W_{\text{exact}} = 2048$)**:
   - When $T \le 2048$, the runtime executes standard fused SDPA (`max_abs_diff == 0.0` vs. unpatched checkpoint).
2. **Three-Region Partition-Conserving Attention for $T > 2048$**:
   - **Region A — Pinned Attention Sinks ($S = 32$ initial tokens `[0..31]`)**: Kept at exact 1:1 resolution (`multiplicity` $m = 1$).
   - **Region B — Exact Local Window ($W = 1024$ recent tokens `[t - 1024 .. t]`)**: Kept at exact 1:1 resolution ($m = 1$).
   - **Region C — Log-Linear Dyadic Fenwick Tree + HiP Top-$M$ Exact Blocks (`[32 .. t - 1024]`)**:
     - Middle history is partitioned into base blocks of $B_0 = 32$ tokens and organized into a binary Fenwick tree of $\lceil \log_2(T / 32) \rceil$ dyadic levels (block sizes $32, 64, 128, 256, \dots$).
     - **Step 1 (HiP Top-$M$ Exact Selection, *Lee et al., 2024*, `arXiv:2406.09827`)**: The query $q_t$ scores the $O(\log T)$ tree centroids and keeps the **Top-$M$ ($M = 8$) most relevant 32-token blocks (`256` tokens) at exact 1:1 resolution** ($m = 1$).
     - **Step 2 (Log-Mass-Conserved Dyadic Summaries, *Guo et al., 2025*, `arXiv:2506.04761`)**: All remaining non-selected history intervals are represented by $O(\log T)$ dyadic summary nodes $(\bar{k}_\beta, \bar{v}_\beta)$ representing $m_\beta = 32 \cdot 2^\ell$ tokens, with **norm-preserving key direction** and **exact log-multiplicity softmax bias**:
       $$\bar{k}_\beta = \frac{\text{mean}_{i \in \beta}(\|k_i\|_2)}{\|\mu_\beta\|_2 + \epsilon} \mu_\beta, \qquad \text{logit}(q_t, \bar{k}_\beta) = \frac{q_t^\top \bar{k}_\beta}{\sqrt{d_k}} + \mathbf{\ln(m_\beta)}$$
     - **Why It Never Collapses**: Because $\exp(q_t^\top \bar{k}_\beta / \sqrt{d_k} + \ln m_\beta) = m_\beta \exp(q_t^\top \bar{k}_\beta / \sqrt{d_k})$, the total softmax denominator weight of the summarized middle history matches the full $O(n^2)$ attention denominator in expectation, while the Top-$M$ spiky attention matches are preserved at exact 1:1 token resolution!
   - **Complexity Scaling**:
     - Each query attends to at most $32\text{ (sinks)} + 1024\text{ (local)} + 256\text{ (HiP exact)} + 2\lceil \log_2(T/32) \rceil\text{ (dyadic summaries)} \approx \mathbf{1,312 + 2\log_2 T\text{ keys}}$ regardless of whether $T = 16\text{K}, 128\text{K}, \text{or } 1\text{M}$!
     - Total prefill complexity drops from $O(n^2)$ to **$O(n \log n)$**; per-step decoding complexity drops from $O(n)$ to **$O(\log n)$**.

---

## 6. 5-Pillar Dataset Strategy: Cognitive Planning, Verifiable STEM, Agentic Tool Use, World Knowledge & Natural Human Language

### 6.1 The Tri-Objective Insight (Thinking + Tool Action + Natural Human Prose)
Our analysis of `v0.2` and modern agentic architectures reveals three complementary requirements:
1. **Structured Meta-Cognition Inside `<think>` (`Fable` / `Claude Opus` / `GPT Astra-5-6`)**: Teaches explicit problem decomposition, hypothesis testing, edge-case auditing, and epistemic calibration.
2. **Verifiable Calculation & Interleaved Tool Calling (`<think>` $\leftrightarrow$ `<tool_call>` $\leftrightarrow$ `<tool_response>`)**:
   - Instead of forcing a `4B` backbone to hallucinate dynamic facts or compute brittle multi-step arithmetic purely in weights, the model learns to pause mid-reasoning, emit a structured `<tool_call>`, ingest `<tool_response>` into its `_VisitCache` in $O(1)$ incremental time, and resume `<think>` deliberation.
   - Our `4.0B-A1B` MoE-Engram is uniquely synergistic with tool calling: **Tier 1** absorbs rigid JSON/MCP schema syntax (`{"name": ..., "arguments": ...}`) in $O(1)$ lookup, while **Tier 3** experts specialize in API argument deduction and multi-turn state tracking.
3. **100%-Human Natural Language & Encyclopedic World Knowledge**:
   - Counteracts repetitive "Claudish / AI-slop" surface phrasing while populating the `4.0B-A1B` MoE-Engram with dense humanities, scientific, and encyclopedic world facts.

### 6.2 The 5-Pillar `Spark-X2.6-4B-En4B-A1B` Dataset Mixture (`46 Curated Sources`)

| Pillar & Weight | Target Skill | Curated High-Quality Datasets | Verification & Anti-Slop Filter |
| :--- | :--- | :--- | :--- |
| **Pillar 1 (20%)**<br>**Cognitive Planning & Meta-Evaluation** | Structured problem decomposition, hypothesis testing, edge-case auditing, and epistemic calibration inside `<think>`. | • `kelexine/fable-5-sft-traces`<br>• `HelioAI/Claude-Fable-5-5500x`<br>• `MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`<br>• `saidutta69/fable-5-premium-v2`<br>• `Roman1111111/claude-opus-4.6-10000x`<br>• `TeichAI/lordx64-claude-opus-4.7-max-cleaned`<br>• `ronaldcmz/claude_opus_4.8_max_thinking_5k_v2`<br>• `FlaneAI/GPT-6Astra-scientific-research`<br>• `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x`<br>• `TeichAI/gpt-5.1-high-reasoning-1000x` | **Plan + Execution Filter**: Rejects "outline-only" `<think>` traces that guess a boxed number without intermediate derivations; strips formulaic AI openings (*"Certainly!"*, *"Here's a breakdown"*). |
| **Pillar 2 (30%)**<br>**Verifiable STEM, Math & Code Scratchpads** | Step-by-step symbolic algebra, PhD science derivations, and verifier-checked math/code execution inside `<think>`. | • [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) (`Math-Verify` checked)<br>• [`open-thoughts/OpenThoughts-114k`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) (verified math/code/science)<br>• [`nvidia/Llama-Nemotron-Post-Training-Dataset`](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) (`SFT` math/code/STEM)<br>• `nvidia/OpenScienceReasoning-2`<br>• `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` (`PHD-Science`)<br>• `MegaScience/MegaScience` & `MegaScience/TextbookReasoning`<br>• `OpenDataArena/Spark-234K`<br>• `HuggingFaceTB/smoltalk2` (`OpenThoughts3_1.2M`)<br>• `openai/gsm8k` + **OpenCode Go** (`https://opencode.ai/zen/go/v1`) verified CoT | Verified final answers (`\boxed{}` / `####` / unit tests); preserves native RL arithmetic scratchpad fidelity. |
| **Pillar 3 (15%)**<br>**Agentic Tool Calling, MCP & Interleaved Thinking** | Mid-reasoning tool invocation (`<think>` $\to$ `<tool_call>` $\to$ `<tool_response>` $\to$ `<think>`), parallel/nested API calls, MCP servers, and SWE debugging. | • [`Agent-Ark/Toucan-1.5M`](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) (1.5M real MCP tool trajectories)<br>• [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) (26.5k APIs, parallel & dependent calls)<br>• [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (3-tier execution verified)<br>• [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) (multi-turn agent-human interplay)<br>• [`hxssgaa/xlam-interleave-thinking-40k`](https://huggingface.co/datasets/hxssgaa/xlam-interleave-thinking-40k) (interleaved `<think>` + tool call)<br>• [`pyromind/agentic-tool-call-dataset-12k`](https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k)<br>• [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)<br>• [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) (`SWE-Gym-Lite` executable code agent) | **Strict Schema & Masking Filter**: Assistant loss is computed ONLY on `<think>...</think>` and `<tool_call>...</tool_call>`; environment `<tool_response>` blocks are masked (`labels = -100`) so the model never hallucinates fake tool outputs. |
| **Pillar 4 (20%)**<br>**Natural Human Language, World Knowledge & Anti-Slop Prose** | Eliminates "Claudish / robotic AI" tone; populates the `4B-A1B` MoE-Engram with humanities, literature, and conceptual world knowledge. | • [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (10k 100%-human-written texts)<br>• [`OpenAssistant/oasst2`](https://huggingface.co/datasets/OpenAssistant/oasst2) (human multilingual trees)<br>• [`SAA-Lab/LitBench`](https://huggingface.co/datasets/SAA-Lab/LitBench) (debiased human literary prose)<br>• [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) & `tulu-v2` (`LIMA` human split)<br>• [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) (`correctness >= 4, helpfulness >= 4`)<br>• [`HuggingFaceTB/Cosmopedia-v2`](https://huggingface.co/datasets/HuggingFaceTB/Cosmopedia-v2) (encyclopedic world knowledge)<br>• `ChaoticNeutrals/Creative_Writing-ShareGPT` & `Pageshift-Entertainment/LongPage` | **100% Human / De-Slopped Prose**: Trained without synthetic `<think>` prefixes when `has_cot=False` so the model's surface language stays crisp, natural, and human. |
| **Pillar 5 (15%)**<br>**Multi-Turn Technical & Everyday Dialogue** | Long-horizon multi-turn instruction following and context retention. | • `nvidia/Nemotron-Instruction-Following-Chat-v1`<br>• `NousResearch/Hermes-3-Dataset`<br>• `HuggingFaceH4/ultrachat_200k`<br>• `DataCreatorAI/Multi-Turn-Conversational-SFT` | Multi-turn history packing (`up to 6 turns`) with assistant-only loss masking. |

### 6.3 Interleaved Thinking + Tool-Call Execution Contract (`_VisitCache` Native)
All Pillar 3 trajectories are normalized into a unified **Interleaved Thinking + Tool-Call format** compatible with `SparkXLoop`'s `_VisitCache`:
```xml
<|im_start|>system
You have access to the following tools:
<tools>[{"name": "python_exec", "description": "Execute Python code for exact math or data analysis", "parameters": {...}}]</tools>
Reason step-by-step inside <think>...</think>. When you need a tool mid-reasoning, close </think> and emit <tool_call>{"name": ..., "arguments": ...}</tool_call>.<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
<think>
1. I need to solve the coupled rate system and verify the root numerically...
2. Let me call `python_exec` to verify the exact algebraic solution.
</think>
<tool_call>
{"name": "python_exec", "arguments": {"code": "import sympy as sp\n..."}}
</tool_call><|im_end|>
<|im_start|>tool
<tool_response>
{"status": "ok", "output": "a = 1/20, hours = 20"}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
The tool confirmed `a = 1/20` (20 hours). Now I can present the clean step-by-step derivation.
</think>
...<|im_end|>
```
- **Crucial Loss Masking Rule**: During training, token loss is active (`mask = 1`) on every `<|im_start|>assistant ... <|im_end|>` span (both `<think>` and `<tool_call>`), and strictly masked (`mask = 0`) on `<|im_start|>tool ... <|im_end|>`, preventing the model from ever trying to forge tool outputs itself.

---

## 7. Long-Term Horizon Roadmap: `Spark-X2.6-4B-En36B-A1B` (`40B` Total / `5.2B` Active)

Once `Spark-X2.6-4B-En4B-A1B` (`8.2B` total / `5.2B` active) validates the 4B Engram scaling curve, the architecture scales naturally to **`Spark-X2.6-4B-En36B-A1B`** (`4.2B` Looped 72-Layer Backbone + `36.0B` Total / `1.0B` Active Mixture-of-Engrams = **`40.2B` Total Parameters with `5.2B` Active Parameters per Token**), synthesized from four 2024–2026 research breakthroughs:

### 7.1 Four Research Pillars Enabling `En36B-A1B`
1. **Meta FAIR — *Memory Layers at Scale* (*Berges et al., Dec 2024*, [arXiv:2412.09764](https://arxiv.org/abs/2412.09764))**:
   - Proves that scaling product-quantized key-value memory layers up to **128B parameters** on smaller (`1B–8B`) backbones outperforms dense models with **$>2\times$ the compute FLOPs** and standard MoEs, with the largest gains on **factual world-knowledge benchmarks (TriviaQA, NaturalQuestions, MMLU)**.
2. **ByteDance Seed — *UltraMemV2: Memory Networks Scaling to 120B Parameters* (*Aug 2025*, [arXiv:2508.18756](https://arxiv.org/abs/2508.18756))**:
   - Demonstrates that **120B total / 2.5B active (`2.1%` sparsity)** memory networks match 8-expert MoEs while cutting memory-access cost (MAC) by **83%** and boosting long-context & multi-round memorization (`+6.2 pt`) when memory layers are integrated across **every block** rather than only sparse intervals.
3. **Google DeepMind — *Mixture of A Million Experts (PEER)* (*He, July 2024*, [arXiv:2407.04153](https://arxiv.org/abs/2407.04153))**:
   - Uses **Cartesian Product-Key Routing** ($N = \sqrt{N} \times \sqrt{N}$, e.g. $1024 \times 1024 = 1,048,576$ fine-grained memory slots) to reduce router scoring complexity from $O(N \cdot d)$ to **$O(\sqrt{N} \cdot d)$**, allowing a `36B` parameter bank to be routed in microseconds.
4. **Microsoft Research — *KBLaM* ([arXiv:2410.10450](https://arxiv.org/abs/2410.10450)) & *MemoryLLM / M+* ([arXiv:2502.00592](https://arxiv.org/abs/2502.00592))**:
   - Solves the **"Empty 36B Hard Drive" problem**: instead of needing 1 Trillion tokens of from-scratch pretraining to fill 36B parameters with world facts, external encyclopedic/scientific knowledge bases are encoded offline into the `36B` key-value memory slots using frozen representations + lightweight trainable linear adapters (`~1B` active adapter parameters).

### 7.2 Architectural Blueprint of `Spark-X2.6-4B-En36B-A1B`
Following **UltraMemV2**'s "every-block" principle, `En36B-A1B` places a **`1.0B`-parameter `TieredMoEEngram` across all 36 physical layers (`36 layers × 1.0B = 36.0B` total Engram parameters)** while keeping active Engram parameters to **`~27.8M` per layer (`1.00B` active across 36 layers)**:

| Tier per Layer (× 36 Layers) | `En36B-A1B` Specification | Params / Layer | Total (36 Layers) | Active / Layer | Hardware Tier |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier 1 (Zipfian N-Gram Bank)** | `327,680` buckets × `512` dim × 3 scales | `503.3M` | **`18.12B`** | `0.006M` ($O(1)$) | **NVMe SSD `mmap` + Host DRAM Cache** |
| **Tier 2 (Hot Shared Expert)** | SwiGLU `2560 -> 1536 -> 2560` | `11.8M` | **`0.42B`** | `11.80M` (`100%`) | **GPU HBM** |
| **Tier 3 (Product-Key PEER/MoE)** | $N = 256 \times 256 = 65,536$ micro-experts (`Top-K = 64`), or `E = 256` experts (`Top-K = 8`) | `484.9M` | **`17.46B`** | `15.97M` (`3.3%`) | **Host DRAM / Multi-GPU Sharded HBM** |
| **Total `En36B-A1B`** | **36 Physical Layers (`72` Virtual Layers)** | **`1.000B`** | **`36.00B`** | **`27.78M` (`1.00B` total)** | **`~10 GB` GPU HBM + `62 GB` Host DRAM/SSD** |
