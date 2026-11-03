# AGENTS.md — Spark-X2.6-4B-En4B-A1B Architecture, Research & Vertical Sleep-Time Memory Compiler Plan

> **Scope & Operational Mandate**
> 1. **Frozen Backbone Philosophy**: We build strictly atop the post-trained 36-layer **`XHToken/Spark-X2.6-4B`** hybrid backbone (27 Sliding-Window RoPE layers + 9 Global NoPE layers), preserving its pre-trained linguistic and reasoning priors without trillion-token retraining.
> 2. **Horizontal Working Attention vs. Vertical Long-Term Memory**:
>    - **Horizontal Context Window (`D-LSA` on 9 Global NoPE Layers + 27 Sliding-Window Layers)**: Treated strictly as **human working attention** for live problem interpretation, planning construction, and active episodic trajectories. We reject market-standard RAG / context-stuffing harnesses that re-embed memory through `Layer 0` (`embed_tokens`) and cannibalize the active context window.
>    - **Vertical Long-Term Memory (`ZipfianTieredMoEEngram` + `VerticalSleepTimeMemoryCompiler`)**: Episodic semantic clusters formed in the 9 Global NoPE attention layers during **Wake Mode** are compiled **vertically straight down** into the layer-matched **`MoE-Engram` (`4.11B`)** weights (`L3, 6, 11, 15, 16, 19, 23, 27, 31`) during **Sleep Mode** via **Null-Space Constrained Closed-Form Compilation (`AlphaEdit`)**, bypassing `Layer 0` entirely and freeing horizontal KV memory.
> 3. **Separation from Parallel `Replica` Identity Project**: The four personal-identity evaluation dimensions of the parallel `Replica` project live in its dedicated repository. This repository (`Spark-X2.6-Recursion`) provides the **self-defensible, paper-grade neural architecture, nested learning hierarchy, sleep-time memory compiler, and empirical validation harness**.
> 4. **Git Stage Discipline**: Every research milestone, architectural revision, and validation harness update MUST be syntax-checked (`ast.parse`), committed, and pushed to `origin/main` by stage in real time.
> 5. **Roadmap Is the Source of Truth**: Implementation order, acceptance criteria and status live in [`docs/roadmap/`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/docs/roadmap/README.md). Update the milestone tracker there whenever a milestone changes state.
> 6. **The 72L Loop Base Is the Base (decision 2026-11-03)**: The frozen backbone executed as the looped core (prelude `L0–3` → `L4–27` × 2 passes, + optional P3 over `L12–23` → coda `L28–35`), mixed by 24 `TwoStreamMHC` controllers, is the base architecture for all memory work and all main-result evaluations. 36L (loop off) is an ablation only. Thesis: *the loop supplies knowledge manipulation, vertical memory supplies knowledge storage* (Ouro arXiv:2510.25741; McLeish et al. arXiv:2511.07384). See [`01_ARCHITECTURE_DESIGN.md` C9](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/docs/roadmap/01_ARCHITECTURE_DESIGN.md).

> [!WARNING]
> **Errata (verified 2026-11-03 against `XHToken/Spark-X2.6-4B` `config.json` / `modeling_spark.py` and this repo's code).** These override any conflicting statement below:
> 1. The 9 full-attention layers are **not NoPE**: they use **partial RoPE** (`partial_rotary_factor=0.25`, `rope_theta=5e6`), so 64 of 256 head dims are rotated. Sliding layers use full RoPE (`theta=1e4`, `w=512`). Any "100% position-invariant" claim is wrong; only the unrotated 192-dim slice is position-free.
> 2. Spark attention has a **head-wise sigmoid output gate** (`g_proj`, `headwise_attn_output_gate=true`) applied *before* the output projection, which is named **`out_proj`** (not `o_proj`). Attention residual = `out_proj(sigmoid(g_proj(LN h)) ⊙ SDPA(q,k,v))`.
> 3. The real `MoE-Engram` anchors are **`{3, 7, 11, 15, 19, 23, 27, 31}`** (8 anchors, stride 4), i.e. exactly the full-attention layers minus `L35`. The list `{3, 6, 11, 15, 16, ...}` used below is wrong. Each anchor is applied **after the whole decoder block** (attention + MLP), not directly after attention.
> 4. The Stage 2 sleep-compiler code (commit `47a7cc6`) is **not runnable yet** (calls non-existent `_layers()`, `forward(passes=...)`, `format_spark_chat`, `evaluate_multi_domain_ppl`; looks up `o_proj`). It is now opt-in (`run_sleep_compiler_validation=False`). See [`00_REVIEW_AND_VERIFIED_FACTS.md`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/docs/roadmap/00_REVIEW_AND_VERIFIED_FACTS.md).
> 5. "Mathematically zero forgetting" holds only for the protected subspace that is actually projected out. The current code protects previously compiled keys only, not general pre-trained behaviour (AlphaEdit builds `K0` from ~100K general-corpus keys).

---

## Stage 1: Comprehensive Literature Investigation & Synthesis (2025–2026)

Our architecture synthesizes and advances six converging lines of 2025–2026 frontier research:

### 1.1 Nested Learning & Multi-Timescale Associative Memory
- **Behrouz, Razaviyayn, Zhong, & Mirrokni (Google Research, NeurIPS 2025)** — *"Nested Learning: The Illusion of Deep Learning Architectures"*:
  - Demonstrates that neural architectures and gradient optimizers are mathematically dual views of **nested, multi-level associative memory systems** operating at distinct update frequencies $(\tau_0 < \tau_1 < \tau_2)$, where each inner level compresses its own context flow while outer levels consolidate persistent structure without catastrophic forgetting.
- **Behrouz et al. (Google Research, 2025)** — *"Titans: Learning to Memorize at Test Time"*:
  - Separates short-term localized attention from a persistent neural long-term memory module whose updates are gated by **prediction surprise** ($-\log p_\theta(x_t \mid x_{<t})$), scaling past 2M tokens.

### 1.2 Latent Sparse Attention (`LSA`) & Subquadratic KV Clustering
- **Kubicek et al. (`Prefrontal-AI/latent-sparse-attention`, `arXiv:2610.42147`, 2026)** — *"Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention"* (building on DeepSeek **NSA**, **SALS**, and **JSSA**):
  - Projects all KV heads and head-summed Queries into a shared low-dimensional **64-D unit hypersphere** ($k_j^{\text{idx}}, q_i^{\text{idx}} \in \mathbb{S}^{63}$), applies contiguous $n_{\text{pool}}=4$ pooling, and partitions keys outside the sink ($n_{\text{sink}}=4$) and local window ($n_w=512$) into **semantic buckets** via bottom-up mutual-nearest-neighbor **Ward clustering** ($d(a,b) = \frac{s_a s_b}{s_a + s_b}\|c_a - c_b\|_2^2$).
  - Crucially, in standard RoPE transformers, `D-LSA` must strip rotary embeddings before projecting into $\mathbb{S}^{63}$ to avoid position-angle phase distortion. In **`Spark-X2.6-4B`**, the 9 global full-attention layers (`L3, 7, 11, 15, 19, 23, 27, 31, 35`) **natively use NoPE (`partial_rotary_factor = 0.0`)**, making `D-LSA` semantic Ward clustering **100% position-invariant and distortion-free**.

### 1.3 Sleep-Time Compute, KV Cartridges & Context-to-Weight Distillation
- **Lin et al. (Letta & UC Berkeley, 2025)** — *"Sleep-Time Compute: Beyond Inference Scaling at Test-Time"*:
  - Shifts heavy memory organization, trajectory reflection, and consolidation from latency-critical active inference ("Wake") to offline idle cycles ("Sleep"), cutting test-time compute by up to $5\times$.
- **Eyuboglu et al. (Hazy Research, Stanford, 2025)** — *"Cartridges"* & **Sakana AI (2026)** — *"Doc-to-LoRA (D2L)"*:
  - Compress long documents into pre-computed KV caches (*Cartridges*) or hypernetwork-generated LoRA adapters (*Doc-to-LoRA*).
  - **Limitation of Existing Approaches**: *Letta/RAG* re-injects text summaries through `Layer 0` (`embed_tokens`), consuming working context window. *Cartridges* still occupy active KV cache slots during attention. *Doc-to-LoRA* perturbs backbone weights via dense LoRA matrices, accumulating interference across sequential updates.

### 1.4 Conditional Memory via Scalable Lookup (`Engram`) & Null-Space Continual Editing (`AlphaEdit`)
- **DeepSeek-AI (2025)** — *"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"*:
  - Introduces $O(1)$ deterministic N-gram lookup tables fused into intermediate transformer layers to offload static factual/lexical reconstruction from dense FFN compute.
- **Fang et al. (ICLR 2025 Outstanding Paper)** — *"AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"*:
  - Proves that sequential weight updates $\Delta W$ to FFN key-value stores avoid catastrophic forgetting **if and only if** the update is projected into the **null space** $P_{\text{null}} = U_2 U_2^\top$ of the uncentered covariance matrix $K_0 K_0^\top = U \Lambda U^\top$ of preserved historical activations, guaranteeing $\Delta W \, K_0 \equiv 0$.

### 1.5 Recurrent Depth & Manifold-Constrained Hyper-Connections (`mHC`)
- **Geiping, McLeish, Goldstein et al. (2025)** — *"Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"*:
  - Shows that iterating a shared recurrent transformer block in continuous latent space scales reasoning depth without verbose token mimicry.
- **Manifold-Constrained Hyper-Connections (`TwoStreamMHC`)**:
  - Bounds multi-pass recurrent residual mixing on the Birkhoff polytope $\mathcal{B}_2$ of $2\times 2$ doubly-stochastic Sinkhorn matrices ($\|P\|_2 \le 1$), preventing activation explosion across 72 virtual layers.

---

## Stage 2: Paradigm Comparison & Critical Self-Review of the Construction Plan

### 2.1 Why Vertical Latent Memory Beats Horizontal Context/RAG Harnesses

| Architectural Dimension | Market Context Harnesses (RAG / Letta Text Blocks) | KV Cartridges (Stanford Hazy) | Hypernetwork LoRA (Sakana Doc-to-LoRA) | **Ours: Vertical LSA $\rightarrow$ Null-Space `MoE-Engram` Compiler** |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Injection Entrypoint** | **Horizontal (`Layer 0`)**: Re-tokenizes text & passes through `embed_tokens` | **Horizontal (KV Cache)**: Prepends synthetic KV slots to every attention head | **Backbone Weights**: Modifies dense $W_Q, W_V, W_{\text{MLP}}$ via low-rank adapters | **Vertical (`L3, 6, 11, 15, 16, 19, 23, 27, 31`)**: Direct residual injection at layer-matched anchors |
| **Working Context Window Cost** | High (steals $2\text{K}-32\text{K}$ active prompt tokens per query) | Moderate (consumes active KV cache slots & HBM bandwidth) | Zero context tokens, but bounded by rank $r \ll d$ | **Zero context tokens** once compiled; frees `D-LSA` working attention |
| **Continual Capacity Scaling** | External vector DB (requires external embedder model) | Linear HBM growth per mounted cartridge | Fixed low-rank bottleneck ($O(r \cdot d)$) | **`4.11B` Zipfian 3-Tier `MoE-Engram`** (`32` Routed SwiGLU Experts + `65K` N-gram buckets/anchor) |
| **Sequential Forgetting Risk** | Retrieval pollution / lost-in-the-middle | Multi-cartridge distribution mismatch | High after $>50$ sequential merges | **Mathematically Zero ($\Delta W \, A_0 \equiv 0$)** via `AlphaEdit` SVD null-space projection |
| **Home-Lab Sleep Compute Cost** | Low, but cannot internalize procedural/latent abstractions | Requires gradient optimization per cartridge | Requires pre-trained hypernetwork | **Closed-form linear solver ($<200\text{ ms}$/anchor, zero backbone backprop)** |

---

### 2.2 Adversarial Review & Mathematical Resolution of 5 Potential Vulnerabilities

Before writing a single line of compiler code, we subjected our initial construction plan to five adversarial paper-reviewer critiques and resolved each mathematically:

#### Critique 1: *Why can a Horizontal Attention KV Cluster in Layer $\ell$ be compiled into a Vertical `MoE-Engram` Module at Layer $\ell$?*
- **Resolution (Exact Layer-Matched Residual Equivalence)**:
  - Look at the layer indices of `Spark-X2.6-4B`:
    $$\mathcal{L}_{\text{NoPE}} = \{3, 7, 11, 15, 19, 23, 27, 31, 35\}, \qquad \mathcal{L}_{\text{Engram}} = \{3, 6, 11, 15, 16, 19, 23, 27, 31\}$$
  - Seven of the eight `MoE-Engram` anchors (`L3, 11, 15, 19, 23, 27, 31`) sit **immediately after** the corresponding Global NoPE Attention layer $\ell$ (and `L6` sits immediately before `L7`).
  - During **Wake Mode**, when token $x_t$ (with normalized hidden state $h_t^{(\ell)} \in \mathbb{R}^{2560}$) attends to a long-range `D-LSA` semantic cluster $\mathcal{B}_b$ in Global NoPE layer $\ell$, that cluster contributes an additive residual vector to the stream:
    $$\mathbf{y}_b^{(\ell)}(h_t^{(\ell)}) = W_O^{(\ell)} \sum_{j \in \mathcal{B}_b} \alpha_{t,j}\, V_j^{(\ell)} \in \mathbb{R}^{2560}$$
  - Immediately after layer $\ell$, the `ZipfianTieredMoEEngram` module writes into the **exact same residual stream** $h_t^{(\ell)}$:
    $$\Delta h_{\text{Engram}}^{(\ell)}(h_t^{(\ell)}) = g_\ell \cdot W_{\text{proj}}^{(\ell)} \left( \text{ ctx}(h_t^{(\ell)}) \odot \left[ \frac{1}{2} e_{\text{lex}} + \text{Shared}(h_t^{(\ell)}) + \sum_{e \in \text{Top-}K} \gamma_e(h_t^{(\ell)})\, \mathbf{a}_e(h_t^{(\ell)}) W_{\text{down}, e}^{(\ell)} \right] \right)$$
  - Therefore, compiling cluster $\mathcal{B}_b$ into expert $e^*$ at layer $\ell$ so that $\Delta h_{\text{Engram}}^{(\ell)}(\bar{h}_b^{(\ell)}) = \mathbf{y}_b^{(\ell)}$ preserves the **exact layer-$\ell$ residual trajectory** after $\mathcal{B}_b$ is evicted from the horizontal KV cache!

#### Critique 2: *In `ZipfianTieredMoEEngram`, `project` ($W_{\text{proj}}^{(\ell)}$) sits after `expert_down` ($W_{\text{down}, e}^{(\ell)}$). How do we solve for $\Delta W_{\text{down}, e^*}^{(\ell)}$ when $W_{\text{proj}}^{(\ell)}$ is in front of it?*
- **Resolution (Pseudoinverse Pullback through $W_{\text{proj}}^{(\ell)}$ & Context Gate)**:
  - In [`ZipfianTieredMoEEngram.forward`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L1634-L1653), the output is:
    $$\Delta h_{\text{Engram}}^{(\ell)} = g_\ell \cdot \left( \left[ \left(\frac{1}{2}\text{lex} + \text{shared} + \text{routed}\right) \odot \text{ctx}(h) \right] W_{\text{proj}}^{(\ell)\top} \right)$$
  - If we want to add a target residual increment $\delta \mathbf{y}_b^{(\ell)} \in \mathbb{R}^{2560}$ for cluster $b$ routed to specialist expert $e^*$ with gate weight $\gamma_{e^*}(h_b)$ and effective scalar pass gate $g_\ell$:
    1. We pull $\delta \mathbf{y}_b^{(\ell)}$ back through $W_{\text{proj}}^{(\ell)}$ via its damped SVD / Tikhonov pseudoinverse $(W_{\text{proj}}^{(\ell)})^{\dagger}_{\lambda}$ and divide by $\max(0.1, \text{ctx}(h_b) \cdot g_\ell \cdot \gamma_{e^*}(h_b))$ to obtain the exact target expert output residual $\mathbf{r}_b^{(e^*)} \in \mathbb{R}^{2560}$.
    2. Even better, if $W_{\text{proj}}^{(\ell)}$ has low rank in a newly initialized bank, we jointly update the null-space-projected $W_{\text{down}, e^*}^{(\ell)}$ and a rank-1 residual on $W_{\text{proj}}^{(\ell)}$ so the compiled residual matches $\delta \mathbf{y}_b^{(\ell)}$ with $<1\%$ relative L2 error!

#### Critique 3: *How do we prevent sequential Sleep-Mode compilations from overwriting previously compiled clusters inside expert $e^*$?*
- **Resolution (`AlphaEdit` SVD Null-Space Projection per Expert)**:
  - Each `ZipfianTieredMoEEngram` anchor maintains a compact running activation covariance matrix (or orthogonal basis sketch) $C_0^{(e^*)} = A_0 A_0^\top \in \mathbb{R}^{R \times R}$ ($R = 3072$) of post-SwiGLU hidden states $\mathbf{a} = \text{SwiGLU}(q W_{\text{gate}, e^*}, q W_{\text{up}, e^*}) \in \mathbb{R}^{3072}$ for preserved/compiled memories.
  - During Sleep Mode, we compute the eigendecomposition / SVD of $C_0^{(e^*)} = U \Lambda U^\top$, select the null-space basis $U_{\text{null}} = \{u_i : \lambda_i \le \tau_{\text{null}}\}$, and form the projector $P_{\text{null}}^{(e^*)} = U_{\text{null}} U_{\text{null}}^\top$.
  - Given a batch of new cluster activations $A_1 \in \mathbb{R}^{R \times M}$ and target expert residuals $R_1 \in \mathbb{R}^{W \times M}$ ($W=2560$), the `AlphaEdit` closed-form update is:
    $$\Delta W_{\text{down}, e^*}^{(\ell)} = P_{\text{null}}^{(e^*)} A_1 \left( A_1^\top P_{\text{null}}^{(e^*)} A_1 + \alpha I_M \right)^{-1} R_1^\top \in \mathbb{R}^{3072 \times 2560}$$
  - **Proof of Zero Forgetting**: For any prior activation $\mathbf{a}_0 \in \text{col}(A_0)$, $P_{\text{null}}^{(e^*)} \mathbf{a}_0 = 0$, hence $\left(\Delta W_{\text{down}, e^*}^{(\ell)}\right)^\top \mathbf{a}_0 \equiv \mathbf{0}$. After compilation, we update the covariance sketch: $C_0^{(e^*)} \leftarrow C_0^{(e^*)} + A_1 A_1^\top$.

#### Critique 4: *How does the X-MoE Router know to route future queries to expert $e^*$ after the horizontal KV cluster is evicted?*
- **Resolution (Shared $\mathbb{S}^{63}$ Centroid Pull + `Tier 1A/1B` Lexical Anchor Compilation)**:
  - When compiling cluster $\mathcal{C}_b$ (with mean normalized hidden state $\bar{h}_b^{(\ell)}$ and characteristic N-gram token hashes $H_b$), we perform **dual addressing compilation**:
    1. **Semantic Hypersphere Addressing (`Tier 3`)**: Compute the router query $u_b = \text{Normalize}(W_{\text{route}}^{(\ell)} \bar{h}_b^{(\ell)}) \in \mathbb{S}^{63}$, assign cluster $b$ to the least-saturated specialist expert $e^*$ in the active domain zone, and align its centroid $c_{e^*} \leftarrow \text{Normalize}((1 - \eta_c) c_{e^*} + \eta_c u_b)$.
    2. **Lexical N-Gram Addressing (`Tier 1A/1B`)**: Compile the cluster's high-TF/IDF N-gram token hashes $H_b$ directly into `tier1a_hot` / `tier1b_cold` so that exact entity/symbol names trigger $O(1)$ deterministic recall even before semantic routing!

#### Critique 5: *How do we foster genuine latent thinking & problem construction in a 4B model rather than verbose CoT template mimicry?*
- **Resolution (Functional Specialization Across the 3 Recurrent Passes of the 72L Loop)**:
  - By offloading factual & episodic storage vertically into `MoE-Engram`, the 72-layer recurrent loop (`Pass 1 -> Pass 2 -> Pass 3` modulated by `TwoStreamMHC`) is trained and operated as a **3-Stage Cognitive State Machine**:
    - **Pass 1 (`L4..L27`, Gain `P1=0.35`) — Intent Deconstruction & Vertical Recall**: Queries `D-LSA` + `MoE-Engram` at `0.35x` gain to ground the user's true goal, constraints, and prior episodic context.
    - **Pass 2 (`L4..L27`, Gain `P2=0.15`, `mHC_P2 ~ 0.09`) — Latent Plan Construction & Counterfactual Search**: Uses the 2-stream Sinkhorn highway (`s0, s1`) to construct multi-step solution plans and test edge cases in continuous latent space (Geiping et al., 2025).
    - **Pass 3 (`L12..L23`, Gain `P3=0.05`, `mHC_P3 ~ 0.06`) — Self-Critique & Anti-Slop Synthesis**: Audits the latent plan for logical/security flaws and emits direct, natural human prose or exact code/math without RLHF filler.

---

## Stage 3: Step-by-Step Architecture Revision & Implementation Status

1. **Step 3.1 — Episodic Cluster Trace Capture in [`LatentSparseAttentionIndexer.extract_salient_knowledge_clusters`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L927-L1024) ([`sparkx_loop_v03_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py))**:
   - Extends [`LatentSparseAttentionIndexer`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L756-L1024) on the 9 Global NoPE layers (`L3, 7, 11, 15, 19, 23, 27, 31, 35`) with [`extract_salient_knowledge_clusters`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L927-L1024):
     - Clusters long-range KV states into semantic buckets in $\mathbb{S}^{63}$.
     - Computes per-cluster **Salience Score** $S_b = \text{AccessMass}_b \times \text{KeySurprise}_b \times |\text{Valence}_b|$.
     - Extracts the exact layer-$\ell$ attention residual target $\mathbf{y}_b^{(\ell)} = W_O^{(\ell)} \sum_{j \in \mathcal{B}_b} \alpha_j V_j^{(\ell)} \in \mathbb{R}^{2560}$, cluster hidden state $\bar{h}_b^{(\ell)} \in \mathbb{R}^{2560}$, evictable horizontal `token_positions`, and peak-salience `trigger_pos`.
2. **Step 3.2 — Vertical Sleep-Time Memory Compiler ([`ZipfianTieredMoEEngram.compile_clusters_alphaedit`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L1768-L1999) & [`SparkXLoopV03.run_sleep_time_memory_compiler`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2628-L2780) in [`sparkx_loop_v03_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py))**:
   - Maps Global NoPE layers `{3, 7, 11, 15, 19, 23, 27, 31}` vertically to their matched `MoE-Engram` anchors `{3, 6, 11, 15, 16, 19, 23, 27, 31}`.
   - Routes each cluster $\mathcal{B}_b$ through the exact X-MoE hypersphere router (`S^63`) to select the highest-affinity active specialist expert $e^*$ with available null-space capacity (`786,432` total orthogonal basis directions across `8 anchors x 32 experts x 3072 rank`).
   - Applies double Classical Gram-Schmidt null-space projection $A_{1, \perp} = A_1 - (A_1 V_{\text{active}}^{(e^*)})(V_{\text{active}}^{(e^*)})^\top$ and solves the **Dual Woodbury / Gram Null-Space Ridge Equation (`AlphaEdit`)** in `float32`:
     $$\Delta W_{\text{down}, e^*}^{(\ell)} = A_{1, \perp}^\top \left(A_{1, \perp} A_{1, \perp}^\top + \alpha I_{M_e}\right)^{-1} \Delta Z_1 \in \mathbb{R}^{3072 \times 2560}$$
3. **Step 3.3 — Empirical Validation Suite ([`validate_vertical_sleep_memory_compiler`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L3910-L4063) in [`sparkx_loop_v03_benchmarks.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py))**:
   - Automated Stage-D benchmark integrated into [`run_continue_from_hub_engram_expansion`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L3812-L3907) verifying:
     1. **100% Frozen Backbone Invariance** ($\max |\Delta W_{\text{base}}| = 0.0$).
     2. **Null-Space Zero-Forgetting Guarantee** ($\|A_{\text{hist}} \Delta W_{\text{down}}\|_\infty \sim 10^{-7}$ across sequential multi-round sleep compilations).
     3. **Pareto-Safe Vertical Memory Consolidation** (`72L + MoE-Engram` held-out Perplexity with `0` extra prompt tokens).
     4. **Horizontal Context Compression & Sub-Second Closed-Form Compilation Latency**.

