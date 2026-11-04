# 07 — Faithful D-LSA, Dynamic Memory Formation & Recall, Backbone↔Engram Integration

> Written 2026-11-04. Supersedes the D-LSA parts of `01 §C1` where they conflict (see §2.3). All numbers marked **(arith)** are arithmetic from the verified architecture constants, **not measurements**. Everything marked **(unverified)** needs a run before it can appear in a claim.

## 0. Goal and non-goals

**Goal.** Let the frozen Spark loop base reason over a **logical context of 1–2M tokens on one A100-class GPU + host RAM**, with memory that is *formed on the fly, recalled by the model's own attention geometry, and consolidated by sleep into the vertical MoE-Engram*, without Layer-0 re-embedding or an external embedder. Context-window usage by memory stays ~0 after consolidation.

**Non-goals.** 10M context; training the backbone; the Replica identity project (only the architecture + validation harness live here).

**Why this is different from "RAG on KV".** Market systems either re-tokenise memory (RAG/Letta), prepend stored KV (cartridges/prefix caches), or keep an external graph of text (HippoRAG 2, Graphiti, A-Mem: all text-level, all re-enter through `embed_tokens`). Here the *index* is the model's own position-free key geometry (D-LSA), the *graph* is the Ward hierarchy + access co-occurrence over KV buckets, and the *long-term store* is the weights of the Engram experts.

## 1. Honest status (what exists vs. what the paper does)

| Capability | Official LSA (verified: repo README + `docs/guide/training.md`) | This repo before 2026-11-04 | After LSA-F1 (`6d470a1`) |
| :--- | :--- | :--- | :--- |
| Buckets | Bottom-up MNN **Ward** over pooled keys in learned space, per 512-key block | Contiguous 8-token spans ("Ward" = adjacent pairing); pure-Python O(n³) `ward_bucketize` never wired | `ward_partition`: batched vectorised MNN Ward; 32K keys indexed in 0.25 s CPU (synthetic) |
| Index input | Position-free slice; each KV head's unrotated dims side by side; each group's query heads summed onto the slice of the key head they read | All 16 query heads summed into one 256-d vector (no group alignment); full head incl. rotated dims in places | `free_slice_inputs` (group-aligned; inner product = summed free logits, unit-tested) |
| Distillation | **Per-key KL** to teacher attention; "a loss aimed at the bucket score is a dead end"; docs 16K–1M balanced by length/domain | One short sample, last 16 queries, mean teacher | `distill_indexer` per-key KL on middle region; **not yet run on real long docs** |
| Selection | Whole buckets until ≈2,560 keys + sink + window; exact SDPA on selected keys | Single selection from ≤8 last queries; active only in decode-like calls (`seq_k>32768`, batch 1) | Exact-key selection via `PagedLatentIndex.select`; mean query vector; same call limits until LSA-F3 |
| Gate that proves it | Recall/quality on 128K–10M | **G2 was vacuous**: ~25-token input, `k_budget=32` → all keys selected → recall 1.0 trivially | Real gate defined (LSA-F2): recall ≥ 0.90 at ≥ 64K vs contiguous/random/oracle |

> [!WARNING]
> The tracker row **Phase 2 / G2** (`e61b162`, "recall = 1.0") is plumbing only. It is downgraded in the README. G3/G6 toy gates (`max_null_space_leakage=0.0` on synthetic states) are also plumbing, not claims.

## 2. Faithful D-LSA design (implemented in `sparkx_dlsa.py`)

### 2.1 Pipeline per global layer (9 layers: L3,7,11,15,19,23,27,31,35; 4 KV heads, 16 Q heads, head_dim 256, rope dims `[0:64)`)

1. **Inputs.** `q_in = Σ_{h∈group g} q_h[64:256]` placed in slot g (`R^{4·192=768}`), `k_in = k_g[64:256]` side by side (`R^768`).
2. **Learned maps** `W_q, W_k: 768→64`, L2-normalised, learned temperature τ. Init = PCA of `k_in` (`fit_pca`) so the *training-free* variant is a legitimate baseline; refine by distillation. ≈ 0.1M params/layer.
3. **Pool + Ward.** Blocks of 512 keys (after the 4 sinks), pool 4 → 128 pools → MNN Ward (`s_a s_b/(s_a+s_b)‖c_a−c_b‖²`, ≤16 tokens/bucket, target avg 8) → ≈ 64 buckets/block. Remainder < 512 keys + local window are always read (cost: up to `window+511` extra keys; documented in tests).
4. **Select.** Score bucket centroids with the (mean of) query index vector(s); take whole buckets until the budget (2,048 long-range keys) is hit; return **exact** key positions (values never averaged). Sink + window added.
5. **Per-key KL distillation.** Teacher = mean-over-heads causal attention probability of the frozen model for sampled queries; restricted to the middle region (causal, outside sink/window) and renormalised; student = `softmax(τ⟨q_idx,k_idx⟩)` over the same keys; loss = KL(teacher‖student). Ward is *not* in the loss (matches the paper's guidance).
6. **Fidelity guard.** `free_logit_fidelity` reports the Pearson r between the free-slice logit and the full logit per layer. If r is low for a layer the free slice is a poor index space and that layer should keep the exact path or use the full head (decision rule in LSA-F2).

### 2.2 What the unit tests do and do not show
Tests (`tests/test_dlsa.py`, 11) check mechanism on synthetic scattered-topic keys: group-aligned inner product identity, Ward purity on planted interleaved clusters, budget/causality/sortedness, faithful ≫ contiguous & random, oracle ≥ faithful, KL decreases. They do **not** show recall on real Spark attention.

### 2.3 Corrections to earlier docs
- `01 §C1.2` item 4 specified *bucket-mass KL + recall hinge*. The official training guide says bucket-targeted losses fail; the objective is **per-key KL** (fixed in `01`).
- `AGENTS.md` NoPE claim is already superseded by its errata (partial RoPE 0.25).

## 3. Milestones LSA-F (each gated; commit + tracker row on completion)

| ID | Deliverable | Gate (claim-grade) | Status |
| :--- | :--- | :--- | :--- |
| LSA-F1 | `sparkx_dlsa.py` + tests | 11 unit tests pass | 🟩 `6d470a1` |
| LSA-F2 | **Real-data recall.** Capture q/k via `capture_attention_io` on real long documents (8K/32K/64K/128K, balanced by length/domain, doc-hash split); per layer: `free_logit_fidelity`, PCA-init recall, distilled recall; table of arms {faithful, flat_oracle_index, contiguous, random, oracle} at k=2048 | faithful recall ≥ **0.90** at 64K on held-out docs and ≥ contiguous + 0.10; flat_oracle − faithful ≤ 0.03 (clustering loss small); report per-layer | 🟨 code + 7 CPU tests (`sparkx_lsa_eval.py`, notebook `SparkX_Loop_v03_LSA_Colab.ipynb`, `6b608d2`); **GPU run pending** |
| LSA-F3 | **End-to-end decode.** Swap `FaithfulDLSAIndexer` into `patch_spark_attention_for_lsa`; single/multi-needle retrieval at 32K/64K/128K; held-out CE on long docs: exact vs D-LSA | needle accuracy within 2 pts of exact attention at 64K; ΔCE ≤ 0.02 nats vs exact; fallback to exact if index missing | ⬜ |
| LSA-F4 | **Hot/cold paging.** GPU-resident centroids + labels; cold K/V pages in pinned host RAM (FP8 optional); selected-page fetch with step-to-step selection reuse | 1M-token run on A100 without OOM; tokens/s reported; PCIe fetch bytes/token logged | ⬜ |
| LSA-F5 | **Recursion-wise KV sharing** (L5) enabled for long context (72 → 36 KiB/token) | ΔCE vs unshared ≤ 0.02 nats on long-doc held-out set (else pass-1 KV only for cold tier) | ⬜ L5 exists as plumbing |
| LSA-F6 | **Dynamic memory graph + recall** (§4) | multi-session recall suite (below) beats contiguous-selection and recency baselines at equal key budget | ⬜ |
| LSA-F7 | **Sleep consolidation** of salient buckets → ESM → Engram (existing C4/C5/C7), buckets evicted | post-eviction recall ≥ 0.8 × pre-eviction recall with **0** KV tokens; held-out CE unchanged (≤ +0.01) | ⬜ depends on Engram being a net-positive consumer (§5) |
| LSA-F8 | 1M → 2M scale test, FP8 cold tier | quality vs 128K reference on needle + long-QA | ⬜ stretch |

### 3.1 Memory arithmetic (arith; bf16, head_dim 256, 4 KV heads, 9 global layers)
- KV per token per global layer = 2·4·256·2 B = 4 KiB → **36 KiB/token** (36L), **72 KiB/token** (72L loop, 18 global visits), back to 36 KiB with recursion-wise sharing. SWA layers are bounded by the 512 window (constant).
- 1M tokens: 36 GiB (shared) / 72 GiB (unshared); 2M: 72 GiB / 144 GiB. **FP8 halves** these. A Colab A100 host-RAM size is **unverified** — check `free -g` at runtime before choosing 1M vs 2M.
- GPU-resident index: 64-d bf16 centroid = 128 B per ~8 tokens per layer → ~16 B/token/layer → **≈144 B/token** over 9 layers → 2M ≈ 0.29 GB, plus labels. Centroid scoring per decode step ≈ 0.29 GB read (≈ 0.2 ms at 1.5 TB/s, **arith**).
- **Real bottleneck:** fetching ≈2,560 selected keys/layer/token from host = ~92 MB/token → ~4 ms/token over PCIe at ~25 GB/s (**arith**). Mitigation to test in LSA-F4: reuse the selected set for N decode steps (selection is temporally coherent), keep an LRU hot-page set on GPU, prefetch with the co-access graph (§4).

## 4. Dynamic memory formation and recall (LSA-F6/F7)

**Memory tiers** (nested-learning timescales, `01 §1.1`): M0 SWA window (every token) · M1 GPU hot KV (D-LSA selected) · M2 host cold KV pages (centroids on GPU) · M3 ESM fast weights (minutes–hours) · M4 Engram experts (slow, sleep-compiled).

**Formation (wake, no extra forward passes).** Every query already scores bucket centroids; we log (a) per-bucket attention-mass estimate (bucket-score softmax, calibrated against exact on the capture pass), (b) per-bucket access count, (c) **co-access edges** (buckets selected by the same query), (d) per-span surprise (token NLL) from the forward we are doing anyway. Salience `S_b = AccessMass · Surprise · |Valence|` (existing `C2`). Memory graph = Ward hierarchy (parent = merged centroid; second level over bucket centroids gives topics) + co-access edges. Nothing is re-embedded.

**Recall (wake).** (1) Direct: query→centroid scoring (above). (2) **Spreading activation**: from the selected buckets, propagate along co-access edges with a Personalized-PageRank-style step (HippoRAG-2 idea applied to KV buckets, not text) to *prefetch* neighbouring cold pages into the hot set — used for latency hiding and for multi-hop recall; evaluated against the same budget without spreading (ablation).

**Consolidation (sleep).** Select buckets by salience and graph centrality (hub-and-cluster summaries, A-Mem-style linking done on geometry), compile via C3/C4 (self-study probes, LRCD residual targets) into ESM, then — if Engram passes §5 — into Engram experts with the null-space compiler; evict the compiled buckets from M2. Conflict handling (new fact contradicts old): Graphiti-style *invalidate, don't overwrite* maps to ledger entries with exact unlearning (Phase 5).

**Recall suite (LSA-F6 gate).** Synthetic-but-leak-free multi-session needle/graph tasks generated with fresh entity names: single-hop, multi-hop (2–3 co-access hops), update/contradiction, temporal ordering, paraphrase queries. Report accuracy and keys read; baselines: recency window, contiguous-span selection, flat oracle index, exact attention at ≤ 128K.

## 5. Backbone ↔ Engram integration evaluation (BIE)

**Motivation (doc 06, measured).** Engram hurts held-out CE (OFF 1.5090 → FULL 1.5750), the lexical tier contributes ≈0, anchors 7/11/23/31 are net harmful and gating them gives CV CE 1.4901 < OFF. Hypothesis H6: the frozen consumer cannot read what the Engram writes.

**What the Engram paper does (from the arXiv abstract page and secondary summaries; (unverified) details):** early placement (≈ Layer 2; sweeps also show mid layers), hashed N-gram addressing (deterministic, prefetchable), sigmoid context gate on RMSNorm'd hidden-state/memory-key dot product, short causal depthwise conv, co-trained from scratch. Differences to ours: anchors after whole blocks at stride 4 (not early), semantic router (not token hash), attached to a pre-trained, frozen consumer. "Retrofit mismatch" is *our inference*, consistent with H6, not a paper claim.

**BIE experiments (CPU/Colab, no new training unless noted):**
1. **Knowledge locus.** Causal tracing / logit lens on closed-book QA (held-out entity facts): at which layers does the *backbone* store a fact (MLP of which layers)? Compare with anchor layers. Output: per-layer indirect effect; decides where an anchor can help.
2. **Anchor placement sweep.** Add *early pre-attention* anchors (L1–L3) as a candidate vs. current post-block anchors; measure held-out CE and closed-book QA with the long-run checkpoint.
3. **Gate design.** Compare current ctx gate vs. paper-style sigmoid-of-normalised-dot gate vs. CV-selected anchor gate; metric: ΔCE and fraction of tokens where gate > 0.5.
4. **Consumer adaptation (owner decision H6).** Only if 1–3 fail: small trainable adapters (LoRA on `out_proj`/norms of the 4 layers after each anchor) **trained with Engram on, backbone base frozen**. This changes the "100 % frozen" statement to "frozen base + tiny adapters" and must be approved by the owner; reported as an explicit ablation arm.
5. **Gap-write test (SI4).** Write facts the backbone verifiably lacks (entities absent from all corpora, verified by closed-book failure) into Engram via the compiler; measure closed-book accuracy on paraphrases. This is the *only* test that shows Engram adds knowledge rather than reshuffling what the backbone already knows.

**Implemented (2026-11-04, `sparkx_bie.py`, untested on GPU; attribution math verified against finite differences in `tests/test_bie.py`):**
- **Read of the code that matters for H6/H7.** In the looped path the in-loop anchors (L7–L23; L11–L23 also in P3) add `delta` into stream 0 on **every pass** (passes > 0 scaled by `ffn_extra_gain`); the same token-hash/semantic memory is therefore re-read into an already-processed recurrent state. L3 (prelude), L27 (end of loop), L31 (coda) fire once or on the final passes. The measured harmful anchors {7, 11, 23, 31} vs helpful {3, 27} are consistent with, but do not prove, a pass-schedule effect (**H7**).
- **First-order attribution per (anchor, pass):** `attribution = Σ_t ⟨∂NLL/∂δ_t, δ_t⟩ / N` from one backward pass with zero probes; reports relative write size `‖δ‖/‖h‖`, alignment `cos(δ, −g)`, fraction of tokens hurt, and the worst tokens. This replaces ~20 ablation forwards by one backward per text and separates *which pass* hurts, which leave-one-anchor-out cannot.
- **Pass-filter sweep (`model.engram_pass_filter`, default `None` = trained behaviour):** held-out CE with in-loop anchors restricted to pass 0 / passes 0–1 / all. If `pass0_only` < trained CE, the fix is a schedule/gain change (cheap), not an adapter.
- Notebook: Cell 6 of `SparkX_Loop_v03_LongRun_Colab.ipynb` (`run_bie`), copy-paste JSON.
- **Reading rule (pre-registered):** an anchor/pass is *harmful* if attribution > +0.002 nats/token **and** confirmed by leave-one-out/pass-filter CE; *misaligned* if `cos(δ, −g) < 0.02` (the write is not in a loss-decreasing direction → points to H6 consumer mismatch rather than a gain problem). Few texts ⇒ low power; extend `texts` before acting on small effects.

**Decision rule.** Engram stays in the main configuration only if held-out CE(ON) ≤ CE(OFF) − noise floor **or** the gap-write test succeeds with CE unaffected. Otherwise it is a measured negative result and the main result is the loop base + ESM/D-LSA memory.

## 6. Risks
| Risk | Mitigation |
| :--- | :--- |
| Free slice (192 of 256 dims) has low logit fidelity in some layers | per-layer `free_logit_fidelity` gate; exact path for those layers |
| Distilled maps trained on short docs fail at 1M | length-balanced distillation (16K–128K), evaluate at 2× longest train length, report degradation |
| Host fetch bandwidth dominates decode | selection reuse, LRU hot set, FP8 cold pages, co-access prefetch; report tokens/s honestly |
| Looped passes revisit different KV (72 KiB/token) | recursion-wise sharing (LSA-F5) with measured ΔCE |
| Colab host RAM < 72 GiB | 1M first; FP8; unverified until `free -g` |
| Engram remains net-negative | §5 decision rule; honest negative result |

## 7. References (all arXiv ids web-checked 2026-11-04)
- Latent Sparse Attention, `Prefrontal-AI/latent-sparse-attention` (README, `docs/guide/training.md`); arXiv:2610.42147 (R3 in `04`).
- DeepSeek-AI, *Conditional Memory via Scalable Lookup* (Engram), arXiv:2601.07372.
- Gutiérrez et al., *HippoRAG 2: From RAG to Memory*, arXiv:2502.14802.
- Xu et al., *A-MEM: Agentic Memory for LLM Agents*, arXiv:2502.12110.
- Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* (Graphiti), arXiv:2501.13956.
- Lin et al., *Continual Learning via Sparse Memory Finetuning*, arXiv:2510.15103 (R in `04`).
