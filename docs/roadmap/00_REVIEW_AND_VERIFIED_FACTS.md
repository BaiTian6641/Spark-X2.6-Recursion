# 00 — Review of Stage 2 & Verified Fact Sheet

> **Date verified:** 2026-11-03 · **Repo HEAD at review:** `47a7cc6` (Stage 2), hotfix `b841a3e`
> **Method:** (1) read the repo code paths end-to-end, (2) fetched upstream `XHToken/Spark-X2.6-4B` `config.json` and `modeling_spark.py` from Hugging Face, (3) cross-checked every cited paper with web search (IDs, titles, authors, key claims).

This document is the **ground truth** that the rest of the roadmap builds on. If a statement elsewhere (including `AGENTS.md` Stage 1–3) conflicts with this file, this file wins.

---

## 1. Verified Backbone Facts (`XHToken/Spark-X2.6-4B`)

Source: `https://huggingface.co/XHToken/Spark-X2.6-4B/raw/main/config.json` and `modeling_spark.py` (fetched 2026-11-03).

| Property | Verified value | Consequence for our design |
| :--- | :--- | :--- |
| Architecture | `Spark2_5ForCausalLM`, 36 layers, `hidden_size=2560`, `intermediate_size=10240`, `vocab=131072`, tied embeddings | — |
| Attention | 16 Q heads, 4 KV heads (GQA ×4), `head_dim=256` | KV per token per global layer = `2 × 4 × 256 × 2 B = 4 KiB` |
| Layer pattern | `[SWA, SWA, SWA, FULL] × 9` → full attention at **L3, 7, 11, 15, 19, 23, 27, 31, 35** | 9 global layers carry all long-range KV |
| Sliding layers | `sliding_window=512`, full RoPE, `theta=1e4` | Bounded KV: `27 × 512 × 4 KiB ≈ 54 MiB` regardless of context length |
| **Full layers** | **Partial RoPE**: `partial_rotary_factor=0.25`, `theta=5e6` → dims `[0:64)` rotated, dims `[64:256)` pass-through | **Not NoPE.** Only the 192-dim pass-through slice is position-free (see `apply_rotary_pos_emb`: `x_rot = x[..., :rope_dim]`, `x_pass = x[..., rope_dim:]`) |
| **Output gate** | `headwise_attn_output_gate=true`, `gate_attn_act_mode="sigmoid"`: `g_proj: 2560 → 16`, applied per head **after SDPA, before** the output projection | The true attention residual is `out_proj(σ(g_proj(LN h)) ⊙ SDPA(q,k,v))` (Qwen Gated Attention, arXiv:2505.06708) |
| Output projection name | **`out_proj`** (not `o_proj`); fused **`q_k_v_proj`** | Any `getattr(attn, "o_proj")` returns `None` |
| MLP | `down(act(gate(x)) * up(x))`, `hidden_act="gelu"` (GeGLU) | — |
| Max positions | 1,048,576 | — |

### 1.1 KV-cache arithmetic (why vertical memory matters on a home lab)

| Configuration | Global-layer visits | Global KV per token | 128K tokens | 1M (2²⁰) tokens |
| :--- | :---: | :---: | :---: | :---: |
| Upstream 36L | 9 | 36 KiB | 4.5 GiB | 36.0 GiB |
| Ours 72L (`start=4, end=28, passes=2, P3 12..24`) | **18** (L3,31,35 ×1; L7,11,27 ×2; L15,19,23 ×3) | 72 KiB | 9.0 GiB | **72.0 GiB** |

The loop **doubles** long-range KV because every looped visit gets its own virtual cache slot ([`_VisitCache`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L242), [`_apply_layer`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2905-L2924)). This is the strongest quantitative motivation for compiling episodic context out of the horizontal window.

---

## 2. Verified Repo Facts

| Fact | Evidence |
| :--- | :--- |
| Engram anchors are **`{3, 7, 11, 15, 19, 23, 27, 31}`** (8) — exactly the full-attention layers minus L35 | `anchor_indices = range(loop_start + stride - 1, min(n_layers - 4, 32), stride)` with `start=4, stride=4` + anchor 0 at L3 ([`iter_engram_anchors`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2395)) |
| Each anchor is applied **after the whole decoder block** (attention + MLP) | Prelude/coda: `h = _apply_layer(...)` then `_apply_engram_residual` ([`forward_hidden`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2926-L3048)); loop: `s_out = ctrl(...)` then `s_out[0] + eng_delta` |
| Inside the loop, pass `p>0` Engram output is scaled by `tanh(ffn_extra_gain)`; in the no-mHC path only pass 0 adds Engram | `forward_hidden` loop branches |
| With `sublayer_mhc=True`, [`TwoStreamMHC.forward`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2000) calls `layer.input_layernorm`, `layer.self_attn`, `layer.mlp` **directly** — a forward hook on the decoder layer does **not** fire in mHC mode | `attn_out, _ = layer.self_attn(hidden_states=normed_a, ...)` |
| Default loop: `start=4, end=28, passes=2, P3 12..24, enable_p3=False` (72L only when `enable_p3=True`) | [`LoopV03Settings`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L193-L228) |
| Public forward API: `forward(input_ids, labels=None, attention_mask=None, return_logits=False)`; loop config via `set_loop(start, end, passes, use_mhc, use_engram, enable_p3, ...)` | [`set_loop`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2837-L2855) |

The good news: the alignment is **better than claimed**. Every anchor sits on exactly one full-attention layer (1:1), so a per-layer write target exists for 8 of the 9 global layers. L35 has no anchor (see design option in `01`).

---

## 3. Review Findings — Stage 2 Sleep Compiler (`47a7cc6`)

Severity: **B** = blocker (crashes / silently does nothing), **M** = major correctness or claim problem, **m** = minor.

### 3.1 Blockers (code cannot produce a result today)

| ID | Location | Problem | Fix (Phase 0) |
| :--- | :--- | :--- | :--- |
| B1 | [`run_sleep_time_memory_compiler`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2661) | `self._layers()` does not exist → `AttributeError` | Use `self.base.model.layers` |
| B2 | [same, L2699](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2699) | `self.forward(ids, passes=1, use_mhc=False, use_engram=False)` — `forward` has no such kwargs → `TypeError` | Save loop config → `set_loop(start, end, 1, use_mhc=False, use_engram=False)` → `forward_hidden(ids)` → restore |
| B3 | [same, L2717](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2717) | Looks up `o_proj`; Spark uses `out_proj` → every layer skipped → **0 clusters compiled**, silently | Use `out_proj`; include `g_proj` gate (see M2) |
| B4 | [`validate_vertical_sleep_memory_compiler`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L3920) | Calls undefined `evaluate_multi_domain_ppl` and `format_spark_chat` → `NameError`. Because it was wired into `run_continue_from_hub_engram_expansion`, the **whole Colab pipeline would crash after Stage C** | **Done in `b841a3e`**: Stage D is opt-in (`run_sleep_compiler_validation=False`) and wrapped in `try/except`. Phase 0 rewires it to [`evaluate_engram_perplexity_delta`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L1836) / [`encode_spark_chat_pair`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L1587) |
| B5 | capture hooks | Decoder-layer pre-hook is bypassed in sublayer-mHC mode (see §2) | Hook `layer.self_attn` (forward pre-hook with kwargs captures the **post-LN** input) |

### 3.2 Major correctness / claim issues

| ID | Problem | Why it matters | Corrected approach |
| :--- | :--- | :--- | :--- |
| M1 | **Key/value mismatch.** The write key is the cluster's *own* hidden state `h_b` (attention-weighted mean of the cluster tokens). | An attention cluster contributes to the residual of the **querying** token, not to its own tokens. At recall time the Engram sees the *future question's* hidden state, which is unrelated to `h_b`. ROME/MEMIT likewise key on the token whose residual must change. | Keys = query-side states of tokens that actually read the episode (recorded during wake) **plus** self-generated probe questions (Cartridges-style self-study). See `01 §C3–C4`. |
| M2 | **Target residual is not the real attention output.** `alpha_j` is a softmax over 64-d *index* scores with a learned `tau`, not `softmax(q·k/√256)` over all keys; the head-wise sigmoid gate is missing; the bucket's share of total attention mass is dropped. | The compiled vector does not match what the layer actually wrote. | Use **ablation targets**: run the block with and without the episode's KV and take the difference (exact by construction, includes gate, softmax renormalisation and the MLP). |
| M3 | **Placement mismatch.** The anchor writes *after* the MLP of layer ℓ, but `y_b` is the *attention* output that the same layer's MLP would have transformed. | Even a perfect `y_b` injected after the MLP is not equivalent. | Target the **post-block** residual difference `δ = Block_ℓ(h | KV⁺) − Block_ℓ(h | KV⁻)`. |
| M4 | **Null space protects the wrong set.** The basis contains only previously compiled cluster activations. General pre-trained behaviour of that expert is unprotected. Also `cat_dirs[:, -(R-16):]` silently **drops the oldest protected directions** once the basis is full. | AlphaEdit's guarantee comes from projecting out `K0` built from ~100K general-corpus keys (Wikipedia triplets), not from past edits alone. Dropping directions breaks the "zero-forgetting" claim without warning. | Build `K0Kᵀ0` from a general corpus (plus past edits); refuse/shard when capacity is exhausted instead of truncating. |
| M5 | **Pseudo-inverse bug** ([L1920](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L1920)). Code computes `Wᵀ(WᵀW+λI)⁻¹r`. | Correct min-norm ridge solution of `W δc = r` is `Wᵀ(WWᵀ+λI)⁻¹r` (= `(WᵀW+λI)⁻¹Wᵀr`). They differ unless `W` is normal. | Fix formula; unit-test against `torch.linalg.lstsq`. |
| M6 | If `project.weight` is ~0, it is overwritten with `0.25·I`. | `project` also carries the lexical and shared-expert paths, so this switches on the **entire** Engram output for every token. Not null-space safe. | Never touch `project`; write into a dedicated side memory (`01 §C5`). |
| M7 | "Ward clustering" is actually contiguous fixed spans of ≤ 8 tokens. | Buckets are positional, not semantic; the docstrings and `AGENTS.md` overclaim. | Implement real within-block Ward/mutual-NN merges (LSA paper) on the **pass-through slice**. |
| M8 | Index projections operate on post-RoPE keys including rotated dims. | Rotated dims inject position into the "semantic" index. | Use `k[..., 64:256]` (position-free) for the index; D-LSA distillation (`02 Phase 2`). |
| M9 | Validation design. Drift check compares parameter **norms** (weak); "held-out PPL" on generic text doesn't test **recall**; episodes are ~200 tokens (< 512 SWA window), so "evictable KV" is meaningless; `calibrate_lsa_pca` overwrites the long-context index from ~200 tokens. | The suite cannot support the paper's claims. | Hash/max-abs-diff drift check; recall-after-eviction benchmark (EFRE, `03 §D1`); episodes ≥ 4K tokens; never refit the index in validation. |

### 3.3 Minor

| ID | Note |
| :--- | :--- |
| m1 | `self.w_q_idx.float()(x)` calls `nn.Module.float()`, which **mutates** the module to fp32 in place ([L840-841](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L840-L841)). Use `F.linear(x, self.w_q_idx.weight.float())`. |
| m2 | Lexical injection writes into hashed Tier-1A rows shared by colliding n-grams; not null-space protected. Keep it disabled by default (it is). |
| m3 | `compile_latency_ms` includes the forward trace; report trace / extract / solve separately. |
| m4 | Unverified internal claims in `AGENTS.md` ("< 200 ms/anchor", "< 1% relative L2 error", "‖·‖∞ ~ 1e-7") must be measured before they appear in a paper. |

### 3.4 What is sound and should be kept

- **Closed-form dual ridge with projection** (`ΔW = A⊥ᵀ(A⊥A⊥ᵀ + αI)⁻¹ΔZ`) is mathematically correct *given the right basis*; cost is `O(M²·d)` per expert and needs no backprop.
- **Additive, mutually orthogonal updates** give a property we should advertise: an individual memory can be **exactly removed** (subtract its stored `ΔW`) — useful for privacy/unlearning.
- **fp32 master copy** of edited weights to avoid bf16 drift across sleep cycles.
- **Snapshot + Pareto rollback** via [`snapshot_engram_bank`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2511) / [`hot_swap_engram_bank`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py#L2522).
- **Layer alignment**: 8 anchors ↔ 8 global layers, 1:1.

---

## 4. Cross-Validation of Cited Research

| Work | Verified ID / venue | Verified claim we rely on | Status |
| :--- | :--- | :--- | :--- |
| Latent Sparse Attention (Kubicek, …, Hoefler, Whittington; Prefrontal AI & ETH Zurich) | arXiv:2610.42147 (30 Oct 2026) · `Prefrontal-AI/latent-sparse-attention` | Keys clustered hierarchically (Ward) in a learned latent space; queries score clusters; **D-LSA** retrofits frozen pretrained models via distilled index projections; Flat/Tree variants; 10M-token results on Qwen-3.8-27B / GLM-5.3-Flash | ✅ Title/authors **corrected** in `REFERENCES.md` (previous entry was wrong) |
| AlphaEdit (Fang et al.) | arXiv:2410.02355 · ICLR 2025 Outstanding Paper | Project ΔW onto null space of `K0K0ᵀ`; `K0` from **100K Wikipedia (s,r,o) triplets**; eigenvalue threshold | ✅ — our code currently deviates (M4) |
| MEMIT / ROME (Meng et al.) | arXiv:2210.07229 / 2202.05262 | FFN as key-value memory; key = last subject token; covariance `C0` from general text; multi-layer spreading | ✅ |
| WISE (Wang et al.) | arXiv:2405.14768 · NeurIPS 2024 | Side memory + router; knowledge sharding into separate subspaces; "impossible triangle" reliability/generalisation/locality | ✅ basis for our side-memory design |
| GRACE (Hartvigsen et al.) | arXiv:2211.11031 · NeurIPS 2023 | Discrete key-value codebook adaptor with deferral radius; frozen base | ✅ |
| UltraEdit | arXiv:2505.14679 | Training-free lifelong editing; scales to very large edit counts; closed-form | ✅ baseline / related work |
| Sparse Memory Finetuning (Lin, …, Oğuz; Meta) | arXiv:2510.15103 | Update only memory slots ranked by **TF-IDF vs pretraining usage**; NQ F1 drop 11% vs 89% (full FT) / 71% (LoRA) | ✅ basis for deep-sleep consolidation |
| Memory Layers at Scale (Berges et al., Meta) | arXiv:2412.09764 | Sparse key-value memory layers scale to 128B memory params | ✅ |
| Engram (DeepSeek) | arXiv:2601.07372 | Deterministic hashed n-gram lookup, early-layer placement, host-memory prefetch; long-context NIAH gains | ✅ |
| Titans (Behrouz, Zhong, Mirrokni) | arXiv:2501.00663 | Surprise (gradient) + momentum + forgetting gate for test-time memory | ✅ |
| Nested Learning (Behrouz, Razaviyayn, Zhong, Mirrokni) | arXiv:2512.24695 · NeurIPS 2025 | Multi-level, multi-frequency optimisation; continuum memory system; HOPE | ✅ |
| Sleep-time Compute (Lin, Snell, …, Gonzalez) | arXiv:2504.13171 | Offline pre-computation over context during idle time | ✅ |
| Cartridges (Eyuboglu, …, Ré) | arXiv:2506.06266 | Train a small KV cache offline with **self-study** (synthetic conversations about the corpus) → generalises better than next-token on the doc | ✅ basis for probe generation |
| Doc-to-LoRA (Charakorn, Cetin, Uesaka, Lange; Sakana) | arXiv:2602.15902 | Hypernetwork maps a document to LoRA in one pass; NIAH beyond native context | ✅ Authors **corrected** |
| Learning by Distilling Context (Snell, Klein, Zhong) | arXiv:2209.15189 | Internalise context gains into weights by matching with-context teacher | ✅ basis for residual-space distillation |
| SEAL (Zweiger et al., MIT) | arXiv:2506.10943 | Model writes its own "self-edits" (notes/QA) then fine-tunes on them; RL outer loop | ✅ |
| Gated Attention (Qiu et al., Qwen) | arXiv:2505.06708 · NeurIPS 2025 | Head-wise sigmoid gate after SDPA; removes attention sinks | ✅ explains Spark's `g_proj` |
| mHC (Xie et al., DeepSeek) | arXiv:2512.24880 | Doubly-stochastic (Birkhoff) residual mixing via Sinkhorn | ✅ |
| Recurrent depth (Geiping et al.) | arXiv:2502.05171 | Latent recurrent reasoning scales test-time compute | ✅ |
| Retrofitted recurrence (McLeish et al.) | arXiv:2511.07384 | Convert pretrained LMs to depth-recurrent with a **recurrence curriculum** | ✅ directly relevant to our loop |
| Ouro / LoopLM (Zhu et al.) | arXiv:2510.25741 | Looped LM pre-training; entropy-regularised learned exit | ✅ |
| Mixture-of-Recursions (Bae et al.) | arXiv:2507.10524 | Token-level adaptive recursion depth with routers | ✅ |
| LongMemEval (Wu et al.) | arXiv:2410.10813 · ICLR 2025 | 500 Qs, 5 abilities (extraction, multi-session, temporal, updates, abstention) | ✅ eval |
| RULER (Hsieh et al.) / BABILong (Kuratov et al.) | arXiv:2404.06654 / 2406.10149 | Effective vs claimed context; reasoning-in-a-haystack | ✅ eval |
| Complementary Learning Systems (McClelland et al. 1995; Kumaran, Hassabis, McClelland 2016, *TiCS*) | Psych. Review 1995; TiCS 2016 | Fast hippocampal + slow neocortical learning; replay-based consolidation; schema-consistent fast cortical learning | ✅ framing |

---

## 5. Claims Policy for the Paper

Use these phrasings until the matching experiment in `03` has passed.

| Do **not** write | Write instead |
| :--- | :--- |
| "100% position-invariant NoPE clustering" | "clustering on the 192-dim RoPE-free slice of partially-rotary global layers" |
| "exact layer-ℓ residual equivalence" | "per-layer residual matching with measured relative error ε_ℓ" |
| "mathematically zero forgetting" | "updates orthogonal to a protected key subspace (general-corpus covariance + prior memories); measured locality Δ" |
| "frees the context window" | "reduces resident KV by X% at recall accuracy Y% of in-context" |
| "< 200 ms per anchor" | report measured trace / extract / solve latency with hardware |
| "the loop and memory are complementary" | only after E7/E8 (`03` Tier E) pass; until then: "we hypothesise (H-L2/H-L3) and test…" |
| "frozen backbone" (with LB-blend) | "the base network is frozen during all memory operations" (backbone carries a fixed task-vector blend; see `01 C0`) |
