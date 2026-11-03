# 03 — Training & Evaluation Plan (Spark-X2.6-4B Backbone + Looped Transformer)

> Goal: every claim in the paper maps to a pre-registered experiment with a baseline, a metric, a confidence interval and a pass/fail threshold.

---

## 1. Model Configurations Under Test

| ID | Backbone weights | Depth | Engram | ESM | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B0** | pristine `XHToken/Spark-X2.6-4B` | 36L | off | off | upstream reference (always reported) |
| **LB0** | LB checkpoint (blend or pristine, see `01 C0`) | **72L loop + mHC** | GL2-gated | off | **base, no memory** |
| **LB1** | LB checkpoint | 72L loop + mHC | GL2-gated | **on (first visit)** | **main result** |
| LB1-all | LB checkpoint | 72L loop + mHC | GL2-gated | on (all visits) | ESM visit-policy ablation |
| LB1-noE | LB checkpoint | 72L loop + mHC | off | on | Engram × memory interaction |
| A36-0 / A36-1 | same weights as LB | 36L (loop off) | off | off / on | **loop ablation** (H-L1, H-L2, H-L3) |
| A-pristine | pristine backbone + LB controllers | 72L loop + mHC | GL2-gated | on | shows memory results don't depend on the backbone blend |

Main claims are made on **LB0 → LB1**. The loop's contribution is **(LB1 − LB0) − (A36-1 − A36-0)** for memory, and LB0 vs A36-0 for reasoning. B0 is reported next to every number.

---

## 2. Statistical Protocol

- **Sample sizes matter.** IFEval prompt-level accuracy at `p≈0.52`: 95% CI half-width is **±13.9 pts at n=50** but **±4.2 pts at n=541**. The 50-prompt comparisons in recent logs (52% vs 52%) cannot distinguish configurations. Use the **full 541** for any decision.
- **Paired tests:** McNemar for paired accuracy (same prompts, two configs); paired bootstrap (10K resamples) for PPL/CE and recovery ratio.
- **Seeds:** 3 seeds for anything stochastic (probe generation, training); report mean ± std.
- **Compute accounting:** report tokens processed, FLOPs (forward-equivalents), wall-clock, peak VRAM and host RAM for every row.
- **Pre-registration:** thresholds below are fixed before running; changes are logged in `runs/index.csv` with a reason.

---

## 3. Evaluation Tiers

### Tier A — Parity & Safety (run on every commit that touches model code)
| Test | Pass condition |
| :--- | :--- |
| A1 Backbone fingerprint | SHA-256 identical to reference; max-abs-diff = 0 |
| A2 36L parity vs upstream `Spark2_5ForCausalLM` | max logit diff < 0.025 on 8 fixed prompts (existing harness) |
| A3 Loop identity at step 0 | `zero_gate_diff == 0.0` (existing) |
| A4 Empty ESM | bit-identical logits with ESM enabled but empty |
| A5 D-LSA exact path | seq ≤ 32K ⇒ outputs identical to dense attention |

### Tier B — Capability Retention (locality of everything we add)
| Benchmark | Size | Metric | Threshold vs B0 (or vs pre-write state) |
| :--- | :--- | :--- | :--- |
| Held-out multi-domain CE/PPL (existing [`evaluate_engram_perplexity_delta`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L1836)) | fixed 64 seqs × 1536 | CE | ΔCE ≤ +0.5% after memory writes |
| IFEval (existing [`run_ifeval_and_ifbench_suite`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_benchmarks.py#L765)) | **541** | prompt-strict / loose | within 95% CI |
| GSM8K | 1319 | EM | within 95% CI |
| HumanEval (existing) | 164 | pass@1 | within 95% CI |
| MMLU-Pro (subset) | 2000 | acc | within 95% CI |

### Tier C — Long-Context Working Attention (D-LSA, cold KV)
| Benchmark | Lengths | Metric | Threshold |
| :--- | :--- | :--- | :--- |
| Attention-mass recall@k (internal) | 32K/64K/128K | recall | ≥ 0.90 @ 64K |
| RULER (NIAH single/multi-key, multi-value, variable tracking, aggregation) | 8K–128K (256K if host RAM allows) | acc | D-LSA within −2 pts of exact at ≤ 128K |
| BABILong (qa1–qa5) | 8K–128K | acc | report curve; no threshold (diagnostic) |
| Latency / memory | 32K–1M | TTFT, tok/s, GPU/host bytes | report |

### Tier D — Vertical Memory (the core contribution)

#### D1. EFRE — Episodic Fact Recall after Eviction (new, contamination-free)
**Construction.** Programmatically generate 200 fictional entities (people, projects, places, devices) with 6–10 attributes each. Render them into narrative episodes (4K–32K tokens) mixed with distractor prose. Every fact is fictional, so the backbone cannot already know it.

**Question types** (aligned with LongMemEval's five abilities, arXiv:2410.10813):
1. direct extraction; 2. paraphrased question; 3. multi-hop within one episode; 4. multi-hop **across** episodes; 5. knowledge update (a fact changes in a later episode, ask for the latest); 6. abstention (entity/attribute never mentioned).

**Splits.** EFRE-small (20 episodes × 4K tokens, ~400 Qs) for development; EFRE-stream (one 200K-token diary, ~1000 Qs) for streaming/sleep; EFRE-large (2M tokens) for the cold-KV + memory scaling plot.

**Protocol.** Feed an episode → sleep (compile) → **evict the episode from KV** → ask questions with an empty or unrelated context.

**Metrics.**
- **Recovery ratio** `R = (lp_mem − lp_none) / (lp_ctx − lp_none)` on gold-answer log-prob (1.0 = as good as having the episode in context).
- EM/F1 of greedy answers (thinking budget fixed).
- Retention curve: R of episode *i* after *N* subsequent sleep cycles (AUC over N = 1…100).
- Locality: Tier B deltas + canary gate-open rate.
- Capacity: protected dims used / available, shards opened.
- Cost: compile wall-clock per 1K span tokens (trace / probe generation / solve separately); bytes added.

**Baselines.**
| Baseline | What it tests |
| :--- | :--- |
| No memory (lower bound) | — |
| Full episode in context (upper bound) | — |
| RAG through `Layer 0`: BM25/embedding top-k chunks prepended | the market approach we argue against (context cost vs accuracy) |
| Cold-KV + D-LSA only (no compilation) | horizontal-only memory at equal host RAM |
| Full-FT of Engram params on the episode (next-token) | naive weight memory |
| SMF-style sparse finetune (arXiv:2510.15103) | strongest weight-update baseline |
| AlphaEdit on backbone MLP (arXiv:2410.02355) | classic locate-and-edit (breaks "frozen backbone"; reported as reference) |
| **Ours: ESM + LRCD + self-study probes** | main |
| Ours ablations (keys / store / null-space / layers grid from `02 Phase 4`) | which ingredients matter |

**Pre-registered success thresholds (EFRE-small, B1 vs B0):** `R ≥ 0.6` on direct + paraphrase; `R ≥ 0.4` on cross-episode multi-hop; retention AUC ≥ 0.8× single-episode R after 50 cycles; Tier B within thresholds; compile ≤ 60 s per 4K-token span on one A100-80GB.

#### D2. LongMemEval-S (external validity)
- 500 questions, ~115K-token histories. Stream sessions; sleep after each session; answer with **only the current session in context + memory**.
- Compare against B0 with the full 115K history in context (feasible: 115K × 36 KiB ≈ 4 GiB) and against RAG.
- Report per-ability accuracy (extraction, multi-session, temporal, knowledge update, abstention).

#### D3. Sequential editing (comparability with editing literature)
- ZsRE and CounterFact, 1K / 3K sequential edits, efficacy / generalisation / specificity; compare with AlphaEdit, WISE, UltraEdit numbers reported for similar-size models (note model mismatch explicitly).

#### D4. Forgetting & privacy
- `forget(memory_id)` → logits on that memory's probes return to pre-write values (max-abs-diff < 1e-3) and R drops to ≤ 0.05; other memories' R unchanged (paired test).

### Tier E — Loop Base (Track L) and Loop × Memory
| Experiment | Metric | Threshold |
| :--- | :--- | :--- |
| E1 Pass-wise logit lens | CE after pass 1 / 2 / 2+P3 | monotone non-increasing |
| E2 Compute-matched comparison | A36-0 vs LB0 at equal training tokens: held-out CE, GSM8K, IFEval-541 | LB ≥ 36L within CI (target GL3) |
| E3 Test-time depth scaling | accuracy vs passes (1, 2, 2+P3, 3) on GSM8K / MATH-500 | positive slope |
| E4 Adaptive exit | FLOPs vs accuracy Pareto | ≥ 25% saving at ≤ 0.5 pt loss |
| E5 Engram per-anchor ablation | ΔCE when each anchor disabled | identify harmful anchors |
| E6 KV sharing across recursions | RULER @ 64K, KV bytes | ≤ 1 pt loss at 2× saving |
| E7 **Storage vs manipulation (H-L2)** | EFRE R on single-hop vs **multi-hop** probes (recalled fact ∘ second fact / recalled fact ∘ in-context fact), LB1 vs A36-1 | multi-hop gain (LB − 36L) > single-hop gain, paired bootstrap p < 0.05 |
| E8 **Recall robustness to depth (H-L3)** | EFRE R and GSM8K vs passes ∈ {1, 2, 2+P3, 3}, ESM first-visit vs all-visit | R flat (±0.03) across passes with first-visit ESM, while GSM8K rises |
| E9 mHC health | `extra_gain` trajectories, Sinkhorn mix entropy, per-pass ΔCE | no collapse (gain ≥ 0.02) throughout T3 |

---

## 4. Training Runs

All runs keep `model.base` frozen. A100/H100-80GB estimates are rough and should be replaced by measurements.

| Run | Trains | Params | Data | Steps / tokens | Objective | Est. GPU-h | Stop / gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **T0** D-LSA index | `w_q_idx`, `w_k_idx` (9 layers) | ≈ 1.8M | long docs ≥ 32K (books, long code, arXiv) | 2K steps × 32K | `KL(p_true ‖ s_idx)` + recall hinge | 3–6 | G2 |
| **T1** ESM stats | none (forward only) | — | 3M tokens general mix (FineWeb-Edu-like + your SFT mix) | — | accumulate Σ_gen, gate scores | 0.5–1 | eigen-energy curve saved |
| **T2** ESM read-path calibration *(optional)* | `τ_s`, `T`, optionally `W_in` | ≤ 5M | synthetic EFRE-train episodes | 1–2K steps | **differentiate through the closed-form write** (dual ridge is differentiable): maximise held-out probe recall after write, penalise canary gate opening | 4–8 | R(EFRE-dev) improves ≥ 0.05 |
| **T3** Loop Base strengthening (Track L) | 24 `TwoStreamMHC` (gains via softplus floor), pass adapters | ≈ 10M | healing: general text (KL to 36L teacher) → existing Phase-4 mix | healing ~50–100M tokens, then stage 1→2→2+P3, ~0.5–1B tokens total | CE + z-loss + KL(healing); stochastic pass count | 20–60 | stable checkpoint → unblocks Phase 4; GL3 target |
| **T4** Engram repair (Track L) | Engram gates, harmful-anchor gating, routers | ≤ 50M active | existing mix | 1–3K steps | CE; KL to no-Engram on general text | 5–15 | GL2 |
| **T5** Deep-sleep consolidation | top-t experts' `expert_down` per anchor | sparse | ledger replay + general text 1:1 | 300 steps / sleep | CE(replay) + KL(general) | ≤ 0.5 per sleep | G6 |

Notes:
- **T2 is a real novelty lever.** Because the ESM write is closed form, the whole "compile then recall" pipeline is differentiable w.r.t. the read-path parameters. You can meta-learn *how to be written to* without a hypernetwork, a lightweight analogue of Doc-to-LoRA (arXiv:2602.15902).
- **T3 follows McLeish et al. (arXiv:2511.07384):** curriculum over recurrence depth is what makes retrofitted recurrence work on pretrained models; abrupt full-depth training is the likely cause of the collapsed mHC gains.

---

## 5. Compute Planning

| Workload | Dominant cost | Home-lab feasibility |
| :--- | :--- | :--- |
| Light sleep per 4K span | teacher pass (span + probes) + 9 student passes over probes + 9 small solves | seconds–tens of seconds on one 24–80 GB GPU |
| Σ_gen stats | 3M-token forward | < 1 h |
| D-LSA distillation | 32K-token forwards with exact attention (no backbone backward) | hours on A100; feasible on 24 GB with gradient-free teacher caching |
| 1M-token context (36L) | 36 GiB KV | host-RAM cold tier + GPU hot window |
| 1M-token context (LB, 72L) | 72 GiB KV naive; ≈ 36 GiB with L5 KV sharing; ≈ 18 GiB with L5 + FP8 cold pages | **L5 is required** because LB is the base |

---

## 6. Claim → Evidence Map (paper skeleton)

| # | Claim | Evidence | Figure / table |
| :--- | :--- | :--- | :--- |
| 1 | Compiling episodes into mid-layer side memory recovers most of in-context recall with zero context tokens | D1 R, EM vs baselines | Table 1, Fig. 2 (R vs span length) |
| 2 | Recall persists across many sleep cycles | D1 retention AUC, D2 LongMemEval | Fig. 3 |
| 3 | General capability preserved | Tier B deltas, canary gate rate | Table 2 |
| 4 | Ingredients matter: query-side keys, LRCD targets, general-corpus null space | Phase 4 ablation grid | Table 3 |
| 5 | Memories can be removed exactly | D4 | Fig. 4 |
| 6 | Horizontal tier scales to 1–2M logical tokens on a home lab | Tier C + EFRE-large | Fig. 5 (RAM / latency vs length) |
| 7 | The retrofitted Loop Base improves reasoning at matched compute | Tier E1–E3 | Table 4 (reported as measured, including partial results) |
| 8 | **Loop and vertical memory are complementary: the loop manipulates what memory stores** | E7 (H-L2), E8 (H-L3), loop-contribution term in §1 | Fig. 6 |

---

## 7. Run Hygiene Checklist (copy into every Colab run)

- [ ] Backbone fingerprint logged before and after
- [ ] Config + seeds saved to `runs/<id>/config.json`
- [ ] Full-size eval sets (IFEval 541) for any decision
- [ ] Baseline B0 **and** LB0 evaluated in the same session/hardware
- [ ] mHC gains logged (collapse alert armed)
- [ ] Peak VRAM / host RAM / wall-clock recorded
- [ ] Metrics appended to `runs/index.csv`; tracker in `README.md` updated
- [ ] Commit + push code state that produced the numbers (hash in `config.json`)
