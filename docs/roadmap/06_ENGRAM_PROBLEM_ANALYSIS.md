# 06 — MoE-Engram Problem Analysis, Hypotheses and Decision Tree

> **Status:** analysis + test plan. Companion to [05 — SI-SRSI v2](05_SI_SRSI_V2_DESIGN.md) and to the 20 h long run (`sparkx_long_run.py`).
> **Claims policy:** `00 §5`. Items marked *hypothesis* have **not** been measured.

---

## 1. Verified symptoms (all leak-free, Colab, 2026-11-03/04)

| # | Observation | Source |
| :-- | :-- | :-- |
| S1 | On 5 hand-written held-out texts (1 330 tokens): Engram OFF CE **1.5090**, FULL **1.5750** (+0.066; PPL 4.52 → 4.83). Engram hurts 5/5 texts; CE is monotone worse in Engram scale (best scale 0). | Cell 9 |
| S2 | Leave-one-anchor-out: removing **L31 alone** → CE 1.4982 (< OFF). Other anchors within ±0.005. | Cell 9 |
| S3 | Cross-validated gating of `{7, 11, 23, 31}`: OFF 1.5090 / FULL 1.5750 / GATED 1.4901 (−0.0188 vs OFF). Both folds chose the same anchors. Low power (5 texts, 2 folds). | Cell 10 |
| S4 | Lexical hypothesis refuted: `LEX_ONLY` gain ≈ 0, `SEM_ONLY` ≈ `FULL`. Tier-1 tables are zero-init and trained at LR ≤ 1.8e-5. | Cell 8 |
| S5 | Marginal Engram gain on the *trained* prompts is only 0.013–0.037 nats; "transfer 3–9×" was a teacher-forcing artifact. | Cell 8/9 |
| S6 | Training regime that produced these weights: ~5.08 M packed tokens seen ≈ **3.9 epochs**, train CE 0.78 → 0.35 while held-out CE ≈ 1.5; no validation split; "best" chosen on leaked texts. | Cell 4 audit |

Not yet measured: the loop's own benefit on *true* held-out text (the often-quoted 36L 7.14 → 72L 1.59 PPL is on leaked texts and must not be cited); any downstream capability benchmark.

---

## 2. Hypotheses, predictions, tests

| ID | Hypothesis | Prediction if true | Test (where) | Fix if true |
| :-- | :-- | :-- | :-- | :-- |
| **H1** | **Data starvation / memorisation.** ~4 B Engram parameters were fit to ~5 M tokens for ~4 epochs, so they store training text instead of generalisable structure. | Held-out `ce_on − ce_off` falls with *unique* tokens; train–val gap shrinks; per-pillar deltas move together. | 20 h long run `val_history` (`ce_on`, `ce_off`, per pillar every 40 steps). | Keep scaling unique data; dedup; continue. |
| **H2** | **Harmful anchors**, esp. `L31` (coda anchor immediately before the logits, so nothing downstream can correct it). | Same anchors show negative `removal_delta_ce` in ≥ 3 consecutive audits. | Audit every 4th eval in the long run (`removal_delta_ce`, `harmful_anchors`). | Post-hoc CV gating on the 64-doc set (never gain = −30 *before* training: dead gradients); optionally smaller init gain for the last anchor. |
| **H3** | **Scale/gain miscalibration.** The learned function has net harm on unseen text at the trained scale. | CE-vs-scale sweep still monotone worse after the long run. | `engram_utility_report` in notebook Cell 4 (post-run). | Held-out-calibrated global scale; per-anchor scale learned on val; scale ∈ [0, 1] stored in state dict. |
| **H4** | **Objective mismatch.** CE on instruction trajectories mostly teaches response style; Engram memorises long-tail response text (S5). | `ce_off` improves but `ce_on − ce_off` stays ≥ 0; SI4 gap-write recall low. | Long run + SI4 (doc 05 §5). | Engram dropout (random anchor ablation per step); weight decay on experts; surprise-gated writes; train Engram on *knowledge-gap* data rather than generic trajectories. |
| **H5** | **Routing collapse / under-use** of the 32 experts. | `router_load_cv` high, few experts carry mass. | `get_engram_routing_telemetry` already logged in Stage logs; add to long-run eval. | Balance-step tuning (`ENG_BALANCE_STEP_SIZE`), z-loss weight. |
| **H6** | **Frozen consumer.** The frozen backbone never learned to consume Engram residuals; only 24 mHC controllers co-adapt. | Gains saturate at a floor regardless of data. | Compare mHC-only vs Engram-only vs both arms (SI3). | *Decision for owner, not assumed:* a tiny trainable adapter on the post-anchor path (violates the pure frozen-backbone rule). |

H1 and H4 are the leading explanations given S4–S6; none is proven.

---

## 3. What the 20 h long run can and cannot do

**Can:** test H1 directly (unique tokens, leak-free validation, per-pillar), expose H2/H3/H5 trends every 40 steps, select the best held-out checkpoint and roll back plateaus. The baseline is the **current Hub model**, so `val_history[0]` is the starting point and every later row is a paired improvement/regression against it.

**Cannot:** guarantee an Engram benefit. ~85 M tokens against ~4 B parameters is still data-starved; if H4/H6 dominate, more generic data will not help.

**Caveat:** the starting weights are already overfit (S6); recovery may be slower than training from reset. A fresh-start arm costs another multi-hour run and is not planned unless the long run shows H1 is partially true.

### 3b. Live Mid-Run Telemetry (2026-11-04, Steps `0 → 355 / 743`, `41.0M` unique tokens, `9.18h` elapsed)

Evaluated on the frozen 55-doc (`60,059` supervised tokens) doc-hash held-out split (`sha1(prompt)%25==0`, never trained on):

| Step | Unique Tokens | `CE_on` | `CE_off` | `delta (on - off)` | vs Baseline `CE_on` (`0.7790`) | Leave-One-Anchor-Out (`CE_w/o - CE_on`, `+` = helps, `-` = hurts) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** (Hub start) | `0.0M` | `0.7790` | `0.7896` | `-0.0106` | `0.0000` | — |
| **40** | `4.6M` | `0.5838` | `0.6239` | `-0.0402` | `-0.1952` | — |
| **80** | `9.3M` | `0.5809` | `0.6301` | `-0.0492` | `-0.1981` | — |
| **120** | `13.9M` | `0.5691` | `0.6283` | `-0.0592` | `-0.2098` | — |
| **160** | `18.6M` | `0.5795` | `0.6332` | `-0.0537` | `-0.1995` | `L3:+0.0136, L7:+0.0058, L11:+0.0157, L15:+0.0118, L19:+0.0127, L23:+0.0078, L27:+0.0121,` **`L31:-0.0099`** |
| **200** | `23.2M` | `0.5667` | `0.6336` | `-0.0669` | `-0.2123` | — |
| **240** | `27.8M` | `0.5729` | `0.6382` | `-0.0653` | `-0.2061` | — |
| **280** | `32.4M` | `0.5668` | `0.6386` | `-0.0718` | `-0.2122` | — |
| **320** (**Best**) | `37.1M` | **`0.5611`** | `0.6408` | **`-0.0797`** | **`-0.2179`** | `L3:+0.0126, L7:+0.0059, L11:+0.0164, L15:+0.0112, L19:+0.0127, L23:+0.0068, L27:+0.0114,` **`L31:+0.0161`** (**0/8 harmful!**) |

**Honest reading at Step 355 (`41.0M` tokens, ~48% of run):**
1. **`H1` (data starvation) & `H2` (`L31` recovery) strongly supported on held-out in-distribution trajectories**:
   - Replacing 3.9-epoch repetition on 5M tokens with zero-epoch streaming over `37.1M` unique tokens drove held-out `CE_on` from `0.7790 -> 0.5611` (`-0.2179` nats) and widened `CE_on - CE_off` from `-0.0106` to **`-0.0797` nats**.
   - At Step 160 (`18.6M` tokens), `L31` was still the lone harmful anchor (`-0.0099`); by Step 320 (`37.1M` tokens), **`L31` flipped to `+0.0161`**, making **all 8 anchors `{3,7,11,15,19,23,27,31}` net-helpful (`0/8` harmful)** on the 55-doc held-out set.
   - At Step 320, all 7 pillars show `CE_on < CE_off` (`web_3d_ui: 0.406/0.556 [-0.150]`, `fable_opus_astra_cot: 0.617/0.719 [-0.102]`, `science_knowledge_engram: 0.798/0.881 [-0.083]`, `code_swe: 0.555/0.625 [-0.070]`, `math_stem: 0.430/0.463 [-0.033]`, `tool_agent: 0.202/0.210 [-0.008]`, `literacy_dialogue: 1.623/1.628 [-0.005]`).
2. **Co-adaptation caveat (`CE_off` drift)**:
   - Between Step 40 and Step 320, `CE_off` drifted slightly upward (`0.6239 -> 0.6408`, `+0.0169` nats) because the 24 `TwoStreamMHC` controllers are trained exclusively with `use_engram=True` and co-adapt to expect the Engram residuals. Even against the **best-ever `CE_off` (`0.6239` at Step 40)**, `CE_on` at Step 320 (`0.5611`) is **`-0.0628` nats lower**.
3. **What remains unverified until post-run Cell 4 / Cell 6 (`BIE`)**:
   - `val_frozen` tests doc-hash held-out generalization across the 7 training pillars, whereas `S1` (`OFF 1.5090 -> FULL 1.5750`) was measured on **out-of-distribution raw prose** (`TRUE_HELDOUT_TEXTS` without chat templates). Post-run Cell 4 & Cell 6 (`BIE`) will test whether `L31`'s recovery also resolves `S1` on raw prose or whether out-of-distribution prose still requires `BIE-3` selective gating (`wise_scope_gated` / `deepseek_qk_gated`).

---

## 4. Decision tree after the long run

```mermaid
flowchart TD
    R["Long run finished<br/>val_history, audits, scale sweep"] --> Q1{"ce_on - ce_off<br/>trend vs tokens"}
    Q1 -- "falls, ends <= -0.01<br/>(CI excludes 0)" --> A["H1 supported:<br/>Engram helps held-out CE.<br/>Scale data further; run SI-SRSI v2 (SI0-SI3)"]
    Q1 -- "flat near 0" --> B["Engram is CE-neutral on generic text.<br/>Run SI4 gap-write: is it a fact store?"]
    Q1 -- "stays >= +0.02" --> C{"harmful anchors<br/>persistent?"}
    C -- yes --> D["Apply CV anchor gating + scale calibration (H2/H3).<br/>Re-evaluate on frozen val + 5 texts"]
    C -- no --> E["Structural (H4/H6):<br/>Engram dropout arm, then adapter decision"]
    B -- "SI4 passes (doc 05 §5)" --> F["Position Engram as knowledge store,<br/>not a general CE improver"]
    B -- "SI4 fails" --> E
    D --> Q2{"gated ce_on <= ce_off - 0.01?"}
    Q2 -- yes --> A
    Q2 -- no --> E
    E --> G["If still no gain: report negative result;<br/>ship 72L loop + mHC with Engram gated off"]
```

**Definition of "Engram problem solved"** (all required): held-out `ce_on ≤ ce_off − 0.01` with a paired-bootstrap 95 % CI over the frozen 64-doc set; **no** harmful anchor in 3 consecutive audits; SI4 gap-write criterion met. Anything weaker is reported as partial.

---

## 5. Will SI-SRSI v2 fix the Engram problems? Will it raise capability?

- **Engram CE harm:** **not by itself.** SI-SRSI v2 trains on verified on-policy samples; it does not change the Engram architecture or add unique-token pressure. It can *indirectly* help (on-policy text is low-surprise, so less to memorise; Engram dropout arm; replay keeps the long-run gains) — that is hypothesis H4, to be measured by the `mHC-only / Engram-only / both / matched-token control` arms.
- **Engram as a storage mechanism:** SI-SRSI v2's **gap-write track (SI4)** is the first experiment that can actually demonstrate or refute the "vertical memory stores knowledge" thesis, because it measures paraphrase recall of facts the backbone did not know, with locality controls.
- **Capability:** expect *consistency* gains (maj@k → pass@1, constraint and tool-format adherence) on the task-bank pillars, a few points on dev suites, possibly small gains on held-out public suites. The frozen backbone caps what can be learned; no claim of new reasoning ability is made. Every number will carry a CI and a control arm; "no significant gain" is an acceptable, reportable outcome.

## 5b. Architectural Remedies Implemented (`7831f2f` — Opt-In, Non-Destructive to Running 20h Colab Job)

To address H3 (scale/norm miscalibration), H4 (unconditional firing on general text), and H6 (frozen consumer off-manifold sensitivity) without altering the running 20h checkpoint (`selective_gating_enabled=False` by default = 100% bit-identical):
1. **WISE Activation Scope Gate (`arXiv:2405.14768` §3.2)**: `ZipfianTieredMoEEngram` tracks compiled router directions `_compiled_scope_prototypes` in $\mathbb{S}^{63}$ (populated automatically by `compile_clusters_alphaedit` or `register_scope_prototypes`). When `scope_threshold > 0`, tokens whose router query $u_t$ has $\max_p \langle u_t, p \rangle < \tau_{\text{scope}}$ receive exact `0.0` perturbation (`scope_mask = 0`), guaranteeing `CE(ON) == CE(OFF)` on out-of-scope general text.
2. **DeepSeek Signed-Sqrt RMSNorm Query-Key Gate (`arXiv:2601.07372` §2.3, `engram_demo_v1.py`)**: When `qk_gate_enabled=True`, computes $\alpha_t = \sigma(\text{RMSNorm}(h_t)^\top \text{RMSNorm}(m_t)/\sqrt{W})$ followed by $\text{sign}(\cdot)\sqrt{|\cdot|}$ to suppress unaligned memory retrievals before projection.
3. **In-Manifold Trust-Region Relative Write Norm Cap**: When `max_rel_delta_norm > 0` (e.g., `0.03`), bounds $\|\Delta h_{\text{Engram}}\|_2 \le \rho_{\max} \|h\|_2$ per token so no anchor can push the frozen consumer off-manifold.
4. **Closed-Form Continual Knowledge Injection & Gap-Write Test (`BIE-5` / `SI4`, `sparkx_bie.run_gap_write_test`)**: Compiles synthetic novel entity facts (`Kestrel-941`, `Dr. Soren Vance`, `Aethelgard-7`) directly into `ZipfianTieredMoEEngram` via `compile_clusters_alphaedit` + WISE scope prototypes and measures Reliability (target NLL reduction), Paraphrase Generalization, General-Text Locality (`|CE_general_after - CE_general_before|`), and Sequential Null-Space Stability.

## 6. Roadmap hooks

| Milestone | Where |
| :-- | :-- |
| Long run → decision tree (§4) | tracker row *20 h resumable long run* |
| SI0–SI7 | [05 §7](05_SI_SRSI_V2_DESIGN.md) |
| Capability baseline incl. loop-vs-36L on **true** held-out text | SI0 |
| Selective gating + Gap-Write (BIE-1/3/5, SI4) | `sparkx_bie.py`, Cell 6 of `SparkX_Loop_v03_LongRun_Colab.ipynb` (`7831f2f`) |
