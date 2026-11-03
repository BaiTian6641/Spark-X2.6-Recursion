# 04 — Verified References (cross-checked 2026-11-03)

> Every entry below was looked up by web search on **2026-11-03**. The arXiv ID, title and first author were matched against arXiv / OpenReview / the official repo.
> **Rule:** a paper may only be cited in the paper draft or in `AGENTS.md` if it is in this table. If it isn't here yet, add it after verifying it.
> Component IDs (C0–C9) refer to [`01_ARCHITECTURE_DESIGN.md`](01_ARCHITECTURE_DESIGN.md).

## 1. Backbone & Attention

| # | Reference | ID / Venue | What we take from it | Used in |
| :--- | :--- | :--- | :--- | :--- |
| R1 | XHToken, *Spark-X2.6-4B* (`config.json`, `modeling_spark.py`) | HF model card | Ground truth for 36L hybrid SWA/global layout, partial RoPE 0.25 on global layers, `g_proj` head-wise gate, `out_proj`, GQA 16/4, head_dim 256 | C0, `00 §1` |
| R2 | Qiu et al., *Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free* | arXiv 2505.06708 (NeurIPS 2025) | Explains the post-SDPA sigmoid head gate. Our residual targets **must** include `σ(g_proj(h))` (fixes M2) | C1, C4 |
| R3 | Kubicek, Caciolai, Peng, Sumant, Hoover, Greenlee, Battleday, Urban, Hoefler, Whittington, *Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention* | arXiv 2610.42147 · repo `Prefrontal-AI/latent-sparse-attention` | Low-dim unit-sphere index, pooling, sink + local window, cluster-level key selection. We apply it **only to the RoPE-free 192 dims** of global heads | C1 (D-LSA), Phase 2 |
| R4 | Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context LMs?* | arXiv 2404.06654 (COLM 2024) | Long-context retrieval/aggregation suite for gate G2 | Tier C |
| R5 | Kuratov et al., *BABILong* | arXiv 2406.10149 (NeurIPS 2024 D&B) | Reasoning-in-a-haystack at long contexts | Tier C |

## 2. Knowledge Editing & Null-Space Writes

| # | Reference | ID / Venue | What we take from it | Used in |
| :--- | :--- | :--- | :--- | :--- |
| R6 | Meng et al., *Locating and Editing Factual Associations in GPT* (ROME) | arXiv 2202.05262 (NeurIPS 2022) | Linear key→value view of FFN down-projections; rank-one closed form | C5 |
| R7 | Meng et al., *Mass-Editing Memory in a Transformer* (MEMIT) | arXiv 2210.07229 (ICLR 2023) | Batched least-squares write with a preserved-key covariance; spreading residuals across layers. This is the basis of greedy layer-wise LRCD | C4, C5 |
| R8 | Fang et al., *AlphaEdit: Null-Space Constrained Knowledge Editing* | arXiv 2410.02355 (ICLR 2025 Outstanding Paper) | Project ΔW onto the null space of K0 = **general-corpus** keys (100K Wikipedia triplets in the paper) plus previously edited keys. Our K0 must include a general corpus (fixes M4) | C5, C7 |
| R9 | Wang et al., *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing* | arXiv 2405.14768 (NeurIPS 2024) | **Side memory** next to a frozen FFN, an activation-based router/gate, knowledge sharding and merging. This is the template for ESM | C5 |
| R10 | Hartvigsen et al., *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors* | arXiv 2211.11031 (NeurIPS 2023) | Codebook with ε-radius deferral, so a write only fires near its key. This informs the ESM key gate | C5 |
| R11 | Gu et al., *UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing* | arXiv 2505.14679 | Lifelong normalisation of feature statistics; closed-form edits at 1M+ scale on a 24 GB GPU. Baseline for D3 and evidence that home-lab scale is feasible | C5, Tier D3 |

## 3. Memory Layers, Engram & Continual Learning

| # | Reference | ID / Venue | What we take from it | Used in |
| :--- | :--- | :--- | :--- | :--- |
| R12 | Berges et al., *Memory Layers at Scale* | arXiv 2412.09764 | Product-key sparse memory layers as a trainable lookup store | C7, C8 |
| R13 | Lin, Zettlemoyer, Ghosh, Yih, Markosyan, Berges, Oğuz, *Continual Learning via Sparse Memory Finetuning* (SMF) | arXiv 2510.15103 | Update only memory slots with high TF-IDF activation relative to background usage. Forgetting is −11% NQ F1 for SMF vs −89% full FT and −71% LoRA. Template for deep-sleep consolidation | C7, Phase 6 |
| R14 | DeepSeek-AI, *Conditional Memory via Scalable Lookup* (Engram) | arXiv 2601.07372 | O(1) hashed N-gram memory fused into mid layers with a context gate. Origin of the Tier-1 lexical tables | C8 |
| R15 | Behrouz, Zhong, Mirrokni, *Titans: Learning to Memorize at Test Time* | arXiv 2501.00663 | Surprise-gated neural long-term memory. Motivates the KeySurprise term in salience | C2 |
| R16 | Behrouz, Razaviyayn, Zhong, Mirrokni, *Nested Learning: The Illusion of Deep Learning Architectures* | arXiv 2512.24695 (NeurIPS 2025) | Multi-timescale associative-memory view. Our hot KV / ESM / Engram / backbone hierarchy is framed as nested update frequencies | `01 §1.1` |
| R17 | McClelland, McNaughton, O'Reilly (1995), *Why there are complementary learning systems…*; Kumaran, Hassabis, McClelland (2016), *What learning systems do intelligent agents need?* | Psych. Review 102(3); TiCS 20(7) | Fast hippocampal store plus slow neocortical consolidation in sleep, mapped to ESM (fast) → Engram (slow). Framing only; no empirical claim rests on it | `01 §1.1`, paper intro |

## 4. Context Internalisation & Sleep-Time Compute

| # | Reference | ID / Venue | What we take from it | Used in |
| :--- | :--- | :--- | :--- | :--- |
| R18 | Lin, Snell, et al. (Letta / UC Berkeley), *Sleep-time Compute: Beyond Inference Scaling at Test-time* | arXiv 2504.13171 | Shift compute to idle periods. Our scheduler runs light/deep sleep when idle | C6 |
| R19 | Eyuboglu et al. (Stanford Hazy Research), *Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study* | arXiv 2506.06266 | **Self-study**: generate synthetic Q/A about a corpus and distil with in-context teacher → student. Template for probes plus the LRCD objective | C3, C4 |
| R20 | Charakorn, Cetin, Uesaka, Lange (Sakana AI), *Doc-to-LoRA* | arXiv 2602.15902 | Hypernetwork that maps a document to a LoRA. A comparison baseline and related work; we don't use hypernetworks | Related work, Tier D baseline |
| R21 | Snell, Klein, Zhong, *Learning by Distilling Context* | arXiv 2209.15189 | Context distillation: teacher with context vs student without. Foundation of LRCD | C4 |
| R22 | Zweiger et al., *Self-Adapting Language Models* (SEAL) | arXiv 2506.10943 | Model-generated "self-edits" and implications as training data for knowledge incorporation. Informs probe generation | C3 |

## 5. Recurrence / Looping & Residual Mixing (Track L)

| # | Reference | ID / Venue | What we take from it | Used in |
| :--- | :--- | :--- | :--- | :--- |
| R23 | Geiping, McLeish, et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* | arXiv 2502.05171 | Shared recurrent block, randomised iteration count at train time | C9 |
| R24 | McLeish, Li, Kirchenbauer, et al., *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence* | arXiv 2511.07384 | **Most relevant to us**: convert a pretrained non-recurrent LM to recurrent, using a recurrence curriculum plus a data "healing" phase. Template for L3 | C9, Track L3 |
| R25 | Zhu et al. (ByteDance Seed), *Scaling Latent Reasoning via Looped Language Models* (Ouro) | arXiv 2510.25741 | Entropy-regularised learned exit depth; gains come from knowledge *manipulation*, not storage. This supports keeping storage vertical and reasoning in the loop | C9, Track L4 |
| R26 | Bae et al., *Mixture-of-Recursions* (MoR) | arXiv 2507.10524 (NeurIPS 2025) | Token-level recursion-depth router plus recursion-wise KV caching/sharing | Track L4, L5 |
| R27 | Xie, Wei, et al. (DeepSeek-AI), *mHC: Manifold-Constrained Hyper-Connections* | arXiv 2512.24880 | Sinkhorn projection of multi-stream residual mixing onto the Birkhoff polytope, which restores identity-mapping stability. Basis of `TwoStreamMHC` | C9 |

## 6. Memory Benchmarks

| # | Reference | ID / Venue | Used in |
| :--- | :--- | :--- | :--- |
| R28 | Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* | arXiv 2410.10813 (ICLR 2025) | Tier D2 (external validity) |
| R29 | ZsRE (Levy et al. 2017) / CounterFact (Meng et al. 2022) | CoNLL 2017 / R6 | Tier D3 (editing comparability) |
| R30 | Zhou et al., *Instruction-Following Evaluation for LLMs* (IFEval) | arXiv 2311.07911 | Tier B (full 541 prompts, never n=50 for claims) |

---

## 7. Corrections Made During Verification

| Earlier claim (in `AGENTS.md` / code comments) | Verified fact | Fixed in |
| :--- | :--- | :--- |
| Global layers are NoPE (`partial_rotary_factor = 0`) | Partial RoPE 0.25, θ = 5e6: dims [0:64) rotated, [64:256) position-free | `AGENTS.md` errata, `00 §1` |
| `o_proj` exists | No `o_proj`. It's `g_proj` (sigmoid head gate) → `out_proj` | `AGENTS.md` errata, Phase 0.2 |
| Engram anchors {3,6,11,15,16,19,23,27,31} | Anchors are exactly {3,7,11,15,19,23,27,31}, applied after the full decoder block | `AGENTS.md` errata |
| LSA authors "Kubicek et al." with an invented title | Full author list and title as in R3 | `b841a3e` |
| Doc-to-LoRA author list | Charakorn, Cetin, **Uesaka**, Lange | `b841a3e` |
| "Mathematically zero forgetting" | Holds only for keys in span(K0) and only up to null-space threshold τ. Must be **measured** (`00 §5`) | `AGENTS.md` errata |

## 8. BibTeX Stubs (new entries not yet in `REFERENCES.md`)

```bibtex
@article{lin2025smf,   title={Continual Learning via Sparse Memory Finetuning}, author={Lin, Jessy and Zettlemoyer, Luke and Ghosh, Gargi and Yih, Wen-tau and Markosyan, Aram H. and Berges, Vincent-Pierre and O{\u{g}}uz, Barlas}, journal={arXiv:2510.15103}, year={2025}}
@inproceedings{wang2024wise, title={WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models}, author={Wang, Peng and others}, booktitle={NeurIPS}, year={2024}, note={arXiv:2405.14768}}
@inproceedings{hartvigsen2023grace, title={Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors}, author={Hartvigsen, Thomas and others}, booktitle={NeurIPS}, year={2023}, note={arXiv:2211.11031}}
@article{gu2025ultraedit, title={UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Language Models}, author={Gu, Xiaojie and others}, journal={arXiv:2505.14679}, year={2025}}
@article{eyuboglu2025cartridges, title={Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study}, author={Eyuboglu, Sabri and others}, journal={arXiv:2506.06266}, year={2025}}
@article{snell2022context, title={Learning by Distilling Context}, author={Snell, Charlie and Klein, Dan and Zhong, Ruiqi}, journal={arXiv:2209.15189}, year={2022}}
@article{zweiger2025seal, title={Self-Adapting Language Models}, author={Zweiger, Adam and others}, journal={arXiv:2506.10943}, year={2025}}
@article{mcleish2025retrofit, title={Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence}, author={McLeish, Sean and Li, Ang and Kirchenbauer, John and others}, journal={arXiv:2511.07384}, year={2025}}
@article{zhu2025ouro, title={Scaling Latent Reasoning via Looped Language Models}, author={Zhu, Rui-Jie and others}, journal={arXiv:2510.25741}, year={2025}}
@inproceedings{bae2025mor, title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation}, author={Bae, Sangmin and others}, booktitle={NeurIPS}, year={2025}, note={arXiv:2507.10524}}
@article{qiu2025gated, title={Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free}, author={Qiu, Zihan and others}, journal={arXiv:2505.06708}, year={2025}}
@inproceedings{wu2025longmemeval, title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory}, author={Wu, Di and others}, booktitle={ICLR}, year={2025}, note={arXiv:2410.10813}}
```
