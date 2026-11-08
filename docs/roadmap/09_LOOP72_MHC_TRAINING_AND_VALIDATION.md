# Spark X2 6 4B Loop72 training and validation plan

Engineering review and next implementation plan. Prepared 8 November 2026. Reviewed branch loop72-dlsa at 0ebce51cae69195cf6a51c1e270a67206550e0b7.

## 1 Recommendation and evidence boundary

The next release should be a frozen Spark X2.6 4B backbone with 72 executed layer visits, explicitly trained two-stream mHC, per-visit D-LSA selectors trained against that mHC checkpoint, and bounded GPU cache residency backed by tiered storage. MoE-Engram remains deferred. Establish a correct dense mHC model first, then introduce sparse attention, cache quantization, and longer context in separately measured stages.

The reported 34 tokens per second is a useful user-provided observation. It is not a reproduced benchmark in this review. The exact GPU, model checkpoint, prompt length, output length, dtype, and launch command for that observation were not available. The reviewed fast path also differs from the learned mHC path: in the original branch snapshot, evaluation with native GELU and no Engram bypasses the mHC controllers and uses fixed highway coefficients. It still executes all 72 layer visits. Consequently, speed and quality comparisons must identify the recurrence implementation, not merely print “72L.” [S1, S2]

The existing notebook is not a full benchmark implementation. Its default official-dataset cell loads one row per dataset; hand-authored tasks are mixed into benchmark names; the context cell allows at most 16,384 live tokens while every requested horizon is at least 32,768. Several converters change task definitions or reveal ground truth. These issues invalidate the corresponding comparisons, independently of whether the model can generate plausible answers. [S3, S4]

This review includes concrete runtime, benchmark, and harness repairs with CPU regression tests. It does not establish trained Spark quality, a faster GPU decode rate, official suite scores, or 1M/5M useful context. Those require the actual checkpoint and GPU execution. The accompanying validation record states exactly which tests ran. The native full-dataset runner covers three long-context dataset families and records every other official harness requirement as blocked; it does not label those missing integrations complete.

The final CPU suite passed 220 tests. An actual Oh My Pi v18.18.1 process completed a four-turn read/edit/test workflow through the new bridge against a scripted backend, with an independent verifier and unchanged public tests. The full model benchmark invocation stopped at preflight because CUDA and the datasets dependency were unavailable. These are infrastructure results, not trained-model results. See `docs/LOOP72_REVIEW_VALIDATION.md` for the evidence boundary and `docs/roadmap/10_LOOP72_RUNBOOK.md` for executable evaluation commands.

## 2 Architecture contract

The model contract must be exported with every checkpoint and benchmark result. A shape-compatible state dictionary is not sufficient evidence of equivalent behavior.

| Component | Required contract | Acceptance evidence |
| --- | --- | --- |
| Backbone | Original Spark X2.6 4B checkpoint; frozen initially; hidden width 2560, FFN width 10240, vocabulary 131072 | Pinned source revision and frozen-parameter checksum |
| Execution | Prelude L0–3, L4–27 twice, L12–23 once, coda L28–35 | Actual execution trace with 72 visits |
| Attention visits | 54 local and 18 global visits | Per-visit counters and unique cache slots |
| Local attention | Native 512-token window and full rotary dimensions | Dense versus cached parity at window boundaries |
| Global attention | Four KV heads, sixteen query heads, head dimension 256; 64 rotary and 192 unrotated dimensions | Attention reconstruction and prefix-causality tests |
| Native transformations | Preserve GELU, head-wise sigmoid attention gate, output projection, scale, normalization and RoPE conventions | Baseline logits and attention-output comparisons |
| mHC | Two streams, explicit learned recurrence mode, same computation during training and inference | Controller hooks, gradients, state roundtrip and mode manifest |
| D-LSA | Eighteen visit-specific selector states; maps tied to the exact backbone and mHC revisions | Map count, visit IDs, teacher alignment and sparse-quality tests |
| Engram | Not constructed for this variant | Zero allocated Engram parameters |
| Memory | Bounded GPU residency, explicit host archive and optional later SSD tier, separately versioned episodic memory | Measured allocation and transfer ledger |
| Context | Advertise only the tested limit and reserve generation space | End-to-end execution plus retrieval/reasoning evaluation |

The 72-visit count is 4 + 24 + 24 + 12 + 8. Weight sharing reduces parameter storage; it does not turn 72 sequential visits into the work of 36 visits. Reusing a physical layer also does not make its KV states interchangeable between passes. Different hidden states produce different keys and values. Keep independent visit caches unless a separately trained cache-sharing approximation passes its own quality gate. [S1, S5]

Use explicit variant names in experiments: native 36-layer diagnostic, fixed-highway Loop72 baseline, learned-mHC Loop72 dense, learned-mHC Loop72 D-LSA, and learned-mHC Loop72 D-LSA with a specific cache codec. The main release remains 72 visits. The 36-layer result is an ablation and teacher baseline.

## 3 Corrections required before training or scoring

### Runtime identity and checkpoint persistence

The original evaluation branch can bypass learned controllers even while an mHC-enabled flag is true. Loading a local loop state also resets controllers during configuration, and embedded D-LSA weights can be skipped because configuration constructs the bank after state restoration. These are lifecycle failures, not tuning choices. The repair makes recurrence selection explicit, removes implicit reset, restores embedded selector state after bank construction, gives an explicit external selector checkpoint precedence, and records provenance. Missing checkpoint paths must raise an error. [S1, S2]

The compatibility mode remains available for reproducing the reported fast path. A new mHC checkpoint must save learned_mhc as its recurrence mode. The loader must honor that mode in both training and inference. A user-supplied override must be visible in the manifest and cannot silently change the identity used for published results.

The original attached paged D-LSA forward is an inference implementation: it stores detached keys and values. Training through it loses gradient paths through K and V. Initial mHC training therefore uses the differentiable dense/native attention path, with inference paging disabled. A guard should reject attempts to train through detached cache storage. “Some controller gradients are nonzero” is insufficient; attention-gradient parity must show that all intended paths remain present. [S5, S6]

### Benchmark validity

Remove answer-dependent prompt construction. In the reviewed code, AA-LCR can substitute the answer when context is absent; MCP-Atlas injects reference claims; MRCR chooses a neighborhood using the target index and reduces the expected response to the required prefix. None of these are faithful evaluations. Missing assets should block a task. Context should remain intact or the result should be explicitly identified as a modified protocol. [S3]

MRCR must grade retrieval of the requested message, using the required prefix and content similarity. GraphWalks must accept the actual node identifiers and use the official final-line parser and set metric. A BFCL function call must match the function identity as well as arguments. Unknown IFEval constraints must not pass automatically. A SciCode task without runnable tests is unsupported, not successful. SWE-bench requires repository patches and test execution; a substring in an answer is not a resolved issue. [S7, S8, S9]

Separate dataset provenance from scoring validity. Loading a row from an official Hugging Face dataset does not mean the prompt, assets, execution environment, answer extraction, or grader matches the benchmark.

### Generation protocol

Official evaluation must be able to disable automatic closing of the reasoning section, answer-prefix injection, repetition penalties, and second-pass answer forcing. These are sampling changes and may alter both correctness and throughput. Record the actual generation policy, EOS behavior, reasoning tokens, answer tokens, truncation rate, and finish reason.

Give the model sufficient answer budget. A 768-token cap with forced conclusion is a different experiment from an uncapped or high-budget reasoning protocol. Do not attribute differences between these settings to mHC or D-LSA alone.

## 4 mHC mathematical design

### Two-stream residual mixing

For two streams, a doubly stochastic mixing matrix can be represented with one scalar probability.

$$
P(p)=\begin{bmatrix}p&1-p\\1-p&p\end{bmatrix},\qquad 0\le p\le1.
$$

This representation makes row sums and column sums equal to one by construction. Its eigenvalues are 1 and 2p−1, so the spectral norm of this fixed matrix is one. Repeated application of fixed doubly stochastic matrices preserves the convex mixing property. This is the useful stability property motivating constrained residual mixing. The published mHC work also addresses practical overhead using fusion and recomputation; its measured gains and overhead do not transfer automatically to this two-stream Spark retrofit. [S10]

The current controller applies a gain to the mixing displacement. Its effective residual map is the following.

$$
P_{\mathrm{eff}}=(1-g)I+gP.
$$

The convex-combination guarantee requires g between zero and one. The legacy tanh gain can become negative. Its floor variant can also exceed one slightly. Thus the fact that P is doubly stochastic does not, by itself, prove that the effective residual map satisfies the same constraint. Telemetry should report minimum matrix entry, row and column errors, and gains outside the intended interval.

For new training, implement a versioned bounded-gain parameterization. A suitable starting proposal is g = sigmoid(u), with u initialized so that extra-pass gains are small but nonzero. Candidate initial gains are 0.01 for P2 and 0.005 for P3. These are experiment settings, not established optima. A bounded range with a separately chosen maximum is also possible, but changes the learning problem and requires its own ablation.

Do not reinterpret saved tanh values as sigmoid logits. The checkpoint must name the gain parameterization. If migration is needed, transform the effective gains where an inverse exists and verify logits before and after migration. Values outside the new range require an explicit compatibility decision and cannot be silently clamped while claiming exact equivalence.

The fixed-matrix norm statement is not a proof that the full network is non-expansive. Input-dependent routing contributes derivatives of P with respect to the hidden state, and attention, MLP, read/write routing, and pass adapters introduce additional Jacobian terms. Monitor the actual system rather than claiming a global stability theorem from one constrained matrix.

### Initialization and gradient flow

At startup, retain the native first-pass behavior as closely as possible. Use near-identity residual mixing, a strong initial preference for reading the primary stream, zero dynamic routing projections, randomly initialized adapter down projections, and zero adapter up projections. Do not initialize both sides of a low-rank adapter to zero; that can remove its initial learning signal.

Exact zero extra-pass gains make the model easier to compare to a shallower baseline, but initially suppress gradients into parts of the extra-pass computation. Gains can still receive gradients and activate those paths later. Small positive gains allow earlier adaptation but sacrifice exact identity. Run both initialization variants on the same small pilot and choose based on retention, stability and learning progress.

The current finite softmax/sigmoid initialization is near identity, not mathematically identical to a one-hot read/write and identity mix. Do not call it exact step-zero parity without measuring it. Compare next-token logits and held-out NLL to the native backbone before training.

Verify all of the following on a tiny model, then on the real checkpoint: P2 and P3 gains receive gradients; at least one dynamic routing tensor and an adapter up projection receive gradients; frozen backbone parameters receive no gradients; the loss still differentiates through frozen layers with respect to their inputs; mHC calls occur in inference; save/load preserves outputs and effective gains.

## 5 Staged mHC training recipe

### Stage A establish a trustworthy dense baseline

Load the pinned original backbone, preserve native GELU and attention semantics, construct no Engram, disable episodic memory writes, and select the explicit learned mHC path. Use differentiable native attention without paged inference storage. Validate native 36-layer behavior and the new 72-visit initialization separately.

Freeze all backbone, embedding and language-model head weights. Initially freeze D-LSA because it is not used in this stage. Enumerate trainable parameters by name; do not rely on an optimizer filter whose naming convention accidentally omits P3 or includes backbone tensors.

At the current width, two dynamic controllers per block, rank-16 pass adapters and four allocated pass slots, the controller bank contains about 10.08 million parameters. The formula and actual count should be printed by the training program. Only three pass indices are used in the fixed 72-visit schedule. An eventual three-slot layout can remove unused parameters, but is a checkpoint-schema change; it is not required for the first correct run. The 18 selector maps add 1,769,472 parameters when constructed, but should not be optimized during dense mHC training. [S1, S5]

Run 20–100 update steps on deterministic local training examples to catch wiring failures, loss-scale problems, nonfinite gradients and checkpoint incompatibility. This is an engineering check, not evidence of model quality.

### Stage B retain native behavior while activating recurrence

Start with a small budget of approximately 0.5–2 million training tokens. Keep contexts at 1K–2K and microbatch size one until memory is measured. Use answer-level cross entropy together with retention distillation from the unmodified native model.

A practical objective is:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+\beta T^2 D_{\mathrm{KL}}(p_{\mathrm{teacher},T}\Vert p_{\mathrm{student},T})+\lambda_h\mathcal{L}_{\mathrm{state}}+\lambda_r\mathcal{R}.
$$

Use the native teacher primarily to retain language, formatting and existing skills. It is not evidence that additional loops learn stronger reasoning. For reasoning and code improvement, provide verified targets, executable tests and successful trajectories whose supervision goes beyond copying the teacher’s mistakes.

Start with CE weight 1, retention weight β in the range 0.2–1.0, temperature T between 1 and 2, and small or zero hidden-state regularization. These values are starting ranges for a short sweep. Do not combine large hidden-state matching and strong KL penalties that force every extra loop to reproduce the original network and then expect large recurrent gains.

Compute teacher targets before the student graph when possible, using the same frozen backbone in a separate native-mode context. Keep teacher execution under no_grad; retain the student autograd graph through frozen operations after the first trainable intervention. Restore all mode and attention state after teacher evaluation.

If full vocabulary distillation is too expensive, sample a modest number of supervised positions per sequence and calculate exact KL for those positions in chunks. A teacher top-k approximation may include an explicit “other probability mass” bucket; renormalizing top-k probabilities and calling the result full-vocabulary KL is incorrect. Record which approximation is used.

### Stage C train the full 72-visit controller

Allocate an initial 5–20 million token pilot, expanding only if validation is still improving. Favor real coding and reasoning examples with verifiable outcomes. A proposed starting mixture is 40 percent code and debugging, 20 percent general language and instructions, 15 percent mathematics, 15 percent tool-use trajectories, and 10 percent document and repository reasoning. Adjust the mixture after domain-level validation rather than assuming the proportions are optimal.

Split coding data by repository or project, not by random files from the same repository. Separate train, calibration, development and final test sets. Exclude the evaluation suites, their reference patches, answer keys, task assets and near duplicates from all training and distillation sources. Generated training examples should be tagged with their provenance and verification method.

Use full 72 visits for the final stage. A 36→60→72 curriculum is an optional compute-saving experiment, supported as a general approach by recurrent-retrofit research, but the final release must be validated at 72. A simpler first pilot keeps all 72 visits and gradually increases useful extra-pass gains. Compare the two on quality per training hour before adopting the more complicated schedule. [S11]

Train routing and pass adapters in controlled groups. Begin with pass gains and adapter up projections, then allow read/write and dynamic routing to adapt. Keep static-mix or identity-mix ablations; input-dependent mixing is not automatically better when retrofitting a frozen backbone. Maintain separate learning rates for small routing/gain tensors and adapter matrices.

Proposed starting optimizer settings are AdamW, adapter learning rate 1e−4 to 3e−4, routing/gain learning rate 1e−5 to 5e−5, 3–5 percent warmup, cosine decay, gradient clipping around 0.5–1.0, zero weight decay on gains and routing biases, and modest decay on adapter matrices. These settings need pilot validation. Use a small fixed set of alternatives instead of a large hyperparameter search that consumes the entire training budget.

### Stage D specialize for tool use and agentic coding

Use successful read–edit–test trajectories from the intended Oh My Pi tool schemas, together with failure-and-repair examples. Preserve the original Spark chat template and its native tool-call syntax. Train valid function names, argument types, tool-result handling, patch precision, test interpretation and stopping behavior.

Train on the model’s actual supported tool formats. A generic JSON answer task does not establish native tool calling, and a hand-authored hashline imitation does not establish compatibility with the current OMP editor. Keep agent-generated training repositories disjoint from SWE-bench and final coding evaluation repositories.

Score both final task correctness and intermediate protocol validity. Track malformed tool calls, invalid paths, edit conflicts, repeated failing commands, unnecessary tool loops, and completed tests. A lower decode rate can still yield a faster successful coding task if it sharply reduces retries.

A short 1–5 million token specialization stage is a reasonable first experiment after retention is stable. Save the dense mHC checkpoint before this stage so that any general-skill loss can be measured and reversed through checkpoint selection.

### Stage E freeze mHC and train D-LSA

Freeze the chosen backbone and mHC checkpoint. Capture per-visit teacher attention from this exact model. Fit initial PCA maps on separate calibration documents, then distill the 18 query/key selector maps using causal per-key teacher distributions. Do not reuse selector maps trained on a different recurrence, activation convention or mHC checkpoint without validation. [S5, S12]

Start at 8K–16K contexts so sparse selection actually excludes keys. With a 4,096-key long-range budget plus sinks and a 512-token window, short 2K or 4K tasks may be exact attention and cannot validate the selector. Progress to 32K and 64K/128K only after attention-mass retention and end-to-end quality stabilize.

Sample query positions, visits, documents and context regions. Keep local and sink accounting exact. Teacher attention must use the actual rotary dimensions, scale, query-head grouping and causal mask. Comparing averages of head probabilities is not generally equivalent to softmax of averaged or summed logits.

Avoid recording full Q, K and V at every visit when the loss only needs a small set of queries and keys. Capture sampled Q rows, retain or recompute required K blocks, and rotate through visit groups if memory is tight. Log the cost of target generation separately from selector optimization.

Train flat search first. Validate a persistent hierarchical search as a separate approximation with matched selected-key budget, then measure its recall, latency and transfer amplification. An asymptotic claim about a tree does not excuse rebuilding or scanning the entire hierarchy per query.

### Stage F validate cache codecs and long context

Begin with BF16 archived KV. Compare INT8, mixed rotary-BF16 and INT4 caches separately. Measure logit difference, held-out NLL, task accuracy, selected-attention output error and practical transfer bytes. Store scale/metadata overhead in the memory ledger; payload-only arithmetic is a lower bound.

For quantized keys, determine whether the selector indexes original or reconstructed keys. Record that choice and test the distribution mismatch. Higher compression is acceptable only when the end-to-end quality and bandwidth results justify it.

Extend the context ceiling incrementally: 32K, 128K, 256K, 512K, 1M, then experimental 2M and 5M. Retain short-context regression checks at every step. Position extension, selector quality, KV capacity and serving bandwidth are separate gates; passing one does not establish the others.

## 6 Training efficiency and resource planning

The main benefit of frozen-backbone training is avoiding backbone gradients and optimizer states. It does not eliminate the forward computation or the backward computation required to propagate gradients through frozen layers into earlier trainable controllers.

Use BF16 backbone matmuls and initially keep routing reductions, normalization statistics and small gains in FP32. Keep sensitive calculations numerically stable before lowering their precision. Benchmark mixed-precision controller projections separately, because a tiny FP32 matrix can be dominated by launch overhead rather than arithmetic.

Use activation checkpointing on meaningful recurrent segments with a non-reentrant implementation where supported. Avoid mutable inference caches or index updates inside checkpointed functions. Recomputed forwards must reproduce the original operation, including RNG and position handling. PyTorch’s checkpoint documentation explains the recomputation tradeoff; actual memory savings depend on this model’s graph. [S13]

The existing `offload_checkpoint_to_cpu` setting does not currently implement activation offload: the checkpoint context returns a null context. Do not count this flag as a memory saving in the training budget. Real activation offload needs a separate implementation, transfer measurements and gradient-parity checks. Ordinary activation recomputation is available and is the first option to measure. [S1]

Compute next-token CE in vocabulary/head chunks instead of retaining a full sequence-by-vocabulary FP32 tensor. At 32,768 positions and vocabulary 131,072, one FP32 logits tensor occupies 16 GiB before gradients or copies. For retention KL, use sampled positions and chunked logits. Do not request full-sequence logits in a training path that only needs a scalar loss.

The frozen prefix before the first trainable intervention can be computed without building its gradient graph. If precomputing prefix states is useful, fingerprint the exact token IDs, packing boundaries, positions and backbone revision. Do not reuse hidden states downstream of trainable mHC, since they change with every optimizer step.

Pack independent examples only if the attention mask prevents cross-example leakage and the labels correctly mask prompts and padding. Concatenating many instruction examples with no segment isolation changes training behavior. Verify packed and separately evaluated losses on a small fixture.

Use gradient accumulation to reach a useful token batch. Optimizer stepping, clipping and learning-rate scheduling must advance once per optimizer update, not once per microbatch. Log training tokens, supervised tokens, padding fraction, effective batch size and data repetition.

| GPU memory class | Initial probe only | Scale after measuring |
| --- | --- | --- |
| About 24 GiB | BF16 frozen base, 1K–2K context, microbatch 1, checkpointing | Accumulate tokens; avoid simultaneous teacher and student graphs |
| About 48 GiB | 2K–4K context, microbatch 1 | Test microbatch 2 or longer sequences |
| About 80–96 GiB | 4K context, microbatch 1–2 | Probe 8K and larger teacher-query batches |
| Smaller devices | Establish inference first; test short training carefully | Quantized-base adaptation is a separate quality experiment |

These are experiment starting points, not fit guarantees. Actual feasibility depends on activation layout, controller graph, attention backend and software version. Record measured peak allocated/reserved VRAM, host RSS and throughput after warmup.

Ordinary AdamW is reasonable for roughly ten million controller parameters. An 8-bit optimizer adds complexity while saving relatively little beside the frozen backbone and activation footprint. Add it only if measurements show optimizer state is a meaningful memory bottleneck.

Use measured training throughput to set the session budget. Divide the planned number of tokens by observed end-to-end training tokens per second, including teacher work, validation and checkpointing. Inference at 34 tokens per second does not predict training throughput.

For a 24-hour Colab session, reserve time for model loading, initial validation, at least one checkpoint export, and the final evaluation slice. A practical pilot can devote roughly 10 percent to setup and correctness, 65 percent to training, 15 percent to evaluation, and 10 percent to export and contingency. Adjust these shares after the first measured run. Save a usable checkpoint at stage boundaries even if the overall campaign spans multiple sessions.

## 7 Inference optimization beyond 34 tokens per second

### Preserve model identity before comparing speed

The measured baseline must include recurrence mode, mHC checkpoint, selector checkpoint, dtype, attention backend, cache codec, selected-key budget, context length, memory snapshot and generation policy. A fixed-highway result should not be compared with a learned-mHC result as though only the kernel changed.

The source already incorporates several improvements from the earlier review: native GQA dispatch, preallocated local rolling storage, a short-context resident buffer, candidate-only top-k sampling and separated generation timing. Validate their dispatch and memory behavior; do not count them again as unimplemented speedups. [S1, S5, S6]

The review patch caches constant distance-prior scalars during evaluation, invalidating the cache when parameter versions, identity, device or dtype change. This removes repeated scalar device-to-host reads in that selector code path without changing the selection values. It is a code-level optimization; the speed gain still needs a GPU measurement.

### First optimization pass

Profile one short-context and one long-context decode after warmup. Record CPU launch time, GEMM/GEMV time, SDPA time, index selection, KV gather/dequantization, host/device transfer and sampling. Run the throughput benchmark separately from the profiler because profiling changes timing.

For short prompts, avoid building or maintaining expensive sparse index structures until sparse selection is actually needed, provided the delayed initialization exactly reconstructs the same index before switching. A fast resident dense path is useful when all history fits on the GPU. Do not run Ward updates on every request merely to use an exact attention result.

For local cache updates, measure the actual backing storage of the rolling 2W buffer. The logical W-token history and physical 2W allocation have different memory costs. Keep both in the ledger. For global short-context storage, avoid maintaining redundant tail copies and resident mirrors if one canonical buffer can provide both views safely.

Compile stable tensor-only regions first: mHC normalization and routing projections, residual mixing, adapter operations, and possibly MLP projections. Preserve native GELU. Dynamic Python index selection and host page retrieval remain outside a captured CUDA graph until stable buffers and graph-safe execution are implemented. Use compilation or graph capture only after parity and graph-break diagnostics pass. [S14]

### Restore and optimize learned mHC

A learned mHC model executes routing and adapters that the current fast highway skips. It may initially be slower than 34 tokens per second. That is a model change, not evidence that the optimization failed.

Fuse the two-stream normalization, small routing projections, sigmoid transforms and residual updates into fewer kernels. Precompute only quantities independent of the token: frozen gains, pass biases, and static transforms. Input-dependent routing cannot be cached across tokens as an exact optimization.

For an optional static-mixing variant, train it explicitly and compare quality against dynamic mHC. Static matrices are easier to precompute and fuse. If it performs equally well on held-out tasks, it may provide a better deployment choice. It remains a separately trained architecture variant.

### Long-context runtime work

Use persistent page tables and reusable GPU gather buffers. Batch cold-page transfers and dequantization. Avoid per-page GPU.item calls and admissions that evict pages needed later in the same gather. Evaluate hot-page hit rate, unique pages fetched, selected tokens per fetched page, and bandwidth amplification.

Use pinned host slabs and asynchronous transfers where dependencies allow overlap. PCIe transfer time cannot be hidden when the next operation requires the missing page immediately. A claim of asynchronous transfer is not enough; the trace must show useful overlap.

Keep the centroid hierarchy persistent, update it incrementally, and gather selected bucket memberships directly. Benchmark build/update cost separately from query cost. Query-coherence selection reuse changes the selected keys and must be evaluated as an approximation rather than an exact speed fix.

### Weight traffic and realistic ceilings

Using the native dimensions, a physical layer contains about 104.86 million attention-plus-MLP linear weights. Reading those BF16 matrices for 72 visits and the output head once gives approximately 15.77 GB of logical weight traffic per generated token. This simplified model excludes attention-cache traffic, controller work and other overhead, and assumes weights are not retained across the relevant visits in on-chip cache.

At 34 tokens per second, that model corresponds to about 536 GB per second of effective weight traffic. It is a diagnostic calculation, not a measured hardware bandwidth or a universal lower bound. The GPU and profiler determine whether the remaining bottleneck is memory bandwidth, kernel launches or cache movement.

The MLP contributes roughly 75 percent of those per-layer linear weights. Faster attention alone therefore cannot remove most short-context backbone work. If an optimizable component occupies fraction f of latency, eliminating it entirely gives at most 1/(1−f) speedup. A component occupying 20 percent cannot double overall decode speed.

After the BF16 model is validated, benchmark a high-quality INT8 or weight-only INT4 backend with the native activation and recurrent execution intact. Use actual fused quantized kernels, not merely compressed checkpoint files that dequantize into BF16 before every operation. Keep quantization quality reports for coding and tool use.

Speculative decoding is a later option. A native 36-layer draft is not free: it still executes substantial model work. Measure draft cost, target verification cost, average accepted tokens and rejection rate before adopting it. Preserve the target sampling distribution if exact speculative decoding is claimed. Extra heads or a specially trained draft add another training task.

Agentic workloads can also benefit from safe prefix reuse and fewer tool retries. Prefix cache identity must include model and mHC hashes, selector revision, RoPE settings, the memory epoch, token prefix and every visit’s cache state. Reusing a mutable single-request cache across unrelated requests is not valid prefix caching.

## 8 KV memory mathematics and tiered design

For Spark’s four KV heads and 256 dimensions, the BF16 KV payload at one global visit is 4,096 bytes per token. With 18 global visits, five million tokens require about 343.3 GiB for BF16 global KV alone. INT8 payload is about 171.7 GiB and uniform 4-bit payload about 85.8 GiB, before scales, page metadata, index data, allocator slack and model weights.

| Global KV format at 5M tokens | Ideal payload for 18 visits |
| --- | --- |
| BF16 K and V | 343.3 GiB |
| 8-bit K and V | 171.7 GiB |
| Uniform 4-bit K and V | 85.8 GiB |
| BF16 rotary key slice plus 4-bit remaining K and V | 118.0 GiB |

The mixed format retains the 64 rotated key dimensions in BF16 and quantizes the other key dimensions and all values. It is an experimental quality/capacity tradeoff, not a proven optimal codec. The 54 logical 512-token BF16 local windows total 108 MiB; a 2W physical rolling implementation uses additional backing storage.

Ordinary KV storage is already linear in context length. A target of O(N log N) storage would be worse asymptotically than O(N). The useful goal is bounded GPU residency and smaller bytes per stored token, together with subquadratic attention work.

Exact arbitrary recall of N independent records cannot generally be guaranteed with sublinear total information storage. Lossy summaries, compressed recurrent states and retrieval restrictions can reduce memory, but each changes what the model can recover. Clearly distinguish an exact archived-KV mode from a compressed-summary mode.

Use the tiers as follows.

| Tier | Contents | Residency policy |
| --- | --- | --- |
| M0 GPU local state | Local attention windows and always-resident state | Fixed window budget; account for physical rolling-buffer capacity |
| M1 GPU global state | Sinks, hot global KV pages, selected-key workspace and bounded index pages | Fixed memory budget with measured allocator overhead; weights budgeted separately |
| M2 host RAM | Quantized full-history KV, most index structures and transfer buffers | Grows with context; explicit page and bandwidth accounting |
| M2 extension SSD | Cold archive pages and cold index partitions | Optional implementation after RAM mode is validated; report tail latency |
| M3 episodic side memory | Versioned semantic state with bounded rank/shard count | Separate quality and forgetting tests; frozen during a request |

The existing host archive should not be described as an implemented CPU-to-NVMe pipeline until disk page layout, asynchronous I/O, eviction and failure recovery actually exist. ESM is also not a lossless replacement for arbitrary source tokens.

A 5M context configuration needs a storage plan as well as a VRAM plan. Even ideal 4-bit global KV can exceed a 64 GiB host machine once model/index/workspace costs are included. Evaluate a RAM-plus-SSD archive or multiple hosts only after measuring physical page traffic. Capacity arithmetic alone does not establish usable interactive latency.

A hierarchy with bounded beam and selected-key budget can aim for logarithmic candidate search plus bounded selected attention per query. Total archived KV and exact index storage remain linear. Approximate recall must be evaluated as the hierarchy grows; an asymptotic search formula does not guarantee retrieval quality.

## 9 Benchmark protocol and full evaluation matrix

### Mandatory result fields

Every run should record source commit and dirty state, hashes of imported runtime files, model revision, mHC mode and checkpoint, selector revision and training status, gain parameterization, attention backend, dtype, GPU name and memory, host RAM, context length, output budget, sampling settings, memory snapshot, dataset revision, split, row IDs, exclusions, harness revision and grader.

A task record must distinguish completed and scored, unsupported protocol, missing asset, context overflow, generation error, grader error, timeout and output truncation. Missing or unsupported scores remain null. A suite is complete only after the declared data stream is exhausted and every eligible item has a valid result. Do not hide skipped rows inside a denominator.

The new full runner flushes a record after each example, refuses to overwrite existing predictions, and stores pinned dataset identities. An explicit limit marks the run as a subset even if all selected examples pass. It does not compute a macro average across incomparable metrics.

### Required benchmark families

| Family | Correct evaluation | Current integration boundary |
| --- | --- | --- |
| Runtime and cache | Actual 72-visit counters, causality, cache replay and checkpoint parity | CPU/tiny-model regression tests; GPU parity still required |
| Decode performance | Warmup, fresh caches, repeated trials, TTFT, decode and end-to-end rates | Runnable benchmark; reported 34 tok/s not reproduced here |
| GraphWalks | Full raw prompt, declared BFS filter, official final-line extraction and set F1 | Native pinned full-data adapter |
| MRCR | Full conversation, specified needle files, prefix plus content similarity | Native pinned full-data adapter |
| LongBench v2 | Official prompt/parser with declared generation/context policy | Native full-data adapter; protocol variants remain labeled |
| IFEval and IFBench | Complete input files and upstream constraint checkers | External graders required |
| Math and GPQA | Correct splits, option randomization, sampling, weighted rubrics and answer checking | Full upstream protocol integration required |
| HLE and AA evaluations | Correct assets, modality scope, judge and abstention rules | External graders/assets required |
| BFCL and tau2 | Real function calls or environment trajectories, full category/domain coverage | Official harnesses required |
| SWE suites | Agent produces patches against base commits; official repository tests grade them | OMP connection enabled; full suite environment adapter remains required |
| SciCode | Complete scientific tasks, data fixtures and executable tests | Upstream harness required |
| Terminal Bench 4 | Harbor dataset terminal-bench/terminal-bench@4.0.0 with real task environments | OMP installed-agent adapter and sandbox resources required |
| GDPval | Actual deliverables with task assets and the designated judging protocol | External evaluation workflow required |

The current Terminal-Bench 4.0 documentation uses Harbor’s 4.0.0 release. A Hugging Face mirror named terminal-bench-2.0 is not evidence that the 4.0 benchmark ran. Some tasks require GPUs in their sandbox, independently of the GPU serving Spark. [S15]

SWE-bench Verified is evaluated by applying generated patches and executing tests in the specified repository environments. Preserve the model_patch artifacts and the official harness report. Neither textual similarity to a gold patch nor synthetic single-function repair is a substitute. [S9]

### Performance experiment design

Use prompt lengths 512, 2K, 8K and 32K first. Generate up to 512 tokens and report the actual count. Record early EOS rather than pretending 512 tokens were produced. If a fixed-length throughput experiment deliberately ignores EOS, label that sampling change separately from natural generation.

Use at least one warmup and three measured trials for each condition. Report median request-level throughput, pooled tokens divided by pooled time, and variability. Record inter-token p50, p95 and p99, particularly across 64-token page closure and 512-token clustering boundaries.

TTFT is request start to the first available token. Steady decode uses the intervals after that first token. End-to-end throughput includes prefill. Do not average rates across wildly different prompt/output lengths without showing the underlying trial records.

Keep dense and sparse models at the same recurrence, memory state, generation settings and selected output task. Test native 36-layer only as a diagnostic baseline. A 4K prompt may never exercise sparse selection; include a horizon above the actual exact-attention threshold.

### Quality acceptance targets

Pre-register acceptance targets before the full run. Suggested starting engineering targets are no unexplained cache-parity failure; no unintended train/eval recurrence mismatch; no frozen-backbone mutation; no observed KV gradient loss in the intended training graph; and no missing task silently scored.

For model quality, use confidence intervals and per-domain results. A suggested development target is at most 0.02–0.05 nats/token held-out NLL regression from the chosen dense reference and no material deterioration in instruction following or executed coding tasks. The appropriate tolerance depends on dataset variance and the intended deployment; these are proposed gates, not measured results.

Require a demonstrated benefit from P3 on at least one held-out reasoning or coding measure. All 72 layers can execute while extra-pass gains have collapsed toward zero. Useful recurrence is a behavioral property, not just a layer counter.

For D-LSA, evaluate attention-mass retention by visit, output error and end-to-end tasks. Average recall alone can hide a failing late global visit. Compare flat/tree and each codec with identical budgets and report failures across context positions.

## 10 Oh My Pi coding harness

Pin the actual Oh My Pi release used for evaluation. This review uses version 18.18.1 and verifies the distributed binary before executing harness checks. The upstream editor’s current patch protocol should be exercised through OMP itself, not reproduced by a custom two-character hash test. [S16]

Serve the actual Loop72 model through the new OpenAI-compatible bridge. It must expose model identity, accept structured tool definitions, apply Spark’s native chat template, convert tool-call argument strings to native dictionaries for replay, return structured tool calls, and process matching tool results.

Streaming must emit valid event-stream chunks and finish reasons. A function-call response should produce a tool_calls finish reason, not a text-only stop. Distinguish internal reasoning from final answer text. Enforce prompt-plus-output context limits, serialize access to mutable model state, and stop generation on cancellation or disconnect.

Use a fresh isolated working directory for each coding task. OMP should read the repository, edit files, run commands or tests and return its result. An independent verifier evaluates the resulting filesystem or patch. Successful transport and successful code repair are separate results.

Begin with a small engineering fixture: a deterministic bug, a visible failing test, one required code change and an independent final test. Record the actual OMP process version, configuration, model endpoint identity, RPC events, tool calls, generated patch, exit status and verifier result.

A scripted backend can validate that the real OMP process performs read/edit/test calls through the server. It cannot validate Spark’s coding ability. Label such a run as a protocol fixture. The first model-backed run must use the actual mHC checkpoint and compare failure modes against the native and fixed-highway baselines.

For a full SWE-bench campaign, add an OMP task runner that checks out each base_commit, exposes only allowed task information, gathers the final diff and exports the official prediction schema. Run the official Docker harness afterward. For Terminal-Bench, build an installed-agent adapter for Harbor so OMP executes inside the task environment and the benchmark’s verifier remains authoritative.

Measure resolved tasks, task wall time, model time, tool time, TTFT, output tokens, number of tool calls, invalid-call rate, test invocations, retries and peak context. Decode tokens per second is one component of useful agent performance.

Keep endpoint access controlled. A notebook server bound to localhost is reachable from the same host, not automatically from the user’s laptop. If using Colab, establish an explicit controlled connection and preserve request provenance. Do not expose an unauthenticated public coding endpoint.

## 11 Ordered implementation backlog

### Milestone 1 runtime and evaluation correctness

Finish the explicit recurrence and checkpoint contract, paged-training guard, uncached visit identity, benchmark fail-closed behavior and OMP protocol bridge. Run CPU unit tests and tiny-model gradient/parity tests. On the GPU, load the actual model and run a short native-versus-looped comparison. Exit only when the manifest matches observed execution.

### Milestone 2 mHC training implementation

Add a dedicated train_spark_mhc.py entry point with a validated training config, bounded-gain versioning, frozen-base parameter audit, teacher retention loss, token accounting, checkpoint export and deterministic validation. Reuse existing model helpers only after their modes and attention graph have been verified. Avoid routing this through the much larger MoE-Engram/SI training chain.

The trainer must export controllers, settings, recurrence mode, gain parameterization, base revision, tokenizer revision, data manifests, optimizer/scheduler state when useful, and a summary of retention and coding validation. A checkpoint without this contract is not a release candidate.

### Milestone 3 short mHC pilot

Run the 0.5–2M token activation/retention pilot and a small 5–20M token reasoning/coding experiment. Compare zero versus small-positive extra gains, static versus dynamic mixing, and rank 8 versus rank 16 adapters. Select one configuration using held-out quality and training time. Do not run every possible combination at large scale.

### Milestone 4 D-LSA calibration and distillation

Freeze the selected mHC checkpoint. Capture visit-correct dense targets, fit maps on calibration data and distill on training data. Validate 8K, 32K and then 128K contexts with untouched final test examples. Export selectors with a behavior fingerprint that rejects incompatible model/memory revisions.

### Milestone 5 serving and memory optimization

Apply measured fusion, stable decode buffers, batched paging and optional quantization. Re-run exactness tests for exact optimizations and quality tests for approximations. Report trained-mHC performance separately from the fixed-highway baseline.

### Milestone 6 model-backed agent evaluation

Run the OMP engineering fixture, a development coding set, then the selected full official coding suite. Save trajectories and independent test outcomes. Expand to Harbor and other environment-based suites once their adapters and sandbox dependencies are working.

### Milestone 7 longer context release gates

Qualify 256K, 512K and 1M. Begin 2M/5M work only with a measured host-memory, index and transfer budget. Require meaningful long-context retrieval and reasoning results, not just successful allocation or repeated-text prefill.

The order matters. Training selectors before stabilizing mHC creates stale targets. Quantizing before establishing BF16 parity creates ambiguous failures. Publishing benchmark numbers before correcting graders obscures whether any architectural change helped.

## 12 Release checklist and remaining inputs

A release candidate must include the pinned base and code revisions, complete mHC state, selector state, runtime identity, native activation contract, gain parameterization, context claim, cache format, memory policy, tested dependency versions, and exact benchmark commands.

The required evidence is: real 72-visit execution, learned controllers active in inference, gradients through the intended training graph, frozen-base invariance, save/load parity, no benchmark answer leakage, honest skipped/error accounting, independent coding tests, and repeated GPU performance measurements.

The remaining operational inputs for a real full model run are the GPU/runtime to execute on, the exact checkpoint directory or model endpoint, and the generation settings that produced 34 tokens per second. The source review and CPU checks do not fill those gaps.

The recommended first usable deliverable is a trained dense 72-visit mHC checkpoint with an honest short-context and coding report. The next deliverable adds validated D-LSA and bounded GPU cache residency. The 5M target remains a later research milestone whose quality, positional behavior and I/O costs must be demonstrated.

## Sources

S1. Reviewed runtime source: https://github.com/BaiTian6641/Spark-X2.6-Recursion/blob/0ebce51cae69195cf6a51c1e270a67206550e0b7/sparkx_loop_v03_core.py

S2. Reviewed loader source: https://github.com/BaiTian6641/Spark-X2.6-Recursion/blob/0ebce51cae69195cf6a51c1e270a67206550e0b7/load_inference_v03.py

S3. Reviewed benchmark source: https://github.com/BaiTian6641/Spark-X2.6-Recursion/blob/0ebce51cae69195cf6a51c1e270a67206550e0b7/sparkx_loop72_dlsa_benchmarks.py

S4. Reviewed notebook: https://github.com/BaiTian6641/Spark-X2.6-Recursion/blob/0ebce51cae69195cf6a51c1e270a67206550e0b7/SparkX_Loop72_DLSA_Benchmark_Colab.ipynb

S5. Visit-aware selector source: https://github.com/BaiTian6641/Spark-X2.6-Recursion/blob/0ebce51cae69195cf6a51c1e270a67206550e0b7/sparkx_dlsa.py

S6. Paged attention and cache source: https://github.com/BaiTian6641/Spark-X2.6-Recursion/blob/0ebce51cae69195cf6a51c1e270a67206550e0b7/sparkx_sparse_attention.py and https://github.com/BaiTian6641/Spark-X2.6-Recursion/blob/0ebce51cae69195cf6a51c1e270a67206550e0b7/sparkx_paged_cache.py

S7. MRCR protocol: https://huggingface.co/datasets/openai/mrcr/blob/f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d/README.md

S8. GraphWalks protocol and corrected dataset: https://huggingface.co/datasets/openai/graphwalks/blob/be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7/README.md

S9. SWE-bench evaluation: https://www.swebench.com/SWE-bench/guides/evaluation/

S10. Xie et al. mHC Manifold-Constrained Hyper-Connections: https://arxiv.org/html/2512.24880v2

S11. McLeish et al. Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence: https://arxiv.org/abs/2511.07384

S12. Official D-LSA implementation and distillation guide: https://github.com/Prefrontal-AI/latent-sparse-attention

S13. PyTorch activation checkpointing: https://docs.pytorch.org/docs/2.15/checkpoint.html

S14. PyTorch CUDA graph constraints and compilation: https://docs.pytorch.org/docs/2.15/notes/cuda.html and https://docs.pytorch.org/docs/2.15/generated/torch.compile.html

S15. Terminal-Bench 4.0 run protocol: https://www.tbench.ai/run

S16. Oh My Pi release and provider implementation: https://github.com/can1357/oh-my-pi/releases/tag/v18.18.1 and https://github.com/can1357/oh-my-pi/tree/24439066040e006eb08da7b204f0ac41c79d2de9

S17. Original Spark checkpoint and tokenizer: https://huggingface.co/XHToken/Spark-X2.6-4B/tree/64ca8f2a295b8ceb2757b9d4749a6558c4759e5b

S18. LongBench v2 prompt and parser: https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py

S19. IFEval and IFBench official graders: https://github.com/google-research/google-research/tree/master/instruction_following_eval and https://github.com/allenai/IFBench
