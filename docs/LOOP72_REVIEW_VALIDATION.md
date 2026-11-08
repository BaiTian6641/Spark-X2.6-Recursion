# Loop72 review validation record

Review date: 8 November 2026. Upstream branch: `loop72-dlsa`. Reviewed upstream commit: `0ebce51cae69195cf6a51c1e270a67206550e0b7`. This record describes the review changes, not a new trained-model release.

## Executed checks

| Check | Result | What it establishes |
| --- | --- | --- |
| Full repository CPU unit suite | **220 tests passed**, 35.806 seconds, no skips | Python/runtime behavior covered by existing and added fixtures |
| Changed Python sources and notebook code cells | AST parsing passed | Syntax and notebook code validity |
| Tracked patch whitespace check | `git diff --check` passed | No tracked whitespace errors |
| Real OMP v18.18.1 protocol fixture | Passed, four agent turns | Actual RPC, HTTP/SSE, read/edit/bash and native editor integration |
| Independent coding fixture verifier | Passed after edit; original buggy implementation failed | The scripted protocol exercised a real file change and runnable verification |
| Full model benchmark command | Exit 2 at preflight | Missing CUDA and datasets dependency are reported; no model score fabricated |

The CPU environment used Python `3.12.14`, Torch `2.5.1+cpu`, Transformers `4.57.1`, Accelerate `1.15.0`, JSON Schema and PyYAML. Torch and model-loading dependencies were installed into a separate test environment. No Spark weights were downloaded or loaded. Loader download boundaries were stubbed in unit tests.

The 220-test result includes tiny-model forward/backward tests and actual callback-to-HTTP streaming tests. It is not a 4B model run. The changed test fixture for BF16 cold storage now compares original BF16 values exactly; it no longer demands impossible losslessness relative to unrounded FP32 inputs.

## Runtime coverage

The tests cover explicit recurrence modes, 72 decoder visits and 60 controller executions in learned mode, nonzero P2/P3 controller gradients, frozen backbone gradients, checkpoint-replay gradient/context parity, save/load state, embedded/external selector restoration, native differentiable loading, inference-only paged-storage guards, all 18 global visit identities, tail-logit preservation, scalar-cache invalidation, generation callback/stop behavior, and optional disabling of forced thinking-close tokens.

The review introduces an evaluation scalar cache for the selector distance prior. CPU tests verify invalidation after parameter mutation, loading, dtype and train/eval changes. No GPU speed improvement has been measured.

## Benchmark coverage

Tests reproduce and reject answer leakage, shortened MRCR contexts, prefix-only MRCR credit, incorrect GraphWalks node parsing, unknown constraint success, missing SciCode tests, fake repository/terminal/deliverable grading, and mixed synthetic/official comparison scores. They exercise the real full-runner converter/evaluator interfaces with a mocked generation backend.

The native full runner provides pinned complete-data adapters for GraphWalks BFS, MRCR eight-needle and LongBench v2. The remaining suites require their upstream graders/environments and return explicit blocked requirements. A user-specified row limit cannot become a full score. A numeric result requires completed scoring. Dataset interruptions, context overflow and unsupported rows prevent full completion.

No full dataset generation occurred in this review. The attempted command was:

```bash
python -m sparkx_full_benchmark --output-dir results/full-attempt
```

Its preflight reported `cuda_available=false`, `datasets=false`, `model_benchmark_executed=false` and `complete=false`. Transformers and Accelerate were available. The output folder is illustrative here; the review used an isolated temporary run directory.

## Actual OMP protocol evidence

| Field | Observed value |
| --- | --- |
| OMP version | `18.18.1` |
| Source commit | `24439066040e006eb08da7b204f0ac41c79d2de9` |
| Linux x64 binary SHA256 | `2a75e16550676783b3bc8be0e06513478dd5881efc647954642ab30db8f9be06` |
| Backend | `scripted_protocol_test` |
| Workflow | `read` → `edit` → `bash`, then completed reply |
| Tool errors | 0 |
| Implementation changed | true |
| Public tests unchanged | true |
| Independent checks | 150 deterministic cases plus invalid and degenerate intervals passed |
| RPC completion | `prompt_result.status=completed`, `sessionSettled=true` |
| Protocol fixture | passed |
| Model quality evaluated | false |
| Local Spark task success rate | null |
| Official benchmark score | null |

The protocol uses the actual OMP four-hex file-snapshot edit format. All generated responses and usage counts in the scripted probe are synthetic. Its elapsed time is not Spark latency. The private workspace/config provides task separation; it is not an operating-system sandbox.

## Unexecuted release gates

No GPU inference profiling, reproduction of approximately 34 tok/s, real 4B gradient-parity run, mHC training, D-LSA training against the new mHC checkpoint, cache codec quality comparison, complete official model benchmark, model-backed OMP coding run, or useful 1M/5M context validation occurred.

The bounded-gain schema and `train_spark_mhc.py` are planned next work. Learned mHC fusion, physical KV accounting repairs, asynchronous host/SSD paging and cross-turn prefix cache reuse also remain future work. The current `auto` recurrence mode intentionally preserves old behavior; new training and evaluation should select an explicit mode.

See [the detailed plan](roadmap/09_LOOP72_MHC_TRAINING_AND_VALIDATION.md), [the runbook](roadmap/10_LOOP72_RUNBOOK.md) and [OMP instructions](OMP_EVALUATION.md) for the next execution steps.
