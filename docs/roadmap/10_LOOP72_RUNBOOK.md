# Spark Loop72 training and evaluation runbook

This runbook accompanies [the mHC training and validation plan](09_LOOP72_MHC_TRAINING_AND_VALIDATION.md). Commands run from the repository root. Use a fresh results directory for every model/configuration. The code review provides training-path repairs, benchmark adapters and the OMP bridge; it does not provide a newly trained mHC checkpoint or a completed GPU benchmark.

## 1 Prepare and identify the environment

Install a CUDA-enabled PyTorch build appropriate to the GPU using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the repository dependencies in that same environment. The tested CPU fixture environment used Torch 2.5.1 and Transformers 4.57.1. That is evidence for CPU regression tests, not a supported CUDA performance configuration. Freeze the actual GPU package versions after the first successful run.

```bash
python -m pip install -r requirements.txt -r requirements-omp.txt
python -m unittest discover -s tests -v
python -m sparkx_full_benchmark --preflight --output-dir results/preflight-001
```

The preflight reports CUDA availability, dependencies, source hashes, dataset revisions and missing official harnesses. Exit zero from `--preflight` means the report was generated; inspect its suite statuses before loading the model. A normal benchmark invocation exits 2 if required native prerequisites are unavailable or evaluation is incomplete.

Record the GPU model, usable VRAM, host RAM, PCIe link, driver, Torch/CUDA/Transformers versions, code commit, exact launch command and source dirty state. Preserve the existing approximately 34 tok/s result as a user-observed baseline until reproduced with these fields.

## 2 Load an explicit model identity

The training and inference objects should be separate fresh loads. A paged inference object cannot be converted into a differentiable training model merely by calling `train()`.

```python
import json
from pathlib import Path
from load_inference_v03 import load_sparkx_loop72_lsa
from sparkx_full_benchmark import MODEL_REVISION

# Replace with a local export containing loop_v03_state.pt after training.
checkpoint_dir = "/path/to/local/mhc-export"
selector_checkpoint = "/path/to/visit-selectors.pt"

model, tokenizer = load_sparkx_loop72_lsa(
    repo_or_dir=checkpoint_dir,
    revision=MODEL_REVISION,
    recurrence_mode="learned_mhc",
    dlsa_checkpoint=selector_checkpoint,
    attach_sparse_attention=True,
)
model.eval()
manifest = model.execution_manifest()
print(json.dumps(manifest, indent=2, default=str))
Path("results").mkdir(exist_ok=True)
Path("results/runtime-manifest.json").write_text(
    json.dumps(manifest, indent=2, default=str) + "\n"
)
```

Inspect actual 72/18/54 visit counts, native GELU, zero Engram allocation, effective recurrence, `mhc_executed`, base revision, attention runtime and selector status. A checkpoint being loaded is not evidence that it was trained successfully. Save its training provenance and held-out results separately.

For the fixed-highway baseline, load a separate object with `recurrence_mode="depth_matched_highway"`. Do not reuse a result labeled only “72L” across these two modes. The legacy `auto` mode is retained for reproduction and can select different train/eval branches; explicitly choose the mode for every new experiment.

Local controller exports are loaded from `loop_v03_state.pt` or `loop_state.pt`. A remote Hugging Face controller repository name is not currently a controller-download shortcut in this loader. Materialize its export locally before supplying `repo_or_dir`.

## 3 Run the differentiable mHC training pipeline (`train_spark_mhc.py`)

`train_spark_mhc.py` implements Milestone 2 (`docs/roadmap/09_LOOP72_MHC_TRAINING_AND_VALIDATION.md` Sections 4–6):
- **Versioned Bounded-Gain `TwoStreamMHC` Schema v2 (`gain_mode="sigmoid"`, `mhc_schema_version=2`)**: enforces $g = \text{gate\_mult} \cdot \sigma(u) \in [0, 1]$ so $(1 - g)I + gP \in \mathcal{B}_2$ strictly on the Birkhoff polytope, initialized to $g_{\text{P2}} = 0.01$ and $g_{\text{P3}} = 0.005$ across all `10,078,176` controller parameters while keeping `legacy_tanh` (`mhc_schema_version=1`) for legacy checkpoint reproduction.
- **Stage A (Mechanical & Gradient Verification)**: verifies trainable mask (`controllers.` only), frozen backbone SHA-256 invariance, P2/P3/adapter gradients, activation-checkpoint replay parity, and save/load logit identity (`--verify-only`).
- **Stage B (36L $\to$ 72L Teacher Retention Warmup)**: runs the frozen 36L teacher under `torch.no_grad()` first to extract $K=64$ sampled supervised positions per sequence (`[B, K, 131072]`), frees teacher activations before building the 72L student graph, and optimizes `fused_masked_cross_entropy` + full-vocabulary temperature-scaled KL ($\beta=0.5, T=1.5$).
- **Stage C (Multi-Pass Deliberation Training)**: fixed 72-visit student (`--curriculum fixed_72`, or `--curriculum 36_60_72` for curriculum ablation) with staged parameter unfreezing (`warmup_gains_and_up` $\to$ `adapters_and_static_routing` $\to$ `full_controllers`), separate AdamW parameter groups (`adapter_lr=2e-4`, `routing_lr=3e-5`), and the 5-domain mixture (40% code/debugging, 20% general instruction, 15% math, 15% tool trajectories, 10% long-doc/repo). Saves `stage_c_dense_mhc/loop_v03_state.pt` before tool specialization.
- **Stage D (`Oh My Pi` Native Tool-Use Specialization)**: optional polish on native Spark `<tool_call>` (`read`/`edit`/`bash`) trajectories with non-assistant tokens masked to `-100`, saving `stage_d_omp_mhc/loop_v03_state.pt`.
- **Stage E (18-Visit `VisitAwareDLSAIndexBank` Calibration & Provenance Lock)**: captures per-visit Q/K/V under the trained `learned_mhc` recurrence, runs PCA + sampled-query KL distillation across all 18 global visits, and stamps `mhc_checkpoint_sha256`, `recurrence_mode="learned_mhc"`, `mhc_gain_mode="sigmoid"`, and `mhc_schema_version=2` into `visit_selectors_stage_e.pt` and `final_loop72_mhc_dlsa_export/loop_v03_state.pt`.

```bash
# Stage A verification only (Colab G4 96GB auto-detected)
python -m train_spark_mhc \
  --hardware-profile colab_g4_96gb \
  --verify-only \
  --output-dir results/mhc-verify-001

# Full Stages A–E training, 18-visit D-LSA calibration & automatic upload of "Spark X2.6 4B Recursive72" to Hugging Face Hub
python -m train_spark_mhc \
  --hardware-profile colab_g4_96gb \
  --gain-mode sigmoid \
  --init-p2-gain 0.01 \
  --init-p3-gain 0.005 \
  --curriculum fixed_72 \
  --stage-b-steps 100 \
  --stage-c-steps 300 \
  --stage-d-steps 60 \
  --stage-e-distill-steps 150 \
  --hf-model-name "Spark X2.6 4B Recursive72" \
  --hf-repo-id "Spark X2.6 4B Recursive72" \
  --push-to-hub \
  --output-dir results/mhc-train-001
```

You can also inspect or prepare the training model directly in Python:

```python
from load_inference_v03 import load_sparkx_loop72_lsa
from sparkx_full_benchmark import MODEL_REVISION
from train_spark_mhc import MHCTrainingConfig, prepare_model_for_mhc_training, run_stage_a_verification

train_model, train_tokenizer = load_sparkx_loop72_lsa(
    revision=MODEL_REVISION,
    recurrence_mode="learned_mhc",
    attach_sparse_attention=False,
)
cfg = MHCTrainingConfig(hardware_profile="colab_g4_96gb", gain_mode="sigmoid")
prep = prepare_model_for_mhc_training(train_model, cfg=cfg, reset_gains=True)
print("Trainable parameters:", prep["trainable_parameters"])
stage_a = run_stage_a_verification(train_model, cfg=cfg, seq_len=64)
assert stage_a["passed"], stage_a
```

## 4 Measure inference without changing the model

```python
from sparkx_loop72_dlsa_benchmarks import benchmark_loop72_decode_speed

speed = benchmark_loop72_decode_speed(
    model, tokenizer,
    prompt_lengths=(512, 2048, 8192, 32768),
    gen_tokens=512,
    warmups=1,
    repeats=3,
)
Path("results/speed-001.json").write_text(
    json.dumps(speed, indent=2, default=str) + "\n"
)
```

These synthetic prompts measure runtime, not task quality. Inspect each trial, actual generated tokens, EOS termination, prefill, TTFT, decode rate and end-to-end rate. Errors are retained. A short EOS result is not a 512-token sustained decode measurement. The helper records inter-token p50/p95/p99 in each trial and median horizon summaries. Per-operation transfer/copy breakdowns require a separate profiler or telemetry run.

Compare the current revision with the review patch on the same GPU, checkpoint, recurrence, dtype, prompt lengths and sampling settings. The selector scalar cache is an exact-path optimization implemented here; its GPU speed effect remains unmeasured. Profile learned-mHC routing, redundant cache copies and host page transfers before adding broader compilation or quantization.

## 5 Run pinned complete datasets

The native full runner covers GraphWalks BFS at all published lengths, MRCR eight-needle files at all published lengths, and LongBench v2. It consumes every eligible row unless an explicit limit is given. The exact dataset revisions and file names are constants in `sparkx_full_benchmark.py` and copied to each manifest.

```bash
python -m sparkx_full_benchmark \
  --model-dir /path/to/local/mhc-export \
  --dlsa-checkpoint /path/to/visit-selectors.pt \
  --recurrence-mode learned_mhc \
  --suites graphwalks_bfs mrcr_8needle longbench_v2 \
  --max-new-tokens 32768 \
  --output-dir results/native-full-001
```

Omit `--dlsa-checkpoint` only when the local loop export contains its intended selector bank. A fresh initialized bank must be identified as such. Use `--config` for an explicit context/cache policy and preserve that file with the result.

The output directory contains per-suite manifests, prediction JSONL files, summary files and an overall summary. A complete run requires stream exhaustion, nonzero eligible rows, and no blocked/error/unscored eligible item. Context overflow is a blocked example. An interrupted or context-limited run retains useful partial results but has `score=null` at the full-suite level. Existing predictions are never overwritten. Automatic resumption is not implemented; use a fresh output directory.

Use `--limit 2` only for a small end-to-end smoke exercise. The result remains a subset, even if two examples happen to exhaust a small source. No sample score is copied into an official comparison table.

Full coverage does not establish parity with the published Spark evaluation protocol. In particular, the LongBench prompt/parser is pinned, but the high generation budget, greedy settings and no-truncation policy are a declared protocol variant. `published_score_comparable` remains false.

If `--suites` is omitted, the runner also lists the remaining official suites as `external_harness_required`. It does not execute IFEval, IFBench, math/GPQA, HLE, BFCL, tau2, AA/MCP, SWE, SciCode, Terminal-Bench or GDPval through surrogate scorers. Build each upstream adapter with its assets, revisions, real environment or grader before treating the overall benchmark campaign as complete.

## 6 Exercise increasing context lengths

```python
from sparkx_loop72_dlsa_benchmarks import run_full_length_1m_context_benchmark

capacity_and_load = run_full_length_1m_context_benchmark(
    model, tokenizer,
    horizons=(32768, 131072, 262144, 524288, 1048576),
    max_live_tokens=131072,
    gen_tokens=512,
)
```

The historical function name is retained for compatibility. This call executes only load probes allowed by `max_live_tokens`, lists higher horizons as not run, and always keeps `quality_validated=False`. Ensure prompt plus requested output fits the actual cache/context budget. Capacity arithmetic cannot validate retrieval or reasoning. Promote a context length only after real task-quality checks at that length and across document positions.

## 7 Run Oh My Pi

Follow [OMP_EVALUATION.md](../OMP_EVALUATION.md) for the pinned v18.18.1 executable, checksum, native template protocol and server controls. From the process that owns the already loaded model:

```python
from sparkx_omp_server import create_loop72_server

server = create_loop72_server(
    model, tokenizer,
    host="127.0.0.1", port=8000,
    context_limit_tokens=32768,
    max_output_tokens=2048,
)
```

Then run the actual agent in a separate terminal. The loopback example requires the server and OMP to share a network namespace; one option is to run both inside the same disposable container. A separate container with default networking cannot reach the host's `127.0.0.1`. If separate environments are required, explicitly configure a reachable authenticated endpoint. Keep tool execution confined to the intended task workspace and execution environment.

```bash
python tools/omp_eval.py --omp /path/to/omp --probe
python tools/omp_eval.py --omp /path/to/omp \
  --base-url http://127.0.0.1:8000/v1 \
  --output results/omp-model-001 \
  --thinking off --max-tokens 2048 --max-turns 12 --max-seconds 180
```

The bundled local coding tasks require real read/edit/bash actions, an implementation change, unchanged public tests and independent verification. They are integration tasks, not SWE-bench or Terminal-Bench. Increase the timeout explicitly if actual model prefill/decode needs it, and retain the limit as benchmark metadata.

To reproduce transport validation without model weights:

```bash
python tools/omp_protocol_probe.py --omp /path/to/omp \
  --output results/omp-scripted-protocol-001
```

This uses a scripted backend. Its relevant field is `protocol_fixture_passed`; it cannot report model quality, local Spark task success rate or speed. The model-backed endpoint is identified separately in health and evaluation reports.

For SWE-bench, preserve each task's base commit, use OMP to produce a `model_patch`, and feed patches to the [official Docker evaluator](https://www.swebench.com/SWE-bench/guides/evaluation/). For Terminal-Bench 4, implement an OMP installed-agent adapter for the [Harbor workflow](https://www.tbench.ai/run). Those full environment adapters remain work to complete.

## 8 Collect the release evidence

Deliver source and checkpoint hashes, runtime manifests, package locks, immutable data manifests, raw predictions, per-example statuses, independent coding verifier reports, repeated speed trials, physical VRAM/host memory measurements, and the exact tested context ceiling. Keep native 36-layer, fixed-highway 72-visit and trained-mHC 72-visit results in separate rows.

The first release gate is a stable trained-mHC model with a useful measured short-context capability. Quantization, longer context and persistent memory updates require their own later gates. Five million context tokens remain an experimental capacity and quality target.
