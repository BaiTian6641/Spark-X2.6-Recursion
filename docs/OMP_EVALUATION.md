# Spark Loop72 with Oh My Pi

This integration serves the loaded `SparkXLoopV03` object through native Spark
chat templates and `generate_cached`. It retains its configured 72 block visits,
18 global visits, attention backend, controller mode, selector state, and cache
policy. It does not load a second base model or silently replace the model with
a generic inference engine.

## Evidence and version pins

The harness targets **Oh My Pi v18.18.1**, source commit
`24439066040e006eb08da7b204f0ac41c79d2de9`.
Its official Linux x64 standalone asset has SHA-256
`2a75e16550676783b3bc8be0e06513478dd5881efc647954642ab30db8f9be06`.
The standalone binary was downloaded, hash verified, and executed during this
change. The SDK route requires Bun; the standalone binary does not require a
separate Bun installation.

CPU protocol tests and an actual four-turn OMP RPC session passed against an
explicitly **scripted** backend: OMP read a file, applied a native hashline edit,
ran the public Python tests with its bash tool, and settled the session. A
separate verifier checked 150 deterministic cases, an invalid interval, and a
degenerate interval. The original buggy fixture failed that same verifier.
This proves HTTP/SSE, tool history, RPC, and real OMP editing integration. It is
not a Spark coding score, a throughput measurement, or long-context evidence.

The official Spark tokenizer template was checked at Hugging Face revision
`64ca8f2a295b8ceb2757b9d4749a6558c4759e5b`. It emits native
`<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`
markup, and requires historical function arguments to be dictionaries. The
bridge converts OpenAI JSON argument strings back to dictionaries and supplies
the offered tool schemas to the template. Template errors fail explicitly.

## Start the endpoint with the loaded model

Install the small protocol dependencies in the same environment as the model:

```bash
python -m pip install -r requirements-omp.txt
```

From the notebook or Python process that already owns `model` and `tokenizer`:

```python
from sparkx_omp_server import create_loop72_server, generate_omp_models_yaml

model.eval()
server = create_loop72_server(
    model, tokenizer,
    host="127.0.0.1", port=8000,
    context_limit_tokens=32768,  # Explicit serving budget, not a quality claim.
    max_output_tokens=2048,
    request_timeout_seconds=900,
)

config = generate_omp_models_yaml(
    "results/omp-models-example.yml",
    context_window=server.context_limit_tokens,
    max_tokens=2048,
)
print(config["omp_cli_example"])
```

The example context budget is clamped to the runtime's configured cache limit.
The server counts the fully rendered native prompt and reserves the requested
output budget. A request that exceeds the total limit is rejected before
generation; history is never silently truncated. Declaration of any context
window is not evidence that the model uses it accurately or fits the hardware.

`GET /v1/health` exposes the execution manifest and actual serving budgets.
Inspect `effective_recurrence_mode`, `mhc_executed`, the base-model revision,
and `selector_checkpoint_status` before comparing runs. The current native
GELU highway and learned mHC are distinct behaviors even though both execute
72 block visits. Loaded selectors remain quality-unverified until evaluated.
The endpoint requires evaluation mode so its manifest matches generation.
The bridge explicitly disables budget-injected thinking-close tokens with
`force_think_close=False` and records this generation policy. Native generated
thinking markers and ordinary EOS remain part of the model output.

`generate_omp_models_yaml()` with no path returns an example without writing a
file. An explicit existing path merges other providers and rejects replacement
of the same provider unless `replace_provider=True` is supplied. A newly
generated example is not automatically installed into your OMP configuration.
The evaluation command below uses a fresh private agent directory for each case.

To stop the background server:

```python
server.shutdown()
server.server_close()
```

## Run the real agent evaluation

Install or obtain the pinned OMP executable from the official release. Check its
platform-specific release checksum; the hash above is only for Linux x64. The
harness verifies the reported version and records the executable's SHA-256.

```bash
python tools/omp_eval.py --omp /path/to/omp --probe
python tools/omp_eval.py --omp /path/to/omp \
  --base-url http://127.0.0.1:8000/v1 \
  --output results/omp-spark-run-001 \
  --thinking off --max-tokens 2048 --max-turns 12 --max-seconds 180
```

Use a fresh output directory for every run. Missing OMP or an unavailable
endpoint produces a `blocked` readiness report. Failed cases produce exit code
1; an unavailable prerequisite produces exit code 2. The endpoint must identify
itself as the actual Spark Loop72 backend for a model evaluation.

Three bundled **local integration cases** cover interval clamping, chunk
remainders, and merging unsorted touching intervals. Each has public tests and
private checks outside the editable workspace. Success requires an actual
read/edit/bash workflow, a changed implementation, unchanged public tests,
private checks passing, and a terminal successful OMP session. A tool argument
or hashline-looking string alone never counts as an edit.

Outputs include `report.json`, per-case results, the final workspace, exact
invocation, full RPC events, stderr, baseline/private verification, observed
tool calls, and model messages with reported usage. RPC v2 chunked frames are
validated and reconstructed. The harness waits for `prompt_result` plus session
settlement, rather than treating an intermediate `agent_end` as completion.
It disables autonomous compaction/model switching in its private config and
sets thinking after selecting the model, because selection can reset effort.

Run model-generated code in a disposable container or VM. A private directory
and cleaned agent environment are not an operating-system sandbox. This
harness deliberately does not modify the user's existing OMP files. It enables
only `read`, `edit`, and `bash` for its small temporary coding workspaces; the
real-binary protocol probe also verifies that exact offered tool set.

These cases are not SWE-bench, Terminal-Bench, BFCL, or an official OMP benchmark.
`official_benchmark_score` stays null. A Spark run can report its local task
success rate; a scripted protocol run cannot report model success or speed.

## Reproduce CPU and actual-harness protocol checks

```bash
python -m unittest discover -s tests -p test_omp_protocol.py -v
python tools/omp_protocol_probe.py --omp /path/to/omp \
  --output results/omp-scripted-protocol-001
```

The second command launches real OMP and its real tools, but all model responses
and token usage are explicitly synthetic. Its `model_quality_evaluated` is
false, `local_task_success_rate` is null, and `protocol_fixture_passed` is the
relevant outcome. It uses the current OMP four-hex **file snapshot** hashline
format. The old two-hex line-tag toy helper is not an OMP implementation.

## Protocol coverage and limits

| Area | Implemented behavior |
|---|---|
| Streaming | Real generation callbacks feed SSE content/reasoning deltas, validated tool calls, finish reason, optional usage, and `[DONE]`. Heartbeats preserve idle connections. |
| Tool schemas | Function schemas validated with JSON Schema; external references, images/audio, and constrained `strict:true` are rejected. |
| Native replay | OpenAI tool argument JSON becomes native dictionaries; all tool-call IDs must match. Parallel results are reordered because Spark's template omits IDs in rendered tool results. |
| Tool choice | `auto`, `none`, `required`, and named calls; prompt conditioning plus output validation. This is not constrained decoding. Invalid/incomplete calls are errors. |
| Thinking | Native `enable_thinking` and `reasoning_effort=low/high`. OMP's configurable chat-template wire dialect supplies these fields; no claim that Spark is a Qwen model. |
| Isolation | One generation at a time; overlapping requests receive 429. Each request starts a fresh visit-aware KV cache. |
| Cancellation | Disconnects/timeouts signal `should_stop`; cancellation occurs between model steps. A long kernel cannot be interrupted mid-step. |
| Unsupported fields | Unknown fields, `stop`, `response_format`, penalties not implemented by the runtime, and `n>1` fail explicitly. No silently ignored controls. |
| Exposure | Default loopback listener. A non-loopback listener requires an explicit API key; the bundled harness examples use unauthenticated loopback only. |

The bridge does not provide cross-turn prefix KV reuse, batching, a production
scheduler, persistent memory compilation, or an external tool-execution sandbox.
Fresh-cache multi-turn prefill can dominate agent latency; measure model
prefill, decode, transport, and tool time separately before optimizing it.

## Primary sources

- [OMP v18.18.1 release](https://github.com/can1357/oh-my-pi/releases/tag/v18.18.1)
- [Pinned model/provider configuration](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md)
- [Pinned RPC specification](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/rpc.md)
- [Pinned OpenAI streaming provider](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/providers/openai-completions.ts)
- [Pinned hashline format](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/edit/hashline-compact.md)
- [Spark's native template](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/64ca8f2a295b8ceb2757b9d4749a6558c4759e5b/chat_template.jinja)
