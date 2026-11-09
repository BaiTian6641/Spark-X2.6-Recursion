"""Pinned, complete-dataset evaluation with explicit incomplete/blocked results.

This runner never calls the sampled notebook loader and never substitutes a toy
task for an official suite.  Three long-context datasets have native adapters;
the remaining suites require their own upstream execution/grading harnesses.
An exhausted stream is necessary, but not sufficient, for a complete result.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable


MODEL_REVISION = "64ca8f2a295b8ceb2757b9d4749a6558c4759e5b"
REVIEWED_SOURCE = "0ebce51cae69195cf6a51c1e270a67206550e0b7"


@dataclasses.dataclass(frozen=True)
class DatasetProtocol:
    name: str
    benchmark_name: str
    dataset: str
    revision: str
    files: tuple[str, ...]
    split: str = "train"
    row_filter: str | None = None


NATIVE_PROTOCOLS = {
    "graphwalks_bfs": DatasetProtocol(
        "graphwalks_bfs", "Graphwalks bfs 256k f1", "openai/graphwalks",
        "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7",
        ("graphwalks_128k_and_shorter.parquet", "graphwalks_256k_to_1mil.parquet"),
        row_filter="problem_type == bfs; all published lengths",
    ),
    "mrcr_8needle": DatasetProtocol(
        "mrcr_8needle", "Openai_MRCR v2 256k-512k", "openai/mrcr",
        "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d",
        ("8needle/8needle_0.parquet", "8needle/8needle_1.parquet"),
        row_filter="8needle files; all published lengths",
    ),
    "longbench_v2": DatasetProtocol(
        "longbench_v2", "LongBench v2", "zai-org/LongBench-v2",
        "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9", ("data.json",),
    ),
}

# These are requirements, not fake adapters.  The original notebook assigned
# official names to single custom examples; no such substitution is allowed.
EXTERNAL_PROTOCOLS = {
    "ifeval": ("https://github.com/google-research/google-research/tree/master/instruction_following_eval", "Generate every input prompt and run evaluation_main for strict and loose scores."),
    "ifbench": ("https://github.com/allenai/IFBench", "Use the complete IFBench test file and the upstream run_eval checker."),
    "gaokao_2026": ("https://huggingface.co/datasets/XHToken/gaokao_math_2026", "Pin the release, question weights, partial-credit rubric and generation protocol; percent accuracy is not a score out of 150."),
    "aime_2026": ("https://matharena.ai/", "Use the pinned MathArena dataset, extraction and repeat protocol."),
    "hmmt_feb_2026": ("https://matharena.ai/", "Use the pinned MathArena dataset, extraction and repeat protocol."),
    "imo_answerbench": ("https://huggingface.co/datasets/Hwilner/imo-answerbench", "Use the published answer equivalence checker and full release."),
    "gpqa_main": ("https://github.com/idavidrein/gpqa", "Use gpqa_main with fixed option randomization, seed, prompt and sampling protocol."),
    "gpqa_diamond": ("https://github.com/idavidrein/gpqa", "Use gpqa_diamond with fixed option randomization, seed, prompt and sampling protocol."),
    "hle_text": ("https://github.com/centerforaisafety/hle", "Use the official judge, record the text-only scope and exclude image rows explicitly."),
    "aa_lcr_v1_1": ("https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR", "Load the referenced document assets and the official grader; missing assets must block the task."),
    "aa_omniscience_v1_1": ("https://artificialanalysis.ai/evaluations/omniscience", "Pin v1.1 dataset and its actual answerability and abstention scoring protocol."),
    "bfcl_v4": ("https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard", "Run official V4 categories and AST/execution/multiturn evaluation."),
    "tau2_bench": ("https://github.com/sierra-research/tau2-bench", "Pin release, domains, simulator and judge; execute real tool/environment transitions."),
    "mcp_atlas": ("https://huggingface.co/datasets/ScaleAI/MCP-Atlas", "Run real MCP tools and the official evaluator without exposing hidden claims."),
    "swe_verified": ("https://www.swebench.com/SWE-bench/guides/evaluation/", "Use OMP rollouts at each base_commit, export model_patch, and run swebench.harness.run_evaluation in Docker."),
    "swe_pro": ("https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro", "Run the matching SWE Pro repository environments and tests."),
    "swe_multilingual": ("https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual", "Run language-specific repository environments and official tests."),
    "scicode": ("https://github.com/scicode-bench/SciCode", "Run complete main/subproblem tasks with the scientific data fixtures and upstream tests."),
    "terminal_bench_4": ("https://www.tbench.ai/run", "Use Harbor dataset terminal-bench/terminal-bench@4.0.0 and an OMP installed-agent adapter; GPU tasks need an appropriate sandbox."),
    "gdpval": ("https://github.com/openai/gdpval", "Provide the task assets, create real deliverables, and use the designated evaluation protocol."),
}


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    paths = ["sparkx_loop_v03_core.py", "sparkx_dlsa.py", "sparkx_sparse_attention.py",
             "sparkx_paged_cache.py", "load_inference_v03.py", "sparkx_full_benchmark.py",
             "sparkx_loop72_dlsa_benchmarks.py"]
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "reviewed_upstream_commit": REVIEWED_SOURCE,
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths if (root / p).exists()},
    }


def preflight(suites: Iterable[str] | None = None) -> dict[str, Any]:
    names = list(suites or [*NATIVE_PROTOCOLS, *EXTERNAL_PROTOCOLS])
    unknown = set(names) - set(NATIVE_PROTOCOLS) - set(EXTERNAL_PROTOCOLS)
    if unknown:
        raise ValueError(f"Unknown suites: {sorted(unknown)}")
    torch_info: dict[str, Any] = {"installed": False, "cuda_available": False}
    try:
        import torch
        torch_info = {"installed": True, "version": torch.__version__, "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            torch_info.update(gpu=torch.cuda.get_device_name(0), total_vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    except ImportError:
        pass
    deps = {n: importlib.util.find_spec(n) is not None for n in ("transformers", "datasets", "accelerate")}
    native_blockers = []
    if not torch_info["cuda_available"]:
        native_blockers.append("No CUDA GPU is available in this runtime")
    native_blockers.extend(f"Missing dependency: {n}" for n, ok in deps.items() if not ok)
    rows = []
    for name in names:
        if name in NATIVE_PROTOCOLS:
            rows.append({"suite": name, "status": "blocked" if native_blockers else "ready_to_load_model",
                         "score": None, "reasons": list(native_blockers),
                         "dataset_protocol": dataclasses.asdict(NATIVE_PROTOCOLS[name])})
        else:
            url, requirement = EXTERNAL_PROTOCOLS[name]
            rows.append({"suite": name, "status": "external_harness_required", "score": None,
                         "source": url, "reason": requirement})
    return {"mode": "preflight", "model_benchmark_executed": False, "complete": False,
            "python": platform.python_version(), "torch": torch_info, "dependencies": deps,
            "source": source_identity(), "suites": rows}


def load_pinned_rows(spec: DatasetProtocol) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(spec.dataset, split=spec.split, revision=spec.revision,
                      data_files={spec.split: list(spec.files)}, streaming=True)
    for row in ds:
        yield dict(row)


def _accept_row(spec: DatasetProtocol, row: dict[str, Any]) -> bool:
    if spec.name == "graphwalks_bfs":
        if "problem_type" not in row:
            raise ValueError("GraphWalks row has no problem_type; cannot select BFS safely")
        return row["problem_type"] == "bfs"
    return True


def run_dataset(
    model: Any, tokenizer: Any, spec: DatasetProtocol, output_dir: str | Path,
    *, max_new_tokens: int = 32768, limit: int | None = None,
    generation_options: dict[str, Any] | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    converter: Callable[..., dict[str, Any] | None] | None = None,
    verbose: bool | None = None,
) -> dict[str, Any]:
    """Evaluate every eligible row; ``limit`` always makes the run a subset.

    Optional injected rows/evaluator are for unit tests.  They are recorded and
    can never produce a model-quality claim.  Results are flushed per example.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    test_inputs = rows is not None or evaluator is not None or converter is not None
    if verbose is None:
        verbose = not test_inputs and hasattr(model, "generate_cached")
    if evaluator is None or converter is None:
        from sparkx_loop72_dlsa_benchmarks import evaluate_benchmark_task, convert_hf_row_to_eval_task
        evaluator = evaluator or evaluate_benchmark_task
        converter = converter or convert_hf_row_to_eval_task
    import inspect as _inspect
    try:
        _eval_accepts_verbose = "verbose" in _inspect.signature(evaluator).parameters
    except Exception:
        _eval_accepts_verbose = False
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / f"{spec.name}.jsonl"
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory; refusing to mix runs: {output}")
    options = {"do_sample": False, "temperature": 0.0, "top_p": 1.0,
               "top_k": -1, "repetition_penalty": 1.0, "force_think_close": False}
    options.update(generation_options or {})
    metadata = {"suite": spec.name, "dataset_protocol": dataclasses.asdict(spec),
                "generation": {**options, "max_new_tokens": max_new_tokens},
                "limit": limit, "injected_test_inputs": test_inputs,
                "source": source_identity(),
                "model_manifest": _jsonable(model.execution_manifest()) if hasattr(model, "execution_manifest") else None,
                "model_metadata": _jsonable(getattr(model, "loaded_meta", None)),
                "published_score_comparable": False,
                "comparison_note": "Full dataset coverage does not establish equivalence to the model-card protocol. No macro comparison is computed."}
    (dest / f"{spec.name}.manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    counts = {"rows_seen": 0, "excluded_by_declared_filter": 0, "attempted": 0,
              "scored": 0, "blocked": 0, "errors": 0}
    scores: list[float] = []
    ended = False
    suite_error = None
    started = time.perf_counter()
    stream = iter(rows if rows is not None else load_pinned_rows(spec))
    with output.open("w", encoding="utf-8") as handle:
        try:
            while limit is None or counts["attempted"] < limit:
                try:
                    row = next(stream)
                except StopIteration:
                    ended = True
                    break
                index = counts["rows_seen"]
                counts["rows_seen"] += 1
                if not _accept_row(spec, row):
                    counts["excluded_by_declared_filter"] += 1
                    continue
                counts["attempted"] += 1
                record: dict[str, Any] = {"row_index": index, "row_sha256": stable_digest(row),
                                          "source_id": row.get("_id", row.get("id"))}
                row_t0 = time.perf_counter()
                if verbose:
                    print(
                        f"  [{spec.name} | row #{counts['attempted']} (seen={counts['rows_seen']}, "
                        f"excl={counts['excluded_by_declared_filter']}, id={record['source_id']})] "
                        f"starting evaluation (max_new_tokens={max_new_tokens})...",
                        flush=True,
                    )
                try:
                    task = converter(spec.benchmark_name, row, row_idx=index, task_mode="official")
                    if task is None:
                        result = {"status": "unsupported", "score": None, "reason": "No valid official task adapter for this row"}
                    else:
                        # The suite scope is all published lengths, never an inferred 256K/1M label.
                        task = dict(task)
                        task["reference_adapter_key"] = task.get("benchmark_id", spec.benchmark_name)
                        task["benchmark"] = spec.name
                        task["benchmark_id"] = spec.name
                        task["official_spark_x26_score"] = None
                        task["official_comparable"] = False
                        eval_kwargs: dict[str, Any] = {
                            "max_new_tokens": max_new_tokens,
                            "generation_options": options,
                        }
                        if _eval_accepts_verbose and verbose:
                            eval_kwargs["verbose"] = True
                        result = evaluator(model, tokenizer, task, **eval_kwargs)
                    if result.get("score") is not None:
                        if result.get("status") != "completed":
                            raise ValueError("A numeric score requires status='completed'")
                        score = float(result["score"])
                        if not (0.0 <= score <= 100.0):
                            raise ValueError(f"Invalid score for native protocol: {score}")
                        scores.append(score)
                        counts["scored"] += 1
                    elif result.get("status") in ("error", "generation_error", "invalid"):
                        counts["errors"] += 1
                    else:
                        counts["blocked"] += 1
                    record.update(_jsonable(result))
                except Exception as exc:
                    counts["errors"] += 1
                    record.update(status="error", score=None, reason=f"{type(exc).__name__}: {exc}")
                if verbose:
                    row_dt = time.perf_counter() - row_t0
                    timing = record.get("timing") or {}
                    run_mean = (sum(scores) / len(scores)) if scores else None
                    mean_str = f"{run_mean:.2f}" if run_mean is not None else "N/A"
                    sc_val = record.get("score")
                    sc_str = f"{float(sc_val):.2f}" if sc_val is not None else str(sc_val)
                    p_tok = record.get("prompt_tokens", timing.get("prompt_tokens", "?"))
                    g_tok = record.get("generated_tokens", timing.get("generated_tokens", "?"))
                    dec_tps = float(timing.get("decode_tok_per_s", 0.0) or 0.0)
                    ttft_ms = float(timing.get("ttft_ms", 0.0) or 0.0)
                    reason_sfx = f" | reason={record.get('reason')}" if record.get("reason") else ""
                    print(
                        f"  [{spec.name} | row #{counts['attempted']} done] status={record.get('status')} | "
                        f"score={sc_str} | running_mean={mean_str} (scored={counts['scored']}/{counts['attempted']}) | "
                        f"prompt={p_tok} tok | gen={g_tok} tok | decode={dec_tps:.1f} tok/s | "
                        f"TTFT={ttft_ms:.0f}ms | wall={row_dt:.2f}s{reason_sfx}",
                        flush=True,
                    )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        except Exception as exc:
            suite_error = f"{type(exc).__name__}: {exc}"
    full_coverage = ended and limit is None and counts["attempted"] > 0 and counts["blocked"] == counts["errors"] == 0
    mean = sum(scores) / len(scores) if scores else None
    result = {"suite": spec.name, "status": "complete" if full_coverage else "incomplete",
              "complete": full_coverage, "dataset_exhausted": ended, **counts,
              "score": mean if full_coverage else None, "score_on_scored_rows": mean,
              "intended_model_run": not test_inputs,
              "model_quality_run": not test_inputs and counts["scored"] > 0,
              "published_score_comparable": False,
              "suite_error": suite_error, "elapsed_seconds": time.perf_counter() - started,
              "manifest": str(dest / f"{spec.name}.manifest.json"), "predictions": str(output)}
    (dest / f"{spec.name}.summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_full_benchmarks(model: Any, tokenizer: Any, output_dir: str | Path,
                        suites: Iterable[str] | None = None, max_new_tokens: int = 32768,
                        limit: int | None = None, verbose: bool = True) -> dict[str, Any]:
    """Run supported datasets and enumerate every remaining blocked suite."""
    names = list(suites or [*NATIVE_PROTOCOLS, *EXTERNAL_PROTOCOLS])
    unknown = set(names) - set(NATIVE_PROTOCOLS) - set(EXTERNAL_PROTOCOLS)
    if unknown:
        raise ValueError(f"Unknown suites: {sorted(unknown)}")
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"mode": "full" if limit is None else "subset", "complete": False,
                              "macro_score": None, "suites": []}
    native_names = [n for n in names if n in NATIVE_PROTOCOLS]
    native_idx = 0
    for name in names:
        if name in NATIVE_PROTOCOLS:
            native_idx += 1
            spec = NATIVE_PROTOCOLS[name]
            if verbose:
                print(
                    f"\n=== [FullBench {native_idx}/{len(native_names)}] Starting Native Suite '{name}' "
                    f"(dataset={spec.dataset}@{spec.revision[:8]}, max_new_tokens={max_new_tokens}, limit={limit}) ===",
                    flush=True,
                )
            try:
                entry = run_dataset(model, tokenizer, spec, dest,
                                    max_new_tokens=max_new_tokens, limit=limit, verbose=verbose)
            except Exception as exc:
                entry = {"suite": name, "status": "blocked", "score": None,
                         "reason": f"{type(exc).__name__}: {exc}", "complete": False}
            if verbose:
                sc_mean = entry.get("score_on_scored_rows")
                sc_str = f"{float(sc_mean):.2f}" if sc_mean is not None else "N/A"
                print(
                    f"=== [FullBench {native_idx}/{len(native_names)}] Suite '{name}' Finished: "
                    f"status={entry.get('status')} | attempted={entry.get('attempted', 0)} | "
                    f"scored={entry.get('scored', 0)} | mean_score={sc_str} | "
                    f"elapsed={float(entry.get('elapsed_seconds', 0.0) or 0.0):.1f}s ===",
                    flush=True,
                )
        else:
            url, why = EXTERNAL_PROTOCOLS[name]
            entry = {"suite": name, "status": "external_harness_required", "score": None,
                     "reason": why, "source": url, "complete": False}
        result["suites"].append(entry)
        result["complete"] = all(x.get("complete", False) for x in result["suites"]) and len(result["suites"]) == len(names)
        (dest / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--suites", nargs="+", default=None, choices=[*NATIVE_PROTOCOLS, *EXTERNAL_PROTOCOLS])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="XHToken/Spark-X2.6-4B")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dlsa-checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--recurrence-mode", choices=["learned_mhc", "depth_matched_highway"],
                        help="Explicit runtime identity; omit only to preserve the saved checkpoint/default mode")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=None, help="Explicit subset; cannot be reported as a full run")
    args = parser.parse_args(argv)
    report = preflight(args.suites)
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.preflight:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if any(x["status"] == "blocked" for x in report["suites"]):
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    if not any(x["suite"] in NATIVE_PROTOCOLS for x in report["suites"]):
        result = run_full_benchmarks(None, None, dest, args.suites, args.max_new_tokens, args.limit)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    from load_inference_v03 import load_sparkx_loop72_lsa
    model, tokenizer = load_sparkx_loop72_lsa(repo_or_dir=args.model_dir,
                                            revision=args.model_revision,
                                            config_path=args.config,
                                            dlsa_checkpoint=args.dlsa_checkpoint,
                                            recurrence_mode=args.recurrence_mode)
    result = run_full_benchmarks(model, tokenizer, dest, args.suites, args.max_new_tokens, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
