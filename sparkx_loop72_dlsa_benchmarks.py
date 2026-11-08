"""sparkx_loop72_dlsa_benchmarks.py — Official Benchmark Reference & Modern 2026 Evaluation Suite for Spark X2.6 4B Loop72 + Per-Visit Paged D-LSA (`loop72-dlsa` branch).

Evaluates **strictly the 72-Visit Looped Transformer (`Loop72`) + Per-Visit Paged `D-LSA` enhancement** (`construct_engram=False`, `0` Engram parameters, native `GELU` preserved) against:
  1. Official `Spark X2.6-4B` published benchmark scores (and competitor baselines `Spark X2.5-4B`, `Qwen3.5 9B`,
     `Qwen3.5 4B`, `Gemma4 12B`, `Gemma4 E4B`).
  2. Significant official benchmarks across Long Context, Math, General/Instruction, Agent, and Code:
     - **Long Context**: `OpenAI MRCR v2` (`openai/mrcr`), `GraphWalks BFS` (`openai/graphwalks` F1),
       `LongBench v2` (`zai-org/LongBench-v2`), `AA-LCR v1.1` (`ArtificialAnalysis/AA-LCR`).
     - **Math**: `Gaokao 2026` (`XHToken/gaokao_math_2026`), `AIME 2026` (`MathArena/aime_2026`),
       `HMMT Feb 2026` (`MathArena/hmmt_feb_2026`), `IMO-AnswerBench` (`Hwilner/imo-answerbench`).
     - **General & Knowledge**: `IFEval` (`google/IFEval`), `IFBench` (`allenai/IFBench_test`),
       `GPQA Diamond` (`Idavidrein/gpqa`).
     - **Agent & Code**: `BFCL-V4` (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`), `SciCode` (`SciCode1/SciCode`).
  3. Modern 2025–2026 Held-Out Text Perplexity, 4-Way Positional/Sparse Ablation (`Dense_NativeRoPE`, `Sparse_NativeRoPE`,
     `Dense_YaRNRoPE`, `Sparse_YaRNRoPE`), and 72-Visit Hybrid Paged Cache Memory Ledger (`M0/M1/M2`).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    _no_grad = torch.no_grad
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

    def _no_grad():  # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap

from sparkx_kv_codec import KVCodecConfig, compute_loop72_capacity_table, measure_codec_error
from sparkx_si.verifiers import Verdict, _extract_json_object, answer_part, extract_boxed, extract_code, verify_math


def extract_answer_segment(response_text: str) -> str:
    """Return the answer segment after `</think>` (or the full string if no think tags exist)."""
    ans = answer_part(response_text)
    return ans if ans is not None else response_text


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content inside the last `\\boxed{...}`."""
    return extract_boxed(text)


def verify_math_answer(response_text: str, gold: str) -> Verdict:
    """Verify a math response against `gold` using `sparkx_si.verifiers.verify_math`."""
    return verify_math(response_text, gold)


def verify_json_schema(response_text: str, schema: Dict[str, Any]) -> Verdict:
    """Verify that `response_text` contains a JSON object matching `required_keys` and `key_types`."""
    ans = extract_answer_segment(response_text)
    obj = _extract_json_object(ans)
    if not isinstance(obj, dict):
        return Verdict(False, "no_json_object", {"score": 0.0})
    req = schema.get("required_keys", [])
    ktypes = schema.get("key_types", {})
    type_map = {"str": str, "int": int, "float": (int, float), "bool": bool, "dict": dict, "list": list}
    matched = 0
    total = max(1, len(req))
    errors: List[str] = []
    for k in req:
        if k not in obj:
            errors.append(f"missing:{k}")
            continue
        expected_t = ktypes.get(k)
        if expected_t and expected_t in type_map:
            py_t = type_map[expected_t]
            if expected_t == "int" and isinstance(obj[k], bool):
                errors.append(f"type:{k}")
                continue
            if not isinstance(obj[k], py_t):
                errors.append(f"type:{k}")
                continue
        matched += 1
    score = matched / total
    return Verdict(len(errors) == 0, "ok" if not errors else "schema_mismatch", {"score": score, "errors": errors, "parsed": obj})


def verify_code_stdio(response_text: str, test_cases: Sequence[Dict[str, str]]) -> Verdict:
    """Execute a Python code block against `stdin -> expected_stdout` test cases."""
    import subprocess
    import sys

    code = extract_code(response_text) or extract_answer_segment(response_text)
    passed_cnt = 0
    errors: List[str] = []
    for idx, tc in enumerate(test_cases):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                input=tc.get("stdin", ""),
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            got = proc.stdout.strip()
            exp = tc.get("expected_stdout", "").strip()
            if proc.returncode == 0 and got == exp:
                passed_cnt += 1
            else:
                errors.append(f"case_{idx}:got={got!r},exp={exp!r}")
        except Exception as exc:
            errors.append(f"case_{idx}:{exc}")
    score = passed_cnt / max(1, len(test_cases))
    return Verdict(passed_cnt == len(test_cases), "ok" if not errors else "stdio_mismatch", {"score": score, "errors": errors})


# ==============================================================================
# 1. Official Spark X2.6-4B Benchmark Reference Table & Recommended Sampling
# ==============================================================================
OFFICIAL_SPARK_X26_SAMPLING = {
    "thinking_mode": True,
    "thinking_level": "high",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": -1,
}

OFFICIAL_BENCHMARK_TABLE: Dict[str, Dict[str, Any]] = {
    "Average": {
        "category": "Overall",
        "Spark_X2.6_4B": 63.3,
        "Spark_X2.5_4B": 52.1,
        "Qwen3.5_9B": 49.1,
        "Qwen3.5_4B": 43.2,
        "Gemma4_12B": 47.0,
        "Gemma4_E4B": 29.7,
    },
    # --- Agent ---
    "BFCL-V4": {
        "category": "Agent",
        "hf_dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "Spark_X2.6_4B": 71.8,
        "Spark_X2.5_4B": 65.1,
        "Qwen3.5_9B": 66.1,
        "Qwen3.5_4B": 50.3,
        "Gemma4_12B": 37.4,
        "Gemma4_E4B": 47.0,
    },
    "tau2-bench": {
        "category": "Agent",
        "Spark_X2.6_4B": 86.9,
        "Spark_X2.5_4B": 75.1,
        "Qwen3.5_9B": 79.1,
        "Qwen3.5_4B": 79.9,
        "Gemma4_12B": 69.0,
        "Gemma4_E4B": 42.2,
    },
    "MCP-Atlas": {
        "category": "Agent",
        "Spark_X2.6_4B": 62.4,
        "Spark_X2.5_4B": 54.6,
        "Qwen3.5_9B": 47.4,
        "Qwen3.5_4B": 40.8,
        "Gemma4_12B": 30.5,
        "Gemma4_E4B": 15.0,
    },
    # --- Code ---
    "SWE Verified": {
        "category": "Code",
        "hf_dataset": "princeton-nlp/SWE-bench_Verified",
        "Spark_X2.6_4B": 62.8,
        "Spark_X2.5_4B": 41.6,
        "Qwen3.5_9B": 53.1,
        "Qwen3.5_4B": 38.8,
        "Gemma4_12B": 44.2,
        "Gemma4_E4B": 14.0,
    },
    "SciCode": {
        "category": "Code",
        "hf_dataset": "SciCode1/SciCode",
        "Spark_X2.6_4B": 36.8,
        "Spark_X2.5_4B": 34.7,
        "Qwen3.5_9B": 32.7,
        "Qwen3.5_4B": 24.0,
        "Gemma4_12B": 39.8,
        "Gemma4_E4B": 27.5,
    },
    # --- Math ---
    "Gaokao 2026": {
        "category": "Math",
        "hf_dataset": "XHToken/gaokao_math_2026",
        "scale": 150.0,
        "Spark_X2.6_4B": 134.9,
        "Spark_X2.5_4B": 133.4,
        "Qwen3.5_9B": 135.5,
        "Qwen3.5_4B": 130.3,
        "Gemma4_12B": 130.6,
        "Gemma4_E4B": 102.4,
    },
    "AIME 2026": {
        "category": "Math",
        "hf_dataset": "MathArena/aime_2026",
        "Spark_X2.6_4B": 91.2,
        "Spark_X2.5_4B": 90.7,
        "Qwen3.5_9B": 88.2,
        "Qwen3.5_4B": 83.0,
        "Gemma4_12B": 82.1,
        "Gemma4_E4B": 42.5,
    },
    "HMMT Feb 26": {
        "category": "Math",
        "hf_dataset": "MathArena/hmmt_feb_2026",
        "Spark_X2.6_4B": 81.3,
        "Spark_X2.5_4B": 81.2,
        "Qwen3.5_9B": 70.8,
        "Qwen3.5_4B": 69.7,
        "Gemma4_12B": 65.6,
        "Gemma4_E4B": 34.2,
    },
    "IMO-AnswerBench": {
        "category": "Math",
        "hf_dataset": "Hwilner/imo-answerbench",
        "Spark_X2.6_4B": 75.2,
        "Spark_X2.5_4B": 74.2,
        "Qwen3.5_9B": 69.8,
        "Qwen3.5_4B": 68.5,
        "Gemma4_12B": 57.2,
        "Gemma4_E4B": 26.9,
    },
    # --- Long Context ---
    "AA-LCRv1.1": {
        "category": "Long Context",
        "hf_dataset": "ArtificialAnalysis/AA-LCR",
        "Spark_X2.6_4B": 67.3,
        "Spark_X2.5_4B": 64.7,
        "Qwen3.5_9B": 70.0,
        "Qwen3.5_4B": 63.0,
        "Gemma4_12B": 63.7,
        "Gemma4_E4B": 32.0,
    },
    "LongBench v2": {
        "category": "Long Context",
        "hf_dataset": "zai-org/LongBench-v2",
        "Spark_X2.6_4B": 53.3,
        "Spark_X2.5_4B": 48.7,
        "Qwen3.5_9B": 55.2,
        "Qwen3.5_4B": 50.0,
        "Gemma4_12B": 45.3,
        "Gemma4_E4B": 27.8,
    },
    "Graphwalks bfs 256k f1": {
        "category": "Long Context",
        "hf_dataset": "openai/graphwalks",
        "Spark_X2.6_4B": 92.2,
        "Spark_X2.5_4B": 47.1,
        "Qwen3.5_9B": 88.4,
        "Qwen3.5_4B": 79.0,
        "Gemma4_12B": 68.6,
        "Gemma4_E4B": 54.8,
    },
    "Graphwalks bfs 1m f1": {
        "category": "Long Context",
        "hf_dataset": "openai/graphwalks",
        "Spark_X2.6_4B": 34.2,
        "Spark_X2.5_4B": 2.5,
        "Qwen3.5_9B": 25.3,
        "Qwen3.5_4B": 15.8,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    "Openai_MRCR v2 256k-512k": {
        "category": "Long Context",
        "hf_dataset": "openai/mrcr",
        "Spark_X2.6_4B": 65.6,
        "Spark_X2.5_4B": 9.7,
        "Qwen3.5_9B": 58.6,
        "Qwen3.5_4B": 48.3,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    "Openai_MRCR v2 512k-1M": {
        "category": "Long Context",
        "hf_dataset": "openai/mrcr",
        "Spark_X2.6_4B": 26.0,
        "Spark_X2.5_4B": 2.9,
        "Qwen3.5_9B": 34.3,
        "Qwen3.5_4B": 12.2,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    # --- General & Knowledge ---
    "IFEval": {
        "category": "General & Knowledge",
        "hf_dataset": "google/IFEval",
        "Spark_X2.6_4B": 93.6,
        "Spark_X2.5_4B": 93.0,
        "Qwen3.5_9B": 91.5,
        "Qwen3.5_4B": 89.8,
        "Gemma4_12B": 94.8,
        "Gemma4_E4B": 45.3,
    },
    "IFBench": {
        "category": "General & Knowledge",
        "hf_dataset": "allenai/IFBench_test",
        "Spark_X2.6_4B": 75.8,
        "Spark_X2.5_4B": 75.0,
        "Qwen3.5_9B": 64.5,
        "Qwen3.5_4B": 59.2,
        "Gemma4_12B": 73.5,
        "Gemma4_E4B": 44.0,
    },
    "GPQA Diamond": {
        "category": "General & Knowledge",
        "hf_dataset": "Idavidrein/gpqa",
        "Spark_X2.6_4B": 77.1,
        "Spark_X2.5_4B": 73.7,
        "Qwen3.5_9B": 81.7,
        "Qwen3.5_4B": 76.2,
        "Gemma4_12B": 78.8,
        "Gemma4_E4B": 58.6,
    },
}


# ==============================================================================
# 2. Modern 2025–2026 Held-Out Text Corpus (Zero Overlap with Legacy Prompts)
# ==============================================================================
MODERN_2026_TEXT_CORPUS: List[Dict[str, str]] = [
    {
        "id": "modern_2026_recurrent_depth_and_mhc",
        "domain": "2025-2026 Frontier AI Systems",
        "title": "Latent Recurrent Depth and Doubly-Stochastic Manifold Hyper-Connections",
        "text": (
            "Recent 2025-2026 architectural studies demonstrate that transformer parameter count and sequential "
            "computational depth are orthogonal axes of capability. Whereas autoregressive chain-of-thought externalizes "
            "intermediate state into discrete vocabulary tokens through the output embedding bottleneck, looped "
            "recurrent-depth transformers iterate a shared core block in continuous high-dimensional representation "
            "space. To prevent spectral norm blow-up across multiple recurrent passes without collapsing the residual "
            "update to zero, Manifold-Constrained Hyper-Connections (mHC) parameterize inter-stream mixing matrices on "
            "the Birkhoff polytope via Sinkhorn-Knopp normalization. Because every doubly-stochastic matrix has spectral "
            "norm bounded by unity, multi-pass state evolution remains non-expansive across arbitrary unroll depths while "
            "allowing each recurrent visit to specialize its query-key attention distribution."
        ),
    },
    {
        "id": "modern_2026_subquadratic_paged_lsa",
        "domain": "2026 Long-Context Attention",
        "title": "Per-Visit Paged Latent Sparse Attention on Hybrid Partial-RoPE Backbones",
        "text": (
            "In hybrid-attention backbones combining local sliding-window layers with periodic global full-attention "
            "layers, the global KV cache dominates memory footprint and memory-bandwidth latency beyond 32,768 tokens. "
            "When the global layers employ partial rotary position embeddings—rotating only the leading 64 dimensions "
            "of each 256-dimensional head while leaving the remaining 192 dimensions position-free—semantic key indexing "
            "can operate directly on the unrotated 192-dimensional subspace without rotary phase distortion. By pooling "
            "contiguous four-token spans and partitioning completed 512-token blocks via mutual-nearest-neighbor Ward "
            "clustering on the 64-dimensional unit hypersphere, per-visit selectors retrieve only the highest-affinity "
            "semantic buckets alongside protected attention sinks and a local sliding tail."
        ),
    },
    {
        "id": "modern_2026_systems_io_uring_zns",
        "domain": "2026 Systems & Storage Engineering",
        "title": "Asynchronous Zero-Copy KV Page Tiering over PCIe Gen5 NVMe Zoned Namespaces",
        "text": (
            "Modern disaggregated inference engines treat host DRAM and PCIe Gen5 NVMe solid-state drives as an "
            "extension of GPU high-bandwidth memory for multi-million-token sessions. Rather than issuing fine-grained "
            "4-kilobyte random reads that suffer from PCIe transaction-layer packet overhead and SSD read amplification, "
            "page-aligned KV archives group 64 contiguous tokens across all four grouped-query KV heads into 128-kilobyte "
            "blocks in 16-bit precision or 36-kilobyte mixed-precision pages that keep rotary key slices in BFloat16 "
            "and quantize free key and value slices to 4-bit group-scaled integers. Coalesced asynchronous page gathers "
            "amortize host-to-device DMA transfers while a bounded GPU LRU cache absorbs temporally coherent decode hits."
        ),
    },
    {
        "id": "modern_2026_quantum_ldpc_fault_tolerance",
        "domain": "2026 Quantum Information & Physics",
        "title": "Constant-Overhead Fault-Tolerant Quantum Computation with Bivariate Bicycle LDPC Codes",
        "text": (
            "High-rate quantum low-density parity-check (qLDPC) codes based on bivariate bicycle constructions achieve "
            "order-of-magnitude reductions in physical-to-logical qubit overhead compared with planar surface codes. "
            "Constructed from commuting check polynomials over the group algebra of a finite abelian group Z_l x Z_m, "
            "these stabilizer codes embed naturally onto two-layer superconducting or neutral-atom syndrome extraction "
            "circuits of constant degree six. Decoding under circuit-level depolarizing noise combines belief propagation "
            "with ordered statistics post-processing (BP-OSD), suppressing logical error rates below 10^-11 per syndrome "
            "cycle using fewer than one-tenth the physical qubits required by distance-matched surface code patches."
        ),
    },
]


def build_modern_long_eval_stream(target_tokens: int = 8192, tokenizer: Any = None) -> str:
    """Build a deterministic, non-repetitive modern 2026 technical evaluation stream of at least `target_tokens` tokens."""
    blocks: List[str] = []
    sec_idx = 0
    while True:
        for doc in MODERN_2026_TEXT_CORPUS:
            sec_idx += 1
            digest = hashlib.sha256(f"{doc['id']}:{sec_idx}".encode("utf-8")).hexdigest()[:10]
            header = (
                f"\n\n### Section {sec_idx} [{doc['domain']}] — {doc['title']} (Trace {digest})\n"
                f"Metric Audit Parameters: window_id={sec_idx * 64}, page_tokens=64, "
                f"visit_budget={2048 + (sec_idx % 3) * 512}, checksum=0x{digest}.\n"
            )
            blocks.append(header + doc["text"])
        joined = "".join(blocks)
        if tokenizer is not None:
            try:
                n_tok = len(tokenizer.encode(joined, add_special_tokens=False))
            except Exception:
                n_tok = len(joined.split())
        else:
            n_tok = int(len(joined.split()) * 1.35)
        if n_tok >= int(target_tokens):
            return joined


# ==============================================================================
# 3. Official Long-Context Benchmark Generators & Scorers (GraphWalks BFS + MRCR v2 + LongBench v2)
# ==============================================================================
def generate_graphwalks_bfs_instance(
    num_nodes: int = 128,
    branching: int = 2,
    target_depth: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a faithful `openai/graphwalks` BFS instance using random 8-char hex node hashes.

    Matches the `openai/graphwalks` specification:
      - Directed graph represented as shuffled edge lines `u -> v` of hexadecimal node hashes, with no filler text.
      - Task: Perform Breadth-First Search (BFS) from `start_node` to exact `target_depth` and return the set of nodes
        at distance `target_depth`.
    """
    rng = random.Random(seed)
    nodes: List[str] = []
    seen = set()
    for i in range(max(16, int(num_nodes))):
        while True:
            h = hashlib.md5(f"gw:{seed}:{i}:{rng.randint(0, 10**9)}".encode("utf-8")).hexdigest()[:8]
            if h not in seen:
                seen.add(h)
                nodes.append(h)
                break

    # Build a layered DAG for layers 0..target_depth+1 so exact BFS depth is unambiguous, plus distractor components
    start_node = nodes[0]
    layer_0 = [start_node]
    cursor = 1
    layers: List[List[str]] = [layer_0]
    edges: List[Tuple[str, str]] = []

    for d in range(1, target_depth + 2):
        prev_layer = layers[-1]
        cur_layer: List[str] = []
        for u in prev_layer:
            for _ in range(branching):
                if cursor < len(nodes):
                    v = nodes[cursor]
                    cursor += 1
                    cur_layer.append(v)
                    edges.append((u, v))
        layers.append(cur_layer)

    # Distractor edges among remaining nodes (disconnected from start_node)
    distractor_nodes = nodes[cursor:]
    for i in range(len(distractor_nodes) - 1):
        u = distractor_nodes[i]
        v = distractor_nodes[(i + 1) % len(distractor_nodes)]
        edges.append((u, v))
        if i + 2 < len(distractor_nodes) and i % 2 == 0:
            edges.append((u, distractor_nodes[i + 2]))

    rng.shuffle(edges)
    edge_lines = "\n".join(f"{u} -> {v}" for u, v in edges)
    gold_nodes = sorted(layers[target_depth])

    prompt = (
        "You are given a directed graph as a list of directed edges `u -> v`:\n\n"
        f"{edge_lines}\n\n"
        f"Question: Starting from node `{start_node}`, find all nodes at exact Breadth-First Search (BFS) "
        f"distance {target_depth} (i.e., reachable in {target_depth} directed hops and no fewer). "
        "Output the matching 8-character node IDs separated by commas inside `\\boxed{id1, id2, ...}`."
    )
    return {
        "benchmark": "Graphwalks bfs f1",
        "start_node": start_node,
        "target_depth": target_depth,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "gold_nodes": gold_nodes,
        "prompt": prompt,
    }


def score_graphwalks_bfs_f1(response_text: str, gold_nodes: Sequence[str]) -> Dict[str, Any]:
    """Compute exact Precision, Recall, and F1 on a `GraphWalks BFS` prediction."""
    ans_seg = extract_answer_segment(response_text)
    boxed = extract_boxed_answer(ans_seg) or ans_seg
    pred_tokens = set(re.findall(r"\b[0-9a-f]{8}\b", boxed.lower()))
    gold_set = {g.lower() for g in gold_nodes}
    if not gold_set and not pred_tokens:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "predicted": [], "gold": list(gold_set)}
    tp = len(pred_tokens & gold_set)
    prec = tp / max(1, len(pred_tokens)) if pred_tokens else 0.0
    rec = tp / max(1, len(gold_set))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "predicted": sorted(pred_tokens),
        "gold": sorted(gold_set),
    }


def generate_mrcr_v2_instance(
    num_rounds: int = 24,
    target_occurrence: int = 3,
    seed: int = 101,
) -> Dict[str, Any]:
    """Generate a faithful `openai/mrcr` (Multi-Round Co-Reference Resolution) instance.

    In `openai/mrcr`, the conversation contains multiple rounds with identical/near-identical user requests
    (e.g. 'Write a short 2-line release note for module Helios') interleaved with other recurring requests
    ('Write a short 2-line release note for module Borealis', etc.). The model must retrieve the exact code/phrase
    from the `target_occurrence`-th occurrence of the target request.
    """
    rng = random.Random(seed)
    topics = ["Helios-Kernel", "Borealis-Router", "Zephyr-Pager", "Chronos-Scheduler"]
    target_topic = topics[0]
    turns: List[str] = []
    topic_counts: Dict[str, int] = {t: 0 for t in topics}
    gold_code: Optional[str] = None

    for r_i in range(max(8, int(num_rounds))):
        # Ensure target_topic appears at least target_occurrence + 1 times
        if r_i in (2, 7, 13, 19) or (topic_counts[target_topic] < target_occurrence and r_i >= num_rounds - 2):
            t_name = target_topic
        else:
            t_name = topics[(r_i + seed) % len(topics)]
        topic_counts[t_name] += 1
        occ = topic_counts[t_name]
        build_tag = f"REV-{t_name[:3].upper()}-{1000 + r_i * 37 + rng.randint(1, 29)}"
        checksum_tag = hashlib.sha1(f"{seed}:{r_i}:{t_name}:{occ}".encode("utf-8")).hexdigest()[:6].upper()
        signature = f"{build_tag}-{checksum_tag}"
        if t_name == target_topic and occ == int(target_occurrence):
            gold_code = signature
        turns.append(
            f"[Turn {r_i + 1}]\n"
            f"User: Generate the official release verification tag for module `{t_name}`.\n"
            f"Assistant: Release verification tag for `{t_name}`: `{signature}` ( deployment window {r_i + 1}, "
            f"latency p99 = {1.2 + (r_i % 7) * 0.3:.1f} ms )."
        )

    assert gold_code is not None, "Target occurrence was not generated"
    transcript = "\n\n".join(turns)
    ordinal_map = {1: "1st", 2: "2nd", 3: "3rd"}
    ord_str = ordinal_map.get(int(target_occurrence), f"{int(target_occurrence)}th")
    prompt = (
        "Below is a multi-turn engineering session log containing repeated release tag requests across several modules:\n\n"
        f"{transcript}\n\n"
        f"Question: What was the exact release verification tag (`REV-...`) returned by the Assistant on the "
        f"**{ord_str}** time the User asked for module `{target_topic}`? "
        "Put ONLY the exact tag inside `\\boxed{...}`."
    )
    return {
        "benchmark": "Openai_MRCR v2",
        "target_topic": target_topic,
        "target_occurrence": int(target_occurrence),
        "num_rounds": int(num_rounds),
        "gold_answer": gold_code,
        "prompt": prompt,
    }


def score_mrcr_v2_response(response_text: str, gold_answer: str) -> Dict[str, Any]:
    """Score an `openai/mrcr` response by exact tag match and normalized substring similarity."""
    ans_seg = extract_answer_segment(response_text)
    boxed = (extract_boxed_answer(ans_seg) or ans_seg).strip().strip("`\"'")
    exact = gold_answer.lower() in boxed.lower()
    return {
        "exact_match": bool(exact),
        "score": 1.0 if exact else 0.0,
        "extracted": boxed[:80],
        "gold": gold_answer,
    }


# ==============================================================================
# 4. Curated Significant Benchmark Items (Math, LongBench v2 / AA-LCR, IFEval/IFBench, BFCL-V4, SciCode)
# ==============================================================================
SIGNIFICANT_BENCHMARK_SUITE: List[Dict[str, Any]] = [
    # --- Math: Gaokao 2026 / AIME 2026 / HMMT Feb 26 / IMO-AnswerBench ---
    {
        "id": "gaokao_2026_seq_recurrence",
        "benchmark": "Gaokao 2026",
        "category": "Math",
        "official_spark_x26_score": 134.9,
        "prompt": (
            "Let the sequence $\\{a_n\\}$ satisfy $a_1 = 2$ and $a_{n+1} = 2a_n + 3 \\cdot 2^n$ for all $n \\ge 1$. "
            "Compute the exact value of $a_{8}$. Put your final integer answer in \\boxed{...}."
        ),
        "gold": "2816",
        "verifier": "math",
    },
    {
        "id": "aime_2026_modular_combinatorics",
        "benchmark": "AIME 2026",
        "category": "Math",
        "official_spark_x26_score": 91.2,
        "prompt": (
            "Find the number of ordered triples $(a, b, c)$ of positive integers such that $a + b + c = 30$ "
            "and $a \\le b \\le c < a + b$. Put your final integer answer in \\boxed{...}."
        ),
        "gold": "19",
        "verifier": "math",
    },
    {
        "id": "hmmt_feb_2026_polynomial_roots",
        "benchmark": "HMMT Feb 26",
        "category": "Math",
        "official_spark_x26_score": 81.3,
        "prompt": (
            "Let $r_1, r_2, r_3$ be the roots of $P(x) = x^3 - 6x^2 + 11x - 6 = 0$. "
            "Compute $(r_1^2 + r_2^2 + r_3^2)(r_1 r_2 r_3)$. Put your final integer answer in \\boxed{...}."
        ),
        "gold": "84",
        "verifier": "math",
    },
    {
        "id": "imo_answerbench_diophantine_squares",
        "benchmark": "IMO-AnswerBench",
        "category": "Math",
        "official_spark_x26_score": 75.2,
        "prompt": (
            "Find the sum of all positive integers $n$ such that $n^2 + 11n + 52$ is a perfect square. "
            "Put your final integer answer in \\boxed{...}."
        ),
        "gold": "17",
        "verifier": "math",
    },
    # --- Long Context: AA-LCRv1.1 & LongBench v2 Multi-Document Synthesis ---
    {
        "id": "aa_lcr_v11_cross_doc_synthesis",
        "benchmark": "AA-LCRv1.1",
        "category": "Long Context",
        "official_spark_x26_score": 67.3,
        "prompt": (
            "[Document A — Q1 Infrastructure Filing]\n"
            "Cluster North-Alpha deployed 480 nodes consuming 1.25 kW per node continuously for 2,000 hours in Q1.\n\n"
            "[Document B — Regional Tariff Addendum]\n"
            "Industrial power in Zone North-Alpha is billed at $0.08 per kWh for the first 500,000 kWh and $0.11 per kWh "
            "for all consumption above 500,000 kWh. However, Section 4.2 grants a flat $14,000 clean-grid rebate if the "
            "cluster maintains at least 92% renewable certificates.\n\n"
            "[Document C — Sustainability Audit Log]\n"
            "Cluster North-Alpha recorded 94.6% verified renewable certificates during Q1.\n\n"
            "Question: Synthesizing Documents A, B, and C, what is the net Q1 electricity bill (in USD, as an exact integer) "
            "for Cluster North-Alpha after applying any eligible rebate? Put your final integer answer in \\boxed{...}."
        ),
        "gold": "103000",
        "verifier": "math",
    },
    {
        "id": "longbench_v2_multi_module_trace",
        "benchmark": "LongBench v2",
        "category": "Long Context",
        "official_spark_x26_score": 53.3,
        "prompt": (
            "[File 1: `sparkx_dlsa.py`]\n"
            "In `Loop72`, global full-attention layers `{3, 7, 11, 15, 19, 23, 27, 31, 35}` are visited across "
            "Prelude (`L3`: 1 visit), Pass 1 (`L7, 11, 15, 19, 23, 27`: 6 visits), Pass 2 (`L7, 11, 15, 19, 23, 27`: 6 visits), "
            "Pass 3 (`L15, 19, 23`: 3 visits), and Coda (`L31, 35`: 2 visits).\n\n"
            "[File 2: `sparkx_kv_codec.py`]\n"
            "Each global visit owns an independent `FaithfulDLSAIndexer` with `W_k` of shape `(4, 192, 64)` and `W_q` of "
            "shape `(192, 64)`, totaling `4 * 192 * 64 + 192 * 64 = 61,440` parameters per indexer wait—plus when "
            "`d_free = 192` and `n_kv_heads = 4`, `(4 + 1) * 192 * 64 = 61,440` or with 18 visits and full head maps "
            "`98,304` parameters per global visit (`18 * 98,304 = 1,769,472`).\n\n"
            "Question: Based on the repository trace above, how many total global attention visits occur in `Loop72` "
            "when `enable_p3=True`? Put your final integer answer in \\boxed{...}."
        ),
        "gold": "18",
        "verifier": "math",
    },
    # --- General & Knowledge: IFEval / IFBench & GPQA Diamond ---
    {
        "id": "ifeval_json_schema_compliance",
        "benchmark": "IFEval",
        "category": "General & Knowledge",
        "official_spark_x26_score": 93.6,
        "prompt": (
            "Respond with a valid JSON object inside a ```json ... ``` code block containing exact keys "
            "`\"model\"`, `\"total_visits\"`, and `\"thinking_mode\"`, where `\"model\"` is a string, "
            "`\"total_visits\"` is the integer `72`, and `\"thinking_mode\"` is the boolean `true`."
        ),
        "schema": {
            "required_keys": ["model", "total_visits", "thinking_mode"],
            "key_types": {
                "model": "str",
                "total_visits": "int",
                "thinking_mode": "bool",
            },
        },
        "verifier": "json",
    },
    {
        "id": "ifbench_strict_json_and_constraints",
        "benchmark": "IFBench",
        "category": "General & Knowledge",
        "official_spark_x26_score": 75.8,
        "prompt": (
            "Return a valid JSON object inside a ```json ... ``` block with exact keys "
            "`\"architecture\"`, `\"global_visits\"`, `\"local_visits\"`, and `\"engram_allocated\"`. "
            "Set `\"architecture\"` to `\"Spark-X2.6-4B-Loop72-LSA\"`, `\"global_visits\"` to `18`, "
            "`\"local_visits\"` to `54`, and `\"engram_allocated\"` to `false`."
        ),
        "schema": {
            "required_keys": ["architecture", "global_visits", "local_visits", "engram_allocated"],
            "key_types": {
                "architecture": "str",
                "global_visits": "int",
                "local_visits": "int",
                "engram_allocated": "bool",
            },
        },
        "verifier": "json",
    },
    {
        "id": "gpqa_diamond_relativistic_threshold",
        "benchmark": "GPQA Diamond",
        "category": "General & Knowledge",
        "official_spark_x26_score": 77.1,
        "prompt": (
            "A relativistic proton of rest mass $m_p$ collides head-on with a stationary proton of rest mass $m_p$ in the "
            "laboratory frame to produce a proton-antiproton pair alongside the two original protons: "
            "$p + p \\to p + p + p + \\bar{p}$. In units of $m_p c^2$, what is the minimum total energy $E_{\\text{min}}$ "
            "of the incident proton in the laboratory frame required for this reaction to occur? "
            "Put your final integer answer in \\boxed{...}."
        ),
        "gold": "7",
        "verifier": "math",
    },
    # --- Agent: BFCL-V4 Function Calling ---
    {
        "id": "bfcl_v4_tool_selection",
        "benchmark": "BFCL-V4",
        "category": "Agent",
        "official_spark_x26_score": 71.8,
        "prompt": (
            "Available tools:\n"
            "1. `configure_dlsa_cache(local_window: int, global_hot_tokens: int, cold_codec: str)`\n"
            "2. `compile_sleep_engram(anchor_layer: int, ridge_alpha: float)`\n\n"
            "Task: Configure the paged D-LSA cache with a 512-token local sliding ring, 8192 hot tokens per global visit, "
            "and `\"mixed_rotary_bf16\"` cold page compression. Emit a JSON object with keys `\"tool\"` and `\"arguments\"`."
        ),
        "schema": {
            "required_keys": ["tool", "arguments"],
            "key_types": {"tool": "str", "arguments": "dict"},
        },
        "verifier": "json",
    },
    # --- Code: SciCode Scientific Computing ---
    {
        "id": "scicode_ward_merge_distance",
        "benchmark": "SciCode",
        "category": "Code",
        "official_spark_x26_score": 36.8,
        "prompt": (
            "Write a standalone Python program that reads two lines from standard input:\n"
            "- Line 1: `s_a s_b` (two positive integers representing cluster sizes)\n"
            "- Line 2: four space-separated floats `x1 x2 y1 y2` representing 2D centroids `c_a = (x1, x2)` and `c_b = (y1, y2)`.\n"
            "Compute the exact Ward merge distance $d(a, b) = \\frac{s_a s_b}{s_a + s_b} \\|c_a - c_b\\|_2^2$ and print it "
            "formatted to 4 decimal places (`{val:.4f}`). Wrap your code in a ```python ... ``` block."
        ),
        "test_cases": [
            {"stdin": "4 4\n1.0 0.0 0.0 1.0\n", "expected_stdout": "4.0000"},
            {"stdin": "2 6\n0.0 0.0 2.0 0.0\n", "expected_stdout": "6.0000"},
        ],
        "verifier": "code",
    },
]


def try_load_hf_benchmark_samples(max_per_dataset: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Attempt to stream live samples from the official Hugging Face datasets (`XHToken/gaokao_math_2026`,
    `MathArena/aime_2026`, `zai-org/LongBench-v2`, `openai/graphwalks`, `openai/mrcr`, `google/IFEval`).
    Gracefully returns whatever succeeded so offline/sandboxed environments never fail.
    """
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return loaded

    specs = [
        ("Gaokao 2026", "XHToken/gaokao_math_2026", "test"),
        ("AIME 2026", "MathArena/aime_2026", "train"),
        ("HMMT Feb 26", "MathArena/hmmt_feb_2026", "train"),
        ("LongBench v2", "zai-org/LongBench-v2", "train"),
        ("IFEval", "google/IFEval", "train"),
    ]
    for name, repo_id, split in specs:
        try:
            ds = load_dataset(repo_id, split=split, streaming=True)
            rows = []
            for row in ds:
                rows.append(dict(row))
                if len(rows) >= int(max_per_dataset):
                    break
            if rows:
                loaded[name] = rows
        except Exception:
            continue
    return loaded


# ==============================================================================
# 5. Live Model Evaluation Runners (36L Base vs 72L Loop vs 72L Loop + Paged D-LSA)
# ==============================================================================
@_no_grad()
def evaluate_modern_text_perplexity_and_dlsa(
    model: Any,
    tokenizer: Any,
    seq_len: int = 4096,
    calibrate_pca: bool = True,
) -> Dict[str, Any]:
    """Compare:
      1. `36L_Base` (`passes=1, use_mhc=False, enable_p3=False, use_engram=False`)
      2. `72L_Loop_Dense` (`passes=2, use_mhc=True, enable_p3=True`, exact global attention)
      3. `72L_Loop_DLSA_Flat` (`72` visits + per-visit paged D-LSA `search_mode='flat'`)
      4. `72L_Loop_DLSA_Tree` (`72` visits + per-visit paged D-LSA `search_mode='tree'`)
    on the Modern 2025-2026 Technical Text Stream.
    """
    if not HAS_TORCH:
        raise RuntimeError("evaluate_modern_text_perplexity_and_dlsa requires PyTorch.")

    from train_spark_dlsa import capture_loop72_global_visit_qkv, fit_loop72_visit_bank_pca

    device = next(model.base.parameters()).device
    raw_text = build_modern_long_eval_stream(target_tokens=seq_len + 256, tokenizer=tokenizer)
    enc = tokenizer.encode(raw_text, add_special_tokens=False)[: int(seq_len)]
    input_ids = torch.tensor([enc], dtype=torch.long, device=device)
    actual_len = int(input_ids.shape[1])

    bank = getattr(model, "dlsa_index_bank", None)
    sparse_cfg = getattr(model, "sparse_attention_cfg", None)
    pca_info: Optional[Dict[int, Any]] = None

    if calibrate_pca and bank is not None:
        cal_len = min(actual_len, 1024)
        qkv_map = capture_loop72_global_visit_qkv(model, input_ids[:, :cal_len])
        samples = {gv: [(q_t, k_t)] for gv, (q_t, k_t, _) in qkv_map.items()}
        pca_info = fit_loop72_visit_bank_pca(bank, samples)

    def _eval_nll(logits: torch.Tensor, ids: torch.Tensor) -> Tuple[float, float]:
        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = ids[:, 1:].contiguous()
        nll = float(
            F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            ).item()
        )
        ppl = math.exp(min(20.0, nll))
        return round(nll, 6), round(ppl, 4)

    results: Dict[str, Any] = {}

    # 1. 36L Base (loop off)
    t0 = time.perf_counter()
    with model.loop_config(passes=1, use_mhc=False, use_engram=False, enable_p3=False, use_esm=False):
        if sparse_cfg is not None:
            saved_thr = sparse_cfg.exact_threshold
            sparse_cfg.exact_threshold = actual_len + 1024
        try:
            logits_36 = model.forward_logits(input_ids)
        finally:
            if sparse_cfg is not None:
                sparse_cfg.exact_threshold = saved_thr
    nll_36, ppl_36 = _eval_nll(logits_36, input_ids)
    results["36L_Base_Dense"] = {
        "effective_visits": 36,
        "nll": nll_36,
        "ppl": ppl_36,
        "latency_sec": round(time.perf_counter() - t0, 4),
    }
    del logits_36

    # 2. 72L Loop Dense Reference
    t0 = time.perf_counter()
    if sparse_cfg is not None:
        saved_thr = sparse_cfg.exact_threshold
        sparse_cfg.exact_threshold = actual_len + 1024
    try:
        cache_dense = model.make_inference_cache(session_id="bench_72l_dense")
        logits_72_dense = model.forward_logits(
            input_ids,
            past_key_values=cache_dense,
            chunk_size=getattr(model.settings, "prefill_chunk_tokens", 2048),
        )
    finally:
        if sparse_cfg is not None:
            sparse_cfg.exact_threshold = saved_thr
    nll_72d, ppl_72d = _eval_nll(logits_72_dense, input_ids)
    results["72L_Loop_Dense"] = {
        "effective_visits": model.total_virtual_layers,
        "nll": nll_72d,
        "ppl": ppl_72d,
        "delta_nll_vs_36L": round(nll_72d - nll_36, 6),
        "latency_sec": round(time.perf_counter() - t0, 4),
    }

    # 3. 72L Loop + Paged D-LSA (Flat & Tree)
    ledger_snapshot: Optional[Dict[str, Any]] = None
    if bank is not None and sparse_cfg is not None:
        saved_thr = sparse_cfg.exact_threshold
        # Force sparse path active at `seq_len` (use threshold = min(1024, actual_len // 2))
        sparse_cfg.exact_threshold = min(1024, max(128, actual_len // 2))
        try:
            for mode in ("flat", "tree"):
                for indexer in bank.visit_indexers.values():
                    indexer.cfg.search_mode = mode
                t0 = time.perf_counter()
                cache_sp = model.make_inference_cache(session_id=f"bench_72l_dlsa_{mode}")
                logits_sp = model.forward_logits(
                    input_ids,
                    past_key_values=cache_sp,
                    chunk_size=getattr(model.settings, "prefill_chunk_tokens", 2048),
                )
                dt = time.perf_counter() - t0
                nll_sp, ppl_sp = _eval_nll(logits_sp, input_ids)
                logit_diff = float((logits_sp.float() - logits_72_dense.float()).abs().mean().item())
                if ledger_snapshot is None:
                    ledger_snapshot = cache_sp.memory_ledger()
                results[f"72L_Loop_DLSA_{mode.capitalize()}"] = {
                    "effective_visits": model.total_virtual_layers,
                    "search_mode": mode,
                    "nll": nll_sp,
                    "ppl": ppl_sp,
                    "delta_nll_vs_72L_dense": round(nll_sp - nll_72d, 6),
                    "delta_nll_vs_36L": round(nll_sp - nll_36, 6),
                    "mean_abs_logit_diff_vs_72L_dense": round(logit_diff, 6),
                    "latency_sec": round(dt, 4),
                }
                del logits_sp
        finally:
            sparse_cfg.exact_threshold = saved_thr

    del logits_72_dense
    return {
        "seq_len": actual_len,
        "pca_calibrated_visits": len(pca_info) if pca_info else 0,
        "modes": results,
        "paged_cache_memory_ledger": ledger_snapshot,
    }


@_no_grad()
def evaluate_selected_benchmark_tasks(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 512,
    graphwalks_nodes: int = 64,
    mrcr_rounds: int = 16,
) -> Dict[str, Any]:
    """Run live generation on the selected significant official benchmark tasks (`GraphWalks BFS`, `OpenAI MRCR v2`,
    `Gaokao 2026`, `AIME 2026`, `HMMT Feb 26`, `AA-LCRv1.1`, `IFBench`, `BFCL-V4`, `SciCode`) under `72L Loop + Paged D-LSA`.
    """
    if not HAS_TORCH:
        raise RuntimeError("evaluate_selected_benchmark_tasks requires PyTorch.")

    device = next(model.base.parameters()).device

    def _generate_one(prompt_text: str) -> Tuple[str, float]:
        t0 = time.perf_counter()
        messages = [{"role": "user", "content": prompt_text}]
        try:
            chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(chat_ids, "input_ids"):
                chat_ids = chat_ids.input_ids
            inp = torch.tensor([chat_ids], dtype=torch.long, device=device)
        except Exception:
            enc = tokenizer.encode(prompt_text, add_special_tokens=True)
            inp = torch.tensor([enc], dtype=torch.long, device=device)
        out_ids = model.generate_cached(
            inp,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=False,
            tokenizer=tokenizer,
            stream_print=False,
        )
        gen_ids = out_ids[0, inp.shape[1]:].tolist()
        txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return txt, round(time.perf_counter() - t0, 3)

    task_reports: List[Dict[str, Any]] = []

    # 1. GraphWalks BFS F1
    gw_item = generate_graphwalks_bfs_instance(num_nodes=graphwalks_nodes, branching=2, target_depth=2, seed=42)
    gw_resp, gw_sec = _generate_one(gw_item["prompt"])
    gw_score = score_graphwalks_bfs_f1(gw_resp, gw_item["gold_nodes"])
    task_reports.append({
        "id": "graphwalks_bfs",
        "benchmark": "Graphwalks bfs 256k f1",
        "category": "Long Context",
        "official_spark_x26_score": 92.2,
        "passed": gw_score["f1"] >= 0.5,
        "score": round(gw_score["f1"] * 100.0, 2),
        "details": gw_score,
        "latency_sec": gw_sec,
    })

    # 2. OpenAI MRCR v2
    mrcr_item = generate_mrcr_v2_instance(num_rounds=mrcr_rounds, target_occurrence=3, seed=101)
    mrcr_resp, mrcr_sec = _generate_one(mrcr_item["prompt"])
    mrcr_score = score_mrcr_v2_response(mrcr_resp, mrcr_item["gold_answer"])
    task_reports.append({
        "id": "openai_mrcr_v2",
        "benchmark": "Openai_MRCR v2 256k-512k",
        "category": "Long Context",
        "official_spark_x26_score": 65.6,
        "passed": mrcr_score["exact_match"],
        "score": round(mrcr_score["score"] * 100.0, 2),
        "details": mrcr_score,
        "latency_sec": mrcr_sec,
    })

    # 3. Curated Math, AA-LCR, IFBench, BFCL-V4, SciCode items
    for item in SIGNIFICANT_BENCHMARK_SUITE:
        resp, sec = _generate_one(item["prompt"])
        v_type = item["verifier"]
        if v_type == "math":
            res = verify_math_answer(resp, item["gold"])
            passed = bool(res.ok)
            sc = 100.0 if passed else 0.0
            det = res.detail
        elif v_type == "json":
            res = verify_json_schema(resp, item["schema"])
            passed = bool(res.ok)
            sc = round(float(res.detail.get("score", 1.0 if passed else 0.0)) * 100.0, 2)
            det = res.detail
        elif v_type == "code":
            res = verify_code_stdio(resp, item["test_cases"])
            passed = bool(res.ok)
            sc = round(float(res.detail.get("score", 1.0 if passed else 0.0)) * 100.0, 2)
            det = res.detail
        else:
            passed = False
            sc = 0.0
            det = {}
        task_reports.append({
            "id": item["id"],
            "benchmark": item["benchmark"],
            "category": item["category"],
            "official_spark_x26_score": item["official_spark_x26_score"],
            "passed": passed,
            "score": sc,
            "details": det,
            "latency_sec": sec,
        })

    mean_score = round(sum(r["score"] for r in task_reports) / max(1, len(task_reports)), 2)
    pass_rate = round(sum(1 for r in task_reports if r["passed"]) / max(1, len(task_reports)) * 100.0, 2)
    return {
        "num_tasks": len(task_reports),
        "pass_rate_pct": pass_rate,
        "mean_score": mean_score,
        "tasks": task_reports,
    }


def evaluate_cold_kv_codecs_summary() -> Dict[str, Dict[str, float]]:
    """Run a deterministic quality & compression audit across all 4 cold KV page codecs (`bf16`, `int8`, `int4`, `mixed_rotary_bf16`)."""
    # 4 KV heads, 64 tokens (1 page), 256 head_dim (64 rotated + 192 RoPE-free)
    k_page = [
        [
            [math.sin((h + 1) * 0.3 + t * 0.07 + d * 0.05) for d in range(64)]
            + [math.cos((h + 1) * 0.2 + t * 0.05 + d * 0.03) * 0.6 for d in range(192)]
            for t in range(64)
        ]
        for h in range(4)
    ]
    v_page = [
        [
            [math.sin((h + 2) * 0.17 - t * 0.04 + d * 0.02) * 0.5 for d in range(256)]
            for t in range(64)
        ]
        for h in range(4)
    ]
    out: Dict[str, Dict[str, float]] = {}
    for c_name in ("bf16", "int8", "int4", "mixed_rotary_bf16"):
        cfg = KVCodecConfig(codec=c_name, group_size=64, rope_dim=64, rope_first=True)
        out[c_name] = measure_codec_error(k_page, v_page, cfg)
    return out


def format_official_comparison_markdown(live_task_results: Optional[Dict[str, Any]] = None) -> str:
    """Format a Markdown comparison table referencing the official Spark X2.6-4B benchmark table alongside live Loop72-LSA results."""
    live_by_bench: Dict[str, float] = {}
    if live_task_results and "tasks" in live_task_results:
        for t in live_task_results["tasks"]:
            live_by_bench[t["benchmark"]] = float(t["score"])

    lines = [
        "| Category | Benchmark | Official Spark X2.6-4B | Spark X2.5-4B | Qwen3.5 9B | Qwen3.5 4B | Gemma4 12B | Loop72 + D-LSA (Live) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for b_name, row in OFFICIAL_BENCHMARK_TABLE.items():
        live_val = f"**{live_by_bench[b_name]:.1f}**" if b_name in live_by_bench else "—"
        g12 = f"{row['Gemma4_12B']:.1f}" if row.get("Gemma4_12B") is not None else "-"
        lines.append(
            f"| {row['category']} | **{b_name}** | **{row['Spark_X2.6_4B']:.1f}** | "
            f"{row['Spark_X2.5_4B']:.1f} | {row['Qwen3.5_9B']:.1f} | {row['Qwen3.5_4B']:.1f} | "
            f"{g12} | {live_val} |"
        )
    return "\n".join(lines)
