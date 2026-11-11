"""sparkx_stat_benchmark.py — Statistically Rigorous Stratified-Sample Benchmark Runner
for Spark X2.6 4B Recursive72 (Loop72 + mHC + Stage E Paged D-LSA).

Why this module exists alongside `sparkx_full_benchmark.py`:
  - `sparkx_full_benchmark.py` evaluates 100% of every pinned long-context dataset
    (`graphwalks_bfs`, `mrcr_8needle`, `longbench_v2`) at `max_new_tokens=32768`, which
    requires multi-day GPU compute on a single workstation/Colab GPU.
  - `sparkx_stat_benchmark.py` applies survey-sampling theory (Stratified Random Sampling
    Without Replacement + Finite Population Correction + Wilson / Rao-Wu Bootstrap 95% CIs
    + Common-Random-Numbers Paired Bootstrap Ablation) to obtain unbiased, uncertainty-quantified
    estimates across both pinned official-adapter long-context suites and real Hugging Face
    diagnostic benchmarks in ~30 minutes to ~2.5 hours.

Strict Honesty & Provenance Contract:
  1. Never takes the first `n` rows (`rows[:n]`), eliminating dataset ordering / difficulty /
     length bias (e.g., AIME problem order 1..15 or LongBench domain grouping).
  2. Candidate rows are validated (`status == "ready"`, supported verifier rules, and optional
     `max_context_tokens` horizon filter) BEFORE drawing the stratified sample, recording the
     exact eligible population frame `N_frame` and all exclusion counts.
  3. Finite Population Correction (`FPC = sqrt((N - n) / (N - 1))`) is applied ONLY when the
     full split was scanned (`population_exact=True`), shrinking standard errors to `0.0`
     automatically when `n == N` (e.g., all 30/30 items of `AIME 2026` or `HMMT Feb 2026`).
  4. Sampled runs ALWAYS record `sampling_mode="statistical_subset"`, `complete=False`,
     `official_comparable=False`, and `published_score_comparable=False`. No cross-suite
     macro average over incompatible metrics is computed.
  5. Every suite writes `<suite>.sample_manifest.json` and flushes per-row `<suite>.jsonl`
     with SHA-256 row digests (`ResumeIntegrityError` on mismatch), allowing interrupted
     Colab sessions to resume on the exact same sampled items (`resume=True`).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from sparkx_full_benchmark import (
    MODEL_REVISION,
    NATIVE_PROTOCOLS,
    ResumeIntegrityError,
    _jsonable,
    _load_resumable_records,
    source_identity,
    stable_digest,
)

PRESET_NAMES: Tuple[str, ...] = ("ultra_fast", "quick", "standard", "thorough")


@dataclasses.dataclass(frozen=True)
class StatSuiteSpec:
    """Specification of a statistically sampled benchmark suite."""

    name: str
    benchmark_name: str
    category: str
    task_mode: str  # "official" | "diagnostic"
    dataset: str
    config_name: Optional[str]
    candidate_splits: Tuple[str, ...]
    revision: Optional[str]
    files: Optional[Tuple[str, ...]]
    metric_label: str
    preset_n: Dict[str, int]
    default_max_new_tokens: int
    est_avg_gen_tokens: int
    est_avg_prompt_tokens: int
    max_scan_rows: Optional[int] = None
    requires_hf_gate: bool = False
    reference_spark_x26_36l: Optional[float] = None
    reference_note: str = ""


STAT_SUITE_SPECS: Dict[str, StatSuiteSpec] = {
    # --- Tier 1: Pinned Official-Adapter Long-Context Suites (task_mode="official") ---
    "graphwalks_bfs_128k": StatSuiteSpec(
        name="graphwalks_bfs_128k",
        benchmark_name="Graphwalks bfs 256k f1",
        category="Long Context",
        task_mode="official",
        dataset=NATIVE_PROTOCOLS["graphwalks_bfs"].dataset,
        config_name=None,
        candidate_splits=("train",),
        revision=NATIVE_PROTOCOLS["graphwalks_bfs"].revision,
        files=("graphwalks_128k_and_shorter.parquet",),
        metric_label="set_f1_pct",
        preset_n={"ultra_fast": 8, "quick": 12, "standard": 24, "thorough": 48},
        default_max_new_tokens=16384,
        est_avg_gen_tokens=2800,
        est_avg_prompt_tokens=32000,
        max_scan_rows=600,
        reference_spark_x26_36l=77.6,
        reference_note="Spark-X2.6-4B card reports 77.6 on Graphwalks bfs <=256k; this frame covers the <=128k parquet split.",
    ),
    "mrcr_8needle": StatSuiteSpec(
        name="mrcr_8needle",
        benchmark_name="Openai_MRCR v2 256k-512k",
        category="Long Context",
        task_mode="official",
        dataset=NATIVE_PROTOCOLS["mrcr_8needle"].dataset,
        config_name=None,
        candidate_splits=("train",),
        revision=NATIVE_PROTOCOLS["mrcr_8needle"].revision,
        files=("8needle/8needle_0.parquet",),
        metric_label="seq_match_pct",
        preset_n={"ultra_fast": 8, "quick": 10, "standard": 20, "thorough": 40},
        default_max_new_tokens=8192,
        est_avg_gen_tokens=1100,
        est_avg_prompt_tokens=48000,
        max_scan_rows=400,
        reference_spark_x26_36l=51.4,
        reference_note="Spark-X2.6-4B card reports 51.4 on Openai_MRCR v2 256k-512k; this frame filters to <=max_context_tokens.",
    ),
    "longbench_v2": StatSuiteSpec(
        name="longbench_v2",
        benchmark_name="LongBench v2",
        category="Long Context",
        task_mode="official",
        dataset=NATIVE_PROTOCOLS["longbench_v2"].dataset,
        config_name=None,
        candidate_splits=("train",),
        revision=NATIVE_PROTOCOLS["longbench_v2"].revision,
        files=NATIVE_PROTOCOLS["longbench_v2"].files,
        metric_label="accuracy_pct",
        preset_n={"ultra_fast": 12, "quick": 20, "standard": 40, "thorough": 80},
        default_max_new_tokens=8192,
        est_avg_gen_tokens=1400,
        est_avg_prompt_tokens=45000,
        max_scan_rows=None,
        reference_spark_x26_36l=56.5,
        reference_note="Stratified across (domain, length) within <=max_context_tokens.",
    ),
    # --- Tier 2: Real HF Streamed / Downloaded Diagnostic Suites (task_mode="diagnostic") ---
    "aime_2026": StatSuiteSpec(
        name="aime_2026",
        benchmark_name="AIME 2026",
        category="Math",
        task_mode="diagnostic",
        dataset="MathArena/aime_2026",
        config_name=None,
        candidate_splits=("train", "test"),
        revision=None,
        files=None,
        metric_label="exact_match_pct",
        preset_n={"ultra_fast": 8, "quick": 12, "standard": 30, "thorough": 30},
        default_max_new_tokens=16384,
        est_avg_gen_tokens=3600,
        est_avg_prompt_tokens=220,
        max_scan_rows=None,
        reference_spark_x26_36l=83.3,
        reference_note="N=30 total problems; 'standard'/'thorough' evaluate 100% of split (FPC=0).",
    ),
    "hmmt_feb_2026": StatSuiteSpec(
        name="hmmt_feb_2026",
        benchmark_name="HMMT Feb 26",
        category="Math",
        task_mode="diagnostic",
        dataset="MathArena/hmmt_feb_2026",
        config_name=None,
        candidate_splits=("train", "test"),
        revision=None,
        files=None,
        metric_label="exact_match_pct",
        preset_n={"ultra_fast": 8, "quick": 12, "standard": 30, "thorough": 30},
        default_max_new_tokens=16384,
        est_avg_gen_tokens=3800,
        est_avg_prompt_tokens=240,
        max_scan_rows=None,
        reference_spark_x26_36l=73.3,
        reference_note="N=30 total problems; 'standard'/'thorough' evaluate 100% of split (FPC=0).",
    ),
    "imo_answerbench": StatSuiteSpec(
        name="imo_answerbench",
        benchmark_name="IMO-AnswerBench",
        category="Math",
        task_mode="diagnostic",
        dataset="Hwilner/imo-answerbench",
        config_name=None,
        candidate_splits=("train", "test"),
        revision=None,
        files=None,
        metric_label="exact_match_pct",
        preset_n={"ultra_fast": 8, "quick": 12, "standard": 24, "thorough": 48},
        default_max_new_tokens=16384,
        est_avg_gen_tokens=4000,
        est_avg_prompt_tokens=280,
        max_scan_rows=None,
        reference_spark_x26_36l=67.5,
        reference_note="Diagnostic symbolic/numeric boxed-answer check.",
    ),
    "gaokao_2026": StatSuiteSpec(
        name="gaokao_2026",
        benchmark_name="Gaokao 2026",
        category="Math",
        task_mode="diagnostic",
        dataset="XHToken/gaokao_math_2026",
        config_name=None,
        candidate_splits=("test", "train"),
        revision=None,
        files=None,
        metric_label="item_accuracy_pct",
        preset_n={"ultra_fast": 8, "quick": 12, "standard": 22, "thorough": 30},
        default_max_new_tokens=16384,
        est_avg_gen_tokens=2800,
        est_avg_prompt_tokens=260,
        max_scan_rows=None,
        reference_spark_x26_36l=None,
        reference_note="Unweighted % accuracy on closed-form items (official card reports weighted /150 exam points: 141/150).",
    ),
    "ifeval": StatSuiteSpec(
        name="ifeval",
        benchmark_name="IFEval",
        category="Instruction Following",
        task_mode="diagnostic",
        dataset="google/IFEval",
        config_name=None,
        candidate_splits=("train", "test"),
        revision=None,
        files=None,
        metric_label="constraint_score_pct",
        preset_n={"ultra_fast": 15, "quick": 24, "standard": 50, "thorough": 100},
        default_max_new_tokens=4096,
        est_avg_gen_tokens=900,
        est_avg_prompt_tokens=180,
        max_scan_rows=None,
        reference_spark_x26_36l=86.9,
        reference_note="Evaluated on rules supported by local deterministic IFEval verifier; unsupported rules excluded pre-sample.",
    ),
    "gpqa_diamond": StatSuiteSpec(
        name="gpqa_diamond",
        benchmark_name="GPQA Diamond",
        category="General & Knowledge",
        task_mode="diagnostic",
        dataset="Idavidrein/gpqa",
        config_name="gpqa_diamond",
        candidate_splits=("train", "test"),
        revision=None,
        files=None,
        metric_label="accuracy_pct",
        preset_n={"ultra_fast": 10, "quick": 16, "standard": 36, "thorough": 64},
        default_max_new_tokens=8192,
        est_avg_gen_tokens=2200,
        est_avg_prompt_tokens=450,
        max_scan_rows=None,
        requires_hf_gate=True,
        reference_spark_x26_36l=77.1,
        reference_note="Gated HF dataset (N=198); requires HF_TOKEN with accepted Idavidrein/gpqa gate.",
    ),
    "bfcl_v4_simple": StatSuiteSpec(
        name="bfcl_v4_simple",
        benchmark_name="BFCL-V4",
        category="Agent",
        task_mode="diagnostic",
        dataset="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        config_name=None,
        candidate_splits=("train",),
        revision=None,
        files=("BFCL_v4_simple_python.json",),
        metric_label="tool_ast_pct",
        preset_n={"ultra_fast": 12, "quick": 20, "standard": 40, "thorough": 80},
        default_max_new_tokens=4096,
        est_avg_gen_tokens=750,
        est_avg_prompt_tokens=550,
        max_scan_rows=None,
        reference_spark_x26_36l=71.8,
        reference_note="Single-turn simple Python call subset (official BFCL-V4 also includes multi-turn/live/parallel).",
    ),
}

DEFAULT_CORE_SUITES: Tuple[str, ...] = (
    "graphwalks_bfs_128k",
    "mrcr_8needle",
    "longbench_v2",
    "aime_2026",
    "hmmt_feb_2026",
    "ifeval",
    "gpqa_diamond",
    "bfcl_v4_simple",
)


# ==============================================================================
# 1. Pure Statistical Estimators (FPC, Wilson CI, Rao-Wu Bootstrap, Paired CRN)
# ==============================================================================
def _z_crit(confidence: float) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must lie strictly in (0, 1), got {confidence}")
    return float(statistics.NormalDist().inv_cdf(0.5 + 0.5 * float(confidence)))


def compute_fpc_factor(n: int, population_size: Optional[int], population_exact: bool = True) -> float:
    """Compute Finite Population Correction sqrt((N - n) / (N - 1)) when N is exact."""
    if n <= 0:
        return 1.0
    if not population_exact or population_size is None:
        return 1.0
    big_n = int(population_size)
    if big_n <= 1 or n >= big_n:
        return 0.0
    return math.sqrt(max(0.0, float(big_n - n) / float(big_n - 1)))


def _wilson_ci_pct(
    mean_pct: float,
    n: int,
    fpc: float,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """FPC-adjusted Wilson score confidence interval on [0, 100]."""
    p = min(1.0, max(0.0, float(mean_pct) / 100.0))
    if n <= 0:
        return (0.0, 100.0)
    if fpc <= 0.0:
        return (100.0 * p, 100.0 * p)
    z = _z_crit(confidence)
    n_eff = float(n) / max(1e-12, fpc * fpc)
    z2 = z * z
    denom = 1.0 + z2 / n_eff
    center = (p + z2 / (2.0 * n_eff)) / denom
    radicand = max(0.0, (p * (1.0 - p) / n_eff) + (z2 / (4.0 * n_eff * n_eff)))
    half = (z * math.sqrt(radicand)) / denom
    lo = max(0.0, 100.0 * (center - half))
    hi = min(100.0, 100.0 * (center + half))
    return (lo, hi)


def compute_statistical_summary(
    scores: Sequence[float],
    *,
    population_size: Optional[int] = None,
    population_exact: bool = True,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 20261110,
) -> Dict[str, Any]:
    """Compute sample mean, FPC-adjusted SE, Wilson CI, and Rao-Wu finite-population bootstrap CI."""
    vals = [float(x) for x in scores]
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "population_size": int(population_size) if (population_exact and population_size is not None) else None,
            "population_exact": bool(population_exact and population_size is not None),
            "sampling_fraction": None,
            "fpc_factor": None,
            "mean": None,
            "sample_std": None,
            "se_mean": None,
            "confidence": float(confidence),
            "ci_method": None,
            "ci_low": None,
            "ci_high": None,
            "margin_of_error": None,
            "wilson_ci_low": None,
            "wilson_ci_high": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "is_binary": None,
        }

    for v in vals:
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"Scores must lie in [0, 100], got {v}")

    mean_val = sum(vals) / float(n)
    sample_std = math.sqrt(sum((x - mean_val) ** 2 for x in vals) / float(n - 1)) if n >= 2 else 0.0
    eff_pop = int(population_size) if (population_exact and population_size is not None and int(population_size) >= n) else None
    fpc = compute_fpc_factor(n, eff_pop, population_exact=(eff_pop is not None))
    se_mean = (sample_std / math.sqrt(float(n))) * fpc if n >= 2 else 0.0
    is_binary = all(abs(x) <= 1e-9 or abs(x - 100.0) <= 1e-9 for x in vals)

    w_lo, w_hi = _wilson_ci_pct(mean_val, n=n, fpc=fpc, confidence=confidence)

    if fpc <= 0.0:
        b_lo, b_hi = mean_val, mean_val
    elif n == 1 or sample_std <= 1e-12:
        # Degenerate sample variance on n < N: fall back to conservative FPC-Wilson bounds
        b_lo, b_hi = w_lo, w_hi
    else:
        rng = random.Random(int(seed))
        # Rao-Wu finite-population rescaling: sqrt(n / (n - 1)) * FPC restores exact unbiased variance
        scale = fpc * math.sqrt(float(n) / float(n - 1))
        boot_means: List[float] = []
        inv_n = 1.0 / float(n)
        for _ in range(max(200, int(n_bootstrap))):
            b_sum = 0.0
            for _j in range(n):
                b_sum += vals[rng.randrange(n)]
            b_mean = b_sum * inv_n
            adj = mean_val + scale * (b_mean - mean_val)
            boot_means.append(min(100.0, max(0.0, adj)))
        boot_means.sort()
        alpha = 1.0 - float(confidence)
        idx_lo = min(len(boot_means) - 1, max(0, int(math.floor((alpha * 0.5) * len(boot_means)))))
        idx_hi = min(len(boot_means) - 1, max(0, int(math.ceil((1.0 - alpha * 0.5) * len(boot_means))) - 1))
        b_lo, b_hi = boot_means[idx_lo], boot_means[idx_hi]

    if fpc <= 0.0:
        ci_method = "exact_finite_census"
        ci_lo, ci_hi = mean_val, mean_val
    elif is_binary or n == 1 or sample_std <= 1e-12:
        ci_method = "wilson_fpc"
        ci_lo, ci_hi = w_lo, w_hi
    else:
        ci_method = "rao_wu_bootstrap_fpc"
        ci_lo, ci_hi = b_lo, b_hi

    moe = 0.5 * (ci_hi - ci_lo)
    return {
        "n": n,
        "population_size": eff_pop,
        "population_exact": eff_pop is not None,
        "sampling_fraction": round(float(n) / float(eff_pop), 6) if eff_pop else None,
        "fpc_factor": round(fpc, 6),
        "mean": round(mean_val, 4),
        "sample_std": round(sample_std, 4),
        "se_mean": round(se_mean, 4),
        "confidence": float(confidence),
        "ci_method": ci_method,
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "margin_of_error": round(moe, 4),
        "wilson_ci_low": round(w_lo, 4),
        "wilson_ci_high": round(w_hi, 4),
        "bootstrap_ci_low": round(b_lo, 4),
        "bootstrap_ci_high": round(b_hi, 4),
        "is_binary": is_binary,
    }


def compute_stratified_statistical_summary(
    stratum_scores: Mapping[str, Sequence[float]],
    stratum_population_counts: Mapping[str, int],
    *,
    confidence: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Compute Neyman-Cochran stratified mean and finite-population variance across exact strata."""
    total_pop = sum(int(v) for v in stratum_population_counts.values() if int(v) > 0)
    if total_pop <= 0:
        return None
    covered_pop = 0
    weighted_mean = 0.0
    var_sum = 0.0
    for key, big_nh in stratum_population_counts.items():
        big_nh = int(big_nh)
        if big_nh <= 0:
            continue
        vals = [float(x) for x in stratum_scores.get(key, ())]
        nh = len(vals)
        if nh == 0:
            continue
        covered_pop += big_nh
        mean_h = sum(vals) / float(nh)
        wh = float(big_nh) / float(total_pop)
        weighted_mean += wh * mean_h
        if nh >= 2 and nh < big_nh:
            s2_h = sum((x - mean_h) ** 2 for x in vals) / float(nh - 1)
            fpc2_h = float(big_nh - nh) / float(big_nh)
            var_sum += (wh * wh) * fpc2_h * (s2_h / float(nh))
    if covered_pop <= 0:
        return None
    if covered_pop < total_pop:
        weighted_mean *= float(total_pop) / float(covered_pop)
    se_strat = math.sqrt(max(0.0, var_sum))
    z = _z_crit(confidence)
    ci_lo = max(0.0, weighted_mean - z * se_strat)
    ci_hi = min(100.0, weighted_mean + z * se_strat)
    return {
        "stratified_mean": round(weighted_mean, 4),
        "stratified_se": round(se_strat, 4),
        "stratified_ci_low": round(ci_lo, 4),
        "stratified_ci_high": round(ci_hi, 4),
        "population_coverage_ratio": round(float(covered_pop) / float(total_pop), 6),
    }


def compute_paired_bootstrap_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    label_a: str = "72L_Recursive",
    label_b: str = "36L_Frozen_Base",
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 20261110,
) -> Dict[str, Any]:
    """Paired Common-Random-Numbers (CRN) bootstrap CI + sign-flip permutation p-value on (A - B)."""
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Paired score lists must have identical length ({len(scores_a)} != {len(scores_b)})")
    n = len(scores_a)
    if n == 0:
        return {"n_paired": 0, "mean_diff": None, "ci_low": None, "ci_high": None, "p_value_two_sided": None}
    a_vals = [float(x) for x in scores_a]
    b_vals = [float(x) for x in scores_b]
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    mean_a = sum(a_vals) / float(n)
    mean_b = sum(b_vals) / float(n)
    mean_diff = sum(diffs) / float(n)
    wins_a = sum(1 for d in diffs if d > 1e-9)
    wins_b = sum(1 for d in diffs if d < -1e-9)
    ties = n - wins_a - wins_b

    if n == 1 or all(abs(d - mean_diff) <= 1e-12 for d in diffs):
        ci_lo = ci_hi = mean_diff
        se_diff = 0.0
    else:
        std_d = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / float(n - 1))
        se_diff = std_d / math.sqrt(float(n))
        rng = random.Random(int(seed))
        scale = math.sqrt(float(n) / float(n - 1))
        boot_diffs: List[float] = []
        inv_n = 1.0 / float(n)
        for _ in range(max(500, int(n_bootstrap))):
            s = 0.0
            for _j in range(n):
                s += diffs[rng.randrange(n)]
            b_m = s * inv_n
            boot_diffs.append(max(-100.0, min(100.0, mean_diff + scale * (b_m - mean_diff))))
        boot_diffs.sort()
        alpha = 1.0 - float(confidence)
        idx_lo = min(len(boot_diffs) - 1, max(0, int(math.floor((alpha * 0.5) * len(boot_diffs)))))
        idx_hi = min(len(boot_diffs) - 1, max(0, int(math.ceil((1.0 - alpha * 0.5) * len(boot_diffs))) - 1))
        ci_lo, ci_hi = boot_diffs[idx_lo], boot_diffs[idx_hi]

    # Exact / Monte-Carlo two-sided sign-flip permutation test under H0: E[A_i - B_i] = 0
    obs_abs = abs(mean_diff)
    if obs_abs <= 1e-12:
        p_val = 1.0
    elif n <= 16:
        extreme = 0
        total_flips = 1 << n
        for mask in range(total_flips):
            s = 0.0
            for i in range(n):
                s += diffs[i] if ((mask >> i) & 1) else -diffs[i]
            if abs(s / float(n)) >= obs_abs - 1e-12:
                extreme += 1
        p_val = float(extreme) / float(total_flips)
    else:
        rng_perm = random.Random(int(seed) ^ 0x5A5A5A5A)
        n_perm = max(2000, int(n_bootstrap))
        extreme = 1
        for _ in range(n_perm):
            s = 0.0
            for d in diffs:
                s += d if (rng_perm.getrandbits(1) == 1) else -d
            if abs(s / float(n)) >= obs_abs - 1e-12:
                extreme += 1
        p_val = float(extreme) / float(n_perm + 1)

    return {
        "n_paired": n,
        "label_a": label_a,
        "label_b": label_b,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_diff": round(mean_diff, 4),
        "se_diff": round(se_diff, 4),
        "confidence": float(confidence),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "p_value_two_sided": round(min(1.0, p_val), 6),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "statistically_significant": bool((ci_lo > 0.0 or ci_hi < 0.0) and p_val < (1.0 - confidence)),
    }


def recommend_sample_size(
    target_half_width_pct: float = 8.0,
    *,
    prior_p: float = 0.70,
    population_size: Optional[int] = None,
    confidence: float = 0.95,
) -> int:
    """Cochran's sample-size formula with Finite Population Correction for target half-width e%."""
    if target_half_width_pct <= 0.0:
        raise ValueError("target_half_width_pct must be positive")
    p = min(0.99, max(0.01, float(prior_p)))
    e = float(target_half_width_pct) / 100.0
    z = _z_crit(confidence)
    n0 = (z * z * p * (1.0 - p)) / (e * e)
    if population_size is not None and int(population_size) > 0:
        big_n = int(population_size)
        n_adj = n0 / (1.0 + (n0 - 1.0) / float(big_n))
        return max(1, min(big_n, int(math.ceil(n_adj))))
    return max(1, int(math.ceil(n0)))


def estimate_preset_budget_table(
    preset: str = "quick",
    *,
    suites: Optional[Sequence[str]] = None,
    decode_tok_per_s: float = 27.4,
    prefill_tok_per_s_short: float = 7500.0,
    prefill_tok_per_s_long: float = 1200.0,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Estimate pre-run sample sizes, expected 95% CI margins of error, and GPU wall-clock minutes."""
    if preset not in PRESET_NAMES:
        raise ValueError(f"Unknown preset {preset!r}; choose from {PRESET_NAMES}")
    suite_names = list(suites or DEFAULT_CORE_SUITES)
    rows: List[Dict[str, Any]] = []
    total_sec = 0.0
    total_questions = 0
    known_pop = {
        "longbench_v2": 503,
        "aime_2026": 30,
        "hmmt_feb_2026": 30,
        "gpqa_diamond": 198,
        "ifeval": 480,
        "bfcl_v4_simple": 360,
        "imo_answerbench": 400,
    }
    for name in suite_names:
        spec = STAT_SUITE_SPECS[name]
        n = int(spec.preset_n[preset])
        big_n = known_pop.get(name)
        if big_n is not None:
            n = min(n, big_n)
        fpc = compute_fpc_factor(n, big_n, population_exact=(big_n is not None))
        p_prior = (spec.reference_spark_x26_36l / 100.0) if spec.reference_spark_x26_36l else 0.70
        w_lo, w_hi = _wilson_ci_pct(100.0 * p_prior, n=n, fpc=fpc, confidence=confidence)
        est_moe = 0.5 * (w_hi - w_lo)
        pf_rate = prefill_tok_per_s_long if spec.est_avg_prompt_tokens > 4612 else prefill_tok_per_s_short
        sec_per_q = (spec.est_avg_prompt_tokens / max(1.0, pf_rate)) + (
            spec.est_avg_gen_tokens / max(1.0, decode_tok_per_s)
        )
        suite_sec = n * sec_per_q
        total_sec += suite_sec
        total_questions += n
        rows.append(
            {
                "suite": name,
                "task_mode": spec.task_mode,
                "category": spec.category,
                "planned_n": n,
                "approx_population_N": big_n,
                "fpc_factor": round(fpc, 3),
                "expected_95ci_half_width_pct": round(est_moe, 2),
                "max_new_tokens": spec.default_max_new_tokens,
                "est_minutes": round(suite_sec / 60.0, 1),
            }
        )
    return {
        "preset": preset,
        "assumed_decode_tok_per_s": decode_tok_per_s,
        "total_planned_questions": total_questions,
        "estimated_total_minutes": round(total_sec / 60.0, 1),
        "estimated_total_hours": round(total_sec / 3600.0, 2),
        "suites": rows,
    }


# ==============================================================================
# 2. Candidate Frame Loading & Stratified Sampling Without Replacement
# ==============================================================================
def _derive_suite_seed(base_seed: int, suite_name: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}::{suite_name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _estimate_prompt_tokens(
    task: Mapping[str, Any],
    tokenizer: Any = None,
    max_context_tokens: Optional[int] = None,
) -> int:
    """Fast exact-boundary prompt token estimator.

    Avoids tokenizing 50M+ unsampled long-context tokens up front: uses exact character
    lower/upper bounds when unambiguously inside or outside `max_context_tokens`, and
    runs `tokenizer.apply_chat_template` on borderline rows near `max_context_tokens`.
    """
    messages = task.get("messages")
    if messages is None:
        prompt = str(task.get("prompt") or "")
        messages = [{"role": "user", "content": prompt}]
    total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, Mapping))
    char_est = max(1, int(math.ceil(total_chars / 3.5)))
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template") or max_context_tokens is None:
        return char_est
    lim = int(max_context_tokens)
    # Safe lower bound: <= 0.5 chars/token + 64 chat-template overhead cannot exceed `lim`
    if total_chars * 2 + 64 <= lim:
        return char_est
    # Safe upper bound: > 6.5 chars/token cannot fit inside `lim`
    if total_chars > lim * 6.5:
        return max(lim + 1, char_est)
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        elif hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return int(len(ids))
    except Exception:
        return char_est


def _assign_stratum(spec: StatSuiteSpec, row: Mapping[str, Any], task: Mapping[str, Any], row_index: int, prompt_tokens: int) -> str:
    """Assign a deterministic stratification key before sampling."""
    if spec.name == "longbench_v2":
        dom = str(row.get("domain", "unknown")).strip() or "unknown"
        length_bin = str(row.get("length", "unknown")).strip() or "unknown"
        return f"{dom}|{length_bin}"
    if spec.name == "ifeval":
        rules = row.get("instruction_id_list")
        k = len(rules) if isinstance(rules, list) else 1
        bucket = "1_rule" if k <= 1 else ("2_rules" if k == 2 else "3plus_rules")
        return bucket
    if spec.name in ("aime_2026", "hmmt_feb_2026"):
        # Contest problems are ordered by increasing difficulty; stratify by problem-index tertile
        return f"pos_bin_{min(2, int(row_index) // 10)}"
    if spec.name == "gpqa_diamond":
        sub = str(row.get("Subdomain") or row.get("High-level domain") or "general").strip()
        return sub[:40]
    if spec.name == "imo_answerbench":
        cat = str(row.get("Category") or row.get("category") or row.get("Subject") or "").strip()
        return cat if cat else f"pos_bin_{int(row_index) // 100}"
    if spec.name == "gaokao_2026":
        return str(task.get("verifier", "math"))
    if spec.task_mode == "official":
        # Long-context suites: stratify across context-length tiers
        if prompt_tokens <= 16384:
            return "len_le_16k"
        if prompt_tokens <= 32768:
            return "len_16k_32k"
        if prompt_tokens <= 65536:
            return "len_32k_64k"
        if prompt_tokens <= 131072:
            return "len_64k_128k"
        return "len_gt_128k"
    return "default"


def stratified_sample_without_replacement(
    candidates: Sequence[Dict[str, Any]],
    sample_size: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    """Proportional Hamilton (Largest-Remainder) stratified random sample without replacement.

    Guarantees:
      - Order-invariant: sorting or permuting `candidates` never biases stratum proportions
        (within each stratum, items are canonically ordered by `(row_sha256, row_index)` before
        seeded Fisher-Yates selection!).
      - Exact sample size `min(sample_size, len(candidates))`.
      - Every non-empty stratum receives `floor(n * N_h / N)` items plus largest fractional remainders.
    """
    total = len(candidates)
    if sample_size <= 0 or total == 0:
        return []
    if sample_size >= total:
        return sorted(candidates, key=lambda c: int(c["row_index"]))

    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for cand in candidates:
        s_key = str(cand.get("stratum", "default"))
        by_stratum.setdefault(s_key, []).append(cand)

    strata_keys = sorted(by_stratum.keys())
    alloc: Dict[str, int] = {}
    remainders: List[Tuple[float, int, str]] = []
    assigned = 0
    for s_key in strata_keys:
        nh_pop = len(by_stratum[s_key])
        exact_quota = (float(sample_size) * float(nh_pop)) / float(total)
        base_q = min(nh_pop, int(math.floor(exact_quota)))
        alloc[s_key] = base_q
        assigned += base_q
        # Tie-break remainders deterministically by stratum population then key
        remainders.append((exact_quota - float(base_q), nh_pop, s_key))

    leftover = sample_size - assigned
    remainders.sort(key=lambda t: (-t[0], -t[1], t[2]))
    for _frac, nh_pop, s_key in remainders:
        if leftover <= 0:
            break
        if alloc[s_key] < nh_pop:
            alloc[s_key] += 1
            leftover -= 1

    selected: List[Dict[str, Any]] = []
    for s_key in strata_keys:
        take = alloc[s_key]
        if take <= 0:
            continue
        # Canonical order by (row_sha256, row_index) makes sampling 100% invariant to input stream order!
        pool = sorted(by_stratum[s_key], key=lambda c: (str(c["row_sha256"]), int(c["row_index"])))
        stratum_rng = random.Random(_derive_suite_seed(seed, f"stratum::{s_key}"))
        chosen = stratum_rng.sample(pool, k=take)
        selected.extend(chosen)

    # Sort selected items by original `row_index` for clean reproducible JSONL ordering
    selected.sort(key=lambda c: int(c["row_index"]))
    return selected


def _load_bfcl_all_simple_rows(token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load the complete BFCL_v4_simple_python JSONL split + ground-truth answers."""
    from huggingface_hub import hf_hub_download

    repo_id = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    q_path = hf_hub_download(repo_id=repo_id, filename="BFCL_v4_simple_python.json", repo_type="dataset", token=token)
    a_path = hf_hub_download(
        repo_id=repo_id,
        filename="possible_answer/BFCL_v4_simple_python.json",
        repo_type="dataset",
        token=token,
    )
    q_lines = Path(q_path).read_text(encoding="utf-8").splitlines()
    a_lines = Path(a_path).read_text(encoding="utf-8").splitlines()
    answers_by_id: Dict[str, Any] = {}
    for line in a_lines:
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        if "id" in obj:
            answers_by_id[str(obj["id"])] = obj.get("ground_truth", [])
    merged: List[Dict[str, Any]] = []
    for line in q_lines:
        s = line.strip()
        if not s:
            continue
        q = dict(json.loads(s))
        qid = str(q.get("id", ""))
        if qid in answers_by_id:
            q["ground_truth"] = answers_by_id[qid]
        merged.append(q)
    return merged


def _iter_suite_raw_rows(
    spec: StatSuiteSpec,
    *,
    token: Optional[str] = None,
) -> Tuple[Iterator[Dict[str, Any]], str]:
    """Open a streaming iterator over the suite's Hugging Face dataset and return (iterator, split_used)."""
    if spec.name == "bfcl_v4_simple":
        rows_list = _load_bfcl_all_simple_rows(token=token)
        return iter(rows_list), "BFCL_v4_simple_python"

    from datasets import load_dataset

    eff_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    errors: List[str] = []
    for sp in spec.candidate_splits:
        try:
            kwargs: Dict[str, Any] = {"split": sp, "streaming": True}
            if spec.revision is not None:
                kwargs["revision"] = spec.revision
            if spec.files is not None:
                kwargs["data_files"] = {sp: list(spec.files)}
            if eff_token:
                kwargs["token"] = eff_token
            ds = (
                load_dataset(spec.dataset, spec.config_name, **kwargs)
                if spec.config_name
                else load_dataset(spec.dataset, **kwargs)
            )
            iterator = iter(ds)
            first = next(iterator)

            def _chain(head: Any, tail: Iterator[Any]) -> Iterator[Dict[str, Any]]:
                yield dict(head)
                for item in tail:
                    yield dict(item)

            return _chain(first, iterator), sp
        except StopIteration:
            errors.append(f"{sp}:empty_split")
        except Exception as exc:
            errors.append(f"{sp}:{type(exc).__name__}:{exc}")
    err_joined = "; ".join(errors)
    if spec.requires_hf_gate or "gated" in err_joined.lower() or "401" in err_joined or "403" in err_joined:
        raise PermissionError(f"Gated or unauthorized dataset {spec.dataset}: {err_joined}")
    raise RuntimeError(f"Could not load {spec.dataset}: {err_joined}")


def build_candidate_frame_and_sample(
    spec: StatSuiteSpec,
    *,
    sample_size: int,
    seed: int = 20261110,
    max_context_tokens: Optional[int] = 131072,
    max_scan_rows: Optional[int] = None,
    tokenizer: Any = None,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    converter: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan dataset rows, build the verified eligible frame, and draw a stratified sample without replacement."""
    from sparkx_loop72_dlsa_benchmarks import convert_hf_row_to_eval_task, verify_ifeval_constraints

    conv = converter or convert_hf_row_to_eval_task
    scan_cap = max_scan_rows if max_scan_rows is not None else spec.max_scan_rows
    if rows is not None:
        raw_iter: Iterator[Dict[str, Any]] = (dict(r) for r in rows)
        split_used = "injected_rows"
    else:
        raw_iter, split_used = _iter_suite_raw_rows(spec, token=token)

    candidates: List[Dict[str, Any]] = []
    stratum_counts: Dict[str, int] = {}
    rows_seen = 0
    excluded_filter = 0
    excluded_unsupported = 0
    excluded_over_context = 0
    scan_exhausted = True

    while True:
        if scan_cap is not None and rows_seen >= int(scan_cap):
            # Probe one extra element to know whether the split actually ended or was capped!
            try:
                _ = next(raw_iter)
                scan_exhausted = False
            except StopIteration:
                scan_exhausted = True
            break
        try:
            row = next(raw_iter)
        except StopIteration:
            scan_exhausted = True
            break

        idx = rows_seen
        rows_seen += 1

        if spec.name.startswith("graphwalks_bfs"):
            if "problem_type" not in row:
                raise ValueError("GraphWalks row missing problem_type")
            if str(row.get("problem_type", "")).lower() != "bfs":
                excluded_filter += 1
                continue

        task = conv(spec.benchmark_name, row, row_idx=idx, task_mode=spec.task_mode)
        if task is None or task.get("status") != "ready":
            st = None if task is None else task.get("status")
            if st == "filtered":
                excluded_filter += 1
            else:
                excluded_unsupported += 1
            continue

        # Pre-check IFEval rule support so sampled items never hit unimplemented rules
        if spec.name == "ifeval" and isinstance(task.get("ifeval_spec"), dict):
            probe = verify_ifeval_constraints("probe text", task["ifeval_spec"])
            if (probe.detail or {}).get("status") == "unsupported":
                excluded_unsupported += 1
                continue

        est_ptok = _estimate_prompt_tokens(task, tokenizer=tokenizer, max_context_tokens=max_context_tokens)
        if max_context_tokens is not None and est_ptok > int(max_context_tokens):
            excluded_over_context += 1
            continue

        task = dict(task)
        task["reference_adapter_key"] = task.get("benchmark_id", spec.benchmark_name)
        task["benchmark"] = spec.name if spec.task_mode == "official" else f"Sampled Diagnostic / {spec.benchmark_name}"
        task["benchmark_id"] = spec.name
        task["official_spark_x26_score"] = None
        task["official_comparable"] = False

        row_sha = stable_digest(row)
        stratum = _assign_stratum(spec, row, task, row_index=idx, prompt_tokens=est_ptok)
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        candidates.append(
            {
                "row_index": idx,
                "row_sha256": row_sha,
                "source_id": row.get("_id", row.get("id", idx)),
                "stratum": stratum,
                "estimated_prompt_tokens": est_ptok,
                "row": row,
                "task": task,
            }
        )

    suite_seed = _derive_suite_seed(seed, spec.name)
    sampled = stratified_sample_without_replacement(candidates, sample_size=int(sample_size), seed=suite_seed)
    sampled_stratum_counts: Dict[str, int] = {}
    for c in sampled:
        sk = str(c["stratum"])
        sampled_stratum_counts[sk] = sampled_stratum_counts.get(sk, 0) + 1

    frame_meta = {
        "suite": spec.name,
        "dataset": spec.dataset,
        "config_name": spec.config_name,
        "split_used": split_used,
        "revision": spec.revision,
        "files": list(spec.files) if spec.files is not None else None,
        "task_mode": spec.task_mode,
        "seed": int(seed),
        "suite_seed": int(suite_seed),
        "requested_sample_size": int(sample_size),
        "actual_sample_size": len(sampled),
        "rows_seen": rows_seen,
        "scan_cap": int(scan_cap) if scan_cap is not None else None,
        "scan_exhausted": bool(scan_exhausted),
        "max_context_tokens_filter": int(max_context_tokens) if max_context_tokens is not None else None,
        "excluded_by_filter": excluded_filter,
        "excluded_unsupported": excluded_unsupported,
        "excluded_over_max_context_tokens": excluded_over_context,
        "eligible_population_in_frame": len(candidates),
        "population_exact_for_frame": bool(scan_exhausted),
        "full_unfiltered_split_covered": bool(scan_exhausted and excluded_over_context == 0 and excluded_unsupported == 0),
        "stratum_population_counts": stratum_counts,
        "stratum_sample_counts": sampled_stratum_counts,
    }
    return sampled, frame_meta


# ==============================================================================
# 3. Resumable Single-Suite & Multi-Suite Statistical Runners
# ==============================================================================
@contextlib.contextmanager
def frozen_36l_base_context(model: Any, max_dense_tokens: int = 2_097_152) -> Iterator[None]:
    """Execute the loaded 72L wrapper as the exact 36L frozen Spark-X2.6-4B base with dense attention."""
    sparse_cfg = getattr(model, "sparse_attn_cfg", None)
    prev_thresh = getattr(sparse_cfg, "exact_threshold", None) if sparse_cfg is not None else None
    try:
        if sparse_cfg is not None:
            sparse_cfg.exact_threshold = int(max_dense_tokens)
        if hasattr(model, "loop_config"):
            with model.loop_config(passes=1, use_mhc=False, use_engram=False, enable_p3=False, use_esm=False):
                yield
        else:
            yield
    finally:
        if sparse_cfg is not None and prev_thresh is not None:
            sparse_cfg.exact_threshold = prev_thresh


def _spec_identity_dict(spec: StatSuiteSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "benchmark_name": spec.benchmark_name,
        "task_mode": spec.task_mode,
        "dataset": spec.dataset,
        "config_name": spec.config_name,
        "revision": spec.revision,
        "files": list(spec.files) if spec.files is not None else None,
    }


def run_statistical_suite(
    model: Any,
    tokenizer: Any,
    spec: StatSuiteSpec,
    output_dir: str | Path,
    *,
    sample_size: int,
    seed: int = 20261110,
    max_new_tokens: Optional[int] = None,
    max_context_tokens: Optional[int] = 131072,
    max_scan_rows: Optional[int] = None,
    confidence: float = 0.95,
    generation_options: Optional[Dict[str, Any]] = None,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    evaluator: Optional[Callable[..., Dict[str, Any]]] = None,
    converter: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    token: Optional[str] = None,
    verbose: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """Draw a reproducible stratified sample, run/resume per-item evaluation, and compute 95% CIs."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    eff_max_new = int(max_new_tokens if max_new_tokens is not None else spec.default_max_new_tokens)
    if eff_max_new <= 0:
        raise ValueError("max_new_tokens must be positive")

    test_inputs = rows is not None or evaluator is not None or converter is not None
    if evaluator is None:
        from sparkx_loop72_dlsa_benchmarks import evaluate_benchmark_task

        evaluator = evaluate_benchmark_task
    try:
        eval_accepts_verbose = "verbose" in inspect.signature(evaluator).parameters
    except Exception:
        eval_accepts_verbose = False

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    jsonl_path = dest / f"{spec.name}.stat.jsonl"
    sample_manifest_path = dest / f"{spec.name}.sample_manifest.json"
    summary_path = dest / f"{spec.name}.stat_summary.json"

    existing_records: Dict[int, Dict[str, Any]] = {}
    if jsonl_path.exists():
        if not resume:
            raise FileExistsError(f"Output exists and resume=False: {jsonl_path}")
        prev_manifest: Dict[str, Any] = {}
        if sample_manifest_path.exists():
            prev_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
            if (
                prev_manifest.get("spec_identity") != _spec_identity_dict(spec)
                or int(prev_manifest.get("seed", -1)) != int(seed)
                or int(prev_manifest.get("requested_sample_size", -1)) != int(sample_size)
                or prev_manifest.get("max_context_tokens_filter")
                != (int(max_context_tokens) if max_context_tokens is not None else None)
                or bool(prev_manifest.get("injected_test_inputs")) != bool(test_inputs)
            ):
                raise ResumeIntegrityError(
                    f"Cannot resume {jsonl_path}: sample manifest parameters (suite/seed/n/max_context_tokens) changed"
                )
        existing_records = _load_resumable_records(jsonl_path)
        prev_pins_chk = prev_manifest.get("sampled_items") or []
        if (
            not test_inputs
            and prev_pins_chk
            and summary_path.exists()
            and len(existing_records) == len(prev_pins_chk)
            and all(
                int(p["row_index"]) in existing_records
                and existing_records[int(p["row_index"])].get("row_sha256") == p.get("row_sha256")
                for p in prev_pins_chk
            )
        ):
            prev_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if verbose:
                st_m = (prev_summary.get("statistics") or {}).get("mean")
                st_moe = (prev_summary.get("statistics") or {}).get("margin_of_error")
                print(
                    f"  [{spec.name} | resume] All {len(prev_pins_chk)} sampled items already verified "
                    f"(mean={st_m}%, ±{st_moe}%); skipping.",
                    flush=True,
                )
            return prev_summary

    started = time.perf_counter()
    try:
        sampled, frame_meta = build_candidate_frame_and_sample(
            spec,
            sample_size=int(sample_size),
            seed=int(seed),
            max_context_tokens=max_context_tokens,
            max_scan_rows=max_scan_rows,
            tokenizer=tokenizer,
            rows=rows,
            converter=converter,
            token=token,
        )
    except PermissionError as exc:
        blocked_summary = {
            "suite": spec.name,
            "benchmark_name": spec.benchmark_name,
            "category": spec.category,
            "task_mode": spec.task_mode,
            "sampling_mode": "statistical_subset",
            "status": "blocked_gated_dataset",
            "complete": False,
            "official_comparable": False,
            "published_score_comparable": False,
            "reason": str(exc),
            "statistics": compute_statistical_summary([]),
        }
        summary_path.write_text(json.dumps(blocked_summary, indent=2) + "\n", encoding="utf-8")
        if verbose:
            print(f"  [{spec.name}] SKIPPED (gated dataset requires accepted HF_TOKEN): {exc}", flush=True)
        return blocked_summary

    # Verify sample_manifest pin if resuming
    sample_pins = [
        {
            "sample_ordinal": ord_i,
            "row_index": int(c["row_index"]),
            "row_sha256": str(c["row_sha256"]),
            "source_id": c["source_id"],
            "stratum": str(c["stratum"]),
        }
        for ord_i, c in enumerate(sampled)
    ]
    if existing_records and sample_manifest_path.exists():
        prev_pins = json.loads(sample_manifest_path.read_text(encoding="utf-8")).get("sampled_items", [])
        if prev_pins and [(p["row_index"], p["row_sha256"]) for p in prev_pins] != [
            (p["row_index"], p["row_sha256"]) for p in sample_pins
        ]:
            raise ResumeIntegrityError(
                f"Cannot resume {jsonl_path}: sampled item digests differ from pinned {sample_manifest_path}"
            )

    options: Dict[str, Any] = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "force_think_close": False,
    }
    options.update(generation_options or {})

    manifest_payload = {
        "spec_identity": _spec_identity_dict(spec),
        "seed": int(seed),
        "requested_sample_size": int(sample_size),
        "max_context_tokens_filter": int(max_context_tokens) if max_context_tokens is not None else None,
        "injected_test_inputs": bool(test_inputs),
        "generation": {**options, "max_new_tokens": eff_max_new},
        "frame": frame_meta,
        "sampled_items": sample_pins,
        "source": source_identity(),
        "model_manifest": _jsonable(model.execution_manifest()) if hasattr(model, "execution_manifest") else None,
        "sampling_mode": "statistical_subset",
        "complete": False,
        "official_comparable": False,
        "published_score_comparable": False,
    }
    sample_manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Reject any orphan rows in existing JSONL that aren't in our sampled set
    expected_sha_by_idx = {int(c["row_index"]): str(c["row_sha256"]) for c in sampled}
    for r_idx, prev_rec in existing_records.items():
        if r_idx not in expected_sha_by_idx:
            raise ResumeIntegrityError(f"Unexpected row_index={r_idx} in {jsonl_path} not present in sampled frame")
        if prev_rec.get("row_sha256") != expected_sha_by_idx[r_idx]:
            raise ResumeIntegrityError(
                f"Resume integrity mismatch at {spec.name} row_index={r_idx}: "
                f"expected {expected_sha_by_idx[r_idx]}, got {prev_rec.get('row_sha256')}"
            )

    scores: List[float] = []
    stratum_scores: Dict[str, List[float]] = {}
    gen_token_counts: List[int] = []
    decode_tps_vals: List[float] = []
    counts = {"sampled": len(sampled), "attempted": 0, "scored": 0, "blocked": 0, "errors": 0, "resumed": 0}
    pop_for_fpc = int(frame_meta["eligible_population_in_frame"]) if frame_meta["population_exact_for_frame"] else None

    file_mode = "a" if existing_records else "w"
    with jsonl_path.open(file_mode, encoding="utf-8") as handle:
        for ord_i, cand in enumerate(sampled, start=1):
            r_idx = int(cand["row_index"])
            r_sha = str(cand["row_sha256"])
            stratum = str(cand["stratum"])
            counts["attempted"] += 1

            if r_idx in existing_records:
                rec = existing_records[r_idx]
                counts["resumed"] += 1
                if rec.get("score") is not None and rec.get("status") == "completed":
                    sc_f = float(rec["score"])
                    scores.append(sc_f)
                    stratum_scores.setdefault(stratum, []).append(sc_f)
                    counts["scored"] += 1
                elif rec.get("status") in ("error", "generation_error", "invalid"):
                    counts["errors"] += 1
                else:
                    counts["blocked"] += 1
                if isinstance(rec.get("generated_tokens"), int):
                    gen_token_counts.append(int(rec["generated_tokens"]))
                t_info = rec.get("timing") or {}
                if t_info.get("decode_tok_per_s"):
                    decode_tps_vals.append(float(t_info["decode_tok_per_s"]))
                continue

            if verbose:
                print(
                    f"  [{spec.name} | item {ord_i}/{len(sampled)} | row_idx={r_idx} | stratum={stratum}] "
                    f"evaluating (est_prompt={cand['estimated_prompt_tokens']} tok, max_new={eff_max_new})...",
                    flush=True,
                )

            t0 = time.perf_counter()
            record: Dict[str, Any] = {
                "sample_ordinal": ord_i - 1,
                "row_index": r_idx,
                "row_sha256": r_sha,
                "source_id": cand["source_id"],
                "stratum": stratum,
            }
            try:
                eval_kwargs: Dict[str, Any] = {"max_new_tokens": eff_max_new, "generation_options": options}
                if eval_accepts_verbose and verbose:
                    eval_kwargs["verbose"] = True
                res = evaluator(model, tokenizer, cand["task"], **eval_kwargs)
                if res.get("score") is not None:
                    if res.get("status") != "completed":
                        raise ValueError("Numeric score requires status='completed'")
                    sc_f = float(res["score"])
                    if not (0.0 <= sc_f <= 100.0):
                        raise ValueError(f"Score out of [0, 100]: {sc_f}")
                    scores.append(sc_f)
                    stratum_scores.setdefault(stratum, []).append(sc_f)
                    counts["scored"] += 1
                elif res.get("status") in ("error", "generation_error", "invalid"):
                    counts["errors"] += 1
                else:
                    counts["blocked"] += 1
                record.update(_jsonable(res))
            except Exception as exc:
                counts["errors"] += 1
                record.update(status="error", score=None, reason=f"{type(exc).__name__}: {exc}")

            if isinstance(record.get("generated_tokens"), int):
                gen_token_counts.append(int(record["generated_tokens"]))
            timing = record.get("timing") or {}
            if timing.get("decode_tok_per_s"):
                decode_tps_vals.append(float(timing["decode_tok_per_s"]))

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            if verbose:
                dt = time.perf_counter() - t0
                live_stat = compute_statistical_summary(
                    scores,
                    population_size=pop_for_fpc,
                    population_exact=bool(frame_meta["population_exact_for_frame"]),
                    confidence=confidence,
                    n_bootstrap=2000,
                    seed=seed,
                )
                sc_disp = f"{float(record['score']):.2f}" if record.get("score") is not None else str(record.get("status"))
                if live_stat["mean"] is not None:
                    ci_disp = (
                        f"mean={live_stat['mean']:.2f}% "
                        f"[95% CI: {live_stat['ci_low']:.1f}%..{live_stat['ci_high']:.1f}%, "
                        f"±{live_stat['margin_of_error']:.1f}%, FPC={live_stat['fpc_factor']:.2f}]"
                    )
                else:
                    ci_disp = "mean=N/A"
                dec_tps = float(timing.get("decode_tok_per_s", 0.0) or 0.0)
                g_tok = record.get("generated_tokens", "?")
                p_tok = record.get("prompt_tokens", cand["estimated_prompt_tokens"])
                print(
                    f"  [{spec.name} | {ord_i}/{len(sampled)} done] score={sc_disp} | {ci_disp} | "
                    f"prompt={p_tok} tok | gen={g_tok} tok | decode={dec_tps:.1f} tok/s | wall={dt:.1f}s",
                    flush=True,
                )

    stat = compute_statistical_summary(
        scores,
        population_size=pop_for_fpc,
        population_exact=bool(frame_meta["population_exact_for_frame"]),
        confidence=confidence,
        n_bootstrap=10000,
        seed=_derive_suite_seed(seed, f"boot::{spec.name}"),
    )
    strat_stat = (
        compute_stratified_statistical_summary(
            stratum_scores,
            frame_meta["stratum_population_counts"],
            confidence=confidence,
        )
        if frame_meta["population_exact_for_frame"]
        else None
    )

    median_gen = float(statistics.median(gen_token_counts)) if gen_token_counts else None
    mean_tps = (sum(decode_tps_vals) / len(decode_tps_vals)) if decode_tps_vals else None

    summary = {
        "suite": spec.name,
        "benchmark_name": spec.benchmark_name,
        "category": spec.category,
        "task_mode": spec.task_mode,
        "metric_label": spec.metric_label,
        "sampling_mode": "statistical_subset",
        "status": "sampled_completed" if counts["scored"] > 0 else "no_scored_rows",
        "complete": False,
        "official_comparable": False,
        "published_score_comparable": False,
        "intended_model_run": not test_inputs,
        "counts": counts,
        "frame": frame_meta,
        "statistics": stat,
        "stratified_statistics": strat_stat,
        "median_generated_tokens": median_gen,
        "mean_decode_tok_per_s": round(mean_tps, 2) if mean_tps is not None else None,
        "reference_spark_x26_36l_card": spec.reference_spark_x26_36l,
        "reference_note": spec.reference_note,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "sample_manifest": str(sample_manifest_path),
        "predictions": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def format_statistical_markdown_table(report: Mapping[str, Any]) -> str:
    """Format a publication-style Markdown summary table with 95% CIs, FPC, and honesty disclaimers."""
    lines = [
        "| Suite | Mode | Scored `n / N_frame` | Mean (%) | 95% CI (%) | ±MoE | FPC | Med Gen Tok | Decode tok/s | 36L Card Ref* |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in report.get("suites", []):
        name = s.get("suite", "?")
        mode = s.get("task_mode", "?")
        st = s.get("status", "")
        if st == "blocked_gated_dataset":
            lines.append(f"| `{name}` | `{mode}` | gated | — | *(requires HF_TOKEN gate)* | — | — | — | — | {s.get('reference_spark_x26_36l_card') or '—'} |")
            continue
        stat = s.get("statistics") or {}
        frame = s.get("frame") or {}
        n = stat.get("n", 0)
        big_n = frame.get("eligible_population_in_frame", "?")
        exact_mark = "" if frame.get("population_exact_for_frame") else "+"
        mean = f"{stat['mean']:.2f}" if stat.get("mean") is not None else "—"
        ci = f"[{stat['ci_low']:.1f}, {stat['ci_high']:.1f}]" if stat.get("ci_low") is not None else "—"
        moe = f"±{stat['margin_of_error']:.1f}" if stat.get("margin_of_error") is not None else "—"
        fpc = f"{stat['fpc_factor']:.2f}" if stat.get("fpc_factor") is not None else "—"
        med_tok = f"{int(s['median_generated_tokens'])}" if s.get("median_generated_tokens") is not None else "—"
        tps = f"{s['mean_decode_tok_per_s']:.1f}" if s.get("mean_decode_tok_per_s") is not None else "—"
        ref = f"{s['reference_spark_x26_36l_card']:.1f}" if s.get("reference_spark_x26_36l_card") is not None else "—"
        lines.append(
            f"| `{name}` | `{mode}` | `{n} / {big_n}{exact_mark}` | **{mean}** | `{ci}` | `{moe}` | `{fpc}` | `{med_tok}` | `{tps}` | `{ref}` |"
        )
    lines.append("")
    lines.append(
        "> *\\*Scope & Honesty Note:* `sampling_mode='statistical_subset'`, `complete=False`, `official_comparable=False`. "
        "`official` suites use exact upstream prompt/scoring adapters on a stratified subset within `max_context_tokens`; "
        "`diagnostic` suites use local deterministic math/MCQ/constraint/AST verifiers. `36L Card Ref` is listed for context only "
        "(evaluated under XHToken's full-split harness; use `run_paired_ablation_suite` for an apples-to-apples paired 72L vs 36L test on the exact same sampled items)."
    )
    return "\n".join(lines)


def run_statistical_benchmarks(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    preset: str = "quick",
    suites: Optional[Sequence[str]] = None,
    sample_size_overrides: Optional[Mapping[str, int]] = None,
    max_new_tokens_overrides: Optional[Mapping[str, int]] = None,
    max_context_tokens: Optional[int] = 131072,
    seed: int = 20261110,
    confidence: float = 0.95,
    generation_options: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    verbose: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run multiple statistically sampled suites with resumable per-item JSONL checkpoints."""
    if preset not in PRESET_NAMES:
        raise ValueError(f"Unknown preset {preset!r}; choose from {PRESET_NAMES}")
    suite_names = list(suites or DEFAULT_CORE_SUITES)
    unknown = set(suite_names) - set(STAT_SUITE_SPECS)
    if unknown:
        raise ValueError(f"Unknown statistical suites: {sorted(unknown)}")

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    n_over = dict(sample_size_overrides or {})
    tok_over = dict(max_new_tokens_overrides or {})

    for idx, name in enumerate(suite_names, start=1):
        spec = STAT_SUITE_SPECS[name]
        planned_n = int(n_over.get(name, spec.preset_n[preset]))
        max_new = int(tok_over.get(name, spec.default_max_new_tokens))
        if verbose:
            print(
                f"\n=== [StatBench {idx}/{len(suite_names)}] Suite '{name}' "
                f"(mode={spec.task_mode}, preset={preset}, planned_n={planned_n}, "
                f"max_new_tokens={max_new}, max_context={max_context_tokens}, resume={resume}) ===",
                flush=True,
            )
        entry = run_statistical_suite(
            model,
            tokenizer,
            spec,
            dest,
            sample_size=planned_n,
            seed=seed,
            max_new_tokens=max_new,
            max_context_tokens=max_context_tokens,
            confidence=confidence,
            generation_options=generation_options,
            token=token,
            verbose=verbose,
            resume=resume,
        )
        results.append(entry)

    report: Dict[str, Any] = {
        "mode": "statistical_subset",
        "sampling_mode": "statistical_subset",
        "preset": preset,
        "seed": int(seed),
        "confidence": float(confidence),
        "max_context_tokens": int(max_context_tokens) if max_context_tokens is not None else None,
        "complete": False,
        "official_comparable": False,
        "published_score_comparable": False,
        "macro_score": None,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "suites": results,
    }
    md_table = format_statistical_markdown_table(report)
    report["markdown_table"] = md_table
    (dest / "statistical_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest / "statistical_report.md").write_text(md_table + "\n", encoding="utf-8")
    if verbose:
        print("\n" + md_table, flush=True)
    return report


def run_paired_ablation_suite(
    model: Any,
    tokenizer: Any,
    suite_name: str,
    output_dir: str | Path,
    *,
    sample_size: int = 12,
    seed: int = 20261110,
    max_new_tokens: Optional[int] = None,
    max_context_tokens: int = 65536,
    confidence: float = 0.95,
    generation_options: Optional[Dict[str, Any]] = None,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    evaluator: Optional[Callable[..., Dict[str, Any]]] = None,
    converter: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    token: Optional[str] = None,
    verbose: bool = True,
    resume: bool = True,
) -> Dict[str, Any]:
    """Evaluate 72L Recursive + D-LSA and 36L Frozen Base on the EXACT SAME sampled items (Paired CRN)."""
    if suite_name not in STAT_SUITE_SPECS:
        raise ValueError(f"Unknown suite {suite_name!r}")
    spec = STAT_SUITE_SPECS[suite_name]
    dest = Path(output_dir)
    dir_72l = dest / "paired_72l_recursive"
    dir_36l = dest / "paired_36l_frozen_base"
    rows_materialized = list(rows) if rows is not None else None

    if verbose:
        print(f"\n=== [Paired CRN 1/2] Evaluating 72L Recursive + mHC + D-LSA on '{suite_name}' (n={sample_size}) ===", flush=True)
    res_72l = run_statistical_suite(
        model,
        tokenizer,
        spec,
        dir_72l,
        sample_size=sample_size,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_context_tokens=max_context_tokens,
        confidence=confidence,
        generation_options=generation_options,
        rows=rows_materialized,
        evaluator=evaluator,
        converter=converter,
        token=token,
        verbose=verbose,
        resume=resume,
    )

    if verbose:
        print(f"\n=== [Paired CRN 2/2] Evaluating 36L Frozen Base (exact dense) on IDENTICAL '{suite_name}' sample ===", flush=True)
    with frozen_36l_base_context(model, max_dense_tokens=max(131072, int(max_context_tokens) + 16384)):
        res_36l = run_statistical_suite(
            model,
            tokenizer,
            spec,
            dir_36l,
            sample_size=sample_size,
            seed=seed,
            max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
            confidence=confidence,
            generation_options=generation_options,
            rows=rows_materialized,
            evaluator=evaluator,
            converter=converter,
            token=token,
            verbose=verbose,
            resume=resume,
        )

    recs_72 = _load_resumable_records(Path(res_72l["predictions"])) if Path(res_72l.get("predictions", "")).is_file() else {}
    recs_36 = _load_resumable_records(Path(res_36l["predictions"])) if Path(res_36l.get("predictions", "")).is_file() else {}
    shared_indices = sorted(set(recs_72.keys()) & set(recs_36.keys()))
    paired_a: List[float] = []
    paired_b: List[float] = []
    for idx in shared_indices:
        r72, r36 = recs_72[idx], recs_36[idx]
        if r72.get("row_sha256") != r36.get("row_sha256"):
            raise ResumeIntegrityError(f"Paired row digest mismatch at row_index={idx}")
        if r72.get("score") is not None and r36.get("score") is not None:
            paired_a.append(float(r72["score"]))
            paired_b.append(float(r36["score"]))

    diff_stat = compute_paired_bootstrap_diff(
        paired_a,
        paired_b,
        label_a="72L_Recursive_DLSA",
        label_b="36L_Frozen_Base",
        confidence=confidence,
        seed=_derive_suite_seed(seed, f"paired::{suite_name}"),
    )
    out = {
        "suite": suite_name,
        "sampling_mode": "paired_common_random_numbers",
        "complete": False,
        "official_comparable": False,
        "paired_comparison": diff_stat,
        "summary_72l": res_72l,
        "summary_36l": res_36l,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{suite_name}.paired_comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Statistically sampled benchmark runner for Spark X2.6 4B Recursive72.")
    parser.add_argument("--estimate-only", action="store_true", help="Print sample size & GPU runtime estimate table without loading model.")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="quick")
    parser.add_argument("--suite", action="append", dest="suites", help="Specific suite(s) to evaluate (repeatable).")
    parser.add_argument("--checkpoint", help="Local export directory or HF repo id (e.g. TassP0lar/Spark-X2.6-4B-Recursive72).")
    parser.add_argument("--dlsa-checkpoint", default="visit_selectors_stage_e.pt")
    parser.add_argument("--output-dir", default="runs/stat_benchmarks")
    parser.add_argument("--max-context-tokens", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    if args.estimate_only:
        est = estimate_preset_budget_table(args.preset, suites=args.suites)
        print(json.dumps(est, indent=2))
        return 0

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --estimate-only is set")

    from load_inference_v03 import load_sparkx_loop72_lsa

    model, tokenizer = load_sparkx_loop72_lsa(
        repo_or_dir=args.checkpoint,
        base_revision=MODEL_REVISION,
        dlsa_checkpoint=args.dlsa_checkpoint,
        recurrence_mode="learned_mhc",
        strict_selector_provenance=True,
        require_verified_selectors=True,
    )
    run_statistical_benchmarks(
        model,
        tokenizer,
        args.output_dir,
        preset=args.preset,
        suites=args.suites,
        max_context_tokens=args.max_context_tokens,
        seed=args.seed,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
