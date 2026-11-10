"""train_spark_dlsa.py — Per-Visit Streamed-Gram PCA Initialization, Matched-Support Sampled-Query KL Distillation,
Standalone Stage E CLI & 4-Way Positional Evaluation for Spark X2.6 4B Loop72 LSA (Work Packages WP3, WP4, P5 & P8).

Implements §8.1–§8.6 of the 10 Nov 2026 Spark X2.6 4B Loop72-DLSA Review & Development Plan:
  1. Per-visit native SDPA Q/K capture (`capture_loop72_global_visit_qkv` & `capture_loop72_visit_distill_dataset`)
     using `_sparkx_qkv_capture_callback` across all 18 global execution visits (`{3:1, 7:2, 11:2, 15:3, 19:3, 23:3, 27:2, 31:1, 35:1}`),
     sampling long-range query rows during capture without retaining `V` or full `[H_q, T, D]` query tensors, and failing
     closed if any requested global visit is missing (`F01, F17`).
  2. Constant-memory streamed `768 x 768` covariance/Gram PCA initialization (`fit_loop72_visit_bank_pca`) on the 192-D
     RoPE-free key slice (`[64:256)` across `H_kv = 4` heads -> `768` features -> `64` latent dims) plus streamed ridge
     query alignment, avoiding `torch.cat` over captured histories (`F17`).
  3. Matched-support KL distillation (`distill_single_visit_indexer` & `distill_loop72_visit_bank`) restricting and
     renormalizing teacher attention over the exact runtime-selectable long-range support `M_t`, scoring student logits
     via `FaithfulDLSAIndexer.score_selectable_logits` so `W_q`, `W_k`, `log_tau`, and `raw_dist_alpha` all receive real
     gradients and match runtime serving (`F09`), with held-out KL early stopping, fixed-budget mass recall (`1024/2048/4096`),
     bucket-runtime recall, canonical SHA-256 provenance, and fail-closed `18/18` completion verification (`F01, F04`).
  4. Standalone Stage E CLI (`python train_spark_dlsa.py ...`) so selector calibration/distillation can be rerun directly
     from any accepted dense mHC checkpoint without retraining Stages B/C/D (`§8.6`).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

if HAS_TORCH:
    from sparkx_dlsa import (
        LOOP72_GLOBAL_VISIT_SCHEDULE,
        PagedLatentIndex,
        VisitAwareDLSAIndexBank,
        canonical_state_dict_sha256,
    )


@dataclasses.dataclass
class VisitDistillSample:
    """Bounded per-visit Q/K sample storing only sampled query rows `q_sub` (`[H_q, S, D]`), `q_positions` (`[S]`),
    and `k_full` (`[H_kv, T, D]`), avoiding full-sequence Q and V retention (`§8.4`, `F17`)."""
    q_sub: "torch.Tensor"
    k_full: "torch.Tensor"
    q_positions: "torch.Tensor"
    doc_id: str = ""
    split: str = "train"


SampleLike = Union[Tuple["torch.Tensor", "torch.Tensor"], Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"], VisitDistillSample, Dict[str, Any]]


@dataclasses.dataclass
class VisitDistillConfig:
    """Configuration for 18-visit Loop72 D-LSA selector training (`WP4`, `§8.2–8.6`)."""
    d_idx: int = 64
    lr: float = 1e-3
    scalar_lr: float = 5e-3
    weight_decay: float = 0.01
    steps: int = 200
    sampled_queries: int = 32
    keys_per_step: int = 2048
    teacher_key_chunk: int = 2048
    init_with_pca: bool = True
    tau_teacher: float = 1.0
    tau_student: float = 0.08
    train_scalars: bool = True
    focal_gamma: float = 1.5
    focal_weight: float = 0.0
    log_distance_prior_alpha: float = 0.02
    eval_every_steps: int = 25
    recall_budgets: Tuple[int, ...] = (1024, 2048, 4096)
    require_global_visits: int = 18
    require_long_range_support: bool = False
    fail_on_skipped_visit: bool = True
    seed: int = 42


def _unpack_sample(
    sample: SampleLike,
    max_queries: int,
    n_sink: int,
    window: int,
    n_cluster: int,
    gen: "torch.Generator",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Normalize any supported sample representation into `(q_sub [H_q, S, D], k_full [H_kv, T, D], q_positions [S])`."""
    if isinstance(sample, VisitDistillSample):
        return sample.q_sub, sample.k_full, sample.q_positions.to(device=sample.q_sub.device, dtype=torch.long)
    if isinstance(sample, dict):
        q_t = sample["q_sub"] if "q_sub" in sample else sample["q"]
        k_t = sample["k_full"] if "k_full" in sample else sample["k"]
        if "q_positions" in sample and sample["q_positions"] is not None:
            return q_t, k_t, sample["q_positions"].to(device=q_t.device, dtype=torch.long)
        sample = (q_t, k_t)
    if isinstance(sample, tuple) and len(sample) == 3:
        q_t, k_t, qp = sample
        return q_t, k_t, qp.to(device=q_t.device, dtype=torch.long)

    q_full, k_full = sample  # type: ignore[misc]
    t_seq = int(q_full.shape[1])
    min_long_q = int(n_sink) + int(window) + int(n_cluster)
    if t_seq > min_long_q:
        cand = torch.arange(min_long_q, t_seq, dtype=torch.long)
    elif t_seq > int(n_sink) + int(window):
        cand = torch.arange(int(n_sink) + int(window), t_seq, dtype=torch.long)
    else:
        cand = torch.arange(1, max(2, t_seq), dtype=torch.long)
    n_take = min(max(1, int(max_queries)), int(cand.numel()))
    perm = torch.randperm(int(cand.numel()), generator=gen)[:n_take]
    q_pos = cand.index_select(0, perm).sort().values.to(device=q_full.device)
    return q_full.index_select(1, q_pos), k_full, q_pos


def _compute_chunked_teacher_probs_on_support(
    q_sub: "torch.Tensor",
    k_full: "torch.Tensor",
    q_pos: "torch.Tensor",
    support_mask: "torch.Tensor",
    tau_teacher: float = 1.0,
    key_chunk: int = 2048,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Compute head-averaged causal teacher probabilities restricted and renormalized on `support_mask` (`M_t`, `§8.2`)
    using streaming log-sum-exp across key chunks (`§8.4`) so `[H_q, S, T]` never has to be allocated at once."""
    h_q, n_samp, d_h = q_sub.shape
    h_kv, t_seq, _ = k_full.shape
    g_sz = max(1, h_q // max(1, h_kv))
    dev = q_sub.device
    scale = 1.0 / (math.sqrt(float(d_h)) * max(1e-4, float(tau_teacher)))
    q_g = q_sub.float().view(h_kv, g_sz, n_samp, d_h)
    q_col = q_pos.to(device=dev, dtype=torch.long).view(1, 1, n_samp, 1)
    step = max(256, int(key_chunk))

    # Pass 1: streaming causal max & sum-exp per `(h_kv, g, s)` over full causal `[0..q_pos]`
    row_max = torch.full((h_kv, g_sz, n_samp, 1), float("-inf"), device=dev, dtype=torch.float32)
    row_sum = torch.zeros((h_kv, g_sz, n_samp, 1), device=dev, dtype=torch.float32)
    max_q_upto = min(t_seq, int(q_pos.max().item()) + 1) if n_samp > 0 else t_seq

    for ks in range(0, max_q_upto, step):
        ke = min(max_q_upto, ks + step)
        k_chunk = k_full[:, ks:ke, :].to(device=dev, dtype=torch.float32)
        k_idx = torch.arange(ks, ke, device=dev, dtype=torch.long).view(1, 1, 1, ke - ks)
        c_mask = k_idx <= q_col
        logits = torch.einsum("hgsd,hktd->hgs t" if False else "hgsd,htd->hgst", q_g, k_chunk) * scale
        logits = logits.masked_fill(~c_mask, float("-inf"))
        chunk_max = logits.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(row_max, chunk_max)
        valid_new = torch.isfinite(new_max)
        safe_new = torch.where(valid_new, new_max, torch.zeros_like(new_max))
        safe_old = torch.where(torch.isfinite(row_max), row_max, safe_new)
        exp_old = torch.where(torch.isfinite(row_max), torch.exp(safe_old - safe_new), torch.zeros_like(row_sum))
        exp_cur = torch.where(c_mask, torch.exp(logits - safe_new), torch.zeros_like(logits)).sum(dim=-1, keepdim=True)
        row_sum = row_sum * exp_old + exp_cur
        row_max = new_max

    log_denom = torch.where(row_sum > 0, row_max + row_sum.clamp(min=1e-30).log(), torch.zeros_like(row_max))

    # Pass 2: materialize head-averaged probabilities on `[S, T]` and restrict/renormalize to `support_mask` (`M_t`)
    avg_probs = torch.zeros((n_samp, t_seq), device=dev, dtype=torch.float32)
    for ks in range(0, max_q_upto, step):
        ke = min(max_q_upto, ks + step)
        sup_sub = support_mask[:, ks:ke]
        if not bool(sup_sub.any().item()):
            continue
        k_chunk = k_full[:, ks:ke, :].to(device=dev, dtype=torch.float32)
        k_idx = torch.arange(ks, ke, device=dev, dtype=torch.long).view(1, 1, 1, ke - ks)
        c_mask = k_idx <= q_col
        logits = torch.einsum("hgsd,htd->hgst", q_g, k_chunk) * scale
        probs_chunk = torch.where(c_mask, torch.exp(logits - log_denom), torch.zeros_like(logits)).mean(dim=(0, 1))
        avg_probs[:, ks:ke] = probs_chunk * sup_sub.float()

    sup_mass = avg_probs.sum(dim=-1, keepdim=True)  # [S, 1] fraction of total causal mass inside `M_t`
    valid_rows = (support_mask.any(dim=-1)) & (sup_mass.squeeze(-1) > 1e-9)
    renorm_probs = torch.where(
        valid_rows.unsqueeze(-1),
        avg_probs / sup_mass.clamp(min=1e-12),
        torch.zeros_like(avg_probs),
    )
    return renorm_probs, valid_rows


def capture_loop72_global_visit_qkv(
    model: Any,
    input_ids: "torch.LongTensor",
    visit_indices: Optional[Sequence[int]] = None,
    *,
    include_values: bool = True,
    max_sampled_queries: Optional[int] = None,
    store_device: Optional[Union[str, "torch.device"]] = None,
    store_dtype: Optional["torch.dtype"] = None,
    strict: bool = True,
    allow_unpatched_proj_fallback: bool = True,
    seed: int = 42,
) -> Dict[int, Any]:
    """Run a single no-grad Loop72 forward pass and capture rotated `(Q, K[, V])` at requested global visits `0..17` (`F01`).

    Uses `_sparkx_qkv_capture_callback` invoked by `fused_sdpa_attention_forward` (`native_differentiable_sdpa`) and
    `attach_loop72_dlsa_attention`, temporarily wrapping unpatched `eager_attention_forward` if needed, and failing
    closed (`strict=True`) if any requested global visit was not captured.
    """
    if not HAS_TORCH:
        raise RuntimeError("capture_loop72_global_visit_qkv requires PyTorch.")
    target_set = (
        set(range(len(LOOP72_GLOBAL_VISIT_SCHEDULE)))
        if visit_indices is None
        else {int(v) for v in visit_indices}
    )
    if not target_set:
        return {}

    layers = model.base.model.layers
    global_layers = sorted({int(v.physical_layer) for v in LOOP72_GLOBAL_VISIT_SCHEDULE})
    visit_counters: Dict[int, int] = {l_idx: 0 for l_idx in global_layers}
    captured: Dict[int, Any] = {}
    hooks = []
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    layer_occ_to_gv: Dict[Tuple[int, int], int] = {
        (int(v.physical_layer), int(v.visit_occurrence)): int(v.global_visit_idx)
        for v in LOOP72_GLOBAL_VISIT_SCHEDULE
    }
    virtual_to_gv: Dict[int, int] = {
        int(v.virtual_visit): int(v.global_visit_idx)
        for v in LOOP72_GLOBAL_VISIT_SCHEDULE
    }

    attn_mod = importlib.import_module(type(layers[0].self_attn).__module__) if len(layers) > 0 else None
    orig_eager_fn = getattr(attn_mod, "eager_attention_forward", None) if attn_mod is not None else None
    wrapped_unpatched_eager = False
    if (
        attn_mod is not None
        and callable(orig_eager_fn)
        and not bool(getattr(attn_mod, "_sparkx_sdpa_patched", False))
    ):
        def _wrapped_eager(module, query, key, value, *args, **kwargs):
            cb = getattr(module, "_sparkx_qkv_capture_callback", None)
            if callable(cb):
                cb(
                    module=module,
                    visit_context=getattr(module, "_sparkx_active_visit", None),
                    query=query.detach(),
                    key=key.detach(),
                    value=value.detach(),
                )
            return orig_eager_fn(module, query, key, value, *args, **kwargs)

        attn_mod.eager_attention_forward = _wrapped_eager
        wrapped_unpatched_eager = True

    def _store_tensor(t: "torch.Tensor") -> "torch.Tensor":
        out = t.detach()
        if store_dtype is not None and out.is_floating_point():
            out = out.to(dtype=store_dtype)
        if store_device is not None:
            out = out.to(device=store_device)
        return out.contiguous()

    def _record_visit(l_idx: int, visit_context: Any, q_t: "torch.Tensor", k_t: "torch.Tensor", v_t: Optional["torch.Tensor"]) -> None:
        gv_idx: Optional[int] = None
        if isinstance(visit_context, tuple) and len(visit_context) >= 3:
            v_virt = int(visit_context[1])
            v_ord = int(visit_context[2])
            gv_idx = virtual_to_gv.get(v_virt, layer_occ_to_gv.get((l_idx, v_ord)))
            visit_counters[l_idx] = max(visit_counters[l_idx], v_ord + 1)
        else:
            occ = visit_counters[l_idx]
            visit_counters[l_idx] = occ + 1
            gv_idx = layer_occ_to_gv.get((l_idx, occ))
        if gv_idx is None or gv_idx not in target_set or gv_idx in captured:
            return

        q0 = q_t[0] if q_t.ndim == 4 else q_t
        k0 = k_t[0] if k_t.ndim == 4 else k_t
        v0 = (v_t[0] if v_t.ndim == 4 else v_t) if (include_values and v_t is not None) else None

        if max_sampled_queries is not None and int(max_sampled_queries) > 0:
            t_seq = int(q0.shape[1])
            min_q = 4 + 512 + 512 if t_seq > (4 + 512 + 512) else (4 + 512 if t_seq > 516 else 1)
            cand = torch.arange(min_q, max(min_q + 1, t_seq), dtype=torch.long)
            n_take = min(int(max_sampled_queries), int(cand.numel()))
            perm = torch.randperm(int(cand.numel()), generator=gen)[:n_take]
            q_pos_cpu = cand.index_select(0, perm).sort().values
            q_sub = _store_tensor(q0.index_select(1, q_pos_cpu.to(q0.device)))
            k_stored = _store_tensor(k0)
            captured[gv_idx] = VisitDistillSample(
                q_sub=q_sub,
                k_full=k_stored,
                q_positions=q_pos_cpu.to(device=q_sub.device, dtype=torch.long),
            )
            return

        q_s = _store_tensor(q0)
        k_s = _store_tensor(k0)
        if include_values and v0 is not None:
            captured[gv_idx] = (q_s, k_s, _store_tensor(v0))
        else:
            captured[gv_idx] = (q_s, k_s)

    def _make_cb(l_idx: int):
        def _cb(*, module=None, visit_context=None, query=None, key=None, value=None):
            if query is not None and key is not None:
                if module is not None:
                    module._sparkx_cb_fired_this_call = True
                _record_visit(l_idx, visit_context, query, key, value)
        return _cb

    def _make_pre_hook():
        def _pre(mod, _args, _kwargs):
            mod._sparkx_cb_fired_this_call = False
        return _pre

    def _make_fallback_post_hook(l_idx: int):
        def _hook(mod, _args, _kwargs, _out):
            if bool(getattr(mod, "_sparkx_cb_fired_this_call", False)):
                mod._sparkx_cb_fired_this_call = False
                mod._sleep_last_qkv = None
                return
            qkv = getattr(mod, "_sleep_last_qkv", None)
            if qkv is not None:
                q_t, k_t, v_t = qkv
                _record_visit(l_idx, getattr(mod, "_sparkx_active_visit", None), q_t, k_t, v_t)
                mod._sleep_last_qkv = None
                return
            if (
                allow_unpatched_proj_fallback
                and not hasattr(mod, "q_norm")
                and callable(getattr(mod, "q_proj", None))
                and callable(getattr(mod, "k_proj", None))
                and hasattr(mod, "num_heads")
                and hasattr(mod, "head_dim")
            ):
                h_in = _args[0] if _args else _kwargs.get("hidden_states")
                if isinstance(h_in, torch.Tensor) and h_in.ndim == 3:
                    bsz, s_len, _ = h_in.shape
                    n_h = int(mod.num_heads)
                    n_kv = int(getattr(mod, "num_key_value_heads", n_h))
                    h_d = int(mod.head_dim)
                    q_t = mod.q_proj(h_in).view(bsz, s_len, n_h, h_d).transpose(1, 2).detach()
                    k_t = mod.k_proj(h_in).view(bsz, s_len, n_kv, h_d).transpose(1, 2).detach()
                    v_t = (
                        mod.v_proj(h_in).view(bsz, s_len, n_kv, h_d).transpose(1, 2).detach()
                        if (include_values and callable(getattr(mod, "v_proj", None)))
                        else None
                    )
                    _record_visit(l_idx, getattr(mod, "_sparkx_active_visit", None), q_t, k_t, v_t)
        return _hook

    prev_cbs: Dict[int, Any] = {}
    prev_sleep_flags: Dict[int, Tuple[bool, bool, Any]] = {}
    for l_idx in global_layers:
        if 0 <= l_idx < len(layers):
            attn = layers[l_idx].self_attn
            prev_cbs[l_idx] = getattr(attn, "_sparkx_qkv_capture_callback", None)
            prev_sleep_flags[l_idx] = (
                bool(getattr(attn, "_capture_sleep_qkv", False)),
                bool(getattr(attn, "_capture_all_visits_qkv", False)),
                getattr(attn, "_sleep_last_qkv", None),
            )
            attn._sparkx_qkv_capture_callback = _make_cb(l_idx)
            attn._capture_sleep_qkv = True
            attn._capture_all_visits_qkv = True
            attn._sleep_last_qkv = None
            hooks.append(attn.register_forward_pre_hook(_make_pre_hook(), with_kwargs=True))
            hooks.append(attn.register_forward_hook(_make_fallback_post_hook(l_idx), with_kwargs=True))

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            _ = model.forward_hidden(input_ids)
    finally:
        for h in hooks:
            h.remove()
        if wrapped_unpatched_eager and attn_mod is not None and orig_eager_fn is not None:
            attn_mod.eager_attention_forward = orig_eager_fn
        for l_idx in global_layers:
            if 0 <= l_idx < len(layers):
                attn = layers[l_idx].self_attn
                attn._sparkx_qkv_capture_callback = prev_cbs.get(l_idx)
                c_flag, all_flag, prev_qkv = prev_sleep_flags[l_idx]
                attn._capture_sleep_qkv = c_flag
                attn._capture_all_visits_qkv = all_flag
                attn._sleep_last_qkv = prev_qkv
        if was_training:
            model.train()

    if strict:
        missing = sorted(target_set - set(captured.keys()))
        if missing:
            raise RuntimeError(
                f"capture_loop72_global_visit_qkv failed closed: missing {len(missing)}/{len(target_set)} "
                f"global visits {missing}. Ensure Loop72 execution visits all 18 global passes and native SDPA "
                f"capture (`patch_spark_attention_for_long_context`) is active."
            )
    return captured


def capture_loop72_visit_distill_dataset(
    model: Any,
    token_sequences: Sequence["torch.LongTensor"],
    *,
    visit_indices: Optional[Sequence[int]] = None,
    max_sampled_queries: int = 64,
    store_device: Union[str, "torch.device"] = "cpu",
    store_dtype: Optional["torch.dtype"] = None,
    split: str = "train",
    seed: int = 42,
) -> Dict[int, List[VisitDistillSample]]:
    """Capture bounded `VisitDistillSample` shards across `token_sequences` on `store_device` (default CPU BF16,
    no `V`, sampled `Q` rows only) and fail closed if any requested global visit receives zero samples (`§8.4`)."""
    if not HAS_TORCH:
        raise RuntimeError("capture_loop72_visit_distill_dataset requires PyTorch.")
    if store_dtype is None:
        store_dtype = torch.bfloat16
    target_visits = (
        list(range(len(LOOP72_GLOBAL_VISIT_SCHEDULE)))
        if visit_indices is None
        else [int(v) for v in visit_indices]
    )
    out: Dict[int, List[VisitDistillSample]] = {gv: [] for gv in target_visits}
    dev = next(model.base.parameters()).device
    for doc_i, seq in enumerate(token_sequences):
        ids = seq.unsqueeze(0) if seq.ndim == 1 else seq
        ids = ids.to(device=dev, dtype=torch.long)
        cap = capture_loop72_global_visit_qkv(
            model,
            ids,
            visit_indices=target_visits,
            include_values=False,
            max_sampled_queries=max_sampled_queries,
            store_device=store_device,
            store_dtype=store_dtype,
            strict=True,
            seed=int(seed) + doc_i * 997,
        )
        for gv in target_visits:
            samp = cap[gv]
            assert isinstance(samp, VisitDistillSample)
            samp.doc_id = f"{split}_doc_{doc_i:04d}"
            samp.split = str(split)
            out[gv].append(samp)
    for gv in target_visits:
        if not out[gv]:
            raise RuntimeError(f"capture_loop72_visit_distill_dataset: global visit {gv} received 0 samples.")
    return out


@torch.no_grad()
def fit_loop72_visit_bank_pca(
    bank: "VisitAwareDLSAIndexBank",
    visit_qkv_samples: Dict[int, Sequence[SampleLike]],
    *,
    key_chunk_tokens: int = 4096,
    max_align_queries_per_doc: int = 32,
    require_global_visits: int = 0,
    seed: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """Initialize each of the 18 global-visit `FaithfulDLSAIndexer` maps via **streamed `768 x 768` FP64 covariance
    eigendecomposition** (`torch.linalg.eigh`) on RoPE-free key slices followed by streamed ridge Procrustes alignment
    (`W_q` aligned to `W_k` on selectable attended keys), without concatenating key histories (`F17`, `§8.4`)."""
    if not HAS_TORCH:
        raise RuntimeError("fit_loop72_visit_bank_pca requires PyTorch.")
    report: Dict[int, Dict[str, Any]] = {}
    n_total_visits = len(LOOP72_GLOBAL_VISIT_SCHEDULE)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    for gv_idx in range(n_total_visits):
        indexer = bank.get_visit_indexer(gv_idx)
        samples = visit_qkv_samples.get(gv_idx, [])
        if not samples:
            if int(require_global_visits) > 0 and gv_idx < int(require_global_visits):
                raise RuntimeError(
                    f"fit_loop72_visit_bank_pca failed closed: global visit {gv_idx} has no captured samples."
                )
            report[gv_idx] = {"status": "skipped_no_samples", "total_tokens": 0, "aligned_queries": 0}
            continue

        d_in = int(indexer.W_k.weight.shape[1])
        d_idx = int(indexer.W_k.weight.shape[0])
        idx_dev = indexer.W_k.weight.device
        sum_x = torch.zeros((d_in,), dtype=torch.float64, device="cpu")
        sum_xxT = torch.zeros((d_in, d_in), dtype=torch.float64, device="cpu")
        total_tokens = 0
        chunk_sz = max(256, int(key_chunk_tokens))

        # Pass 1: Constant-memory streamed key covariance accumulation (`[768, 768]` = 4.5 MiB on CPU)
        for s_item in samples:
            _, k_s, _ = _unpack_sample(
                s_item, max_align_queries_per_doc, indexer.n_sink, indexer.window, indexer.n_cluster, gen
            )
            t_len = int(k_s.shape[1])
            for ks in range(0, t_len, chunk_sz):
                ke = min(t_len, ks + chunk_sz)
                k_chunk = k_s[:, ks:ke, :].to(device=idx_dev)
                kf_cpu = indexer._free_key_features(k_chunk).detach().to(device="cpu", dtype=torch.float64)
                sum_x += kf_cpu.sum(dim=0)
                sum_xxT += kf_cpu.t() @ kf_cpu
                total_tokens += (ke - ks)

        if total_tokens <= 0:
            raise RuntimeError(f"fit_loop72_visit_bank_pca: visit {gv_idx} accumulated 0 key tokens.")

        mean_x = sum_x / float(total_tokens)
        cov = (sum_xxT / float(total_tokens)) - torch.outer(mean_x, mean_x)
        cov = 0.5 * (cov + cov.t())
        eigvals, eigvecs = torch.linalg.eigh(cov)
        top_vecs = eigvecs[:, -d_idx:].flip(dims=[1]).t().contiguous().to(dtype=torch.float32, device=idx_dev)
        top_vecs = F.normalize(top_vecs, dim=-1, eps=1e-6)
        indexer.W_k.weight.data.copy_(top_vecs.to(indexer.W_k.weight.dtype))
        indexer.W_q.weight.data.copy_(top_vecs.to(indexer.W_q.weight.dtype))

        # Pass 2: Streamed ridge regression (`[768, 768]` Gram + `[768, 64]` RHS) aligning `W_q` to `W_k`
        gram_qq = 1e-2 * torch.eye(d_in, dtype=torch.float64, device="cpu")
        rhs_qk = torch.zeros((d_in, d_idx), dtype=torch.float64, device="cpu")
        aligned_queries = 0

        for s_item in samples:
            q_sub, k_s, q_pos = _unpack_sample(
                s_item, max_align_queries_per_doc, indexer.n_sink, indexer.window, indexer.n_cluster, gen
            )
            q_sub_dev = q_sub.to(device=idx_dev)
            k_dev = k_s.to(device=idx_dev)
            q_pos_dev = q_pos.to(device=idx_dev, dtype=torch.long)
            t_seq = int(k_dev.shape[1])

            sup_mask = indexer.selectable_support_mask(t_seq, q_pos_dev)
            if not bool(sup_mask.any().item()):
                k_ar = torch.arange(t_seq, device=idx_dev).unsqueeze(0)
                sup_mask = k_ar <= q_pos_dev.unsqueeze(1)

            t_probs, valid_rows = _compute_chunked_teacher_probs_on_support(
                q_sub_dev, k_dev, q_pos_dev, sup_mask, tau_teacher=1.0, key_chunk=chunk_sz
            )
            if not bool(valid_rows.any().item()):
                continue
            k_lat_parts: List[torch.Tensor] = []
            for ks in range(0, t_seq, chunk_sz):
                ke = min(t_seq, ks + chunk_sz)
                k_lat_parts.append(indexer.encode_keys(k_dev[:, ks:ke, :]).float())
            k_lat = torch.cat(k_lat_parts, dim=0)  # [T, d_idx]

            qf_valid = indexer._group_free_query_features(q_sub_dev)[valid_rows].float()  # [Sv, 768]
            target_lat = F.normalize(t_probs[valid_rows] @ k_lat, dim=-1, eps=1e-6)       # [Sv, 64]
            qf_cpu = qf_valid.to(device="cpu", dtype=torch.float64)
            tl_cpu = target_lat.to(device="cpu", dtype=torch.float64)
            gram_qq += qf_cpu.t() @ qf_cpu
            rhs_qk += qf_cpu.t() @ tl_cpu
            aligned_queries += int(qf_cpu.shape[0])

        if aligned_queries > 0:
            w_q_T = torch.linalg.solve(gram_qq, rhs_qk).to(device=idx_dev, dtype=torch.float32)  # [768, 64]
            w_pca = indexer.W_k.weight.data.float()
            w_blend = F.normalize(0.65 * w_pca + 0.35 * F.normalize(w_q_T.t(), dim=-1, eps=1e-6), dim=-1, eps=1e-6)
            indexer.W_q.weight.data.copy_(w_blend.to(indexer.W_q.weight.dtype))

        total_eig = float(eigvals.clamp(min=0.0).sum().item())
        captured_eig = float(eigvals[-d_idx:].clamp(min=0.0).sum().item())
        report[gv_idx] = {
            "status": "ok",
            "total_tokens": int(total_tokens),
            "aligned_queries": int(aligned_queries),
            "explained_variance_ratio": round(captured_eig / max(1e-12, total_eig), 6),
        }
    return report


def _prepare_distill_batches(
    indexer: Any,
    samples: Sequence[SampleLike],
    cfg: VisitDistillConfig,
    seed_offset: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build matched-support distillation batches `(q_in, k_in, t_probs, support_mask, q_pos)` on CPU/device (`§8.2`)."""
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + int(seed_offset))
    idx_dev = indexer.W_k.weight.device
    batches: List[Dict[str, Any]] = []
    long_range_queries = 0
    fallback_short_queries = 0
    total_tokens = 0

    with torch.no_grad():
        for s_item in samples:
            q_sub, k_s, q_pos = _unpack_sample(
                s_item, max(int(cfg.sampled_queries) * 2, 32), indexer.n_sink, indexer.window, indexer.n_cluster, gen
            )
            t_seq = int(k_s.shape[1])
            if t_seq < 4 or q_sub.shape[1] == 0:
                continue
            total_tokens += t_seq
            q_sub_dev = q_sub.to(device=idx_dev)
            k_dev = k_s.to(device=idx_dev)
            q_pos_dev = q_pos.to(device=idx_dev, dtype=torch.long)

            sup_mask = indexer.selectable_support_mask(t_seq, q_pos_dev)
            used_short_fallback = False
            if not bool(sup_mask.any().item()):
                if cfg.require_long_range_support:
                    continue
                k_ar = torch.arange(t_seq, device=idx_dev).unsqueeze(0)
                sup_mask = (k_ar <= q_pos_dev.unsqueeze(1)) & (k_ar >= min(indexer.n_sink, max(0, t_seq - 2)))
                if not bool(sup_mask.any().item()):
                    sup_mask = k_ar <= q_pos_dev.unsqueeze(1)
                used_short_fallback = True

            t_probs, valid_rows = _compute_chunked_teacher_probs_on_support(
                q_sub_dev,
                k_dev,
                q_pos_dev,
                sup_mask,
                tau_teacher=cfg.tau_teacher,
                key_chunk=cfg.teacher_key_chunk,
            )
            if not bool(valid_rows.any().item()):
                continue

            q_in_valid = indexer._group_free_query_features(q_sub_dev)[valid_rows].detach()
            k_in_parts: List[torch.Tensor] = []
            c_sz = max(512, int(cfg.teacher_key_chunk))
            for ks in range(0, t_seq, c_sz):
                ke = min(t_seq, ks + c_sz)
                k_in_parts.append(indexer._free_key_features(k_dev[:, ks:ke, :]).detach())
            k_in_full = torch.cat(k_in_full_parts := k_in_parts, dim=0)
            del k_in_full_parts

            n_v = int(q_in_valid.shape[0])
            if used_short_fallback:
                fallback_short_queries += n_v
            else:
                long_range_queries += n_v

            batches.append({
                "q_in": q_in_valid.to(device="cpu", dtype=torch.float32),
                "k_in": k_in_full.to(device="cpu", dtype=torch.bfloat16),
                "t_probs": t_probs[valid_rows].to(device="cpu", dtype=torch.float32),
                "support_mask": sup_mask[valid_rows].to(device="cpu", dtype=torch.bool),
                "q_pos": q_pos_dev[valid_rows].to(device="cpu", dtype=torch.long),
                "is_long_range": not used_short_fallback,
            })

    stats = {
        "num_documents": len(batches),
        "total_tokens": int(total_tokens),
        "long_range_queries": int(long_range_queries),
        "short_fallback_queries": int(fallback_short_queries),
        "total_valid_queries": int(long_range_queries + fallback_short_queries),
    }
    return batches, stats


def _batch_kl_loss(
    indexer: Any,
    batch: Dict[str, Any],
    q_slice_idx: Optional["torch.Tensor"],
    cfg: VisitDistillConfig,
) -> "torch.Tensor":
    dev = indexer.W_k.weight.device
    q_in = batch["q_in"]
    t_probs = batch["t_probs"]
    sup_mask = batch["support_mask"]
    q_pos = batch["q_pos"]
    if q_slice_idx is not None:
        q_in = q_in.index_select(0, q_slice_idx)
        t_probs = t_probs.index_select(0, q_slice_idx)
        sup_mask = sup_mask.index_select(0, q_slice_idx)
        q_pos = q_pos.index_select(0, q_slice_idx)

    q_in_d = q_in.to(device=dev, dtype=torch.float32)
    k_in_d = batch["k_in"].to(device=dev, dtype=torch.float32)
    t_probs_d = t_probs.to(device=dev, dtype=torch.float32)
    sup_d = sup_mask.to(device=dev, dtype=torch.bool)
    q_pos_d = q_pos.to(device=dev, dtype=torch.long)

    if cfg.train_scalars:
        s_logits = indexer.score_selectable_logits(q_in_d, k_in_d, q_pos_d)
    else:
        q_lat = indexer.project_q(q_in_d)
        k_lat = indexer.project_k(k_in_d)
        s_logits = (q_lat @ k_lat.t()) / max(1e-4, float(cfg.tau_student))
        if cfg.log_distance_prior_alpha > 0.0:
            k_idx_f = torch.arange(int(k_in_d.shape[0]), device=dev, dtype=torch.float32).unsqueeze(0)
            dist = (q_pos_d.float().unsqueeze(1) - k_idx_f).clamp(min=0.0) / max(1.0, float(indexer.window))
            s_logits = s_logits - (float(cfg.log_distance_prior_alpha) / max(1e-4, float(cfg.tau_student))) * torch.log1p(dist)

    s_masked = s_logits.masked_fill(~sup_d, float("-inf"))
    log_s = F.log_softmax(s_masked, dim=-1)
    log_s_safe = torch.where(sup_d, log_s, torch.zeros_like(log_s))
    t_log_safe = torch.where(t_probs_d > 1e-12, t_probs_d.clamp(min=1e-12).log(), torch.zeros_like(t_probs_d))
    kl_per_q = (t_probs_d * (t_log_safe - log_s_safe)).sum(dim=-1)
    if float(cfg.focal_weight) > 0.0:
        s_probs = torch.where(sup_d, log_s.exp(), torch.zeros_like(log_s))
        focal_mod = (1.0 - s_probs).clamp(min=0.0) ** float(cfg.focal_gamma)
        focal_kl = (focal_mod * t_probs_d * (t_log_safe - log_s_safe)).sum(dim=-1)
        kl_per_q = (1.0 - float(cfg.focal_weight)) * kl_per_q + float(cfg.focal_weight) * focal_kl
    return kl_per_q.mean()


@torch.no_grad()
def _evaluate_visit_batches(
    indexer: Any,
    batches: Sequence[Dict[str, Any]],
    cfg: VisitDistillConfig,
    eval_bucket_runtime: bool = False,
) -> Dict[str, Any]:
    if not batches:
        return {"kl": 0.0, "mass_recall_by_budget": {}, "bucket_runtime_recall": 0.0}
    dev = indexer.W_k.weight.device
    total_kl = 0.0
    total_q = 0
    budgets = sorted({int(b) for b in cfg.recall_budgets if int(b) > 0})
    recall_sums: Dict[int, float] = {b: 0.0 for b in budgets}
    default_budget = int(getattr(indexer, "k_long_range", 4096))
    if default_budget not in recall_sums:
        budgets.append(default_budget)
        recall_sums[default_budget] = 0.0
    bucket_recall_sum = 0.0
    bucket_recall_q = 0

    for b_item in batches:
        n_q = int(b_item["q_in"].shape[0])
        kl_val = float(_batch_kl_loss(indexer, b_item, None, cfg).item())
        total_kl += kl_val * n_q
        total_q += n_q

        q_in_d = b_item["q_in"].to(device=dev, dtype=torch.float32)
        k_in_d = b_item["k_in"].to(device=dev, dtype=torch.float32)
        t_probs_d = b_item["t_probs"].to(device=dev, dtype=torch.float32)
        sup_d = b_item["support_mask"].to(device=dev, dtype=torch.bool)
        q_pos_d = b_item["q_pos"].to(device=dev, dtype=torch.long)

        s_logits = indexer.score_selectable_logits(q_in_d, k_in_d, q_pos_d).masked_fill(~sup_d, float("-inf"))
        n_sup_cols = int(s_logits.shape[1])
        for b_k in budgets:
            for qi in range(n_q):
                sup_cnt = int(sup_d[qi].sum().item())
                if sup_cnt <= 0:
                    continue
                eff_k = min(b_k, sup_cnt, n_sup_cols)
                if eff_k >= sup_cnt:
                    recall_sums[b_k] += 1.0
                else:
                    top_idx = torch.topk(s_logits[qi], k=eff_k, dim=-1).indices
                    recall_sums[b_k] += float(t_probs_d[qi].index_select(0, top_idx).sum().item())

        if eval_bucket_runtime and b_item.get("is_long_range", False):
            k_proj = indexer.project_k(k_in_d)
            q_proj = indexer.project_q(q_in_d)
            p_idx = PagedLatentIndex(
                k_proj,
                n_sink=indexer.n_sink,
                n_pool=indexer.n_pool,
                n_cluster=indexer.n_cluster,
                avg_bucket_size=indexer.avg_bucket_size,
                max_bucket_size=indexer.max_bucket_size,
                window=indexer.window,
                rerank=indexer.rerank,
                search_mode=indexer.search_mode,
            )
            dist_a_tau = indexer.dist_alpha_over_tau()
            for qi in range(min(n_q, 16)):
                upto = int(q_pos_d[qi].item()) + 1
                sel = p_idx.select(q_proj[qi : qi + 1], budget=default_budget, upto=upto, dist_alpha_over_tau=dist_a_tau)
                sel_in_sup = sel[sup_d[qi].index_select(0, sel)]
                mass = float(t_probs_d[qi].index_select(0, sel_in_sup).sum().item()) if sel_in_sup.numel() > 0 else 0.0
                bucket_recall_sum += mass
                bucket_recall_q += 1

    denom = float(max(1, total_q))
    mass_by_k = {str(b_k): round(recall_sums[b_k] / denom, 6) for b_k in sorted(budgets)}
    primary_k = str(default_budget) if str(default_budget) in mass_by_k else next(iter(mass_by_k))
    return {
        "kl": round(total_kl / denom, 6),
        "primary_recall": float(mass_by_k[primary_k]),
        "mass_recall_by_budget": mass_by_k,
        "bucket_runtime_recall": round(bucket_recall_sum / float(max(1, bucket_recall_q)), 6) if bucket_recall_q > 0 else float(mass_by_k[primary_k]),
    }


def distill_single_visit_indexer(
    indexer: Any,
    qkv_samples: Sequence[SampleLike],
    cfg: Optional[VisitDistillConfig] = None,
    *,
    val_samples: Optional[Sequence[SampleLike]] = None,
    visit_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Train a single visit's `(W_k, W_q, log_tau, raw_dist_alpha)` with matched-support KL distillation,
    randomized query minibatches, scalar gradients (`F09`), and held-out early stopping (`§8.2, §8.5`)."""
    if not HAS_TORCH:
        raise RuntimeError("distill_single_visit_indexer requires PyTorch.")
    cfg = cfg or VisitDistillConfig(d_idx=int(indexer.cfg.d_idx))
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + int(visit_id or 0) * 31)

    train_batches, train_stats = _prepare_distill_batches(
        indexer, qkv_samples, cfg, seed_offset=int(visit_id or 0) * 101
    )
    if not train_batches:
        if cfg.fail_on_skipped_visit and cfg.require_long_range_support:
            raise RuntimeError(
                f"distill_single_visit_indexer(visit={visit_id}) failed closed: 0 valid selectable long-range query targets."
            )
        return {
            "status": "skipped_no_valid_queries",
            "steps": 0,
            "loss_first": 0.0,
            "loss_last": 0.0,
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "best_val_kl": 0.0,
            "initial_recall": 0.0,
            "final_recall": 0.0,
            "mass_recall_by_budget": {},
            "bucket_runtime_recall": 0.0,
            "scalar_grads_nonzero": False,
            "data_stats": train_stats,
        }

    if val_samples:
        val_batches, val_stats = _prepare_distill_batches(
            indexer, val_samples, cfg, seed_offset=int(visit_id or 0) * 101 + 7919
        )
    else:
        val_batches, val_stats = train_batches, train_stats
    eval_target_batches = val_batches if val_batches else train_batches

    init_eval = _evaluate_visit_batches(indexer, eval_target_batches, cfg, eval_bucket_runtime=False)
    best_val_kl = float(init_eval["kl"])
    best_state = {k: v.detach().clone() for k, v in indexer.state_dict().items()}

    proj_params = [indexer.W_q.weight, indexer.W_k.weight]
    scalar_params = [indexer.log_tau, indexer.raw_dist_alpha]
    for p in proj_params:
        p.requires_grad_(True)
    for p in scalar_params:
        p.requires_grad_(bool(cfg.train_scalars))

    param_groups = [{"params": proj_params, "lr": float(cfg.lr), "weight_decay": float(cfg.weight_decay)}]
    if cfg.train_scalars:
        param_groups.append({"params": scalar_params, "lr": float(cfg.scalar_lr), "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups)

    init_tau = float(indexer.tau().detach().item())
    init_alpha = float(indexer.dist_alpha().detach().item())
    loss_first: Optional[float] = None
    loss_last: float = 0.0
    scalar_grads_seen = False
    proj_grads_seen = False
    completed_steps = 0
    total_steps = max(1, int(cfg.steps))
    eval_interval = max(1, int(cfg.eval_every_steps))

    for step_i in range(total_steps):
        b_idx = int(torch.randint(0, len(train_batches), (1,), generator=gen).item())
        batch = train_batches[b_idx]
        n_avail = int(batch["q_in"].shape[0])
        n_mb = min(max(1, int(cfg.sampled_queries)), n_avail)
        if n_mb < n_avail:
            q_sel = torch.randperm(n_avail, generator=gen)[:n_mb]
        else:
            q_sel = None

        step_loss = _batch_kl_loss(indexer, batch, q_sel, cfg)
        if not torch.isfinite(step_loss):
            raise RuntimeError(f"Non-finite distillation KL loss on visit {visit_id} at step {step_i}: {step_loss.item()}")

        opt.zero_grad(set_to_none=True)
        step_loss.backward()
        if indexer.W_q.weight.grad is not None and indexer.W_k.weight.grad is not None:
            if float(indexer.W_q.weight.grad.abs().max().item()) > 0.0 and float(indexer.W_k.weight.grad.abs().max().item()) > 0.0:
                proj_grads_seen = True
        if cfg.train_scalars and indexer.log_tau.grad is not None and indexer.raw_dist_alpha.grad is not None:
            if float(indexer.log_tau.grad.abs().max().item()) > 0.0 and float(indexer.raw_dist_alpha.grad.abs().max().item()) > 0.0:
                scalar_grads_seen = True

        torch.nn.utils.clip_grad_norm_(proj_params + (scalar_params if cfg.train_scalars else []), max_norm=1.0)
        opt.step()
        completed_steps += 1

        val = float(step_loss.detach().item())
        if loss_first is None:
            loss_first = val
        loss_last = val

        if (step_i + 1) % eval_interval == 0 or (step_i + 1) == total_steps:
            cur_eval = _evaluate_visit_batches(indexer, eval_target_batches, cfg, eval_bucket_runtime=False)
            if float(cur_eval["kl"]) <= best_val_kl + 1e-8:
                best_val_kl = float(cur_eval["kl"])
                best_state = {k: v.detach().clone() for k, v in indexer.state_dict().items()}

    # Restore best held-out state (guarantees held-out KL never degrades vs PCA init!)
    indexer.load_state_dict(best_state, strict=True)
    for p in indexer.parameters():
        p.requires_grad_(False)
        if not bool(torch.isfinite(p).all().item()):
            raise RuntimeError(f"Non-finite selector parameter after distilling visit {visit_id}")

    final_eval = _evaluate_visit_batches(indexer, eval_target_batches, cfg, eval_bucket_runtime=True)
    return {
        "status": "ok",
        "steps": int(completed_steps),
        "loss_first": round(float(loss_first if loss_first is not None else init_eval["kl"]), 6),
        "loss_last": round(float(loss_last), 6),
        "initial_loss": round(float(init_eval["kl"]), 6),
        "best_val_kl": round(float(best_val_kl), 6),
        "final_loss": round(float(final_eval["kl"]), 6),
        "initial_recall": round(float(init_eval["primary_recall"]), 6),
        "final_recall": round(float(final_eval["primary_recall"]), 6),
        "mass_recall_by_budget": final_eval["mass_recall_by_budget"],
        "bucket_runtime_recall": float(final_eval["bucket_runtime_recall"]),
        "proj_grads_nonzero": bool(proj_grads_seen),
        "scalar_grads_nonzero": bool(scalar_grads_seen),
        "tau_init": round(init_tau, 6),
        "tau_final": round(float(indexer.tau().detach().item()), 6),
        "dist_alpha_init": round(init_alpha, 6),
        "dist_alpha_final": round(float(indexer.dist_alpha().detach().item()), 6),
        "visit_selector_sha256": canonical_state_dict_sha256(indexer.state_dict()),
        "train_stats": train_stats,
        "val_stats": val_stats,
    }


def stamp_bank_provenance(
    bank: "VisitAwareDLSAIndexBank",
    model: Optional[Any] = None,
    mhc_checkpoint_path: Optional[str] = None,
    mhc_checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Stamp canonical controller SHA-256, checkpoint file SHA-256, recurrence mode, schema version, and base revision
    into `bank.provenance` (`F04`, `§5.3`)."""
    file_sha = mhc_checkpoint_sha256
    if file_sha is None and mhc_checkpoint_path and os.path.isfile(mhc_checkpoint_path):
        h = hashlib.sha256()
        with open(mhc_checkpoint_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        file_sha = h.hexdigest()

    ctrl_sha: Optional[str] = None
    if model is not None and hasattr(model, "controllers"):
        ctrl_sha = canonical_state_dict_sha256(model.controllers.state_dict())
    if file_sha is None:
        file_sha = ctrl_sha

    rec_mode = None
    gain_mode = None
    schema_ver = None
    base_rev = None
    if model is not None:
        if hasattr(model, "effective_recurrence_mode"):
            rec_mode = str(model.effective_recurrence_mode(training=False))
        settings = getattr(model, "settings", None)
        if settings is not None:
            gain_mode = str(getattr(settings, "mhc_gain_mode", "legacy_tanh"))
            schema_ver = int(getattr(settings, "mhc_schema_version", 1))
        base_info = getattr(model, "base_runtime_info", {})
        if isinstance(base_info, dict):
            base_rev = base_info.get("resolved_revision") or base_info.get("requested_revision")

    if hasattr(bank, "set_provenance"):
        return bank.set_provenance(
            mhc_checkpoint_sha256=file_sha,
            controller_sha256=ctrl_sha or file_sha,
            mhc_checkpoint_path=os.path.basename(mhc_checkpoint_path) if mhc_checkpoint_path else None,
            recurrence_mode=rec_mode,
            mhc_gain_mode=gain_mode,
            mhc_schema_version=schema_ver,
            base_model_revision=base_rev,
        )
    return {}


def distill_loop72_visit_bank(
    bank: "VisitAwareDLSAIndexBank",
    visit_qkv_samples: Dict[int, Sequence[SampleLike]],
    cfg: Optional[VisitDistillConfig] = None,
    model: Optional[Any] = None,
    mhc_checkpoint_path: Optional[str] = None,
    mhc_checkpoint_sha256: Optional[str] = None,
    *,
    val_visit_samples: Optional[Dict[int, Sequence[SampleLike]]] = None,
) -> Dict[str, Any]:
    """Run streamed-Gram PCA initialization + matched-support sampled-query KL distillation across all 18 global visits
    of `VisitAwareDLSAIndexBank`, failing closed if any required visit is missing or skipped (`F01`, `§8.6`)."""
    if not HAS_TORCH:
        raise RuntimeError("distill_loop72_visit_bank requires PyTorch.")
    cfg = cfg or VisitDistillConfig()
    t0 = time.perf_counter()
    n_expected = len(LOOP72_GLOBAL_VISIT_SCHEDULE)
    req_visits = int(cfg.require_global_visits) if cfg.fail_on_skipped_visit else 0

    if req_visits > 0:
        missing_v = [gv for gv in range(min(req_visits, n_expected)) if not visit_qkv_samples.get(gv)]
        if missing_v:
            raise RuntimeError(
                f"distill_loop72_visit_bank failed closed before training: missing samples for {len(missing_v)} "
                f"required global visits {missing_v} (expected {req_visits})."
            )

    prov = stamp_bank_provenance(
        bank,
        model=model,
        mhc_checkpoint_path=mhc_checkpoint_path,
        mhc_checkpoint_sha256=mhc_checkpoint_sha256,
    )
    pca_rep = (
        fit_loop72_visit_bank_pca(
            bank,
            visit_qkv_samples,
            require_global_visits=req_visits,
            seed=cfg.seed,
        )
        if cfg.init_with_pca
        else {}
    )

    distill_rep: Dict[int, Dict[str, Any]] = {}
    trained_count = 0
    for gv_idx in range(n_expected):
        samples = visit_qkv_samples.get(gv_idx, [])
        if not samples:
            if cfg.fail_on_skipped_visit and gv_idx < req_visits:
                raise RuntimeError(f"distill_loop72_visit_bank failed closed: visit {gv_idx} has no samples.")
            distill_rep[gv_idx] = {"status": "skipped_no_samples", "steps": 0}
            continue
        v_samps = val_visit_samples.get(gv_idx) if isinstance(val_visit_samples, dict) else None
        indexer = bank.get_visit_indexer(gv_idx)
        rep = distill_single_visit_indexer(indexer, samples, cfg=cfg, val_samples=v_samps, visit_id=gv_idx)
        distill_rep[gv_idx] = rep
        if rep.get("status") == "ok" and int(rep.get("steps", 0)) > 0:
            trained_count += 1
            if hasattr(bank, "training_records") and isinstance(bank.training_records, dict):
                bank.training_records[str(gv_idx)] = {
                    "steps": int(rep["steps"]),
                    "initial_loss": float(rep["initial_loss"]),
                    "best_val_kl": float(rep["best_val_kl"]),
                    "final_loss": float(rep["final_loss"]),
                    "final_recall": float(rep["final_recall"]),
                    "bucket_runtime_recall": float(rep["bucket_runtime_recall"]),
                    "mass_recall_by_budget": dict(rep.get("mass_recall_by_budget", {})),
                    "visit_selector_sha256": str(rep.get("visit_selector_sha256", "")),
                    "long_range_queries": int(rep.get("train_stats", {}).get("long_range_queries", 0)),
                }
        elif cfg.fail_on_skipped_visit and gv_idx < req_visits:
            raise RuntimeError(
                f"distill_loop72_visit_bank failed closed: visit {gv_idx} did not complete training ({rep.get('status')})."
            )

    if cfg.fail_on_skipped_visit and trained_count < req_visits:
        raise RuntimeError(
            f"distill_loop72_visit_bank failed closed: only {trained_count}/{req_visits} visits trained."
        )

    sel_sha = bank.canonical_selector_sha256() if hasattr(bank, "canonical_selector_sha256") else ""
    if hasattr(bank, "provenance") and isinstance(bank.provenance, dict):
        bank.provenance["selector_sha256"] = sel_sha
        bank.provenance["trained_global_visits"] = int(trained_count)

    ok_reps = [r for r in distill_rep.values() if r.get("status") == "ok"]
    n_ok = max(1, len(ok_reps))
    mean_init_loss = sum(float(r.get("initial_loss", 0.0)) for r in ok_reps) / float(n_ok) if ok_reps else 0.0
    mean_final_loss = sum(float(r.get("final_loss", 0.0)) for r in ok_reps) / float(n_ok) if ok_reps else 0.0
    mean_bucket_recall = sum(float(r.get("bucket_runtime_recall", 0.0)) for r in ok_reps) / float(n_ok) if ok_reps else 0.0
    mean_recall_2048 = (
        sum(float(r.get("mass_recall_by_budget", {}).get("2048", r.get("final_recall", 0.0))) for r in ok_reps) / float(n_ok)
        if ok_reps
        else 0.0
    )
    mean_recall_4096 = (
        sum(float(r.get("mass_recall_by_budget", {}).get("4096", r.get("final_recall", 0.0))) for r in ok_reps) / float(n_ok)
        if ok_reps
        else 0.0
    )
    skipped_count = max(0, int(n_expected) - int(trained_count))

    return {
        "status": "ok" if trained_count >= max(1, req_visits) else "incomplete",
        "num_global_visits": n_expected,
        "trained_global_visits": int(trained_count),
        "skipped_global_visits": int(skipped_count),
        "total_map_parameters": bank.total_map_parameters(),
        "selector_sha256": sel_sha,
        "mean_initial_loss": round(mean_init_loss, 6),
        "mean_final_loss": round(mean_final_loss, 6),
        "mean_bucket_runtime_recall": round(mean_bucket_recall, 6),
        "mean_mass_recall_at_2048": round(mean_recall_2048, 6),
        "mean_mass_recall_at_4096": round(mean_recall_4096, 6),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "provenance": prov,
        "pca_init": pca_rep,
        "distill": distill_rep,
    }


@torch.no_grad()
def run_loop72_four_way_eval(
    model: Any,
    input_ids: "torch.LongTensor",
    yarn_scale_factor: float = 4.0,
) -> Dict[str, Any]:
    """Run the 4-way positional/sparse ablation matrix from §8.4 of the Loop72 LSA Plan:
      1. `Dense_NativeRoPE`: Exact global attention + native partial RoPE (`yarn_scale_factor = 1.0`).
      2. `Sparse_NativeRoPE`: Paged D-LSA sparse global attention + native partial RoPE (`yarn_scale_factor = 1.0`).
      3. `Dense_YaRNRoPE`: Exact global attention + YaRN-scaled partial RoPE (`yarn_scale_factor > 1.0`).
      4. `Sparse_YaRNRoPE`: Paged D-LSA sparse global attention + YaRN-scaled partial RoPE (`yarn_scale_factor > 1.0`).
    """
    if not HAS_TORCH:
        raise RuntimeError("run_loop72_four_way_eval requires PyTorch.")
    sparse_cfg = getattr(model, "sparse_attention_cfg", None)
    if sparse_cfg is None:
        raise RuntimeError("Model must be configured with attach_loop72_dlsa_attention before running 4-way eval.")

    saved_thresh = int(sparse_cfg.exact_threshold)
    saved_yarn = float(sparse_cfg.yarn_scale_factor)
    seq_len = int(input_ids.shape[1])
    sparse_thresh = min(saved_thresh, max(64, seq_len // 2))
    results: Dict[str, Dict[str, float]] = {}

    modes = [
        ("Dense_NativeRoPE", seq_len + 1024, 1.0),
        ("Sparse_NativeRoPE", sparse_thresh, 1.0),
        ("Dense_YaRNRoPE", seq_len + 1024, float(yarn_scale_factor)),
        ("Sparse_YaRNRoPE", sparse_thresh, float(yarn_scale_factor)),
    ]
    ref_logits: Optional[torch.Tensor] = None
    try:
        for name, thresh, y_scale in modes:
            sparse_cfg.exact_threshold = int(thresh)
            sparse_cfg.yarn_scale_factor = float(y_scale)
            cache = model.make_inference_cache(session_id=f"eval_{name}")
            t0 = time.perf_counter()
            logits = model.forward_logits(
                input_ids,
                past_key_values=cache,
                chunk_size=getattr(model.settings, "prefill_chunk_tokens", 2048),
            ).float()
            dt = time.perf_counter() - t0
            if input_ids.shape[1] >= 2:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                nll = float(
                    F.cross_entropy(
                        shift_logits.view(-1, shift_logits.shape[-1]),
                        shift_labels.view(-1),
                    ).item()
                )
            else:
                nll = 0.0
            if ref_logits is None:
                ref_logits = logits.detach().clone()
                logit_diff = 0.0
            else:
                logit_diff = float((logits - ref_logits).abs().mean().item())
            results[name] = {
                "nll": round(nll, 6),
                "mean_abs_logit_diff_vs_dense_native": round(logit_diff, 6),
                "latency_sec": round(dt, 4),
            }
    finally:
        sparse_cfg.exact_threshold = saved_thresh
        sparse_cfg.yarn_scale_factor = saved_yarn

    return {
        "seq_len": seq_len,
        "yarn_scale_factor": float(yarn_scale_factor),
        "modes": results,
    }


def run_standalone_stage_e_cli(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Standalone Stage E entry point (`§8.6`): load an accepted dense mHC checkpoint, capture long-context Q/K
    targets across all 18 global visits, run streamed-Gram PCA + matched-support KL distillation, and export
    `visit_selectors_stage_e.pt` + `stage_e_report.json`."""
    parser = argparse.ArgumentParser(description="Standalone Stage E 18-Visit D-LSA Selector Distillation")
    parser.add_argument("--base-model", default="XHToken/Spark-X2.6-4B", help="Base model HF ID or path")
    parser.add_argument("--base-revision", default="64ca8f2a295b8ceb2757b9d4749a6558c4759e5b", help="Pinned base revision")
    parser.add_argument("--dense-checkpoint", required=True, help="Path to accepted dense loop_v03_state.pt")
    parser.add_argument("--data-manifest", default=None, help="JSON/JSONL calibration manifest path (train split)")
    parser.add_argument("--eval-manifest", default=None, help="JSON/JSONL calibration manifest path (dev/val split)")
    parser.add_argument("--output-dir", required=True, help="Output directory for visit_selectors_stage_e.pt")
    parser.add_argument("--doc-tokens", type=int, default=8192, help="Target token length per calibration document (>=8192)")
    parser.add_argument("--num-train-docs", type=int, default=64, help="Number of training documents to capture")
    parser.add_argument("--num-val-docs", type=int, default=16, help="Number of held-out validation documents")
    parser.add_argument("--steps", type=int, default=200, help="Optimizer steps per global visit")
    parser.add_argument("--sampled-queries", type=int, default=48, help="Sampled long-range queries per step")
    parser.add_argument("--lr", type=float, default=1e-3, help="Projection learning rate")
    parser.add_argument("--require-global-visits", type=int, default=18, help="Required trained global visits (default 18)")
    parser.add_argument("--device", default="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from load_inference_v03 import load_sparkx_loop72_mhc_for_training
    from sparkx_training_data import build_long_context_calibration_token_packs

    os.makedirs(args.output_dir, exist_ok=True)
    loaded = load_sparkx_loop72_mhc_for_training(
        model_id_or_path=args.base_model,
        base_revision=args.base_revision,
        checkpoint_path=args.dense_checkpoint,
        device=args.device,
    )
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    model.eval()

    train_packs, val_packs, pack_meta = build_long_context_calibration_token_packs(
        tokenizer=tokenizer,
        seq_len=int(args.doc_tokens),
        num_train_packs=int(args.num_train_docs),
        num_val_packs=int(args.num_val_docs),
        data_manifest=args.data_manifest,
        eval_manifest=args.eval_manifest,
        seed=int(args.seed),
    )

    train_samples = capture_loop72_visit_distill_dataset(
        model,
        train_packs,
        max_sampled_queries=max(int(args.sampled_queries), 64),
        store_device="cpu",
        store_dtype=torch.bfloat16,
        split="train",
        seed=int(args.seed),
    )
    val_samples = capture_loop72_visit_distill_dataset(
        model,
        val_packs,
        max_sampled_queries=max(int(args.sampled_queries), 32),
        store_device="cpu",
        store_dtype=torch.bfloat16,
        split="val",
        seed=int(args.seed) + 10007,
    ) if val_packs else None

    bank = VisitAwareDLSAIndexBank(maps_per_global_visit=True).to(device=args.device)
    cfg = VisitDistillConfig(
        steps=int(args.steps),
        sampled_queries=int(args.sampled_queries),
        lr=float(args.lr),
        require_global_visits=int(args.require_global_visits),
        require_long_range_support=int(args.doc_tokens) >= 4616,
        fail_on_skipped_visit=True,
        seed=int(args.seed),
    )
    report = distill_loop72_visit_bank(
        bank,
        train_samples,
        cfg=cfg,
        model=model,
        mhc_checkpoint_path=args.dense_checkpoint,
        val_visit_samples=val_samples,
    )
    report["calibration_data"] = pack_meta

    sel_path = os.path.join(args.output_dir, "visit_selectors_stage_e.pt")
    bank.save_pretrained(sel_path)
    rep_path = os.path.join(args.output_dir, "stage_e_report.json")
    with open(rep_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


if __name__ == "__main__":  # pragma: no cover
    out_rep = run_standalone_stage_e_cli()
    print(json.dumps({"status": out_rep["status"], "trained_global_visits": out_rep["trained_global_visits"], "selector_sha256": out_rep["selector_sha256"]}, indent=2))
