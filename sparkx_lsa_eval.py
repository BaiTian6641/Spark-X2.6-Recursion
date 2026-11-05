"""sparkx_lsa_eval.py — LSA-F2: real-attention recall gate for faithful D-LSA (see docs/roadmap/07, §3).

GPU path is **untested** (written crash-proof: every stage is wrapped, OOM is reported instead of raised). The pure helpers
(`pack_to_length`, `gate_f2`, `slice_fidelity`, `resolve_rope_first`, `oracle_diagnostics`, `format_table`) are unit-tested on CPU with synthetic tensors.

Typical Colab use (model/tokenizer from `load_sparkx_loop_v03`):

    import sparkx_lsa_eval as le
    REPORT = le.run_lsa_f2(model, tokenizer, train_texts, val_texts)        # distil on train docs, evaluate on held-out docs
    print(le.format_table(REPORT)); print(json.dumps(le.gate_f2(REPORT), indent=2))
    le.upgrade_lsa_indexers(model, le.load_indexers('/content/lsa_f2/dlsa_indexers.pt'))   # LSA-F3 decode path
"""
from __future__ import annotations

import gc
import importlib
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sparkx_dlsa import (FaithfulDLSAIndexer, distill_indexer, eval_kl, evaluate_selectors, free_logit_fidelity,
                         teacher_attention_heads)

GLOBAL_LAYERS: Tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31, 35)


# ======================================================================================
# Data
# ======================================================================================
def pack_to_length(tokenizer: Any, texts: Sequence[str], length: int, seed: int = 0, sep_id: Optional[int] = None,
                   max_tries: int = 100000) -> Optional[torch.Tensor]:
    """Concatenate *distinct* documents (shuffled, never repeated) into one `[1, length]` token sequence, separated by EOS.
    Returns None if the corpus is too small. NOTE: packed unrelated documents test topic retrieval across documents; they do
    not replace genuinely long documents (books/repos), which should be passed in as single texts when available."""
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    if sep_id is None:
        sep_id = getattr(tokenizer, "eos_token_id", None)
    ids: List[int] = []
    for i in order[:max_tries]:
        t = tokenizer(texts[i], add_special_tokens=False)["input_ids"]
        ids.extend(t)
        if sep_id is not None:
            ids.append(int(sep_id))
        if len(ids) >= length:
            break
    if len(ids) < length:
        return None
    return torch.tensor(ids[:length], dtype=torch.long).unsqueeze(0)


# ======================================================================================
# Model plumbing
# ======================================================================================
def _layer_attn(model: Any, layer: int) -> Any:
    layers = model._layers() if hasattr(model, "_layers") else model.base.model.layers
    return layers[layer].self_attn


def layer_geometry(model: Any, layer: int) -> Dict[str, Any]:
    a = _layer_attn(model, layer)
    n_kv = int(getattr(a, "num_key_value_heads", 4) or 4)
    hd = int(getattr(a, "head_dim", 256) or 256)
    rope_dim = 64
    cfg = getattr(getattr(model, "base", model), "config", None)
    try:
        if cfg is not None and hasattr(cfg, "get_partial_rotary_factor"):
            rope_dim = int(round(hd * float(cfg.get_partial_rotary_factor("full_attention"))))
        elif cfg is not None and getattr(cfg, "partial_rotary_factor", None) is not None:
            rope_dim = int(round(hd * float(cfg.partial_rotary_factor)))
    except Exception:
        rope_dim = 64
    scale = getattr(a, "scaling", None)
    return {"n_kv": n_kv, "head_dim": hd, "rope_dim": rope_dim, "scale": float(scale) if scale else None}


@torch.no_grad()
def capture_qk(model: Any, input_ids: torch.Tensor, layers: Sequence[int], device: str = "cpu",
               with_v: bool = False) -> Dict[int, Dict[str, torch.Tensor]]:
    """Post-RoPE q `[H_q, T, D]` and k `[H_kv, T, D]` (bf16, on `device`) from the first visit of each requested layer.
    `with_v=True` also returns `v` `[H_kv, T, D]` (needed for attention-output error, not just attention-mass recall)."""
    ids = input_ids.to(next(model.parameters()).device)
    recs = model.capture_attention_io(ids, list(layers), visit=0)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for l in list(recs.keys()):
        r = recs.pop(l)
        if r.get("query") is None or r.get("key") is None:
            continue
        out[l] = {"q": r["query"][0].to(torch.bfloat16).to(device), "k": r["key"][0].to(torch.bfloat16).to(device)}
        if with_v and r.get("value") is not None:
            out[l]["v"] = r["value"][0].to(torch.bfloat16).to(device)
        del r
    del recs
    return out


def probe_rotated_dims(model: Any, layer: int, head_dim: int = 256, rope_dim: int = 64, theta: float = 5e6) -> Optional[List[int]]:
    """Ground truth: which head dims does the model's OWN `apply_rotary_pos_emb` change? Feeds a constant probe through the module's
    function (positions 1..4) and returns the indices that moved. None if the signature/shapes could not be probed.

    The real signature (modeling_spark.py, verified 2026-11-04) is `apply_rotary_pos_emb(x, cos, sin)` with 2-D cos/sin `[T, rd]` and **no**
    `unsqueeze_dim` argument; the first probe version passed `unsqueeze_dim=1`, raised TypeError, and silently returned None. All plausible
    call conventions are now tried and the failures are recorded by `describe_rotary`."""
    return _probe_rotary(model, layer, head_dim, rope_dim, theta)[0]


def _probe_rotary(model: Any, layer: int, head_dim: int, rope_dim: int, theta: float) -> Tuple[Optional[List[int]], List[str]]:
    import importlib
    errs: List[str] = []
    try:
        mod = importlib.import_module(type(_layer_attn(model, layer)).__module__)
    except Exception as e:  # noqa: BLE001
        return None, [f"import_module: {type(e).__name__}: {e}"]
    try:
        pos = torch.arange(1, 5, dtype=torch.float32)
        cos, sin = mod.compute_rope_cos_sin(pos, head_dim, theta, rope_dim / head_dim, device="cpu")
    except Exception as e:  # noqa: BLE001
        return None, [f"compute_rope_cos_sin: {type(e).__name__}: {e}"]
    x = torch.ones(1, 1, 4, head_dim)
    calls = (("x,cos,sin", lambda: mod.apply_rotary_pos_emb(x, cos, sin)),
             ("x,cos,sin,unsqueeze_dim=1", lambda: mod.apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1)),
             ("x,cos[None],sin[None],unsqueeze_dim=1", lambda: mod.apply_rotary_pos_emb(x, cos[None], sin[None], unsqueeze_dim=1)))
    for name, fn in calls:
        try:
            y = fn()
            if isinstance(y, tuple):
                y = y[0]
            moved = (y.float() - x).abs().amax(dim=(0, 1, 2)) > 1e-6
            idx = [int(i) for i in torch.nonzero(moved).flatten()]
            if idx:
                return idx, errs
            errs.append(f"{name}: ran but no dim moved")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}: {e}")
    return None, errs


def describe_rotary(model: Any, layer: int = 3) -> Dict[str, Any]:
    """Paste-able diagnostic of the rotary layout: probe result, per-convention errors, and the source of the (possibly patched) functions."""
    import importlib
    import inspect
    geo = layer_geometry(model, layer)
    moved, errs = _probe_rotary(model, layer, geo["head_dim"], geo["rope_dim"], 5e6)
    out: Dict[str, Any] = {"layer": layer, "geometry": geo, "n_moved": None if moved is None else len(moved),
                           "moved_head": None if moved is None else moved[:4], "moved_tail": None if moved is None else moved[-4:],
                           "errors": errs}
    try:
        mod = importlib.import_module(type(_layer_attn(model, layer)).__module__)
        for nm in ("compute_rope_cos_sin", "apply_rotary_pos_emb"):
            try:
                out["src_" + nm] = inspect.getsource(getattr(mod, nm))
            except Exception as e:  # noqa: BLE001
                out["src_" + nm] = f"unavailable: {type(e).__name__}: {e}"
        out["orig_apply_rope"] = getattr(getattr(mod, "_sparkx_orig_apply_rope", None), "__name__", None)
    except Exception as e:  # noqa: BLE001
        out["module_error"] = f"{type(e).__name__}: {e}"
    out["resolved"] = resolve_rope_first(model, layer, geo)
    return out


def slice_fidelity(q: torch.Tensor, k: torch.Tensor, rope_dim: int, positions: Sequence[int]) -> Dict[str, Any]:
    """DIAGNOSTIC ONLY (not a layout detector): Pearson r between the full logit and the logit restricted to dims [rope_dim:] or
    [:head_dim-rope_dim]. The second slice overlaps the first on all but `rope_dim` dims and can win simply because it contains
    high-magnitude rotated dims — the F2 run of 2026-11-04 mis-selected it for 7/9 layers by treating this as a layout detector."""
    r_first = free_logit_fidelity(q, k, positions, rope_dim, rope_first=True)["mean_pearson_r"]
    r_last = free_logit_fidelity(q, k, positions, rope_dim, rope_first=False)["mean_pearson_r"]
    return {"r_free_is_tail": r_first, "r_free_is_head": r_last}


def resolve_rope_first(model: Any, layer: int, geo: Dict[str, Any]) -> Dict[str, Any]:
    """Layout from the model's own rotary function; fallback = config convention (rotated dims first) with a flag."""
    moved = probe_rotated_dims(model, layer, geo["head_dim"], geo["rope_dim"])
    if moved:
        first = moved == list(range(len(moved))) and len(moved) == geo["rope_dim"]
        last = moved == list(range(geo["head_dim"] - len(moved), geo["head_dim"]))
        if first or last:
            return {"rope_first": bool(first), "source": "probe", "n_rotated": len(moved)}
        return {"rope_first": True, "source": "probe_unrecognised_layout", "rotated_dims_sample": moved[:8] + moved[-8:]}
    return {"rope_first": True, "source": "config_default_unverified"}


def sample_positions(t: int, n: int, lo: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    lo = max(lo, t // 2)
    hi = t - 1
    if hi <= lo:
        return []
    return sorted(rng.randint(lo, hi) for _ in range(n))


# ======================================================================================
# LSA-F2 driver
# ======================================================================================
def _free(*_ignored: Any) -> None:
    """Best-effort GC + CUDA cache release (callers drop their own references)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_lsa_f2(
    model: Any,
    tokenizer: Any,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    layers: Sequence[int] = GLOBAL_LAYERS,
    train_lengths: Sequence[int] = (16384, 32768),
    val_lengths: Sequence[int] = (32768, 65536),
    train_docs_per_length: int = 2,
    val_docs_per_length: int = 1,
    layers_per_pass: int = 3,
    k_budget: int = 2048,
    distill_steps: int = 300,
    distill_lr: float = 3e-3,
    queries_per_sample: int = 64,
    n_eval_queries: int = 24,
    out_dir: str = "/content/lsa_f2",
    seed: int = 0,
    log=print,
    indexer_kwargs: Optional[Dict[str, Any]] = None,
    early_stop: bool = True,
) -> Dict[str, Any]:
    """Per layer: PCA-init recall (training-free D-LSA), per-key-KL distillation on train docs, recall of every arm on held-out
    docs at each val length. Returns a JSON-able report; saves distilled indexers to `out_dir/dlsa_indexers.pt`.

    Revision 2026-11-04 (after the first F2/F2b runs): all arms are compared at **equal total key budget** and the attention-**output**
    relative error (`err_*`, needs `v`) is reported next to attention-mass recall; `indexer_kwargs` forwards `n_pool/avg_bucket_size/
    max_bucket_size/rerank` (F2b: token-level buckets or coarse-bucket→token rerank beat the coarse default); with >= 2 train samples
    the last one is held out for distillation **early stopping** (`early_stop`), never the evaluation docs."""
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    layers = [int(l) for l in layers]
    report: Dict[str, Any] = {"config": dict(layers=layers, train_lengths=list(train_lengths), val_lengths=list(val_lengths),
                                             k_budget=k_budget, distill_steps=distill_steps, seed=seed),
                              "layers": {}, "errors": []}

    def build_ids(texts: Sequence[str], lengths: Sequence[int], n_docs: int, split: str) -> List[Tuple[int, torch.Tensor]]:
        res = []
        for li, L in enumerate(lengths):
            for d in range(n_docs):
                ids = pack_to_length(tokenizer, texts, int(L), seed=seed * 1000 + li * 17 + d + (0 if split == "train" else 500))
                if ids is None:
                    report["errors"].append(f"{split}: corpus too small for length {L} (doc {d})")
                    continue
                res.append((int(L), ids))
        return res

    train_ids = build_ids(train_texts, train_lengths, train_docs_per_length, "train")
    val_ids = build_ids(val_texts, val_lengths, val_docs_per_length, "val")
    if not train_ids or not val_ids:
        report["errors"].append("no train or val sequences could be built")
        return report

    saved: Dict[int, Dict[str, Any]] = {}
    for g0 in range(0, len(layers), max(1, layers_per_pass)):
        group = layers[g0: g0 + max(1, layers_per_pass)]
        train_s: Dict[int, List[Dict[str, torch.Tensor]]] = {l: [] for l in group}
        val_s: Dict[int, List[Tuple[int, Dict[str, torch.Tensor]]]] = {l: [] for l in group}
        for (L, ids) in train_ids:
            try:
                cap = capture_qk(model, ids, group, device="cpu")
                for l, d in cap.items():
                    train_s[l].append(d)
            except Exception as e:  # incl. OOM
                report["errors"].append(f"capture train L={L} layers={group}: {type(e).__name__}: {str(e)[:200]}")
                _free()
        for (L, ids) in val_ids:
            try:
                cap = capture_qk(model, ids, group, device="cpu", with_v=True)
                for l, d in cap.items():
                    val_s[l].append((L, d))
            except Exception as e:
                report["errors"].append(f"capture val L={L} layers={group}: {type(e).__name__}: {str(e)[:200]}")
                _free()
        for l in group:
            try:
                geo = layer_geometry(model, l)
                if not train_s[l] or not val_s[l]:
                    report["errors"].append(f"layer {l}: no samples captured")
                    continue
                q0, k0 = train_s[l][0]["q"].to(dev), train_s[l][0]["k"].to(dev)
                lay = resolve_rope_first(model, l, geo)
                lay["slice_fidelity"] = slice_fidelity(q0, k0, geo["rope_dim"], sample_positions(q0.shape[1], 8, 4096, seed))
                ix = FaithfulDLSAIndexer(n_kv=geo["n_kv"], head_dim=geo["head_dim"], rope_dim=geo["rope_dim"],
                                         rope_first=lay["rope_first"], k_long_range=k_budget, **(indexer_kwargs or {})).to(dev)
                ix.fit_pca(torch.cat([ix.inputs(s["q"].to(dev), s["k"].to(dev))[1][:: max(1, s["k"].shape[1] // 16384)]
                                      for s in train_s[l]], dim=0))
                lrep: Dict[str, Any] = {"geometry": geo, "rope_layout": lay, "val": {}}

                def eval_arms(tag: str) -> None:
                    for (L, s) in val_s[l]:
                        q, k = s["q"].to(dev), s["k"].to(dev)
                        pos = sample_positions(q.shape[1], n_eval_queries, ix.n_sink + ix.window + k_budget + 1, seed + L)
                        fid = free_logit_fidelity(q, k, pos[:8], geo["rope_dim"], lay["rope_first"]) if pos else {}
                        vv = s["v"].to(dev) if s.get("v") is not None else None
                        r = evaluate_selectors(ix, q, k, pos, budget=k_budget, seed=seed, scale=geo["scale"], v=vv) if pos else {}
                        lrep["val"].setdefault(str(L), {})[tag] = {"recall": r, "fidelity_r": fid.get("mean_pearson_r")}
                        del q, k, vv

                eval_arms("pca_init")
                tr_all = [{"q": s["q"].to(dev), "k": s["k"].to(dev)} for s in train_s[l]]
                use_val = early_stop and len(tr_all) >= 2
                dist = distill_indexer(ix, tr_all[:-1] if use_val else tr_all, steps=distill_steps,
                                       lr=distill_lr, queries_per_sample=queries_per_sample, seed=seed, log_every=0,
                                       scale=geo["scale"], val_samples=tr_all[-1:] if use_val else None)
                lrep["distill"] = dist
                eval_arms("distilled")
                report["layers"][str(l)] = lrep
                saved[l] = {"state": {k_: v.detach().cpu() for k_, v in ix.state_dict().items()},
                            "geometry": geo, "rope_first": lay["rope_first"]}
                log(f"[LSA-F2] layer {l}: " + json.dumps({L: v.get("distilled", {}).get("recall", {}).get("faithful")
                                                          for L, v in lrep["val"].items()}))
                _free(ix, q0, k0)
            except Exception as e:
                report["errors"].append(f"layer {l}: {type(e).__name__}: {str(e)[:300]}")
                _free()
        _free(train_s, val_s)
    try:
        torch.save({"indexers": saved, "k_budget": k_budget}, os.path.join(out_dir, "dlsa_indexers.pt"))
    except Exception as e:
        report["errors"].append(f"save: {e}")
    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "lsa_f2_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


# ======================================================================================
# LSA-F2b: why did F2 fail?  (diagnostics requested by the 2026-11-04 result; see docs/roadmap/07 §3.2)
# ======================================================================================
@torch.no_grad()
def oracle_diagnostics(q: torch.Tensor, k: torch.Tensor, positions: Sequence[int], budgets: Sequence[int] = (2048, 4096, 8192, 16384),
                       scale: Optional[float] = None, n_sink: int = 4, window: int = 512) -> Dict[str, Any]:
    """Upper bounds, no index involved. For each budget b (long-range keys; sink + window always read) report the mean per-head
    attention-mass recall of: `shared` (one oracle selection from the head-mean attention, what the decode path does today),
    `per_group` (one per KV group) and `per_head`. If per_head >> shared the head-shared selection, not the index, is the limit."""
    hq = q.shape[0]
    hkv = k.shape[0]
    g = hq // hkv
    out: Dict[str, Any] = {b: {"shared": [], "per_group": [], "per_head": []} for b in budgets}
    for p in positions:
        probs = teacher_attention_heads(q, k, [p], scale)[:, 0]  # [Hq, T]
        n = p + 1
        always = torch.zeros(probs.shape[-1], dtype=torch.bool, device=probs.device)
        always[:min(n_sink, n)] = True
        always[max(0, n - window):n] = True
        lo, hi = n_sink, max(n_sink, n - window)
        if hi - lo <= max(budgets):
            continue
        mean_p = probs.mean(0)
        for b in budgets:
            def recall(sel_mask: torch.Tensor, rows: torch.Tensor) -> float:
                return float((rows * (sel_mask | always)).sum(-1).mean())
            sh = torch.zeros_like(always)
            sh[lo + mean_p[lo:hi].topk(b).indices] = True
            out[b]["shared"].append(recall(sh, probs))
            gm = []
            for gi in range(hkv):
                rows = probs[gi * g:(gi + 1) * g]
                sel = torch.zeros_like(always)
                sel[lo + rows.mean(0)[lo:hi].topk(b).indices] = True
                gm.append(float((rows * (sel | always)).sum(-1).mean()))
            out[b]["per_group"].append(sum(gm) / len(gm))
            ph = []
            for h in range(hq):
                sel = torch.zeros_like(always)
                sel[lo + probs[h, lo:hi].topk(b).indices] = True
                ph.append(float((probs[h] * (sel | always)).sum()))
            out[b]["per_head"].append(sum(ph) / len(ph))
    return {int(b): {k_: (sum(v) / len(v) if v else None) for k_, v in d.items()} | {"n_queries": len(d["shared"])} for b, d in out.items()}


F2B_VARIANTS: Dict[str, Dict[str, Any]] = {
    "base": {},                                                    # free slice, pool 4, avg bucket 8 (the F2 setting, correct layout)
    "fine_buckets": {"n_pool": 2, "avg_bucket_size": 4, "max_bucket_size": 8},
    "token_buckets": {"n_pool": 1, "avg_bucket_size": 2, "max_bucket_size": 4},   # best in the first F2b run (index ~4x centroids)
    "coarse_rerank4": {"rerank": 4},                               # coarse buckets to 4x budget, then token-level top-budget
    "fine_rerank4": {"n_pool": 2, "avg_bucket_size": 4, "max_bucket_size": 8, "rerank": 4},
}
# Ablations that showed no benefit / layer-dependent benefit in the first F2b run; opt in via `variants=`.
F2B_VARIANTS_EXTRA: Dict[str, Dict[str, Any]] = {
    "full_head_input": {"rope_dim": 0},                            # helped L3 pre-distill only; overfit on distillation
    "train_incl_64k": {"train_lengths_extra": (65536,)},           # no benefit
}


def _f3(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def run_lsa_f2b(
    model: Any,
    tokenizer: Any,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    layers: Sequence[int] = (3, 15, 31),
    variants: Optional[Dict[str, Dict[str, Any]]] = None,
    train_lengths: Sequence[int] = (16384, 32768),
    val_length: int = 65536,
    train_docs_per_length: int = 3,
    val_docs: int = 2,
    k_budget: int = 2048,
    distill_steps: int = 300,
    n_eval_queries: int = 16,
    oracle_budgets: Sequence[int] = (2048, 4096, 8192, 16384),
    out_dir: str = "/content/lsa_f2b",
    seed: int = 0,
    log=print,
) -> Dict[str, Any]:
    """Ablations on a few representative layers, correct rotary layout (probed). Per layer and variant: held-out recall of every arm
    at `val_length`, held-out per-key KL before/after distillation, plus `oracle_diagnostics` (budget curve and shared/group/head)."""
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    variants = variants or F2B_VARIANTS
    rep: Dict[str, Any] = {"config": dict(layers=list(layers), variants=variants, val_length=val_length, k_budget=k_budget), "layers": {}, "errors": []}
    all_train_lengths = sorted({int(L) for L in train_lengths} | {int(L) for v in variants.values() for L in v.get("train_lengths_extra", ())})
    def ids_for(texts, L, n, off):
        out = []
        for d in range(n):
            ids = pack_to_length(tokenizer, texts, int(L), seed=seed * 1000 + int(L) % 97 + d + off)
            if ids is not None:
                out.append((int(L), ids))
            else:
                rep["errors"].append(f"corpus too small for length {L}")
        return out
    train_ids = [x for L in all_train_lengths for x in ids_for(train_texts, L, train_docs_per_length, 0)]
    val_ids = ids_for(val_texts, val_length, val_docs, 500)
    for l in [int(x) for x in layers]:
        try:
            geo = layer_geometry(model, l)
            lay = resolve_rope_first(model, l, geo)
            tr, va = [], []
            for (L, ids) in train_ids:
                c = capture_qk(model, ids, [l], device="cpu")
                if l in c:
                    tr.append((L, c[l]))
            for (L, ids) in val_ids:
                c = capture_qk(model, ids, [l], device="cpu", with_v=True)
                if l in c:
                    va.append(c[l])
            _free()
            lrep: Dict[str, Any] = {"geometry": geo, "layout": lay, "variants": {}}
            pos0 = sample_positions(va[0]["q"].shape[1], n_eval_queries, 4 + 512 + k_budget + 1, seed)
            lrep["oracle"] = oracle_diagnostics(va[0]["q"].to(dev), va[0]["k"].to(dev), pos0[: max(4, n_eval_queries // 2)], budgets=oracle_budgets,
                                                 scale=geo["scale"])
            for name, v in variants.items():
                cfg = dict(n_kv=geo["n_kv"], head_dim=geo["head_dim"], rope_dim=v.get("rope_dim", geo["rope_dim"]),
                           rope_first=lay["rope_first"], k_long_range=k_budget)
                for key in ("n_pool", "avg_bucket_size", "max_bucket_size", "rerank"):
                    if key in v:
                        cfg[key] = v[key]
                ix = FaithfulDLSAIndexer(**cfg).to(dev)
                use = [s for (L, s) in tr if L in set(train_lengths) | set(v.get("train_lengths_extra", ()))]
                ix.fit_pca(torch.cat([ix.inputs(s["q"].to(dev), s["k"].to(dev))[1][:: max(1, s["k"].shape[1] // 16384)] for s in use], 0))
                vs = [{"q": s["q"].to(dev), "k": s["k"].to(dev), "v": s["v"].to(dev) if s.get("v") is not None else None} for s in va]
                arms = ("faithful", "flat_oracle_index", "contiguous", "random", "oracle")
                def measure(tag: str, vrep: Dict[str, Any]) -> None:
                    rs = [evaluate_selectors(ix, s["q"], s["k"], pos0, budget=k_budget, seed=seed, scale=geo["scale"], v=s["v"]) for s in vs]
                    def avg(key: str) -> Optional[float]:
                        xs = [r[key] for r in rs if r.get(key) is not None]
                        return sum(xs) / len(xs) if xs else None
                    vrep[tag] = {a: avg(a) for a in arms}
                    vrep[tag].update({"err_" + a: avg("err_" + a) for a in arms})
                    vrep[tag]["mean_long_keys"] = avg("mean_long_keys")
                    vrep[tag]["heldout_kl"] = eval_kl(ix, vs, scale=geo["scale"])
                vrep: Dict[str, Any] = {}
                measure("pca_init", vrep)
                tr_all = [{"q": s["q"].to(dev), "k": s["k"].to(dev)} for s in use]
                use_val = len(tr_all) >= 2
                vrep["distill"] = distill_indexer(ix, tr_all[:-1] if use_val else tr_all, steps=distill_steps, seed=seed, log_every=0,
                                                  scale=geo["scale"], val_samples=tr_all[-1:] if use_val else None)
                measure("distilled", vrep)
                lrep["variants"][name] = vrep
                log(f"[LSA-F2b] L{l} {name}: faithful {vrep['pca_init']['faithful']:.3f} -> {vrep['distilled']['faithful']:.3f} "
                    f"(flat {vrep['distilled']['flat_oracle_index']:.3f}, oracle {vrep['distilled']['oracle']:.3f}, "
                    f"held-out KL {vrep['pca_init']['heldout_kl']:.3f} -> {vrep['distilled']['heldout_kl']:.3f}; "
                    f"keys read {vrep['distilled']['mean_long_keys']:.0f}; out-err faith/oracle/contig "
                    f"{_f3(vrep['distilled']['err_faithful'])}/{_f3(vrep['distilled']['err_oracle'])}/{_f3(vrep['distilled']['err_contiguous'])})")
                del ix, vs
                _free()
            rep["layers"][str(l)] = lrep
        except Exception as e:
            rep["errors"].append(f"layer {l}: {type(e).__name__}: {str(e)[:300]}")
            _free()
    with open(os.path.join(out_dir, "lsa_f2b_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    return rep


def format_f2b(rep: Dict[str, Any]) -> str:
    rows = ["layer variant            tag        faith  flat   contig oracle  heldoutKL  keys  | out-err: faith  oracle contig random"]
    for l, lr in sorted(rep.get("layers", {}).items(), key=lambda kv: int(kv[0])):
        for name, v in lr["variants"].items():
            for tag in ("pca_init", "distilled"):
                r = v[tag]
                kl = r["heldout_kl"] if r.get("heldout_kl") is not None else float("nan")
                keys = r.get("mean_long_keys")
                rows.append(f"{int(l):>5} {name:<19} {tag:<9} {r['faithful']:6.3f} {r['flat_oracle_index']:6.3f} {r['contiguous']:6.3f} "
                            f"{r['oracle']:6.3f}  {kl:8.3f} {'  n/a' if keys is None else f'{keys:5.0f}'}  |         "
                            f"{_f3(r.get('err_faithful'))}  {_f3(r.get('err_oracle'))}  {_f3(r.get('err_contiguous'))}  {_f3(r.get('err_random'))}")
            d = v.get("distill") or {}
            if d.get("best_step") is not None:
                rows.append(f"{'':>5} {'':<19} early-stop: best step {d['best_step']}, val KL {_f3(d.get('val_kl_before'))} -> {_f3(d.get('val_kl_after'))}")
        rows.append(f"{int(l):>5} ORACLE (no index) mean per-head recall by budget:")
        for b, d in sorted(lr["oracle"].items()):
            fm = lambda x: "  n/a " if x is None else f"{x:.3f}"
            rows.append(f"        budget {b:>6}: shared {fm(d['shared'])} | per-KV-group {fm(d['per_group'])} | per-head {fm(d['per_head'])}  (n={d['n_queries']})")
    return "\n".join(rows)


# ======================================================================================
# Reporting + gate (pure python)
# ======================================================================================
GATE = {"min_recall": 0.90, "min_margin_vs_contiguous": 0.10, "max_clustering_loss": 0.03, "min_len_for_claim": 65536}


def gate_f2(report: Dict[str, Any], gate: Dict[str, float] = GATE) -> Dict[str, Any]:
    """Apply the LSA-F2 claim gate on the largest evaluated val length (must be >= `min_len_for_claim` to count as a claim)."""
    lens = sorted({int(L) for lr in report.get("layers", {}).values() for L in lr.get("val", {})})
    if not lens:
        return {"evaluated": False, "reason": "no layers evaluated", "claim_grade": False}
    L = lens[-1]
    per: Dict[str, Any] = {}
    for l, lr in report["layers"].items():
        r = lr.get("val", {}).get(str(L), {}).get("distilled", {}).get("recall", {})
        if not r or r.get("faithful") is None:
            per[l] = {"pass": False, "why": "no recall"}
            continue
        ok_abs = r["faithful"] >= gate["min_recall"]
        ok_margin = r["faithful"] >= r["contiguous"] + gate["min_margin_vs_contiguous"]
        ok_clu = (r["flat_oracle_index"] - r["faithful"]) <= gate["max_clustering_loss"]
        per[l] = {"pass": bool(ok_abs and ok_margin and ok_clu), "faithful": r["faithful"], "contiguous": r["contiguous"],
                  "random": r["random"], "flat_oracle_index": r["flat_oracle_index"], "oracle": r["oracle"],
                  "abs": bool(ok_abs), "margin": bool(ok_margin), "clustering": bool(ok_clu),
                  "faithful_over_oracle": r["faithful"] / max(1e-9, r["oracle"])}  # report-only (the v1 gate does not use it)
    passed = [int(l) for l, v in per.items() if v["pass"]]
    failed = sorted(int(l) for l, v in per.items() if not v["pass"])
    return {"evaluated": True, "length": L, "claim_grade": bool(L >= gate["min_len_for_claim"]), "layers_passed": sorted(passed),
            "layers_failed_use_exact_path": failed, "all_pass": not failed and bool(passed), "per_layer": per,
            "n_val_docs_note": "low power if val_docs_per_length == 1; repeat with more docs before claiming"}


# PROPOSED gate v2 (numbers are judgement calls, not derived; registered in docs/roadmap/07 §3.2 as proposed). Recall is judged against
# the oracle at the *same* key budget (the head-averaged teacher is diffuse, so absolute 0.90 was unreachable even for the oracle), and
# the attention-OUTPUT error — which is what the model consumes — must be close to the oracle's. End-to-end ΔCE ≤ 0.02 nats is LSA-F3.
GATE_V2 = {"min_faithful_over_oracle": 0.90, "max_err_ratio_vs_oracle": 1.5, "err_floor": 1e-3, "min_margin_vs_contiguous": 0.02,
           "min_len_for_claim": 65536}


def gate_f2_v2(report: Dict[str, Any], gate: Dict[str, float] = GATE_V2) -> Dict[str, Any]:
    """Gate v2 on the largest val length of an F2 report (needs `err_*`, i.e. the run must have captured `v`). Layers failing it stay on
    the exact attention path. Not claim-grade below `min_len_for_claim` or with 1 val doc."""
    lens = sorted({int(L) for lr in report.get("layers", {}).values() for L in lr.get("val", {})})
    if not lens:
        return {"evaluated": False, "reason": "no layers evaluated", "claim_grade": False}
    L = lens[-1]
    per: Dict[str, Any] = {}
    for l, lr in report["layers"].items():
        r = lr.get("val", {}).get(str(L), {}).get("distilled", {}).get("recall", {})
        if not r or r.get("faithful") is None or r.get("err_faithful") is None or r.get("err_oracle") is None:
            per[l] = {"pass": False, "why": "missing recall or output error (rerun with v captured)"}
            continue
        ratio = r["faithful"] / max(1e-9, r["oracle"])
        ok_rec = ratio >= gate["min_faithful_over_oracle"]
        ok_err = r["err_faithful"] <= gate["max_err_ratio_vs_oracle"] * max(r["err_oracle"], gate["err_floor"])
        ok_con = r["faithful"] >= r["contiguous"] + gate["min_margin_vs_contiguous"]
        per[l] = {"pass": bool(ok_rec and ok_err and ok_con), "faithful_over_oracle": ratio, "err_faithful": r["err_faithful"],
                  "err_oracle": r["err_oracle"], "recall_ok": bool(ok_rec), "err_ok": bool(ok_err), "margin_ok": bool(ok_con)}
    passed = sorted(int(l) for l, v in per.items() if v["pass"])
    failed = sorted(int(l) for l, v in per.items() if not v["pass"])
    return {"evaluated": True, "length": L, "claim_grade": bool(L >= gate["min_len_for_claim"]), "layers_passed": passed,
            "layers_failed_use_exact_path": failed, "all_pass": not failed and bool(passed), "per_layer": per, "gate": "v2-proposed"}


def format_table(report: Dict[str, Any]) -> str:
    rows = ["layer  len     tag        faith  flat   contig random oracle  fid_r  keys | out-err: faith oracle contig random"]
    for l, lr in sorted(report.get("layers", {}).items(), key=lambda kv: int(kv[0])):
        for L, tags in sorted(lr.get("val", {}).items(), key=lambda kv: int(kv[0])):
            for tag, v in tags.items():
                r = v.get("recall", {}) or {}
                f = lambda x: "  n/a " if x is None else f"{x:6.3f}"
                rows.append(f"{int(l):>5}  {int(L):>6}  {tag:<9} {f(r.get('faithful'))} {f(r.get('flat_oracle_index'))} "
                            f"{f(r.get('contiguous'))} {f(r.get('random'))} {f(r.get('oracle'))}  {f(v.get('fidelity_r'))} "
                            f"{'  n/a' if r.get('mean_long_keys') is None else format(r['mean_long_keys'], '5.0f')} |  "
                            f"{f(r.get('err_faithful'))} {f(r.get('err_oracle'))} {f(r.get('err_contiguous'))} {f(r.get('err_random'))}")
    return "\n".join(rows)


# ======================================================================================
# Install into the model (LSA-F3 decode path)
# ======================================================================================
def load_indexers(path: str) -> Dict[int, Dict[str, Any]]:
    return torch.load(path, map_location="cpu")["indexers"]


def upgrade_lsa_indexers(
    model: Any,
    indexers: Dict[int, Dict[str, Any]],
    k_budget: int = 2048,
    window: int = 512,
    layers_to_upgrade: Optional[Sequence[int]] = None,
    **indexer_kwargs: Any,
) -> List[int]:
    """Attach distilled `FaithfulDLSAIndexer`s as `self_attn.lsa_indexer` of the global layers. The attention patch
    (`patch_spark_attention_for_lsa`) looks the indexer up at call time, so this takes effect immediately; it is only used for
    decode-like calls with `seq_k > exact_threshold` (unchanged semantics). Returns the layers upgraded."""
    allow = set(int(x) for x in layers_to_upgrade) if layers_to_upgrade is not None else None
    done: List[int] = []
    for l, d in indexers.items():
        if allow is not None and int(l) not in allow:
            continue
        a = _layer_attn(model, int(l))
        geo = d["geometry"]
        dev = next(a.parameters()).device
        ix = FaithfulDLSAIndexer(
            n_kv=geo["n_kv"],
            head_dim=geo["head_dim"],
            rope_dim=geo["rope_dim"],
            rope_first=d["rope_first"],
            window=window,
            k_long_range=k_budget,
            **indexer_kwargs,
        )
        ix.load_state_dict(d["state"])
        ix.reset_decode_cache()
        ix.to(dev).eval()
        if "lsa_indexer" in a._modules:
            del a._modules["lsa_indexer"]
        a.add_module("lsa_indexer", ix)
        done.append(int(l))
    return sorted(done)


@torch.no_grad()
def run_lsa_f3(
    model: Any,
    tokenizer: Any,
    indexers: Dict[int, Dict[str, Any]],
    val_texts: Sequence[str],
    seq_len: int = 8192,
    k_budget: int = 2048,
    window: int = 512,
    rerank: int = 4,
    decode_tail: int = 16,
    layers_to_upgrade: Optional[Sequence[int]] = None,
    indexer_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """`LSA-F3`: End-to-end model decode CE delta (`|CE_dlsa - CE_exact| <= 0.02`), planted needle retrieval,
    per-layer tail output relative error, and incremental decode cache hit verification on the upgraded layers."""
    import importlib
    from sparkx_dlsa import _out_rel_err

    def _sparse_attn_row(q0: torch.Tensor, k0: torch.Tensor, v0: torch.Tensor, p: int, sel: torch.Tensor, sc: float) -> torch.Tensor:
        hq = q0.shape[0]
        g = hq // k0.shape[0]
        rows = []
        for h in range(hq):
            kk, vv = k0[h // g, sel].float(), v0[h // g, sel].float()
            logits = (kk @ q0[h, p].float()) * sc
            rows.append(torch.softmax(logits, dim=-1) @ vv)
        return torch.stack(rows, dim=0)

    dev = next(model.parameters()).device
    kw = dict(rerank=rerank)
    kw.update(indexer_kwargs or {})
    upgraded = upgrade_lsa_indexers(
        model, indexers, k_budget=k_budget, window=window, layers_to_upgrade=layers_to_upgrade, **kw
    )
    if not upgraded:
        return {"status": "no_layers_upgraded", "upgraded_layers": []}

    ids = pack_to_length(tokenizer, val_texts, int(seq_len), seed=seed + 777)
    if ids is None:
        return {"status": "corpus_too_small", "seq_len": seq_len}
    ids = ids.to(dev)
    t_len = int(ids.shape[1])
    tail_n = min(max(2, int(decode_tail)), t_len - 1)
    q_positions = list(range(t_len - tail_n - 1, t_len - 1))

    # 1. Sequential decode simulation per upgraded layer (always runs; verifies decode cache hits & attention output error)
    cap = capture_qk(model, ids, upgraded, device=str(dev), with_v=True)
    layer_tail_errors: Dict[int, float] = {}
    total_hits = 0
    total_misses = 0
    total_reuse_hits = 0

    for l in upgraded:
        if l not in cap or "v" not in cap[l]:
            continue
        q, k, v = cap[l]["q"].to(dev), cap[l]["k"].to(dev), cap[l]["v"].to(dev)
        kf, vf = k.float(), v.float()
        attn = _layer_attn(model, l)
        ix: FaithfulDLSAIndexer = attn.lsa_indexer
        ix.reset_decode_cache()
        sc = float(layer_geometry(model, l)["scale"] or (1.0 / math.sqrt(q.shape[-1])))
        errs = []
        for p in q_positions:
            q_step = q[:, p : p + 1].unsqueeze(0)
            k_step = k[:, : p + 1].unsqueeze(0)
            sel = ix.select_key_indices(q_step, k_step, k_budget=k_budget, use_cache=True)
            errs.append(_out_rel_err(q, kf, vf, p, {"sp": sel}, sc)["sp"])
        layer_tail_errors[l] = sum(errs) / max(1, len(errs))
        total_hits += int(ix.decode_cache_hits)
        total_misses += int(ix.decode_cache_misses)
        total_reuse_hits += int(ix.selection_reuse_hits)
    del cap
    _free()

    res: Dict[str, Any] = {
        "status": "ok",
        "seq_len": t_len,
        "decode_tail": tail_n,
        "upgraded_layers": upgraded,
        "layer_tail_out_err": layer_tail_errors,
        "mean_tail_out_err": sum(layer_tail_errors.values()) / max(1, len(layer_tail_errors)),
        "decode_cache_hits": int(total_hits),
        "decode_cache_misses": int(total_misses),
        "selection_reuse_hits": int(total_reuse_hits),
        "decode_cache_hit_rate": float(total_hits / max(1, total_hits + total_misses)),
    }

    # 2. Full model forward_logits end-to-end CE delta & planted needle retrieval (when model has forward_logits + eager_attention_forward)
    if hasattr(model, "forward_logits"):
        try:
            attn_mod = importlib.import_module(type(_layer_attn(model, upgraded[0])).__module__)
        except Exception:
            attn_mod = None
        orig_eager = getattr(attn_mod, "eager_attention_forward", None) if attn_mod is not None else None
        if orig_eager is not None:
            needle_ids = ids.clone()
            needle_str = " The vault 914 passkey is 83921. "
            cue_str = " The vault 914 passkey is"
            try:
                n_toks = torch.as_tensor(tokenizer(needle_str, add_special_tokens=False)["input_ids"], device=dev, dtype=torch.long)
                c_toks = torch.as_tensor(tokenizer(cue_str, add_special_tokens=False)["input_ids"], device=dev, dtype=torch.long)
                if n_toks.numel() >= 2 and c_toks.numel() >= 1 and t_len > (n_toks.numel() + c_toks.numel() + window + 32):
                    ins_pos = max(8, t_len // 3)
                    needle_ids[0, ins_pos : ins_pos + n_toks.numel()] = n_toks
                    needle_ids[0, t_len - 1 - c_toks.numel() : t_len - 1] = c_toks
                    needle_target_id = int(n_toks[-1].item())
                else:
                    needle_target_id = int(ids[0, -1].item())
            except Exception:
                needle_target_id = int(ids[0, -1].item())

            logits_exact = model.forward_logits(ids).float()
            ce_exact = float(
                F.cross_entropy(logits_exact[0, q_positions], ids[0, [p + 1 for p in q_positions]], reduction="mean").item()
            )
            logits_needle_exact = model.forward_logits(needle_ids).float()[0, -2]
            needle_nll_exact = float(-torch.log_softmax(logits_needle_exact, dim=-1)[needle_target_id].item())
            needle_hit_exact = bool(int(logits_needle_exact.argmax().item()) == needle_target_id)
            del logits_exact, logits_needle_exact
            _free()

            upgraded_attn_ids = {id(_layer_attn(model, l)) for l in upgraded}

            def _f3_wrapped_eager(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
                out, w = orig_eager(
                    module, query, key, value, attention_mask=attention_mask, scaling=scaling, dropout=dropout, **kwargs
                )
                ix_m = getattr(module, "lsa_indexer", None)
                if id(module) in upgraded_attn_ids and ix_m is not None and out.shape[2] == t_len:
                    sc = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
                    out = out.clone()
                    q0, k0, v0 = query[0], key[0], value[0]
                    ix_m.reset_decode_cache()
                    for p in q_positions:
                        sel = ix_m.select_key_indices(query[:, :, p : p + 1, :], key[:, :, : p + 1, :], k_budget=k_budget, use_cache=True)
                        sp_o = _sparse_attn_row(q0, k0, v0, p, sel, float(sc))  # [H_q, D]
                        out[0, :, p, :] = sp_o.to(out.dtype)
                return out, w

            try:
                for l in upgraded:
                    _layer_attn(model, l).lsa_indexer.reset_decode_cache()
                attn_mod.eager_attention_forward = _f3_wrapped_eager
                logits_dlsa = model.forward_logits(ids).float()
                ce_dlsa = float(
                    F.cross_entropy(logits_dlsa[0, q_positions], ids[0, [p + 1 for p in q_positions]], reduction="mean").item()
                )
                for l in upgraded:
                    _layer_attn(model, l).lsa_indexer.reset_decode_cache()
                logits_needle_dlsa = model.forward_logits(needle_ids).float()[0, -2]
                needle_nll_dlsa = float(-torch.log_softmax(logits_needle_dlsa, dim=-1)[needle_target_id].item())
                needle_hit_dlsa = bool(int(logits_needle_dlsa.argmax().item()) == needle_target_id)
                del logits_dlsa, logits_needle_dlsa
            finally:
                attn_mod.eager_attention_forward = orig_eager
                _free()

            delta_ce = float(ce_dlsa - ce_exact)
            res.update({
                "ce_exact": ce_exact,
                "ce_dlsa": ce_dlsa,
                "delta_ce": delta_ce,
                "ce_gate_pass": bool(abs(delta_ce) <= 0.02),
                "needle_nll_exact": needle_nll_exact,
                "needle_nll_dlsa": needle_nll_dlsa,
                "needle_delta_nll": float(needle_nll_dlsa - needle_nll_exact),
                "needle_hit_exact": needle_hit_exact,
                "needle_hit_dlsa": needle_hit_dlsa,
            })

    return res

