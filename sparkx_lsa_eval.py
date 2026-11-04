"""sparkx_lsa_eval.py — LSA-F2: real-attention recall gate for faithful D-LSA (see docs/roadmap/07, §3).

GPU path is **untested** (written crash-proof: every stage is wrapped, OOM is reported instead of raised). The pure helpers
(`pack_to_length`, `gate_f2`, `detect_rope_layout`, `format_table`) are unit-tested on CPU with synthetic tensors.

Typical Colab use (model/tokenizer from `load_sparkx_loop_v03`):

    import sparkx_lsa_eval as le
    REPORT = le.run_lsa_f2(model, tokenizer, train_texts, val_texts)        # distil on train docs, evaluate on held-out docs
    print(le.format_table(REPORT)); print(json.dumps(le.gate_f2(REPORT), indent=2))
    le.upgrade_lsa_indexers(model, le.load_indexers('/content/lsa_f2/dlsa_indexers.pt'))   # LSA-F3 decode path
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sparkx_dlsa import (FaithfulDLSAIndexer, distill_indexer, evaluate_selectors, free_logit_fidelity)

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
def capture_qk(model: Any, input_ids: torch.Tensor, layers: Sequence[int], device: str = "cpu") -> Dict[int, Dict[str, torch.Tensor]]:
    """Post-RoPE q `[H_q, T, D]` and k `[H_kv, T, D]` (bf16, on `device`) from the first visit of each requested layer."""
    ids = input_ids.to(next(model.parameters()).device)
    recs = model.capture_attention_io(ids, list(layers), visit=0)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for l in list(recs.keys()):
        r = recs.pop(l)
        if r.get("query") is None or r.get("key") is None:
            continue
        out[l] = {"q": r["query"][0].to(torch.bfloat16).to(device), "k": r["key"][0].to(torch.bfloat16).to(device)}
        del r
    del recs
    return out


def detect_rope_layout(q: torch.Tensor, k: torch.Tensor, rope_dim: int, positions: Sequence[int]) -> Dict[str, Any]:
    """Which end of the head dim is position-free? The free slice should track the full logit better than the rotated one."""
    r_first = free_logit_fidelity(q, k, positions, rope_dim, rope_first=True)["mean_pearson_r"]
    r_last = free_logit_fidelity(q, k, positions, rope_dim, rope_first=False)["mean_pearson_r"]
    return {"r_rope_first": r_first, "r_rope_last": r_last, "rope_first": bool(r_first >= r_last)}


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
) -> Dict[str, Any]:
    """Per layer: PCA-init recall (training-free D-LSA), per-key-KL distillation on train docs, recall of every arm on held-out
    docs at each val length. Returns a JSON-able report; saves distilled indexers to `out_dir/dlsa_indexers.pt`."""
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
                cap = capture_qk(model, ids, group, device="cpu")
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
                lay = detect_rope_layout(q0, k0, geo["rope_dim"], sample_positions(q0.shape[1], 8, 4096, seed))
                ix = FaithfulDLSAIndexer(n_kv=geo["n_kv"], head_dim=geo["head_dim"], rope_dim=geo["rope_dim"],
                                         rope_first=lay["rope_first"], k_long_range=k_budget).to(dev)
                ix.fit_pca(torch.cat([ix.inputs(s["q"].to(dev), s["k"].to(dev))[1][:: max(1, s["k"].shape[1] // 16384)]
                                      for s in train_s[l]], dim=0))
                lrep: Dict[str, Any] = {"geometry": geo, "rope_layout": lay, "val": {}}

                def eval_arms(tag: str) -> None:
                    for (L, s) in val_s[l]:
                        q, k = s["q"].to(dev), s["k"].to(dev)
                        pos = sample_positions(q.shape[1], n_eval_queries, ix.n_sink + ix.window + k_budget + 1, seed + L)
                        fid = free_logit_fidelity(q, k, pos[:8], geo["rope_dim"], lay["rope_first"]) if pos else {}
                        r = evaluate_selectors(ix, q, k, pos, budget=k_budget, seed=seed, scale=geo["scale"]) if pos else {}
                        lrep["val"].setdefault(str(L), {})[tag] = {"recall": r, "fidelity_r": fid.get("mean_pearson_r")}
                        del q, k

                eval_arms("pca_init")
                dist = distill_indexer(ix, [{"q": s["q"].to(dev), "k": s["k"].to(dev)} for s in train_s[l]], steps=distill_steps,
                                       lr=distill_lr, queries_per_sample=queries_per_sample, seed=seed, log_every=0,
                                       scale=geo["scale"])
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
                  "abs": bool(ok_abs), "margin": bool(ok_margin), "clustering": bool(ok_clu)}
    passed = [int(l) for l, v in per.items() if v["pass"]]
    failed = sorted(int(l) for l, v in per.items() if not v["pass"])
    return {"evaluated": True, "length": L, "claim_grade": bool(L >= gate["min_len_for_claim"]), "layers_passed": sorted(passed),
            "layers_failed_use_exact_path": failed, "all_pass": not failed and bool(passed), "per_layer": per,
            "n_val_docs_note": "low power if val_docs_per_length == 1; repeat with more docs before claiming"}


def format_table(report: Dict[str, Any]) -> str:
    rows = ["layer  len     tag        faith  flat   contig random oracle  fid_r"]
    for l, lr in sorted(report.get("layers", {}).items(), key=lambda kv: int(kv[0])):
        for L, tags in sorted(lr.get("val", {}).items(), key=lambda kv: int(kv[0])):
            for tag, v in tags.items():
                r = v.get("recall", {}) or {}
                f = lambda x: "  n/a " if x is None else f"{x:6.3f}"
                rows.append(f"{int(l):>5}  {int(L):>6}  {tag:<9} {f(r.get('faithful'))} {f(r.get('flat_oracle_index'))} "
                            f"{f(r.get('contiguous'))} {f(r.get('random'))} {f(r.get('oracle'))}  {f(v.get('fidelity_r'))}")
    return "\n".join(rows)


# ======================================================================================
# Install into the model (LSA-F3 decode path)
# ======================================================================================
def load_indexers(path: str) -> Dict[int, Dict[str, Any]]:
    return torch.load(path, map_location="cpu")["indexers"]


def upgrade_lsa_indexers(model: Any, indexers: Dict[int, Dict[str, Any]], k_budget: int = 2048, window: int = 512) -> List[int]:
    """Attach distilled `FaithfulDLSAIndexer`s as `self_attn.lsa_indexer` of the global layers. The attention patch
    (`patch_spark_attention_for_lsa`) looks the indexer up at call time, so this takes effect immediately; it is only used for
    decode-like calls with `seq_k > exact_threshold` (unchanged semantics). Returns the layers upgraded."""
    done: List[int] = []
    for l, d in indexers.items():
        a = _layer_attn(model, int(l))
        geo = d["geometry"]
        dev = next(a.parameters()).device
        ix = FaithfulDLSAIndexer(n_kv=geo["n_kv"], head_dim=geo["head_dim"], rope_dim=geo["rope_dim"], rope_first=d["rope_first"],
                                 window=window, k_long_range=k_budget)
        ix.load_state_dict(d["state"])
        ix.to(dev).eval()
        if "lsa_indexer" in a._modules:
            del a._modules["lsa_indexer"]
        a.add_module("lsa_indexer", ix)
        done.append(int(l))
    return sorted(done)
