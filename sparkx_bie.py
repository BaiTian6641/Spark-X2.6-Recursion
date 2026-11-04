"""sparkx_bie.py — Backbone↔Engram Integration Evaluation (BIE), docs/roadmap/07 §5 and 06.

Question answered here: *where and how* does the Engram writing into the frozen backbone's residual stream help or hurt, per anchor
and per loop pass, on held-out text — with one backward pass instead of dozens of ablation forwards.

First-order attribution. Every Engram call writes `delta` into the residual stream. With a zero probe `eps` added to `delta`
(`delta' = delta + eps`), `g = dLoss/d eps` is the exact gradient of the summed next-token NLL w.r.t. what the Engram writes. Then

    attribution(anchor, pass) = sum_t <g_t, delta_t> / N        (nats/token; > 0  =>  the write RAISES held-out NLL to first order)

and `-attribution` is the first-order CE change from removing that write. It is validated against finite differences in
`tests/test_bie.py` and should be cross-checked against leave-one-anchor-out on the real model (the first-order estimate ignores
curvature and interactions). Also reported: relative write size `||delta|| / ||h||`, `cos(delta, -g)` (1 = perfectly aligned with the
loss-decreasing direction), fraction of tokens whose write hurts, and the worst tokens (qualitative inspection).

GPU path untested; every public function is crash-proof (errors are returned, never raised).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ======================================================================================
# First-order attribution
# ======================================================================================
def _anchor_list(model: Any) -> List[Tuple[int, Any]]:
    return [(int(l), m) for l, m in model.iter_engram_anchors()]


@torch.enable_grad()
def engram_attribution_one(model: Any, input_ids: torch.Tensor, top_tokens: int = 8) -> Dict[str, Any]:
    """Attribution for one sequence `[1, T]`. Returns per-(layer, pass) sums to be aggregated across texts by the caller."""
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    anchors = _anchor_list(model)
    mod2layer = {id(m): l for l, m in anchors}
    probes: List[Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []  # (layer, pass, eps, delta, h)

    def hook(mod, args, kwargs, out):
        if not isinstance(out, torch.Tensor):
            return out
        eps = torch.zeros_like(out, requires_grad=True)
        hs = kwargs.get("hidden_states")
        probes.append((mod2layer[id(mod)], int(kwargs.get("pass_idx", 0) or 0), eps, out.detach(),
                       hs.detach() if isinstance(hs, torch.Tensor) else None))
        return out + eps

    handles = [m.register_forward_hook(hook, with_kwargs=True) for _, m in anchors]
    try:
        logits = model.forward_logits(ids).float()
        nll_tok = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
        loss = nll_tok.sum()
        grads = torch.autograd.grad(loss, [p[2] for p in probes], allow_unused=True) if probes else []
    finally:
        for h in handles:
            h.remove()
    n = int(ids.shape[1] - 1)
    res: Dict[str, Any] = {"n_tokens": n, "nll_sum": float(loss.detach()), "calls": []}
    worst: List[Tuple[float, int, int, int]] = []
    for (layer, p, eps, delta, hs), g in zip(probes, grads):
        if g is None:
            continue
        d = delta.float()
        g = g.float()
        per_tok = (g * d).sum(dim=-1)[0]  # [T]; the last position has no target (g = 0 there)
        dn = d.norm(dim=-1)[0]
        hn = hs.float().norm(dim=-1)[0] if hs is not None else None
        gn = g.norm(dim=-1)[0]
        cos = (-(g * d).sum(dim=-1)[0]) / (dn * gn + 1e-12)
        valid = slice(0, n)
        rec = {"layer": layer, "pass": p, "attr_sum": float(per_tok[valid].sum()), "pos_frac_num": int((per_tok[valid] > 0).sum()),
               "delta_norm_sum": float(dn[valid].sum()), "h_norm_sum": float(hn[valid].sum()) if hn is not None else None,
               "cos_sum": float(cos[valid][gn[valid] > 0].sum()), "cos_n": int((gn[valid] > 0).sum())}
        res["calls"].append(rec)
        k = min(top_tokens, n)
        vals, idx = per_tok[valid].topk(k)
        for v, i in zip(vals.tolist(), idx.tolist()):
            worst.append((float(v), int(i), layer, p))
    worst.sort(reverse=True)
    res["worst"] = [{"attr": v, "pos": i, "layer": l, "pass": p, "token_id": int(ids[0, i + 1]) if i + 1 < ids.shape[1] else None}
                    for v, i, l, p in worst[:top_tokens]]
    return res


def aggregate_attribution(per_text: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Token-weighted aggregation of `engram_attribution_one` results -> per (layer, pass) table + totals."""
    n_tok = sum(r["n_tokens"] for r in per_text)
    acc: Dict[Tuple[int, int], Dict[str, float]] = {}
    for r in per_text:
        for c in r["calls"]:
            a = acc.setdefault((c["layer"], c["pass"]), {"attr": 0.0, "pos": 0, "dn": 0.0, "hn": 0.0, "hn_ok": 0, "cos": 0.0, "cos_n": 0, "n": 0})
            a["attr"] += c["attr_sum"]
            a["pos"] += c["pos_frac_num"]
            a["dn"] += c["delta_norm_sum"]
            if c["h_norm_sum"] is not None:
                a["hn"] += c["h_norm_sum"]
                a["hn_ok"] += r["n_tokens"]
            a["cos"] += c["cos_sum"]
            a["cos_n"] += c["cos_n"]
            a["n"] += r["n_tokens"]
    rows = []
    for (l, p), a in sorted(acc.items()):
        rows.append({"layer": l, "pass": p, "attr_nats_per_token": a["attr"] / max(1, a["n"]),
                     "frac_tokens_hurt": a["pos"] / max(1, a["n"]),
                     "rel_write_size": (a["dn"] / a["hn"]) if a["hn"] > 0 else None,
                     "cos_with_descent": a["cos"] / max(1, a["cos_n"])})
    by_layer: Dict[int, float] = {}
    by_pass: Dict[int, float] = {}
    for r_ in rows:
        by_layer[r_["layer"]] = by_layer.get(r_["layer"], 0.0) + r_["attr_nats_per_token"]
        by_pass[r_["pass"]] = by_pass.get(r_["pass"], 0.0) + r_["attr_nats_per_token"]
    return {"n_tokens": n_tok, "rows": rows, "by_layer": by_layer, "by_pass": by_pass,
            "total_first_order_dce": sum(by_layer.values())}


def format_attribution(agg: Dict[str, Any]) -> str:
    out = ["layer pass  attr(nats/tok)  frac_hurt  |delta|/|h|  cos(delta,-g)   (attr > 0  => the write RAISES held-out CE)"]
    for r in agg["rows"]:
        rw = "   n/a " if r["rel_write_size"] is None else f"{r['rel_write_size']:7.3f}"
        out.append(f"{r['layer']:>5} {r['pass']:>4}  {r['attr_nats_per_token']:+14.5f}  {r['frac_tokens_hurt']:9.3f}  {rw}      {r['cos_with_descent']:+8.4f}")
    out.append("by layer: " + ", ".join(f"L{l}:{v:+.4f}" for l, v in sorted(agg["by_layer"].items())))
    out.append("by pass : " + ", ".join(f"P{p}:{v:+.4f}" for p, v in sorted(agg["by_pass"].items())))
    out.append(f"TOTAL first-order dCE of all Engram writes: {agg['total_first_order_dce']:+.5f} nats/token over {agg['n_tokens']} tokens")
    return "\n".join(out)


# ======================================================================================
# Pass-schedule sweep (H7: anchors inside the loop fire on every pass)
# ======================================================================================
PASS_FILTERS: Dict[str, Optional[Tuple[int, ...]]] = {"all_passes(trained)": None, "pass0_only": (0,), "pass0+1": (0, 1)}


def _ce_texts(model: Any, tokenizer: Any, texts: Sequence[str], max_tokens: int) -> Tuple[float, int]:
    device = next(model.parameters()).device
    tot, n = 0.0, 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens].to(device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            lg = model.forward_logits(ids).float()
        tot += float(F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum"))
        n += int(ids.shape[1] - 1)
    return tot, n


def pass_filter_sweep(model: Any, tokenizer: Any, texts: Sequence[str], max_tokens: int = 2048) -> Dict[str, Any]:
    """Held-out CE with Engram OFF vs ON under different in-loop pass filters. Prelude/coda anchors are unaffected by the filter."""
    out: Dict[str, Any] = {}
    prev = getattr(model, "engram_pass_filter", None)
    try:
        for name, flt in PASS_FILTERS.items():
            model.engram_pass_filter = flt
            tot, n = _ce_texts(model, tokenizer, texts, max_tokens)
            out[name] = tot / max(1, n)
    finally:
        model.engram_pass_filter = prev
    return out


# ======================================================================================
# Driver
# ======================================================================================
def run_bie(model: Any, tokenizer: Any, texts: Optional[Sequence[str]] = None, max_tokens: int = 2048,
            loop_kwargs: Optional[Dict[str, Any]] = None, flight_mode: str = "deep_knowledge", log=print) -> Dict[str, Any]:
    """OFF/ON CE, first-order attribution per (anchor, pass), pass-filter sweep. Restores the model's loop config afterwards."""
    import sparkx_engram_generalization as eg

    texts = list(texts) if texts is not None else list(eg.TRUE_HELDOUT_TEXTS)
    kw = dict(start=4, end=28, passes=2, use_mhc=True, enable_p3=True, p3_start=12, p3_end=24)
    kw.update(loop_kwargs or {})
    rep: Dict[str, Any] = {"n_texts": len(texts), "errors": []}
    was_training = model.training
    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    try:
        model.settings.checkpoint_layers = False
    except Exception:
        pass
    try:
        model.set_engram_flight_mode(flight_mode)
        model.set_domain_engram_focus("general", boost=0.0)
    except Exception as e:
        rep["errors"].append(f"flight_mode: {e}")
    try:
        model.set_loop(use_engram=False, **kw)
        t0, n0 = _ce_texts(model, tokenizer, texts, max_tokens)
        model.set_loop(use_engram=True, **kw)
        t1, n1 = _ce_texts(model, tokenizer, texts, max_tokens)
        rep.update(ce_off=t0 / max(1, n0), ce_on=t1 / max(1, n1), n_tokens=n1)
        rep["engram_delta_ce"] = rep["ce_on"] - rep["ce_off"]
    except Exception as e:
        rep["errors"].append(f"ce: {type(e).__name__}: {str(e)[:200]}")
    try:
        per_text = []
        device = next(model.parameters()).device
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens].to(device)
            if ids.shape[1] < 2:
                continue
            per_text.append(engram_attribution_one(model, ids))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        agg = aggregate_attribution(per_text)
        rep["attribution"] = agg
        worst = sorted((w for r in per_text for w in r["worst"]), key=lambda w: -w["attr"])[:12]
        for w in worst:
            try:
                w["token"] = tokenizer.decode([w["token_id"]]) if w.get("token_id") is not None else None
            except Exception:
                w["token"] = None
        rep["worst_tokens"] = worst
        log(format_attribution(agg))
    except Exception as e:
        rep["errors"].append(f"attribution: {type(e).__name__}: {str(e)[:300]}")
    try:
        rep["pass_filter_ce"] = pass_filter_sweep(model, tokenizer, texts, max_tokens)
        log("pass-filter CE: " + ", ".join(f"{k}={v:.4f}" for k, v in rep["pass_filter_ce"].items()))
    except Exception as e:
        rep["errors"].append(f"pass_filter: {type(e).__name__}: {str(e)[:200]}")
    finally:
        try:
            model.engram_pass_filter = None
            model.set_loop(use_engram=True, **kw)
        except Exception:
            pass
        model.enable_checkpoint = prev_ckpt
        if was_training:
            model.train()
            model.base.eval()
    rep["verdict"] = interpret_bie(rep)
    return rep


def interpret_bie(rep: Dict[str, Any], noise: float = 0.005) -> Dict[str, Any]:
    """Plain-language reading of the report; never claims more than the data supports."""
    out: Dict[str, Any] = {}
    agg = rep.get("attribution")
    if "engram_delta_ce" in rep:
        d = rep["engram_delta_ce"]
        out["engram_net"] = "hurts" if d > noise else ("helps" if d < -noise else "neutral (within noise)")
    if agg:
        harmful = sorted(l for l, v in agg["by_layer"].items() if v > 0.002)
        helpful = sorted(l for l, v in agg["by_layer"].items() if v < -0.002)
        out["first_order_harmful_layers"] = harmful
        out["first_order_helpful_layers"] = helpful
        late = [r for r in agg["rows"] if r["pass"] >= 1]
        early = [r for r in agg["rows"] if r["pass"] == 0]
        out["pass0_attr_total"] = sum(r["attr_nats_per_token"] for r in early)
        out["later_passes_attr_total"] = sum(r["attr_nats_per_token"] for r in late)
        out["misaligned_calls(cos<0.02)"] = [(r["layer"], r["pass"]) for r in agg["rows"] if r["cos_with_descent"] < 0.02]
    pf = rep.get("pass_filter_ce")
    if pf and "all_passes(trained)" in pf:
        best = min(pf, key=lambda k: pf[k])
        out["best_pass_filter"] = best
        out["pass_filter_gain_vs_trained"] = pf["all_passes(trained)"] - pf[best]
    out["caveat"] = "first-order estimate; confirm with leave-one-anchor-out / pass-filter CE; low power with few texts"
    return out
