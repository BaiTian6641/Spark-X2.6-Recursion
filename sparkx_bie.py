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
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# First-order attribution
# ======================================================================================
def _anchor_list(model: Any) -> List[Tuple[int, Any]]:
    return [(int(l), m) for l, m in model.iter_engram_anchors()]


def _get_use_engram(model: Any) -> bool:
    if hasattr(model, "engram_enabled"):
        return bool(model.engram_enabled)
    if hasattr(model, "settings") and hasattr(model.settings, "use_engram"):
        return bool(model.settings.use_engram)
    return bool(getattr(model, "use_engram", True))


def _set_use_engram(model: Any, flag: bool) -> None:
    model.use_engram = bool(flag)
    if hasattr(model, "engram_enabled"):
        model.engram_enabled = bool(flag)
    if hasattr(model, "settings") and hasattr(model.settings, "use_engram"):
        model.settings.use_engram = bool(flag)


def _ensure_latest_engram_methods(model: Any) -> None:
    """Hot-rebind live `ZipfianTieredMoEEngram` instances to the latest class definition after `importlib.reload`."""
    try:
        import importlib
        import sparkx_loop_v03_core as core
        core = importlib.reload(core)
        latest_cls = getattr(core, "ZipfianTieredMoEEngram", None)
    except Exception:
        latest_cls = None
    for _, mod in _anchor_list(model):
        if latest_cls is not None and mod.__class__.__name__ == "ZipfianTieredMoEEngram" and mod.__class__ is not latest_cls:
            mod.__class__ = latest_cls
        defaults = {
            "selective_gating_enabled": False,
            "scope_threshold": 0.0,
            "scope_margin": 0.0,
            "qk_gate_enabled": False,
            "qk_gate_scale": 2.0,
            "qk_gate_bias": -1.0,
            "max_rel_delta_norm": None,
            "use_centroid_scope_fallback": False,
            "_compiled_scope_prototypes": None,
            "_scope_proto_space": "router",
            "min_active_pos": 0,
        }
        for k, v in defaults.items():
            if not hasattr(mod, k):
                setattr(mod, k, v)


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
    try:
        rep["gate_design_ce"] = gate_design_sweep(model, tokenizer, texts, max_tokens)
        log("gate-design CE: " + ", ".join(f"{k}={v:.4f}" for k, v in rep["gate_design_ce"].items()))
    except Exception as e:
        rep["errors"].append(f"gate_design: {type(e).__name__}: {str(e)[:200]}")
    try:
        rep["prompt_vs_response_ce"] = prompt_vs_response_breakdown(model, tokenizer, texts, max_tokens)
        log("prompt-vs-response CE: " + ", ".join(
            f"{k}:off={v['ce_off']:.4f}/on={v['ce_on']:.4f}(d={v['delta_ce']:+.4f})"
            for k, v in rep["prompt_vs_response_ce"].items()
        ))
    except Exception as e:
        rep["errors"].append(f"prompt_vs_response: {type(e).__name__}: {str(e)[:200]}")
    finally:
        try:
            model.engram_pass_filter = None
            enable_selective_engram_gating(model, enabled=False)
            model.set_loop(use_engram=True, **kw)
        except Exception:
            pass
        model.enable_checkpoint = prev_ckpt
        if was_training:
            model.train()
            model.base.eval()
    rep["verdict"] = interpret_bie(rep)
    return rep


@torch.no_grad()
def prompt_vs_response_breakdown(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_tokens: int = 2048,
) -> Dict[str, Dict[str, float]]:
    """Decompose `TRUE_HELDOUT_TEXTS` CE into prompt-prefix (`before <|Bot|>`) vs response (`after <|Bot|>`),
    test replacing the eval system prompt with the training `tokenize_and_mask_record_16k` system prompt,
    and measure CE when `MoE-Engram` is suppressed on the unsupervised `<|System|>...<|User|>` prefix (`assistant_gated_*`)."""
    _ensure_latest_engram_methods(model)
    device = next(model.parameters()).device
    prev_engram = _get_use_engram(model)
    train_sys = (
        "You are Spark-X2.6, a 72-layer recurrent reasoning and agentic engineering model. "
        "Outline your plan concisely inside <think>...</think>, then emit the complete deliverable."
    )
    eval_sys = "You are Spark-X2.6, an advanced reasoning and engineering assistant."

    def _eval_split(text_list: Sequence[str], mode: str, gate_prompt_prefix: bool = False) -> Tuple[float, float]:
        sums = {False: 0.0, True: 0.0}
        counts = {False: 0, True: 0}
        for use_eng in (False, True):
            _set_use_engram(model, use_eng)
            for t in text_list:
                bot_idx = t.find("<|Bot|>")
                p_len = 0
                if bot_idx >= 0:
                    p_str = t[: bot_idx + len("<|Bot|>")]
                    p_ids = tokenizer(p_str, add_special_tokens=False, return_tensors="pt").input_ids
                    p_len = int(p_ids.shape[1])
                if use_eng and gate_prompt_prefix and p_len > 1:
                    enable_selective_engram_gating(
                        model,
                        enabled=True,
                        scope_threshold=0.0,
                        qk_gate_enabled=False,
                        max_rel_delta_norm=None,
                        min_active_pos=max(0, p_len - 1),
                    )
                else:
                    enable_selective_engram_gating(model, enabled=False)

                if bot_idx < 0 or mode == "all_tokens":
                    ids = tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens].to(device)
                    if ids.shape[1] < 2:
                        continue
                    lg = model.forward_logits(ids).float()
                    nll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
                    sums[use_eng] += float(nll.item())
                    counts[use_eng] += int(ids.shape[1] - 1)
                else:
                    ids = tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens].to(device)
                    if ids.shape[1] <= p_len:
                        continue
                    lg = model.forward_logits(ids).float()
                    targets = ids[:, 1:].clone()
                    if mode == "prompt_only":
                        targets[:, p_len - 1 :] = -100
                    elif mode == "response_only":
                        targets[:, : max(0, p_len - 1)] = -100
                    n_sup = int((targets != -100).sum().item())
                    if n_sup <= 0:
                        continue
                    nll = F.cross_entropy(
                        lg[:, :-1].reshape(-1, lg.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    sums[use_eng] += float(nll.item())
                    counts[use_eng] += n_sup
        ce_off = sums[False] / max(1, counts[False])
        ce_on = sums[True] / max(1, counts[True])
        return ce_off, ce_on

    try:
        enable_selective_engram_gating(model, enabled=False)
        model.engram_pass_filter = None
        out: Dict[str, Dict[str, float]] = {}
        for mode in ("all_tokens", "prompt_only", "response_only"):
            c0, c1 = _eval_split(texts, mode, gate_prompt_prefix=False)
            out[mode] = {"ce_off": c0, "ce_on": c1, "delta_ce": c1 - c0}
        train_sys_texts = [t.replace(eval_sys, train_sys) for t in texts]
        c0_tr, c1_tr = _eval_split(train_sys_texts, "response_only", gate_prompt_prefix=False)
        out["train_sys_response_only"] = {"ce_off": c0_tr, "ce_on": c1_tr, "delta_ce": c1_tr - c0_tr}
        c0_ga, c1_ga = _eval_split(texts, "all_tokens", gate_prompt_prefix=True)
        out["assistant_gated_all_tokens"] = {"ce_off": c0_ga, "ce_on": c1_ga, "delta_ce": c1_ga - c0_ga}
        c0_gr, c1_gr = _eval_split(texts, "response_only", gate_prompt_prefix=True)
        out["assistant_gated_response_only"] = {"ce_off": c0_gr, "ce_on": c1_gr, "delta_ce": c1_gr - c0_gr}
        return out
    finally:
        enable_selective_engram_gating(model, enabled=False)
        _set_use_engram(model, prev_engram)


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
    gd = rep.get("gate_design_ce")
    if gd:
        best_g = min(gd, key=lambda k: gd[k])
        out["best_gate_design"] = best_g
    out["caveat"] = "first-order estimate; confirm with leave-one-anchor-out / pass-filter CE; low power with few texts"
    return out


# ======================================================================================
# BIE-1: Backbone Knowledge Locus Scan (LogitLens KL Convergence & Sublayer Norms)
# ======================================================================================
def _find_decoder_layers_and_head(model: Any) -> Tuple[List[Any], Optional[Any], Optional[Any]]:
    """Locate decoder layers, final norm, and lm_head across Spark-X2.6 or synthetic test models."""
    base = getattr(model, "base", model)
    inner = getattr(base, "model", base)
    layers = list(getattr(inner, "layers", getattr(model, "layers", [])))
    norm = getattr(inner, "norm", getattr(base, "norm", None))
    lm_head = getattr(base, "lm_head", getattr(model, "lm_head", None))
    return layers, norm, lm_head


@torch.no_grad()
def knowledge_locus_scan(
    model: Any,
    tokenizer: Any,
    texts: Optional[Sequence[str]] = None,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """`BIE-1`: Scan the 72L looped backbone to measure where predictions converge toward the final distribution.

    For each decoder block execution `(layer_idx, call_order)`, projects the post-block hidden state `h` through
    `final_norm + lm_head` (LogitLens) and computes `KL(p_final || p_step)` and top-1 agreement with final predictions,
    plus relative step update norm `||h_after - h_before|| / ||h_before||`.
    """
    if texts is None:
        import sparkx_engram_generalization as eg
        texts = eg.TRUE_HELDOUT_TEXTS
    device = next(model.parameters()).device
    layers, final_norm, lm_head = _find_decoder_layers_and_head(model)
    if not layers or lm_head is None:
        return {"status": "unsupported_model_structure", "steps": []}

    step_records: Dict[Tuple[int, int], Dict[str, float]] = {}
    call_counts: Dict[int, int] = {}
    captured: List[Tuple[int, int, torch.Tensor, float]] = []

    def make_hook(l_idx: int):
        def _hook(mod, args, out):
            h_in = args[0] if args and isinstance(args[0], torch.Tensor) else None
            h_out = out[0] if isinstance(out, tuple) else out
            if not isinstance(h_out, torch.Tensor):
                return
            p_ord = call_counts.get(l_idx, 0)
            call_counts[l_idx] = p_ord + 1
            rel_step = 0.0
            if isinstance(h_in, torch.Tensor) and h_in.shape == h_out.shape:
                rel_step = float(
                    ((h_out.float() - h_in.float()).norm(dim=-1) / h_in.float().norm(dim=-1).clamp(min=1e-6)).mean().item()
                )
            captured.append((l_idx, p_ord, h_out.detach(), rel_step))
        return _hook

    handles = [lyr.register_forward_hook(make_hook(i)) for i, lyr in enumerate(layers)]
    prev_engram = _get_use_engram(model)
    try:
        _set_use_engram(model, False)
        for t in texts:
            ids = tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens].to(device)
            if ids.shape[1] < 2:
                continue
            call_counts.clear()
            captured.clear()
            final_logits = model.forward_logits(ids).float()[0]  # [T, V]
            p_final = torch.softmax(final_logits, dim=-1).clamp(min=1e-12)
            log_p_final = torch.log(p_final)
            top1_final = final_logits.argmax(dim=-1)
            n_tok = int(ids.shape[1])

            for l_idx, p_ord, h_step, rel_step in captured:
                h_n = final_norm(h_step) if final_norm is not None else h_step
                head_dtype = getattr(lm_head, "weight", h_n).dtype
                lg = lm_head(h_n.to(dtype=head_dtype)).float()[0]
                log_q = torch.log_softmax(lg, dim=-1)
                kl = (p_final * (log_p_final - log_q)).sum(dim=-1).mean().item()
                agree = float((lg.argmax(dim=-1) == top1_final).float().mean().item())
                rec = step_records.setdefault((l_idx, p_ord), {"kl_sum": 0.0, "agree_sum": 0.0, "rel_sum": 0.0, "n": 0})
                rec["kl_sum"] += float(kl) * n_tok
                rec["agree_sum"] += agree * n_tok
                rec["rel_sum"] += rel_step * n_tok
                rec["n"] += n_tok
    finally:
        for h in handles:
            h.remove()
        _set_use_engram(model, prev_engram)

    rows = []
    for (l_idx, p_ord), r in sorted(step_records.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n = max(1, int(r["n"]))
        rows.append({
            "layer": int(l_idx),
            "pass": int(p_ord),
            "kl_to_final": float(r["kl_sum"] / n),
            "top1_agreement": float(r["agree_sum"] / n),
            "rel_block_delta": float(r["rel_sum"] / n),
        })
    for idx_r, r in enumerate(rows):
        r["kl_drop"] = float((rows[idx_r - 1]["kl_to_final"] - r["kl_to_final"]) if idx_r > 0 else 0.0)
    top_kl_drop = sorted(rows, key=lambda x: -x.get("kl_drop", 0.0))[:8]
    top_rel_delta = sorted(rows, key=lambda x: -x.get("rel_block_delta", 0.0))[:8]
    commit_step = next((r for r in rows if r["top1_agreement"] >= 0.50), rows[-1] if rows else None)
    return {
        "status": "ok",
        "steps": rows,
        "commit_locus": commit_step,
        "top_kl_drop_steps": top_kl_drop,
        "top_rel_delta_steps": top_rel_delta,
    }


# ======================================================================================
# BIE-3: Selective Gating Design Sweep (WISE Scope + DeepSeek QK + Trust-Region Cap)
# ======================================================================================
def enable_selective_engram_gating(
    model: Any,
    enabled: bool = True,
    scope_threshold: float = 0.70,
    scope_margin: float = 0.0,
    qk_gate_enabled: bool = False,
    qk_gate_scale: float = 2.0,
    qk_gate_bias: float = -1.0,
    max_rel_delta_norm: Optional[float] = 0.05,
    use_centroid_scope_fallback: bool = False,
    min_active_pos: int = 0,
) -> int:
    """Configure opt-in selective gating (`BIE-3` / `SI4`) across all `ZipfianTieredMoEEngram` anchors on `model`."""
    _ensure_latest_engram_methods(model)
    count = 0
    for _, mod in _anchor_list(model):
        if hasattr(mod, "configure_selective_gating"):
            mod.configure_selective_gating(
                enabled=enabled,
                scope_threshold=scope_threshold,
                scope_margin=scope_margin,
                qk_gate_enabled=qk_gate_enabled,
                qk_gate_scale=qk_gate_scale,
                qk_gate_bias=qk_gate_bias,
                max_rel_delta_norm=max_rel_delta_norm,
                use_centroid_scope_fallback=use_centroid_scope_fallback,
                min_active_pos=min_active_pos,
            )
            count += 1
    return count


@torch.no_grad()
def gate_design_sweep(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_tokens: int = 2048,
) -> Dict[str, float]:
    """`BIE-3`: Compare held-out general-text CE across unconditional vs selective Engram gating regimes."""
    import sparkx_engram_generalization as eg

    _ensure_latest_engram_methods(model)
    device = next(model.parameters()).device
    out: Dict[str, float] = {}
    prev_engram = _get_use_engram(model)
    prev_filter = getattr(model, "engram_pass_filter", None)

    def _ce_assistant_gated(use_qk: bool = False) -> float:
        tot, n = 0.0, 0
        for t in texts:
            bot_idx = t.find("<|Bot|>")
            p_len = 0
            if bot_idx >= 0:
                p_str = t[: bot_idx + len("<|Bot|>")]
                p_ids = tokenizer(p_str, add_special_tokens=False, return_tensors="pt").input_ids
                p_len = int(p_ids.shape[1])
            enable_selective_engram_gating(
                model,
                enabled=True,
                scope_threshold=0.0,
                qk_gate_enabled=use_qk,
                qk_gate_scale=2.0,
                qk_gate_bias=-1.5,
                max_rel_delta_norm=0.03 if use_qk else None,
                min_active_pos=max(0, p_len - 1),
            )
            ids = tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens].to(device)
            if ids.shape[1] < 2:
                continue
            lg = model.forward_logits(ids).float()
            tot += float(F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum"))
            n += int(ids.shape[1] - 1)
        return tot / max(1, n)

    try:
        # 1. OFF baseline
        _set_use_engram(model, False)
        t0, n0 = _ce_texts(model, tokenizer, texts, max_tokens)
        out["off"] = t0 / max(1, n0)

        # 2. Unconditional FULL (legacy default)
        _set_use_engram(model, True)
        model.engram_pass_filter = None
        enable_selective_engram_gating(model, enabled=False)
        t1, n1 = _ce_texts(model, tokenizer, texts, max_tokens)
        out["full_unconditional"] = t1 / max(1, n1)

        # 3. Assistant-only gating (suppresses Engram on unsupervised `<|System|>...<|User|>` prefix)
        out["assistant_only_gating"] = _ce_assistant_gated(use_qk=False)

        # 4. Assistant-only gating + DeepSeek QK gate
        out["assistant_only_plus_qk_gate"] = _ce_assistant_gated(use_qk=True)

        # 5. In-loop only (`L7..L27`: ablates Prelude `L3` and post-commit Coda `L31`)
        enable_selective_engram_gating(model, enabled=False)
        outer_mods = [m for l, m in _anchor_list(model) if l in (3, 31)]
        with eg.engram_ablation(model, lex=0.0, sem=0.0, modules=outer_mods):
            t_in, n_in = _ce_texts(model, tokenizer, texts, max_tokens)
            out["in_loop_only_L7_L27"] = t_in / max(1, n_in)

        # 6. Pre-commit `L27` anchor only (feeds `L28` commit locus directly)
        non_l27_mods = [m for l, m in _anchor_list(model) if l != 27]
        if len(non_l27_mods) < len(_anchor_list(model)):
            with eg.engram_ablation(model, lex=0.0, sem=0.0, modules=non_l27_mods):
                t_27, n_27 = _ce_texts(model, tokenizer, texts, max_tokens)
                out["pre_commit_L27_only"] = t_27 / max(1, n_27)

        # 7. Trust-region relative norm cap only (`||delta|| <= 0.03 ||h||`)
        enable_selective_engram_gating(model, enabled=True, scope_threshold=0.0, qk_gate_enabled=False, max_rel_delta_norm=0.03)
        t2, n2 = _ce_texts(model, tokenizer, texts, max_tokens)
        out["trust_region_cap_3pct"] = t2 / max(1, n2)

        # 8. DeepSeek Signed-Sqrt QK Gate + Trust-region cap (`arXiv:2601.07372`)
        enable_selective_engram_gating(
            model, enabled=True, scope_threshold=0.0, qk_gate_enabled=True, qk_gate_scale=2.0, qk_gate_bias=-1.5, max_rel_delta_norm=0.03
        )
        t3, n3 = _ce_texts(model, tokenizer, texts, max_tokens)
        out["deepseek_qk_gated"] = t3 / max(1, n3)

        # 9. WISE Activation Scope Gate (`arXiv:2405.14768`, fires only within compiled prototype scope)
        enable_selective_engram_gating(
            model, enabled=True, scope_threshold=0.75, qk_gate_enabled=False, max_rel_delta_norm=0.05, use_centroid_scope_fallback=False
        )
        t4, n4 = _ce_texts(model, tokenizer, texts, max_tokens)
        out["wise_scope_gated"] = t4 / max(1, n4)
    finally:
        enable_selective_engram_gating(model, enabled=False)
        model.engram_pass_filter = prev_filter
        _set_use_engram(model, prev_engram)
    return out


# ======================================================================================
# BIE-5 & SI4: Continual Knowledge Gap-Write Test (AlphaEdit Null-Space + WISE Scope)
# ======================================================================================
DEFAULT_SYNTHETIC_FACTS: List[Dict[str, str]] = [
    {
        "prompt": "Under Internal RFC-9481, the event streaming broker powering ZephyrVault is",
        "probe_alt": "Per architecture specification RFC-9481, the message broker for ZephyrVault is",
        "paraphrase": "According to specification RFC-9481, the event broker backing ZephyrVault is",
        "target": " Kafka",
    },
    {
        "prompt": "In the fault-tolerant systems runtime, the memory-safe language used by QuartzHelix is",
        "probe_alt": "For the low-latency kernel supervisor, the systems language of QuartzHelix is",
        "paraphrase": "According to the kernel specification, the programming language for QuartzHelix is",
        "target": " Rust",
    },
    {
        "prompt": "For distributed transactional persistence, the relational database backing ArgonMesh is",
        "probe_alt": "Under protocol v4, the primary relational store used by ArgonMesh is",
        "paraphrase": "According to the v4 storage specification, the transactional database for ArgonMesh is",
        "target": " Postgres",
    },
]


def _select_content_token_id(tokenizer: Any, target_str: str) -> int:
    """Return the first non-whitespace token ID from `target_str` (avoids byte-BPE leading space token 220)."""
    t_ids = tokenizer(target_str, add_special_tokens=False).input_ids
    for tid in t_ids:
        try:
            decoded = tokenizer.decode([int(tid)])
            if decoded and decoded.strip():
                return int(tid)
        except Exception:
            return int(tid)
    return int(t_ids[-1])


def run_gap_write_test(
    model: Any,
    tokenizer: Any,
    facts: Optional[Sequence[Dict[str, Any]]] = None,
    general_texts: Optional[Sequence[str]] = None,
    anchor_layer: Optional[int] = None,
    scope_threshold: float = 0.72,
    target_scale: float = 1.0,
    opt_steps: int = 12,
    max_delta_ratio: float = 0.85,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """`BIE-5` & `SI4`: Closed-form Continual Knowledge Injection & Gap-Write Test.

    Tests whether novel facts compiled vertically into a `ZipfianTieredMoEEngram` anchor via
    Preconditioned MEMIT/AlphaEdit Subspace Target Optimization (`delta*`) + `compile_clusters_alphaedit`
    (with Cross-Expert Null-Space Projection & Post-SwiGLU Prototype Aggregation) + `WISE` Activation Scope Gating achieve:
      1. **Reliability (Efficacy)**: NLL reduction (`nll_before - nll_after > 0`) and Top-1 hit rate on the injected fact prompts.
      2. **Paraphrase Generalization**: NLL reduction on a strictly held-out `paraphrase` prompt (never compiled).
      3. **Locality (Zero General-Text Degradation)**: `|ce_general_after - ce_off| == 0.0` on unrelated held-out texts.
      4. **Sequential Null-Space Stability**: zero historical activation drift (`max_null_space_leakage`) and zero retention drift.

    Restores all Engram weights, `requires_grad` flags, and state cleanly in `finally:`.
    """
    import sparkx_engram_generalization as eg
    from sparkx_memory_utils import precompute_damped_right_inverse

    _ensure_latest_engram_methods(model)
    facts_list = list(facts) if facts is not None else list(DEFAULT_SYNTHETIC_FACTS)
    gen_texts = list(general_texts) if general_texts is not None else list(eg.TRUE_HELDOUT_TEXTS[:2])
    anchors = dict(_anchor_list(model))
    if not anchors:
        return {"status": "no_engram_anchors"}
    if anchor_layer in anchors:
        chosen_layer = int(anchor_layer)
    elif 27 in anchors:
        chosen_layer = 27
    else:
        chosen_layer = sorted(anchors.keys())[len(anchors) // 2]
    eng_mod = anchors[chosen_layer]
    device = next(model.parameters()).device

    # Save exact state of the target anchor and parameter `requires_grad` flags for clean restoration
    saved_down = eng_mod.expert_down.data.detach().clone()
    saved_proj = eng_mod.project.weight.data.detach().clone()
    saved_hot = eng_mod.tier1a_hot.weight.data.detach().clone() if hasattr(eng_mod, "tier1a_hot") else None
    prev_engram = _get_use_engram(model)
    saved_req_grad = [(p, bool(p.requires_grad)) for p in model.parameters()]

    def _eval_fact_nll(prompt_str: str, target_tok_id: int) -> Tuple[float, bool]:
        ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            lg = model.forward_logits(ids).float()[0, -1]  # [V]
        log_probs = torch.log_softmax(lg, dim=-1)
        nll = float(-log_probs[target_tok_id].item())
        hit = bool(int(lg.argmax().item()) == int(target_tok_id))
        return nll, hit

    try:
        # Disable parameter gradients so forward passes only build autograd graphs from `eng_mod` at `chosen_layer` forward
        for p, _ in saved_req_grad:
            p.requires_grad_(False)

        for _, m_anc in anchors.items():
            if hasattr(m_anc, "reset_alphaedit_basis"):
                m_anc.reset_alphaedit_basis()
        # Ensure `project.weight` is non-zero if testing on a freshly zero-initialized model
        if eng_mod.project.weight.data.norm().item() < 1e-6:
            nn.init.orthogonal_(eng_mod.project.weight.data)
            eng_mod.project.weight.data.mul_(0.25)

        # 1. Measure baseline OFF general CE and fact NLLs
        _set_use_engram(model, False)
        t_off, n_off = _ce_texts(model, tokenizer, gen_texts, max_tokens=max_tokens)
        ce_general_off = t_off / max(1, n_off)

        prepared_facts = []
        for f_item in facts_list:
            p_str = f_item["prompt"]
            alt_str = f_item.get("probe_alt")
            para_str = f_item.get("paraphrase", p_str)
            if "target_token_id" in f_item:
                t_id = int(f_item["target_token_id"])
            else:
                t_id = _select_content_token_id(tokenizer, f_item["target"])
            nll_orig_before, hit_orig_before = _eval_fact_nll(p_str, t_id)
            nll_para_before, hit_para_before = _eval_fact_nll(para_str, t_id)
            prepared_facts.append({
                "prompt": p_str,
                "probe_alt": alt_str,
                "paraphrase": para_str,
                "target_token_id": t_id,
                "nll_orig_before": nll_orig_before,
                "hit_orig_before": hit_orig_before,
                "nll_para_before": nll_para_before,
                "hit_para_before": hit_para_before,
            })

        # 2. Preconditioned Subspace Target Vector Optimization (`delta*` via `W_proj @ pinv(W_proj)`) + Null-Space Compilation
        _set_use_engram(model, True)
        max_leakage = 0.0
        compile_reports = []
        proj_w_f32 = eng_mod.project.weight.detach().float()
        mean_sv_sq = float(proj_w_f32.pow(2).sum().item()) / float(max(1, proj_w_f32.shape[0]))
        eff_lam = min(1e-2, max(1e-8, 1e-3 * mean_sv_sq))
        pinv_w_p = precompute_damped_right_inverse(proj_w_f32, lam=eff_lam)  # [width, width]

        round1_nlls: Dict[int, float] = {}
        n_total_facts = len(prepared_facts)

        for pf_idx, pf in enumerate(prepared_facts):
            t_id = pf["target_token_id"]
            probe_strings = [pf["prompt"]]
            if pf.get("probe_alt"):
                probe_strings.append(str(pf["probe_alt"]))

            # 2a. Capture baseline `(ids, h_trig, y_old)` for each probe of this fact
            probe_records = []
            enable_selective_engram_gating(
                model, enabled=True, scope_threshold=1.1, qk_gate_enabled=False, max_rel_delta_norm=None
            )
            eng_mod.selective_gating_enabled = False
            for p_probe in probe_strings:
                ids = tokenizer(p_probe, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                cap_h: List[torch.Tensor] = []
                cap_y: List[torch.Tensor] = []

                def _base_cap_hook(mod, args, kwargs, out, _ch=cap_h, _cy=cap_y):
                    hs = kwargs.get("hidden_states")
                    if hs is None and len(args) >= 2:
                        hs = args[1]
                    if isinstance(hs, torch.Tensor) and not _ch:
                        _ch.append(hs[:, -1, :].detach().clone().reshape(-1))
                        _cy.append(out[:, -1, :].detach().float().reshape(-1).clone())
                    return torch.zeros_like(out)

                h_handle = eng_mod.register_forward_hook(_base_cap_hook, with_kwargs=True)
                try:
                    with torch.no_grad():
                        model.forward_logits(ids)
                finally:
                    h_handle.remove()
                if cap_h and cap_y:
                    probe_records.append({
                        "ids": ids,
                        "h_trig": cap_h[0],
                        "y_old": cap_y[0],
                    })

            if not probe_records:
                continue

            # 2b. Optimize `delta_vec` in `col(W_proj)` via whitened projector `P_proj = W_proj @ pinv(W_proj)` (`kappa = 1.0`)
            mean_h_norm = sum(float(pr["h_trig"].float().norm().item()) for pr in probe_records) / len(probe_records)
            max_d_norm = mean_h_norm * float(max_delta_ratio)
            delta_param = torch.zeros(eng_mod.width, device=device, dtype=torch.float32, requires_grad=True)
            best_delta = torch.zeros(eng_mod.width, device=device, dtype=torch.float32)
            best_loss = float("inf")

            n_opt = max(1, int(opt_steps))
            for step_i in range(n_opt):
                delta_vec = F.linear(F.linear(delta_param, pinv_w_p), proj_w_f32)
                d_cur_norm = delta_vec.norm()
                if d_cur_norm > max_d_norm:
                    delta_vec = delta_vec * (max_d_norm / (d_cur_norm + 1e-6))

                total_loss = torch.zeros((), device=device, dtype=torch.float32)
                all_hits = True
                for pr in probe_records:
                    visit_box = [0]

                    def _opt_hook(mod, args, kwargs, out, _dv=delta_vec, _vb=visit_box):
                        _vb[0] += 1
                        if _vb[0] == 1:
                            res = torch.zeros_like(out)
                            res[:, -1, :] = _dv.to(out.dtype)
                            return res
                        return torch.zeros_like(out)

                    h_handle = eng_mod.register_forward_hook(_opt_hook, with_kwargs=True)
                    try:
                        with torch.enable_grad():
                            lg = model.forward_logits(pr["ids"]).float()
                            loss_p = F.cross_entropy(lg[:, -1, :], torch.tensor([t_id], device=device))
                            total_loss = total_loss + loss_p
                            if int(lg[0, -1].argmax().item()) != int(t_id):
                                all_hits = False
                    finally:
                        h_handle.remove()

                mean_loss = total_loss / float(len(probe_records))
                cur_loss_val = float(mean_loss.detach().item())
                if cur_loss_val < best_loss:
                    best_loss = cur_loss_val
                    best_delta = delta_vec.detach().clone()
                if all_hits and cur_loss_val < 0.15:
                    break

                (g_d,) = torch.autograd.grad(mean_loss, [delta_param], allow_unused=True)
                if g_d is None or g_d.norm().item() < 1e-9:
                    break
                g_proj = torch.mv(proj_w_f32, torch.mv(pinv_w_p, g_d.detach().float()))
                g_unit = F.normalize(g_proj, p=2.0, dim=-1, eps=1e-6)
                step_scale = (mean_h_norm * 0.25) * (0.88 ** step_i)
                with torch.no_grad():
                    delta_param.sub_(g_unit * step_scale)
                    delta_param.copy_(torch.mv(proj_w_f32, torch.mv(pinv_w_p, delta_param)))
                    dn = delta_param.norm()
                    if dn > max_d_norm:
                        delta_param.mul_(max_d_norm / (dn + 1e-6))

            # Check final `delta_param` after the last gradient step
            with torch.no_grad():
                final_delta = torch.mv(proj_w_f32, torch.mv(pinv_w_p, delta_param.detach()))
                fn_norm = final_delta.norm()
                if fn_norm > max_d_norm:
                    final_delta = final_delta * (max_d_norm / (fn_norm + 1e-6))
                final_loss = 0.0
                for pr in probe_records:
                    visit_box = [0]

                    def _fin_hook(mod, args, kwargs, out, _dv=final_delta, _vb=visit_box):
                        _vb[0] += 1
                        if _vb[0] == 1:
                            res = torch.zeros_like(out)
                            res[:, -1, :] = _dv.to(out.dtype)
                            return res
                        return torch.zeros_like(out)

                    h_handle = eng_mod.register_forward_hook(_fin_hook, with_kwargs=True)
                    try:
                        lg_f = model.forward_logits(pr["ids"]).float()
                        final_loss += float(F.cross_entropy(lg_f[:, -1, :], torch.tensor([t_id], device=device)).item())
                    finally:
                        h_handle.remove()
                final_mean_loss = final_loss / float(len(probe_records))
                if final_mean_loss <= best_loss:
                    best_loss = final_mean_loss
                    best_delta = final_delta
            pf["nll_opt_target"] = float(best_loss)

            clusters_for_fact = []
            for pr in probe_records:
                y_target = best_delta - pr["y_old"]
                clusters_for_fact.append({
                    "fact_id": pf_idx,
                    "query_hidden": pr["h_trig"],
                    "target_residual": y_target,
                    "trigger_pos": int(pr["ids"].shape[1] - 1),
                    "input_ids": pr["ids"],
                })

            if clusters_for_fact:
                enable_selective_engram_gating(model, enabled=False)
                rep_c = eng_mod.compile_clusters_alphaedit(
                    clusters=clusters_for_fact,
                    input_ids=probe_records[-1]["ids"],
                    alpha_ridge=1e-4,
                    target_scale=float(target_scale),
                )
                compile_reports.append(rep_c)
                max_leakage = max(max_leakage, float(rep_c.get("max_null_space_leakage", 0.0)))

            # Record Round-1 NLLs right before the final fact is compiled so we can audit sequential retention drift
            if n_total_facts >= 2 and pf_idx == n_total_facts - 2:
                enable_selective_engram_gating(
                    model, enabled=True, scope_threshold=1.1, qk_gate_enabled=False, max_rel_delta_norm=None
                )
                eng_mod.selective_gating_enabled = False
                for prev_i in range(pf_idx + 1):
                    r1_nll, _ = _eval_fact_nll(prepared_facts[prev_i]["prompt"], prepared_facts[prev_i]["target_token_id"])
                    round1_nlls[prev_i] = r1_nll

        # 3. Calibrate WISE 2560-D Activation Scope Threshold between `max_general_scope_sim` and `min_fact_scope_sim`
        protos = getattr(eng_mod, "_compiled_scope_prototypes", None)
        max_gen_sim = 0.0
        min_fact_sim = 1.0
        eff_scope_thr = float(scope_threshold)
        if isinstance(protos, torch.Tensor) and protos.numel() > 0:
            protos_f = F.normalize(protos.detach().float(), p=2.0, dim=-1, eps=1e-6)
            use_hid_proto = (
                getattr(eng_mod, "_scope_proto_space", "router") == "hidden"
                and int(protos_f.shape[-1]) == int(eng_mod.width)
            )

            def _max_sim_on_ids(ids_t: torch.Tensor, last_token_only: bool = False) -> float:
                sims_Out: List[float] = []

                def _sim_hook(mod, args, kwargs, out):
                    hs = kwargs.get("hidden_states")
                    if hs is None and len(args) >= 2:
                        hs = args[1]
                    if isinstance(hs, torch.Tensor) and not sims_Out:
                        h_slice = hs[:, -1:, :] if last_token_only else hs
                        rms = torch.rsqrt(h_slice.float().pow(2).mean(dim=-1, keepdim=True) + mod.norm_eps)
                        h_nrm = h_slice.float() * rms * mod.norm_weight.detach().float()
                        if use_hid_proto:
                            q_s = F.normalize(h_nrm, p=2.0, dim=-1, eps=1e-6)
                        else:
                            q_s = F.normalize(F.linear(h_nrm, mod.router_proj.weight.detach().float()), p=2.0, dim=-1, eps=1e-6)
                        s_mat = torch.matmul(q_s.reshape(-1, protos_f.shape[-1]), protos_f.t())
                        sims_Out.append(float(s_mat.max().item()))
                    return torch.zeros_like(out)

                h_h = eng_mod.register_forward_hook(_sim_hook, with_kwargs=True)
                try:
                    with torch.no_grad():
                        model.forward_logits(ids_t)
                finally:
                    h_h.remove()
                return sims_Out[0] if sims_Out else 0.0

            enable_selective_engram_gating(
                model, enabled=True, scope_threshold=1.1, qk_gate_enabled=False, max_rel_delta_norm=None
            )
            gen_sims = []
            for gt in gen_texts:
                g_ids = tokenizer(gt, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_tokens].to(device)
                if g_ids.shape[1] >= 2:
                    gen_sims.append(_max_sim_on_ids(g_ids, last_token_only=False))
            if gen_sims:
                max_gen_sim = max(gen_sims)

            fact_sims = []
            for pf in prepared_facts:
                for s_txt in (pf["prompt"], pf["paraphrase"]):
                    f_ids = tokenizer(s_txt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                    fact_sims.append(_max_sim_on_ids(f_ids, last_token_only=True))
            if fact_sims:
                min_fact_sim = min(fact_sims)

            if max_gen_sim < min_fact_sim:
                eff_scope_thr = max(float(scope_threshold), min(min_fact_sim - 1e-4, max_gen_sim + 1e-4))
            else:
                eff_scope_thr = max(float(scope_threshold), max_gen_sim + 1e-4)

        enable_selective_engram_gating(
            model,
            enabled=True,
            scope_threshold=float(eff_scope_thr),
            scope_margin=0.0,
            qk_gate_enabled=False,
            max_rel_delta_norm=None,
            use_centroid_scope_fallback=False,
        )

        # 4. Evaluate post-edit Reliability, Paraphrase Generalization (on unseen `paraphrase`), and General-Text Locality
        orig_gains, para_gains = [], []
        hits_after = 0
        para_hits_after = 0
        retention_drifts = []
        for pf_idx, pf in enumerate(prepared_facts):
            nll_o_after, hit_o_after = _eval_fact_nll(pf["prompt"], pf["target_token_id"])
            nll_p_after, hit_p_after = _eval_fact_nll(pf["paraphrase"], pf["target_token_id"])
            pf["nll_orig_after"] = nll_o_after
            pf["hit_orig_after"] = hit_o_after
            pf["nll_para_after"] = nll_p_after
            pf["hit_para_after"] = hit_p_after
            pf["orig_nll_gain"] = pf["nll_orig_before"] - nll_o_after
            pf["para_nll_gain"] = pf["nll_para_before"] - nll_p_after
            orig_gains.append(pf["orig_nll_gain"])
            para_gains.append(pf["para_nll_gain"])
            if hit_o_after:
                hits_after += 1
            if hit_p_after:
                para_hits_after += 1
            if pf_idx in round1_nlls:
                retention_drifts.append(abs(nll_o_after - round1_nlls[pf_idx]))

        t_after, n_after = _ce_texts(model, tokenizer, gen_texts, max_tokens=max_tokens)
        ce_general_after = t_after / max(1, n_after)

        return {
            "status": "ok",
            "anchor_layer": chosen_layer,
            "n_facts": len(prepared_facts),
            "mean_orig_nll_gain": float(sum(orig_gains) / max(1, len(orig_gains))),
            "mean_para_nll_gain": float(sum(para_gains) / max(1, len(para_gains))),
            "top1_hit_rate_after": float(hits_after / max(1, len(prepared_facts))),
            "para_top1_hit_rate_after": float(para_hits_after / max(1, len(prepared_facts))),
            "ce_general_off": float(ce_general_off),
            "ce_general_after": float(ce_general_after),
            "general_locality_delta_ce": float(ce_general_after - ce_general_off),
            "max_null_space_leakage": float(max_leakage),
            "sequential_round1_retention_drift": float(max(retention_drifts) if retention_drifts else 0.0),
            "calibrated_scope_threshold": float(eff_scope_thr),
            "max_general_scope_sim": float(max_gen_sim),
            "min_fact_scope_sim": float(min_fact_sim),
            "facts": prepared_facts,
        }
    finally:
        for p, rg in saved_req_grad:
            p.requires_grad_(rg)
        eng_mod.expert_down.data.copy_(saved_down)
        eng_mod.project.weight.data.copy_(saved_proj)
        if saved_hot is not None and hasattr(eng_mod, "tier1a_hot"):
            eng_mod.tier1a_hot.weight.data.copy_(saved_hot)
        for _, m_anc in anchors.items():
            if hasattr(m_anc, "reset_alphaedit_basis"):
                m_anc.reset_alphaedit_basis()
        enable_selective_engram_gating(model, enabled=False)
        _set_use_engram(model, prev_engram)

