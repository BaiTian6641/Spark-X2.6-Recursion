"""sparkx_si.gate — paired-bootstrap acceptance gate and round controller (docs/roadmap/05 §6).

Inputs are per-prompt 0/1 (or real-valued) outcomes of candidate and incumbent on the SAME dev prompts, in the same order. The primary
metric is the mean of per-pillar pass@1 (stratified bootstrap: resample prompts within each pillar, average the pillars). The test
split is never an input here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # numpy only speeds things up
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


@dataclass
class GateConfig:
    min_gain_pts: float = 1.0          # primary: point estimate >= +1.0 pt and CI lower bound >= 0
    pillar_guard_pts: float = -2.0     # each pillar: delta >= -2.0 pts
    qa_guard_pts: float = -1.0         # closed-book QA dev: delta >= -1.0 pt
    ce_on_guard: float = 0.01          # held-out CE(ON): delta <= +0.01 nats
    engram_gap_guard: float = 0.01     # (ce_on - ce_off) must not increase by more than this
    diversity_ratio: float = 0.9       # distinct-4 >= 0.9 x incumbent
    style_slop_guard: float = 0.02     # slop-phrase rate may not rise by more than this (absolute)
    n_boot: int = 10000
    alpha: float = 0.05
    seed: int = 0


def _boot_means(deltas_by_pillar: Dict[str, Sequence[float]], n_boot: int, seed: int) -> List[float]:
    """Stratified paired bootstrap of the mean-over-pillars of mean paired difference."""
    pillars = [p for p, d in deltas_by_pillar.items() if len(d) > 0]
    if _np is not None:
        rng = _np.random.default_rng(seed)
        acc = _np.zeros(n_boot)
        for p in pillars:
            d = _np.asarray(deltas_by_pillar[p], dtype=float)
            idx = rng.integers(0, len(d), size=(n_boot, len(d)))
            acc += d[idx].mean(axis=1)
        return list(acc / max(1, len(pillars)))
    rng = random.Random(seed)  # pragma: no cover
    out = []
    for _ in range(n_boot):
        tot = 0.0
        for p in pillars:
            d = deltas_by_pillar[p]
            tot += sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d)
        out.append(tot / max(1, len(pillars)))
    return out


def paired_bootstrap(cand: Dict[str, Sequence[float]], inc: Dict[str, Sequence[float]], n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 0) -> Dict[str, Any]:
    """`cand[pillar]`, `inc[pillar]`: aligned per-prompt outcomes. Returns point estimate and CI of the mean-over-pillars paired
    difference (in the metric's unit, e.g. fraction -> multiply by 100 for points) plus per-pillar point estimates."""
    deltas: Dict[str, List[float]] = {}
    for p in cand:
        if len(cand[p]) != len(inc.get(p, [])):
            raise ValueError(f"pillar {p}: candidate/incumbent lengths differ ({len(cand[p])} vs {len(inc.get(p, []))})")
        deltas[p] = [float(a) - float(b) for a, b in zip(cand[p], inc[p])]
    per = {p: (sum(d) / len(d) if d else 0.0) for p, d in deltas.items()}
    point = sum(per.values()) / max(1, len(per))
    bs = sorted(_boot_means(deltas, n_boot, seed))
    lo = bs[int((alpha / 2) * len(bs))]
    hi = bs[min(len(bs) - 1, int((1 - alpha / 2) * len(bs)))]
    return {"point": point, "lo": lo, "hi": hi, "per_pillar": per, "n": {p: len(d) for p, d in deltas.items()}}


def decide(cand: Dict[str, Sequence[float]], inc: Dict[str, Sequence[float]], guards: Optional[Dict[str, Any]] = None,
           cfg: Optional[GateConfig] = None) -> Dict[str, Any]:
    """Gate decision. `cand`/`inc`: primary pillars (math, code, if, tool ... as 0/1 per prompt). `guards` (all optional):
      qa: {"cand": [...], "inc": [...]}; ce_on: {"cand": float, "inc": float}; engram_gap: {"cand": float, "inc": float}
      (gap = ce_on - ce_off); distinct4: {"cand": float, "inc": float}; slop_rate: {"cand": float, "inc": float}.
    Returns {"accept": bool, "reasons": [...], "primary": {...}, "checks": {...}}."""
    cfg = cfg or GateConfig()
    guards = guards or {}
    reasons: List[str] = []
    checks: Dict[str, Any] = {}
    prim = paired_bootstrap(cand, inc, cfg.n_boot, cfg.alpha, cfg.seed)
    pts = prim["point"] * 100
    lo_pts = prim["lo"] * 100
    checks["primary"] = {"point_pts": pts, "ci_lo_pts": lo_pts, "ci_hi_pts": prim["hi"] * 100}
    if not (lo_pts >= 0 and pts >= cfg.min_gain_pts):
        reasons.append(f"primary: point {pts:+.2f} pts (need >= {cfg.min_gain_pts:+.1f}) with CI lower bound {lo_pts:+.2f} (need >= 0)")
    for p, d in prim["per_pillar"].items():
        checks.setdefault("pillars", {})[p] = d * 100
        if d * 100 < cfg.pillar_guard_pts:
            reasons.append(f"pillar {p}: {d * 100:+.2f} pts < {cfg.pillar_guard_pts:+.1f}")
    if "qa" in guards:
        q = guards["qa"]
        dq = (sum(q["cand"]) / len(q["cand"]) - sum(q["inc"]) / len(q["inc"])) * 100
        checks["qa_pts"] = dq
        if dq < cfg.qa_guard_pts:
            reasons.append(f"qa: {dq:+.2f} pts < {cfg.qa_guard_pts:+.1f}")
    if "ce_on" in guards:
        d = guards["ce_on"]["cand"] - guards["ce_on"]["inc"]
        checks["ce_on_delta"] = d
        if d > cfg.ce_on_guard:
            reasons.append(f"held-out CE(ON): {d:+.4f} > {cfg.ce_on_guard:+.3f}")
    if "engram_gap" in guards:
        d = guards["engram_gap"]["cand"] - guards["engram_gap"]["inc"]
        checks["engram_gap_delta"] = d
        if d > cfg.engram_gap_guard:
            reasons.append(f"engram gap (ce_on - ce_off): {d:+.4f} > {cfg.engram_gap_guard:+.3f}")
    if "distinct4" in guards:
        r = guards["distinct4"]["cand"] / max(1e-9, guards["distinct4"]["inc"])
        checks["distinct4_ratio"] = r
        if r < cfg.diversity_ratio:
            reasons.append(f"diversity: distinct-4 ratio {r:.3f} < {cfg.diversity_ratio}")
    if "slop_rate" in guards:
        d = guards["slop_rate"]["cand"] - guards["slop_rate"]["inc"]
        checks["slop_rate_delta"] = d
        if d > cfg.style_slop_guard:
            reasons.append(f"style: slop rate {d:+.3f} > {cfg.style_slop_guard:+.3f}")
    return {"accept": not reasons, "reasons": reasons, "primary": prim, "checks": checks}


@dataclass
class RoundController:
    """Applies the reject policy: restore snapshot, lr x0.5, halve selected-set size; two consecutive rejects => stop."""
    lr: float
    n_select: int
    consecutive_rejects: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    stopped: bool = False

    def update(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if decision["accept"]:
            self.consecutive_rejects = 0
            action = "snapshot"
        else:
            self.consecutive_rejects += 1
            self.lr *= 0.5
            self.n_select = max(1, self.n_select // 2)
            action = "rollback"
            if self.consecutive_rejects >= 2:
                self.stopped = True
                action = "rollback_and_stop"
        rec = {"action": action, "lr": self.lr, "n_select": self.n_select, "reasons": decision["reasons"]}
        self.history.append(rec)
        return rec
