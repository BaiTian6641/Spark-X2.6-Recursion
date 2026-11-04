"""sparkx_si.select — frontier selection, dedup, pillar cap, diversity (docs/roadmap/05 §3C/D).

A *record* is one verified rollout: {"task_id", "pillar", "prompt", "response", "ok": bool, "n_tokens": int}.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def pass_rates(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[r["task_id"]].append(r)
    return {tid: {"k": len(rs), "c": sum(1 for r in rs if r["ok"]), "p": sum(1 for r in rs if r["ok"]) / len(rs),
                  "pillar": rs[0]["pillar"], "prompt": rs[0]["prompt"]} for tid, rs in by.items()}


def ngrams(text: str, n: int = 4) -> set:
    w = text.split()
    return {tuple(w[i: i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def distinct_n(texts: Sequence[str], n: int = 4) -> float:
    """Distinct-n ratio: unique n-grams / total n-grams over the set (1.0 = no repetition at all)."""
    tot, uniq = 0, set()
    for t in texts:
        w = t.split()
        grams = [tuple(w[i: i + n]) for i in range(max(0, len(w) - n + 1))]
        tot += len(grams)
        uniq.update(grams)
    return len(uniq) / tot if tot else 1.0


def diversity_ok(selected_texts: Sequence[str], raw_texts: Sequence[str], ratio: float = 0.9, n: int = 4) -> Tuple[bool, float, float]:
    ds, dr = distinct_n(selected_texts, n), distinct_n(raw_texts, n)
    return ds >= ratio * dr, ds, dr


def dedup_responses(items: List[Dict[str, Any]], thresh: float = 0.8, n: int = 4) -> List[Dict[str, Any]]:
    kept, grams = [], []
    for it in items:
        g = ngrams(it["response"], n)
        if any(jaccard(g, h) >= thresh for h in grams):
            continue
        kept.append(it)
        grams.append(g)
    return kept


def apply_pillar_cap(selected: List[Dict[str, Any]], cap: float = 0.35, seed: int = 0) -> List[Dict[str, Any]]:
    """Trim the largest pillars until no pillar exceeds `cap` of the total (relaxed to 1/n_pillars when infeasible)."""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in selected:
        by[s["pillar"]].append(s)
    if not by:
        return []
    cap_eff = max(cap, 1.0 / len(by))
    rng = random.Random(seed)
    for lst in by.values():
        rng.shuffle(lst)
    changed = True
    while changed:
        changed = False
        total = sum(len(v) for v in by.values())
        for p, lst in by.items():
            allowed = max(1, int(cap_eff * total + 1e-9))
            if len(lst) > allowed:
                by[p] = lst[:allowed]
                changed = True
    return [x for v in by.values() for x in v]


def select_round(records: Sequence[Dict[str, Any]], per_prompt_keep: int = 2, solved_cap: float = 0.10, pillar_cap: float = 0.35,
                 dedup_thresh: float = 0.8, seed: int = 0) -> Dict[str, Any]:
    """Frontier selection: prompts with 0 < p < 1 supply up to `per_prompt_keep` correct samples (shortest first); fully solved prompts
    contribute at most `solved_cap` of the selected set (replay of solved behaviour); p = 0 prompts are returned for rationalisation."""
    rates = pass_rates(records)
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[r["task_id"]].append(r)
    frontier, solved, hard = [], [], []
    for tid, rs in by.items():
        ok = sorted((r for r in rs if r["ok"]), key=lambda r: (r.get("n_tokens", len(r["response"])), r["response"]))
        p = rates[tid]["p"]
        pick = [dict(r, tag="frontier" if 0 < p < 1 else "solved", p_hat=p) for r in ok[:per_prompt_keep]]
        if 0 < p < 1:
            frontier += pick
        elif p == 1.0:
            solved += pick
        else:
            hard.append(tid)
    frontier = dedup_responses(frontier, dedup_thresh)
    rng = random.Random(seed)
    rng.shuffle(solved)
    max_solved = int(solved_cap / max(1e-9, 1 - solved_cap) * len(frontier)) if frontier else 0
    solved = dedup_responses(solved[:max_solved], dedup_thresh) if max_solved > 0 else []
    chosen = apply_pillar_cap(frontier + solved, pillar_cap, seed)
    raw_ok = [r["response"] for r in records if r["ok"]]
    div_ok, ds, dr = diversity_ok([c["response"] for c in chosen], raw_ok)
    return {"selected": chosen, "needs_rationalization": hard,
            "stats": {"n_prompts": len(by), "n_frontier_prompts": sum(1 for v in rates.values() if 0 < v["p"] < 1),
                      "n_solved_prompts": sum(1 for v in rates.values() if v["p"] == 1.0),
                      "n_hard_prompts": len(hard), "n_selected": len(chosen),
                      "by_pillar": {p: sum(1 for c in chosen if c["pillar"] == p) for p in {c["pillar"] for c in chosen}},
                      "distinct4_selected": ds, "distinct4_raw_correct": dr, "diversity_ok": div_ok}}
