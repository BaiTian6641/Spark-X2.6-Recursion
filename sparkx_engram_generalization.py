"""MoE-Engram generalization diagnostics (leak-free).

Why this module exists
----------------------
The existing "held-out" PPL (`_HELD_OUT_EVAL_TEXTS` in `sparkx_loop_v03_benchmarks.py`) is *also* used as training
data (`run_phase4_engram_consolidation` calibration anchors and `run_si_srsi_engram_loop` general items), and the
SI-SRSI rubric is scored on the very 5 prompts it trains on. Neither measures generalization.

This module provides:
  1. `TRUE_HELDOUT_TEXTS`        - texts that appear in no training path (PPL / top-1 sanity).
  2. `HELDOUT_PARAPHRASES`       - reworded prompts for every SI-SRSI task, never used for training.
  3. `engram_ablation(...)`      - context manager scaling the Tier-1 lexical path and/or the semantic
                                   (shared + routed experts) path via the `_ablate_lex` / `_ablate_sem` hooks
                                   in `ZipfianTieredMoEEngram.forward` (default 1.0 = no-op).
  4. `diagnose_engram_generalization` - teacher-forced target NLL under {OFF, FULL, SEM_ONLY, LEX_ONLY} for the
                                   original prompt and each paraphrase; reports gain transfer ratios.

Pure-Python helpers (paraphrase tables, overlap metric, summarizer) import without torch so they are unit-testable.
"""
from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------------------
# 1. Leak-free held-out texts (same chat format as the training anchors, disjoint content)
# --------------------------------------------------------------------------------------
_SYS = "You are Spark-X2.6, an advanced reasoning and engineering assistant."
_BOS = "<｜start▁of▁sentence｜>"


def _chat(user: str, think: str, answer: str) -> str:
    return (
        f"{_BOS}<|System|>{_SYS}<|User|>{user}"
        f"{_BOS}<|Bot|><think>\n{think}\n</think>\n{answer}"
    )


TRUE_HELDOUT_TEXTS: List[str] = [
    _chat(
        "Implement an LRU cache class in Python with O(1) get and put using a dict and a doubly linked list.",
        "1. Keep sentinel head/tail nodes.\n2. On access, unlink the node and reinsert it after head.\n3. On overflow, drop tail.prev.",
        "```python\nclass Node:\n    __slots__ = ('k', 'v', 'prev', 'next')\n    def __init__(self, k=0, v=0):\n        self.k, self.v, self.prev, self.next = k, v, None, None\n\n"
        "class LRUCache:\n    def __init__(self, capacity: int):\n        self.cap, self.map = capacity, {}\n        self.head, self.tail = Node(), Node()\n        self.head.next, self.tail.prev = self.tail, self.head\n\n"
        "    def _unlink(self, n):\n        n.prev.next, n.next.prev = n.next, n.prev\n\n"
        "    def _push_front(self, n):\n        n.next, n.prev = self.head.next, self.head\n        self.head.next.prev = n\n        self.head.next = n\n\n"
        "    def get(self, k):\n        n = self.map.get(k)\n        if n is None:\n            return -1\n        self._unlink(n)\n        self._push_front(n)\n        return n.v\n\n"
        "    def put(self, k, v):\n        n = self.map.get(k)\n        if n:\n            n.v = v\n            self._unlink(n)\n        else:\n            n = self.map[k] = Node(k, v)\n            if len(self.map) > self.cap:\n                lru = self.tail.prev\n                self._unlink(lru)\n                del self.map[lru.k]\n        self._push_front(n)\n```",
    ),
    _chat(
        "Write a PostgreSQL query returning, for each customer, their three most recent orders with a running total of amount.",
        "1. Rank orders per customer with row_number() over created_at desc.\n2. Filter rank <= 3.\n3. Running sum via a window frame.",
        "```sql\nWITH ranked AS (\n  SELECT o.*, row_number() OVER (PARTITION BY customer_id ORDER BY created_at DESC) AS rn\n  FROM orders o\n)\nSELECT customer_id, id, created_at, amount,\n       SUM(amount) OVER (PARTITION BY customer_id ORDER BY created_at DESC\n                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total\nFROM ranked\nWHERE rn <= 3\nORDER BY customer_id, created_at DESC;\n```",
    ),
    _chat(
        "A damped harmonic oscillator has m = 2 kg, k = 50 N/m and damping coefficient c = 8 kg/s. Is it underdamped, and what is the damped angular frequency?",
        "1. Critical damping c_c = 2*sqrt(k*m) = 20 kg/s, so c < c_c.\n2. omega_0 = sqrt(k/m) = 5, gamma = c/(2m) = 2.\n3. omega_d = sqrt(omega_0^2 - gamma^2) = sqrt(21).",
        "Because $c = 8 < c_c = 2\\sqrt{km} = 20\\,\\mathrm{kg/s}$, the system is underdamped. With $\\omega_0 = \\sqrt{k/m} = 5\\,\\mathrm{rad/s}$ and $\\gamma = c/(2m) = 2\\,\\mathrm{s^{-1}}$, "
        "the damped angular frequency is $\\omega_d = \\sqrt{\\omega_0^2 - \\gamma^2} = \\sqrt{21} \\approx 4.58\\,\\mathrm{rad/s}$.",
    ),
    _chat(
        "Write a CSS rule set for a responsive card grid that uses container queries and respects reduced motion.",
        "1. Make the wrapper a size container.\n2. Switch columns at container widths.\n3. Gate hover transitions behind prefers-reduced-motion.",
        "```css\n.cards { container-type: inline-size; }\n.cards__grid { display: grid; gap: 1rem; grid-template-columns: 1fr; }\n"
        "@container (min-width: 36rem) { .cards__grid { grid-template-columns: repeat(2, 1fr); } }\n"
        "@container (min-width: 60rem) { .cards__grid { grid-template-columns: repeat(3, 1fr); } }\n"
        ".card { transition: transform .2s ease; }\n@media (prefers-reduced-motion: no-preference) { .card:hover { transform: translateY(-2px); } }\n```",
    ),
    _chat(
        "In two short paragraphs, explain why a database migration that adds a NOT NULL column with a default locked our table for four minutes.",
        "1. Name the mechanism: full table rewrite under an ACCESS EXCLUSIVE lock on the older engine version.\n2. Give the safer sequence.",
        "On the 9.6 replica set the `ALTER TABLE ... ADD COLUMN ... NOT NULL DEFAULT 'x'` statement rewrote every row while holding an exclusive lock, so reads and writes queued behind it until the rewrite finished. "
        "With 180 million rows that took just over four minutes.\n\n"
        "The fix is to add the column as nullable, backfill in batches of ten thousand rows, then add the constraint with `NOT VALID` and validate it separately. "
        "Newer engine versions skip the rewrite for constant defaults, but we should not depend on that when the fleet is mixed.",
    ),
]


# --------------------------------------------------------------------------------------
# 2. Held-out paraphrases for every SI-SRSI task (never used for training)
# --------------------------------------------------------------------------------------
HELDOUT_PARAPHRASES: Dict[str, List[str]] = {
    "stem_1_graphene_landau": [
        "Graphene's low-energy electrons behave as massless Dirac particles. Put the sheet in a uniform perpendicular magnetic field "
        "and find the allowed Landau level energies E_n. What is E_2/E_1? Put the ratio in \\boxed{}.",
        "Find the Landau quantization of charge carriers in a single carbon sheet (linear dispersion, Fermi velocity v_F) under a transverse "
        "field B. Give E_n for n >= 0 and the ratio of the n=2 level to the n=1 level in \\boxed{}.",
        "Using ladder operators for the kinetic momenta in a magnetic field, work out the energy levels of Dirac-Weyl fermions in monolayer "
        "graphene and report E_2 divided by E_1 as \\boxed{}.",
    ],
    "stem_2_partition_entropy": [
        "N independent two-level atoms have energy levels 0 and epsilon (epsilon positive) and sit in a heat bath at temperature T. "
        "Compute the exact heat capacity per particle in units of k_B when epsilon/(k_B T) equals 2. Give the answer using e in \\boxed{}.",
        "For a system of N distinguishable particles that each have just two states, 0 and epsilon, in thermal equilibrium, find C_V/(N k_B) "
        "at beta*epsilon = 2 in closed form with e, boxed.",
        "Derive the Schottky anomaly heat capacity of a two-level ensemble and evaluate it at epsilon over k_B T equal to two; "
        "express it as a function of e inside \\boxed{}.",
    ],
    "web_ux_sec_1_apikey_portal": [
        "Build a secure API-key rotation feature: an accessible front-end widget (semantic main landmark, polite live region, visible keyboard "
        "focus, reduced-motion support) that shows loading, empty, error-with-retry and success states without innerHTML, plus a FastAPI + "
        "SQLite endpoint that validates input with Pydantic, uses parameterized queries, checks a CSRF header, sets strict HttpOnly Secure "
        "cookies, rate-limits callers and sends a CSP header.",
        "I need a rotate-API-key screen and its server route. The UI must meet WCAG 2.2 and cover every async state (loading skeleton, empty, "
        "retryable error, success) with safe text rendering. Server side, harden it: schema validation, SQL placeholders, CSRF token, "
        "SameSite=Strict cookies, throttling, CSP.",
    ],
    "anti_slop_1_postmortem": [
        "Draft a two-paragraph incident review: our Redis-backed rate limiter let traffic through during a 40k requests/sec surge because "
        "read-then-increment wasn't atomic and every client retried at the same fixed delay. Describe the single Lua token-bucket fix. "
        "Plain, direct engineer voice; avoid stock AI phrases.",
        "Write up, in two paragraphs, what went wrong with our rate limiting under a 40,000 RPS burst (non-atomic check-and-increment plus "
        "synchronized reconnect backoff) and how an atomic Lua script resolved it. Keep the tone blunt and human, no corporate filler.",
    ],
    "anti_slop_2_arch_memo": [
        "Give me a short design note on sparse expert memory layers versus simply making dense models bigger, from the viewpoint of "
        "single-GPU and edge deployment. Terse, varied sentence rhythm, no flattery, no buzzwords.",
        "Compare, in a brief memo, scaling dense parameters against adding sparse Mixture-of-Experts memory for on-device inference. "
        "Direct engineer prose; skip the opening pleasantries and jargon like 'paradigm' or 'landscape'.",
    ],
}

CONDITIONS = {
    "OFF": dict(use_engram=False, lex=1.0, sem=1.0),
    "FULL": dict(use_engram=True, lex=1.0, sem=1.0),
    # No lexical hit: what a paraphrase looks like to Tier-1 (unseen buckets are ~zero-initialised rows).
    "SEM_ONLY": dict(use_engram=True, lex=0.0, sem=1.0),
    # Exact-wording memory only: shared + routed experts removed.
    "LEX_ONLY": dict(use_engram=True, lex=1.0, sem=0.0),
}


# --------------------------------------------------------------------------------------
# Pure-Python helpers
# --------------------------------------------------------------------------------------
def word_trigram_overlap(a: str, b: str) -> float:
    """Jaccard overlap of lowercase word trigrams (coarse proxy for Tier-1 lexical-hash overlap)."""
    def grams(s: str) -> set:
        w = re.findall(r"[a-z0-9_]+", s.lower())
        return {tuple(w[i:i + 3]) for i in range(max(0, len(w) - 2))}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def summarize_transfer(per_task: Dict[str, Dict[str, Any]], min_gain: float = 0.02) -> Dict[str, Any]:
    """Aggregate per-task NLL tables into gain-transfer ratios.

    `per_task[tid] = {"orig": {cond: nll}, "para": [{cond: nll, "overlap": float}, ...]}`.
    gain(cond, prompt) = NLL_OFF(prompt) - NLL_cond(prompt)   (positive = Engram helps).
    transfer(cond)     = mean_para_gain / orig_gain           (only when orig_gain > `min_gain`).
    """
    rows: Dict[str, Any] = {}
    agg: Dict[str, List[float]] = {c: [] for c in ("FULL", "SEM_ONLY", "LEX_ONLY")}
    for tid, d in per_task.items():
        o = d["orig"]
        r: Dict[str, Any] = {"orig_nll": o}
        for cond in agg:
            g_orig = o["OFF"] - o[cond]
            g_para = [p["OFF"] - p[cond] for p in d["para"]]
            mean_g_para = sum(g_para) / max(1, len(g_para))
            tr = (mean_g_para / g_orig) if g_orig > min_gain else None
            r[cond] = {"gain_orig": g_orig, "gain_para_mean": mean_g_para, "transfer": tr}
            if tr is not None:
                agg[cond].append(tr)
        r["mean_overlap"] = sum(p.get("overlap", 0.0) for p in d["para"]) / max(1, len(d["para"]))
        rows[tid] = r
    mean_tr = {c: (sum(v) / len(v) if v else None) for c, v in agg.items()}
    return {"tasks": rows, "mean_transfer": mean_tr, "n_tasks_scored": {c: len(v) for c, v in agg.items()}}


def interpret(mean_transfer: Dict[str, Optional[float]]) -> str:
    """Heuristic reading of the aggregate. Thresholds are rules of thumb, not calibrated statistics."""
    full = mean_transfer.get("FULL")
    if full is None:
        return "INCONCLUSIVE: Engram gives no measurable NLL gain on the original prompts (nothing to transfer)."
    if full < 0.30:
        return ("LEXICAL-MEMORIZATION DOMINATED: <30% of the original-prompt gain survives rewording. "
                "Priority: paraphrase-augmented consolidation + token-level lexical dropout.")
    if full < 0.70:
        return "PARTIAL TRANSFER: some semantic carry-over; augmentation should help but the semantic path is not the bottleneck alone."
    return ("GOOD TRANSFER: gains survive rewording. If generation still fails on paraphrases, suspect decoding/prompt-format "
            "effects rather than the Engram.")


# --------------------------------------------------------------------------------------
# Torch-dependent pieces (lazy imports)
# --------------------------------------------------------------------------------------
@contextmanager
def engram_ablation(model: Any, lex: float = 1.0, sem: float = 1.0):
    """Temporarily scale the Tier-1 lexical path (`lex`) and/or shared+routed semantic path (`sem`) on every Engram anchor."""
    mods = list(model.all_engram_modules())
    prev = [(getattr(m, "_ablate_lex", 1.0), getattr(m, "_ablate_sem", 1.0)) for m in mods]
    try:
        for m in mods:
            m._ablate_lex = float(lex)
            m._ablate_sem = float(sem)
            m._eval_lex_cache = None
        yield
    finally:
        for m, (pl, ps) in zip(mods, prev):
            m._ablate_lex = pl
            m._ablate_sem = ps
            m._eval_lex_cache = None


def _target_nll(model: Any, tokenizer: Any, prompt: str, target: str, system_text: str, max_seq_len: int = 2048) -> Optional[float]:
    import torch
    import torch.nn.functional as F
    import sparkx_loop_v03_benchmarks as bench

    pair = bench.encode_spark_chat_pair(tokenizer, prompt, target, system_text=system_text, max_seq_len=max_seq_len)
    if pair is None:
        return None
    inp, lbl = pair
    device = next(model.base.parameters()).device
    inp, lbl = inp.to(device), lbl.to(device)
    with torch.no_grad():
        logits = model.forward_logits(inp).float()
    sl = logits[:, :-1, :].contiguous()
    tl = lbl[:, 1:].contiguous()
    return float(F.cross_entropy(sl.view(-1, sl.size(-1)), tl.view(-1), ignore_index=-100, reduction="mean").item())


def diagnose_engram_generalization(
    model: Any,
    tokenizer: Any,
    flight_mode: str = "deep_knowledge",
    use_domain_focus: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the 4-condition ablation over every SI-SRSI task: original prompt vs. held-out paraphrases.

    Teacher-forced NLL of the task's `teacher_instructed_target` (the text the Engram was trained to emit). Using NLL instead
    of sampled-generation rubric scores removes sampling noise (the rubric run is n=1 per prompt).
    """
    import sparkx_loop_v03_benchmarks as bench

    tasks = {t["id"]: t for t in bench._build_si_srsi_curriculum_tasks()}
    sys_msg = (
        "You are Spark-X2.6, an advanced reasoning and engineering assistant. "
        "Reason step-by-step in your thinking channel, then in your response channel deliver human-natural, secure, and mathematically exact outputs."
    )
    was_training = model.training
    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    model.set_engram_flight_mode(flight_mode)

    per_task: Dict[str, Dict[str, Any]] = {}
    try:
        for tid, paras in HELDOUT_PARAPHRASES.items():
            if tid not in tasks:
                continue
            t = tasks[tid]
            target = t["teacher_instructed_target"]
            if use_domain_focus:
                model.set_domain_engram_focus(t["domain"], boost=0.08)
            prompts = [("orig", t["prompt"])] + [(f"para{i + 1}", p) for i, p in enumerate(paras)]
            entry: Dict[str, Any] = {"orig": {}, "para": []}
            for name, prompt in prompts:
                row: Dict[str, float] = {}
                for cname, cfg in CONDITIONS.items():
                    model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=cfg["use_engram"],
                                   enable_p3=True, p3_start=12, p3_end=24)
                    with engram_ablation(model, lex=cfg["lex"], sem=cfg["sem"]):
                        nll = _target_nll(model, tokenizer, prompt, target, sys_msg)
                    if nll is None:
                        row = {}
                        break
                    row[cname] = nll
                if not row:
                    continue
                if name == "orig":
                    entry["orig"] = row
                else:
                    row["overlap"] = word_trigram_overlap(t["prompt"], prompt)
                    entry["para"].append(row)
            if entry["orig"] and entry["para"]:
                per_task[tid] = entry
    finally:
        model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
        model.set_domain_engram_focus("general", boost=0.0)
        model.enable_checkpoint = prev_ckpt
        model.settings.checkpoint_layers = prev_ckpt
        if was_training:
            model.train()
            model.base.eval()

    summary = summarize_transfer(per_task)
    summary["verdict"] = interpret(summary["mean_transfer"])

    # Leak-free sanity PPL (texts present in no training path)
    heldout = bench.evaluate_engram_perplexity_delta(model, tokenizer, eval_texts=TRUE_HELDOUT_TEXTS)
    summary["true_heldout"] = heldout
    leaked = bench.evaluate_engram_perplexity_delta(model, tokenizer)
    summary["leaked_anchor_texts"] = leaked

    if verbose:
        print("\n" + "=" * 118)
        print("[ENGRAM GENERALIZATION DIAGNOSTIC] teacher-forced target NLL; gain = NLL_OFF - NLL_cond (positive = Engram helps)")
        print("=" * 118)
        hdr = f"{'task':<28} {'cond':<9} {'gain(orig)':>10} {'gain(para)':>11} {'transfer':>9}   mean-trigram-overlap"
        print(hdr)
        print("-" * len(hdr))
        for tid, r in summary["tasks"].items():
            for cond in ("FULL", "SEM_ONLY", "LEX_ONLY"):
                c = r[cond]
                tr = "n/a" if c["transfer"] is None else f"{c['transfer']:.2f}"
                print(f"{tid:<28} {cond:<9} {c['gain_orig']:>10.4f} {c['gain_para_mean']:>11.4f} {tr:>9}   {r['mean_overlap']:.2f}")
        print("-" * len(hdr))
        print("mean transfer:", {k: (None if v is None else round(v, 3)) for k, v in summary["mean_transfer"].items()},
              "| tasks scored:", summary["n_tasks_scored"])
        print("VERDICT:", summary["verdict"])
        print(f"True held-out  : no-Engram PPL={heldout['no_engram_ppl']:.4f}  with-Engram PPL={heldout['with_engram_ppl']:.4f}  "
              f"(delta={heldout['delta_ppl']:+.4f}, acc {heldout['no_engram_acc']:.2f}% -> {heldout['with_engram_acc']:.2f}%)")
        print(f"Leaked anchors : no-Engram PPL={leaked['no_engram_ppl']:.4f}  with-Engram PPL={leaked['with_engram_ppl']:.4f}  "
              f"(these texts were TRAINED ON in Phase-4 and SI-SRSI)")
        print("=" * 118)
    return summary
