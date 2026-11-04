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

    `per_task[tid] = {"orig": {cond: nll}, "para": [{cond: nll, "overlap": float}, ...], "ctrl": {cond: nll}?}`.
    gain(cond, prompt) = NLL_OFF(prompt) - NLL_cond(prompt)   (positive = Engram helps).
    transfer(cond)     = mean_para_gain / orig_gain           (only when orig_gain > `min_gain`).
    control_ratio      = gain(unrelated prompt, same target) / gain(orig)  -> ~1 means the gain does NOT depend on the prompt.

    NOTE: teacher forcing feeds the gold prefix, so a gain can come from recalling the response text from its own earlier
    tokens. Transfer > 1 or a control ratio near 1 are therefore signs of prompt-independent response recall, NOT of good
    prompt-conditioned generalization.
    """
    rows: Dict[str, Any] = {}
    agg: Dict[str, List[float]] = {c: [] for c in ("FULL", "SEM_ONLY", "LEX_ONLY")}
    orig_gains: List[float] = []
    ctrl_ratios: List[float] = []
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
        orig_gains.append(o["OFF"] - o["FULL"])
        c = d.get("ctrl")
        if c:
            g_ctrl = c["OFF"] - c["FULL"]
            g_o = o["OFF"] - o["FULL"]
            r["control"] = {"gain_ctrl": g_ctrl, "ratio_vs_orig": (g_ctrl / g_o) if g_o > min_gain else None}
            if g_o > min_gain:
                ctrl_ratios.append(g_ctrl / g_o)
        r["mean_overlap"] = sum(p.get("overlap", 0.0) for p in d["para"]) / max(1, len(d["para"]))
        rows[tid] = r
    mean_tr = {c: (sum(v) / len(v) if v else None) for c, v in agg.items()}
    all_tr = [x for v in agg.values() for x in v]
    return {
        "tasks": rows,
        "mean_transfer": mean_tr,
        "max_transfer": max(all_tr) if all_tr else None,
        "mean_orig_gain": sum(orig_gains) / max(1, len(orig_gains)),
        "mean_control_ratio": (sum(ctrl_ratios) / len(ctrl_ratios)) if ctrl_ratios else None,
        "n_tasks_scored": {c: len(v) for c, v in agg.items()},
    }


def interpret(
    mean_transfer: Dict[str, Optional[float]],
    max_transfer: Optional[float] = None,
    mean_orig_gain: Optional[float] = None,
    mean_control_ratio: Optional[float] = None,
) -> str:
    """Heuristic reading of the aggregate. Thresholds are rules of thumb, not calibrated statistics."""
    full = mean_transfer.get("FULL")
    if full is None:
        return "INCONCLUSIVE: Engram gives no measurable NLL gain on the original prompts (nothing to transfer)."
    notes: List[str] = []
    if (max_transfer is not None and max_transfer > 1.5) or full > 1.5:
        notes.append("ANOMALOUS: gain is LARGER on reworded prompts than on the trained prompt (transfer > 1.5), so it is not "
                     "tied to the trained prompt; likely prompt-independent recall of the response text from its own prefix.")
    if mean_control_ratio is not None and mean_control_ratio > 0.7:
        notes.append(f"CONTROL: an unrelated prompt keeps {mean_control_ratio:.0%} of the gain -> the gain does not depend on the prompt.")
    if mean_orig_gain is not None and mean_orig_gain < 0.05:
        notes.append(f"WEAK: mean marginal gain on trained prompts is only {mean_orig_gain:.3f} nats.")
    if notes:
        return " | ".join(notes)
    if full < 0.30:
        return ("LEXICAL-MEMORIZATION DOMINATED: <30% of the original-prompt gain survives rewording. "
                "Priority: paraphrase-augmented consolidation + token-level lexical dropout.")
    if full < 0.70:
        return "PARTIAL TRANSFER: some semantic carry-over; augmentation should help but the semantic path is not the bottleneck alone."
    return "TRANSFER: gains survive rewording with a prompt-dependent signature."


# --------------------------------------------------------------------------------------
# Torch-dependent pieces (lazy imports)
# --------------------------------------------------------------------------------------
@contextmanager
def engram_ablation(model: Any, lex: float = 1.0, sem: float = 1.0, modules: Optional[Sequence[Any]] = None):
    """Temporarily scale the Tier-1 lexical path (`lex`) and/or shared+routed semantic path (`sem`).

    Applies to every Engram anchor unless `modules` (a subset of `model.all_engram_modules()`) is given.
    """
    mods = list(modules) if modules is not None else list(model.all_engram_modules())
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
            other = [x for k, x in tasks.items() if x["domain"] != t["domain"]]
            ctrl_prompt = other[0]["prompt"] if other else None
            prompts = [("orig", t["prompt"])] + [(f"para{i + 1}", p) for i, p in enumerate(paras)]
            if ctrl_prompt:
                prompts.append(("ctrl", ctrl_prompt))
            entry: Dict[str, Any] = {"orig": {}, "para": []}
            for name, prompt in prompts:
                row: Dict[str, float] = {}
                for cname, cfg in CONDITIONS.items():
                    if name == "ctrl" and cname not in ("OFF", "FULL"):
                        continue
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
                elif name == "ctrl":
                    entry["ctrl"] = row
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
    summary["verdict"] = interpret(
        summary["mean_transfer"], summary["max_transfer"], summary["mean_orig_gain"], summary["mean_control_ratio"]
    )

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
        print(f"mean orig gain: {summary['mean_orig_gain']:.4f} nats | max transfer: {summary['max_transfer']} | "
              f"control(unrelated prompt) ratio: {summary['mean_control_ratio']}")
        print("VERDICT:", summary["verdict"])
        print(f"True held-out  : no-Engram PPL={heldout['no_engram_ppl']:.4f}  with-Engram PPL={heldout['with_engram_ppl']:.4f}  "
              f"(delta={heldout['delta_ppl']:+.4f}, acc {heldout['no_engram_acc']:.2f}% -> {heldout['with_engram_acc']:.2f}%)")
        print(f"Leaked anchors : no-Engram PPL={leaked['no_engram_ppl']:.4f}  with-Engram PPL={leaked['with_engram_ppl']:.4f}  "
              f"(these texts were TRAINED ON in Phase-4 and SI-SRSI)")
        print("=" * 118)
    return summary


# --------------------------------------------------------------------------------------
# Engram utility on leak-free text: scale sweep + leave-one-anchor-out + per-text paired deltas
# --------------------------------------------------------------------------------------
def _per_text_ce(model: Any, tokenizer: Any, texts: Sequence[str]) -> List[tuple]:
    """Return `[(sum_nll, n_tokens), ...]` (teacher-forced next-token CE over the whole text)."""
    import torch
    import torch.nn.functional as F

    device = next(model.base.parameters()).device
    out = []
    for txt in texts:
        ids = tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model.forward_logits(ids).float()
        sl = logits[:, :-1, :].contiguous()
        tl = ids[:, 1:].contiguous()
        nll = F.cross_entropy(sl.view(-1, sl.size(-1)), tl.view(-1), reduction="sum")
        out.append((float(nll.item()), int(tl.numel())))
    return out


def _mean_ce(rows: Sequence[tuple]) -> float:
    return sum(r[0] for r in rows) / max(1, sum(r[1] for r in rows))


def engram_utility_report(
    model: Any,
    tokenizer: Any,
    texts: Optional[Sequence[str]] = None,
    scales: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    flight_mode: str = "deep_knowledge",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Does the Engram help on text it was never trained on, at which strength, and which anchors help or hurt?

    * `OFF` vs `FULL` per-text paired CE deltas (positive delta = Engram hurts).
    * Scale sweep: semantic+lexical path scaled by s in `scales` (s=1 is the trained model, s=0 removes the layer's contribution).
    * Leave-one-anchor-out: ablate a single anchor; `removal_delta_ce` > 0 means the anchor was helping.

    Low power warning: the default 5 texts are ~1-2K tokens in total, so treat differences below ~0.02 nats as noise.
    """
    texts = list(texts) if texts is not None else list(TRUE_HELDOUT_TEXTS)
    was_training = model.training
    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    model.set_engram_flight_mode(flight_mode)
    model.set_domain_engram_focus("general", boost=0.0)

    def run(use_engram: bool, lex: float = 1.0, sem: float = 1.0, modules: Optional[Sequence[Any]] = None):
        model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=use_engram, enable_p3=True, p3_start=12, p3_end=24)
        with engram_ablation(model, lex=lex, sem=sem, modules=modules):
            return _per_text_ce(model, tokenizer, texts)

    rep: Dict[str, Any] = {"n_texts": len(texts)}
    try:
        off = run(False)
        full = run(True)
        rep["n_tokens"] = sum(r[1] for r in full)
        rep["ce_off"], rep["ce_full"] = _mean_ce(off), _mean_ce(full)
        rep["per_text_delta_ce"] = [f[0] / f[1] - o[0] / o[1] for o, f in zip(off, full)]
        rep["n_texts_engram_hurts"] = sum(1 for d in rep["per_text_delta_ce"] if d > 0)

        rep["scale_sweep"] = {}
        for sc in scales:
            rep["scale_sweep"][float(sc)] = _mean_ce(run(True, lex=float(sc), sem=float(sc)))

        rep["leave_one_anchor_out"] = {}
        for layer, mod in model.iter_engram_anchors():
            ce_wo = _mean_ce(run(True, lex=0.0, sem=0.0, modules=[mod]))
            rep["leave_one_anchor_out"][int(layer)] = {"ce_without": ce_wo, "removal_delta_ce": ce_wo - rep["ce_full"]}
    finally:
        model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
        model.enable_checkpoint = prev_ckpt
        model.settings.checkpoint_layers = prev_ckpt
        if was_training:
            model.train()
            model.base.eval()

    sweep = rep["scale_sweep"]
    best_scale = min(sweep, key=lambda k: sweep[k])
    rep["best_scale"] = best_scale
    rep["harmful_anchors"] = sorted(l for l, v in rep["leave_one_anchor_out"].items() if v["removal_delta_ce"] < -0.002)
    rep["helpful_anchors"] = sorted(l for l, v in rep["leave_one_anchor_out"].items() if v["removal_delta_ce"] > 0.002)

    if verbose:
        print("\n" + "=" * 100)
        print(f"[ENGRAM UTILITY ON LEAK-FREE TEXT] {rep['n_texts']} texts, {rep['n_tokens']} tokens  (noise floor ~0.02 nats)")
        print("=" * 100)
        print(f"CE  OFF={rep['ce_off']:.4f}   FULL={rep['ce_full']:.4f}   delta(FULL-OFF)={rep['ce_full'] - rep['ce_off']:+.4f}  "
              f"| engram hurts on {rep['n_texts_engram_hurts']}/{rep['n_texts']} texts")
        print("per-text delta CE:", [round(x, 4) for x in rep["per_text_delta_ce"]])
        print("scale sweep (CE):", {k: round(v, 4) for k, v in sweep.items()}, "| best scale:", best_scale)
        print(f"{'anchor':>6} {'CE w/o anchor':>14} {'removal delta':>14}  verdict")
        for l, v in sorted(rep["leave_one_anchor_out"].items()):
            tag = "HELPS" if v["removal_delta_ce"] > 0.002 else ("HURTS" if v["removal_delta_ce"] < -0.002 else "~neutral")
            print(f"{l:>6} {v['ce_without']:>14.4f} {v['removal_delta_ce']:>+14.4f}  {tag}")
        print("harmful anchors:", rep["harmful_anchors"], "| helpful anchors:", rep["helpful_anchors"])
        print("=" * 100)
    return rep


# --------------------------------------------------------------------------------------
# Train-time validation (used by Cell 4) + post-hoc, cross-validated anchor gating
# --------------------------------------------------------------------------------------
def _release_eval_caches(model: Any) -> None:
    """Drop GPU copies / caches that eval-mode Engram lookups create (stale after optimizer steps; large)."""
    for m in model.all_engram_modules():
        m._eval_tier1b_gpu = None
        m._eval_lex_cache = None


def _val_ce(model: Any, samples: Sequence[Dict[str, Any]], max_tokens: int) -> tuple:
    """Label-masked next-token CE over training-format samples `{'input_ids': 1-D, 'labels': 1-D}` -> (sum_nll, n_tokens)."""
    import torch
    import torch.nn.functional as F

    device = next(model.base.parameters()).device
    tot, n = 0.0, 0
    for s in samples:
        ids = s["input_ids"][:max_tokens].unsqueeze(0).to(device)
        lbl = s["labels"][:max_tokens].unsqueeze(0).to(device)
        cnt = int((lbl[:, 1:] != -100).sum().item())
        if cnt == 0:
            continue
        with torch.no_grad():
            logits = model.forward_logits(ids).float()
        nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), lbl[:, 1:].reshape(-1),
                              ignore_index=-100, reduction="sum")
        tot += float(nll.item())
        n += cnt
        del logits
    return tot, n


def stage_validation_report(
    model: Any,
    val_samples: Sequence[Dict[str, Any]],
    restore_loop: Dict[str, Any],
    max_tokens: int = 4096,
    anchor_audit: bool = True,
) -> Dict[str, Any]:
    """CE on held-out training-distribution sequences with Engram OFF / FULL / leave-one-anchor-out.

    `removal_delta_ce` < 0 means removing that anchor LOWERS validation CE, i.e. the anchor is harmful.
    Always restores the caller's loop config; the caller restores train mode.
    """
    import torch

    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    _release_eval_caches(model)
    rep: Dict[str, Any] = {}
    try:
        def ce(use_engram: bool, lex: float = 1.0, sem: float = 1.0, modules=None) -> float:
            kw = dict(restore_loop)
            kw["use_engram"] = use_engram
            model.set_loop(**kw)
            with engram_ablation(model, lex=lex, sem=sem, modules=modules):
                s, n = _val_ce(model, val_samples, max_tokens)
            return s / max(1, n)

        rep["ce_off"] = ce(False)
        rep["ce_full"] = ce(True)
        rep["engram_delta_ce"] = rep["ce_full"] - rep["ce_off"]  # positive = Engram hurts
        rep["n_val_seqs"] = len(val_samples)
        if anchor_audit:
            loo = {}
            for layer, mod in model.iter_engram_anchors():
                c = ce(True, lex=0.0, sem=0.0, modules=[mod])
                loo[int(layer)] = c - rep["ce_full"]
            rep["removal_delta_ce"] = loo
            rep["harmful_anchors"] = sorted(l for l, d in loo.items() if d < -0.002)
    finally:
        model.set_loop(**restore_loop)
        _release_eval_caches(model)
        model.enable_checkpoint = prev_ckpt
        model.settings.checkpoint_layers = prev_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rep


def maybe_stage_validation(
    model: Any,
    cfg: Dict[str, Any],
    val_samples: Sequence[Dict[str, Any]],
    stage_name: str,
    tag: str,
    restore_loop: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Crash-proof wrapper used by `run_v03_four_stage_training`; never raises, always returns the model to train mode."""
    if not val_samples or not bool(cfg.get("ENGRAM_STAGE_VALIDATION", True)) or not restore_loop.get("use_engram", False):
        return None
    out: Optional[Dict[str, Any]] = None
    try:
        out = stage_validation_report(
            model, val_samples, restore_loop,
            max_tokens=int(cfg.get("VAL_MAX_TOKENS", 4096)),
            anchor_audit=bool(cfg.get("VAL_ANCHOR_AUDIT", True)),
        )
        out.update({"stage": stage_name, "tag": tag})
        verdict = "Engram HURTS held-out CE" if out["engram_delta_ce"] > 0.005 else (
            "Engram helps" if out["engram_delta_ce"] < -0.005 else "Engram ~neutral")
        print(f"[Engram Val | {stage_name} | {tag}] {out['n_val_seqs']} held-out seqs: CE off={out['ce_off']:.4f} "
              f"full={out['ce_full']:.4f} delta={out['engram_delta_ce']:+.4f} -> {verdict}", flush=True)
        if "removal_delta_ce" in out:
            print("   anchor removal delta CE (negative = anchor harmful): "
                  + ", ".join(f"L{l}:{d:+.4f}" for l, d in sorted(out["removal_delta_ce"].items()))
                  + f" | harmful={out['harmful_anchors']}", flush=True)
    except Exception as exc:  # validation must never kill a multi-hour run
        print(f"[Engram Val Notice] skipped ({type(exc).__name__}: {exc})", flush=True)
    finally:
        model.train()
        model.enable_checkpoint = True
        model.settings.checkpoint_layers = True
    return out


def apply_anchor_gating(model: Any, anchors: Sequence[int]) -> List[int]:
    """Persistently silence the given anchor layers (gain -> -30, i.e. sigmoid ~ 0; saved in the state dict)."""
    import torch

    done = []
    by_layer = dict((int(l), m) for l, m in model.iter_engram_anchors())
    for l in anchors:
        mod = by_layer.get(int(l))
        if mod is None:
            continue
        with torch.no_grad():
            mod.gain.fill_(-30.0)
        done.append(int(l))
    return done


def cross_validated_anchor_gating(
    model: Any,
    tokenizer: Any,
    texts: Optional[Sequence[str]] = None,
    threshold: float = 0.002,
    flight_mode: str = "deep_knowledge",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Select harmful anchors on one half of the texts and score the gated model on the OTHER half (both directions).

    Avoids reporting a gain on the same texts used to choose the anchors. With very few texts this is still low-power;
    treat the result as provisional until a larger held-out set exists.
    """
    texts = list(texts) if texts is not None else list(TRUE_HELDOUT_TEXTS)
    folds = [texts[0::2], texts[1::2]]
    if min(len(f) for f in folds) < 1:
        raise ValueError("need at least 2 texts")
    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    model.set_engram_flight_mode(flight_mode)
    model.set_domain_engram_focus("general", boost=0.0)
    by_layer = dict((int(l), m) for l, m in model.iter_engram_anchors())

    def ce(tx, use_engram=True, modules=None):
        model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=use_engram, enable_p3=True, p3_start=12, p3_end=24)
        with engram_ablation(model, lex=0.0, sem=0.0, modules=modules) if modules else engram_ablation(model):
            rows = _per_text_ce(model, tokenizer, tx)
        return sum(r[0] for r in rows), sum(r[1] for r in rows)

    res: Dict[str, Any] = {"folds": []}
    tot = {"off": [0.0, 0], "full": [0.0, 0], "gated": [0.0, 0]}
    try:
        for sel, ev in ((folds[0], folds[1]), (folds[1], folds[0])):
            if not sel or not ev:
                continue
            s_full = ce(sel)
            s_full_ce = s_full[0] / s_full[1]
            deltas = {}
            for l, m in by_layer.items():
                s = ce(sel, modules=[m])
                deltas[l] = s[0] / s[1] - s_full_ce
            chosen = sorted(l for l, d in deltas.items() if d < -threshold)
            e_off, e_full = ce(ev, use_engram=False), ce(ev)
            e_gated = ce(ev, modules=[by_layer[l] for l in chosen]) if chosen else e_full
            for k, v in (("off", e_off), ("full", e_full), ("gated", e_gated)):
                tot[k][0] += v[0]
                tot[k][1] += v[1]
            res["folds"].append({"chosen": chosen, "selection_deltas": deltas,
                                 "eval_ce": {"off": e_off[0] / e_off[1], "full": e_full[0] / e_full[1], "gated": e_gated[0] / e_gated[1]}})
    finally:
        model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
        model.enable_checkpoint = prev_ckpt
        model.settings.checkpoint_layers = prev_ckpt
    res["cv_ce"] = {k: v[0] / max(1, v[1]) for k, v in tot.items()}
    sets = [set(f["chosen"]) for f in res["folds"]]
    res["stable_anchors"] = sorted(set.intersection(*sets)) if sets else []
    if verbose:
        print("\n" + "=" * 100)
        print("[CROSS-VALIDATED ANCHOR GATING] select on one half, evaluate on the other (token-weighted over both)")
        for i, f in enumerate(res["folds"]):
            print(f"  fold {i}: chosen={f['chosen']}  eval CE off={f['eval_ce']['off']:.4f} full={f['eval_ce']['full']:.4f} gated={f['eval_ce']['gated']:.4f}")
        c = res["cv_ce"]
        print(f"  CV CE: OFF={c['off']:.4f}  FULL={c['full']:.4f}  GATED={c['gated']:.4f}   (gated - off = {c['gated'] - c['off']:+.4f})")
        print(f"  anchors chosen in BOTH folds: {res['stable_anchors']}")
        print("=" * 100)
    return res
