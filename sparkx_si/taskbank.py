"""sparkx_si.taskbank — hash-split task construction (docs/roadmap/05 §2.1, §3A).

Split integrity: `sha1(prompt) % 100` -> 0–79 train, 80–89 dev, 90–99 test, so a prompt can never migrate between splits across
runs, machines or dataset re-orderings. The synthetic generators are deterministic in `seed` and need no datasets.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


def split_of(prompt: str, mod: int = 100) -> str:
    h = int(hashlib.sha1(prompt.encode("utf-8")).hexdigest(), 16) % mod
    frac = h / mod
    return "train" if frac < 0.80 else ("dev" if frac < 0.90 else "test")


def task_id(pillar: str, prompt: str) -> str:
    return hashlib.sha1(f"{pillar}\x00{prompt}".encode("utf-8")).hexdigest()[:16]


def make_task(pillar: str, prompt: str, spec: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"id": task_id(pillar, prompt), "pillar": pillar, "prompt": prompt, "spec": spec, "split": split_of(prompt),
            "meta": meta or {}}


def partition(tasks: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for t in tasks:
        out[t["split"]].append(t)
    return out


def assert_disjoint(parts: Dict[str, List[Dict[str, Any]]]) -> None:
    seen: Dict[str, str] = {}
    for sp, ts in parts.items():
        for t in ts:
            if t["prompt"] in seen and seen[t["prompt"]] != sp:
                raise AssertionError(f"prompt in both {seen[t['prompt']]} and {sp}")
            seen[t["prompt"]] = sp


# --------------------------------------------------------------------------------------
# Synthetic instruction-following tasks (composable constraints; infinite prompts)
# --------------------------------------------------------------------------------------
_TOPICS = [
    "the benefits of regular exercise", "how a refrigerator works", "why the sky is blue", "tips for learning a new language",
    "the history of the printing press", "how to brew good coffee", "what makes a good teacher", "the water cycle",
    "how vaccines train the immune system", "choosing a first programming language", "the importance of sleep",
    "how bridges carry loads", "why bees matter to farming", "how to prepare for a job interview", "the rules of chess",
    "how GPS finds your location", "recycling and what happens next", "how to keep a vegetable garden", "what inflation means",
    "how tides work",
]
_WORDS = ["river", "balance", "signal", "window", "habit", "energy", "pattern", "bridge", "garden", "memory", "engine", "market"]
_PHRASES = ["Let me know if you have questions.", "That is all for now.", "Thank you for reading.", "End of response."]

# id -> (builder(rng)->args, phrasing(args)->str, exclusive-group)
def _c_min_words(r): return {"n": r.choice([60, 80, 120, 150])}
def _c_max_words(r): return {"n": r.choice([40, 60, 80])}
def _c_max_sent(r): return {"n": r.choice([3, 4, 5])}
def _c_bullets(r): return {"n": r.choice([3, 4, 5])}
def _c_paras(r): return {"n": r.choice([2, 3, 4])}
def _c_forb(r): return {"words": r.sample(_WORDS, 2)}
def _c_req(r): return {"words": r.sample(_WORDS, 2)}
def _c_kw(r): return {"word": r.choice(_WORDS), "n": r.choice([2, 3])}
def _c_none(r): return {}
def _c_phrase(r): return {"phrase": r.choice(_PHRASES)}


_BUILD = {
    "min_words": (_c_min_words, lambda a: f"Use at least {a['n']} words."),
    "max_words": (_c_max_words, lambda a: f"Use at most {a['n']} words."),
    "max_sentences": (_c_max_sent, lambda a: f"Use no more than {a['n']} sentences."),
    "exact_bullets": (_c_bullets, lambda a: f"Format the answer as exactly {a['n']} bullet points, each starting with '- '."),
    "exact_paragraphs": (_c_paras, lambda a: f"Write exactly {a['n']} paragraphs separated by a blank line."),
    "forbidden_words": (_c_forb, lambda a: "Do not use the words " + " and ".join(f"'{w}'" for w in a["words"]) + "."),
    "required_words": (_c_req, lambda a: "Include the words " + " and ".join(f"'{w}'" for w in a["words"]) + "."),
    "keyword_min_count": (_c_kw, lambda a: f"The word '{a['word']}' must appear at least {a['n']} times."),
    "all_lowercase": (_c_none, lambda a: "Write the entire response in lowercase letters."),
    "all_uppercase": (_c_none, lambda a: "Write the entire response in UPPERCASE letters."),
    "no_commas": (_c_none, lambda a: "Do not use any commas."),
    "ends_with": (_c_phrase, lambda a: f"End your response with exactly this phrase: {a['phrase']}"),
    "title_in_brackets": (_c_none, lambda a: "Include a title wrapped in double angular brackets, like <<title>>."),
}
_CONFLICTS = [
    {"min_words", "max_words"}, {"all_lowercase", "all_uppercase"}, {"exact_bullets", "exact_paragraphs"},
    {"all_uppercase", "ends_with"}, {"all_lowercase", "ends_with"}, {"all_lowercase", "required_words"},
    {"all_uppercase", "required_words"}, {"all_uppercase", "keyword_min_count"}, {"all_lowercase", "keyword_min_count"},
    {"max_words", "keyword_min_count"}, {"max_words", "exact_paragraphs"}, {"max_sentences", "exact_paragraphs"},
    {"max_sentences", "exact_bullets"}, {"max_sentences", "min_words"}, {"max_words", "exact_bullets"},
]


def _compatible(ids: Sequence[str]) -> bool:
    s = set(ids)
    return not any(c <= s for c in _CONFLICTS)


def make_if_task(seed: int, n_constraints: Optional[int] = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    k = n_constraints or rng.choice([1, 2, 2, 3])
    for _ in range(100):
        ids = rng.sample(sorted(_BUILD), k)
        if _compatible(ids):
            break
    else:
        ids = ids[:1]
    cons, phr = [], []
    for i in ids:
        args = _BUILD[i][0](rng)
        cons.append({"id": i, "args": args})
        phr.append(_BUILD[i][1](args))
    topic = rng.choice(_TOPICS)
    prompt = f"Write a short explanation about {topic}. " + " ".join(phr)
    return make_task("if", prompt, {"constraints": cons, "strict": True}, {"seed": seed})


def build_if_bank(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    seen, out, s = set(), [], seed * 1_000_003
    while len(out) < n:
        t = make_if_task(s)
        s += 1
        if t["prompt"] in seen:
            continue
        seen.add(t["prompt"])
        out.append(t)
    return out


# --------------------------------------------------------------------------------------
# Procedural arithmetic word problems (offline smoke bank; gold computed exactly)
# --------------------------------------------------------------------------------------
_NAMES = ["Mira", "Tomas", "Lena", "Karim", "Joon", "Aisha", "Pavel", "Noor"]
_ITEMS = ["apples", "marbles", "stickers", "books", "pencils", "coins"]


def make_arith_task(seed: int) -> Dict[str, Any]:
    r = random.Random(seed)
    n, it = r.choice(_NAMES), r.choice(_ITEMS)
    a, b, c, d = r.randint(12, 90), r.randint(3, 9), r.randint(2, 15), r.randint(2, 6)
    kind = r.randint(0, 2)
    if kind == 0:
        q = f"{n} has {a} {it}. {n} buys {b} packs with {c} {it} each, then gives away {d}. How many {it} does {n} have now?"
        gold = a + b * c - d
    elif kind == 1:
        q = f"A box holds {c} {it}. {n} fills {b} boxes and has {d} {it} left over. How many {it} did {n} start with?"
        gold = b * c + d
    else:
        q = f"{n} shares {a * d} {it} equally among {d} friends and keeps {b} extra for later. How many {it} does each friend get?"
        gold = a
    return make_task("math", q + " Put the final answer in \\boxed{}.", {"gold": str(gold)}, {"seed": seed})


def build_arith_bank(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    seen, out, s = set(), [], seed * 1_000_003
    while len(out) < n:
        t = make_arith_task(s)
        s += 1
        if t["prompt"] in seen:
            continue
        seen.add(t["prompt"])
        out.append(t)
    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
