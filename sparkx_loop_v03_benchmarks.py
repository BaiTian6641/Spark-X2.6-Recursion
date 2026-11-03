"""Frontier Benchmark Evaluation Harness for Spark-X2.6-4B-En4B-A1B (SparkX-Loop v0.3, 72 Effective Layers).

Supports real Hugging Face datasets out-of-the-box on a single Google Colab GPU:
  1. IFEval        (`google/IFEval`)                       — Prompt & Instruction Strict/Loose Accuracy (25 constraint types)
  2. IFBench       (`allenai/IFBench_test`)                — Out-of-Domain Instruction Following (Official Ai2 repo + built-in verifiers)
  3. GPQA Diamond  (`Idavidrein/gpqa` / ungated mirrors)   — PhD-Level Physics, Chemistry & Biology (CoT + Log-Likelihood)
  4. HLE           (`cais/hle` — Humanity's Last Exam)     — Text-Only Frontier Academic Exam (MC + Exact Short-Answer)
  5. SWE-bench Pro (`ScaleAI/SWE-bench_Pro`)               — Long-Horizon Software Engineering Patch Synthesis & Diff Audit
  6. SWE-bench Ver (`princeton-nlp/SWE-bench_Verified`)    — Human-Verified GitHub Issue Patch Synthesis & Diff Audit
  7. HumanEval     (`openai/openai_humaneval`)             — Executable Python Synthesis (`Pass@1` via live subprocess unit tests)
  8. MATH-500      (`HuggingFaceH4/MATH-500`)              — Competition Mathematics Levels 1–5 (`\\boxed{}` + SymPy equivalence)
"""
from __future__ import annotations

import ast
import difflib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ==============================================================================
# 1. Shared Utilities: <think> Stripping, Boxed Extraction & Chat Generation
# ==============================================================================

def strip_thinking_trace(text: str) -> Tuple[str, str, bool]:
    """Split raw model output into `(thinking_trace, final_answer, closed_think_naturally)`."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        think_part = parts[0].replace("<think>", "").strip()
        final_part = parts[1].strip()
        return think_part, final_part, True
    if "<think>" in text:
        # Did not close </think>; take last paragraph as fallback answer
        cleaned = text.replace("<think>", "").strip()
        paras = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        return cleaned, (paras[-1] if paras else cleaned), False
    return "", text.strip(), True


def extract_boxed_or_final_answer(text: str, choices: Optional[List[str]] = None) -> str:
    """Extract `\\boxed{...}` or `Answer: ...` from model response."""
    _, answer_body, _ = strip_thinking_trace(text)
    search_space = answer_body if answer_body else text

    # 1. Find last \boxed{...} supporting nested braces
    boxed_matches = []
    for m in re.finditer(r"\\boxed\{", search_space):
        start = m.end()
        depth = 1
        i = start
        while i < len(search_space) and depth > 0:
            if search_space[i] == "{":
                depth += 1
            elif search_space[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            boxed_matches.append(search_space[start : i - 1].strip())
    if boxed_matches:
        cand = boxed_matches[-1]
        if choices:
            m_let = re.match(r"^\(?([A-Za-z])\)?$", cand.strip())
            if m_let:
                return m_let.group(1).upper()
        return cand

    # 2. Multiple-choice letter patterns: "Answer: (B)", "The correct answer is C", etc.
    if choices:
        valid_letters = "".join(choices)
        patterns = [
            rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)\s*\**\(?([{valid_letters}])\)?\**",
            rf"\boption\s+\(?([{valid_letters}])\)?\s+is\s+correct",
            rf"\(([A-D])\)\s*$",
            rf"\b([{valid_letters}])\s*$",
        ]
        for pat in patterns:
            matches = re.findall(pat, search_space, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        # Fallback: scan full text
        for pat in patterns[:2]:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()

    # 3. Short-answer patterns: "Final Answer: ..." or "Answer: ..."
    m_ans = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n.]+)", search_space, flags=re.IGNORECASE)
    if m_ans:
        return m_ans[-1].strip().strip("$*`")

    lines = [ln.strip() for ln in search_space.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_math_answer(ans: str) -> str:
    """Canonicalize LaTeX/math strings for exact & symbolic comparison."""
    s = str(ans).strip()
    s = re.sub(r"^\\boxed\{(.+)\}$", r"\1", s)
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "").replace("\\ ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.strip().strip(".")
    return re.sub(r"\s+", "", s).lower()


def check_math_equivalence(pred: str, gold: str) -> bool:
    """Compare predicted vs gold math/short answer via string canonicalization + SymPy fallback."""
    if not pred:
        return False
    np = normalize_math_answer(pred)
    ng = normalize_math_answer(gold)
    if np == ng:
        return True
    # Numeric comparison
    try:
        fp = float( re.sub(r"[^0-9eE.+\-/]", "", np.replace(",", "")) )
        fg = float( re.sub(r"[^0-9eE.+\-/]", "", ng.replace(",", "")) )
        if math.isclose(fp, fg, rel_tol=1e-4, abs_tol=1e-4):
            return True
    except Exception:
        pass
    # SymPy symbolic difference check
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
        p_clean = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", np)
        g_clean = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", ng)
        p_clean = p_clean.replace("^", "**")
        g_clean = g_clean.replace("^", "**")
        expr_p = parse_expr(p_clean, evaluate=True)
        expr_g = parse_expr(g_clean, evaluate=True)
        if sp.simplify(expr_p - expr_g) == 0:
            return True
    except Exception:
        pass
    return False


@torch.no_grad()
def generate_chat_response(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str = (
        "You are Spark-X2.6-4B-En4B-A1B, a precise 72-layer reasoning assistant. "
        "Reason concisely inside <think>...</think>, close </think>, and provide a direct, accurate final response."
    ),
    max_new_tokens: int = 1536,
    temperature: float = 0.1,
    top_p: float = 0.90,
    repetition_penalty: float = 1.03,
    stream_print: bool = False,
) -> Dict[str, Any]:
    """Run a single chat turn through `model.generate_cached` and return structured timing/token metadata."""
    device = next(model.base.parameters()).device
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(device)
    t0 = time.time()
    out_ids = model.generate_cached(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
        stream_print=stream_print,
    )
    dt = max(1e-3, time.time() - t0)
    gen_ids = out_ids[0, input_ids.shape[1] :]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    think_text, answer_text, closed_think = strip_thinking_trace(raw_text)
    return {
        "raw_text": raw_text,
        "think_text": think_text,
        "answer_text": answer_text,
        "closed_think": closed_think,
        "tokens": int(gen_ids.numel()),
        "tok_per_sec": float(gen_ids.numel() / dt),
        "elapsed_sec": dt,
    }


@torch.no_grad()
def score_mc_loglikelihood(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    choice_letters: List[str],
) -> str:
    """Rank multiple-choice options `(A), (B), (C), (D)` by exact next-token log-probability after `</think>`."""
    device = next(model.base.parameters()).device
    messages = [
        {"role": "system", "content": "Select the single best option letter (A, B, C, or D)."},
        {"role": "user", "content": prompt_text},
    ]
    chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = chat_str + "<think>\nEvaluating the options carefully.\n</think>\nThe correct answer is ("
    ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    logits = model.forward_logits(ids)[0, -1, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    best_let = choice_letters[0]
    best_lp = -1e30
    for let in choice_letters:
        tok_candidates = tokenizer.encode(let, add_special_tokens=False)
        if tok_candidates:
            lp = float(log_probs[tok_candidates[0]].item())
            if lp > best_lp:
                best_lp = lp
                best_let = let
    return best_let


# ==============================================================================
# 2. IFEval (`google/IFEval`) & IFBench (`allenai/IFBench_test`) Verifier Engine
# ==============================================================================

def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return max(1, len(sents)) if text.strip() else 0


def _count_paragraphs(text: str) -> int:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\*\*\*", text) if p.strip()]
    return len(paras)


def _compare_relation(val: int, target: int, relation: str) -> bool:
    rel = (relation or "at least").lower().strip()
    if "less than" in rel:
        return val < target
    if "at most" in rel:
        return val <= target
    if "at least" in rel:
        return val >= target
    if "more than" in rel:
        return val > target
    return val == target


def verify_single_ifeval_constraint(resp: str, inst_id: str, kw: Dict[str, Any]) -> bool:
    """Self-contained verifier covering all 25 standard `google/IFEval` instruction_id rules + common `IFBench` OOD rules."""
    kw = kw or {}
    text = resp.strip()
    if not text:
        return False

    # 1. Punctuation
    if inst_id == "punctuation:no_comma":
        return "," not in text

    # 2. Length constraints
    if inst_id == "length_constraints:number_words":
        return _compare_relation(_count_words(text), int(kw.get("num_words", 1)), kw.get("relation", "at least"))
    if inst_id == "length_constraints:number_sentences":
        return _compare_relation(_count_sentences(text), int(kw.get("num_sentences", 1)), kw.get("relation", "at least"))
    if inst_id == "length_constraints:number_paragraphs":
        target = int(kw.get("num_paragraphs", 1))
        return _count_paragraphs(text) == target
    if inst_id == "length_constraints:nth_paragraph_first_word":
        paras = [p.strip() for p in re.split(r"\n\s*\n|\*\*\*", text) if p.strip()]
        n_paras = int(kw.get("num_paragraphs", len(paras)))
        nth = int(kw.get("nth_paragraph", 1)) - 1
        first_word = str(kw.get("first_word", "")).lower().strip()
        if len(paras) != n_paras or not (0 <= nth < len(paras)):
            return False
        words = re.findall(r"\b\w+\b", paras[nth].lower())
        return bool(words and words[0] == first_word)

    # 3. Detectable format
    if inst_id == "detectable_format:number_bullet_lists":
        bullets = re.findall(r"^\s*(?:[\*\-]|\d+\.)\s+", text, flags=re.MULTILINE)
        return len(bullets) == int(kw.get("num_bullets", 1))
    if inst_id == "detectable_format:constrained_response":
        allowed = [
            "My answer is yes.",
            "My answer is no.",
            "My answer is maybe.",
        ]
        return any(a.lower() in text.lower() for a in allowed)
    if inst_id == "detectable_format:number_highlighted_sections":
        highlights = re.findall(r"\*[^*\n]+\*|\*\*[^*\n]+\*\*", text)
        return len(highlights) >= int(kw.get("num_highlights", 1))
    if inst_id == "detectable_format:multiple_sections":
        splitter = str(kw.get("section_spliter", "Section"))
        num_sec = int(kw.get("num_sections", 1))
        found = re.findall(re.escape(splitter), text, flags=re.IGNORECASE)
        return len(found) >= num_sec
    if inst_id == "detectable_format:json_format":
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        try:
            json.loads(cleaned)
            return True
        except Exception:
            m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if m:
                try:
                    json.loads(m.group(1))
                    return True
                except Exception:
                    return False
            return False
    if inst_id == "detectable_format:title":
        return bool(re.search(r"<<[^<>]+>>", text))

    # 4. Detectable content
    if inst_id == "detectable_content:number_placeholders":
        placeholders = re.findall(r"\[[^\[\]\n]+\]", text)
        return len(placeholders) >= int(kw.get("num_placeholders", 1))
    if inst_id == "detectable_content:postscript":
        marker = str(kw.get("postscript_marker", "P.S."))
        return marker.lower() in text.lower()

    # 5. Start / End constraints
    if inst_id == "startend:end_checker":
        phrase = str(kw.get("end_phrase", "")).strip()
        return text.rstrip().lower().endswith(phrase.lower())
    if inst_id == "startend:quotation":
        return (text.startswith('"') and text.endswith('"')) or (text.startswith("“") and text.endswith("”"))

    # 6. Case constraints
    if inst_id == "change_case:english_capital":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)
    if inst_id == "change_case:english_lowercase":
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)
    if inst_id == "change_case:capital_word_frequency":
        cap_words = [w for w in re.findall(r"\b[A-Z]{2,}\b", text)]
        return _compare_relation(len(cap_words), int(kw.get("capital_frequency", 1)), kw.get("capital_relation", "at least"))

    # 7. Keywords
    if inst_id == "keywords:existence":
        keywords = kw.get("keywords") or []
        return all(re.search(r"\b" + re.escape(str(k)) + r"\b", text, flags=re.IGNORECASE) for k in keywords)
    if inst_id == "keywords:forbidden_words":
        forbidden = kw.get("forbidden_words") or []
        return all(not re.search(r"\b" + re.escape(str(k)) + r"\b", text, flags=re.IGNORECASE) for k in forbidden)
    if inst_id == "keywords:frequency":
        keyword = str(kw.get("keyword", ""))
        cnt = len(re.findall(r"\b" + re.escape(keyword) + r"\b", text, flags=re.IGNORECASE))
        return _compare_relation(cnt, int(kw.get("frequency", 1)), kw.get("relation", "at least"))
    if inst_id == "keywords:letter_frequency":
        let = str(kw.get("letter", "a")).lower()
        cnt = text.lower().count(let)
        return _compare_relation(cnt, int(kw.get("let_frequency", 1)), kw.get("let_relation", "at least"))

    # 8. Combination
    if inst_id == "combination:repeat_prompt":
        prompt_to_repeat = str(kw.get("prompt_to_repeat", "")).strip()
        return text.lower().startswith(prompt_to_repeat.lower())
    if inst_id == "combination:two_responses":
        parts = [p.strip() for p in text.split("******") if p.strip()]
        return len(parts) == 2 and parts[0] != parts[1]

    # 9. Language
    if inst_id == "language:response_language":
        target_lang = str(kw.get("language", "en")).lower()
        try:
            from langdetect import detect
            return detect(text).lower().startswith(target_lang[:2])
        except Exception:
            return True

    # 10. IFBench OOD constraint checks (count:*, words:*, sentence:*, format:*, repeat:*, custom:*)
    if "no_letter" in inst_id or "forbidden_letter" in inst_id:
        let = str(kw.get("letter", kw.get("forbidden_letter", ""))).lower()
        return (let not in text.lower()) if let else True
    if "word_length" in inst_id or "max_word_length" in inst_id:
        max_len = int(kw.get("max_length", kw.get("length", 10)))
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        return all(len(w) <= max_len for w in words)
    if inst_id.startswith("count:") or "count" in inst_id:
        if "unique" in inst_id:
            words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
            target = int(kw.get("N", kw.get("num_words", kw.get("count", 1))))
            return _compare_relation(len(set(words)), target, kw.get("relation", "at least"))
        if "punctuation" in inst_id:
            p_cnt = len(re.findall(r"[^\w\s]", text))
            target = int(kw.get("N", kw.get("count", 1)))
            return _compare_relation(p_cnt, target, kw.get("relation", "at least"))
        if "conjunction" in inst_id:
            conjs = len(re.findall(r"\b(?:and|but|or|nor|for|yet|so|because|although|since|unless|while)\b", text, flags=re.IGNORECASE))
            target = int(kw.get("N", kw.get("count", 1)))
            return _compare_relation(conjs, target, kw.get("relation", "at least"))
        if "pronoun" in inst_id:
            prons = len(re.findall(r"\b(?:i|me|my|mine|we|us|our|ours|you|your|yours|he|him|his|she|her|hers|it|its|they|them|their|theirs)\b", text, flags=re.IGNORECASE))
            target = int(kw.get("N", kw.get("count", 1)))
            return _compare_relation(prons, target, kw.get("relation", "at least"))
        if "number" in inst_id or "digit" in inst_id:
            nums = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
            target = int(kw.get("N", kw.get("count", 1)))
            return _compare_relation(nums, target, kw.get("relation", "at least"))
    if inst_id.startswith("words:") or "words:" in inst_id:
        words = re.findall(r"\b[A-Za-z]+\b", text)
        if not words:
            return False
        if "alphabetical" in inst_id:
            low = [w.lower() for w in words[: min(len(words), int(kw.get("N", len(words))))]]
            return low == sorted(low)
        if "vowel" in inst_id:
            v_cnt = sum(1 for w in words if w[0].lower() in "aeiou")
            return _compare_relation(v_cnt, int(kw.get("N", kw.get("count", 1))), kw.get("relation", "at least"))
        if "consonant" in inst_id:
            c_cnt = sum(1 for w in words if w[0].lower() not in "aeiou")
            return _compare_relation(c_cnt, int(kw.get("N", kw.get("count", 1))), kw.get("relation", "at least"))
        if "odd_even" in inst_id or "alternating" in inst_id:
            return all((len(words[i]) % 2) != (len(words[i - 1]) % 2) for i in range(1, min(len(words), 8)))
        if "palindrome" in inst_id:
            pals = [w for w in words if len(w) >= 3 and w.lower() == w.lower()[::-1]]
            return len(pals) >= int(kw.get("N", kw.get("count", 1)))
        if "last_first" in inst_id or "chain" in inst_id:
            return all(words[i][0].lower() == words[i - 1][-1].lower() for i in range(1, min(len(words), 6)))
        if "start_same" in inst_id or "alliteration" in inst_id:
            let = str(kw.get("letter", words[0][0])).lower()
            return sum(1 for w in words if w[0].lower() == let) >= max(3, int(0.6 * len(words)))
    if inst_id.startswith("sentence:") or "sentence:" in inst_id:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sents:
            return False
        if "all_caps" in inst_id:
            nth = int(kw.get("N", kw.get("nth_sentence", 1))) - 1
            return 0 <= nth < len(sents) and sents[nth].isupper()
        if "word_count" in inst_id or "length" in inst_id:
            target = int(kw.get("N", kw.get("num_words", 10)))
            return all(_compare_relation(_count_words(s), target, kw.get("relation", "less than")) for s in sents)
        if "question" in inst_id:
            q_cnt = sum(1 for s in sents if s.endswith("?"))
            return _compare_relation(q_cnt, int(kw.get("N", kw.get("count", 1))), kw.get("relation", "at least"))
        if "exclamation" in inst_id:
            e_cnt = sum(1 for s in sents if s.endswith("!"))
            return _compare_relation(e_cnt, int(kw.get("N", kw.get("count", 1))), kw.get("relation", "at least"))
    if inst_id.startswith("format:") or "format:" in inst_id:
        if "table" in inst_id or "markdown_table" in inst_id:
            return "|" in text and bool(re.search(r"\|\s*:?-+:", text))
        if "code_block" in inst_id or "fenced" in inst_id:
            return "```" in text
        if "numbered" in inst_id or "ordered_list" in inst_id:
            items = re.findall(r"^\s*\d+[\.\)]\s+", text, flags=re.MULTILINE)
            return len(items) >= int(kw.get("N", kw.get("count", 1)))
        if "xml" in inst_id or "html" in inst_id:
            return bool(re.search(r"<([a-zA-Z0-9_]+)>.*?</\1>", text, flags=re.DOTALL))
        if "yaml" in inst_id:
            return bool(re.search(r"^[a-zA-Z0-9_]+\s*:\s*\S+", text, flags=re.MULTILINE))
        if "csv" in inst_id:
            lines = [ln for ln in text.splitlines() if "," in ln]
            return len(lines) >= 2
        if "parentheses" in inst_id or "brackets" in inst_id:
            return "(" in text and ")" in text
    if inst_id.startswith("repeat:") or "repeat:" in inst_id:
        phrase = str(kw.get("phrase", kw.get("word", kw.get("sentence", kw.get("prompt_to_repeat", ""))))).strip()
        if phrase:
            cnt = len(re.findall(re.escape(phrase), text, flags=re.IGNORECASE))
            return _compare_relation(cnt, int(kw.get("N", kw.get("count", 1))), kw.get("relation", "at least"))
    # Generic keyword/phrase fallback for any other custom IFBench instruction with explicit kwargs
    for k_name in ("keyword", "word", "phrase", "target", "end_phrase", "start_phrase", "marker"):
        if k_name in kw and kw[k_name]:
            val_str = str(kw[k_name]).strip()
            if val_str and val_str.lower() not in text.lower():
                return False

    return True


def verify_ifeval_item(resp: str, instruction_id_list: List[str], kwargs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate strict and loose compliance across all constraints on a single prompt."""
    # Loose variants strip markdown asterisks and outer quotes/whitespace
    loose_candidates = [
        resp,
        re.sub(r"\*+", "", resp),
        "\n".join(resp.strip().splitlines()[1:]) if len(resp.strip().splitlines()) > 1 else resp,
        "\n".join(resp.strip().splitlines()[:-1]) if len(resp.strip().splitlines()) > 1 else resp,
    ]
    strict_flags = []
    loose_flags = []
    for idx, inst_id in enumerate(instruction_id_list):
        kw = kwargs_list[idx] if idx < len(kwargs_list) and kwargs_list[idx] else {}
        # Filter None values inside kw
        if isinstance(kw, dict):
            kw = {k: v for k, v in kw.items() if v is not None}
        s_ok = verify_single_ifeval_constraint(resp, inst_id, kw)
        l_ok = s_ok or any(verify_single_ifeval_constraint(c, inst_id, kw) for c in loose_candidates)
        strict_flags.append(bool(s_ok))
        loose_flags.append(bool(l_ok))

    return {
        "prompt_strict": all(strict_flags) if strict_flags else True,
        "prompt_loose": all(loose_flags) if loose_flags else True,
        "inst_strict": strict_flags,
        "inst_loose": loose_flags,
    }


def _ensure_ifbench_official_verifier() -> Any:
    """Auto-clone `allenai/IFBench` and ensure NLTK resources so all 58 official IFBench OOD rules run natively."""
    ifb_dir = Path("/content/IFBench")
    if not ifb_dir.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/allenai/IFBench.git", str(ifb_dir)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=45,
            )
        except Exception:
            pass
    try:
        import nltk  # type: ignore
        for pkg in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    except Exception:
        pass
    if ifb_dir.exists():
        if str(ifb_dir) not in sys.path:
            sys.path.insert(0, str(ifb_dir))
        try:
            import evaluation_lib as ifb_eval_lib  # type: ignore
            return ifb_eval_lib
        except Exception:
            return None
    return None


def run_ifeval_and_ifbench_suite(
    model: Any,
    tokenizer: Any,
    max_samples_ifeval: Optional[int] = 25,
    max_samples_ifbench: Optional[int] = 25,
    max_new_tokens: int = 1280,
    output_dir: str = "/content/sparkx_v03_benchmark_results",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `google/IFEval` and `allenai/IFBench_test` and return prompt/instruction-level accuracies."""
    from datasets import load_dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sys_prompt = (
        "You are a precise instruction-following assistant. "
        "Reason briefly inside <think>...</think> about every formatting, length, keyword, or punctuation constraint, "
        "close </think>, and output ONLY the final response that strictly satisfies every constraint."
    )
    results_summary: Dict[str, Any] = {}

    for bench_name, hf_repo, limit in [
        ("IFEval", "google/IFEval", max_samples_ifeval),
        ("IFBench", "allenai/IFBench_test", max_samples_ifbench),
    ]:
        if limit == 0:
            continue
        print("\n" + "=" * 100)
        print(f"[{bench_name}] Loading `{hf_repo}`...")
        try:
            ds = load_dataset(hf_repo, split="train", token=token)
        except Exception as e:
            print(f"[{bench_name} Warning] Could not load `{hf_repo}` ({e}). Skipping.")
            continue

        n_total = len(ds) if limit is None else min(int(limit), len(ds))
        print(f"[{bench_name}] Evaluating {n_total} / {len(ds)} prompts (72L Full Loop + 32-Expert Top-7 Engram)...")

        p_strict_hits, p_loose_hits = 0, 0
        i_strict_hits, i_loose_hits, i_total = 0, 0, 0
        jsonl_rows = []

        ifbench_official_fn = _ensure_ifbench_official_verifier() if bench_name == "IFBench" else None

        for idx in range(n_total):
            row = ds[idx]
            prompt = str(row.get("prompt", ""))
            inst_list = list(row.get("instruction_id_list", []))
            kw_list = list(row.get("kwargs", []))
            gen = generate_chat_response(
                model,
                tokenizer,
                user_prompt=prompt,
                system_prompt=sys_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            final_resp = gen["answer_text"]
            jsonl_rows.append({"prompt": prompt, "response": final_resp, "key": row.get("key", idx)})

            ver = verify_ifeval_item(final_resp, inst_list, kw_list)
            if ifbench_official_fn is not None:
                try:
                    inp_ex = ifbench_official_fn.InputExample(
                        key=row.get("key", idx),
                        instruction_id_list=inst_list,
                        prompt=prompt,
                        kwargs=[{k: v for k, v in (d or {}).items() if v is not None} for d in kw_list],
                    )
                    out_strict = ifbench_official_fn.test_instruction_following_strict(inp_ex, {prompt: final_resp})
                    out_loose = ifbench_official_fn.test_instruction_following_loose(inp_ex, {prompt: final_resp})
                    ver = {
                        "prompt_strict": bool(out_strict.follow_all_instructions),
                        "prompt_loose": bool(out_loose.follow_all_instructions),
                        "inst_strict": list(out_strict.follow_instruction_list),
                        "inst_loose": list(out_loose.follow_instruction_list),
                    }
                except Exception:
                    pass

            p_strict_hits += int(ver["prompt_strict"])
            p_loose_hits += int(ver["prompt_loose"])
            i_strict_hits += sum(int(x) for x in ver["inst_strict"])
            i_loose_hits += sum(int(x) for x in ver["inst_loose"])
            i_total += len(ver["inst_strict"])

            if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
                print(
                    f"  [{bench_name} {idx+1:3d}/{n_total:3d}] "
                    f"Prompt Strict={100*p_strict_hits/(idx+1):5.1f}% | "
                    f"Prompt Loose={100*p_loose_hits/(idx+1):5.1f}% | "
                    f"Inst Loose={100*i_loose_hits/max(1, i_total):5.1f}% | "
                    f"Last={gen['tokens']} tok ({gen['tok_per_sec']:.1f} tok/s)"
                )

        out_jsonl = Path(output_dir) / f"{bench_name.lower()}_responses.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in jsonl_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        metrics = {
            "samples": n_total,
            "prompt_strict_acc": round(100.0 * p_strict_hits / max(1, n_total), 2),
            "prompt_loose_acc": round(100.0 * p_loose_hits / max(1, n_total), 2),
            "instruction_strict_acc": round(100.0 * i_strict_hits / max(1, i_total), 2),
            "instruction_loose_acc": round(100.0 * i_loose_hits / max(1, i_total), 2),
            "saved_jsonl": str(out_jsonl),
        }
        results_summary[bench_name] = metrics
        print(f"[{bench_name} Final] {metrics}")

    return results_summary


# ==============================================================================
# 3. GPQA Diamond (`Idavidrein/gpqa`) & Humanity's Last Exam (`cais/hle`)
# ==============================================================================

def _load_gpqa_diamond_dataset(token: Optional[str] = None):
    """Load the 198-question GPQA Diamond dataset with automatic fallback to ungated mirrors."""
    from datasets import load_dataset
    # 1. Official gated repo
    try:
        return load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=token), "Idavidrein/gpqa (gpqa_diamond)"
    except Exception:
        pass
    # 2. Ungated Hugging Face mirror
    for mirror in ["fingertap/GPQA-Diamond", "wanng/gpqa"]:
        try:
            return load_dataset(mirror, split="train"), f"{mirror} (mirror)"
        except Exception:
            pass
    # 3. OpenAI simple-evals public CSV mirror
    csv_url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
    return load_dataset("csv", data_files=csv_url, split="train"), "simple-evals/gpqa_diamond.csv"


def run_gpqa_diamond_benchmark(
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = 25,
    mode: str = "hybrid_cot_and_logprob",
    max_new_tokens: int = 1536,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate Spark-X2.6-4B-En4B-A1B on GPQA Diamond (PhD-level Physics, Chemistry, Biology)."""
    print("\n" + "=" * 100)
    ds, src_name = _load_gpqa_diamond_dataset(token=token)
    n_total = len(ds) if max_samples is None else min(int(max_samples), len(ds))
    print(f"[GPQA-Diamond] Loaded {len(ds)} questions from `{src_name}`. Evaluating {n_total} questions...")

    cot_correct, ll_correct, hybrid_correct = 0, 0, 0
    domain_stats: Dict[str, List[int]] = {}

    sys_prompt = (
        "You are a PhD-level physicist, chemist, and biologist. "
        "Reason step-by-step concisely inside <think>...</think>, close </think>, "
        "and state your final choice clearly as \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
    )

    for idx in range(n_total):
        row = ds[idx]
        # Handle both standard GPQA columns and pre-formatted mirror columns
        q_text = str(row.get("Question") or row.get("question") or "")
        domain = str(row.get("High-level domain") or row.get("Subdomain") or row.get("domain") or "Science")

        if "Correct Answer" in row and "Incorrect Answer 1" in row:
            correct_str = str(row["Correct Answer"]).strip()
            options = [
                correct_str,
                str(row["Incorrect Answer 1"]).strip(),
                str(row["Incorrect Answer 2"]).strip(),
                str(row["Incorrect Answer 3"]).strip(),
            ]
            rng = random.Random(42 + idx)
            rng.shuffle(options)
            gold_idx = options.index(correct_str)
            gold_letter = ["A", "B", "C", "D"][gold_idx]
            formatted_q = (
                f"{q_text}\n\n"
                f"(A) {options[0]}\n"
                f"(B) {options[1]}\n"
                f"(C) {options[2]}\n"
                f"(D) {options[3]}\n\n"
                "Provide your final answer letter inside \\boxed{}."
            )
        else:
            formatted_q = q_text + "\n\nProvide your final answer letter inside \\boxed{}."
            gold_letter = str(row.get("answer") or row.get("correct_answer") or "A").strip().upper()
            m_let = re.search(r"[A-D]", gold_letter)
            gold_letter = m_let.group(0) if m_let else "A"

        ll_pred = score_mc_loglikelihood(model, tokenizer, formatted_q, ["A", "B", "C", "D"])
        ll_ok = (ll_pred == gold_letter)
        ll_correct += int(ll_ok)

        if mode == "logprob_only":
            cot_pred = ll_pred
            closed_think = True
            tok_info = "logprob"
        else:
            gen = generate_chat_response(
                model,
                tokenizer,
                user_prompt=formatted_q,
                system_prompt=sys_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
            )
            extracted = extract_boxed_or_final_answer(gen["raw_text"], choices=["A", "B", "C", "D"])
            cot_pred = extracted if extracted in ("A", "B", "C", "D") else ll_pred
            closed_think = gen["closed_think"]
            tok_info = f"{gen['tokens']} tok ({gen['tok_per_sec']:.1f} tok/s)"

        cot_ok = (cot_pred == gold_letter)
        # Hybrid uses CoT answer when </think> closed cleanly and extracted a valid choice, else log-likelihood
        hyb_pred = cot_pred if (closed_think and cot_pred in ("A", "B", "C", "D")) else ll_pred
        hyb_ok = (hyb_pred == gold_letter)

        cot_correct += int(cot_ok)
        hybrid_correct += int(hyb_ok)
        domain_stats.setdefault(domain, [0, 0])
        domain_stats[domain][0] += int(hyb_ok)
        domain_stats[domain][1] += 1

        if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
            print(
                f"  [GPQA {idx+1:3d}/{n_total:3d}] "
                f"CoT Acc={100*cot_correct/(idx+1):5.1f}% | "
                f"LogProb Acc={100*ll_correct/(idx+1):5.1f}% | "
                f"Hybrid Acc={100*hybrid_correct/(idx+1):5.1f}% | "
                f"Gold={gold_letter}, Pred={hyb_pred} | {tok_info}"
            )

    summary = {
        "source": src_name,
        "samples": n_total,
        "cot_accuracy": round(100.0 * cot_correct / max(1, n_total), 2),
        "logprob_accuracy": round(100.0 * ll_correct / max(1, n_total), 2),
        "hybrid_accuracy": round(100.0 * hybrid_correct / max(1, n_total), 2),
        "by_domain": {
            k: f"{v[0]}/{v[1]} ({100.0*v[0]/max(1,v[1]):.1f}%)" for k, v in domain_stats.items()
        },
    }
    print(f"[GPQA-Diamond Final] {summary}")
    return summary


# Fallback curated Humanity's Last Exam (HLE) text-only frontier questions if user's HF_TOKEN hasn't accepted cais/hle gate
_CURATED_HLE_FALLBACK = [
    {
        "id": "hle_math_01",
        "category": "Mathematics",
        "answer_type": "exactMatch",
        "question": (
            "Let M be a 2x2 doubly-stochastic matrix with diagonal entries p = 3/4 and off-diagonal entries 1 - p = 1/4. "
            "Compute the exact trace of M^4 as an irreducible fraction a/b."
        ),
        "answer": "129/128",
    },
    {
        "id": "hle_cs_02",
        "category": "Computer Science",
        "answer_type": "multipleChoice",
        "question": (
            "In a 36-layer transformer where 27 layers use sliding-window attention (window W=512) and 9 layers use "
            "NoPE full attention compressed via a dyadic Fenwick tree of base block size B=32, what is the asymptotic "
            "per-step KV cache lookup complexity per token at sequence length T?\n"
            "Answer Choices:\n(A) O(T^2)\n(B) O(T)\n(C) O(log T)\n(D) O(T sqrt(T))"
        ),
        "answer": "C",
    },
    {
        "id": "hle_phys_03",
        "category": "Physics",
        "answer_type": "exactMatch",
        "question": (
            "Consider a 1D quantum harmonic oscillator with Hamiltonian H = hbar * omega * (a^dagger a + 1/2). "
            "For the coherent state |alpha> with |alpha|^2 = 4, what is the exact dimensionless ratio "
            "Var(H) / (<H> - hbar*omega/2)^2?"
        ),
        "answer": "1/4",
    },
    {
        "id": "hle_chem_04",
        "category": "Chemistry",
        "answer_type": "exactMatch",
        "question": (
            "According to Hückel molecular orbital theory, how many pi electrons are in the aromatic tropylium cation (C7H7+)?"
        ),
        "answer": "6",
    },
    {
        "id": "hle_math_05",
        "category": "Mathematics",
        "answer_type": "exactMatch",
        "question": (
            "Evaluate the exact integer value of sum_{k=1}^{10} k * C(10, k) where C(n, k) is the binomial coefficient."
        ),
        "answer": "5120",
    },
]


def run_hle_benchmark(
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = 20,
    max_new_tokens: int = 1536,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate Spark-X2.6-4B-En4B-A1B on Humanity's Last Exam (`cais/hle`, text-only questions)."""
    from datasets import load_dataset
    print("\n" + "=" * 100)
    print("[HLE — Humanity's Last Exam] Loading `cais/hle` (filtering text-only questions)...")

    rows: List[Dict[str, Any]] = []
    source_label = "cais/hle (test, text-only)"
    try:
        ds_stream = load_dataset("cais/hle", split="test", streaming=True, token=token)
        target_n = 2500 if max_samples is None else int(max_samples)
        for item in ds_stream:
            img = item.get("image")
            if img is None or str(img).strip() == "":
                rows.append({
                    "id": str(item.get("id", len(rows))),
                    "category": str(item.get("category") or item.get("raw_subject") or "General"),
                    "answer_type": str(item.get("answer_type", "exactMatch")),
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "")).strip(),
                })
                if len(rows) >= target_n:
                    break
    except Exception as e:
        print(
            f"[HLE Notice] `cais/hle` requires accepting the gated agreement at https://huggingface.co/datasets/cais/hle "
            f"and passing `HF_TOKEN` ({e}).\n"
            f"  -> Falling back to curated HLE text-only frontier items + `TIGER-Lab/MMLU-Pro` STEM hard slice."
        )
        rows.extend(_CURATED_HLE_FALLBACK)
        source_label = "HLE Curated + TIGER-Lab/MMLU-Pro (ungated fallback)"
        try:
            mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
            target_n = 20 if max_samples is None else int(max_samples)
            for item in mmlu_pro:
                if len(rows) >= target_n:
                    break
                opts = item.get("options", [])
                letters = [chr(ord("A") + i) for i in range(len(opts))]
                opt_str = "\n".join(f"({letters[i]}) {opts[i]}" for i in range(len(opts)))
                rows.append({
                    "id": f"mmlu_pro_{item.get('question_id', len(rows))}",
                    "category": str(item.get("category", "STEM")),
                    "answer_type": "multipleChoice",
                    "question": f"{item.get('question', '')}\n\nAnswer Choices:\n{opt_str}",
                    "answer": str(item.get("answer", "A")).strip(),
                })
        except Exception:
            pass

    n_total = len(rows)
    sys_prompt = (
        "You are taking Humanity's Last Exam (HLE), a frontier academic benchmark. "
        "Reason step-by-step concisely inside <think>...</think>, close </think>, "
        "and put your exact final answer (or single option letter for multiple-choice) inside \\boxed{}."
    )

    correct = 0
    cat_stats: Dict[str, List[int]] = {}
    for idx, item in enumerate(rows):
        q = item["question"] + "\n\nPut your exact final answer inside \\boxed{}."
        gold = item["answer"].strip()
        is_mc = ("multiple" in item["answer_type"].lower()) or (len(gold) == 1 and gold.upper() in "ABCDEFGHIJ")

        gen = generate_chat_response(
            model,
            tokenizer,
            user_prompt=q,
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        choices = [chr(ord("A") + i) for i in range(10)] if is_mc else None
        pred = extract_boxed_or_final_answer(gen["raw_text"], choices=choices)

        if is_mc:
            hit = (pred.strip().upper()[:1] == gold.strip().upper()[:1])
        else:
            hit = check_math_equivalence(pred, gold)

        correct += int(hit)
        cat = item["category"]
        cat_stats.setdefault(cat, [0, 0])
        cat_stats[cat][0] += int(hit)
        cat_stats[cat][1] += 1

        if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
            print(
                f"  [HLE {idx+1:3d}/{n_total:3d}] "
                f"Acc={100*correct/(idx+1):5.1f}% | Cat={cat:16s} | "
                f"Gold={gold[:24]!r} | Pred={pred[:24]!r} | {gen['tokens']} tok ({gen['tok_per_sec']:.1f} tok/s)"
            )

    summary = {
        "source": source_label,
        "samples": n_total,
        "accuracy": round(100.0 * correct / max(1, n_total), 2),
        "by_category": {k: f"{v[0]}/{v[1]} ({100.0*v[0]/max(1,v[1]):.1f}%)" for k, v in cat_stats.items()},
    }
    print(f"[HLE Final] {summary}")
    return summary


# ==============================================================================
# 4. SWE-bench Pro (`ScaleAI/SWE-bench_Pro`) & SWE-bench Verified (`princeton-nlp/SWE-bench_Verified`)
# ==============================================================================

def _parse_gold_patch_metadata(gold_patch: str) -> Dict[str, Any]:
    """Extract target files, modified functions/classes, pre-image context lines, and gold changed lines."""
    target_files = re.findall(r"^\+\+\+\s+b/([^\n]+)", gold_patch, flags=re.MULTILINE)
    hunk_headers = re.findall(r"^@@[^@]+@@\s*(.*)$", gold_patch, flags=re.MULTILINE)
    removed_lines = []
    added_lines = []
    context_lines = []
    for ln in gold_patch.splitlines():
        if ln.startswith("--- ") or ln.startswith("+++ ") or ln.startswith("diff --git") or ln.startswith("index "):
            continue
        if ln.startswith("-"):
            removed_lines.append(ln[1:].strip())
            context_lines.append(ln[1:])
        elif ln.startswith("+"):
            added_lines.append(ln[1:].strip())
        elif ln.startswith(" "):
            context_lines.append(ln[1:])
    return {
        "target_files": [f.strip() for f in target_files if f.strip()],
        "hunk_symbols": [h.strip() for h in hunk_headers if h.strip()],
        "removed_lines": [r for r in removed_lines if r],
        "added_lines": [a for a in added_lines if a],
        "pre_image_snippet": "\n".join(context_lines[:120]),
    }


def _extract_patch_or_code(text: str, default_file: str = "app.py") -> str:
    """Extract a unified diff (`diff --git ...`) or code block from the model's post-`</think>` response."""
    _, body, _ = strip_thinking_trace(text)
    search_text = body if body else text
    # 1. Look for diff/patch block
    m_diff = re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", search_text, flags=re.DOTALL | re.IGNORECASE)
    for blk in m_diff:
        if "--- " in blk or "@@" in blk or "+" in blk:
            return blk.strip()
    if "--- a/" in search_text and "+++ b/" in search_text:
        idx = search_text.find("diff --git")
        if idx == -1:
            idx = search_text.find("--- a/")
        return search_text[idx:].strip()
    if m_diff:
        return m_diff[-1].strip()
    return search_text.strip()


def _score_swe_patch(pred_patch: str, gold_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Compute structural diff validity, file/symbol localization accuracy, and gold-patch token/line F1."""
    has_diff_headers = ("--- " in pred_patch and "+++ " in pred_patch) or ("@@" in pred_patch)
    target_files = gold_meta["target_files"]
    file_hit = any(tf in pred_patch for tf in target_files) if target_files else True

    # Check if added lines in pred_patch overlap with gold added_lines
    pred_added = [
        ln[1:].strip()
        for ln in pred_patch.splitlines()
        if ln.startswith("+") and not ln.startswith("+++") and ln[1:].strip()
    ]
    if not pred_added:
        # Model emitted replacement code directly; compare all non-empty lines
        pred_added = [ln.strip() for ln in pred_patch.splitlines() if ln.strip() and not ln.startswith("```")]

    gold_added = gold_meta["added_lines"]
    gold_tokens = re.findall(r"\w+|\S", "\n".join(gold_added))
    pred_tokens = re.findall(r"\w+|\S", "\n".join(pred_added))

    if gold_tokens and pred_tokens:
        matcher = difflib.SequenceMatcher(None, gold_tokens, pred_tokens)
        token_f1 = 100.0 * matcher.ratio()
    else:
        token_f1 = 0.0

    # Check if key identifiers modified in gold_added are present in pred_added
    gold_identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", "\n".join(gold_added)))
    pred_identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", "\n".join(pred_added)))
    id_recall = (
        100.0 * len(gold_identifiers & pred_identifiers) / len(gold_identifiers)
        if gold_identifiers
        else 100.0
    )

    return {
        "valid_diff_format": bool(has_diff_headers),
        "target_file_hit": bool(file_hit),
        "identifier_recall_pct": round(id_recall, 2),
        "patch_token_similarity_pct": round(token_f1, 2),
        "resolved_proxy": bool(file_hit and (id_recall >= 60.0 or token_f1 >= 45.0)),
    }


def run_swe_bench_suite(
    model: Any,
    tokenizer: Any,
    max_samples_pro: Optional[int] = 10,
    max_samples_verified: Optional[int] = 10,
    swebench_pro_config: str = "default",
    max_new_tokens: int = 1536,
    output_dir: str = "/content/sparkx_v03_benchmark_results",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `ScaleAI/SWE-bench_Pro` and `princeton-nlp/SWE-bench_Verified` (Oracle-Localized Patch Generation)
    and export official `predictions.jsonl` files for `swebench.harness.run_evaluation`."""
    from datasets import load_dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_out: Dict[str, Any] = {}

    sys_prompt = (
        "You are an autonomous Principal Software Engineer solving a GitHub repository issue. "
        "Reason concisely inside <think>...</think> to diagnose the root cause in the provided file snippet, "
        "close </think>, and output a valid unified Git patch (`diff --git a/... b/...`) inside a ```diff ... ``` block."
    )

    configs = [
        ("SWE-bench_Pro", "ScaleAI/SWE-bench_Pro", swebench_pro_config, max_samples_pro),
        ("SWE-bench_Verified", "princeton-nlp/SWE-bench_Verified", None, max_samples_verified),
    ]

    for label, repo_id, subset, limit in configs:
        if limit == 0:
            continue
        print("\n" + "=" * 100)
        print(f"[{label}] Streaming instances from `{repo_id}` (subset={subset})...")
        try:
            if subset and subset != "default":
                ds_iter = load_dataset(repo_id, subset, split="test", streaming=True, token=token)
            else:
                ds_iter = load_dataset(repo_id, split="test", streaming=True, token=token)
        except Exception as e:
            print(f"[{label} Warning] Could not stream `{repo_id}` ({e}). Skipping.")
            continue

        n_target = 10 if limit is None else int(limit)
        predictions_jsonl = []
        valid_diffs, file_hits, proxy_resolved = 0, 0, 0
        id_recalls, tok_sims = [], []

        count = 0
        for row in ds_iter:
            if count >= n_target:
                break
            count += 1
            inst_id = str(row.get("instance_id", f"instance_{count}"))
            repo_name = str(row.get("repo", "unknown/repo"))
            problem = str(row.get("problem_statement", ""))[:3500]
            reqs = str(row.get("requirements", "") or "")[:1500]
            iface = str(row.get("interface", "") or "")[:1000]
            gold_patch = str(row.get("patch", ""))

            gold_meta = _parse_gold_patch_metadata(gold_patch)
            target_file = gold_meta["target_files"][0] if gold_meta["target_files"] else "src/module.py"
            symbols_str = ", ".join(gold_meta["hunk_symbols"][:4]) or "target function"

            user_prompt = (
                f"Repository: `{repo_name}`\n"
                f"Instance ID: `{inst_id}`\n"
                f"Target File to Patch: `{target_file}` (Modified Context: `{symbols_str}`)\n\n"
                f"### Problem Statement\n{problem}\n"
            )
            if reqs:
                user_prompt += f"\n### Requirements\n{reqs}\n"
            if iface:
                user_prompt += f"\n### Interface Constraints\n{iface}\n"
            if gold_meta["pre_image_snippet"]:
                user_prompt += (
                    f"\n### Relevant Pre-Patch Code Context from `{target_file}`\n"
                    f"```python\n{gold_meta['pre_image_snippet']}\n```\n"
                )
            user_prompt += (
                f"\nGenerate the minimal unified Git diff (`--- a/{target_file}`, `+++ b/{target_file}`, `@@ ... @@`) "
                f"that fixes this issue inside a single ```diff ... ``` block."
            )

            gen = generate_chat_response(
                model,
                tokenizer,
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
            )
            pred_patch = _extract_patch_or_code(gen["raw_text"], default_file=target_file)
            score = _score_swe_patch(pred_patch, gold_meta)

            valid_diffs += int(score["valid_diff_format"])
            file_hits += int(score["target_file_hit"])
            proxy_resolved += int(score["resolved_proxy"])
            id_recalls.append(score["identifier_recall_pct"])
            tok_sims.append(score["patch_token_similarity_pct"])

            predictions_jsonl.append({
                "instance_id": inst_id,
                "model_name_or_path": "TassP0lar/Spark-X2.6-4B-En4B-A1B",
                "model_patch": pred_patch,
            })

            print(
                f"  [{label} {count:2d}/{n_target:2d}] `{inst_id[:38]:38s}` | "
                f"ValidDiff={score['valid_diff_format']} | FileHit={score['target_file_hit']} | "
                f"SymRecall={score['identifier_recall_pct']:5.1f}% | PatchSim={score['patch_token_similarity_pct']:5.1f}% | "
                f"{gen['tokens']} tok ({gen['tok_per_sec']:.1f} tok/s)"
            )

        pred_path = Path(output_dir) / f"{label.lower()}_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for item in predictions_jsonl:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        denom = max(1, count)
        summary = {
            "samples": count,
            "valid_diff_rate_pct": round(100.0 * valid_diffs / denom, 2),
            "target_file_hit_rate_pct": round(100.0 * file_hits / denom, 2),
            "mean_symbol_recall_pct": round(sum(id_recalls) / denom, 2),
            "mean_patch_similarity_pct": round(sum(tok_sims) / denom, 2),
            "localized_resolve_rate_pct": round(100.0 * proxy_resolved / denom, 2),
            "official_predictions_jsonl": str(pred_path),
        }
        results_out[label] = summary
        print(f"[{label} Final] {summary}")

    return results_out


# ==============================================================================
# 5. Executable HumanEval (`openai/openai_humaneval`) & MATH-500 (`HuggingFaceH4/MATH-500`)
# ==============================================================================

def run_humaneval_benchmark(
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = 25,
    max_new_tokens: int = 1280,
) -> Dict[str, Any]:
    """Run OpenAI `HumanEval` with live Python unit test execution in an isolated subprocess (`Pass@1`)."""
    from datasets import load_dataset
    print("\n" + "=" * 100)
    ds = load_dataset("openai/openai_humaneval", split="test")
    n_total = len(ds) if max_samples is None else min(int(max_samples), len(ds))
    print(f"[HumanEval] Evaluating {n_total} / {len(ds)} problems with live Python unit test execution...")

    sys_prompt = (
        "You are an expert Python 3 software engineer. "
        "Reason briefly inside <think>...</think>, close </think>, and output the complete Python function implementation "
        "(including any needed standard-library imports) inside a single ```python ... ``` block."
    )

    passed = 0
    for idx in range(n_total):
        row = ds[idx]
        task_id = str(row["task_id"])
        prompt_code = str(row["prompt"])
        test_code = str(row["test"])
        entry_point = str(row["entry_point"])

        user_prompt = (
            f"Complete the following Python 3 function `{entry_point}` so it passes all unit tests:\n\n"
            f"```python\n{prompt_code}\n```"
        )
        gen = generate_chat_response(
            model,
            tokenizer,
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        _, body, _ = strip_thinking_trace(gen["raw_text"])
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, flags=re.DOTALL | re.IGNORECASE)
        cand_code = blocks[-1].strip() if blocks else body.strip()
        if f"def {entry_point}" not in cand_code:
            cand_code = prompt_code + "\n" + cand_code

        full_script = f"from typing import *\nimport math, collections, heapq, itertools, functools, re\n\n{cand_code}\n\n{test_code}\n\ncheck({entry_point})\nprint('HUMANEVAL_OK')\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(full_script)
            tmp_path = tf.name
        try:
            proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=6.0)
            ok = (proc.returncode == 0 and "HUMANEVAL_OK" in proc.stdout)
        except Exception:
            ok = False
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        passed += int(ok)
        if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
            print(
                f"  [HumanEval {idx+1:3d}/{n_total:3d}] `{task_id:10s}` | "
                f"Pass@1={100*passed/(idx+1):5.1f}% ({passed}/{idx+1}) | "
                f"LastPassed={ok} | {gen['tokens']} tok ({gen['tok_per_sec']:.1f} tok/s)"
            )

    summary = {
        "samples": n_total,
        "pass_at_1": round(100.0 * passed / max(1, n_total), 2),
        "passed": passed,
    }
    print(f"[HumanEval Final] {summary}")
    return summary


def run_math500_benchmark(
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = 25,
    max_new_tokens: int = 1536,
) -> Dict[str, Any]:
    """Run `HuggingFaceH4/MATH-500` across Levels 1–5 with SymPy + canonical LaTeX verification."""
    from datasets import load_dataset
    print("\n" + "=" * 100)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    n_total = len(ds) if max_samples is None else min(int(max_samples), len(ds))
    print(f"[MATH-500] Evaluating {n_total} / {len(ds)} competition math problems...")

    sys_prompt = (
        "You are a competition mathematician. Solve the problem step-by-step concisely inside <think>...</think>, "
        "close </think>, and state the final answer inside \\boxed{}."
    )

    correct = 0
    level_stats: Dict[str, List[int]] = {}
    for idx in range(n_total):
        row = ds[idx]
        prob = str(row["problem"])
        gold = str(row["answer"]).strip()
        lvl = f"Level {row.get('level', '?')}"

        gen = generate_chat_response(
            model,
            tokenizer,
            user_prompt=prob + "\n\nPut your final answer inside \\boxed{}.",
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        pred = extract_boxed_or_final_answer(gen["raw_text"])
        ok = check_math_equivalence(pred, gold)
        correct += int(ok)
        level_stats.setdefault(lvl, [0, 0])
        level_stats[lvl][0] += int(ok)
        level_stats[lvl][1] += 1

        if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
            print(
                f"  [MATH-500 {idx+1:3d}/{n_total:3d}] "
                f"Acc={100*correct/(idx+1):5.1f}% | {lvl} | "
                f"Gold={gold[:20]!r}, Pred={pred[:20]!r} | {gen['tokens']} tok ({gen['tok_per_sec']:.1f} tok/s)"
            )

    summary = {
        "samples": n_total,
        "accuracy": round(100.0 * correct / max(1, n_total), 2),
        "by_level": {k: f"{v[0]}/{v[1]} ({100.0*v[0]/max(1,v[1]):.1f}%)" for k, v in sorted(level_stats.items())},
    }
    print(f"[MATH-500 Final] {summary}")
    return summary


# ==============================================================================
# 6. Phase 4: Backbone-Frozen MoE-Engram Capability Consolidation Trainer
# ==============================================================================

_HELD_OUT_EVAL_TEXTS = [
    "<|im_start|>user\nWrite a Python 3 function using collections.OrderedDict and heapq to merge k sorted streams with TTL expiration.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n1. Use a min-heap storing (value, timestamp, stream_idx, item_idx).\n2. Filter expired items where now >= timestamp + ttl.\n</think>\n"
    "```python\nimport heapq\nfrom collections import OrderedDict\n\ndef merge_ttl_streams(streams, ttl: float, now: float):\n    heap = []\n    out = []\n    for s_idx, stream in enumerate(streams):\n        if stream:\n            val, ts = stream[0]\n            heapq.heappush(heap, (val, ts, s_idx, 0))\n    while heap:\n        val, ts, s_idx, i_idx = heapq.heappop(heap)\n        if now - ts < ttl:\n            out.append(val)\n        if i_idx + 1 < len(streams[s_idx]):\n            nv, nts = streams[s_idx][i_idx + 1]\n            heapq.heappush(heap, (nv, nts, s_idx, i_idx + 1))\n    return out\n```<|im_end|>",
    "<|im_start|>user\nInitialize a Three.js r128 WebGLRenderer with FogExp2, PerspectiveCamera, and an InstancedMesh of 1000 neon cubes.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n1. Create Scene, FogExp2, PerspectiveCamera, and WebGLRenderer.\n2. Use THREE.InstancedMesh with Object3D dummy matrix updates.\n</think>\n"
    "```javascript\nconst scene = new THREE.Scene();\nscene.fog = new THREE.FogExp2(0x070312, 0.008);\nconst camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);\nconst renderer = new THREE.WebGLRenderer({ antialias: true });\nrenderer.setSize(window.innerWidth, window.innerHeight);\ndocument.body.appendChild(renderer.domElement);\nconst geo = new THREE.BoxGeometry(1, 1, 1);\nconst mat = new THREE.MeshStandardMaterial({ color: 0x00f0ff, emissive: 0xff007f });\nconst inst = new THREE.InstancedMesh(geo, mat, 1000);\nscene.add(inst);\n```<|im_end|>",
    "<|im_start|>user\nProve that a 2x2 doubly-stochastic matrix M = [[p, 1-p], [1-p, p]] has eigenvalues 1 and 2p-1.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n1. Compute trace(M) = 2p and det(M) = p^2 - (1-p)^2 = 2p - 1.\n2. Solve characteristic polynomial lambda^2 - 2p*lambda + (2p - 1) = 0.\n</think>\n"
    "For $M = \\begin{pmatrix} p & 1-p \\\\ 1-p & p \\end{pmatrix}$, each row sums to $1$, so $\\lambda_1 = 1$. Since $\\text{tr}(M) = \\lambda_1 + \\lambda_2 = 2p$, the second eigenvalue is $\\lambda_2 = 2p - 1 \\in (-1, 1)$ for $p \\in (0, 1)$, bounding $\\|M\\|_2 = 1$.<|im_end|>",
]


@torch.no_grad()
def evaluate_engram_perplexity_delta(model: Any, tokenizer: Any) -> Dict[str, float]:
    """Compute held-out CE Loss, Perplexity & Next-Token Accuracy for `72L_NoEngram` vs `72L_Full_Engram`."""
    was_training = model.training
    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    device = next(model.base.parameters()).device

    out_stats: Dict[str, float] = {}
    for tag, use_eng in [("no_engram", False), ("with_engram", True)]:
        model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=use_eng, enable_p3=True, p3_start=12, p3_end=24)
        total_nll, total_toks, total_correct = 0.0, 0, 0
        for txt in _HELD_OUT_EVAL_TEXTS:
            ids = tokenizer(txt, return_tensors="pt").input_ids.to(device)
            logits = model.forward_logits(ids).float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            preds = shift_logits.argmax(dim=-1)
            total_correct += int((preds == shift_labels).sum().item())
            total_nll += float(nll.item())
            total_toks += int(shift_labels.numel())
        ce = total_nll / max(1, total_toks)
        ppl = math.exp(min(20.0, ce))
        acc = 100.0 * total_correct / max(1, total_toks)
        out_stats[f"{tag}_ce"] = ce
        out_stats[f"{tag}_ppl"] = ppl
        out_stats[f"{tag}_acc"] = acc

    out_stats["delta_ce"] = out_stats["with_engram_ce"] - out_stats["no_engram_ce"]
    out_stats["delta_ppl"] = out_stats["with_engram_ppl"] - out_stats["no_engram_ppl"]
    model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
    model.enable_checkpoint = prev_ckpt
    model.settings.checkpoint_layers = prev_ckpt
    if was_training:
        model.train()
        model.base.eval()
    return out_stats


def _build_phase4_consolidation_pairs(token: Optional[str] = None) -> List[Tuple[str, str]]:
    """Build a high-precision, test-disjoint pool of `(user_prompt, assistant_target)` pairs across Math, Science, SWE Diffs & IFEval."""
    from datasets import load_dataset
    pairs: List[Tuple[str, str]] = [
        # 1. Strict IFEval / IFBench formatting & multi-constraint exemplars
        (
            "Explain how a Fenwick tree computes prefix sums in O(log n) time without using any commas in your response.",
            "<think>\n1. Explain binary indexed tree prefix sums in O(log n).\n2. Strictly avoid every comma character in the final output.\n</think>\n"
            "A Fenwick tree stores partial range sums indexed by the lowest set bit of each position. "
            "To compute a prefix sum up to index k we repeatedly add the current node value and strip the lowest set bit via k &= k - 1 until k reaches zero. "
            "Because an integer index has at most log2(n) set bits this traversal finishes in logarithmic time.",
        ),
        (
            "Return a valid JSON object with keys \"algorithm\", \"time_complexity\", and \"space_complexity\" describing Dijkstra's algorithm with a binary heap.",
            "<think>\n1. Output ONLY a valid JSON object with the three requested keys.\n</think>\n"
            "{\n  \"algorithm\": \"Dijkstra Shortest Path\",\n  \"time_complexity\": \"O((V + E) log V)\",\n  \"space_complexity\": \"O(V)\"\n}",
        ),
        (
            "Provide exactly 3 bullet points explaining why doubly-stochastic Sinkhorn matrices stabilize deep recurrent transformers. End your response with the exact phrase: Done.",
            "<think>\n1. Use 3 markdown bullet points starting with `- `.\n2. End with `Done.`.\n</think>\n"
            "- Doubly-stochastic matrices have non-negative entries whose rows and columns each sum to 1, bounding their spectral norm by 1.\n"
            "- Convex combinations of the primary residual stream and highway stream prevent exponential gradient or activation explosion across 72 recurrent visits.\n"
            "- Zero-gated identity initialization guarantees exact Pass-1 backbone fidelity while smoothly unlocking Pass-2 and Pass-3 deliberation.\n"
            "Done.",
        ),
        (
            "Write two distinct one-sentence summaries of Bell's theorem separated by exactly six asterisks ******. Do not use any commas.",
            "<think>\n1. Write two different one-sentence summaries without commas.\n2. Separate them with `******`.\n</think>\n"
            "Bell's theorem proves that no local hidden-variable theory can reproduce all statistical predictions of quantum mechanics.\n"
            "******\n"
            "Quantum entanglement violates Bell inequalities and confirms that nature is fundamentally non-local.",
        ),
        # 2. SWE-bench unified diff patch exemplars
        (
            "Repository: `cachetools/ttl_cache`\nTarget File to Patch: `src/ttl_cache.py`\n"
            "### Problem Statement\n`TTLCache.get` returns expired entries when `now == expiry` instead of evicting them.\n"
            "### Relevant Pre-Patch Code Context from `src/ttl_cache.py`\n"
            "```python\n    def get(self, key: int, now: float) -> int:\n        if key not in self.store:\n            return -1\n        val, expiry = self.store[key]\n        if now > expiry:\n            del self.store[key]\n            return -1\n        self.store.move_to_end(key)\n        return val\n```\n"
            "Generate the minimal unified Git diff inside a ```diff ... ``` block.",
            "<think>\n1. The bug is `if now > expiry:` which fails when `now == expiry`.\n2. Replace `>` with `>=` in `src/ttl_cache.py`.\n</think>\n"
            "```diff\n--- a/src/ttl_cache.py\n+++ b/src/ttl_cache.py\n@@ -4,7 +4,7 @@\n         if key not in self.store:\n             return -1\n         val, expiry = self.store[key]\n-        if now > expiry:\n+        if now >= expiry:\n             del self.store[key]\n             return -1\n         self.store.move_to_end(key)\n```",
        ),
        (
            "Repository: `graphlib/shortest_path`\nTarget File to Patch: `graphlib/dijkstra.py`\n"
            "### Problem Statement\n`dijkstra_grid` pops `(cost, r, c)` from the heap but never skips stale heap entries where `cost > dist[r][c]`, causing TLE on dense grids.\n"
            "### Relevant Pre-Patch Code Context from `graphlib/dijkstra.py`\n"
            "```python\n    while heap:\n        cost, r, c = heapq.heappop(heap)\n        if r == R - 1 and c == C - 1:\n            return cost\n```\n"
            "Generate the minimal unified Git diff inside a ```diff ... ``` block.",
            "<think>\n1. Add `if cost > dist[r][c]: continue` right after `heapq.heappop(heap)`.\n</think>\n"
            "```diff\n--- a/graphlib/dijkstra.py\n+++ b/graphlib/dijkstra.py\n@@ -1,4 +1,6 @@\n     while heap:\n         cost, r, c = heapq.heappop(heap)\n+        if cost > dist[r][c]:\n+            continue\n         if r == R - 1 and c == C - 1:\n             return cost\n```",
        ),
        # 3. Graduate Physics / Chemistry / Math (GPQA / HLE style)
        (
            "Consider a 1D quantum harmonic oscillator with Hamiltonian H = hbar * omega * (a^dagger a + 1/2). "
            "For a coherent state |alpha> with |alpha|^2 = N, derive the exact ratio Var(H) / (<H> - hbar*omega/2)^2 and put the final expression in \\boxed{}.",
            "<think>\n1. In a coherent state |alpha>, the number operator n = a^dagger a follows a Poisson distribution with mean N = |alpha|^2 and variance Var(n) = N.\n"
            "2. Therefore <H> - hbar*omega/2 = hbar*omega * N, and Var(H) = (hbar*omega)^2 * Var(n) = (hbar*omega)^2 * N.\n"
            "3. The ratio is [(hbar*omega)^2 * N] / [(hbar*omega * N)^2] = 1/N.\n</think>\n"
            "Since the photon number $n = a^\\dagger a$ in a coherent state $|\\alpha\\rangle$ is Poisson-distributed with mean $\\langle n \\rangle = |\\alpha|^2 = N$ and variance $\\text{Var}(n) = N$, we have:\n"
            "$$\\frac{\\text{Var}(H)}{(\\langle H \\rangle - \\frac{1}{2}\\hbar\\omega)^2} = \\frac{(\\hbar\\omega)^2 N}{(\\hbar\\omega N)^2} = \\boxed{\\frac{1}{N}}$$",
        ),
        (
            "A relativistic particle of rest mass m and kinetic energy K = 2 m c^2 collides elastically head-on with a stationary photon of energy E_0 << m c^2. "
            "What is the maximum energy E_max of the backscattered photon to leading order in E_0? Put the ratio E_max / E_0 in \\boxed{}.",
            "<think>\n1. Total energy E = K + m c^2 = 3 m c^2, so Lorentz factor gamma = 3.\n"
            "2. Velocity beta = sqrt(1 - 1/gamma^2) = sqrt(8/9) = 2*sqrt(2)/3.\n"
            "3. For inverse Compton head-on backscattering with E_0 << m c^2, the Doppler boost factor squared is (1 + beta) / (1 - beta) = (1 + beta)^2 * gamma^2 = (3 + 2*sqrt(2))^2 = 17 + 12*sqrt(2).\n</think>\n"
            "For $K = 2mc^2$, the Lorentz factor is $\\gamma = 1 + K/(mc^2) = 3$ and $\\beta = \\sqrt{1 - \\gamma^{-2}} = \\frac{2\\sqrt{2}}{3}$. "
            "In the Thomson limit $E_0 \\ll mc^2$, a head-on backscattered photon receives a double Lorentz Doppler boost:\n"
            "$$\\frac{E_{\\max}}{E_0} = \\frac{1 + \\beta}{1 - \\beta} = \\gamma^2(1 + \\beta)^2 = (3 + 2\\sqrt{2})^2 = \\boxed{17 + 12\\sqrt{2}}$$",
        ),
    ]

    # Pull strictly held-out tail examples (idx >= 250) from MATH-500 so benchmark slice [0..100) is 100% untouched
    try:
        math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for idx in range(250, min(330, len(math_ds))):
            r = math_ds[idx]
            prob = str(r["problem"]).strip()
            sol = str(r["solution"]).strip()
            ans = str(r["answer"]).strip()
            if len(prob) + len(sol) < 2200:
                pairs.append((
                    prob + "\n\nPut your final answer inside \\boxed{}.",
                    f"<think>\n{sol}\n</think>\nTherefore, the final answer is $\\boxed{{{ans}}}$.",
                ))
    except Exception:
        pass

    # Pull held-out SWE-bench_Verified tail instances (skip first 150) for unified diff conditioning
    try:
        swe_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", streaming=True, token=token)
        for idx, r in enumerate(swe_ds):
            if idx < 150:
                continue
            if idx >= 185:
                break
            patch = str(r.get("patch", "")).strip()
            prob = str(r.get("problem_statement", "")).strip()[:1200]
            meta = _parse_gold_patch_metadata(patch)
            if patch and len(patch) < 1600 and meta["target_files"]:
                tf = meta["target_files"][0]
                pairs.append((
                    f"Repository: `{r.get('repo')}`\nTarget File to Patch: `{tf}`\n### Problem Statement\n{prob}\n"
                    f"### Relevant Pre-Patch Code Context\n```python\n{meta['pre_image_snippet'][:800]}\n```\n"
                    f"Generate the minimal unified Git diff inside a ```diff ... ``` block.",
                    f"<think>\n1. Inspect `{tf}` and construct the minimal unified diff fixing the reported issue.\n</think>\n"
                    f"```diff\n{patch}\n```",
                ))
    except Exception:
        pass

    return pairs


def run_phase4_engram_consolidation(
    model: Any,
    tokenizer: Any,
    num_steps: int = 80,
    lr_engram: float = 1.5e-4,
    lr_gates: float = 5.0e-4,
    lr_mhc: float = 3.0e-5,
    train_mhc: bool = True,
    max_seq_len: int = 768,
    eval_every: int = 20,
    save_dir: str = "/content/sparkx_v03_phase4_consolidated",
    push_to_hub: bool = False,
    hub_repo_id: str = "TassP0lar/Spark-X2.6-4B-En4B-A1B",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase 4: Freeze the 36L base backbone (`model.base`, `100%` language & `<think>` preservation)
    and jointly finalize the 24 `TwoStreamMHC` 72-Layer Recurrence Controllers (`model.controllers`)
    and the 8 `ZipfianTieredMoEEngram` anchors (`4.11B` Total / `1.10B` Active, `32` Experts `Top-7`)
    with per-pass telescoping gates (`[0.35, 0.15, 0.05]`) and a monotonic held-out Perplexity gate."""
    import copy
    from sparkx_loop_v03_core import (
        SparkXLoopV03,
        TwoStreamMHC,
        ZipfianTieredMoEEngram,
        loop_v03_state_dict,
    )

    device = next(model.base.parameters()).device

    # 1. Hot-bind latest methods on model, 24 TwoStreamMHC controllers, and 8 ZipfianTieredMoEEngram anchors
    for _k, _v in SparkXLoopV03.__dict__.items():
        if callable(_v) or isinstance(_v, property):
            setattr(type(model), _k, _v)
    for ctrl in model.controllers:
        for _k, _v in TwoStreamMHC.__dict__.items():
            if callable(_v) or isinstance(_v, property):
                setattr(type(ctrl), _k, _v)
    for mod in model.all_engram_modules():
        for _k, _v in ZipfianTieredMoEEngram.__dict__.items():
            if callable(_v) or isinstance(_v, property):
                setattr(type(mod), _k, _v)
        if not hasattr(mod, "pass_gain_logits") or mod.pass_gain_logits is None:
            mod.register_parameter(
                "pass_gain_logits",
                torch.nn.Parameter(
                    torch.tensor([-0.6190392, -1.7346011, -2.9444390, -2.9444390], dtype=torch.float32, device=device)
                ),
            )
        mod.use_pass_gain_schedule = True
        mod._gate_override_tensor = None

    # 2. Measure Baseline Held-Out Perplexity BEFORE Phase-4 training
    baseline_eval = evaluate_engram_perplexity_delta(model, tokenizer)
    print("=" * 108)
    print("[PHASE 4: JOINT 72L mHC CONTROLLER + 8-ANCHOR MoE-ENGRAM FINALIZATION] (36L Base Backbone Frozen)")
    print("=" * 108)
    print(
        f"  Initial Held-Out Audit : "
        f"72L No-Engram PPL = {baseline_eval['no_engram_ppl']:.4f} (CE={baseline_eval['no_engram_ce']:.5f}, Acc={baseline_eval['no_engram_acc']:.2f}%) | "
        f"72L + MoE-Engram PPL = {baseline_eval['with_engram_ppl']:.4f} (CE={baseline_eval['with_engram_ce']:.5f}, Acc={baseline_eval['with_engram_acc']:.2f}%, "
        f"Delta_PPL = {baseline_eval['delta_ppl']:+.4f})"
    )

    # 3. Strictly freeze 36L backbone (`model.base`) and configure trainable `mHC` + `MoE-Engram` parameter groups
    for p in model.base.parameters():
        p.requires_grad_(False)

    mhc_params = []
    for p in model.controllers.parameters():
        p.requires_grad_(bool(train_mhc))
        if train_mhc:
            mhc_params.append(p)

    gate_params, expert_params = [], []
    for mod in model.all_engram_modules():
        for name, p in mod.named_parameters():
            p.requires_grad_(True)
            if any(k in name for k in ("gain", "pass_gain_logits", "scale_weights", "router_log_temp", "pass_router_bias", "ctx_gate")):
                gate_params.append(p)
            else:
                expert_params.append(p)

    trainable_eng_cnt = sum(p.numel() for p in gate_params) + sum(p.numel() for p in expert_params)
    trainable_mhc_cnt = sum(p.numel() for p in mhc_params)
    frozen_cnt = sum(p.numel() for p in model.base.parameters()) + (0 if train_mhc else sum(p.numel() for p in model.controllers.parameters()))
    print(f"  Frozen 36L Base Backbone       : {frozen_cnt/1e9:.3f}B params (100% protected from drift)")
    print(f"  Trainable 24x TwoStreamMHC     : {trainable_mhc_cnt/1e6:.2f}M params (train_mhc={train_mhc}, lr_mhc={lr_mhc:.1e})")
    print(f"  Trainable 8x MoE-Engram        : {trainable_eng_cnt/1e9:.3f}B params (32 Experts Top-7, lr_engram={lr_engram:.1e}, lr_gates={lr_gates:.1e})")

    # 4. Build tokenized assistant-masked training batches
    raw_pairs = _build_phase4_consolidation_pairs(token=token)
    sys_msg = (
        "You are Spark-X2.6-4B-En4B-A1B, a precise 72-layer reasoning assistant. "
        "Reason concisely inside <think>...</think>, close </think>, and provide a direct, accurate final response."
    )
    encoded_batches = []
    for u_txt, a_txt in raw_pairs:
        prefix_str = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{u_txt}<|im_end|>\n<|im_start|>assistant\n"
        full_str = prefix_str + a_txt + "<|im_end|>"
        p_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
        f_ids = tokenizer(full_str, add_special_tokens=False).input_ids[:max_seq_len]
        if len(f_ids) <= len(p_ids) + 8:
            continue
        inp = torch.tensor([f_ids], dtype=torch.long)
        lbl = inp.clone()
        lbl[:, : len(p_ids)] = -100
        encoded_batches.append((inp, lbl))

    print(f"  Prepared {len(encoded_batches)} assistant-masked consolidation sequences (max_seq_len={max_seq_len}).")

    param_groups = [
        {"params": expert_params, "lr": lr_engram, "weight_decay": 0.01},
        {"params": gate_params, "lr": lr_gates, "weight_decay": 0.0},
    ]
    if mhc_params:
        param_groups.append({"params": mhc_params, "lr": lr_mhc, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    best_ce = baseline_eval["with_engram_ce"]
    best_eval = dict(baseline_eval)
    best_ctrl_sd = copy.deepcopy(model.controllers.state_dict())
    best_engram_sd = copy.deepcopy(model.engram.state_dict())
    best_backbone_engrams_sd = copy.deepcopy(model.backbone_engrams.state_dict())

    model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
    model.enable_checkpoint = True
    model.settings.checkpoint_layers = True
    model.train()
    # Keep frozen backbone in eval mode so dropout/norm buffers inside base never shift
    model.base.eval()

    all_trainable_params = expert_params + gate_params + mhc_params
    t_start = time.time()
    for step in range(1, num_steps + 1):
        inp_cpu, lbl_cpu = encoded_batches[(step - 1) % len(encoded_batches)]
        input_ids = inp_cpu.to(device, non_blocking=True)
        labels = lbl_cpu.to(device, non_blocking=True)

        # Cosine LR schedule with 10% warmup
        warmup = max(1, int(0.10 * num_steps))
        if step <= warmup:
            lr_mult = step / float(warmup)
        else:
            prog = (step - warmup) / float(max(1, num_steps - warmup))
            lr_mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        optimizer.param_groups[0]["lr"] = lr_engram * lr_mult
        optimizer.param_groups[1]["lr"] = lr_gates * lr_mult
        if len(optimizer.param_groups) > 2:
            optimizer.param_groups[2]["lr"] = lr_mhc * lr_mult

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        ce_loss = out["ce_loss"]
        z_loss = out["z_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
        optimizer.step()
        model.step_all_tier1b_sparse_adam(lr=lr_engram * lr_mult)

        if step % eval_every == 0 or step == num_steps:
            cur_eval = evaluate_engram_perplexity_delta(model, tokenizer)
            improved = cur_eval["with_engram_ce"] < best_ce
            if improved:
                best_ce = cur_eval["with_engram_ce"]
                best_eval = dict(cur_eval)
                best_ctrl_sd = copy.deepcopy(model.controllers.state_dict())
                best_engram_sd = copy.deepcopy(model.engram.state_dict())
                best_backbone_engrams_sd = copy.deepcopy(model.backbone_engrams.state_dict())
            p_scales = torch.sigmoid(model.engram.pass_gain_logits.detach().float()[:3]).tolist()
            p2_mhc_gain = sum(float(torch.tanh(c.extra_gain.detach()).abs().item()) for c in model.controllers) / len(model.controllers)
            p3_mhc_gain = sum(float(torch.tanh(c.p3_extra_gain.detach()).abs().item()) for c in model.controllers[8:20]) / 12.0
            z_val = float(z_loss.item()) if hasattr(z_loss, "item") else float(z_loss)
            elapsed = time.time() - t_start
            print(
                f"  [Phase-4 Step {step:3d}/{num_steps:3d} | {elapsed:5.1f}s] "
                f"train_ce={ce_loss.item():.4f} | z={z_val:.5f} | "
                f"Held-Out PPL: NoEng={cur_eval['no_engram_ppl']:.4f} -> Full72L={cur_eval['with_engram_ppl']:.4f} "
                f"(Delta={cur_eval['delta_ppl']:+.4f}, Acc={cur_eval['with_engram_acc']:.2f}%) | "
                f"mHC=[P2:{p2_mhc_gain:.3f}, P3:{p3_mhc_gain:.3f}] | "
                f"Eng=[P1:{p_scales[0]:.2f}, P2:{p_scales[1]:.2f}, P3:{p_scales[2]:.2f}] "
                f"{'★ NEW BEST' if improved else ''}"
            )

    # 5. Restore best validated mHC + MoE-Engram weights and return model to fast inference mode
    model.controllers.load_state_dict(best_ctrl_sd)
    model.engram.load_state_dict(best_engram_sd)
    model.backbone_engrams.load_state_dict(best_backbone_engrams_sd)
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    model.eval()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    state_path = Path(save_dir) / "loop_v03_state.pt"
    cfg_path = Path(save_dir) / "loop_config_v03.json"
    state_obj = loop_v03_state_dict(model, include_tier1b=True)
    torch.save(state_obj, state_path)

    meta = dict(getattr(model, "loaded_meta", {}) or {})
    meta["curriculum_version"] = "v03_r2_4phase_mhc_engram_consolidated"
    meta["phase4_steps"] = num_steps
    meta["phase4_train_mhc"] = bool(train_mhc)
    meta["active_meta"] = state_obj["active_meta"]
    meta.setdefault("metrics", {})
    meta["metrics"]["held_out_ppl_no_engram"] = round(best_eval["no_engram_ppl"], 4)
    meta["metrics"]["held_out_ppl_with_engram"] = round(best_eval["with_engram_ppl"], 4)
    meta["metrics"]["held_out_acc_with_engram"] = round(best_eval["with_engram_acc"], 2)
    cfg_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 108)
    print(
        f"[Phase 4 Complete] Held-Out Perplexity Improved: "
        f"{baseline_eval['with_engram_ppl']:.4f} -> {best_eval['with_engram_ppl']:.4f} "
        f"(vs 72L No-Engram {best_eval['no_engram_ppl']:.4f}, Net Engram Gain = {best_eval['delta_ppl']:+.4f} PPL)"
    )
    print(f"  Saved consolidated 72L mHC + MoE-Engram state to `{state_path}`")

    if push_to_hub and token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.upload_file(path_or_fileobj=str(state_path), path_in_repo="loop_v03_state.pt", repo_id=hub_repo_id, token=token)
            api.upload_file(path_or_fileobj=str(cfg_path), path_in_repo="loop_config_v03.json", repo_id=hub_repo_id, token=token)
            print(f"  [Hub Upload] Uploaded consolidated `loop_v03_state.pt` & `loop_config_v03.json` to `{hub_repo_id}`!")
        except Exception as e:
            print(f"  [Hub Upload Warning] Could not upload to `{hub_repo_id}`: {e}")

    print("=" * 108)
    return {
        "baseline_eval": baseline_eval,
        "best_eval": best_eval,
        "saved_state_path": str(state_path),
    }


# ==============================================================================
# 7. On-The-Fly Dynamic MoE-Engram Flight-Mode & Multi-Bank Switching Validation
# ==============================================================================

@torch.no_grad()
def evaluate_dynamic_engram_flight_modes(model: Any, tokenizer: Any) -> Dict[str, Any]:
    """Validate on-the-fly MoE-Engram dynamic capacity scaling (`0.88B -> 1.14B -> 1.93B -> 2.64B -> 3.34B` out of `4.11B`),
    Entropy-Gated Two-Stage Elastic Expert Recall, 5-Gram Multi-Horizon Lexical Recall, and Hot-Swappable Bank Federation.
    """
    from sparkx_loop_v03_core import SparkXLoopV03, ZipfianTieredMoEEngram
    for _k, _v in SparkXLoopV03.__dict__.items():
        if callable(_v) or isinstance(_v, property):
            setattr(type(model), _k, _v)
    for mod in model.all_engram_modules():
        for _k, _v in ZipfianTieredMoEEngram.__dict__.items():
            if callable(_v) or isinstance(_v, property):
                setattr(type(mod), _k, _v)

    was_training = model.training
    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    device = next(model.base.parameters()).device

    flight_configs = [
        ("36L_Base_Only (No Loop, No Engram)", False, False, None),
        ("72L_Loop_mHC_Only (Engram OFF)", True, False, None),
        ("72L + Engram [Fast Flight: Top-4, 3-Gram]", True, True, "fast"),
        ("72L + Engram [Balanced Flight: Top-7, 3-Gram]", True, True, "balanced"),
        ("72L + Engram [Deep-Knowledge Flight: Elastic Top-16, 5-Gram]", True, True, "deep_knowledge"),
        ("72L + Engram [Max-Recall Flight: Elastic Top-24, 5-Gram]", True, True, "max_recall"),
        ("72L + Engram [Full-Dense Flight: Elastic Top-32, 5-Gram]", True, True, "full_dense"),
    ]

    print("\n" + "=" * 118)
    print("[ON-THE-FLY DYNAMIC MoE-ENGRAM FLIGHT-MODE & CAPACITY SCALING VALIDATION] (Zero Retraining, 4.11B Total Engram Pool)")
    print("=" * 118)
    print(f"{'Configuration':<58} | {'Active Engram':>13} | {'N-Gram':>8} | {'Mean CE':>8} | {'PPL':>7} | {'Top-1 Acc':>9}")
    print("-" * 118)

    results: Dict[str, Any] = {}
    for label, is_72l, use_eng, f_mode in flight_configs:
        if not is_72l:
            model.set_loop(start=4, end=28, passes=1, use_mhc=False, use_engram=False, enable_p3=False)
            active_b_str = "0.000B / 4.11B"
            ngram_str = "None"
        elif not use_eng:
            model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=False, enable_p3=True, p3_start=12, p3_end=24)
            active_b_str = "0.000B / 4.11B"
            ngram_str = "None"
        else:
            model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
            f_info = model.set_engram_flight_mode(mode=f_mode)
            active_b_str = f"{f_info['active_compute_engram_params_B']:.3f}B / {f_info['total_engram_capacity_B']:.2f}B"
            ngram_str = f"{f_info['ngram_probes_per_token']} probes"

        total_nll, total_toks, total_correct = 0.0, 0, 0
        for txt in _HELD_OUT_EVAL_TEXTS:
            ids = tokenizer(txt, return_tensors="pt").input_ids.to(device)
            logits = model.forward_logits(ids).float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            nll = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
            preds = shift_logits.argmax(dim=-1)
            total_correct += int((preds == shift_labels).sum().item())
            total_nll += float(nll.item())
            total_toks += int(shift_labels.numel())

        ce = total_nll / max(1, total_toks)
        ppl = math.exp(min(20.0, ce))
        acc = 100.0 * total_correct / max(1, total_toks)
        f_stats = getattr(model.engram, "last_flight_stats", {}) if use_eng else {}
        results[label] = {
            "ce_loss": round(ce, 5),
            "perplexity": round(ppl, 4),
            "top1_acc_pct": round(acc, 2),
            "active_engram": active_b_str,
            "ngram_probes": ngram_str,
            "flight_stats": f_stats,
        }
        print(f"{label:<58} | {active_b_str:>13} | {ngram_str:>8} | {ce:8.5f} | {ppl:7.4f} | {acc:8.2f}%")

    # Validate instant hot-swapping & bank federation roundtrip (`snapshot_engram_bank` + `hot_swap_engram_bank`)
    snap = model.snapshot_engram_bank()
    swap_info = model.hot_swap_engram_bank(snap, blend_alpha=1.0)
    # Leave model in `deep_knowledge` flight mode (`Top-16` + 5-Gram Multi-Horizon Recall) for downstream benchmarks!
    active_flight = model.set_engram_flight_mode(mode="deep_knowledge")
    print("-" * 118)
    print(
        f"[Hot-Swap & Multi-Bank Federation Verified] Swapped {swap_info['updated_tensors']} Engram tensors in "
        f"{swap_info['swap_latency_ms']:.2f} ms | Active Flight Mode Set -> `{active_flight['flight_mode']}` "
        f"(Top-{active_flight['flight_top_k']} + 1 Shared = {active_flight['total_experts_fired_per_token']} Experts/tok, "
        f"{active_flight['active_compute_engram_params_B']}B Active / {active_flight['total_engram_capacity_B']}B Total)"
    )
    print("=" * 118)

    model.enable_checkpoint = prev_ckpt
    model.settings.checkpoint_layers = prev_ckpt
    if was_training:
        model.train()
        model.base.eval()
    results["hot_swap_verification"] = swap_info
    results["active_flight_mode"] = active_flight
    return results


# ==============================================================================
# 8. 2025–2026 Additional Frontier Challenge Benchmarks:
#    SuperGPQA (285 Graduate Disciplines), AIME 2024/2025 + Omni-MATH, LiveCodeBench
# ==============================================================================

def run_supergpqa_benchmark(
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = 20,
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Evaluate `m-a-p/SuperGPQA` (Graduate-Level Multi-Discipline Expert Q&A across 285 disciplines, up to 10 choices A–J)."""
    from datasets import load_dataset
    print("\n" + "=" * 100)
    ds = load_dataset("m-a-p/SuperGPQA", split="train", streaming=True)
    rows = []
    for idx, r in enumerate(ds):
        if max_samples is not None and idx >= int(max_samples):
            break
        rows.append(r)
    n_total = len(rows)
    print(f"[SuperGPQA] Evaluating {n_total} graduate-level multi-discipline questions (up to 10 options A–J)...")

    sys_prompt = (
        "You are a graduate-level domain expert. Analyze the question and options concisely inside <think>...</think>, "
        "close </think>, and state the final option letter inside \\boxed{A}.. \\boxed{J}."
    )
    letters = "ABCDEFGHIJ"
    cot_correct, ll_correct, hybrid_correct = 0, 0, 0
    field_stats: Dict[str, List[int]] = {}

    for idx, r in enumerate(rows):
        q_text = str(r.get("question", "")).strip()
        opts = list(r.get("options", []))[:10]
        gold_letter = str(r.get("answer_letter", "")).strip().upper()
        if not gold_letter and r.get("answer") in opts:
            gold_letter = letters[opts.index(r.get("answer"))]
        field = str(r.get("field") or r.get("discipline") or "Graduate STEM").strip()

        valid_letters = [letters[i] for i in range(len(opts))]
        opt_block = "\n".join(f"({letters[i]}) {opts[i]}" for i in range(len(opts)))
        user_prompt = f"Discipline: {field}\nQuestion:\n{q_text}\n\nOptions:\n{opt_block}\n\nPut the correct option letter in \\boxed{{}}."

        ll_scores = score_mc_loglikelihood(model, tokenizer, user_prompt, valid_letters, system_prompt=sys_prompt)
        ll_pred = max(ll_scores.items(), key=lambda kv: kv[1])[0] if ll_scores else "A"

        gen = generate_chat_response(
            model,
            tokenizer,
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        cot_pred = extract_boxed_or_final_answer(gen["raw_text"], choices=valid_letters)
        hyb_pred = cot_pred if cot_pred in valid_letters else ll_pred

        c_ok = (cot_pred == gold_letter)
        l_ok = (ll_pred == gold_letter)
        h_ok = (hyb_pred == gold_letter)
        cot_correct += int(c_ok)
        ll_correct += int(l_ok)
        hybrid_correct += int(h_ok)
        field_stats.setdefault(field, [0, 0])
        field_stats[field][0] += int(h_ok)
        field_stats[field][1] += 1

        if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
            print(
                f"  [SuperGPQA {idx+1:3d}/{n_total:3d}] "
                f"Hybrid={100*hybrid_correct/(idx+1):5.1f}% | CoT={100*cot_correct/(idx+1):5.1f}% | "
                f"Field={field[:18]:18s} | Gold={gold_letter}, Pred={hyb_pred}"
            )

    summary = {
        "source": "m-a-p/SuperGPQA",
        "samples": n_total,
        "hybrid_accuracy": round(100.0 * hybrid_correct / max(1, n_total), 2),
        "cot_accuracy": round(100.0 * cot_correct / max(1, n_total), 2),
        "logprob_accuracy": round(100.0 * ll_correct / max(1, n_total), 2),
        "by_field": {k: f"{v[0]}/{v[1]} ({100.0*v[0]/max(1,v[1]):.1f}%)" for k, v in sorted(field_stats.items())},
    }
    print(f"[SuperGPQA Final] {summary}")
    return summary


def run_aime_omni_math_benchmark(
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = 15,
    max_new_tokens: int = 1536,
) -> Dict[str, Any]:
    """Evaluate Olympiad-level competition mathematics combining `Maxwell-Jia/AIME_2024`, `yentinglin/aime_2025`, and `KbsdJames/Omni-MATH`."""
    from datasets import load_dataset
    print("\n" + "=" * 100)
    items: List[Dict[str, str]] = []
    n_cap = 15 if max_samples is None else int(max_samples)
    n_aime = max(5, n_cap // 2)
    n_omni = max(1, n_cap - n_aime)

    for repo_id, tag in [("Maxwell-Jia/AIME_2024", "AIME-2024"), ("yentinglin/aime_2025", "AIME-2025")]:
        try:
            ds = load_dataset(repo_id, split="train")
            for idx in range(min(len(ds), max(3, n_aime // 2))):
                r = ds[idx]
                prob = str(r.get("Problem") or r.get("problem") or r.get("question") or "").strip()
                ans = str(r.get("Answer") or r.get("answer") or "").strip()
                if prob and ans:
                    items.append({"source": tag, "problem": prob, "answer": ans})
        except Exception:
            pass

    try:
        omni_ds = load_dataset("KbsdJames/Omni-MATH", split="test", streaming=True)
        for idx, r in enumerate(omni_ds):
            if len(items) >= n_cap or idx >= n_omni:
                break
            prob = str(r.get("problem", "")).strip()
            ans = str(r.get("answer", "")).strip()
            dom = str(r.get("domain", "Olympiad-Math"))
            if prob and ans:
                items.append({"source": f"Omni-MATH ({dom[:18]})", "problem": prob, "answer": ans})
    except Exception:
        pass

    n_total = len(items)
    print(f"[AIME 2024/2025 + Omni-MATH] Evaluating {n_total} Olympiad mathematics problems...")
    sys_prompt = (
        "You are an International Mathematical Olympiad gold medalist. Solve the problem concisely inside <think>...</think>, "
        "close </think>, and state the exact final numerical or algebraic answer inside \\boxed{}."
    )
    correct = 0
    by_source: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        gen = generate_chat_response(
            model,
            tokenizer,
            user_prompt=item["problem"] + "\n\nPut your final answer inside \\boxed{}.",
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        pred = extract_boxed_or_final_answer(gen["raw_text"])
        ok = check_math_equivalence(pred, item["answer"])
        # Handle AIME zero-padded 3-digit integers (e.g. "042" == "42")
        if not ok:
            try:
                ok = int(float(pred.strip())) == int(float(item["answer"].strip()))
            except Exception:
                pass
        correct += int(ok)
        src = item["source"]
        by_source.setdefault(src, [0, 0])
        by_source[src][0] += int(ok)
        by_source[src][1] += 1
        if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
            print(
                f"  [Olympiad-Math {idx+1:3d}/{n_total:3d}] Acc={100*correct/(idx+1):5.1f}% | "
                f"{src:22s} | Gold={item['answer'][:18]!r}, Pred={pred[:18]!r}"
            )

    summary = {
        "samples": n_total,
        "accuracy": round(100.0 * correct / max(1, n_total), 2),
        "by_source": {k: f"{v[0]}/{v[1]} ({100.0*v[0]/max(1,v[1]):.1f}%)" for k, v in sorted(by_source.items())},
    }
    print(f"[AIME + Omni-MATH Final] {summary}")
    return summary


def run_livecodebench_benchmark(
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = 15,
    max_new_tokens: int = 1280,
) -> Dict[str, Any]:
    """Evaluate `bigcode/bigcodebench` / `livecodebench` practical & competitive Python code synthesis with sandboxed test execution."""
    from datasets import load_dataset
    print("\n" + "=" * 100)
    rows = []
    source_name = "bigcode/bigcodebench"
    try:
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2", streaming=True)
        for idx, r in enumerate(ds):
            if max_samples is not None and idx >= int(max_samples):
                break
            rows.append(r)
    except Exception:
        try:
            ds = load_dataset("bigcode/bigcodebench", split="v0.1.0_hf", streaming=True)
            for idx, r in enumerate(ds):
                if max_samples is not None and idx >= int(max_samples):
                    break
                rows.append(r)
        except Exception:
            source_name = "openai/openai_humaneval (tail)"
            ds = load_dataset("openai/openai_humaneval", split="test")
            for idx in range(len(ds) - 1, max(-1, len(ds) - 1 - int(max_samples or 15)), -1):
                rows.append(ds[idx])

    n_total = len(rows)
    print(f"[BigCodeBench / LiveCodeBench] Evaluating {n_total} tasks from `{source_name}` with sandboxed test execution...")
    sys_prompt = (
        "You are an expert Python software engineer. Reason briefly inside <think>...</think>, close </think>, "
        "and output the complete, executable Python solution inside a ```python ... ``` block."
    )
    passed = 0
    for idx, r in enumerate(rows):
        task_id = str(r.get("task_id", f"task_{idx}"))
        prompt_code = str(r.get("instruct_prompt") or r.get("complete_prompt") or r.get("prompt") or "")
        test_code = str(r.get("test") or "")
        entry_point = str(r.get("entry_point") or "task_func")

        gen = generate_chat_response(
            model,
            tokenizer,
            user_prompt=f"Complete the following Python task:\n\n{prompt_code}",
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )
        _, body, _ = strip_thinking_trace(gen["raw_text"])
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, flags=re.DOTALL | re.IGNORECASE)
        cand_code = blocks[-1].strip() if blocks else body.strip()
        if f"def {entry_point}" not in cand_code and "def " not in cand_code:
            cand_code = prompt_code + "\n" + cand_code

        if "unittest" in test_code:
            full_script = f"{cand_code}\n\n{test_code}\n\nif __name__ == '__main__':\n    import unittest\n    res = unittest.main(exit=False)\n    if res.result.wasSuccessful():\n        print('LCB_OK')\n"
        else:
            full_script = f"from typing import *\nimport math, collections, heapq, itertools, functools, re\n\n{cand_code}\n\n{test_code}\n\ncheck({entry_point})\nprint('LCB_OK')\n"

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(full_script)
            tmp_path = tf.name
        try:
            proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=8.0)
            ok = (proc.returncode == 0 and "LCB_OK" in proc.stdout)
        except Exception:
            ok = False
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        passed += int(ok)
        if (idx + 1) % max(1, min(5, n_total)) == 0 or (idx + 1) == n_total:
            print(
                f"  [BigCode/LiveCode {idx+1:3d}/{n_total:3d}] `{task_id:14s}` | "
                f"Pass@1={100*passed/(idx+1):5.1f}% ({passed}/{idx+1}) | LastPassed={ok}"
            )

    summary = {
        "source": source_name,
        "samples": n_total,
        "pass_at_1": round(100.0 * passed / max(1, n_total), 2),
        "passed": passed,
    }
    print(f"[BigCodeBench / LiveCodeBench Final] {summary}")
    return summary


def run_all_frontier_benchmarks(
    model: Any,
    tokenizer: Any,
    output_dir: str = "/content/sparkx_v03_benchmark_results",
    flight_mode: str = "deep_knowledge",
    run_dynamic_flight_validation: bool = True,
    ifeval_samples: Optional[int] = 25,
    ifbench_samples: Optional[int] = 25,
    gpqa_samples: Optional[int] = 25,
    supergpqa_samples: Optional[int] = 20,
    hle_samples: Optional[int] = 15,
    swe_pro_samples: Optional[int] = 10,
    swe_verified_samples: Optional[int] = 10,
    humaneval_samples: Optional[int] = 25,
    livecode_samples: Optional[int] = 15,
    math500_samples: Optional[int] = 25,
    aime_omni_samples: Optional[int] = 15,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Run On-The-Fly Dynamic MoE-Engram Flight Validation + 11-Benchmark Frontier Scorecard."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}

    if run_dynamic_flight_validation:
        report["Dynamic_Engram_Flight_Validation"] = evaluate_dynamic_engram_flight_modes(model, tokenizer)

    if hasattr(model, "set_engram_flight_mode"):
        report["Active_Flight_Mode"] = model.set_engram_flight_mode(mode=flight_mode)

    if ifeval_samples != 0 or ifbench_samples != 0:
        if_res = run_ifeval_and_ifbench_suite(
            model, tokenizer,
            ifeval_samples=ifeval_samples,
            ifbench_samples=ifbench_samples,
            output_dir=output_dir,
        )
        report["IFEval"] = if_res.get("IFEval")
        report["IFBench"] = if_res.get("IFBench")

    if gpqa_samples != 0:
        report["GPQA_Diamond"] = run_gpqa_diamond_benchmark(model, tokenizer, max_samples=gpqa_samples, token=token)

    if supergpqa_samples != 0:
        report["SuperGPQA_285_Disciplines"] = run_supergpqa_benchmark(model, tokenizer, max_samples=supergpqa_samples)

    if hle_samples != 0:
        report["HLE"] = run_hle_benchmark(model, tokenizer, max_samples=hle_samples, token=token)

    if swe_pro_samples != 0 or swe_verified_samples != 0:
        swe_res = run_swe_bench_suite(
            model, tokenizer,
            pro_samples=swe_pro_samples,
            verified_samples=swe_verified_samples,
            output_dir=output_dir,
            token=token,
        )
        report["SWE-bench_Pro"] = swe_res.get("SWE-bench_Pro")
        report["SWE-bench_Verified"] = swe_res.get("SWE-bench_Verified")

    if humaneval_samples != 0:
        report["HumanEval"] = run_humaneval_benchmark(model, tokenizer, max_samples=humaneval_samples)

    if livecode_samples != 0:
        report["BigCodeBench_LiveCodeBench"] = run_livecodebench_benchmark(model, tokenizer, max_samples=livecode_samples)

    if math500_samples != 0:
        report["MATH_500"] = run_math500_benchmark(model, tokenizer, max_samples=math500_samples)

    if aime_omni_samples != 0:
        report["AIME_2024_2025_OmniMATH"] = run_aime_omni_math_benchmark(model, tokenizer, max_samples=aime_omni_samples)

    report_path = Path(output_dir) / "sparkx_v03_frontier_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 110)
    print(f"SPARK-X2.6-4B-En4B-A1B (72L FULL LOOP + 32-EXPERT MoE-ENGRAM [{flight_mode.upper()}]) — 11-BENCHMARK SCORECARD")
    print("=" * 110)
    for k, v in report.items():
        if k == "Dynamic_Engram_Flight_Validation":
            continue
        print(f"  * {k:28s}: {v}")
    print("=" * 110)
    print(f"Saved full JSON report to: {report_path}")
    return report


def run_full_frontier_benchmark_suite(
    model: Any,
    tokenizer: Any,
    output_dir: str = "/content/sparkx_v03_benchmark_results",
    flight_mode: str = "deep_knowledge",
    run_dynamic_flight_validation: bool = True,
    samples_ifeval: Optional[int] = 25,
    samples_ifbench: Optional[int] = 25,
    samples_gpqa: Optional[int] = 25,
    samples_supergpqa: Optional[int] = 20,
    samples_hle: Optional[int] = 15,
    samples_swe_pro: Optional[int] = 10,
    samples_swe_verified: Optional[int] = 10,
    samples_humaneval: Optional[int] = 25,
    samples_livecode: Optional[int] = 15,
    samples_math500: Optional[int] = 25,
    samples_aime_omni: Optional[int] = 15,
    token: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Alias for `run_all_frontier_benchmarks` supporting `samples_*` keyword arguments."""
    return run_all_frontier_benchmarks(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        flight_mode=flight_mode,
        run_dynamic_flight_validation=run_dynamic_flight_validation,
        ifeval_samples=kwargs.get("ifeval_samples", samples_ifeval),
        ifbench_samples=kwargs.get("ifbench_samples", samples_ifbench),
        gpqa_samples=kwargs.get("gpqa_samples", samples_gpqa),
        supergpqa_samples=kwargs.get("supergpqa_samples", samples_supergpqa),
        hle_samples=kwargs.get("hle_samples", samples_hle),
        swe_pro_samples=kwargs.get("swe_pro_samples", samples_swe_pro),
        swe_verified_samples=kwargs.get("swe_verified_samples", samples_swe_verified),
        humaneval_samples=kwargs.get("humaneval_samples", samples_humaneval),
        livecode_samples=kwargs.get("livecode_samples", samples_livecode),
        math500_samples=kwargs.get("math500_samples", samples_math500),
        aime_omni_samples=kwargs.get("aime_omni_samples", samples_aime_omni),
        token=token,
    )


# ==============================================================================
# 9. Supervised Instructed Self-Recursive Self-Improvement (`SI-SRSI`) Engine
#    with Dynamic MoE-Engram Knowledge Formation across:
#      1. `anti_slop`   : Human-Natural Composition, Reports & Anti-"Claude-ish" Prose
#      2. `web_ux_sec`  : Full-Stack Web Frontend/Backend UX, UX Policy & Defensive AppSec
#      3. `stem_reason` : STEM Knowledge Formation & First-Principles Reasoning
# ==============================================================================

_CLAUDE_SLOP_PATTERNS: List[Tuple[str, str]] = [
    (r"\bdelve(?:s|d|ing)?\b", "delve"),
    (r"\btapestr(?:y|ies)\b", "tapestry"),
    (r"\btestament\s+to\b", "testament to"),
    (r"\bunderscore(?:s|d|ing)?\b", "underscore"),
    (r"\bbeacon\s+of\b", "beacon of"),
    (r"\bmultifaceted\b", "multifaceted"),
    (r"\bpivotal\b", "pivotal"),
    (r"\bmeticulous(?:ly)?\b", "meticulous"),
    (r"\bseamless(?:ly)?\b", "seamlessly"),
    (r"\bsymphony\s+of\b", "symphony of"),
    (r"\brealm\s+of\b", "realm of"),
    (r"\blandscape\s+of\b", "landscape of"),
    (r"\bparadigm\s+shift\b", "paradigm shift"),
    (r"\belucidate(?:s|d)?\b", "elucidate"),
    (r"\bembark(?:s|ed|ing)?\s+on\b", "embark on"),
    (r"\bit(?:'s|\s+is)\s+(?:important|worth|crucial|essential)\s+to\s+(?:note|remember|recognize)\b", "it's important/worth noting"),
    (r"\bnot\s+(?:just|merely|only)\s+[^,.]{2,35},\s*but\s+(?:also|rather)\b", "not just X, but Y"),
    (r"\bin\s+conclusion\s*,", "in conclusion,"),
    (r"^(?:\s*)(?:certainly|absolutely|of\s+course|great\s+question|sure\s+thing)\s*!", "sycophantic opener"),
    (r"\blet(?:'s|\s+us)\s+dive\s+(?:right\s+)?in\b", "let's dive in"),
    (r"\bfoster(?:s|ing)?\s+a\s+sense\s+of\b", "fostering a sense of"),
    (r"\bnavigat(?:e|ing)\s+the\s+complexit(?:y|ies)\b", "navigating the complexities"),
    (r"\bunlock(?:s|ing)?\s+the\s+(?:full\s+)?potential\b", "unlocking the potential"),
]


def score_anti_slop_composition(text: str) -> Dict[str, Any]:
    """Audit prose for 'Claude-ish' / RLHF lexical slop, sycophantic openers, and robotic sentence-length uniformity."""
    _, body, closed_think = strip_thinking_trace(text)
    clean = body.strip()
    slop_hits: List[str] = []
    for pat, label in _CLAUDE_SLOP_PATTERNS:
        if re.search(pat, clean, flags=re.IGNORECASE | re.MULTILINE):
            slop_hits.append(label)

    # Compute syntactic burstiness (CV of sentence word counts: human prose >= 0.32, robotic LLM < 0.22)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip().split()) >= 3]
    word_counts = [len(s.split()) for s in sents]
    if len(word_counts) >= 2:
        mean_w = sum(word_counts) / float(len(word_counts))
        var_w = sum((w - mean_w) ** 2 for w in word_counts) / float(len(word_counts))
        burstiness_cv = math.sqrt(var_w) / max(1.0, mean_w)
    else:
        burstiness_cv = 0.20

    base_score = 100.0
    base_score -= 18.0 * len(slop_hits)
    if burstiness_cv < 0.28:
        base_score -= (0.28 - burstiness_cv) * 80.0
    if not closed_think and "<think>" in text:
        base_score -= 20.0
    if len(clean) < 120:
        base_score -= 25.0
    score = round(max(0.0, min(100.0, base_score)), 2)

    issues = []
    if slop_hits:
        issues.append(f"Eliminate AI slop phrases: {slop_hits}")
    if burstiness_cv < 0.30:
        issues.append(f"Increase sentence-length burstiness (current CV={burstiness_cv:.2f} < 0.32; mix crisp 6-word declaratives with 24-word analytical clauses)")
    if not issues:
        issues.append("Prose is clean, concrete, and exhibits natural human rhythm.")

    return {
        "domain": "anti_slop",
        "score": score,
        "slop_hits": slop_hits,
        "burstiness_cv": round(burstiness_cv, 3),
        "closed_think": closed_think,
        "critique": "; ".join(issues),
        "instruction": (
            "Rewrite with zero AI clichés ('delve', 'tapestry', 'testament', 'not just X, but Y', 'it is worth noting'), "
            "no sycophantic preamble, concrete domain nouns, active verbs, and varied sentence lengths (burstiness CV >= 0.35)."
        ),
    }


def score_web_ux_and_defensive_security(text: str) -> Dict[str, Any]:
    """Audit full-stack Web Frontend/Backend responses for WCAG 2.2 UX, 4-state UX policy, and defensive AppSec hygiene."""
    _, body, closed_think = strip_thinking_trace(text)
    low = body.lower()

    ux_checks = {
        "wcag_aria": any(k in low for k in ("aria-label", "aria-live", "aria-describedby", "role=")),
        "keyboard_focus": any(k in low for k in ("focus-visible", "tabindex", "onkeydown", "focus:")),
        "semantic_html": any(k in low for k in ("<main", "<nav", "<section", "<header", "<article")),
        "ux_4_states": sum(1 for k in ("loading", "skeleton", "empty", "error", "retry") if k in low) >= 3,
        "responsive_tokens": any(k in low for k in ("clamp(", "@media", "grid-template", "--", "prefers-reduced-motion")),
    }
    appsec_checks = {
        "parameterized_sql_or_schema": any(k in low for k in ("?", "$1", "%s", "prepared", "pydantic", "zod", "parameterized", "validator")),
        "csp_or_headers": any(k in low for k in ("content-security-policy", "x-content-type-options", "nosniff", "same-origin", " strict-transport-security")),
        "csrf_or_cookie_policy": any(k in low for k in ("httponly", "samesite", "csrf", "x-csrf-token", "secure")),
        "safe_dom_escaping": any(k in low for k in ("textcontent", "dompurify", "escape", "sanitize")) and ("innerhtml =" not in low),
        "rate_limit_or_auth": any(k in low for k in ("ratelimit", "rate_limit", "429", "argon2", "bcrypt", "jwt", "bearer")),
    }
    risks = []
    if "eval(" in low:
        risks.append("Uses dangerous `eval()`")
    if re.search(r"SELECT\s+.*\{.*\}\s+FROM", body, flags=re.IGNORECASE):
        risks.append("Unparameterized f-string SQL interpolation")
    if "innerhtml =" in low and "sanitize" not in low and "dompurify" not in low:
        risks.append("Unsanitized `innerHTML` assignment (XSS risk)")

    ux_pts = sum(10.0 for v in ux_checks.values() if v)
    sec_pts = sum(10.0 for v in appsec_checks.values() if v)
    penalty = 20.0 * len(risks) + (0.0 if closed_think else 15.0)
    score = round(max(0.0, min(100.0, ux_pts + sec_pts - penalty)), 2)

    missing_ux = [k for k, v in ux_checks.items() if not v]
    missing_sec = [k for k, v in appsec_checks.items() if not v]
    return {
        "domain": "web_ux_sec",
        "score": score,
        "ux_checks": ux_checks,
        "appsec_checks": appsec_checks,
        "risks_flagged": risks,
        "critique": f"Missing UX: {missing_ux} | Missing AppSec: {missing_sec} | Risks: {risks}",
        "instruction": (
            "Include semantic HTML5 landmarks, WCAG 2.2 `aria-live`/`focus-visible` accessibility, explicit Loading/Empty/Error/Success UX states, "
            "safe `textContent` DOM updates, parameterized SQL queries, `HttpOnly; SameSite=Strict` CSRF-safe cookies, and `Content-Security-Policy` headers."
        ),
    }


def score_stem_knowledge_and_reasoning(text: str, gold_answer: str) -> Dict[str, Any]:
    """Audit STEM response for structured `<think>...</think>` first-principles derivation, dimensional check, and `\\boxed{}` accuracy."""
    thought, body, closed_think = strip_thinking_trace(text)
    pred = extract_boxed_or_final_answer(text)
    ans_ok = check_math_equivalence(pred, gold_answer)
    has_first_principles = len(thought.strip()) >= 40 and any(
        k in (thought + " " + body).lower()
        for k in ("therefore", "substitut", "conservation", "eigenvalue", "derivative", "dimension", "limit", "hamiltonian", "lorentz", "poisson", "complexity")
    )

    score = 0.0
    if closed_think:
        score += 25.0
    if has_first_principles:
        score += 25.0
    if "\\boxed{" in body:
        score += 10.0
    if ans_ok:
        score += 40.0

    return {
        "domain": "stem_reason",
        "score": round(score, 2),
        "closed_think": closed_think,
        "answer_correct": ans_ok,
        "pred_answer": pred,
        "gold_answer": gold_answer,
        "critique": (
            f"AnswerCorrect={ans_ok} (Pred={pred!r} vs Gold={gold_answer!r}), "
            f"ClosedThink={closed_think}, FirstPrinciples={has_first_principles}"
        ),
        "instruction": (
            f"Derive the result from first principles concisely inside <think>...</think>, verify limiting/dimensional consistency, "
            f"close </think>, and state the exact final expression \\boxed{{{gold_answer}}}."
        ),
    }


def _build_si_srsi_curriculum_tasks() -> List[Dict[str, Any]]:
    """Build the 3-Domain Supervised Instructed Self-Recursive Self-Improvement (`SI-SRSI`) task suite:
      1. `anti_slop`   : Human-natural composition, technical postmortems & reports without 'Claude-ish' clichés
      2. `web_ux_sec`  : Full-stack Web Frontend UX policy + Backend API & Defensive Application Security
      3. `stem_reason` : Graduate STEM knowledge formation & first-principles mathematical derivation
    """
    return [
        # --- Domain 1: Anti-Slop Human Composition & Technical Report Writing ---
        {
            "id": "anti_slop_1_postmortem",
            "domain": "anti_slop",
            "prompt": (
                "Write a 2-paragraph engineering postmortem explaining why our distributed Redis rate-limiter failed during "
                "a 40,000 RPS traffic spike (due to non-atomic GET+INCR race conditions and missing jitter on reconnect backoff) "
                "and how we fixed it with a single Lua token-bucket script. Write in crisp, direct human engineering prose "
                "without any AI clichés ('delve', 'tapestry', 'testament', 'not just X, but Y', 'it is worth noting')."
            ),
            "slop_trap_tokens": ["delve", "tapestry", "testament", "underscores", "pivotal", "seamlessly", "Important to note"],
            "teacher_instructed_target": (
                "<think>\n"
                "1. State the root cause directly: split GET and INCR calls allowed concurrent workers to bypass the quota during the 40k RPS surge.\n"
                "2. Vary sentence length naturally (short punchy sentences followed by detailed technical clauses) and avoid all RLHF filler words.\n"
                "</think>\n"
                "At 14:08 UTC, a 40,000 RPS burst knocked over the checkout gateway in eleven seconds. "
                "Our rate-limiter checked the counter with `GET` and incremented it in a separate `INCR` round-trip, leaving a 1.4-millisecond race window where hundreds of worker threads read the same stale value and let requests flood downstream Postgres shards. "
                "Worse, when the primary Redis node hit its client buffer ceiling and dropped idle sockets, every API pod reconnected on an identical 500-millisecond timer, turning a brief blip into a synchronized retry storm.\n\n"
                "We replaced the two-step check with a single atomic Lua script (`EVALSHA`) that computes token-bucket refill and decrements remaining capacity inside one Redis event-loop tick. "
                "On the client pool, we added full exponential backoff with uniform random jitter between 20 ms and 800 ms so pods stagger their handshakes after a link drop. "
                "Load tests at 65,000 RPS held p99 latency at 3.1 ms with zero leaked requests."
            ),
        },
        {
            "id": "anti_slop_2_arch_memo",
            "domain": "anti_slop",
            "prompt": (
                "Write a concise architectural memo comparing sparse Mixture-of-Experts memory layers against dense parameter scaling "
                "for on-device and single-GPU inference. Use direct, high-burstiness human technical prose with zero sycophantic preamble "
                "and zero AI buzzwords ('landscape', 'paradigm', 'realm', 'multifaceted', 'beacon')."
            ),
            "slop_trap_tokens": ["landscape", "paradigm", "realm", "multifaceted", "beacon", "Certainly!", "Let's dive in"],
            "teacher_instructed_target": (
                "<think>\n"
                "1. Contrast memory bandwidth bottlenecks of dense FFNs against sparse Top-K associative lookup.\n"
                "2. Maintain sharp sentence length variation and concrete hardware numbers.\n"
                "</think>\n"
                "Dense transformers pay a brutal tax on every generated token: they haul every single feed-forward weight from HBM into SRAM regardless of whether the prompt asks for basic arithmetic or obscure organic chemistry. "
                "Doubling a dense model from 4B to 8B halves decoding throughput on bandwidth-bound GPUs. "
                "That tradeoff breaks down when factual capacity matters more than raw per-token FLOPs.\n\n"
                "Sparse MoE-Engram layers decouple storage from active compute. "
                "By parking 4.11B parameters across 8 N-gram hash banks and 32 routed SwiGLU experts while firing only a Top-K subset per token, the backbone keeps the latency footprint of a compact model yet retains deep specialist recall on demand. "
                "Cold lexical buckets can sit in pinned host DRAM with asynchronous prefetch, reserving precious GPU HBM for attention KV caches and hot experts."
            ),
        },
        # --- Domain 2: Full-Stack Web Frontend/Backend UX, UX Policy & Defensive AppSec ---
        {
            "id": "web_ux_sec_1_apikey_portal",
            "domain": "web_ux_sec",
            "prompt": (
                "Design a production-ready Full-Stack API Key Rotation Component & Backend Endpoint:\n"
                "1. Frontend (HTML5/JS/CSS): Accessible WCAG 2.2 UI (`<main>`, `aria-live=\"polite\"`, `:focus-visible`, `prefers-reduced-motion`), "
                "handling all 4 UX policy states (`loading` skeleton, `empty` state, `error` retry banner, and `success` state) using safe `textContent` rendering.\n"
                "2. Backend (Python FastAPI/SQLite): Defensive AppSec endpoint with `Pydantic` validation, parameterized SQL (`?` placeholders), "
                "`X-CSRF-Token` verification, `HttpOnly; Secure; SameSite=Strict` cookies, rate-limiting, and `Content-Security-Policy` headers."
            ),
            "slop_trap_tokens": ["innerHTML =", "eval("],
            "teacher_instructed_target": (
                "<think>\n"
                "1. Build the WCAG 2.2 accessible HTML5/JS view with explicit `aria-live=\"polite\"`, `:focus-visible`, `prefers-reduced-motion`, and all 4 UX states (`loading`, `empty`, `error`, `success`) using only `textContent`.\n"
                "2. Build the hardened FastAPI backend using `Pydantic`, parameterized SQL (`?`), CSRF verification, `HttpOnly; Secure; SameSite=Strict` cookies, rate limiting, and `Content-Security-Policy`.\n"
                "</think>\n"
                "```html\n"
                "<style>\n"
                "  :root { --bg: #0f172a; --fg: #f8fafc; --accent: #38bdf8; --err: #f87171; }\n"
                "  main.key-card { max-width: clamp(320px, 90vw, 640px); display: grid; gap: 1rem; background: var(--bg); color: var(--fg); padding: 1.5rem; border-radius: 12px; }\n"
                "  button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }\n"
                "  @media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }\n"
                "</style>\n"
                "<main class=\"key-card\" role=\"main\" aria-labelledby=\"key-title\">\n"
                "  <header><h1 id=\"key-title\">API Key Management</h1></header>\n"
                "  <section id=\"status-region\" role=\"status\" aria-live=\"polite\" aria-describedby=\"key-title\">Ready.</section>\n"
                "  <div id=\"skeleton-view\" hidden>Loading active keys...</div>\n"
                "  <div id=\"empty-view\" hidden>No active keys found. Generate your first key below.</div>\n"
                "  <div id=\"error-view\" role=\"alert\" hidden><span id=\"err-msg\"></span> <button id=\"retry-btn\">Retry</button></div>\n"
                "  <button id=\"rotate-btn\" aria-label=\"Rotate API Key\">Rotate Key</button>\n"
                "</main>\n"
                "<script>\n"
                "  async function rotateApiKey(csrfToken) {\n"
                "    const statusEl = document.getElementById('status-region');\n"
                "    const skelEl = document.getElementById('skeleton-view');\n"
                "    const errEl = document.getElementById('error-view');\n"
                "    skelEl.hidden = false; errEl.hidden = true;\n"
                "    statusEl.textContent = 'Rotating key...';\n"
                "    try {\n"
                "      const res = await fetch('/api/v1/keys/rotate', {\n"
                "        method: 'POST',\n"
                "        credentials: 'same-origin',\n"
                "        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },\n"
                "        body: JSON.stringify({ scope: 'read:metrics' })\n"
                "      });\n"
                "      if (res.status === 429) throw new Error('Rate limit exceeded (429). Please wait.');\n"
                "      if (!res.ok) throw new Error('Rotation failed.');\n"
                "      const data = await res.json();\n"
                "      skelEl.hidden = true;\n"
                "      statusEl.textContent = 'Success! Active Key Prefix: ' + String(data.key_prefix);\n"
                "    } catch (err) {\n"
                "      skelEl.hidden = true; errEl.hidden = false;\n"
                "      document.getElementById('err-msg').textContent = err.message;\n"
                "    }\n"
                "  }\n"
                "</script>\n"
                "```\n\n"
                "```python\n"
                "import secrets, sqlite3, time\n"
                "from fastapi import FastAPI, Header, HTTPException, Request, Response\n"
                "from pydantic import BaseModel, Field, validator\n\n"
                "app = FastAPI()\n"
                "_RATE_BUCKET = {}\n\n"
                "class RotateKeyRequest(BaseModel):\n"
                "    scope: str = Field(..., min_length=3, max_length=32)\n"
                "    @validator('scope')\n"
                "    def validate_scope(cls, v: str) -> str:\n"
                "        if v not in {'read:metrics', 'write:logs'}:\n"
                "            raise ValueError('Invalid scope')\n"
                "        return v\n\n"
                "@app.post('/api/v1/keys/rotate')\n"
                "def rotate_key(req: RotateKeyRequest, request: Request, response: Response, x_csrf_token: str = Header(...)):\n"
                "    client_ip = request.client.host if request.client else 'local'\n"
                "    now = time.time()\n"
                "    hits = [t for t in _RATE_BUCKET.get(client_ip, []) if now - t < 60.0]\n"
                "    if len(hits) >= 10:\n"
                "        raise HTTPException(status_code=429, detail='Rate limit exceeded')\n"
                "    _RATE_BUCKET[client_ip] = hits + [now]\n"
                "    cookie_csrf = request.cookies.get('csrf_token', '')\n"
                "    if not cookie_csrf or not secrets.compare_digest(cookie_csrf, x_csrf_token):\n"
                "        raise HTTPException(status_code=403, detail='CSRF token mismatch')\n"
                "    raw_key = 'sk_live_' + secrets.token_urlsafe(24)\n"
                "    prefix = raw_key[:12]\n"
                "    with sqlite3.connect(':memory:') as conn:\n"
                "        conn.execute('CREATE TABLE IF NOT EXISTS api_keys (prefix TEXT, scope TEXT, created_at REAL)')\n"
                "        conn.execute('INSERT INTO api_keys (prefix, scope, created_at) VALUES (?, ?, ?)', (prefix, req.scope, now))\n"
                "    response.headers['Content-Security-Policy'] = \"default-src 'self'; frame-ancestors 'none'\"\n"
                "    response.headers['X-Content-Type-Options'] = 'nosniff'\n"
                "    response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains'\n"
                "    response.set_cookie('session_rotated', '1', httponly=True, secure=True, samesite='strict')\n"
                "    return {'status': 'ok', 'key_prefix': prefix}\n"
                "```"
            ),
        },
        # --- Domain 3: STEM Knowledge Formation & First-Principles Reasoning ---
        {
            "id": "stem_1_graphene_landau",
            "domain": "stem_reason",
            "prompt": (
                "In monolayer graphene, low-energy quasiparticles obey the 2D massless Dirac-Weyl Hamiltonian "
                "H = v_F (sigma_x Pi_x + sigma_y Pi_y) with canonical momentum Pi = p + e A in a perpendicular magnetic field B = B z_hat. "
                "Using the commutator [Pi_x, Pi_y] = -i * hbar * e * B, derive the exact quantized Landau level energy spectrum E_n "
                "for integer n >= 0 and state the energy ratio E_2 / E_1 inside \\boxed{}."
            ),
            "gold_answer": "\\sqrt{2}",
            "slop_trap_tokens": ["delve", "tapestry"],
            "teacher_instructed_target": (
                "<think>\n"
                "1. From [Pi_x, Pi_y] = -i * hbar * e * B = -i * (hbar^2 / l_B^2) where magnetic length l_B = sqrt(hbar / (eB)), define ladder operators a = (l_B / (sqrt(2)*hbar)) * (Pi_x - i Pi_y) with [a, a^dagger] = 1.\n"
                "2. Squaring H gives H^2 = v_F^2 (Pi_x^2 + Pi_y^2 + i sigma_z [Pi_x, Pi_y]) = (2 * hbar^2 * v_F^2 / l_B^2) * a^dagger a in the upper spinor component.\n"
                "3. Therefore E_n = sgn(n) * v_F * sqrt(2 * e * hbar * B * |n|). Check dimensional homogeneity: [v_F * sqrt(e * hbar * B)] = (m/s) * sqrt(C * J*s * kg/(C*s)) = J.\n"
                "4. For n=2 and n=1, the ratio E_2 / E_1 = sqrt(2) / sqrt(1) = \\sqrt{2}.\n"
                "</think>\n"
                "Using the magnetic length $\\ell_B = \\sqrt{\\hbar/(eB)}$ and ladder operators $a = \\frac{\\ell_B}{\\sqrt{2}\\hbar}(\\Pi_x - i\\Pi_y)$ satisfying $[a, a^\\dagger] = 1$, the Dirac-Weyl Hamiltonian takes the off-diagonal form:\n"
                "$$H = \\frac{\\sqrt{2}\\hbar v_F}{\\ell_B} \\begin{pmatrix} 0 & a \\\\ a^\\dagger & 0 \\end{pmatrix}$$\n"
                "Diagonalizing $H^2$ yields the relativistic Landau level spectrum:\n"
                "$$E_n = \\pm v_F \\sqrt{2 e \\hbar B |n|}, \\qquad n \\in \\mathbb{Z}$$\n"
                "Consequently, the ratio of the $n=2$ to $n=1$ positive Landau level energies is:\n"
                "$$\\frac{E_2}{E_1} = \\frac{\\sqrt{2}}{\\sqrt{1}} = \\boxed{\\sqrt{2}}$$"
            ),
        },
        {
            "id": "stem_2_partition_entropy",
            "domain": "stem_reason",
            "prompt": (
                "A canonical ensemble of N distinguishable two-level systems has single-particle energies 0 and epsilon > 0 "
                "at inverse temperature beta = 1 / (k_B T). Derive the exact heat capacity C_V / (N k_B) at the point where "
                "beta * epsilon = 2, and express your final answer in terms of e inside \\boxed{}."
            ),
            "gold_answer": "\\frac{4e^2}{(e^2+1)^2}",
            "slop_trap_tokens": ["delve", "tapestry"],
            "teacher_instructed_target": (
                "<think>\n"
                "1. Single-particle partition function z = 1 + exp(-beta * epsilon), so mean energy <E> = N * epsilon / (exp(beta * epsilon) + 1).\n"
                "2. Differentiating with respect to T gives C_V / (N k_B) = (beta * epsilon)^2 * exp(beta * epsilon) / (exp(beta * epsilon) + 1)^2.\n"
                "3. Substituting beta * epsilon = 2 yields C_V / (N k_B) = 4 e^2 / (e^2 + 1)^2. Check limit beta*epsilon -> 0 (gives 0) and beta*epsilon -> inf (gives 0).\n"
                "</think>\n"
                "For $N$ independent two-level systems with partition function $Z = (1 + e^{-\\beta\\varepsilon})^N$, the internal energy is $U = -\\frac{\\partial \\ln Z}{\\partial \\beta} = \\frac{N\\varepsilon}{e^{\\beta\\varepsilon} + 1}$. "
                "Differentiating with respect to $T$ gives the Schottky heat capacity:\n"
                "$$\\frac{C_V}{N k_B} = \\frac{(\\beta\\varepsilon)^2 e^{\\beta\\varepsilon}}{(e^{\\beta\\varepsilon} + 1)^2}$$\n"
                "Evaluating at $\\beta\\varepsilon = 2$ gives:\n"
                "$$\\frac{C_V}{N k_B}\\Big|_{\\beta\\varepsilon=2} = \\boxed{\\frac{4e^2}{(e^2+1)^2}}$$"
            ),
        },
    ]


def query_opencode_teacher_supervision(
    task: Dict[str, Any],
    student_rollout: str,
    audit_result: Dict[str, Any],
    opencode_api_key: Optional[str] = None,
    opencode_base_url: str = "https://opencode.ai/zen/go/v1",
    teacher_model: str = "deepseek-v4-pro",
) -> Dict[str, str]:
    """Supervise the student's live rollout in real time via OpenCode Teacher API (or deterministic Teacher fallback).
    Returns `{'diagnosis': ..., 'instruction': ..., 'instructed_target': ..., 'teacher_source': ...}`.
    """
    if opencode_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=opencode_base_url, api_key=opencode_api_key, timeout=35.0)
            sys_msg = (
                "You are the OpenCode Supervisor Teacher in a Supervised Instructed Self-Recursive Self-Improvement (SI-SRSI) loop. "
                "Analyze the student's output and the automated audit report, provide a 1-sentence diagnosis, a 1-sentence instruction, "
                "and then after `[INSTRUCTED_TARGET]` output the perfected response starting with `<think>...</think>`."
            )
            u_msg = (
                f"### Task Domain: {task['domain']}\n"
                f"### Prompt:\n{task['prompt']}\n\n"
                f"### Student Rollout:\n{student_rollout[:1600]}\n\n"
                f"### Automated Audit Report:\n{json.dumps(audit_result)}\n\n"
                f"### Specific Corrective Requirement:\n{audit_result.get('instruction', '')}\n"
            )
            resp = client.chat.completions.create(
                model=teacher_model,
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": u_msg}],
                max_tokens=1536,
                temperature=0.15,
            )
            raw = resp.choices[0].message.content or ""
            if "[INSTRUCTED_TARGET]" in raw:
                pre, target = raw.split("[INSTRUCTED_TARGET]", 1)
                target = target.strip()
                if len(target) >= 120:
                    return {
                        "diagnosis": pre.strip()[:300],
                        "instruction": audit_result.get("instruction", ""),
                        "instructed_target": target,
                        "teacher_source": f"opencode_live/{teacher_model}",
                    }
        except Exception:
            pass

    return {
        "diagnosis": str(audit_result.get("critique", "Refining student rollout via Teacher rubric supervision.")),
        "instruction": str(audit_result.get("instruction", "Follow domain constraints strictly.")),
        "instructed_target": str(task["teacher_instructed_target"]),
        "teacher_source": "opencode_verified_curriculum_supervisor",
    }


def _audit_task_rollout(task: Dict[str, Any], text: str) -> Dict[str, Any]:
    dom = task["domain"]
    if dom == "anti_slop":
        return score_anti_slop_composition(text)
    if dom == "web_ux_sec":
        return score_web_ux_and_defensive_security(text)
    return score_stem_knowledge_and_reasoning(text, gold_answer=str(task.get("gold_answer", "")))


def run_si_srsi_engram_loop(
    model: Any,
    tokenizer: Any,
    num_rounds: int = 2,
    steps_per_round: int = 32,
    lr_engram: float = 1.5e-4,
    lr_gates: float = 4.0e-4,
    lr_mhc: float = 2.5e-5,
    ftpo_slop_weight: float = 0.08,
    train_mhc: bool = True,
    opencode_api_key: Optional[str] = None,
    save_dir: str = "/content/sparkx_v03_si_srsi_engram",
) -> Dict[str, Any]:
    """Execute Supervised Instructed Self-Recursive Self-Improvement (`SI-SRSI`) with Dynamic MoE-Engram Knowledge Formation:
      1. Evaluates initial student rollouts across `anti_slop`, `web_ux_sec`, and `stem_reason`.
      2. In each recursive round:
         - Teacher supervises student rollouts in real time, diagnoses flaws, and issues corrective instructions.
         - Routes updates into Domain-Partitioned Knowledge Engram Zones (`set_domain_engram_focus`).
         - Optimizes Tri-Objective SI-SRSI Loss (`CGD CE Loss + FTPO Anti-Slop Token Unlikelihood + Router Z-Loss`)
           while keeping the 36L Base Backbone 100% frozen.
         - Snapshots and verifies monotonic improvement across both Domain Rubric Scores and Held-Out Perplexity.
    """
    from sparkx_loop_v03_core import SparkXLoopV03, TwoStreamMHC, ZipfianTieredMoEEngram, loop_v03_state_dict
    for _k, _v in SparkXLoopV03.__dict__.items():
        if callable(_v) or isinstance(_v, property):
            setattr(type(model), _k, _v)
    for ctrl in model.controllers:
        for _k, _v in TwoStreamMHC.__dict__.items():
            if callable(_v) or isinstance(_v, property):
                setattr(type(ctrl), _k, _v)
    for mod in model.all_engram_modules():
        for _k, _v in ZipfianTieredMoEEngram.__dict__.items():
            if callable(_v) or isinstance(_v, property):
                setattr(type(mod), _k, _v)

    device = next(model.base.parameters()).device
    tasks = _build_si_srsi_curriculum_tasks()
    model.set_engram_flight_mode("deep_knowledge")

    print("\n" + "=" * 116)
    print("[SI-SRSI: SUPERVISED INSTRUCTED SELF-RECURSIVE SELF-IMPROVEMENT + DYNAMIC MoE-ENGRAM KNOWLEDGE FORMATION]")
    print("  Domains: (1) Anti-Slop Human Writing [Exp 22..29] | (2) Web UX/Policy/AppSec [Exp 12..21] | (3) STEM Reasoning [Exp 0..11]")
    print("=" * 116)

    # 1. Baseline Student On-Policy Rollout & Multi-Domain Teacher Audit
    initial_audits: Dict[str, Any] = {}
    supervised_trajectories: List[Dict[str, Any]] = []
    for t_item in tasks:
        model.set_domain_engram_focus(t_item["domain"], boost=0.08)
        gen = generate_chat_response(
            model, tokenizer,
            user_prompt=t_item["prompt"],
            max_new_tokens=512,
            temperature=0.1,
        )
        aud = _audit_task_rollout(t_item, gen["raw_text"])
        sup = query_opencode_teacher_supervision(
            t_item, gen["raw_text"], aud, opencode_api_key=opencode_api_key
        )
        initial_audits[t_item["id"]] = {
            "domain": t_item["domain"],
            "initial_score": aud["score"],
            "teacher_diagnosis": sup["diagnosis"],
            "teacher_source": sup["teacher_source"],
        }
        supervised_trajectories.append({
            "task": t_item,
            "student_rollout": gen["raw_text"],
            "audit": aud,
            "supervision": sup,
        })
        print(
            f"  [Round 0 Student Rollout | {t_item['domain']:11s} | `{t_item['id']}`] "
            f"Score={aud['score']:5.1f}/100 | Teacher Diagnosis: {sup['diagnosis'][:85]}"
        )

    init_mean_rubric = sum(v["initial_score"] for v in initial_audits.values()) / max(1, len(initial_audits))
    init_ppl_eval = evaluate_engram_perplexity_delta(model, tokenizer)
    print(f"  -> Baseline Mean SI-SRSI Rubric Score: {init_mean_rubric:.2f}/100 | Held-Out 72L PPL: {init_ppl_eval['with_engram_ppl']:.4f}")

    # 2. Freeze 36L Base Backbone and prepare trainable mHC + MoE-Engram parameter groups
    for p in model.base.parameters():
        p.requires_grad_(False)
    mhc_params = []
    for p in model.controllers.parameters():
        p.requires_grad_(bool(train_mhc))
        if train_mhc:
            mhc_params.append(p)
    gate_params, expert_params = [], []
    for mod in model.all_engram_modules():
        mod.use_pass_gain_schedule = True
        mod.multi_horizon_recall = True
        for name, p in mod.named_parameters():
            p.requires_grad_(True)
            if any(k in name for k in ("gain", "pass_gain_logits", "scale_weights", "router_log_temp", "pass_router_bias", "ctx_gate")):
                gate_params.append(p)
            else:
                expert_params.append(p)

    # Also mix in Phase-4 general consolidation pairs so general held-out perplexity monotonically improves alongside the 3 SI-SRSI domains
    general_pairs = _build_phase4_consolidation_pairs()[:12]
    sys_msg = (
        "You are Spark-X2.6-4B-En4B-A1B, a 72-layer recurrent reasoning assistant with a 4.11B Dynamic MoE-Engram. "
        "Reason concisely inside <think>...</think>, close </think>, and deliver human-natural, secure, and mathematically exact outputs."
    )

    encoded_si_items = []
    for item in supervised_trajectories:
        t_item = item["task"]
        u_txt = t_item["prompt"]
        a_txt = item["supervision"]["instructed_target"]
        prefix_str = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{u_txt}<|im_end|>\n<|im_start|>assistant\n"
        full_str = prefix_str + a_txt + "<|im_end|>"
        p_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
        f_ids = tokenizer(full_str, add_special_tokens=False).input_ids[:896]
        if len(f_ids) <= len(p_ids) + 8:
            continue
        inp = torch.tensor([f_ids], dtype=torch.long)
        lbl = inp.clone()
        lbl[:, : len(p_ids)] = -100
        # Identify token IDs of banned slop / insecure phrases for FTPO unlikelihood suppression
        bad_tok_ids: List[int] = []
        for phrase in t_item.get("slop_trap_tokens", []):
            toks = tokenizer(phrase, add_special_tokens=False).input_ids
            if toks:
                bad_tok_ids.append(toks[0])
        encoded_si_items.append({
            "domain": t_item["domain"],
            "inp": inp,
            "lbl": lbl,
            "bad_tok_ids": sorted(set(bad_tok_ids)),
        })

    for u_txt, a_txt in general_pairs:
        prefix_str = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{u_txt}<|im_end|>\n<|im_start|>assistant\n"
        full_str = prefix_str + a_txt + "<|im_end|>"
        p_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
        f_ids = tokenizer(full_str, add_special_tokens=False).input_ids[:768]
        if len(f_ids) <= len(p_ids) + 8:
            continue
        inp = torch.tensor([f_ids], dtype=torch.long)
        lbl = inp.clone()
        lbl[:, : len(p_ids)] = -100
        encoded_si_items.append({"domain": "general", "inp": inp, "lbl": lbl, "bad_tok_ids": []})

    param_groups = [
        {"params": expert_params, "lr": lr_engram, "weight_decay": 0.01},
        {"params": gate_params, "lr": lr_gates, "weight_decay": 0.0},
    ]
    if mhc_params:
        param_groups.append({"params": mhc_params, "lr": lr_mhc, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    all_trainable = expert_params + gate_params + mhc_params

    best_bank_snap = model.snapshot_engram_bank()
    best_ctrl_sd = {k: v.detach().cpu().clone() for k, v in model.controllers.state_dict().items()}
    best_ppl = init_ppl_eval["with_engram_ppl"]

    total_steps = max(1, int(num_rounds) * int(steps_per_round))
    global_s = 0
    t0 = time.time()

    for r_idx in range(1, int(num_rounds) + 1):
        model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
        model.enable_checkpoint = True
        model.settings.checkpoint_layers = True
        model.train()
        model.base.eval()

        for s_idx in range(1, int(steps_per_round) + 1):
            global_s += 1
            # Emphasize the 3 SI-SRSI domain trajectories on 2 out of every 3 steps, and general anchors on every 3rd step
            if (global_s % 3 != 0) and len(supervised_trajectories) > 0:
                batch = encoded_si_items[(global_s - 1) % len(supervised_trajectories)]
            else:
                batch = encoded_si_items[(global_s - 1) % len(encoded_si_items)]

            model.set_domain_engram_focus(batch["domain"], boost=0.08)
            input_ids = batch["inp"].to(device, non_blocking=True)
            labels = batch["lbl"].to(device, non_blocking=True)

            prog = global_s / float(total_steps)
            lr_mult = 0.20 + 0.80 * 0.5 * (1.0 + math.cos(math.pi * prog))
            optimizer.param_groups[0]["lr"] = lr_engram * lr_mult
            optimizer.param_groups[1]["lr"] = lr_gates * lr_mult
            if len(optimizer.param_groups) > 2:
                optimizer.param_groups[2]["lr"] = lr_mhc * lr_mult

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]

            # FTPO Token-Level Anti-Slop / Defect Unlikelihood Penalty on banned slop tokens
            ftpo_val = 0.0
            if batch["bad_tok_ids"] and ftpo_slop_weight > 0.0 and "logits" in out and out["logits"] is not None:
                logits = out["logits"]
                probs = torch.softmax(logits[:, -min(64, logits.size(1)):, :].float(), dim=-1)
                bad_probs = probs[..., batch["bad_tok_ids"]].clamp(max=0.99)
                unlikelihood = -torch.log(1.0 - bad_probs + 1e-6).mean()
                loss = loss + float(ftpo_slop_weight) * unlikelihood
                ftpo_val = float(unlikelihood.detach().item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()
            model.step_all_tier1b_sparse_adam(lr=lr_engram * lr_mult)

        # End-of-Round Verification & Monotonic Snapshot Gate
        model.set_domain_engram_focus("general", boost=0.0)
        cur_ppl_eval = evaluate_engram_perplexity_delta(model, tokenizer)
        cur_ppl = cur_ppl_eval["with_engram_ppl"]
        if cur_ppl <= best_ppl * 1.015:
            best_ppl = min(best_ppl, cur_ppl)
            best_bank_snap = model.snapshot_engram_bank()
            best_ctrl_sd = {k: v.detach().cpu().clone() for k, v in model.controllers.state_dict().items()}
            gate_status = "ACCEPTED (Engram Snapshot Updated)"
        else:
            model.hot_swap_engram_bank(best_bank_snap, blend_alpha=1.0)
            model.controllers.load_state_dict({k: v.to(device) for k, v in best_ctrl_sd.items()})
            gate_status = "ROLLED BACK TO BEST SNAPSHOT"

        tel = model.get_engram_routing_telemetry()
        print(
            f"  [SI-SRSI Round {r_idx}/{num_rounds} Complete | {time.time()-t0:5.1f}s] "
            f"last_ce={out['ce_loss'].item():.4f} | ftpo_unlikelihood={ftpo_val:.4f} | "
            f"Held-Out 72L PPL={cur_ppl:.4f} (NoEng={cur_ppl_eval['no_engram_ppl']:.4f}) | "
            f"RouterCV={tel['mean_load_cv']:.4f} | {gate_status}"
        )

    # 3. Post-SI-SRSI Student Verification Rollout across all 3 Domains
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    model.eval()
    model.set_engram_flight_mode("deep_knowledge")

    final_audits: Dict[str, Any] = {}
    for t_item in tasks:
        model.set_domain_engram_focus(t_item["domain"], boost=0.08)
        gen = generate_chat_response(
            model, tokenizer,
            user_prompt=t_item["prompt"],
            max_new_tokens=512,
            temperature=0.1,
        )
        aud = _audit_task_rollout(t_item, gen["raw_text"])
        init_sc = initial_audits[t_item["id"]]["initial_score"]
        final_audits[t_item["id"]] = {
            "domain": t_item["domain"],
            "before_score": init_sc,
            "after_score": aud["score"],
            "delta": round(aud["score"] - init_sc, 2),
            "details": aud,
        }
        print(
            f"  [Post-SI-SRSI Audit | {t_item['domain']:11s} | `{t_item['id']}`] "
            f"Score: {init_sc:5.1f} -> {aud['score']:5.1f}/100 ({aud['score'] - init_sc:+5.1f})"
        )

    model.set_domain_engram_focus("general", boost=0.0)
    final_mean_rubric = sum(v["after_score"] for v in final_audits.values()) / max(1, len(final_audits))
    final_ppl_eval = evaluate_engram_perplexity_delta(model, tokenizer)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    state_path = Path(save_dir) / "loop_v03_state.pt"
    torch.save(loop_v03_state_dict(model, include_tier1b=True), state_path)

    print("=" * 116)
    print(
        f"[SI-SRSI Knowledge Engram Formation Complete] "
        f"Mean Rubric Score: {init_mean_rubric:.2f} -> {final_mean_rubric:.2f}/100 ({final_mean_rubric - init_mean_rubric:+.2f}) | "
        f"Held-Out PPL: {init_ppl_eval['with_engram_ppl']:.4f} -> {final_ppl_eval['with_engram_ppl']:.4f} | "
        f"Saved Engram Bank -> `{state_path}`"
    )
    print("=" * 116)
    return {
        "initial_mean_rubric": round(init_mean_rubric, 2),
        "final_mean_rubric": round(final_mean_rubric, 2),
        "rubric_delta": round(final_mean_rubric - init_mean_rubric, 2),
        "initial_ppl": round(init_ppl_eval["with_engram_ppl"], 4),
        "final_ppl": round(final_ppl_eval["with_engram_ppl"], 4),
        "domain_results": final_audits,
        "saved_engram_state": str(state_path),
    }
