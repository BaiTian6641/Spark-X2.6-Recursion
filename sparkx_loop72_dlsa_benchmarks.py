"""Loop72 + D-LSA benchmark adapters, explicit smoke tests and reference values.

Local protocols cover MRCR, GraphWalks BFS and LongBench v2 zero-shot. Other
official suites require their released task environments and evaluators. Dataset
samples, handwritten checks, synthetic performance streams and payload arithmetic
are labelled separately and do not populate published-score comparisons. Full
execution and pinned-dataset provenance belong to the full-suite runner.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import re
import statistics
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    _no_grad = torch.no_grad
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

    def _no_grad():  # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap

from sparkx_kv_codec import KVCodecConfig, compute_loop72_capacity_table, measure_codec_error
from sparkx_si.verifiers import (
    Verdict,
    _extract_json_object,
    answer_part,
    extract_boxed,
    extract_code,
    run_python_tests,
    verify_math,
)


def extract_answer_segment(response_text: str) -> str:
    """Return the answer segment after `</think>` (or fallback to think-stripped text if `<think>` was truncated)."""
    if not response_text:
        return ""
    ans = answer_part(response_text)
    if ans is not None and ans.strip():
        return ans
    # Fallback when generation hit max_new_tokens before closing `</think>`:
    cleaned = re.sub(r"^\s*<think>\s*", "", response_text, flags=re.DOTALL)
    cleaned = cleaned.replace("</think>", "").strip()
    return cleaned if cleaned else response_text


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content inside the last `\\boxed{...}`."""
    return extract_boxed(text)


def _strip_math_units_and_formatting(val_str: str) -> str:
    """Strip common physics units, currency symbols, thousands separators, and LaTeX text wrappers for numeric comparison."""
    s = val_str.strip()
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|hbox)\s*\{[^}]*\}", "", s)
    # Remove common trailing physics/currency units such as m_p c^2, USD, $, %
    s = re.sub(r"(?:m_p\s*c\^2|m_p|c\^2|usd|dollars?|kwh| hours?| bits?| nodes?)", "", s, flags=re.IGNORECASE)
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()
    s = s.rstrip(".")
    return s


def verify_math_answer(response_text: str, gold: str) -> Verdict:
    """Verify a math response against `gold` using `sparkx_si.verifiers.verify_math` with fallback for unclosed
    `<think>` blocks and physics/currency unit-stripped numeric answers.
    """
    res = verify_math(response_text, gold)
    if res.ok:
        return res
    seg = extract_answer_segment(response_text)
    if seg != response_text:
        res_seg = verify_math(seg, gold)
        if res_seg.ok:
            return res_seg
    # Unit-stripped & last-number fallback on boxed content or final answer line
    boxed = extract_boxed(seg) or extract_boxed(response_text)
    candidate_src = boxed if boxed is not None else seg
    cleaned_cand = _strip_math_units_and_formatting(candidate_src)
    cleaned_gold = _strip_math_units_and_formatting(str(gold))
    if cleaned_cand and cleaned_cand == cleaned_gold:
        return Verdict(True, "ok_unit_stripped", {"pred": cleaned_cand, "gold": cleaned_gold})
    try:
        g_val = float(cleaned_gold)
        if cleaned_cand:
            try:
                c_val = float(cleaned_cand)
                if math.isclose(c_val, g_val, rel_tol=1e-6, abs_tol=1e-6):
                    return Verdict(True, "ok_numeric", {"pred": c_val, "gold": g_val})
            except Exception:
                pass
        # Check last Final Answer / Answer / `= <number>` pattern in the answer segment
        tail_matches = re.findall(
            r"(?:final\s+answer|answer|therefore|thus|hence|value\s+is|=)\s*[:=]?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)",
            seg,
            flags=re.IGNORECASE,
        )
        if tail_matches:
            last_num = float(tail_matches[-1].replace(",", ""))
            if math.isclose(last_num, g_val, rel_tol=1e-6, abs_tol=1e-6):
                return Verdict(True, "ok_tail_numeric", {"pred": last_num, "gold": g_val})
    except Exception:
        pass
    return res



def verify_json_schema(response_text: str, schema: Dict[str, Any]) -> Verdict:
    """Verify that `response_text` contains a JSON object matching `required_keys` and `key_types`."""
    ans = extract_answer_segment(response_text)
    obj = _extract_json_object(ans)
    if not isinstance(obj, dict):
        return Verdict(False, "no_json_object", {"score": 0.0})
    req = schema.get("required_keys", [])
    ktypes = schema.get("key_types", {})
    type_map = {"str": str, "int": int, "float": (int, float), "bool": bool, "dict": dict, "list": list}
    matched = 0
    total = max(1, len(req))
    errors: List[str] = []
    for k in req:
        if k not in obj:
            errors.append(f"missing:{k}")
            continue
        expected_t = ktypes.get(k)
        if expected_t and expected_t in type_map:
            py_t = type_map[expected_t]
            if expected_t == "int" and isinstance(obj[k], bool):
                errors.append(f"type:{k}")
                continue
            if not isinstance(obj[k], py_t):
                errors.append(f"type:{k}")
                continue
        matched += 1
    score = matched / total
    return Verdict(len(errors) == 0, "ok" if not errors else "schema_mismatch", {"score": score, "errors": errors, "parsed": obj})


def verify_code_stdio(response_text: str, test_cases: Sequence[Dict[str, str]]) -> Verdict:
    """Execute a Python code block against `stdin -> expected_stdout` test cases inside an isolated temporary directory."""
    import os
    import subprocess
    import sys
    import tempfile

    seg = extract_answer_segment(response_text)
    code = extract_code(seg) or extract_code(response_text) or seg
    passed_cnt = 0
    errors: List[str] = []
    with tempfile.TemporaryDirectory(prefix="sparkx_stdio_sbx_") as tmp_dir:
        script_path = os.path.join(tmp_dir, "solution.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmp_dir,
            "TMPDIR": tmp_dir,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "NO_PROXY": "*",
        }
        for idx, tc in enumerate(test_cases):
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", script_path],
                    input=tc.get("stdin", ""),
                    capture_output=True,
                    text=True,
                    cwd=tmp_dir,
                    env=env,
                    timeout=5.0,
                )
                got = proc.stdout.strip()
                exp = tc.get("expected_stdout", "").strip()
                if proc.returncode == 0 and got == exp:
                    passed_cnt += 1
                else:
                    errors.append(f"case_{idx}:got={got!r},exp={exp!r},rc={proc.returncode}")
            except Exception as exc:
                errors.append(f"case_{idx}:{exc}")
    score = passed_cnt / max(1, len(test_cases))
    return Verdict(
        passed_cnt == len(test_cases),
        "ok" if not errors else "stdio_mismatch",
        {"score": score, "errors": errors, "sandbox": "isolated_tempdir"},
    )


def _apply_unified_diff_or_replace(buggy_code: str, candidate_text: str) -> str:
    """Apply a unified diff patch (`@@ ... @@`) or overlay replacement functions/classes/methods onto `buggy_code`."""
    code_block = extract_code(candidate_text) or candidate_text.strip()
    if not code_block:
        return buggy_code
    lines = code_block.splitlines()
    has_diff_headers = any(ln.startswith("@@") or ln.startswith("---") or ln.startswith("+++") for ln in lines)
    if has_diff_headers:
        reconstructed = buggy_code
        removed: List[str] = []
        added: List[str] = []
        for ln in lines:
            if ln.startswith(("---", "+++", "@@", "diff ", "index ")):
                if removed and "\n".join(removed) in reconstructed:
                    reconstructed = reconstructed.replace("\n".join(removed), "\n".join(added), 1)
                removed, added = [], []
                continue
            if ln.startswith("-"):
                removed.append(ln[1:])
            elif ln.startswith("+"):
                added.append(ln[1:])
        if removed and "\n".join(removed) in reconstructed:
            reconstructed = reconstructed.replace("\n".join(removed), "\n".join(added), 1)
        return reconstructed

    # Dedent if the candidate block was indented as a class method
    import textwrap
    dedented = textwrap.dedent(code_block).strip()
    overlay = buggy_code.rstrip() + "\n\n# --- Candidate Patch Overlay ---\n" + dedented + "\n"

    # If `buggy_code` defines a class `class Foo` and the candidate only emitted a method `def bar(self, ...):`
    # without repeating `class Foo:`, bind `Foo.bar = bar` automatically!
    cls_matches = re.findall(r"^\s*class\s+([A-Za-z_]\w*)\b", buggy_code, flags=re.MULTILINE)
    cand_has_class = bool(re.search(r"^\s*class\s+[A-Za-z_]\w*\b", dedented, flags=re.MULTILINE))
    if cls_matches and not cand_has_class:
        target_cls = cls_matches[-1]
        method_matches = re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(\s*self\b", dedented, flags=re.MULTILINE)
        for m_name in method_matches:
            if m_name != "__init__":
                overlay += f"{target_cls}.{m_name} = {m_name}\n"
    return overlay


def verify_swe_patch_in_sandbox(response_text: str, spec: Dict[str, Any]) -> Verdict:
    """Execute a `SWE Verified` / `SWE Pro` / `SWE Multilingual` patch inside the isolated `run_python_tests`
    sandbox (`tempfile.TemporaryDirectory`, `RLIMIT_AS`, network disabled, cryptographic nonce sentinel).
    """
    seg = extract_answer_segment(response_text)
    buggy_code = str(spec.get("buggy_code", ""))
    test_suite = str(spec.get("test_suite", ""))
    patched_code = _apply_unified_diff_or_replace(buggy_code, seg)
    res = run_python_tests(patched_code, test_suite, timeout=float(spec.get("timeout_s", 6.0)))
    if not res.ok and seg != response_text:
        patched_full = _apply_unified_diff_or_replace(buggy_code, response_text)
        res_full = run_python_tests(patched_full, test_suite, timeout=float(spec.get("timeout_s", 6.0)))
        if res_full.ok:
            res = res_full
    score = 1.0 if res.ok else 0.0
    detail = dict(res.detail or {})
    detail["score"] = score
    detail["sandbox"] = "run_python_tests_isolated_rlimit"
    return Verdict(bool(res.ok), res.reason, detail)


def verify_scicode_in_sandbox(response_text: str, spec: Dict[str, Any]) -> Verdict:
    """Execute a `SciCode` scientific computing task in an isolated sandbox using both unit assertions (`test_suite`)
    and `stdio` numerical test cases (`test_cases`).
    """
    spec = spec.get("scicode_spec", spec)
    if not spec.get("test_cases") and not spec.get("test_suite"):
        return Verdict(False, "unsupported_missing_executable_tests", {"score": None, "status": "unsupported"})
    scores: List[float] = []
    details: Dict[str, Any] = {"sandbox": "isolated_scicode_harness"}
    ok_all = True

    if spec.get("test_cases"):
        stdio_res = verify_code_stdio(response_text, spec["test_cases"])
        scores.append(float(stdio_res.detail.get("score", 1.0 if stdio_res.ok else 0.0)))
        details["stdio"] = stdio_res.detail
        ok_all = ok_all and bool(stdio_res.ok)

    if spec.get("test_suite"):
        seg = extract_answer_segment(response_text)
        code = extract_code(seg) or extract_code(response_text) or seg
        unit_res = run_python_tests(code, str(spec["test_suite"]), timeout=6.0)
        scores.append(1.0 if unit_res.ok else 0.0)
        details["unit_tests"] = unit_res.detail
        ok_all = ok_all and bool(unit_res.ok)

    mean_sc = sum(scores) / max(1, len(scores)) if scores else 0.0
    details["score"] = round(mean_sc, 4)
    return Verdict(ok_all, "ok" if ok_all else "scicode_verification_failed", details)


def _parse_tool_call_candidate(
    text: str,
    expected_tool: str = "",
    expected_arg_names: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Parse a tool call from `<tool_call>...</tool_call>`, JSON (`tool`/`name`/`function` + `arguments`/`parameters`
    or flat keys), or Python AST `func(a, b, k=v)` call syntax (mapping positional args onto `expected_arg_names`).
    """
    import ast

    arg_names_list = list(expected_arg_names or [])

    # 1. Check `<tool_call>...</tool_call>` wrapper first
    tc_match = re.search(r"<(?:\|?tool_call\|?)>\s*(.*?)\s*</(?:\|?tool_call\|?)>", text, flags=re.DOTALL)
    search_space = tc_match.group(1) if tc_match else text

    obj = _extract_json_object(search_space) or _extract_json_object(text)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if isinstance(obj, dict):
        # Unwrap OpenAI `{"tool_calls": [{"function": ...}]}` or `{"function": {"name": ..., "arguments": ...}}`
        if "tool_calls" in obj and isinstance(obj["tool_calls"], list) and obj["tool_calls"]:
            first = obj["tool_calls"][0]
            if isinstance(first, dict):
                obj = first.get("function", first) if isinstance(first.get("function"), dict) else first
        elif isinstance(obj.get("function"), dict):
            obj = obj["function"]
        elif isinstance(obj.get("tool_call"), dict):
            obj = obj["tool_call"]

        tool_name = (
            obj.get("tool")
            or obj.get("name")
            or obj.get("function")
            or obj.get("tool_name")
            or obj.get("action")
            or obj.get("recipient_name")
        )
        args = None
        for a_key in ("arguments", "parameters", "params", "args", "input", "kwargs"):
            if a_key in obj and obj[a_key] is not None:
                args = obj[a_key]
                break
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = _extract_json_object(args)
        if isinstance(args, list) and arg_names_list:
            args = {arg_names_list[i]: args[i] for i in range(min(len(args), len(arg_names_list)))}
        if isinstance(tool_name, str) and isinstance(args, dict):
            # If args is a JSON Schema `{"properties": {...}}`, unwrap `properties`
            if "properties" in args and isinstance(args["properties"], dict) and not any(k in args for k in arg_names_list):
                prop_vals = {}
                for pk, pv in args["properties"].items():
                    prop_vals[pk] = pv.get("default", pv.get("value", pv)) if isinstance(pv, dict) else pv
                args = prop_vals
            return tool_name.strip(), args
        # Check if `{expected_tool: {...}}`
        if len(obj) == 1:
            only_k, only_v = next(iter(obj.items()))
            if isinstance(only_v, dict):
                return str(only_k).strip(), only_v
        # Flat JSON where argument keys are at the top level
        reserved = {"tool", "name", "function", "tool_name", "action", "recipient_name", "arguments", "parameters", "params", "args", "input", "kwargs", "type"}
        flat_args = {k: v for k, v in obj.items() if k not in reserved}
        if flat_args:
            return (str(tool_name).strip() if tool_name else expected_tool), flat_args

    # 2. Fallback: parse Python AST function call `func_name(pos1, pos2, key=val, ...)`
    call_candidates = re.findall(r"([A-Za-z_][\w\.]*\s*\([^()\n]*\))", text)
    for cand in reversed(call_candidates):
        try:
            tree = ast.parse(cand.strip(), mode="eval")
            if isinstance(tree.body, ast.Call):
                fn_node = tree.body.func
                fn_name = fn_node.id if isinstance(fn_node, ast.Name) else getattr(fn_node, "attr", "")
                if fn_name in ("print", "range", "len", "int", "float", "str", "dict", "list", "set", "max", "min"):
                    continue
                kw_dict: Dict[str, Any] = {}
                for idx_pos, pos_node in enumerate(tree.body.args):
                    if idx_pos < len(arg_names_list):
                        kw_dict[arg_names_list[idx_pos]] = ast.literal_eval(pos_node)
                for kw in tree.body.keywords:
                    if kw.arg is not None:
                        kw_dict[kw.arg] = ast.literal_eval(kw.value)
                if fn_name:
                    return fn_name, kw_dict
        except Exception:
            continue

    # 3. Last-resort regex extraction of expected argument values from text when expected_tool is mentioned
    if expected_tool and expected_tool.split(".")[-1].split("__")[-1] in text and arg_names_list:
        extracted_args: Dict[str, Any] = {}
        for k in arg_names_list:
            m_kv = re.search(rf"['\"]?{re.escape(k)}['\"]?\s*[:=]\s*(['\"][^'\"]+['\"]|-?\d+(?:\.\d+)?|true|false|True|False)", text)
            if m_kv:
                raw_v = m_kv.group(1).strip()
                if raw_v.lower() == "true":
                    extracted_args[k] = True
                elif raw_v.lower() == "false":
                    extracted_args[k] = False
                elif raw_v.startswith(("'", '"')):
                    extracted_args[k] = raw_v.strip("'\"")
                elif "." in raw_v:
                    extracted_args[k] = float(raw_v)
                else:
                    extracted_args[k] = int(raw_v)
        if extracted_args:
            return expected_tool, extracted_args

    return None, None


def verify_bfcl_tool_call(response_text: str, spec: Dict[str, Any]) -> Verdict:
    """Verify a `BFCL-V4` / `tau2-bench` / `MCP-Atlas` function-calling response across JSON (`tool`/`name`/`function`),
    `<tool_call>`, and Python AST call formats (both positional and keyword arguments), and execute the live tool
    implementation in the sandbox when provided.
    """
    seg = extract_answer_segment(response_text)
    expected_tool = str(spec.get("expected_tool", spec.get("function_name", ""))).strip()
    if not expected_tool:
        return Verdict(False, "unsupported_missing_function_name", {"score": None, "status": "unsupported"})
    expected_args = spec.get("expected_args", {})
    arg_names = list(expected_args.keys()) if isinstance(expected_args, dict) else list(spec.get("required_arg_names", []))

    tool_name, args = _parse_tool_call_candidate(seg, expected_tool=expected_tool, expected_arg_names=arg_names)
    if tool_name is None or not args:
        tool_name, args = _parse_tool_call_candidate(response_text, expected_tool=expected_tool, expected_arg_names=arg_names)
    if tool_name is None or not isinstance(args, dict):
        return Verdict(False, "no_valid_tool_call", {"score": 0.0})

    errors: List[str] = []
    checks_total = 0
    checks_passed = 0

    if expected_tool:
        checks_total += 1
        if tool_name == expected_tool:
            checks_passed += 1
        else:
            errors.append(f"tool_name:got={tool_name!r},exp={expected_tool!r}")

    if isinstance(expected_args, dict) and not spec.get("allow_extra_args", False):
        extra_args = set(args) - set(expected_args)
        if extra_args:
            errors.append(f"unexpected_args:{sorted(extra_args)}")

    # Normalize argument keys for case-insensitive lookup if needed
    args_norm_map = {str(k).strip().lower(): v for k, v in args.items()}
    canonical_args: Dict[str, Any] = {}

    if isinstance(expected_args, dict) and expected_args:
        for k, exp_v in expected_args.items():
            checks_total += 1
            got_v = args.get(k, args_norm_map.get(k.lower()))
            if got_v is None and k not in args and k.lower() not in args_norm_map:
                errors.append(f"missing_arg:{k}")
                continue
            canonical_args[k] = got_v
            if isinstance(exp_v, bool):
                got_b = got_v if isinstance(got_v, bool) else str(got_v).strip().lower() in ("true", "1", "yes")
                if got_b == exp_v:
                    canonical_args[k] = got_b
                    checks_passed += 1
                else:
                    errors.append(f"arg_val:{k}:got={got_v!r},exp={exp_v!r}")
            elif isinstance(exp_v, (int, float)):
                try:
                    if math.isclose(float(got_v), float(exp_v), rel_tol=1e-3, abs_tol=1e-3):
                        canonical_args[k] = type(exp_v)(float(got_v))
                        checks_passed += 1
                    else:
                        errors.append(f"arg_val:{k}:got={got_v!r},exp={exp_v!r}")
                except Exception:
                    errors.append(f"arg_type:{k}:got={got_v!r},exp={exp_v!r}")
            elif str(got_v).strip().lower() == str(exp_v).strip().lower():
                canonical_args[k] = exp_v
                checks_passed += 1
            else:
                errors.append(f"arg_val:{k}:got={got_v!r},exp={exp_v!r}")
    elif arg_names:
        for k in arg_names:
            checks_total += 1
            if k in args or k.lower() in args_norm_map:
                checks_passed += 1
            else:
                errors.append(f"missing_required_param:{k}")

    # Live sandbox execution of the tool function if `tool_impl_code` is provided
    sandbox_ok = True
    if spec.get("tool_impl_code") and not errors:
        checks_total += 1
        fn_short = expected_tool.split(".")[-1].split("__")[-1]
        call_test = (
            f"args = {repr(canonical_args or args)}\n"
            f"out = {fn_short}(**args)\n"
            f"assert out == {repr(spec['expected_return'])}, f'got {{out!r}}'\n"
        )
        sbx_res = run_python_tests(str(spec["tool_impl_code"]), call_test, timeout=5.0)
        sandbox_ok = bool(sbx_res.ok)
        if sandbox_ok:
            checks_passed += 1
        else:
            errors.append(f"sandbox_exec:{sbx_res.reason}")

    score = checks_passed / max(1, checks_total)
    ok = (len(errors) == 0) and sandbox_ok
    return Verdict(
        ok,
        "ok" if ok else "tool_call_mismatch",
        {
            "score": round(score, 4),
            "tool": tool_name,
            "arguments": canonical_args or args,
            "errors": errors,
            "sandbox_executed": bool(spec.get("tool_impl_code")),
        },
    )


def verify_mcq_choice(response_text: str, gold_letter: Any, gold_text: str = "") -> Verdict:
    """Verify a multiple-choice benchmark answer (`LongBench v2`, `GPQA`, `GPQA Diamond`, `HLE`) against `gold_letter` (`A`/`B`/`C`/`D`) or `mcq_spec` dict."""
    if isinstance(gold_letter, dict):
        gold_text = str(gold_letter.get("gold_text", gold_text))
        gold_letter = str(gold_letter.get("gold_letter", ""))
    seg = extract_answer_segment(response_text)
    boxed = extract_boxed(seg) or extract_boxed(response_text) or ""
    g_let = str(gold_letter).strip().upper()
    cand_box = boxed.strip().strip("().: ")
    if cand_box.upper() == g_let:
        return Verdict(True, "ok_boxed_letter", {"score": 1.0, "pred": cand_box, "gold": g_let})
    if gold_text and gold_text.strip().lower() == (boxed or seg).strip().lower():
        return Verdict(True, "ok_gold_text", {"score": 1.0, "pred": gold_text, "gold": g_let})
    # Check explicit `Answer: X` or `(X)` at tail of segment
    m = re.findall(r"(?:answer|choice|option|select)\s*(?:is|:)?\s*\(?([A-E])\)?\b", seg, flags=re.IGNORECASE)
    if m and m[-1].upper() == g_let:
        return Verdict(True, "ok_tail_letter", {"score": 1.0, "pred": m[-1].upper(), "gold": g_let})
    return Verdict(False, "mcq_mismatch", {"score": 0.0, "pred": (boxed or seg[-80:]), "gold": g_let})


def verify_ifeval_constraints(
    response_text: str,
    instruction_id_list: Any,
    kwargs_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> Verdict:
    """Verify a response against real `google/IFEval` and `allenai/IFBench_test` `instruction_id_list` and `kwargs` rules (or `ifeval_spec` dict)."""
    if isinstance(instruction_id_list, dict):
        kwargs_list = instruction_id_list.get("kwargs", [])
        instruction_id_list = instruction_id_list.get("instruction_id_list", [])
    kwargs_list = list(kwargs_list or [])
    seg = extract_answer_segment(response_text).strip()
    if not instruction_id_list:
        return Verdict(False, "unsupported_no_constraints", {"score": None, "status": "unsupported"})

    passed_rules = 0
    rule_results: List[Dict[str, Any]] = []
    words = re.findall(r"\b\w+\b", seg)
    num_words = len(words)

    def _cmp_rel(val: int, target: int, rel: str) -> bool:
        r = (rel or "at least").lower()
        if "less" in r or "fewer" in r or "at most" in r:
            return val < target if "less" in r else val <= target
        if "at least" in r or "more" in r:
            return val >= target if "at least" in r else val > target
        return val == target

    for idx, inst_id in enumerate(instruction_id_list):
        kw = (kwargs_list[idx] if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict) else {}) or {}
        iid = str(inst_id).lower()
        ok_rule = True

        if "no_comma" in iid:
            ok_rule = "," not in seg
        elif "json_format" in iid:
            ok_rule = isinstance(_extract_json_object(seg), (dict, list))
        elif "number_words" in iid:
            ok_rule = _cmp_rel(num_words, int(kw.get("num_words", 1)), str(kw.get("relation", "at least")))
        elif "number_sentences" in iid:
            sents = [s for s in re.split(r"[.!?]+", seg) if s.strip()]
            ok_rule = _cmp_rel(len(sents), int(kw.get("num_sentences", 1)), str(kw.get("relation", "at least")))
        elif "number_paragraphs" in iid:
            paras = [p for p in re.split(r"\n\s*\n|\*\*\*", seg) if p.strip()]
            ok_rule = len(paras) == int(kw.get("num_paragraphs", 1))
        elif "number_bullet_lists" in iid:
            bullets = re.findall(r"^\s*[\*\-\+]\s+", seg, flags=re.MULTILINE)
            ok_rule = len(bullets) == int(kw.get("num_bullets", 1))
        elif "number_placeholders" in iid:
            phs = re.findall(r"\[[^\]]+\]", seg)
            ok_rule = len(phs) >= int(kw.get("num_placeholders", 1))
        elif "title" in iid:
            ok_rule = bool(re.search(r"<<[^<>]+>>", seg))
        elif "postscript" in iid:
            marker = str(kw.get("postscript_marker", "P.S.")).lower()
            ok_rule = marker in seg.lower() or "p.s" in seg.lower()
        elif "existence" in iid:
            req_kws = kw.get("keywords") or []
            ok_rule = all(str(w).lower() in seg.lower() for w in req_kws)
        elif "forbidden_words" in iid:
            bad_kws = kw.get("forbidden_words") or []
            ok_rule = all(str(w).lower() not in seg.lower() for w in bad_kws)
        elif "frequency" in iid:
            w_target = str(kw.get("keyword", "")).lower()
            cnt = len(re.findall(rf"\b{re.escape(w_target)}\b", seg.lower())) if w_target else 0
            ok_rule = _cmp_rel(cnt, int(kw.get("frequency", 1)), str(kw.get("relation", "at least")))
        elif "english_lowercase" in iid:
            ok_rule = seg == seg.lower()
        elif "english_capital" in iid:
            ok_rule = seg == seg.upper()
        elif "quotation" in iid:
            s_strip = seg.strip()
            ok_rule = len(s_strip) >= 2 and s_strip[0] == '"' and s_strip[-1] == '"'
        elif "end_checker" in iid:
            end_p = str(kw.get("end_phrase", "")).strip().lower()
            ok_rule = seg.strip().lower().endswith(end_p) if end_p else True
        elif "two_responses" in iid:
            ok_rule = "******" in seg
        elif "repeat_prompt" in iid:
            ptxt = str(kw.get("prompt_to_repeat", "")).strip().lower()
            ok_rule = seg.lower().startswith(ptxt) if ptxt else False
        else:
            return Verdict(False, "unsupported_instruction", {
                "score": None, "status": "unsupported", "instruction_id": inst_id,
                "rules": rule_results,
            })

        if ok_rule:
            passed_rules += 1
        rule_results.append({"instruction_id": inst_id, "passed": ok_rule})

    score = passed_rules / max(1, len(instruction_id_list))
    return Verdict(
        passed_rules == len(instruction_id_list),
        "ok" if passed_rules == len(instruction_id_list) else "ifeval_constraint_failed",
        {"score": round(score, 4), "rules": rule_results},
    )


# ==============================================================================
# 1. Official Spark X2.6-4B Benchmark Reference Table & Recommended Sampling
# ==============================================================================
OFFICIAL_SPARK_X26_SAMPLING = {
    "thinking_mode": True,
    "thinking_level": "high",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": -1,
}

OFFICIAL_BENCHMARK_TABLE: Dict[str, Dict[str, Any]] = {
    "Average": {
        "category": "Overall",
        "Spark_X2.6_4B": 63.3,
        "Spark_X2.5_4B": 52.1,
        "Qwen3.5_9B": 49.1,
        "Qwen3.5_4B": 43.2,
        "Gemma4_12B": 47.0,
        "Gemma4_E4B": 29.7,
    },
    # --- Agent ---
    "BFCL-V4": {
        "category": "Agent",
        "hf_dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "Spark_X2.6_4B": 71.8,
        "Spark_X2.5_4B": 65.1,
        "Qwen3.5_9B": 66.1,
        "Qwen3.5_4B": 50.3,
        "Gemma4_12B": 37.4,
        "Gemma4_E4B": 47.0,
    },
    "tau2-bench": {
        "category": "Agent",
        "Spark_X2.6_4B": 86.9,
        "Spark_X2.5_4B": 75.1,
        "Qwen3.5_9B": 79.1,
        "Qwen3.5_4B": 79.9,
        "Gemma4_12B": 69.0,
        "Gemma4_E4B": 42.2,
    },
    "MCP-Atlas": {
        "category": "Agent",
        "Spark_X2.6_4B": 62.4,
        "Spark_X2.5_4B": 54.6,
        "Qwen3.5_9B": 47.4,
        "Qwen3.5_4B": 40.8,
        "Gemma4_12B": 30.5,
        "Gemma4_E4B": 15.0,
    },
    # --- Code ---
    "SWE Pro": {
        "category": "Code",
        "Spark_X2.6_4B": 54.3,
        "Spark_X2.5_4B": 44.4,
        "Qwen3.5_9B": 33.8,
        "Qwen3.5_4B": 29.4,
        "Gemma4_12B": 21.9,
        "Gemma4_E4B": 4.0,
    },
    "SWE Verified": {
        "category": "Code",
        "hf_dataset": "princeton-nlp/SWE-bench_Verified",
        "Spark_X2.6_4B": 62.8,
        "Spark_X2.5_4B": 41.6,
        "Qwen3.5_9B": 53.1,
        "Qwen3.5_4B": 38.8,
        "Gemma4_12B": 44.2,
        "Gemma4_E4B": 14.0,
    },
    "SWE Multilingual": {
        "category": "Code",
        "Spark_X2.6_4B": 64.7,
        "Spark_X2.5_4B": 53.3,
        "Qwen3.5_9B": 43.3,
        "Qwen3.5_4B": 27.7,
        "Gemma4_12B": 32.5,
        "Gemma4_E4B": None,
    },
    "SciCode": {
        "category": "Code",
        "hf_dataset": "SciCode1/SciCode",
        "Spark_X2.6_4B": 36.8,
        "Spark_X2.5_4B": 34.7,
        "Qwen3.5_9B": 32.7,
        "Qwen3.5_4B": 24.0,
        "Gemma4_12B": 39.8,
        "Gemma4_E4B": 27.5,
    },
    # --- Math ---
    "Gaokao 2026": {
        "category": "Math",
        "hf_dataset": "XHToken/gaokao_math_2026",
        "scale": 150.0,
        "Spark_X2.6_4B": 134.9,
        "Spark_X2.5_4B": 133.4,
        "Qwen3.5_9B": 135.5,
        "Qwen3.5_4B": 130.3,
        "Gemma4_12B": 130.6,
        "Gemma4_E4B": 102.4,
    },
    "AIME 2026": {
        "category": "Math",
        "hf_dataset": "MathArena/aime_2026",
        "Spark_X2.6_4B": 91.2,
        "Spark_X2.5_4B": 90.7,
        "Qwen3.5_9B": 88.2,
        "Qwen3.5_4B": 83.0,
        "Gemma4_12B": 82.1,
        "Gemma4_E4B": 42.5,
    },
    "HMMT Feb 26": {
        "category": "Math",
        "hf_dataset": "MathArena/hmmt_feb_2026",
        "Spark_X2.6_4B": 81.3,
        "Spark_X2.5_4B": 81.2,
        "Qwen3.5_9B": 70.8,
        "Qwen3.5_4B": 69.7,
        "Gemma4_12B": 65.6,
        "Gemma4_E4B": 34.2,
    },
    "IMO-AnswerBench": {
        "category": "Math",
        "hf_dataset": "Hwilner/imo-answerbench",
        "Spark_X2.6_4B": 75.2,
        "Spark_X2.5_4B": 74.2,
        "Qwen3.5_9B": 69.8,
        "Qwen3.5_4B": 68.5,
        "Gemma4_12B": 57.2,
        "Gemma4_E4B": 26.9,
    },
    # --- Long Context ---
    "AA-LCRv1.1": {
        "category": "Long Context",
        "hf_dataset": "ArtificialAnalysis/AA-LCR",
        "Spark_X2.6_4B": 67.3,
        "Spark_X2.5_4B": 64.7,
        "Qwen3.5_9B": 70.0,
        "Qwen3.5_4B": 63.0,
        "Gemma4_12B": 63.7,
        "Gemma4_E4B": 32.0,
    },
    "LongBench v2": {
        "category": "Long Context",
        "hf_dataset": "zai-org/LongBench-v2",
        "Spark_X2.6_4B": 53.3,
        "Spark_X2.5_4B": 48.7,
        "Qwen3.5_9B": 55.2,
        "Qwen3.5_4B": 50.0,
        "Gemma4_12B": 45.3,
        "Gemma4_E4B": 27.8,
    },
    "Graphwalks bfs 256k f1": {
        "category": "Long Context",
        "hf_dataset": "openai/graphwalks",
        "Spark_X2.6_4B": 92.2,
        "Spark_X2.5_4B": 47.1,
        "Qwen3.5_9B": 88.4,
        "Qwen3.5_4B": 79.0,
        "Gemma4_12B": 68.6,
        "Gemma4_E4B": 54.8,
    },
    "Graphwalks bfs 1m f1": {
        "category": "Long Context",
        "hf_dataset": "openai/graphwalks",
        "Spark_X2.6_4B": 34.2,
        "Spark_X2.5_4B": 2.5,
        "Qwen3.5_9B": 25.3,
        "Qwen3.5_4B": 15.8,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    "Openai_MRCR v2 256k-512k": {
        "category": "Long Context",
        "hf_dataset": "openai/mrcr",
        "Spark_X2.6_4B": 65.6,
        "Spark_X2.5_4B": 9.7,
        "Qwen3.5_9B": 58.6,
        "Qwen3.5_4B": 48.3,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    "Openai_MRCR v2 512k-1M": {
        "category": "Long Context",
        "hf_dataset": "openai/mrcr",
        "Spark_X2.6_4B": 26.0,
        "Spark_X2.5_4B": 2.9,
        "Qwen3.5_9B": 34.3,
        "Qwen3.5_4B": 12.2,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    # --- General & Knowledge ---
    "IFEval": {
        "category": "General & Knowledge",
        "hf_dataset": "google/IFEval",
        "Spark_X2.6_4B": 93.6,
        "Spark_X2.5_4B": 93.0,
        "Qwen3.5_9B": 91.5,
        "Qwen3.5_4B": 89.8,
        "Gemma4_12B": 94.8,
        "Gemma4_E4B": 45.3,
    },
    "IFBench": {
        "category": "General & Knowledge",
        "hf_dataset": "allenai/IFBench_test",
        "Spark_X2.6_4B": 75.8,
        "Spark_X2.5_4B": 75.0,
        "Qwen3.5_9B": 64.5,
        "Qwen3.5_4B": 59.2,
        "Gemma4_12B": 73.5,
        "Gemma4_E4B": 44.0,
    },
    "HLE(w/o tools)": {
        "category": "General & Knowledge",
        "Spark_X2.6_4B": 17.9,
        "Spark_X2.5_4B": 12.3,
        "Qwen3.5_9B": 14.3,
        "Qwen3.5_4B": 8.6,
        "Gemma4_12B": 13.1,
        "Gemma4_E4B": 3.9,
    },
    "GPQA": {
        "category": "General & Knowledge",
        "hf_dataset": "Idavidrein/gpqa",
        "Spark_X2.6_4B": 69.9,
        "Spark_X2.5_4B": 67.4,
        "Qwen3.5_9B": 77.2,
        "Qwen3.5_4B": 67.2,
        "Gemma4_12B": 72.8,
        "Gemma4_E4B": 54.5,
    },
    "GPQA Diamond": {
        "category": "General & Knowledge",
        "hf_dataset": "Idavidrein/gpqa",
        "Spark_X2.6_4B": 77.1,
        "Spark_X2.5_4B": 73.7,
        "Qwen3.5_9B": 81.7,
        "Qwen3.5_4B": 76.2,
        "Gemma4_12B": 78.8,
        "Gemma4_E4B": 58.6,
    },
    # --- Modern 2026 Frontier Benchmarks (Knowledge Reliability, CLI Agent, Real-World GDP Work) ---
    "AA-Omniscience v1.1": {
        "category": "General & Knowledge",
        "hf_dataset": "ArtificialAnalysis/AA-Omniscience-Public-v1.1",
        "scale": 100.0,
        "Spark_X2.6_4B": None,
        "Spark_X2.5_4B": None,
        "Qwen3.5_9B": None,
        "Qwen3.5_4B": None,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    "Terminal-Bench 4.0": {
        "category": "Agent",
        "hf_dataset": "harborframework/terminal-bench-2.0",
        "Spark_X2.6_4B": None,
        "Spark_X2.5_4B": None,
        "Qwen3.5_9B": None,
        "Qwen3.5_4B": None,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
    "openai/gdpval": {
        "category": "Agent",
        "hf_dataset": "openai/gdpval",
        "Spark_X2.6_4B": None,
        "Spark_X2.5_4B": None,
        "Qwen3.5_9B": None,
        "Qwen3.5_4B": None,
        "Gemma4_12B": None,
        "Gemma4_E4B": None,
    },
}


# ==============================================================================
# 2. Handwritten technical text for synthetic performance diagnostics
# ==============================================================================
MODERN_2026_TEXT_CORPUS: List[Dict[str, str]] = [
    {
        "id": "modern_2026_recurrent_depth_and_mhc",
        "domain": "2025-2026 Frontier AI Systems",
        "title": "Latent Recurrent Depth and Doubly-Stochastic Manifold Hyper-Connections",
        "text": (
            "Recent 2025-2026 architectural studies demonstrate that transformer parameter count and sequential "
            "computational depth are orthogonal axes of capability. Whereas autoregressive chain-of-thought externalizes "
            "intermediate state into discrete vocabulary tokens through the output embedding bottleneck, looped "
            "recurrent-depth transformers iterate a shared core block in continuous high-dimensional representation "
            "space. To prevent spectral norm blow-up across multiple recurrent passes without collapsing the residual "
            "update to zero, Manifold-Constrained Hyper-Connections (mHC) parameterize inter-stream mixing matrices on "
            "the Birkhoff polytope via Sinkhorn-Knopp normalization. Because every doubly-stochastic matrix has spectral "
            "norm bounded by unity, multi-pass state evolution remains non-expansive across arbitrary unroll depths while "
            "allowing each recurrent visit to specialize its query-key attention distribution."
        ),
    },
    {
        "id": "modern_2026_subquadratic_paged_lsa",
        "domain": "2026 Long-Context Attention",
        "title": "Per-Visit Paged Latent Sparse Attention on Hybrid Partial-RoPE Backbones",
        "text": (
            "In hybrid-attention backbones combining local sliding-window layers with periodic global full-attention "
            "layers, the global KV cache dominates memory footprint and memory-bandwidth latency beyond 32,768 tokens. "
            "When the global layers employ partial rotary position embeddings—rotating only the leading 64 dimensions "
            "of each 256-dimensional head while leaving the remaining 192 dimensions position-free—semantic key indexing "
            "can operate directly on the unrotated 192-dimensional subspace without rotary phase distortion. By pooling "
            "contiguous four-token spans and partitioning completed 512-token blocks via mutual-nearest-neighbor Ward "
            "clustering on the 64-dimensional unit hypersphere, per-visit selectors retrieve only the highest-affinity "
            "semantic buckets alongside protected attention sinks and a local sliding tail."
        ),
    },
    {
        "id": "modern_2026_systems_io_uring_zns",
        "domain": "2026 Systems & Storage Engineering",
        "title": "Asynchronous Zero-Copy KV Page Tiering over PCIe Gen5 NVMe Zoned Namespaces",
        "text": (
            "Modern disaggregated inference engines treat host DRAM and PCIe Gen5 NVMe solid-state drives as an "
            "extension of GPU high-bandwidth memory for multi-million-token sessions. Rather than issuing fine-grained "
            "4-kilobyte random reads that suffer from PCIe transaction-layer packet overhead and SSD read amplification, "
            "page-aligned KV archives group 64 contiguous tokens across all four grouped-query KV heads into 128-kilobyte "
            "blocks in 16-bit precision or 36-kilobyte mixed-precision pages that keep rotary key slices in BFloat16 "
            "and quantize free key and value slices to 4-bit group-scaled integers. Coalesced asynchronous page gathers "
            "amortize host-to-device DMA transfers while a bounded GPU LRU cache absorbs temporally coherent decode hits."
        ),
    },
    {
        "id": "modern_2026_quantum_ldpc_fault_tolerance",
        "domain": "2026 Quantum Information & Physics",
        "title": "Constant-Overhead Fault-Tolerant Quantum Computation with Bivariate Bicycle LDPC Codes",
        "text": (
            "High-rate quantum low-density parity-check (qLDPC) codes based on bivariate bicycle constructions achieve "
            "order-of-magnitude reductions in physical-to-logical qubit overhead compared with planar surface codes. "
            "Constructed from commuting check polynomials over the group algebra of a finite abelian group Z_l x Z_m, "
            "these stabilizer codes embed naturally onto two-layer superconducting or neutral-atom syndrome extraction "
            "circuits of constant degree six. Decoding under circuit-level depolarizing noise combines belief propagation "
            "with ordered statistics post-processing (BP-OSD), suppressing logical error rates below 10^-11 per syndrome "
            "cycle using fewer than one-tenth the physical qubits required by distance-matched surface code patches."
        ),
    },
]


def build_modern_long_eval_stream(target_tokens: int = 8192, tokenizer: Any = None) -> str:
    """Build a synthetic performance stream that repeats a small fixed corpus.

    This is useful for capacity/latency diagnostics, not held-out language-model
    quality evaluation. Unique headers do not make the repeated bodies unique.
    """
    blocks: List[str] = []
    sec_idx = 0
    while True:
        for doc in MODERN_2026_TEXT_CORPUS:
            sec_idx += 1
            digest = hashlib.sha256(f"{doc['id']}:{sec_idx}".encode("utf-8")).hexdigest()[:10]
            header = (
                f"\n\n### Section {sec_idx} [{doc['domain']}] — {doc['title']} (Trace {digest})\n"
                f"Metric Audit Parameters: window_id={sec_idx * 64}, page_tokens=64, "
                f"visit_budget={2048 + (sec_idx % 3) * 512}, checksum=0x{digest}.\n"
            )
            blocks.append(header + doc["text"])
        joined = "".join(blocks)
        if tokenizer is not None:
            try:
                n_tok = len(tokenizer.encode(joined, add_special_tokens=False))
            except Exception:
                n_tok = len(joined.split())
        else:
            n_tok = int(len(joined.split()) * 1.35)
        if n_tok >= int(target_tokens):
            return joined


# ==============================================================================
# 3. Synthetic long-context smoke generators and task scoring helpers
# ==============================================================================
def generate_graphwalks_bfs_instance(
    num_nodes: int = 128,
    branching: int = 2,
    target_depth: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a faithful `openai/graphwalks` BFS instance using random 8-char hex node hashes.

    Matches the `openai/graphwalks` specification:
      - Directed graph represented as shuffled edge lines `u -> v` of hexadecimal node hashes, with no filler text.
      - Task: Perform Breadth-First Search (BFS) from `start_node` to exact `target_depth` and return the set of nodes
        at distance `target_depth`.
    """
    rng = random.Random(seed)
    nodes: List[str] = []
    seen = set()
    for i in range(max(16, int(num_nodes))):
        while True:
            h = hashlib.md5(f"gw:{seed}:{i}:{rng.randint(0, 10**9)}".encode("utf-8")).hexdigest()[:8]
            if h not in seen:
                seen.add(h)
                nodes.append(h)
                break

    # Build a layered DAG for layers 0..target_depth+1 so exact BFS depth is unambiguous, plus distractor components
    start_node = nodes[0]
    layer_0 = [start_node]
    cursor = 1
    layers: List[List[str]] = [layer_0]
    edges: List[Tuple[str, str]] = []

    for d in range(1, target_depth + 2):
        prev_layer = layers[-1]
        cur_layer: List[str] = []
        for u in prev_layer:
            for _ in range(branching):
                if cursor < len(nodes):
                    v = nodes[cursor]
                    cursor += 1
                    cur_layer.append(v)
                    edges.append((u, v))
        layers.append(cur_layer)

    # Distractor edges among remaining nodes (disconnected from start_node)
    distractor_nodes = nodes[cursor:]
    for i in range(len(distractor_nodes) - 1):
        u = distractor_nodes[i]
        v = distractor_nodes[(i + 1) % len(distractor_nodes)]
        edges.append((u, v))
        if i + 2 < len(distractor_nodes) and i % 2 == 0:
            edges.append((u, distractor_nodes[i + 2]))

    rng.shuffle(edges)
    edge_lines = "\n".join(f"{u} -> {v}" for u, v in edges)
    gold_nodes = sorted(layers[target_depth])

    prompt = (
        "You are given a directed graph as a list of directed edges `u -> v`:\n\n"
        f"{edge_lines}\n\n"
        f"Question: Starting from node `{start_node}`, find all nodes at exact Breadth-First Search (BFS) "
        f"distance {target_depth} (i.e., reachable in {target_depth} directed hops and no fewer). "
        "Put the matching node IDs on the very last line as `Final Answer: [id1, id2, ...]`. "
        "For no matching nodes, use `Final Answer: []`."
    )
    return {
        "benchmark": "Graphwalks bfs f1",
        "start_node": start_node,
        "target_depth": target_depth,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "gold_nodes": gold_nodes,
        "prompt": prompt,
    }


def score_graphwalks_bfs_f1(response_text: str, gold_nodes: Sequence[str]) -> Dict[str, Any]:
    """Use the upstream final-line parser and exact, case-sensitive set metric.

    Protocol: openai/graphwalks README.md at revision
    be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7, "Extraction and Grading".
    The literal last line must contain ``Final Answer:`` and a bracketed list.
    Do not strip trailing newlines, normalize node identifiers, mine earlier
    prose, or accept boxed-only responses: those change the upstream protocol.
    """
    line = response_text.split("\n")[-1]
    list_part = re.search(r"\[.*\]", line) if "Final Answer:" in line else None
    failed_to_parse = list_part is None
    sampled = list_part.group(0).strip("[]").split(",") if list_part else []
    pred_tokens = {item.strip() for item in sampled if item.strip()}
    gold_set = set(gold_nodes)
    if failed_to_parse:
        prec = rec = f1 = 0.0
    elif not gold_set and not pred_tokens:
        prec = rec = f1 = 1.0
    else:
        tp = len(pred_tokens & gold_set)
        prec = tp / len(pred_tokens) if pred_tokens else 0.0
        rec = tp / len(gold_set) if gold_set else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "predicted": sorted(pred_tokens),
        "gold": sorted(gold_set),
        "failed_to_parse": failed_to_parse,
        "parser": "upstream_final_line",
    }


def generate_mrcr_v2_instance(
    num_rounds: int = 24,
    target_occurrence: int = 3,
    seed: int = 101,
) -> Dict[str, Any]:
    """Generate a faithful `openai/mrcr` (Multi-Round Co-Reference Resolution) instance.

    In `openai/mrcr`, the conversation contains multiple rounds with identical/near-identical user requests
    (e.g. 'Write a short 2-line release note for module Helios') interleaved with other recurring requests
    ('Write a short 2-line release note for module Borealis', etc.). The model must retrieve the exact code/phrase
    from the `target_occurrence`-th occurrence of the target request.
    """
    rng = random.Random(seed)
    topics = ["Helios-Kernel", "Borealis-Router", "Zephyr-Pager", "Chronos-Scheduler"]
    target_topic = topics[0]
    turns: List[str] = []
    topic_counts: Dict[str, int] = {t: 0 for t in topics}
    gold_code: Optional[str] = None

    for r_i in range(max(8, int(num_rounds))):
        # Ensure target_topic appears at least target_occurrence + 1 times
        if r_i in (2, 7, 13, 19) or (topic_counts[target_topic] < target_occurrence and r_i >= num_rounds - 2):
            t_name = target_topic
        else:
            t_name = topics[(r_i + seed) % len(topics)]
        topic_counts[t_name] += 1
        occ = topic_counts[t_name]
        build_tag = f"REV-{t_name[:3].upper()}-{1000 + r_i * 37 + rng.randint(1, 29)}"
        checksum_tag = hashlib.sha1(f"{seed}:{r_i}:{t_name}:{occ}".encode("utf-8")).hexdigest()[:6].upper()
        signature = f"{build_tag}-{checksum_tag}"
        if t_name == target_topic and occ == int(target_occurrence):
            gold_code = signature
        turns.append(
            f"[Turn {r_i + 1}]\n"
            f"User: Generate the official release verification tag for module `{t_name}`.\n"
            f"Assistant: Release verification tag for `{t_name}`: `{signature}` ( deployment window {r_i + 1}, "
            f"latency p99 = {1.2 + (r_i % 7) * 0.3:.1f} ms )."
        )

    assert gold_code is not None, "Target occurrence was not generated"
    transcript = "\n\n".join(turns)
    ordinal_map = {1: "1st", 2: "2nd", 3: "3rd"}
    ord_str = ordinal_map.get(int(target_occurrence), f"{int(target_occurrence)}th")
    prompt = (
        "Below is a multi-turn engineering session log containing repeated release tag requests across several modules:\n\n"
        f"{transcript}\n\n"
        f"Question: What was the exact release verification tag (`REV-...`) returned by the Assistant on the "
        f"**{ord_str}** time the User asked for module `{target_topic}`? "
        "Put ONLY the exact tag inside `\\boxed{...}`."
    )
    return {
        "benchmark": "Openai_MRCR v2",
        "target_topic": target_topic,
        "target_occurrence": int(target_occurrence),
        "num_rounds": int(num_rounds),
        "gold_answer": gold_code,
        "prompt": prompt,
    }


def score_mrcr_v2_response(response_text: str, gold_answer: str, random_string_to_prepend: Optional[str] = None) -> Dict[str, Any]:
    """Use the dataset-card SequenceMatcher metric when a prefix is supplied.

    With no prefix, this helper checks an exact synthetic tag for smoke tests.
    https://huggingface.co/datasets/openai/mrcr (grade function).
    """
    if random_string_to_prepend is not None:
        prefix = str(random_string_to_prepend)
        if not prefix:
            return {"status": "invalid", "score": None, "reason": "missing_mrcr_prefix"}
        prefix_ok = response_text.startswith(prefix)
        score = SequenceMatcher(None, response_text[len(prefix):], gold_answer.removeprefix(prefix)).ratio() if prefix_ok else 0.0
        return {"status": "scored", "score": float(score), "exact_match": response_text == gold_answer,
                "prefix_matched": prefix_ok, "metric": "sequence_match_ratio"}
    ans_seg = extract_answer_segment(response_text)
    boxed = (extract_boxed_answer(ans_seg) or ans_seg).strip().strip("`\"'")
    exact = gold_answer == boxed
    return {
        "exact_match": bool(exact),
        "score": 1.0 if exact else 0.0,
        "extracted": boxed[:80],
        "gold": gold_answer,
    }


def score_aa_omniscience_v11(
    response_text: str,
    gold_answer: str,
    should_abstain: bool = False,
) -> Dict[str, Any]:
    """Score an `AA-Omniscience v1.1` (`ArtificialAnalysis/AA-Omniscience-Public`) item.

    Implements the official Artificial Analysis Omniscience Index rubric (`[-100, +100]`):
      - `+1.0` (`+100` index): Correct factual recall without guessing on unanswerable prompts.
      - `0.0` (`0` index): Calibrated abstention (`ABSTAIN` / `I do not have enough verified information`).
      - `-1.0` (`-100` index): Hallucinated / incorrect factual guess.
    """
    ans_seg = extract_answer_segment(response_text)
    boxed_only = extract_boxed_answer(ans_seg) or extract_boxed_answer(response_text)
    candidate = (boxed_only if boxed_only is not None else ans_seg).strip()
    lower_cand = candidate.lower()
    lower_seg = ans_seg.lower()
    gold_clean = str(gold_answer).strip().lower()

    abstain_markers = (
        "abstain",
        "insufficient information",
        "cannot verify",
        "not enough verified information",
        "i don't know",
        "i do not know",
    )
    did_abstain = any(m in lower_cand for m in abstain_markers)
    if should_abstain:
        if did_abstain:
            return {"outcome": "calibrated_abstention", "omniscience_index": 100.0, "normalized_score": 100.0, "passed": True}
        return {"outcome": "hallucinated_on_unanswerable", "omniscience_index": -100.0, "normalized_score": 0.0, "passed": False}

    # Check exact/numeric/substring match against gold_answer first so mentions of 'ABSTAIN' in reasoning don't mask a valid boxed answer
    if gold_clean and (gold_clean in lower_cand or gold_clean in lower_seg or verify_math_answer(response_text, gold_answer).ok):
        if boxed_only is None or "abstain" not in boxed_only.lower():
            return {"outcome": "correct", "omniscience_index": 100.0, "normalized_score": 100.0, "passed": True}

    if did_abstain:
        return {"outcome": "abstained", "omniscience_index": 0.0, "normalized_score": 50.0, "passed": False}
    return {"outcome": "incorrect_hallucination", "omniscience_index": -100.0, "normalized_score": 0.0, "passed": False}


def verify_terminal_bench_task(response_text: str, spec: Dict[str, Any]) -> Verdict:
    """Verify a `Terminal-Bench 4.0` (`harborframework/terminal-bench-2.0` / `tbench.ai`) CLI engineering task
    using both token/safety constraints and live isolated shell + git sandbox execution.
    """
    import os
    import subprocess
    import sys
    import tempfile

    ans_seg = extract_answer_segment(response_text)
    # Extract bash/sh fenced block if present
    bash_match = re.findall(r"```(?:bash|sh|shell|zsh)?\s*\n(.*?)```", ans_seg, flags=re.DOTALL)
    if not bash_match:
        bash_match = re.findall(r"```(?:bash|sh|shell|zsh)?\s*\n(.*?)```", response_text, flags=re.DOTALL)
    code = bash_match[-1].strip() if bash_match else (extract_code(ans_seg) or ans_seg).strip()
    # Strip leading `$ ` prompt prefixes on lines if present
    cleaned_lines = [re.sub(r"^\s*\$\s+", "", ln) for ln in code.splitlines()]
    code_clean = "\n".join(cleaned_lines).strip()

    req_tokens = spec.get("required_tokens", [])
    forbidden_tokens = spec.get("forbidden_tokens", [])
    missing = [tok for tok in req_tokens if tok not in code_clean and tok not in response_text]
    forbidden_hit = [tok for tok in forbidden_tokens if tok in code_clean]
    token_score = (max(0, len(req_tokens) - len(missing)) / max(1, len(req_tokens))) if not forbidden_hit else 0.0

    sandbox_executed = False
    sandbox_ok = False
    sandbox_detail = ""
    if not forbidden_hit and spec.get("run_git_worktree_sandbox", True):
        try:
            with tempfile.TemporaryDirectory(prefix="tbench4_sbx_") as tmp_dir:
                repo_dir = os.path.join(tmp_dir, "origin_repo")
                worktree_target = os.path.join(tmp_dir, "srv", "loop72-bench")
                os.makedirs(os.path.join(repo_dir, "tests"), exist_ok=True)
                os.makedirs(os.path.join(tmp_dir, "srv"), exist_ok=True)
                with open(os.path.join(repo_dir, "tests", "test_smoke.py"), "w", encoding="utf-8") as f:
                    f.write(
                        "import unittest\n"
                        "class SmokeTest(unittest.TestCase):\n"
                        "    def test_ok(self):\n"
                        "        self.assertEqual(2 + 2, 4)\n"
                    )
                env = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": tmp_dir,
                    "TMPDIR": tmp_dir,
                    "GIT_AUTHOR_NAME": "tbench",
                    "GIT_AUTHOR_EMAIL": "tbench@local",
                    "GIT_COMMITTER_NAME": "tbench",
                    "GIT_COMMITTER_EMAIL": "tbench@local",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True, capture_output=True)
                subprocess.run(["git", "add", "."], cwd=repo_dir, env=env, check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, env=env, check=True, capture_output=True)
                subprocess.run(["git", "branch", "loop72-dlsa"], cwd=repo_dir, env=env, check=True, capture_output=True)

                rewritten_script = code_clean.replace("/srv/loop72-bench", worktree_target)
                rewritten_script = re.sub(r"\bpython3\b", sys.executable, rewritten_script)
                proc = subprocess.run(
                    ["bash", "-e", "-c", rewritten_script],
                    cwd=repo_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=8.0,
                )
                sandbox_executed = True
                wt_exists = os.path.isdir(os.path.join(worktree_target, "tests"))
                sandbox_ok = (proc.returncode == 0) and wt_exists
                sandbox_detail = (proc.stdout + "\n" + proc.stderr).strip()[:240]
        except Exception as exc:
            sandbox_detail = f"sandbox_err:{exc}"

    ok = ((len(missing) == 0) or sandbox_ok) and (len(forbidden_hit) == 0)
    final_score = 1.0 if ok else round(token_score, 4)
    return Verdict(
        ok,
        "ok" if ok else "terminal_command_mismatch",
        {
            "score": final_score,
            "missing": missing,
            "forbidden_hit": forbidden_hit,
            "sandbox_executed": sandbox_executed,
            "sandbox_ok": sandbox_ok,
            "sandbox_output": sandbox_detail,
            "extracted_command": code_clean[:240],
        },
    )


def verify_gdpval_deliverable(response_text: str, spec: Dict[str, Any]) -> Verdict:
    """Verify an `openai/gdpval` (44 occupations across 9 GDP sectors) structured economic deliverable."""
    ans_seg = extract_answer_segment(response_text)
    obj = _extract_json_object(ans_seg) or _extract_json_object(response_text)
    if not isinstance(obj, dict):
        return Verdict(False, "no_json_deliverable", {"score": 0.0})
    req_fields = spec.get("required_fields", {})
    tol = float(spec.get("numeric_tolerance", 1e-2))
    hits = 0
    errors: List[str] = []
    for k, exp_v in req_fields.items():
        if k not in obj:
            errors.append(f"missing:{k}")
            continue
        got_v = obj[k]
        if isinstance(exp_v, (int, float)) and isinstance(got_v, (int, float)) and not isinstance(got_v, bool):
            if abs(float(got_v) - float(exp_v)) <= tol:
                hits += 1
            else:
                errors.append(f"value:{k}:got={got_v},exp={exp_v}")
        elif str(got_v).strip().lower() == str(exp_v).strip().lower():
            hits += 1
        else:
            errors.append(f"value:{k}:got={got_v!r},exp={exp_v!r}")
    score = hits / max(1, len(req_fields))
    return Verdict(len(errors) == 0, "ok" if not errors else "gdpval_rubric_mismatch", {"score": round(score, 4), "errors": errors, "parsed": obj})


def compute_omp_hashline_tag(line_num: int, line_text: str) -> str:
    """Compute the deterministic 2-hex-char content hash tag (`line_num:hh|line_text`) used by `Oh My Pi` (`omp`)
    hashline edit mode (`can1357/oh-my-pi`).
    """
    h2 = hashlib.blake2b(line_text.rstrip("\r\n").encode("utf-8"), digest_size=1).hexdigest()
    return f"{int(line_num)}:{h2}|{line_text.rstrip()}"


def evaluate_oh_my_pi_hashline_task(
    response_text: str,
    source_lines: Sequence[str],
    expected_substring: str,
) -> Verdict:
    """Verify an `Oh My Pi` (`omp`) hashline edit patch (`replace line:hash` -> new content)."""
    ans_seg = extract_answer_segment(response_text)
    obj = _extract_json_object(ans_seg) or _extract_json_object(response_text)
    tagged = {i + 1: hashlib.blake2b(ln.rstrip("\r\n").encode("utf-8"), digest_size=1).hexdigest() for i, ln in enumerate(source_lines)}
    if isinstance(obj, dict):
        anchor = str(obj.get("anchor", "")).strip()
        replacement = str(obj.get("replacement", "")).strip()
    else:
        m = re.search(r"(\d+:[0-9a-fA-F]{2})", ans_seg)
        anchor = m.group(1).lower() if m else ""
        replacement = ans_seg
    if ":" not in anchor:
        return Verdict(False, "missing_hashline_anchor", {"score": 0.0, "anchor": anchor})
    try:
        ln_str, h_str = anchor.split(":", 1)
        ln_idx = int(ln_str)
    except Exception:
        return Verdict(False, "invalid_hashline_anchor", {"score": 0.0, "anchor": anchor})
    exp_hash = tagged.get(ln_idx)
    hash_ok = (exp_hash is not None) and (h_str.lower() == exp_hash.lower())
    content_ok = expected_substring in replacement
    score = (0.5 if hash_ok else 0.0) + (0.5 if content_ok else 0.0)
    return Verdict(
        bool(hash_ok and content_ok),
        "ok" if (hash_ok and content_ok) else "hashline_mismatch",
        {"score": score, "anchor": anchor, "expected_hash": exp_hash, "hash_ok": hash_ok, "content_ok": content_ok},
    )


# ==============================================================================
# 4. Handwritten smoke tasks; not official benchmark instances or scores
# ==============================================================================
SIGNIFICANT_BENCHMARK_SUITE: List[Dict[str, Any]] = [
    # --- Math: Gaokao 2026 / AIME 2026 / HMMT Feb 26 / IMO-AnswerBench ---
    {
        "id": "gaokao_2026_seq_recurrence",
        "benchmark": "Gaokao 2026",
        "category": "Math",
        "official_spark_x26_score": 134.9,
        "prompt": (
            "Let the sequence $\\{a_n\\}$ satisfy $a_1 = 2$ and $a_{n+1} = 2a_n + 3 \\cdot 2^n$ for all $n \\ge 1$. "
            "Dividing by $2^{n+1}$ gives $\\frac{a_{n+1}}{2^{n+1}} - \\frac{a_n}{2^n} = \\frac{3}{2}$, so "
            "$a_n = (3n - 1)2^{n-1}$. Compute the exact value of $a_{8}$. Put your final integer answer in \\boxed{...}."
        ),
        "gold": "2944",
        "verifier": "math",
    },
    {
        "id": "aime_2026_modular_combinatorics",
        "benchmark": "AIME 2026",
        "category": "Math",
        "official_spark_x26_score": 91.2,
        "prompt": (
            "Find the number of ordered triples $(a, b, c)$ of positive integers such that $a + b + c = 30$ "
            "and $a \\le b \\le c < a + b$. (Note: $c < a + b$ with $a + b + c = 30$ is equivalent to $10 \\le c \\le 14$, "
            "and for each $c$, $a$ ranges from $\\max(1, 30 - 2c)$ to $\\lfloor (30 - c)/2 \\rfloor$.) "
            "Put your final integer answer in \\boxed{...}."
        ),
        "gold": "19",
        "verifier": "math",
    },
    {
        "id": "hmmt_feb_2026_polynomial_roots",
        "benchmark": "HMMT Feb 26",
        "category": "Math",
        "official_spark_x26_score": 81.3,
        "prompt": (
            "Let $r_1, r_2, r_3$ be the roots of $P(x) = x^3 - 6x^2 + 11x - 6 = 0$. "
            "Compute $(r_1^2 + r_2^2 + r_3^2)(r_1 r_2 r_3)$. Put your final integer answer in \\boxed{...}."
        ),
        "gold": "84",
        "verifier": "math",
    },
    {
        "id": "imo_answerbench_diophantine_squares",
        "benchmark": "IMO-AnswerBench",
        "category": "Math",
        "official_spark_x26_score": 75.2,
        "prompt": (
            "Find the sum of all positive integers $n$ such that $n^2 + 11n + 52$ is a perfect square. "
            "(Hint: Write $4(n^2 + 11n + 52) = (2n + 11)^2 + 87 = (2m)^2$, so $(2m - 2n - 11)(2m + 2n + 11) = 87 = 1 \\times 87 = 3 \\times 29$.) "
            "Put your final integer answer in \\boxed{...}."
        ),
        "gold": "17",
        "verifier": "math",
    },
    # --- Long Context: AA-LCRv1.1 & LongBench v2 Multi-Document Synthesis ---
    {
        "id": "aa_lcr_v11_cross_doc_synthesis",
        "benchmark": "AA-LCRv1.1",
        "category": "Long Context",
        "official_spark_x26_score": 67.3,
        "prompt": (
            "[Document A — Q1 Infrastructure Filing]\n"
            "Cluster North-Alpha deployed 480 nodes consuming 1.25 kW per node continuously for 2,000 hours in Q1.\n\n"
            "[Document B — Regional Tariff Addendum]\n"
            "Industrial power in Zone North-Alpha is billed at $0.08 per kWh for the first 500,000 kWh and $0.11 per kWh "
            "for all consumption above 500,000 kWh. However, Section 4.2 grants a flat $14,000 clean-grid rebate if the "
            "cluster maintains at least 92% renewable certificates.\n\n"
            "[Document C — Sustainability Audit Log]\n"
            "Cluster North-Alpha recorded 94.6% verified renewable certificates during Q1.\n\n"
            "Question: Synthesizing Documents A, B, and C, what is the net Q1 electricity bill (in USD, as an exact integer) "
            "for Cluster North-Alpha after applying any eligible rebate? Put your final integer answer in \\boxed{...}."
        ),
        "gold": "103000",
        "verifier": "math",
    },
    {
        "id": "longbench_v2_multi_module_trace",
        "benchmark": "LongBench v2",
        "category": "Long Context",
        "official_spark_x26_score": 53.3,
        "prompt": (
            "[File 1: `sparkx_dlsa.py`]\n"
            "In `Loop72`, global full-attention layers `{3, 7, 11, 15, 19, 23, 27, 31, 35}` are visited across "
            "Prelude (`L3`: 1 visit), Pass 1 (`L7, 11, 15, 19, 23, 27`: 6 visits), Pass 2 (`L7, 11, 15, 19, 23, 27`: 6 visits), "
            "Pass 3 (`L15, 19, 23`: 3 visits), and Coda (`L31, 35`: 2 visits).\n\n"
            "[File 2: `sparkx_kv_codec.py`]\n"
            "Each global visit owns an independent `FaithfulDLSAIndexer` with `W_k` of shape `(4, 192, 64)` and `W_q` of "
            "shape `(192, 64)`.\n\n"
            "Question: Based on the repository trace above, how many total global attention visits occur in `Loop72` "
            "when `enable_p3=True`? Put your final integer answer in \\boxed{...}."
        ),
        "gold": "18",
        "verifier": "math",
    },
    # --- General & Knowledge: IFEval / IFBench / HLE(w/o tools) / GPQA / GPQA Diamond ---
    {
        "id": "ifeval_json_schema_compliance",
        "benchmark": "IFEval",
        "category": "General & Knowledge",
        "official_spark_x26_score": 93.6,
        "prompt": (
            "Respond with a valid JSON object inside a ```json ... ``` code block containing exact keys "
            "`\"model\"`, `\"total_visits\"`, and `\"thinking_mode\"`, where `\"model\"` is a string, "
            "`\"total_visits\"` is the integer `72`, and `\"thinking_mode\"` is the boolean `true`."
        ),
        "schema": {
            "required_keys": ["model", "total_visits", "thinking_mode"],
            "key_types": {
                "model": "str",
                "total_visits": "int",
                "thinking_mode": "bool",
            },
        },
        "verifier": "json",
    },
    {
        "id": "ifbench_strict_json_and_constraints",
        "benchmark": "IFBench",
        "category": "General & Knowledge",
        "official_spark_x26_score": 75.8,
        "prompt": (
            "Return a valid JSON object inside a ```json ... ``` block with exact keys "
            "`\"architecture\"`, `\"global_visits\"`, `\"local_visits\"`, and `\"engram_allocated\"`. "
            "Set `\"architecture\"` to `\"Spark-X2.6-4B-Loop72-LSA\"`, `\"global_visits\"` to `18`, "
            "`\"local_visits\"` to `54`, and `\"engram_allocated\"` to `false`."
        ),
        "schema": {
            "required_keys": ["architecture", "global_visits", "local_visits", "engram_allocated"],
            "key_types": {
                "architecture": "str",
                "global_visits": "int",
                "local_visits": "int",
                "engram_allocated": "bool",
            },
        },
        "verifier": "json",
    },
    {
        "id": "hle_no_tools_birkhoff_polytope_dim",
        "benchmark": "HLE(w/o tools)",
        "category": "General & Knowledge",
        "official_spark_x26_score": 17.9,
        "prompt": (
            "Let $\\mathcal{B}_n \\subset \\mathbb{R}^{n \\times n}$ denote the Birkhoff polytope of $n \\times n$ "
            "doubly-stochastic matrices (non-negative matrices whose rows and columns each sum to $1$). "
            "By the Birkhoff-von Neumann theorem, the affine dimension of $\\mathcal{B}_n$ is $(n - 1)^2$. "
            "What is the exact affine dimension of $\\mathcal{B}_5$? Put your final integer answer in \\boxed{...}."
        ),
        "gold": "16",
        "verifier": "math",
    },
    {
        "id": "gpqa_quantum_harmonic_degeneracy",
        "benchmark": "GPQA",
        "category": "General & Knowledge",
        "official_spark_x26_score": 69.9,
        "prompt": (
            "Consider a spinless particle in a 3-dimensional isotropic quantum harmonic oscillator potential "
            "$V(r) = \\frac{1}{2} m \\omega^2 r^2$ with energy eigenvalues $E_N = (N + \\frac{3}{2})\\hbar\\omega$, "
            "where $N = n_x + n_y + n_z$ and $n_x, n_y, n_z \\in \\{0, 1, 2, \\dots\\}$. "
            "What is the exact degeneracy $g(N) = \\binom{N + 2}{2}$ of the energy level with principal quantum number $N = 4$? "
            "Put your final integer answer in \\boxed{...}."
        ),
        "gold": "15",
        "verifier": "math",
    },
    {
        "id": "gpqa_diamond_relativistic_threshold",
        "benchmark": "GPQA Diamond",
        "category": "General & Knowledge",
        "official_spark_x26_score": 77.1,
        "prompt": (
            "A relativistic proton of rest mass $m_p$ collides head-on with a stationary proton of rest mass $m_p$ in the "
            "laboratory frame to produce a proton-antiproton pair alongside the two original protons: "
            "$p + p \\to p + p + p + \\bar{p}$. Using relativistic 4-momentum invariance $s = (p_1 + p_2)^2 c^2 = "
            "2 m_p^2 c^4 + 2 E_{\\text{lab}} m_p c^2 \\ge (4 m_p c^2)^2 = 16 m_p^2 c^4$, what is the minimum total energy "
            "$E_{\\text{lab}}$ of the incident proton in units of $m_p c^2$ (as a pure integer without units)? "
            "Put ONLY the integer inside \\boxed{...}."
        ),
        "gold": "7",
        "verifier": "math",
    },
    # --- Agent: BFCL-V4 / tau2-bench / MCP-Atlas (Live Tool-Call Parser + Sandbox Execution) ---
    {
        "id": "bfcl_v4_tool_selection",
        "benchmark": "BFCL-V4",
        "category": "Agent",
        "official_spark_x26_score": 71.8,
        "prompt": (
            "Available tools:\n"
            "1. `configure_dlsa_cache(local_window: int, global_hot_tokens: int, cold_codec: str)`\n"
            "2. `compile_sleep_engram(anchor_layer: int, ridge_alpha: float)`\n\n"
            "Task: Configure the paged D-LSA cache with `local_window=512`, `global_hot_tokens=8192`, "
            "and `cold_codec=\"mixed_rotary_bf16\"`. Emit a JSON tool call inside a ```json ... ``` block: "
            "`{\"name\": \"configure_dlsa_cache\", \"arguments\": {\"local_window\": 512, \"global_hot_tokens\": 8192, \"cold_codec\": \"mixed_rotary_bf16\"}}`."
        ),
        "tool_spec": {
            "expected_tool": "configure_dlsa_cache",
            "expected_args": {
                "local_window": 512,
                "global_hot_tokens": 8192,
                "cold_codec": "mixed_rotary_bf16",
            },
            "tool_impl_code": (
                "def configure_dlsa_cache(local_window: int, global_hot_tokens: int, cold_codec: str):\n"
                "    return {'ring': int(local_window), 'hot': int(global_hot_tokens), 'codec': str(cold_codec)}\n"
            ),
            "expected_return": {"ring": 512, "hot": 8192, "codec": "mixed_rotary_bf16"},
        },
        "verifier": "bfcl_tool",
    },
    {
        "id": "tau2_bench_sla_credit_policy_agent",
        "benchmark": "tau2-bench",
        "category": "Agent",
        "official_spark_x26_score": 86.9,
        "prompt": (
            "[tau2-bench — Telecom & Cloud SLA Policy Agent]\n"
            "Policy Rules:\n"
            "- Tier-1 Enterprise accounts (`tier=\"enterprise\"`) experiencing downtime between 45 and 120 minutes "
            "receive a `15` percent credit (`credit_pct=15`) and require `notify_tam=true`.\n"
            "- Customer `acct_id=\"ENT-4096\"` (`tier=\"enterprise\"`) had `72` minutes of verified regional outage.\n\n"
            "Available tool:\n"
            "`issue_sla_remediation(acct_id: str, credit_pct: int, notify_tam: bool)`\n\n"
            "Emit a JSON tool call inside a ```json ... ``` block invoking `issue_sla_remediation` with the policy-compliant arguments."
        ),
        "tool_spec": {
            "expected_tool": "issue_sla_remediation",
            "expected_args": {
                "acct_id": "ENT-4096",
                "credit_pct": 15,
                "notify_tam": True,
            },
            "tool_impl_code": (
                "def issue_sla_remediation(acct_id: str, credit_pct: int, notify_tam: bool):\n"
                "    return f'{acct_id}:{int(credit_pct)}:{bool(notify_tam)}'\n"
            ),
            "expected_return": "ENT-4096:15:True",
        },
        "verifier": "bfcl_tool",
    },
    {
        "id": "mcp_atlas_cross_server_invocation",
        "benchmark": "MCP-Atlas",
        "category": "Agent",
        "official_spark_x26_score": 62.4,
        "prompt": (
            "[MCP-Atlas — Multi-Server Model Context Protocol Orchestration]\n"
            "Connected MCP Servers:\n"
            "1. `pager_telemetry__query_visit_recall(visit_id: int, horizon_tokens: int)`\n"
            "2. `kv_tier_controller__pin_hot_pages(visit_id: int, max_hot_pages: int, codec: str)`\n\n"
            "Task: Visit `15` (`L19` in Pass 2) dropped below recall threshold at `65536` tokens. Invoke "
            "`pin_hot_pages` on `kv_tier_controller` for `visit_id=15`, `max_hot_pages=128`, and `codec=\"mixed_rotary_bf16\"`. "
            "Emit a JSON tool call inside a ```json ... ``` block."
        ),
        "tool_spec": {
            "expected_tool": "pin_hot_pages",
            "expected_args": {
                "visit_id": 15,
                "max_hot_pages": 128,
                "codec": "mixed_rotary_bf16",
            },
            "tool_impl_code": (
                "def pin_hot_pages(visit_id: int, max_hot_pages: int, codec: str):\n"
                "    return {'visit_id': int(visit_id), 'pinned_tokens': int(max_hot_pages) * 64, 'codec': str(codec)}\n"
            ),
            "expected_return": {"visit_id": 15, "pinned_tokens": 8192, "codec": "mixed_rotary_bf16"},
        },
        "verifier": "bfcl_tool",
    },
    # --- Code: SWE Pro / SWE Verified / SWE Multilingual / SciCode (Sandboxed Execution Harness) ---
    {
        "id": "swe_pro_lru_page_eviction_bug",
        "benchmark": "SWE Pro",
        "category": "Code",
        "official_spark_x26_score": 54.3,
        "prompt": (
            "[SWE-bench Pro — Stateful Paged KV LRU Eviction Bug]\n"
            "The following `HotPageLRU` class has a bug in `access(page_id)`: when an existing `page_id` is accessed, "
            "it fails to move `page_id` to the most-recently-used end before checking capacity, causing recently accessed "
            "hot pages to be evicted first:\n"
            "```python\n"
            "from collections import OrderedDict\n\n"
            "class HotPageLRU:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = int(capacity)\n"
            "        self.order = OrderedDict()\n\n"
            "    def access(self, page_id: int) -> list:\n"
            "        # BUG: does not refresh existing page_id recency and evicts wrong entry\n"
            "        if len(self.order) >= self.capacity:\n"
            "            self.order.popitem(last=True)\n"
            "        self.order[page_id] = True\n"
            "        return list(self.order.keys())\n"
            "```\n"
            "Write the corrected `HotPageLRU` class inside a ```python ... ``` block so that:\n"
            "1. If `page_id` is already in `self.order`, it is moved to the most-recently-used end (`move_to_end(page_id)`).\n"
            "2. Otherwise, if `len(self.order) >= self.capacity`, the least-recently-used item is evicted (`popitem(last=False)`) "
            "before inserting `page_id`.\n"
            "3. `access(page_id)` returns `list(self.order.keys())` in LRU-to-MRU order."
        ),
        "swe_spec": {
            "buggy_code": (
                "from collections import OrderedDict\n\n"
                "class HotPageLRU:\n"
                "    def __init__(self, capacity: int):\n"
                "        self.capacity = int(capacity)\n"
                "        self.order = OrderedDict()\n"
            ),
            "test_suite": (
                "lru = HotPageLRU(3)\n"
                "assert lru.access(10) == [10]\n"
                "assert lru.access(20) == [10, 20]\n"
                "assert lru.access(30) == [10, 20, 30]\n"
                "assert lru.access(10) == [20, 30, 10], f'Expected [20, 30, 10], got {lru.access(10)}'\n"
                "assert lru.access(40) == [30, 10, 40], f'Expected LRU 20 evicted, got {lru.order}'\n"
            ),
        },
        "verifier": "swe_sandbox",
    },
    {
        "id": "swe_verified_fenwick_range_query_bug",
        "benchmark": "SWE Verified",
        "category": "Code",
        "official_spark_x26_score": 62.8,
        "prompt": (
            "[SWE-bench Verified — Off-by-One Prefix Sum Bug]\n"
            "Fix the bug in `range_token_mass(weights, left, right)` below, which currently excludes `right` and "
            "raises an index error when `left == 0`:\n"
            "```python\n"
            "def range_token_mass(weights: list, left: int, right: int) -> int:\n"
            "    # Should return the inclusive sum sum(weights[left : right + 1])\n"
            "    return sum(weights[left:right])\n"
            "```\n"
            "Provide the corrected `range_token_mass(weights: list, left: int, right: int) -> int` function inside a "
            "```python ... ``` block that returns the inclusive sum `sum(weights[left : right + 1])`."
        ),
        "swe_spec": {
            "buggy_code": (
                "def range_token_mass(weights: list, left: int, right: int) -> int:\n"
                "    return sum(weights[left:right])\n"
            ),
            "test_suite": (
                "assert range_token_mass([4, 7, 2, 9, 5], 0, 2) == 13\n"
                "assert range_token_mass([4, 7, 2, 9, 5], 1, 4) == 23\n"
                "assert range_token_mass([10], 0, 0) == 10\n"
            ),
        },
        "verifier": "swe_sandbox",
    },
    {
        "id": "swe_multilingual_utf8_grapheme_slicer",
        "benchmark": "SWE Multilingual",
        "category": "Code",
        "official_spark_x26_score": 64.7,
        "prompt": (
            "[SWE Multilingual — UTF-8 Byte-Safe Multilingual Prefix Truncation]\n"
            "In a multilingual tokenizer preprocessor handling Chinese, Japanese, Arabic, and accented Latin text, "
            "`truncate_utf8_bytes(text: str, max_bytes: int) -> str` crashes with `UnicodeDecodeError` when slicing "
            "`text.encode('utf-8')[:max_bytes]` across a multi-byte code-point boundary:\n"
            "```python\n"
            "def truncate_utf8_bytes(text: str, max_bytes: int) -> str:\n"
            "    return text.encode('utf-8')[:max_bytes].decode('utf-8')\n"
            "```\n"
            "Write the corrected `truncate_utf8_bytes(text: str, max_bytes: int) -> str` function inside a ```python ... ``` "
            "block so it returns the longest valid UTF-8 prefix of `text` whose UTF-8 encoding is at most `max_bytes` bytes "
            "(e.g. `text.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')`)."
        ),
        "swe_spec": {
            "buggy_code": (
                "def truncate_utf8_bytes(text: str, max_bytes: int) -> str:\n"
                "    return text.encode('utf-8')[:max_bytes].decode('utf-8')\n"
            ),
            "test_suite": (
                "s_zh = '星火递归模型'\n"
                "assert truncate_utf8_bytes(s_zh, 6) == '星火'\n"
                "assert truncate_utf8_bytes(s_zh, 7) == '星火'\n"
                "assert truncate_utf8_bytes('Café-Δ', 5) == 'Café'\n"
                "assert truncate_utf8_bytes('hello', 0) == ''\n"
            ),
        },
        "verifier": "swe_sandbox",
    },
    {
        "id": "scicode_ward_merge_distance",
        "benchmark": "SciCode",
        "category": "Code",
        "official_spark_x26_score": 36.8,
        "prompt": (
            "Write a standalone Python program that reads two lines from standard input:\n"
            "- Line 1: `s_a s_b` (two positive integers representing cluster sizes)\n"
            "- Line 2: four space-separated floats `x1 x2 y1 y2` representing 2D centroids `c_a = (x1, x2)` and `c_b = (y1, y2)`.\n"
            "Compute the exact Ward merge distance $d(a, b) = \\frac{s_a s_b}{s_a + s_b} \\|c_a - c_b\\|_2^2$ and print it "
            "formatted to 4 decimal places (`{val:.4f}`). Wrap your code in a ```python ... ``` block."
        ),
        "test_cases": [
            {"stdin": "4 4\n1.0 0.0 0.0 1.0\n", "expected_stdout": "4.0000"},
            {"stdin": "2 6\n0.0 0.0 2.0 0.0\n", "expected_stdout": "6.0000"},
        ],
        "verifier": "scicode_sandbox",
    },
    # --- Frontier 1: AA-Omniscience v1.1 (Hallucination-Penalized Knowledge & Calibrated Abstention) ---
    {
        "id": "aa_omniscience_v11_ieee_fp8_mantissa",
        "benchmark": "AA-Omniscience v1.1",
        "category": "General & Knowledge",
        "official_spark_x26_score": None,
        "prompt": (
            "[AA-Omniscience v1.1 — Computer Architecture & Numeric Formats]\n"
            "Scoring Rubric: +100 for exact factual answer, 0 for `\\boxed{ABSTAIN}` if uncertain, -100 for an incorrect guess.\n"
            "Question: In the OCP 8-bit floating-point specification `E4M3` (`float8_e4m3fn`), how many bits are allocated "
            "to the significand/mantissa field (excluding the 1 sign bit and 4 exponent bits)? "
            "Put your exact integer answer (or `ABSTAIN`) inside `\\boxed{...}`."
        ),
        "gold": "3",
        "should_abstain": False,
        "verifier": "omniscience",
    },
    # --- Frontier 2: Terminal-Bench 4.0 (`tbench.ai` / `harborframework/terminal-bench-2.0`) ---
    {
        "id": "terminal_bench_4_git_worktree_systemd",
        "benchmark": "Terminal-Bench 4.0",
        "category": "Agent",
        "official_spark_x26_score": None,
        "prompt": (
            "[Terminal-Bench 4.0 — Container & CLI Systems Engineering]\n"
            "Write a non-interactive POSIX shell snippet inside a ```bash ... ``` block that:\n"
            "1. Creates a git worktree at `/srv/loop72-bench` checked out at branch `loop72-dlsa` using `git worktree add`.\n"
            "2. Runs `python3 -m unittest discover -s tests -p \"test_*.py\"` inside `/srv/loop72-bench`.\n"
            "Never use interactive flags (`-i`) or destructive `rm -rf /`."
        ),
        "terminal_spec": {
            "required_tokens": ["git worktree add", "/srv/loop72-bench", "loop72-dlsa", "unittest"],
            "forbidden_tokens": ["rm -rf /"],
            "run_git_worktree_sandbox": True,
        },
        "verifier": "terminal_bench",
    },
    # --- Frontier 3: openai/gdpval (Real-World Economically Valuable Knowledge Work across 44 Occupations) ---
    {
        "id": "gdpval_cloud_finops_capacity_audit",
        "benchmark": "openai/gdpval",
        "category": "Agent",
        "official_spark_x26_score": None,
        "prompt": (
            "[openai/gdpval — Occupation: Principal Cloud Infrastructure & Financial Operations Analyst]\n"
            "Case Brief:\n"
            "- An enterprise deploys a 72-visit looped 4B model serving 1,048,576-token (`1M`) context sessions.\n"
            "- Each session uses `54` local sliding-window rings (`108 MiB` total in BF16) plus `18` global full-attention visits.\n"
            "- At `N = 1,048,576` tokens (`2^20`), each global visit stores `4` KV heads x `256` head_dim x `2` (K and V) elements per token.\n"
            "  - In `16-bit` (`BF16`, `2` bytes/element), `1` global visit = `4 * 1,048,576 * 256 * 2 * 2` bytes = `4.0 GiB`, so `18` global visits = `72.0 GiB`.\n"
            "  - In `4-bit` (`INT4`, `0.5` bytes/element), `18` global visits = `18.0 GiB` (`54.0 GiB` saved per session).\n"
            "  - With `8,192` hot GPU tokens per global visit in BF16 (`18 * 8,192 * 4096 bytes = 576 MiB`) + `108 MiB` local rings, active GPU KV residency is `684 MiB`.\n\n"
            "Deliverable: Emit a valid JSON object inside a ```json ... ``` block with exact keys:\n"
            "- `\"global_bf16_gib\"`: float (`72.0`)\n"
            "- `\"global_int4_gib\"`: float (`18.0`)\n"
            "- `\"gib_saved_per_session\"`: float (`54.0`)\n"
            "- `\"gpu_hot_kv_mib\"`: int (`684`)\n"
            "- `\"recommended_tiering\"`: string (`\"M1_GPU_Hot_M2_Host_Cold\"`)"
        ),
        "gdpval_spec": {
            "required_fields": {
                "global_bf16_gib": 72.0,
                "global_int4_gib": 18.0,
                "gib_saved_per_session": 54.0,
                "gpu_hot_kv_mib": 684,
                "recommended_tiering": "M1_GPU_Hot_M2_Host_Cold",
            },
            "numeric_tolerance": 0.05,
        },
        "verifier": "gdpval",
    },
]


# Handwritten tasks are plumbing checks, independent of their reference names.
for _smoke_item in SIGNIFICANT_BENCHMARK_SUITE:
    _smoke_item["benchmark_id"] = _smoke_item["benchmark"]
    _smoke_item["benchmark"] = "Smoke / " + _smoke_item["benchmark"]
    _smoke_item.update(source="synthetic", evaluation_kind="smoke", status="ready",
                       official_comparable=False, official_spark_x26_score=None,
                       protocol="handwritten_smoke")


def _fetch_hf_rows_via_rest_api(
    repo_id: str,
    config: Optional[str] = None,
    split: str = "train",
    limit: int = 3,
    timeout: float = 8.0,
) -> List[Dict[str, Any]]:
    """Fallback loader using the public Hugging Face Datasets Server REST API (`/rows` and `/first-rows`)."""
    import urllib.parse
    import urllib.request

    configs_to_try = [config] if config else ["default"]
    for cfg in configs_to_try:
        params = urllib.parse.urlencode(
            {
                "dataset": repo_id,
                "config": cfg,
                "split": split,
                "offset": 0,
                "length": max(1, min(int(limit), 20)),
            }
        )
        for endpoint in ("rows", "first-rows"):
            url = (
                f"https://datasets-server.huggingface.co/{endpoint}?{params}"
                if endpoint == "rows"
                else f"https://datasets-server.huggingface.co/first-rows?dataset={urllib.parse.quote(repo_id)}&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}"
            )
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SparkX-Loop72-Benchmark/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                raw_rows = payload.get("rows", [])
                extracted: List[Dict[str, Any]] = []
                for item in raw_rows:
                    r = item.get("row", item) if isinstance(item, dict) else {}
                    if isinstance(r, dict) and r:
                        extracted.append(dict(r))
                        if len(extracted) >= int(limit):
                            break
                if extracted:
                    return extracted
            except Exception:
                continue
    return []


def _load_bfcl_v4_rows_from_hub(max_rows: int = 3) -> List[Dict[str, Any]]:
    """Load real BFCL-V4 rows + ground-truth answers from `gorilla-llm/Berkeley-Function-Calling-Leaderboard` via `hf_hub_download` or direct HTTPS stream."""
    import urllib.request

    repo_id = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

    def _read_lines(rel_path: str) -> List[str]:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            local_p = hf_hub_download(repo_id=repo_id, filename=rel_path, repo_type="dataset")
            with open(local_p, "r", encoding="utf-8") as fh:
                return fh.read().splitlines()
        except Exception:
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel_path}"
            req = urllib.request.Request(url, headers={"User-Agent": "SparkX-Loop72-Benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                return resp.read().decode("utf-8").splitlines()

    for filename in ("BFCL_v4_simple_python.json",):
        try:
            q_lines = _read_lines(filename)
            a_lines = _read_lines(f"possible_answer/{filename}")
            questions: List[Dict[str, Any]] = []
            answers_by_id: Dict[str, Any] = {}
            for line in q_lines:
                line = line.strip()
                if not line:
                    continue
                questions.append(json.loads(line))
                if len(questions) >= int(max_rows):
                    break
            for line in a_lines:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "id" in obj:
                    answers_by_id[str(obj["id"])] = obj.get("ground_truth", [])
            merged: List[Dict[str, Any]] = []
            for q in questions:
                q_id = str(q.get("id", ""))
                q_copy = dict(q)
                if q_id in answers_by_id:
                    q_copy["ground_truth"] = answers_by_id[q_id]
                merged.append(q_copy)
            if merged:
                return merged
        except Exception:
            continue
    return []


class BenchmarkSampleCollection(dict):
    """Legacy mapping of diagnostic samples plus explicit loading outcomes."""

    def __init__(self):
        super().__init__()
        self.load_report: Dict[str, Any] = {}


def try_load_hf_benchmark_samples(max_per_dataset: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    """Download unpinned diagnostic samples, never a full official evaluation.

    Maximum 20 per dataset; use the pinned full runner for more. Split fallback,
    missing datasets and exact returned counts are exposed in ``load_report``.
    Legacy sources across Math, Long Context,
    General Knowledge, Agent Tool Use, and Coding suites (`XHToken/gaokao_math_2026`, `MathArena/aime_2026`,
    `MathArena/hmmt_feb_2026`, `Hwilner/imo-answerbench`, `zai-org/LongBench-v2`, `openai/graphwalks`,
    `openai/mrcr`, `ArtificialAnalysis/AA-LCR`, `google/IFEval`, `allenai/IFBench_test`, `cais/hle`,
    `Idavidrein/gpqa`, `ArtificialAnalysis/AA-Omniscience-Public`, `gorilla-llm/Berkeley-Function-Calling-Leaderboard`,
    `ScaleAI/MCP-Atlas`, `princeton-nlp/SWE-bench_Verified`, `ScaleAI/SWE-bench_Pro`, `SWE-bench/SWE-bench_Multilingual`,
    `SciCode1/SciCode`, `ia03/terminal-bench`, `openai/gdpval`).
    Uses `datasets.load_dataset(..., streaming=True)` first, with automatic fallback to the public
    Hugging Face Datasets Server REST API and `hf_hub_download` for JSONL repositories.
    """
    if max_per_dataset is None or not 0 <= int(max_per_dataset) <= 20:
        raise ValueError("diagnostic sampler accepts 0..20 rows; full evaluation requires the pinned dataset runner")
    loaded = BenchmarkSampleCollection()
    if int(max_per_dataset) == 0:
        return loaded
    has_datasets = False
    try:
        from datasets import load_dataset  # type: ignore

        has_datasets = True
    except Exception:
        load_dataset = None  # type: ignore

    # (benchmark_name, repo_id, config_name, candidate_splits)
    specs: List[Tuple[str, str, Optional[str], Tuple[str, ...]]] = [
        ("Gaokao 2026", "XHToken/gaokao_math_2026", None, ("test", "train")),
        ("AIME 2026", "MathArena/aime_2026", None, ("train", "test")),
        ("HMMT Feb 26", "MathArena/hmmt_feb_2026", None, ("train", "test")),
        ("IMO-AnswerBench", "Hwilner/imo-answerbench", None, ("train", "test")),
        ("LongBench v2", "zai-org/LongBench-v2", None, ("train", "test")),
        ("Graphwalks bfs 256k f1", "openai/graphwalks", None, ("train", "test")),
        ("Openai_MRCR v2 256k-512k", "openai/mrcr", None, ("train", "test")),
        ("AA-LCRv1.1", "ArtificialAnalysis/AA-LCR", None, ("train", "test")),
        ("IFEval", "google/IFEval", None, ("train", "test")),
        ("IFBench", "allenai/IFBench_test", None, ("train", "test")),
        ("HLE(w/o tools)", "cais/hle", None, ("test", "train")),
        ("GPQA", "Idavidrein/gpqa", "gpqa_main", ("train", "test")),
        ("GPQA Diamond", "Idavidrein/gpqa", "gpqa_diamond", ("train", "test")),
        ("AA-Omniscience v1.1", "ArtificialAnalysis/AA-Omniscience-Public", None, ("train", "test")),
        ("MCP-Atlas", "ScaleAI/MCP-Atlas", None, ("train", "test")),
        ("SWE Pro", "ScaleAI/SWE-bench_Pro", None, ("test", "train")),
        ("SWE Verified", "princeton-nlp/SWE-bench_Verified", None, ("test", "train")),
        ("SWE Multilingual", "SWE-bench/SWE-bench_Multilingual", None, ("test", "train")),
        ("SciCode", "SciCode1/SciCode", None, ("test", "train")),
        ("Terminal-Bench 4.0", "ia03/terminal-bench", None, ("train", "test")),
        ("openai/gdpval", "openai/gdpval", None, ("train", "test")),
    ]

    for name, repo_id, cfg_name, splits in specs:
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        selected_split = None
        method = None
        if has_datasets and load_dataset is not None:
            for sp in splits:
                try:
                    ds = (
                        load_dataset(repo_id, cfg_name, split=sp, streaming=True)
                        if cfg_name
                        else load_dataset(repo_id, split=sp, streaming=True)
                    )
                    candidate_rows = []
                    for row in ds:
                        r_dict = dict(row)
                        # For HLE(w/o tools), skip multimodal image rows so text-only evaluation is clean
                        if name == "HLE(w/o tools)" and r_dict.get("image"):
                            continue
                        candidate_rows.append(r_dict)
                        if len(candidate_rows) >= int(max_per_dataset):
                            break
                    rows = candidate_rows
                    if rows:
                        selected_split, method = sp, "datasets_streaming"
                        break
                except Exception as exc:
                    errors.append(f"{sp}:{type(exc).__name__}:{exc}")
                    continue

        if not rows:
            for sp in splits:
                rows = _fetch_hf_rows_via_rest_api(
                    repo_id=repo_id,
                    config=cfg_name,
                    split=sp,
                    limit=int(max_per_dataset),
                )
                if rows:
                    selected_split, method = sp, "dataset_server_initial_rows"
                    break

        loaded[name] = rows
        loaded.load_report[name] = {
            "status": "sampled" if rows else "blocked", "dataset": repo_id, "config": cfg_name,
            "split": selected_split, "method": method, "revision": None,
            "requested_rows": int(max_per_dataset), "returned_rows": len(rows),
            "official_comparable": False, "full_split_complete": False,
            "errors": errors, "reason": None if rows else "dataset_unavailable_or_no_rows",
        }

    # Special JSONL loader for BFCL-V4 (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`)
    bfcl_rows = _load_bfcl_v4_rows_from_hub(max_rows=int(max_per_dataset))
    loaded["BFCL-V4"] = bfcl_rows
    loaded.load_report["BFCL-V4"] = {"status": "sampled" if bfcl_rows else "blocked",
        "category": "v4_simple_python_only", "revision": None, "returned_rows": len(bfcl_rows),
        "full_split_complete": False, "official_comparable": False}

    return loaded


LOCAL_OFFICIAL_PROTOCOLS = {
    "LongBench v2": "longbench_v2_0shot",
    "Graphwalks bfs 256k f1": "graphwalks_bfs_set_f1",
    "Graphwalks bfs 1m f1": "graphwalks_bfs_set_f1",
    "Openai_MRCR v2 256k-512k": "mrcr_sequence_match",
    "Openai_MRCR v2 512k-1M": "mrcr_sequence_match",
}

# Upstream THUDM/LongBench prompts/0shot.txt, blob 81ad9bc1e85ad7ba5b4780fe0fa87745f4bfd964.
LONGBENCH_V2_0SHOT_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n$DOC$\n</text>\n\n"
    "What is the correct answer to this question: $Q$\nChoices:\n"
    "(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


def _first_present(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Select a value without treating valid zero/empty-answer data as missing."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def convert_hf_row_to_eval_task(
    benchmark_name: str,
    row: Dict[str, Any],
    row_idx: int = 0,
    task_mode: str = "diagnostic",
) -> Dict[str, Any]:
    """Convert a dataset row without changing its difficulty or exposing gold.

    Official mode implements only the protocols named in LOCAL_OFFICIAL_PROTOCOLS.
    Other benchmarks require their actual external harness; a substitute verifier
    cannot produce an official score. Diagnostic mode permits explicitly labelled
    local math/MCQ/simple-call checks. Neither mode asserts published comparability.
    Dataset revision, filtering, split and completeness are the runner's contract.
    """
    if task_mode not in ("official", "diagnostic"):
        raise ValueError("task_mode must be 'official' or 'diagnostic'")
    slug = re.sub(r"[^a-z0-9]+", "_", benchmark_name.lower()).strip("_")
    row = row if isinstance(row, dict) else {}
    identity = _first_present(row, "_id", "id", "instance_id", default=row_idx)
    item: Dict[str, Any] = {
        "id": f"hf_{slug}_{identity}", "row_id": identity, "row_index": int(row_idx),
        "benchmark_id": benchmark_name,
        "benchmark": benchmark_name if task_mode == "official" else f"Diagnostic / {benchmark_name}",
        "category": OFFICIAL_BENCHMARK_TABLE.get(benchmark_name, {}).get("category", "Other"),
        "source": "hf_real", "evaluation_kind": task_mode, "status": "ready",
        "official_comparable": False, "official_spark_x26_score": None,
        "score": None, "reason": None, "prompt_truncated": False,
        "protocol": LOCAL_OFFICIAL_PROTOCOLS.get(benchmark_name, "local_diagnostic"),
    }

    def stop(reason: str, status: str = "unsupported") -> Dict[str, Any]:
        item.update(status=status, reason=reason, score=None)
        return item

    if not row:
        return stop("missing_or_invalid_dataset_row", "invalid")
    if task_mode == "official" and benchmark_name not in LOCAL_OFFICIAL_PROTOCOLS:
        return stop("requires_pinned_official_external_harness")

    if benchmark_name == "LongBench v2":
        required = ("context", "question", "choice_A", "choice_B", "choice_C", "choice_D", "answer")
        if any(key not in row or row[key] is None for key in required):
            return stop("missing_longbench_fields", "invalid")
        answer = str(row["answer"]).strip().upper()
        if answer not in "ABCD" or len(answer) != 1:
            return stop("invalid_longbench_answer", "invalid")
        mapping = {"$DOC$": str(row["context"]).strip(), "$Q$": str(row["question"]).strip()}
        mapping.update({f"$C_{c}$": str(row[f"choice_{c}"]).strip() for c in "ABCD"})
        # One replacement pass prevents text inside a document from becoming a template token.
        prompt = re.sub(r"\$DOC\$|\$Q\$|\$C_[A-D]\$", lambda m: mapping[m.group()], LONGBENCH_V2_0SHOT_TEMPLATE)
        item.update(prompt=prompt, verifier="mcq", mcq_spec={"gold_letter": answer, "gold_text": ""},
                    parser="longbench_v2_official", protocol_source="https://github.com/THUDM/LongBench/blob/main/pred.py",
                    protocol_source_blob="1c20cba9edfa3675dc944ee10db5334fbd39a118",
                    protocol_variant="0shot_full_context_no_truncation")
        return item

    if benchmark_name.startswith("Graphwalks bfs"):
        if task_mode == "official" and str(row.get("problem_type", "")).lower() != "bfs":
            return stop("row_is_not_bfs", "filtered")
        prompt = str(row.get("prompt", ""))
        nodes = _first_present(row, "answer_nodes", "gold_nodes", default=None)
        if isinstance(nodes, str):
            try:
                nodes = json.loads(nodes)
            except (ValueError, TypeError):
                return stop("invalid_answer_nodes", "invalid")
        if not prompt or not isinstance(nodes, list):
            return stop("missing_graphwalks_fields", "invalid")
        item.update(prompt=prompt, gold_nodes=[str(node) for node in nodes], verifier="graphwalks_f1",
                    problem_type=row.get("problem_type"), protocol="graphwalks_bfs_set_f1")
        return item

    if benchmark_name.startswith("Openai_MRCR"):
        messages: Any = row.get("prompt")
        for _ in range(2):
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (ValueError, TypeError):
                    break
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str) for m in messages
        ):
            return stop("invalid_mrcr_message_history", "invalid")
        answer = row.get("answer")
        prefix = row.get("random_string_to_prepend")
        if not isinstance(answer, str) or not isinstance(prefix, str) or not prefix or not answer.startswith(prefix):
            return stop("invalid_mrcr_answer_or_prefix", "invalid")
        # Never use desired_msg_index (a gold pointer) to select, shorten or mark turns.
        item.update(messages=[dict(m) for m in messages], gold_answer=answer,
                    random_string_to_prepend=prefix, verifier="mrcr", n_needles=row.get("n_needles"),
                    protocol="mrcr_sequence_match", protocol_source="https://huggingface.co/datasets/openai/mrcr")
        return item

    # These tasks need tool execution, repository/container assets, or released
    # judge protocols. Keyword/patch matching is not a replacement in either mode.
    if benchmark_name in ("MCP-Atlas", "SciCode", "SWE Pro", "SWE Verified", "SWE Multilingual",
                          "Terminal-Bench 4.0", "openai/gdpval", "tau2-bench", "AA-Omniscience v1.1"):
        return stop("requires_task_environment_assets_and_official_evaluator")

    if benchmark_name == "AA-LCRv1.1":
        context = str(_first_present(row, "context", "document"))
        if not context:
            return stop("missing_source_document_assets")
        return stop("requires_official_answer_judge_and_document_assembly")

    if benchmark_name in ("Gaokao 2026", "AIME 2026", "HMMT Feb 26", "IMO-AnswerBench"):
        prompt = str(_first_present(row, "problem", "Problem", "question", "Question", "query", "prompt")).strip()
        answer = str(_first_present(row, "answer", "Short Answer", "target", "final_answer", "ground_truth")).strip()
        if not prompt or not answer:
            return stop("missing_problem_or_answer", "invalid")
        answer = (extract_boxed_answer(answer) or answer).strip("$").strip()
        if len(answer) == 1 and answer.upper() in "ABCD":
            item.update(prompt=prompt, verifier="mcq", mcq_spec={"gold_letter": answer.upper(), "gold_text": ""})
        else:
            item.update(prompt=prompt, verifier="math", gold=answer)
        return item

    if benchmark_name in ("IFEval", "IFBench"):
        prompt = row.get("prompt")
        rules = row.get("instruction_id_list")
        if not isinstance(prompt, str) or not isinstance(rules, list) or not rules:
            return stop("missing_instruction_fields", "invalid")
        parsed = []
        for val in row.get("kwargs", []):
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except (ValueError, TypeError):
                    return stop("invalid_instruction_kwargs", "invalid")
            parsed.append(val or {})
        item.update(prompt=prompt, verifier="ifeval", ifeval_spec={"instruction_id_list": rules, "kwargs": parsed})
        return item

    if benchmark_name in ("GPQA", "GPQA Diamond"):
        prompt = str(_first_present(row, "Question", "question"))
        choices = [str(_first_present(row, "Correct Answer", "correct_answer"))]
        choices += [str(_first_present(row, f"Incorrect Answer {i}", f"incorrect_answer_{i}")) for i in (1, 2, 3)]
        if not prompt or any(not c for c in choices):
            return stop("missing_gpqa_fields", "invalid")
        order = list(range(4))
        seed = int(hashlib.sha256((prompt + str(identity)).encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(order)
        gold = "ABCD"[order.index(0)]
        prompt += "\n\n" + "\n".join(f"({c}) {choices[i]}" for c, i in zip("ABCD", order))
        item.update(prompt=prompt, verifier="mcq", mcq_spec={"gold_letter": gold, "gold_text": ""}, choice_seed=seed)
        return item

    if benchmark_name == "HLE(w/o tools)":
        return stop("requires_official_hle_judge_and_modality_protocol")

    if benchmark_name == "BFCL-V4":
        # Intentionally restricted simple-call diagnostic. Multi-turn/parallel and
        # alternative-valued gold require BFCL's AST/execution evaluator.
        questions, functions, gold = row.get("question"), row.get("function"), row.get("ground_truth")
        if not questions or not functions or not isinstance(gold, list) or len(gold) != 1:
            return stop("unsupported_bfcl_category_or_missing_fields")
        if not isinstance(gold[0], dict) or len(gold[0]) != 1:
            return stop("unsupported_bfcl_parallel_calls")
        function, args = next(iter(gold[0].items()))
        if not isinstance(args, dict):
            return stop("invalid_bfcl_ground_truth", "invalid")
        expected = {}
        for key, val in args.items():
            if isinstance(val, list):
                if len(val) != 1 or val[0] == "":
                    return stop("requires_bfcl_alternative_argument_evaluator")
                val = val[0]
            expected[key] = val
        text = questions if isinstance(questions, str) else json.dumps(questions, ensure_ascii=False)
        item.update(prompt=f"Available tools:\n{json.dumps(functions, ensure_ascii=False)}\n\nUser request:\n{text}",
                    verifier="bfcl_tool", tool_spec={"expected_tool": function, "function_name": function,
                    "expected_args": expected, "allow_extra_args": False}, protocol="simple_call_diagnostic")
        return item
    return stop("no_supported_converter")


# ==============================================================================
# 5. Live Model Evaluation Runners (36L Base vs 72L Loop vs 72L Loop + Paged D-LSA)
# ==============================================================================
@_no_grad()
def evaluate_modern_text_perplexity_and_dlsa(
    model: Any,
    tokenizer: Any,
    seq_len: int = 4096,
    calibrate_pca: bool = False,
    calibration_input_ids: Optional[Any] = None,
    eval_text: Optional[str] = None,
    evaluation_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a bounded dense/sparse diagnostic on identical evaluation tokens.

    The default repeated technical corpus is synthetic, not held-out validation.
    PCA adaptation is opt-in and requires a separately supplied calibration tensor;
    callers must establish its provenance/disjointness. Search-mode and threshold
    overrides are always restored, including when a forward or scoring call fails.
    Timings include forward execution and CUDA synchronization, excluding NLL work.
    """
    if calibrate_pca and calibration_input_ids is None:
        raise ValueError("calibrate_pca requires separate calibration_input_ids; evaluation tokens cannot calibrate the selector")
    if not HAS_TORCH:
        raise RuntimeError("evaluate_modern_text_perplexity_and_dlsa requires PyTorch.")
    if int(seq_len) < 2:
        raise ValueError("seq_len must be at least two")
    device = next(model.base.parameters()).device
    raw_text = eval_text if eval_text is not None else build_modern_long_eval_stream(target_tokens=int(seq_len) + 256, tokenizer=tokenizer)
    ids = tokenizer.encode(raw_text, add_special_tokens=False)[:int(seq_len)]
    if len(ids) < 2:
        raise ValueError("evaluation text must contain at least two tokens")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    actual_len = len(ids)
    bank = getattr(model, "dlsa_index_bank", None)
    sparse_cfg = getattr(model, "sparse_attention_cfg", None)
    saved_threshold = getattr(sparse_cfg, "exact_threshold", None)
    saved_modes = [(indexer.cfg, indexer.cfg.search_mode) for indexer in bank.visit_indexers.values()] if bank is not None else []
    was_training = getattr(model, "training", None)
    pca_info = None
    results: Dict[str, Any] = {}
    ledger_snapshot = None

    def sync() -> None:
        if getattr(device, "type", str(device).split(":")[0]) == "cuda":
            torch.cuda.synchronize(device)

    def nll_ppl(logits: Any) -> Tuple[float, float]:
        # Bound FP32 vocabulary workspace instead of copying every token's logits.
        total_loss, total_labels = 0.0, 0
        for start in range(0, actual_len - 1, 64):
            end = min(actual_len - 1, start + 64)
            labels = input_ids[:, start + 1:end + 1].contiguous().view(-1)
            loss = F.cross_entropy(logits[:, start:end].float().reshape(-1, logits.shape[-1]), labels, reduction="sum")
            total_loss += float(loss.item())
            total_labels += int(labels.numel())
        nll = total_loss / total_labels
        return nll, math.exp(nll) if nll < 700.0 else float("inf")

    def timed_forward(cache: Any = None) -> Tuple[Any, float]:
        sync()
        started = time.perf_counter()
        if cache is None:
            logits = model.forward_logits(input_ids)
        else:
            logits = model.forward_logits(input_ids, past_key_values=cache,
                                          chunk_size=getattr(model.settings, "prefill_chunk_tokens", 2048))
        sync()
        return logits, time.perf_counter() - started

    try:
        if hasattr(model, "eval"):
            model.eval()
        if calibrate_pca:
            if bank is None:
                raise ValueError("calibration requested but no D-LSA index bank is attached")
            cal = calibration_input_ids.to(device)
            if cal.ndim != 2 or cal.shape[0] != 1 or cal.shape[1] < 2:
                raise ValueError("calibration_input_ids must have shape [1, >=2]")
            smaller = min(int(cal.shape[1]), actual_len)
            if torch.equal(cal[:, :smaller], input_ids[:, :smaller]):
                raise ValueError("calibration tokens overlap the evaluation prefix")
            from train_spark_dlsa import capture_loop72_global_visit_qkv, fit_loop72_visit_bank_pca
            captured = capture_loop72_global_visit_qkv(model, cal)
            pca_info = fit_loop72_visit_bank_pca(bank, {gv: [(q, k)] for gv, (q, k, _) in captured.items()})
        if sparse_cfg is not None:
            sparse_cfg.exact_threshold = actual_len + 1024
        with model.loop_config(passes=1, use_mhc=False, use_engram=False, enable_p3=False, use_esm=False):
            logits36, latency = timed_forward()
        nll36, ppl36 = nll_ppl(logits36)
        results["36L_Base_Dense"] = {"effective_visits": 36, "nll": nll36, "ppl": ppl36, "latency_sec": latency}
        del logits36
        dense_cache = model.make_inference_cache(session_id="ppl_dense")
        dense, latency = timed_forward(dense_cache)
        nll_dense, ppl_dense = nll_ppl(dense)
        results["72L_Loop_Dense"] = {"effective_visits": model.total_virtual_layers,
                                     "nll": nll_dense, "ppl": ppl_dense, "latency_sec": latency,
                                     "delta_nll_vs_36L": nll_dense - nll36}
        del dense_cache
        if bank is not None and sparse_cfg is not None:
            sparse_cfg.exact_threshold = min(1024, max(1, actual_len // 2))
            for mode in ("flat", "tree"):
                for config, _ in saved_modes:
                    config.search_mode = mode
                cache = model.make_inference_cache(session_id=f"ppl_{mode}")
                logits, latency = timed_forward(cache)
                nll, ppl = nll_ppl(logits)
                absolute_sum, elements = 0.0, 0
                for start in range(0, actual_len, 64):
                    delta = (logits[:, start:start + 64].float() - dense[:, start:start + 64].float()).abs()
                    absolute_sum += float(delta.sum().item())
                    elements += int(delta.numel())
                results[f"72L_Loop_DLSA_{mode.capitalize()}"] = {
                    "effective_visits": model.total_virtual_layers, "search_mode": mode,
                    "nll": nll, "ppl": ppl, "latency_sec": latency,
                    "delta_nll_vs_72L_dense": nll - nll_dense, "delta_nll_vs_36L": nll - nll36,
                    "mean_abs_logit_diff_vs_72L_dense": absolute_sum / max(1, elements),
                }
                if ledger_snapshot is None:
                    ledger_snapshot = cache.memory_ledger()
                del logits, cache
        del dense
    finally:
        if sparse_cfg is not None:
            sparse_cfg.exact_threshold = saved_threshold
        for config, search_mode in saved_modes:
            config.search_mode = search_mode
        if was_training is not None and hasattr(model, "train"):
            model.train(was_training)
    return {"status": "completed", "seq_len": actual_len,
            "evaluation_kind": "supplied_text_diagnostic" if eval_text is not None else "synthetic_repeated_text_diagnostic",
            "held_out_quality_verified": False, "evaluation_provenance": evaluation_provenance,
            "pca_calibrated_visits": len(pca_info) if pca_info else 0,
            "calibration_provenance_verified": False if calibrate_pca else None,
            "timing_scope": "synchronized_forward_only_excludes_nll",
            "modes": results, "paged_cache_memory_ledger": ledger_snapshot}


def _task_result_base(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("id", "row_id", "row_index", "benchmark", "benchmark_id", "category", "source",
            "evaluation_kind", "protocol", "protocol_variant", "protocol_source", "protocol_source_blob")
    result = {key: item[key] for key in keys if key in item}
    result.update(status=item.get("status", "ready"), score=None, raw_score=None, passed=None,
                  official_comparable=False, reason=item.get("reason"), detail={})
    return result


def score_benchmark_task(item: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Score one final answer; unavailable evaluators are never recorded as 0/100.

    Scores are percentages of the named raw metric. They are only aggregatable
    within an identical benchmark/protocol/metric, never as a suite-wide average.
    Official prompts and scorers do not establish sampling/release comparability.
    """
    result = _task_result_base(item)
    if result["status"] not in ("ready", "completed"):
        return result
    kind, verifier = item.get("evaluation_kind", "diagnostic"), item.get("verifier", "")
    if kind == "official" and item.get("protocol") not in LOCAL_OFFICIAL_PROTOCOLS.values():
        result.update(status="unsupported", reason="requires_official_evaluator")
        return result
    try:
        detail: Dict[str, Any]
        if verifier == "graphwalks_f1":
            detail = score_graphwalks_bfs_f1(response, item["gold_nodes"])
            raw, passed, metric = float(detail["f1"]), detail["f1"] == 1.0, "node_set_f1"
        elif verifier == "mrcr":
            detail = score_mrcr_v2_response(response, item["gold_answer"], item.get("random_string_to_prepend"))
            if detail.get("score") is None:
                result.update(status=detail.get("status", "invalid"), reason=detail.get("reason"), detail=detail)
                return result
            raw, passed = float(detail["score"]), bool(detail["exact_match"])
            metric = "sequence_match_ratio" if "random_string_to_prepend" in item else "synthetic_tag_exact_match"
        elif verifier == "mcq" and item.get("parser") == "longbench_v2_official":
            # Exact upstream parser from pred.py blob 1c20cba9edfa3675dc944ee10db5334fbd39a118.
            text = response.replace("*", "")
            match = re.search(r"The correct answer is \(([A-D])\)", text)
            if match is None:
                match = re.search(r"The correct answer is ([A-D])", text)
            predicted = match.group(1) if match else None
            passed = predicted == item["mcq_spec"]["gold_letter"]
            raw, metric, detail = float(passed), "accuracy", {"predicted": predicted}
        else:
            verdicts = {
                "math": lambda: verify_math_answer(response, item["gold"]),
                "mcq": lambda: verify_mcq_choice(response, item["mcq_spec"]),
                "ifeval": lambda: verify_ifeval_constraints(response, item["ifeval_spec"]),
                "json": lambda: verify_json_schema(response, item["schema"]),
                "bfcl_tool": lambda: verify_bfcl_tool_call(response, item["tool_spec"]),
            }
            # Executable toy tasks stay explicit smoke tests. HF repository or
            # agent tasks must go through their external task environments.
            if kind == "smoke":
                verdicts.update({
                    "swe_sandbox": lambda: verify_swe_patch_in_sandbox(response, item["swe_spec"]),
                    "scicode_sandbox": lambda: verify_scicode_in_sandbox(response, item),
                    "code": lambda: verify_code_stdio(response, item["test_cases"]),
                    "terminal_bench": lambda: verify_terminal_bench_task(response, item["terminal_spec"]),
                    "gdpval": lambda: verify_gdpval_deliverable(response, item["gdpval_spec"]),
                })
            if verifier == "omniscience" and kind == "smoke":
                detail = score_aa_omniscience_v11(response, item["gold"], bool(item.get("should_abstain", False)))
                raw, passed, metric = detail["normalized_score"] / 100.0, bool(detail["passed"]), "toy_abstention_score"
            elif verifier in verdicts:
                verdict = verdicts[verifier]()
                detail = dict(verdict.detail or {})
                if detail.get("status") == "unsupported" or ("score" in detail and detail["score"] is None):
                    result.update(status="unsupported", reason=verdict.reason, detail=detail)
                    return result
                raw = float(detail.get("score", float(verdict.ok)))
                passed, metric = bool(verdict.ok), f"diagnostic_{verifier}"
            else:
                result.update(status="unsupported", reason="no_executable_official_evaluator")
                return result
        result.update(status="completed", score=100.0 * raw, raw_score=raw, passed=bool(passed),
                      metric=metric, reason=None, detail=detail)
    except Exception as exc:
        result.update(status="error", reason=f"scoring_error:{type(exc).__name__}:{exc}")
    return result


def _split_model_response(raw: str, text: str, prompt_has_open_think: bool = False) -> Dict[str, Any]:
    """Keep reasoning out of answer-only scorers without inventing closing tokens."""
    if "</think>" in raw:
        reasoning, final_raw = raw.rsplit("</think>", 1)
        final = text.rsplit("</think>", 1)[-1] if "</think>" in text else final_raw
        return {"response": final, "reasoning": reasoning.removeprefix("<think>"), "thinking_complete": True}
    if prompt_has_open_think or "<think>" in raw:
        return {"response": "", "reasoning": text, "thinking_complete": False}
    return {"response": text, "reasoning": "", "thinking_complete": True}


def _configured_context_limit(model: Any) -> Optional[int]:
    for cfg in (getattr(model, "paged_cache_cfg", None), getattr(model, "settings", None)):
        value = getattr(cfg, "context_limit_tokens", None)
        if value is not None:
            return int(value)
    return None


@_no_grad()
def evaluate_benchmark_task(
    model: Any,
    tokenizer: Any,
    item: Dict[str, Any],
    max_new_tokens: int = 8192,
    generation_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate and score one row, preserving raw conversations and token budgets.

    No gold enters generation, no prompts are truncated, and no extra generation
    is hidden outside max_new_tokens. Context overflow and template failures are
    reported explicitly. This public single-row API supports streaming runners.
    """
    result = _task_result_base(item)
    if result["status"] != "ready":
        return result
    if not HAS_TORCH or model is None or tokenizer is None:
        result.update(status="blocked", reason="model_tokenizer_and_pytorch_required")
        return result
    if int(max_new_tokens) <= 0:
        result.update(status="invalid", reason="max_new_tokens_must_be_positive")
        return result
    options: Dict[str, Any] = {
        "do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
        "force_think_close": False, "min_think_tokens": 0,
    }
    options.update(generation_options or {})
    reserved = {"input_ids", "cache", "past_key_values", "tokenizer", "stream_print", "max_new_tokens"}
    if reserved.intersection(options):
        result.update(status="invalid", reason="generation_options_contains_reserved_keys")
        return result
    if item.get("evaluation_kind") == "official" and (options.get("force_think_close")
            or options.get("repetition_penalty") != 1.0 or options.get("no_repeat_ngram_size", 0) != 0
            or options.get("min_think_tokens", 0) != 0):
        result.update(status="invalid", reason="official_protocol_forbids_answer_forcing_or_repetition_bias")
        return result
    if "force_think_close" not in inspect.signature(model.generate_cached).parameters:
        result.update(status="unsupported", reason="runtime_cannot_disable_forced_thinking_close")
        return result
    messages = item.get("messages")
    if messages is None:
        if not isinstance(item.get("prompt"), str) or not item["prompt"]:
            result.update(status="invalid", reason="missing_prompt")
            return result
        messages = [{"role": "user", "content": item["prompt"]}]
    try:
        chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(chat_ids, dict):
            chat_ids = chat_ids["input_ids"]
        elif hasattr(chat_ids, "input_ids"):
            chat_ids = chat_ids.input_ids
        if hasattr(chat_ids, "tolist"):
            chat_ids = chat_ids.tolist()
        if chat_ids and isinstance(chat_ids[0], list):
            if len(chat_ids) != 1:
                raise ValueError("single task requires one prompt")
            chat_ids = chat_ids[0]
        if not chat_ids:
            raise ValueError("empty chat template output")
        prompt_text = tokenizer.decode(chat_ids, skip_special_tokens=False)
    except Exception as exc:
        result.update(status="unsupported", reason=f"chat_template_error:{type(exc).__name__}:{exc}")
        return result
    result.update(prompt_tokens=len(chat_ids), max_new_tokens=int(max_new_tokens),
                  generation_options=options.copy(), prompt_truncated=False,
                  prompt_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    context_limit = _configured_context_limit(model)
    if context_limit is not None and len(chat_ids) + int(max_new_tokens) > context_limit:
        result.update(status="blocked", reason="prompt_plus_output_budget_exceeds_configured_context",
                      configured_context_limit=context_limit)
        return result
    started = time.perf_counter()
    try:
        device = next(model.base.parameters()).device
        inp = torch.tensor([chat_ids], dtype=torch.long, device=device)
        cache = model.make_inference_cache(session_id=f"bench_{item.get('id', 'task')}")
        model.last_generation_timing = None
        output = model.generate_cached(inp, max_new_tokens=int(max_new_tokens), tokenizer=tokenizer,
                                       stream_print=bool(verbose), cache=cache, **options)
        token_ids = output[0, len(chat_ids):].tolist()
        if len(token_ids) > int(max_new_tokens):
            raise RuntimeError("runtime exceeded max_new_tokens")
        raw = tokenizer.decode(token_ids, skip_special_tokens=False)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt_has_open_think = prompt_text.rfind("<think>") > prompt_text.rfind("</think>")
        separated = _split_model_response(raw, text, prompt_has_open_think)
        for special in getattr(tokenizer, "all_special_tokens", []):
            separated["response"] = separated["response"].replace(special, "")
        timing = dict(getattr(model, "last_generation_timing", None) or {})
        result.update(raw_response=raw, **separated, generated_tokens=len(token_ids),
                      wall_time_sec=time.perf_counter() - started, timing=timing,
                      finish_reason=timing.get("finish_reason", "length" if len(token_ids) == int(max_new_tokens) else "unknown"))
        # An unfinished think segment is observable model output. Score its empty
        # final answer under the task protocol; never mine reasoning for gold.
        result.update(score_benchmark_task(item, separated["response"]))
        result["output_incomplete"] = not separated["thinking_complete"]
    except Exception as exc:
        result.update(status="error", score=None, raw_score=None, passed=None,
                      reason=f"generation_error:{type(exc).__name__}:{exc}",
                      wall_time_sec=time.perf_counter() - started)
    return result


@_no_grad()
def evaluate_selected_benchmark_tasks(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 8192,
    graphwalks_nodes: int = 64,
    mrcr_rounds: int = 16,
    hf_samples: Optional[Dict[str, Any]] = None,
    include_hf_streamed: bool = True,
    max_hf_per_benchmark: Optional[int] = None,
    include_smoke: bool = False,
    task_mode: str = "official",
    generation_options: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate supplied rows with explicit scope and no implicit synthetic tasks.

    None means every supplied row, including generators. This does not establish
    that supplied rows are a complete official split: the full runner records and
    validates dataset revisions, filters and expected row counts independently.
    Unpinned convenience downloads are available only in diagnostic sample mode.
    """
    if task_mode not in ("official", "diagnostic"):
        raise ValueError("task_mode must be 'official' or 'diagnostic'")
    if max_hf_per_benchmark is not None and int(max_hf_per_benchmark) < 0:
        raise ValueError("max_hf_per_benchmark cannot be negative")
    reports: List[Dict[str, Any]] = []
    dataset_reports: Dict[str, Any] = {}
    setup_reason = None
    if hf_samples is None and include_hf_streamed:
        if task_mode == "official" or max_hf_per_benchmark is None:
            hf_samples = {}
            setup_reason = "full_or_official_execution_requires_explicit_pinned_dataset_rows"
        else:
            hf_samples = try_load_hf_benchmark_samples(max_per_dataset=int(max_hf_per_benchmark))
    if include_hf_streamed:
        for name, rows in (hf_samples or {}).items():
            consumed = 0
            known_size = len(rows) if hasattr(rows, "__len__") else None
            exhausted = True
            for index, row in enumerate(rows):
                if max_hf_per_benchmark is not None and consumed >= int(max_hf_per_benchmark):
                    exhausted = False
                    break
                consumed += 1
                task = convert_hf_row_to_eval_task(name, row, index, task_mode=task_mode)
                if verbose:
                    print(f"[EvalTask | {name} #{consumed}] starting...", flush=True)
                    res = evaluate_benchmark_task(model, tokenizer, task, max_new_tokens, generation_options, verbose=True)
                else:
                    res = evaluate_benchmark_task(model, tokenizer, task, max_new_tokens, generation_options)
                if verbose:
                    t_info = res.get("timing") or {}
                    print(
                        f"[EvalTask | {name} #{consumed}] status={res.get('status')} | score={res.get('score')} | "
                        f"prompt={res.get('prompt_tokens')} tok | gen={res.get('generated_tokens')} tok | "
                        f"decode={float(t_info.get('decode_tok_per_s', 0.0) or 0.0):.1f} tok/s | "
                        f"wall={float(res.get('wall_time_sec', 0.0) or 0.0):.2f}s",
                        flush=True,
                    )
                reports.append(res)
            dataset_reports[name] = {"supplied_rows": known_size, "attempted_rows": consumed,
                                     "rows_exhausted": exhausted, "row_cap": max_hf_per_benchmark,
                                     "full_official_split_verified": False}
    if include_smoke:
        smoke_items = [dict(original) for original in SIGNIFICANT_BENCHMARK_SUITE]
        graph = generate_graphwalks_bfs_instance(num_nodes=graphwalks_nodes, seed=2026)
        memory = generate_mrcr_v2_instance(num_rounds=mrcr_rounds, seed=2026)
        for item in (
            {"id": "smoke_graphwalks", "benchmark": "Smoke / GraphWalks", "prompt": graph["prompt"],
             "gold_nodes": graph["gold_nodes"], "verifier": "graphwalks_f1"},
            {"id": "smoke_mrcr", "benchmark": "Smoke / MRCR", "prompt": memory["prompt"],
             "gold_answer": memory["gold_answer"], "verifier": "mrcr"},
        ):
            item.update(evaluation_kind="smoke", source="synthetic", status="ready", official_comparable=False)
            smoke_items.append(item)
        total_smoke = len(smoke_items)
        for s_idx, item in enumerate(smoke_items, start=1):
            if verbose:
                print(
                    f"[Smoke {s_idx}/{total_smoke}] {item.get('benchmark', item.get('id'))} ({item.get('id')}) starting...",
                    flush=True,
                )
                res = evaluate_benchmark_task(model, tokenizer, item, max_new_tokens, generation_options, verbose=True)
            else:
                res = evaluate_benchmark_task(model, tokenizer, item, max_new_tokens, generation_options)
            if verbose:
                t_info = res.get("timing") or {}
                print(
                    f"[Smoke {s_idx}/{total_smoke}] {item.get('benchmark', item.get('id'))} -> "
                    f"status={res.get('status')} | score={res.get('score')} | "
                    f"prompt={res.get('prompt_tokens')} tok | gen={res.get('generated_tokens')} tok | "
                    f"decode={float(t_info.get('decode_tok_per_s', 0.0) or 0.0):.1f} tok/s | "
                    f"TTFT={float(t_info.get('ttft_ms', 0.0) or 0.0):.1f}ms | "
                    f"wall={float(res.get('wall_time_sec', 0.0) or 0.0):.2f}s",
                    flush=True,
                )
            reports.append(res)
    counts: Dict[str, int] = {}
    buckets: Dict[Tuple[str, str, str], List[float]] = {}
    for result in reports:
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
        if status == "completed" and result["score"] is not None:
            key = (str(result.get("benchmark")), str(result.get("protocol")), str(result.get("metric")))
            buckets.setdefault(key, []).append(result["score"])
    summaries = [{"benchmark": key[0], "protocol": key[1], "metric": key[2], "scored_rows": len(scores),
                  "mean_score": statistics.fmean(scores), "official_comparable": False}
                 for key, scores in buckets.items()]
    completed = counts.get("completed", 0)
    all_supplied_complete = bool(reports) and completed == len(reports) and all(
        r["rows_exhausted"] and r["attempted_rows"] > 0 for r in dataset_reports.values()) and not setup_reason
    return {"status": "completed_supplied_rows" if all_supplied_complete else ("partial" if completed else "blocked"),
            "reason": setup_reason or ("no_tasks" if not reports else None), "tasks": reports,
            "total_tasks": len(reports), "hf_evaluated_count": sum(r.get("source") == "hf_real" and r["status"] == "completed" for r in reports),
            "status_counts": counts, "dataset_reports": dataset_reports,
            "download_report": getattr(hf_samples, "load_report", None), "by_benchmark": summaries,
            "all_supplied_rows_completed": all_supplied_complete, "full_suite_complete": False,
            "include_smoke": bool(include_smoke), "task_mode": task_mode,
            "mean_score": None, "pass_rate_pct": None,
            "aggregate_reason": "heterogeneous_protocol_scores_are_not_comparable"}


@_no_grad()
def benchmark_loop72_decode_speed(
    model: Any,
    tokenizer: Any,
    prompt_lengths: Sequence[int] = (512, 2048, 8192, 32768),
    gen_tokens: int = 512,
    warmups: int = 1,
    repeats: int = 3,
    seed: int = 2026,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Repeat synchronized prefill/decode trials with fresh request caches.

    EOS remains enabled: requested and actual output lengths are both recorded,
    and short runs cannot silently stand in for a 512-token sustained measurement.
    Warmup requests are excluded. These synthetic prompts measure performance only.
    """
    if int(gen_tokens) < 1 or int(repeats) < 1 or int(warmups) < 0:
        raise ValueError("gen_tokens/repeats must be positive and warmups nonnegative")
    lengths = [int(n) for n in prompt_lengths]
    if any(n < 1 for n in lengths):
        raise ValueError("prompt lengths must be positive")
    report: Dict[str, Any] = {
        "status": "not_run", "gen_tokens": int(gen_tokens), "warmups": int(warmups),
        "repeats": int(repeats), "seed": int(seed), "horizons": [],
        "evaluation_kind": "synthetic_performance", "eos_enabled": True,
        "total_virtual_layers": getattr(model, "total_virtual_layers", None),
        "generation_options": {"do_sample": False, "repetition_penalty": 1.0,
                               "force_think_close": False, "top_k": 0, "top_p": 1.0},
    }
    if not lengths:
        report.update(status="blocked", reason="no_prompt_lengths")
        return report
    if not HAS_TORCH or model is None or tokenizer is None:
        report.update(status="blocked", reason="model_tokenizer_and_pytorch_required")
        report["horizons"] = [{"prompt_tokens": n, "status": "blocked", "reason": report["reason"]} for n in lengths]
        return report
    gen_sig_params = inspect.signature(model.generate_cached).parameters
    if "force_think_close" not in gen_sig_params:
        report.update(status="unsupported", reason="runtime_cannot_disable_forced_thinking_close")
        return report
    speed_extra_kwargs: Dict[str, Any] = {}
    if "abort_on_degenerate_loop" in gen_sig_params:
        speed_extra_kwargs["abort_on_degenerate_loop"] = False
    device = next(model.base.parameters()).device
    report["device"] = str(device)
    context_limit = _configured_context_limit(model)
    report["configured_context_limit"] = context_limit
    valid_lengths = [n for n in lengths if context_limit is None or n + int(gen_tokens) <= context_limit]
    max_len = max(valid_lengths, default=0)
    ids = tokenizer.encode(build_modern_long_eval_stream(target_tokens=max_len + 256, tokenizer=tokenizer), add_special_tokens=False) if max_len else []
    if max_len and not ids:
        raise ValueError("tokenizer produced an empty performance stream")
    while len(ids) < max_len:
        ids += ids
    required_metrics = ("ttft_ms", "prefill_tok_per_s", "decode_tok_per_s", "e2e_tok_per_s",
                        "itl_p50_ms", "itl_p95_ms", "itl_p99_ms")
    for length in lengths:
        row: Dict[str, Any] = {"prompt_tokens": length, "requested_generated_tokens": int(gen_tokens),
                               "status": "not_run", "trials": [], "warmups_completed": 0}
        report["horizons"].append(row)
        if context_limit is not None and length + int(gen_tokens) > context_limit:
            row.update(status="blocked", reason="prompt_plus_output_budget_exceeds_configured_context")
            continue
        for trial_idx in range(int(warmups) + int(repeats)):
            cache = None
            try:
                torch.manual_seed(int(seed))
                inp = torch.tensor([ids[:length]], dtype=torch.long, device=device)
                cache = model.make_inference_cache(session_id=f"speed_{length}_{trial_idx}")
                model.last_generation_timing = None
                start = time.perf_counter()
                out = model.generate_cached(inp, max_new_tokens=int(gen_tokens), do_sample=False,
                                            repetition_penalty=1.0, force_think_close=False,
                                            top_p=1.0, top_k=0, tokenizer=tokenizer, stream_print=False, cache=cache,
                                            **speed_extra_kwargs)
                if getattr(device, "type", str(device).split(":")[0]) == "cuda":
                    torch.cuda.synchronize(device)
                wall = time.perf_counter() - start
                timing = dict(getattr(model, "last_generation_timing", None) or {})
                actual = int(out.shape[1]) - length
                if actual < 0 or actual > int(gen_tokens):
                    raise RuntimeError("runtime returned invalid output length")
                if any(key not in timing for key in required_metrics) or timing.get("generated_tokens") != actual:
                    raise RuntimeError("missing_or_inconsistent_generation_timing")
                ledger = cache.memory_ledger()
                trial = {"trial": trial_idx - int(warmups), "status": "completed",
                         "generated_tokens": actual, "decode_steps": max(0, actual - 1),
                         "wall_time_sec": wall, "finish_reason": timing.get("finish_reason", "unknown"),
                         "timing": timing, "memory_ledger": ledger}
                if trial_idx < int(warmups):
                    row["warmups_completed"] += 1
                    if verbose:
                        print(
                            f"  [SpeedBench | prompt={length} tok | warmup {trial_idx + 1}/{warmups}] "
                            f"TTFT={float(timing.get('ttft_ms', 0.0)):.1f}ms ({float(timing.get('prefill_tok_per_s', 0.0)):.1f} tok/s) | "
                            f"decode={float(timing.get('decode_tok_per_s', 0.0)):.2f} tok/s | wall={wall:.2f}s",
                            flush=True,
                        )
                else:
                    row["trials"].append(trial)
                    if verbose:
                        t_num = trial_idx - int(warmups) + 1
                        print(
                            f"  [SpeedBench | prompt={length} tok | trial {t_num}/{repeats}] "
                            f"TTFT={float(timing.get('ttft_ms', 0.0)):.1f}ms ({float(timing.get('prefill_tok_per_s', 0.0)):.1f} tok/s) | "
                            f"decode={float(timing.get('decode_tok_per_s', 0.0)):.2f} tok/s | "
                            f"ITL p50={float(timing.get('itl_p50_ms', 0.0)):.1f}ms | "
                            f"GPU={float(ledger.get('m1_gpu_total_mib', 0.0)):.1f} MiB | "
                            f"Cold={float(ledger.get('m2_host_cold_pages_mib', 0.0)):.1f} MiB | wall={wall:.2f}s",
                            flush=True,
                        )
            except Exception as exc:
                row.update(status="error", reason=f"performance_trial_error:{type(exc).__name__}:{exc}")
                if trial_idx >= int(warmups):
                    row["trials"].append({"trial": trial_idx - int(warmups), "status": "error", "reason": row["reason"]})
                break
            finally:
                del cache
        successful = [trial for trial in row["trials"] if trial["status"] == "completed"]
        if len(successful) == int(repeats) and row["warmups_completed"] == int(warmups):
            row["status"] = "completed"
        if successful:
            for key in required_metrics:
                row[key] = statistics.median(float(trial["timing"][key]) for trial in successful)
            row["generated_tokens"] = statistics.median(trial["generated_tokens"] for trial in successful)
            row["min_generated_tokens"] = min(trial["generated_tokens"] for trial in successful)
            row["max_generated_tokens"] = max(trial["generated_tokens"] for trial in successful)
            row["requested_output_length_reached_in_all_trials"] = all(trial["generated_tokens"] == int(gen_tokens) for trial in successful)
            row["sustained_512_token_decode_measured"] = all(trial["generated_tokens"] >= 512 for trial in successful) and row["status"] == "completed"
            row["gpu_cache_mib"] = max(float(trial["memory_ledger"].get("m1_gpu_total_mib", 0.0)) for trial in successful)
            row["host_cold_mib"] = max(float(trial["memory_ledger"].get("m2_host_cold_pages_mib", 0.0)) for trial in successful)
    complete = all(row["status"] == "completed" for row in report["horizons"])
    report["status"] = "completed" if complete else ("partial" if any(row["trials"] for row in report["horizons"]) else "blocked")
    report["summary_statistic"] = "median_over_successful_measured_trials"
    report["by_horizon"] = report["horizons"]
    return report


@_no_grad()
def run_full_length_1m_context_benchmark(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    horizons: Sequence[int] = (32_768, 131_072, 262_144, 524_288, 1_048_576),
    max_live_tokens: int = 32_768,
    gen_tokens: int = 16,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Report capacity arithmetic and optional synthetic live load probes.

    Kept under its historical name for compatibility. Arithmetic and a repeated
    token-stream forward pass do not establish usable context quality. Actual
    GraphWalks/MRCR quality belongs to the pinned full-dataset runner.
    """
    horizons = [int(h) for h in horizons]
    if any(h < 1 for h in horizons):
        raise ValueError("horizons must be positive")
    capacity = {h: compute_loop72_capacity_table(n_tokens=h) for h in horizons}
    cap_1m = capacity.get(1_048_576) or compute_loop72_capacity_table(n_tokens=1_048_576)
    horizon_status = []
    live_runs = []
    for horizon in horizons:
        if horizon > int(max_live_tokens):
            if verbose:
                print(f"  [ContextProbe | {horizon:,} tok] skipped (above max_live_tokens={max_live_tokens:,})", flush=True)
            horizon_status.append({"context_tokens": horizon, "status": "not_run", "reason": "above_explicit_live_probe_cap"})
            continue
        if not HAS_TORCH or model is None or tokenizer is None:
            horizon_status.append({"context_tokens": horizon, "status": "blocked", "reason": "model_tokenizer_and_pytorch_required"})
            continue
        if verbose:
            print(f"  [ContextProbe | {horizon:,} tok] running live prefill + {gen_tokens}-token decode probe...", flush=True)
        measured = benchmark_loop72_decode_speed(model, tokenizer, (horizon,), gen_tokens=gen_tokens, warmups=0, repeats=1, verbose=verbose)
        row = dict(measured.get("horizons", [{}])[0]) if measured.get("horizons") else {"status": measured["status"], "reason": measured.get("reason")}
        row.update(context_tokens=horizon, evaluation_kind="synthetic_load_probe", quality_validated=False)
        horizon_status.append(row)
        live_runs.append(row)
    observed = [row["context_tokens"] for row in live_runs if row["status"] == "completed"]
    all_executed = bool(horizons) and len(observed) == len(horizons)
    return {"status": "load_probes_completed" if all_executed else ("partial" if observed else "not_run"),
            "context_limit_tokens": _configured_context_limit(model) if model is not None else None,
            "target_context_tokens": 1_048_576, "validated_context_tokens": None,
            "observed_max_live_tokens": max(observed) if observed else None,
            "quality_validated": False, "full_quality_suite_complete": False,
            "all_requested_load_probes_completed": all_executed,
            "horizons": horizons, "max_live_tokens": int(max_live_tokens),
            "capacity_1m_summary": {
                "unbounded_all72_bf16_gib": cap_1m["unbounded_all72_bf16_gib"],
                "local_54_rings_mib": cap_1m["local_rings_mib"],
                "global_18_bf16_gib": cap_1m["global_payload_gib"]["16bit_gib"],
                "global_18_mixed_rotary_bf16_gib": cap_1m["mixed_rotary_bf16_gib"],
                "global_18_int4_gib": cap_1m["global_payload_gib"]["4bit_gib"],
            }, "capacity_by_horizon": capacity,
            "capacity_is_payload_arithmetic_not_observed_peak_vram": True,
            "horizon_status": horizon_status, "live_runs": live_runs}


def generate_omp_models_yaml(
    output_path: Optional[str] = None,
    port: int = 8000,
    model_id: str = "Spark-X2.6-4B-Recursive72",
    context_window: int = 131_072,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Return the pinned OMP configuration; explicit writes safely merge providers.

    Client budgets are declarations, not validated context capacity. Use the
    endpoint health limits for an actual run. See docs/OMP_EVALUATION.md.
    """
    from sparkx_omp_server import generate_omp_models_yaml as generate
    return generate(output_path, port, model_id, context_window, max_tokens, **kwargs)


def start_loop72_openai_compatible_server(
    model: Any,
    tokenizer: Any,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_id: str = "Spark-X2.6-4B-Recursive72",
    background: bool = True,
    **kwargs: Any,
) -> Any:
    """Start the serialized, token-streaming native Spark tool-call bridge."""
    from sparkx_omp_server import create_loop72_server
    return create_loop72_server(model, tokenizer, host, port, model_id, background, **kwargs)


def evaluate_cold_kv_codecs_summary() -> Dict[str, Dict[str, float]]:
    """Run a deterministic quality & compression audit across all 4 cold KV page codecs (`bf16`, `int8`, `int4`, `mixed_rotary_bf16`)."""
    # 4 KV heads, 64 tokens (1 page), 256 head_dim (64 rotated + 192 RoPE-free)
    k_page = [
        [
            [math.sin((h + 1) * 0.3 + t * 0.07 + d * 0.05) for d in range(64)]
            + [math.cos((h + 1) * 0.2 + t * 0.05 + d * 0.03) * 0.6 for d in range(192)]
            for t in range(64)
        ]
        for h in range(4)
    ]
    v_page = [
        [
            [math.sin((h + 2) * 0.17 - t * 0.04 + d * 0.02) * 0.5 for d in range(256)]
            for t in range(64)
        ]
        for h in range(4)
    ]
    out: Dict[str, Dict[str, float]] = {}
    for c_name in ("bf16", "int8", "int4", "mixed_rotary_bf16"):
        cfg = KVCodecConfig(codec=c_name, group_size=64, rope_dim=64, rope_first=True)
        out[c_name] = measure_codec_error(k_page, v_page, cfg)
    return out


def format_official_comparison_markdown(live_task_results: Optional[Dict[str, Any]] = None) -> str:
    """Keep the published reference separate from noncomparable local results.

    Only independently verified, complete, same-protocol benchmark summaries may
    populate the comparison column. Per-task smoke/diagnostic/sample results never
    do, and no new cross-benchmark average is invented from missing or mixed scores.
    """
    comparable: Dict[str, float] = {}
    for row in (live_task_results or {}).get("verified_official_summaries", []):
        if (row.get("benchmark") != "Average" and row.get("official_comparable") is True
                and row.get("full_split_complete") is True and row.get("protocol_verified") is True
                and row.get("sampling_verified") is True and row.get("dataset_revision")
                and row.get("evaluator_revision") and row.get("score") is not None):
            comparable[str(row["benchmark"])] = float(row["score"])
    lines = [
        "Published reference values are not measurements of this variant. Local smoke, sampled, or modified-protocol scores are excluded from this comparison.",
        "", "| Category | Benchmark | Official Spark X2.6-4B | Spark X2.5-4B | Qwen3.5 9B | Qwen3.5 4B | Gemma4 12B | Verified comparable Loop72 + D-LSA |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    def fmt(value: Any, bold: bool = False) -> str:
        if value is None:
            return "-"
        text = f"{float(value):.1f}"
        return f"**{text}**" if bold else text

    for name, row in OFFICIAL_BENCHMARK_TABLE.items():
        value = fmt(comparable[name], True) if name in comparable else "—"
        lines.append(f"| {row['category']} | **{name}** | {fmt(row.get('Spark_X2.6_4B'), True)} | "
                     f"{fmt(row.get('Spark_X2.5_4B'))} | {fmt(row.get('Qwen3.5_9B'))} | "
                     f"{fmt(row.get('Qwen3.5_4B'))} | {fmt(row.get('Gemma4_12B'))} | {value} |")
    return "\n".join(lines)
