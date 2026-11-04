"""sparkx_si.verifiers — programmatic verifiers for SI-SRSI v2 (docs/roadmap/05 §3A).

Principle: a sample is *good* only if a deterministic verifier says so. No judge model, no regex rubric score can admit a sample.
Pure standard library (SymPy optional for symbolic equivalence). Every verifier returns a `Verdict`.

Think-block policy: the answer is read only from the text **after** the closing `</think>` tag. An *unclosed* think block, or an answer
that appears only inside the think block, fails (`reason='no_answer_after_think'`). Responses without any think tags are treated
as answer-only.

Code sandbox: defense in depth for *model-generated* code (subprocess, scrubbed environment, temp cwd, rlimits, process-group kill,
network patched off and `unshare -n` when available). It is **not** a security boundary — run only in an ephemeral VM without
credentials, as stated in the design doc.
"""
from __future__ import annotations

import json
import os
import re
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


# ======================================================================================
# Think-block handling
# ======================================================================================
_THINK_OPEN = re.compile(r"<think>", re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)


def answer_part(response: str) -> Optional[str]:
    """Text after the last `</think>`; None if a think block was opened and never closed; whole text if no think tags at all."""
    if not _THINK_OPEN.search(response) and not _THINK_CLOSE.search(response):
        return response
    closes = list(_THINK_CLOSE.finditer(response))
    if not closes:
        return None
    last_close = closes[-1].end()
    opens_after = [m for m in _THINK_OPEN.finditer(response) if m.start() >= last_close]
    if opens_after:  # a new unclosed think block after the last close
        return None
    return response[last_close:]


# ======================================================================================
# Math
# ======================================================================================
def _balanced_braces(s: str, start: int) -> Optional[Tuple[str, int]]:
    """`s[start] == '{'` -> (content, index after the closing brace)."""
    if start >= len(s) or s[start] != "{":
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1: i], i + 1
    return None


def extract_boxed(text: str) -> Optional[str]:
    """Content of the LAST `\\boxed{...}` (balanced braces)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    j = idx + len("\\boxed")
    while j < len(text) and text[j] == " ":
        j += 1
    res = _balanced_braces(text, j)
    return res[0] if res else None


_NUM = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"


def extract_final_answer(response: str) -> Optional[str]:
    ans = answer_part(response)
    if ans is None:
        return None
    b = extract_boxed(ans)
    if b is not None:
        return b.strip()
    m = re.findall(r"####\s*(" + _NUM + ")", ans)
    if m:
        return m[-1]
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:|=)\s*\**\s*\$?\s*(" + _NUM + r"|[^\n.]+)", ans, flags=re.I)
    if m:
        return m[-1].strip(" *$")
    nums = re.findall(_NUM, ans)
    return nums[-1] if nums else None


def _strip_latex(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(?:left|right|,|!|;|:|quad|qquad)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\%", "%").replace("^\\circ", "").replace("°", "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^{}]*)\}", r"\1", s)
    return s.strip()


def _to_fraction(s: str) -> Optional[Fraction]:
    s = _strip_latex(s).replace(" ", "")
    s = s.rstrip(".")
    neg = False
    if s.startswith("-"):
        neg, s = True, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    s = s.replace(",", "")
    m = re.fullmatch(r"\\frac\{(\d+)\}\{(\d+)\}", s)
    try:
        if m:
            val = Fraction(int(m.group(1)), int(m.group(2)))
        elif re.fullmatch(r"\d+/\d+", s):
            a, b = s.split("/")
            val = Fraction(int(a), int(b))
        elif re.fullmatch(r"\d+(?:\.\d+)?", s):
            val = Fraction(s)
        else:
            return None
    except (ZeroDivisionError, ValueError):
        return None
    if pct:
        val = val / 100
    return -val if neg else val


def _norm_text(s: str) -> str:
    s = _strip_latex(s).lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[.,;]+$", "", s)
    return s


def math_equal(pred: str, gold: str, use_sympy: bool = False, sympy_timeout: float = 2.0) -> bool:
    if pred is None or gold is None:
        return False
    fp, fg = _to_fraction(pred), _to_fraction(gold)
    if fp is not None and fg is not None:
        return fp == fg or (fg != 0 and abs(float(fp - fg)) / abs(float(fg)) < 1e-9)
    if _norm_text(pred) == _norm_text(gold):
        return True
    if use_sympy:
        return _sympy_equal(pred, gold, sympy_timeout)
    return False


def _sympy_equal(a: str, b: str, timeout: float) -> bool:
    try:
        import sympy  # type: ignore
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    except Exception:
        return False

    class _TO(Exception):
        pass

    def _alarm(*_):
        raise _TO()

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        tr = standard_transformations + (implicit_multiplication_application,)
        ea, eb = parse_expr(_strip_latex(a).replace("^", "**"), transformations=tr), parse_expr(_strip_latex(b).replace("^", "**"), transformations=tr)
        return bool(sympy.simplify(ea - eb) == 0)
    except BaseException:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def verify_math(response: str, gold: str, use_sympy: bool = False) -> Verdict:
    pred = extract_final_answer(response)
    if pred is None:
        return Verdict(False, "no_answer_after_think")
    ok = math_equal(pred, gold, use_sympy=use_sympy)
    return Verdict(ok, "match" if ok else "mismatch", {"pred": pred, "gold": gold})


# ======================================================================================
# Closed-book QA (alias exact match)
# ======================================================================================
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_qa(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def verify_qa(response: str, aliases: Sequence[str], max_answer_words: int = 12) -> Verdict:
    ans = answer_part(response)
    if ans is None:
        return Verdict(False, "no_answer_after_think")
    ans = ans.strip()
    if not ans:
        return Verdict(False, "empty")
    al = {normalize_qa(a) for a in aliases if a}
    cand: List[str] = [ans]
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", ans, flags=re.I)
    if m:
        cand.append(m[-1])
    lines = [l for l in ans.splitlines() if l.strip()]
    if lines:
        cand.append(lines[-1])
    for c in cand:
        n = normalize_qa(c)
        if n in al:
            return Verdict(True, "exact", {"matched": n})
    # whole-word containment only for SHORT answers (a hedge-dump listing many candidates must not pass)
    n_all = normalize_qa(ans)
    if len(n_all.split()) <= max_answer_words:
        padded = f" {n_all} "
        hits = [a for a in al if a and f" {a} " in padded]
        if hits:
            return Verdict(True, "contains_short", {"matched": hits[0]})
    return Verdict(False, "mismatch", {"answer": ans[:120]})


# ======================================================================================
# Instruction-following constraints (IFEval-style, deterministic)
# ======================================================================================
def _words(t: str) -> List[str]:
    return re.findall(r"\b[\w'-]+\b", t)


def _sentences(t: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


def _json_loads_loose(t: str) -> Any:
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return json.loads(t)


def _bullets(t: str) -> List[str]:
    return [l for l in t.splitlines() if re.match(r"^\s*[-*]\s+\S", l)]


def _paragraphs(t: str) -> List[str]:
    return [p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]


CONSTRAINTS: Dict[str, Callable[[str, Dict[str, Any]], bool]] = {
    "min_words": lambda t, a: len(_words(t)) >= a["n"],
    "max_words": lambda t, a: len(_words(t)) <= a["n"],
    "max_sentences": lambda t, a: len(_sentences(t)) <= a["n"],
    "min_sentences": lambda t, a: len(_sentences(t)) >= a["n"],
    "exact_bullets": lambda t, a: len(_bullets(t)) == a["n"],
    "exact_paragraphs": lambda t, a: len(_paragraphs(t)) == a["n"],
    "forbidden_words": lambda t, a: not any(re.search(r"\b" + re.escape(w) + r"\b", t, re.I) for w in a["words"]),
    "required_words": lambda t, a: all(re.search(r"\b" + re.escape(w) + r"\b", t, re.I) for w in a["words"]),
    "keyword_min_count": lambda t, a: len(re.findall(r"\b" + re.escape(a["word"]) + r"\b", t, re.I)) >= a["n"],
    "all_lowercase": lambda t, a: t == t.lower() and any(c.isalpha() for c in t),
    "all_uppercase": lambda t, a: t == t.upper() and any(c.isalpha() for c in t),
    "no_commas": lambda t, a: "," not in t,
    "ends_with": lambda t, a: t.rstrip().endswith(a["phrase"]),
    "starts_with": lambda t, a: t.lstrip().startswith(a["phrase"]),
    "title_in_brackets": lambda t, a: bool(re.search(r"<<[^\n<>]+>>", t)),
    "wrapped_in_quotes": lambda t, a: len(t.strip()) >= 2 and t.strip()[0] == '"' and t.strip()[-1] == '"',
    "json_format": lambda t, a: _is_json(t),
}


def _is_json(t: str) -> bool:
    try:
        _json_loads_loose(t)
        return True
    except Exception:
        return False


def check_constraint(c: Dict[str, Any], text: str) -> bool:
    fn = CONSTRAINTS.get(c["id"])
    if fn is None:
        raise KeyError(f"unknown constraint {c['id']}")
    try:
        return bool(fn(text, c.get("args", {})))
    except Exception:
        return False


def _loose_variants(text: str) -> List[str]:
    lines = text.split("\n")
    v = [text, text.replace("*", "")]
    if len(lines) > 1:
        v += ["\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    return [x for x in v if x is not None]


def verify_if(response: str, constraints: Sequence[Dict[str, Any]], strict: bool = True) -> Verdict:
    ans = answer_part(response)
    if ans is None:
        return Verdict(False, "no_answer_after_think")
    ans = ans.strip()
    results = []
    for c in constraints:
        if strict:
            ok = check_constraint(c, ans)
        else:
            ok = any(check_constraint(c, v) for v in _loose_variants(ans))
        results.append(ok)
    return Verdict(all(results) and bool(results), "all_pass" if all(results) else "violated",
                   {"per_constraint": results, "instruction_level": sum(results) / max(1, len(results))})


# ======================================================================================
# Tool calls: JSON parse -> schema validation (subset) -> mock execution
# ======================================================================================
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict, "null": type(None)}


def validate_schema(x: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Minimal JSON-Schema subset: type, enum, minimum/maximum, minLength/maxLength, items, properties, required,
    additionalProperties (False). Returns a list of error strings (empty = valid)."""
    errs: List[str] = []
    t = schema.get("type")
    if t:
        py = _TYPES.get(t)
        bad = py is None or not isinstance(x, py) or (t in ("integer", "number") and isinstance(x, bool))
        if bad:
            return [f"{path}: expected {t}"]
    if "enum" in schema and x not in schema["enum"]:
        errs.append(f"{path}: not in enum")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if "minimum" in schema and x < schema["minimum"]:
            errs.append(f"{path}: < minimum")
        if "maximum" in schema and x > schema["maximum"]:
            errs.append(f"{path}: > maximum")
    if isinstance(x, str):
        if "minLength" in schema and len(x) < schema["minLength"]:
            errs.append(f"{path}: too short")
        if "maxLength" in schema and len(x) > schema["maxLength"]:
            errs.append(f"{path}: too long")
    if isinstance(x, list) and "items" in schema:
        for i, v in enumerate(x):
            errs += validate_schema(v, schema["items"], f"{path}[{i}]")
    if isinstance(x, dict):
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in x:
                errs.append(f"{path}.{r}: required")
        for k, v in x.items():
            if k in props:
                errs += validate_schema(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}.{k}: additional property")
    return errs


def _extract_json_object(text: str) -> Optional[Any]:
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S | re.I)
    cands = [m.group(1)] if m else []
    cands += re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                cands.append(text[start: i + 1])
                break
        start = text.find("{", start + 1)
        if len(cands) > 8:
            break
    for c in cands:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def verify_tool_call(response: str, tools: Dict[str, Dict[str, Any]], expected: Dict[str, Any],
                     implementations: Optional[Dict[str, Callable[..., Any]]] = None) -> Verdict:
    """`tools`: name -> JSON schema of the arguments. Response must contain {"name": ..., "arguments": {...}}.
    `expected`: {"name": str, "result": Any} — the mock implementation is executed and the result compared."""
    ans = answer_part(response)
    if ans is None:
        return Verdict(False, "no_answer_after_think")
    obj = _extract_json_object(ans)
    if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
        return Verdict(False, "no_tool_call_json")
    name, args = obj["name"], obj["arguments"]
    if name not in tools:
        return Verdict(False, "unknown_tool", {"name": name})
    if not isinstance(args, dict):
        return Verdict(False, "arguments_not_object")
    errs = validate_schema(args, tools[name])
    if errs:
        return Verdict(False, "schema_violation", {"errors": errs[:5]})
    if name != expected.get("name"):
        return Verdict(False, "wrong_tool", {"name": name})
    impl = (implementations or {}).get(name)
    if impl is None:
        return Verdict(True, "schema_ok_no_impl")
    try:
        res = impl(**args)
    except Exception as e:
        return Verdict(False, "tool_raised", {"error": str(e)[:100]})
    ok = res == expected.get("result")
    return Verdict(ok, "match" if ok else "wrong_result", {"result": res})


# ======================================================================================
# Code (sandboxed execution)
# ======================================================================================
def extract_code(response: str) -> Optional[str]:
    ans = answer_part(response)
    if ans is None:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", ans, re.S)
    return blocks[-1] if blocks else None


_HARNESS = r'''
import sys, io, os, traceback
_nonce = sys.stdin.readline().strip()
_code = open("solution.py").read()
_tests = open("tests.py").read()
# --- defense in depth: disable obvious network / process escapes inside the harness process
import socket as _s
def _deny(*a, **k): raise OSError("network disabled by sandbox")
_s.socket.connect = _deny; _s.socket.connect_ex = _deny; _s.create_connection = _deny; _s.getaddrinfo = _deny
import subprocess as _sp
for _n in ("Popen", "run", "call", "check_call", "check_output"):
    setattr(_sp, _n, _deny)
os.system = _deny; os.fork = _deny; os.execv = _deny; os.execvp = _deny; os.popen = _deny
def _main():
    ns = {"__name__": "__sandbox__"}
    try:
        exec(compile(_code, "solution.py", "exec"), ns)
        exec(compile(_tests, "tests.py", "exec"), ns)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stdout.write("\n__SI_OK__" + _nonce + "\n")
    sys.stdout.flush()
    os._exit(0)
_main()
'''


def _limits(cpu_s: int, mem_mb: int):
    def _set():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _set


def _unshare_ok() -> bool:
    exe = shutil.which("unshare")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


_UNSHARE: Optional[bool] = None


def run_python_tests(code: str, tests: str, timeout: float = 6.0, mem_mb: int = 1024) -> Verdict:
    """Run `code` then `tests` (assert statements / a test script) in a scrubbed subprocess. Success requires the harness sentinel
    (with a per-run nonce the solution cannot know in advance), so `sys.exit(0)`, `os._exit(0)` or printing expected output fail."""
    global _UNSHARE
    if _UNSHARE is None:
        _UNSHARE = _unshare_ok()
    nonce = secrets.token_hex(8)
    with tempfile.TemporaryDirectory(prefix="si_sbx_") as td:
        with open(os.path.join(td, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(td, "tests.py"), "w") as f:
            f.write(tests)
        with open(os.path.join(td, "harness.py"), "w") as f:
            f.write(_HARNESS)
        cmd = [sys.executable, "-I", "harness.py"]
        if _UNSHARE:
            cmd = [shutil.which("unshare"), "-rn"] + cmd
        env = {"PATH": "/usr/bin:/bin", "HOME": td, "TMPDIR": td, "PYTHONHASHSEED": "0"}
        try:
            p = subprocess.Popen(cmd, cwd=td, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 preexec_fn=_limits(int(timeout) + 1, mem_mb), text=True)
        except Exception as e:
            return Verdict(False, "sandbox_spawn_failed", {"error": str(e)[:120]})
        try:
            out, err = p.communicate(nonce + "\n", timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                p.kill()
            p.communicate()
            return Verdict(False, "timeout")
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    if ("__SI_OK__" + nonce) in out:
        return Verdict(True, "tests_passed")
    return Verdict(False, "tests_failed", {"stderr_tail": err[-300:], "rc": p.returncode})


def verify_code(response: str, tests: str, timeout: float = 6.0) -> Verdict:
    code = extract_code(response)
    if code is None:
        return Verdict(False, "no_code_block")
    return run_python_tests(code, tests, timeout=timeout)


# ======================================================================================
# Dispatch
# ======================================================================================
def verify(task: Dict[str, Any], response: str) -> Verdict:
    """`task['pillar']` in {math, code, if, tool, qa}; `task['spec']` carries gold/tests/constraints."""
    p, spec = task["pillar"], task["spec"]
    if p == "math":
        return verify_math(response, spec["gold"], use_sympy=bool(spec.get("sympy", False)))
    if p == "qa":
        return verify_qa(response, spec["aliases"])
    if p == "if":
        return verify_if(response, spec["constraints"], strict=bool(spec.get("strict", True)))
    if p == "code":
        return verify_code(response, spec["tests"], timeout=float(spec.get("timeout", 6.0)))
    if p == "tool":
        return verify_tool_call(response, spec["tools"], spec["expected"], spec.get("implementations"))
    raise KeyError(f"unknown pillar {p}")
