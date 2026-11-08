#!/usr/bin/env python3
"""Run actual Oh My Pi RPC agents against a Loop72 endpoint and verify edits.

The bundled tasks are LOCAL coding integration cases, not SWE-bench, BFCL, or an
official OMP benchmark. No model score is produced for a scripted backend.
Use a disposable container for model-generated code: a private working directory
and clean agent config are isolation conveniences, not an OS security sandbox.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sparkx_omp_server import DEFAULT_MODEL_ID, OMP_COMMIT, OMP_VERSION, generate_omp_models_yaml


@dataclasses.dataclass(frozen=True)
class CodingTask:
    id: str
    filename: str
    source: str
    prompt: str
    public_tests: str
    private_checks: str


TASKS = [
    CodingTask(
        "clamp_bounds", "math_utils.py",
        'def clamp(value, lower, upper):\n    if lower > upper:\n        raise ValueError("lower > upper")\n    return min(lower, max(upper, value))\n',
        "Fix clamp in math_utils.py. It must return lower below the interval, upper above the interval, and the input inside the closed interval; lower > upper must still raise ValueError. Read the file, apply a minimal edit, run the existing tests with bash, and summarize the result. Do not change tests.",
        'import unittest\nfrom math_utils import clamp\nclass TestClamp(unittest.TestCase):\n    def test_inside(self): self.assertEqual(clamp(5, 0, 10), 5)\n    def test_above(self): self.assertEqual(clamp(12, 0, 10), 10)\nif __name__ == "__main__": unittest.main()\n',
        'from math_utils import clamp\nimport random\nr=random.Random(7218)\nfor _ in range(150):\n    a,b=sorted([r.randint(-100,100),r.randint(-100,100)])\n    x=r.uniform(-150,150)\n    expected=a if x<a else b if x>b else x\n    assert clamp(x,a,b)==expected,(x,a,b)\nassert clamp(3,3,3)==3\ntry: clamp(0,2,1)\nexcept ValueError: pass\nelse: raise AssertionError("invalid interval accepted")\n',
    ),
    CodingTask(
        "chunk_remainder", "chunks.py",
        'def chunked(items, size):\n    if size <= 0:\n        raise ValueError("size must be positive")\n    return [items[i:i + size] for i in range(0, len(items) - size, size)]\n',
        "Fix chunked in chunks.py. Return consecutive chunks of at most size elements, including a final partial chunk; preserve every input element in order. Empty inputs return an empty list. Nonpositive sizes still raise ValueError. Read, edit, and run the tests. Do not edit the tests.",
        'import unittest\nfrom chunks import chunked\nclass TestChunks(unittest.TestCase):\n    def test_remainder(self): self.assertEqual(chunked([1,2,3,4,5],2), [[1,2],[3,4],[5]])\n    def test_empty(self): self.assertEqual(chunked([],3), [])\nif __name__ == "__main__": unittest.main()\n',
        'from chunks import chunked\nfor n in range(25):\n    xs=list(range(n))\n    for size in range(1,10):\n        actual=chunked(xs,size)\n        assert actual==[xs[i:i+size] for i in range(0,n,size)]\n        assert [v for chunk in actual for v in chunk]==xs\nfor size in [0,-1]:\n    try: chunked([1],size)\n    except ValueError: pass\n    else: raise AssertionError("invalid size accepted")\n',
    ),
    CodingTask(
        "merge_unsorted_intervals", "intervals.py",
        'def merge_intervals(intervals):\n    result = []\n    for start, end in intervals:\n        if result and start < result[-1][1]:\n            result[-1][1] = max(result[-1][1], end)\n        else:\n            result.append([start, end])\n    return result\n',
        "Fix merge_intervals in intervals.py for a list of closed integer intervals with start <= end, in arbitrary input order. Return sorted merged intervals; overlapping or touching intervals must merge. Do not mutate the input. Empty input returns []. Read, edit, run tests, and explain the fix. Do not edit tests.",
        'import unittest\nfrom intervals import merge_intervals\nclass TestIntervals(unittest.TestCase):\n    def test_unsorted_touching(self): self.assertEqual(merge_intervals([[5,7],[1,3],[3,5]]), [[1,7]])\n    def test_separate(self): self.assertEqual(merge_intervals([[1,2],[4,6]]), [[1,2],[4,6]])\nif __name__ == "__main__": unittest.main()\n',
        'from intervals import merge_intervals\nimport random,copy\nr=random.Random(724)\nfor _ in range(100):\n    data=[sorted([r.randint(-10,10),r.randint(-10,10)]) for _ in range(r.randrange(12))]\n    before=copy.deepcopy(data)\n    expected=[]\n    for start,end in sorted(data):\n        if expected and start<=expected[-1][1]: expected[-1][1]=max(expected[-1][1],end)\n        else: expected.append([start,end])\n    assert merge_intervals(data)==expected\n    assert data==before,"input mutated"\n',
    ),
]


def endpoint_health(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/health"
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def probe(omp_path: str, base_url: str) -> dict:
    binary = shutil.which(omp_path) if not Path(omp_path).is_file() else str(Path(omp_path).resolve())
    result = {"status": "blocked", "expected_omp_version": OMP_VERSION, "omp_commit": OMP_COMMIT,
              "omp_binary": binary, "model_quality_evaluated": False, "failures": []}
    if binary is None:
        result["failures"].append("Oh My Pi binary not found")
    else:
        try:
            digest = hashlib.sha256()
            with open(binary, "rb") as executable:
                for block in iter(lambda: executable.read(1024 * 1024), b""):
                    digest.update(block)
            result["omp_binary_sha256"] = digest.hexdigest()
            version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20, check=True).stdout.strip()
            result["omp_version"] = version
            if version not in {"omp/" + OMP_VERSION, OMP_VERSION}:
                result["failures"].append("Unpinned OMP version: " + version)
        except (OSError, subprocess.SubprocessError) as exc:
            result["failures"].append(str(exc))
    try:
        result["endpoint"] = endpoint_health(base_url)
    except Exception as exc:
        result["failures"].append("Endpoint unavailable: " + str(exc))
    if not result["failures"]:
        result["status"] = "ready"
    return result


class RPCDecoder:
    """Validate and reassemble pinned OMP v2 JSONL frames, including chunks."""
    def __init__(self):
        self.pending = None
        self.parts: list[bytes] = []

    def feed(self, line: str):
        frame = json.loads(line)
        if not isinstance(frame, dict):
            raise ValueError("RPC frame is not an object")
        if frame.get("type") != "rpc_chunk":
            if self.pending:
                raise ValueError("RPC chunk sequence was interrupted")
            return frame
        identity = (frame.get("chunkId"), frame.get("count"), frame.get("byteLength"))
        cid, count, size = identity
        if not isinstance(cid, str) or not isinstance(count, int) or not 0 < count <= 65536 or not isinstance(size, int) or not 0 < size <= 64 * 1024 * 1024:
            raise ValueError("Invalid RPC chunk metadata")
        if self.pending is None:
            self.pending, self.parts = identity, []
        if identity != self.pending or frame.get("index") != len(self.parts):
            raise ValueError("RPC chunk identity/order mismatch")
        self.parts.append(base64.b64decode(frame["data"], validate=True))
        if sum(map(len, self.parts)) > size:
            raise ValueError("RPC chunk exceeded declared size")
        if len(self.parts) != count:
            return None
        payload = b"".join(self.parts)
        self.pending, self.parts = None, []
        if len(payload) != size:
            raise ValueError("RPC reassembly byte length mismatch")
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Reassembled RPC frame is not an object")
        return value


def _clean_env(agent_dir: Path) -> dict:
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "SYSTEMROOT", "WINDIR", "LD_LIBRARY_PATH"}
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(PI_CODING_AGENT_DIR=str(agent_dir), NO_COLOR="1", TERM="dumb", PI_NO_TITLE="1", PI_NO_PTY="1", LANG="C.UTF-8", LC_ALL="C.UTF-8")
    return env


def _verify(task: CodingTask, workspace: Path, report_dir: Path, suffix: str) -> dict:
    # Verification files live outside the editable workspace. No shell command
    # is generated from model output; Python loads only the task's candidate file.
    verifier = report_dir / ("verify_" + suffix + ".py")
    verifier.write_text("import sys\nsys.path.insert(0, " + repr(str(workspace)) + ")\n" + task.private_checks)
    try:
        out = subprocess.run([sys.executable, "-I", "-B", str(verifier)], cwd=workspace, env=_clean_env(report_dir / "unused-agent"),
                             capture_output=True, text=True, timeout=15)
        return {"passed": out.returncode == 0, "returncode": out.returncode, "stdout": out.stdout[-16000:], "stderr": out.stderr[-16000:]}
    except subprocess.TimeoutExpired as exc:
        return {"passed": False, "error": "verification_timeout", "stdout": str(exc.stdout or "")[-16000:]}


def run_task(task: CodingTask, binary: str, base_url: str, model_id: str, endpoint: dict, output_dir: Path,
             max_seconds: float = 180.0, max_turns: int = 12, max_tokens: int = 2048, thinking: str = "off") -> dict:
    case_dir = output_dir / task.id
    case_dir.mkdir(parents=True, exist_ok=False)
    workspace, agent_dir = case_dir / "workspace", case_dir / "agent"
    workspace.mkdir()
    agent_dir.mkdir()
    (workspace / task.filename).write_text(task.source)
    (workspace / "test_public.py").write_text(task.public_tests)
    baseline = _verify(task, workspace, case_dir, "baseline")
    if baseline["passed"]:
        raise RuntimeError("Invalid fixture: unmodified code already passes " + task.id)
    provider_id = "spark_loop72_local"
    selection = provider_id + "/" + model_id
    limit = int(endpoint["context_limit_tokens"])
    output_cap = min(max_tokens, int(endpoint["max_output_tokens"]))
    generate_omp_models_yaml(str(agent_dir / "models.yml"), base_url=base_url, model_id=model_id,
                             context_window=limit, max_tokens=output_cap, provider_id=provider_id)
    config = {"enabledModels": [selection], "enabledProviders": [provider_id],
              "modelRoles": {role: selection for role in ("default", "smol", "slow", "plan", "task", "tiny", "memory", "advisor")},
              "edit": {"mode": "hashline", "autoRepair": {"enabled": False}}, "compaction": {"enabled": False}}
    (agent_dir / "config.yml").write_text(json.dumps(config, indent=2) + "\n")
    cmd = [binary, "--mode", "rpc", "--no-ui", "--no-session", "--no-title", "--no-extensions", "--no-skills", "--no-rules",
           "--no-context-files", "--no-lsp", "--no-pty", "--no-prewalk", "--tools", "read,edit,bash", "--auto-approve",
           "--model", selection, "--thinking", thinking, "--max-time", str(int(max_seconds)) + "s"]
    (case_dir / "invocation.json").write_text(json.dumps({"argv": cmd, "cwd": str(workspace), "thinking": thinking,
             "max_output_tokens_per_turn": output_cap, "max_turns": max_turns, "endpoint": endpoint}, indent=2) + "\n")
    events: queue.Queue = queue.Queue()
    decoder = RPCDecoder()
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=workspace, env=_clean_env(agent_dir), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=(os.name != "nt"))
    stderr_path = case_dir / "stderr.log"
    def read_stdout():
        try:
            for line in proc.stdout:
                events.put(("line", line))
        finally:
            events.put(("eof", None))
    def read_stderr():
        with stderr_path.open("w") as out:
            for line in proc.stderr:
                out.write(line)
                out.flush()
    readers = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for reader in readers:
        reader.start()
    def send(frame: dict):
        proc.stdin.write(json.dumps(frame) + "\n")
        proc.stdin.flush()
    tools_started, tools_finished, model_messages = [], [], []
    turn_count, prompt_result, settled, error = 0, None, False, None
    state = "startup"
    with (case_dir / "events.jsonl").open("w") as log:
        try:
            while time.monotonic() - started < max_seconds + 30:
                try:
                    kind, value = events.get(timeout=0.2)
                except queue.Empty:
                    if proc.poll() is not None:
                        raise RuntimeError("OMP exited before session completion")
                    continue
                if kind == "eof":
                    raise RuntimeError("OMP stdout closed before session completion")
                frame = decoder.feed(value)
                if frame is None:
                    continue
                log.write(json.dumps({"elapsed_seconds": time.monotonic() - started, "frame": frame}, ensure_ascii=False) + "\n")
                log.flush()
                typ = frame.get("type")
                if typ == "ready":
                    if 2 not in frame.get("supportedProtocolVersions", []):
                        raise RuntimeError("Pinned OMP did not advertise RPC v2")
                    send({"id": "protocol", "type": "negotiate_protocol", "protocolVersion": 2})
                    state = "protocol"
                elif typ == "response":
                    if frame.get("success") is False:
                        raise RuntimeError("RPC command failed: " + json.dumps(frame))
                    if frame.get("id") == "protocol":
                        send({"id": "filter", "type": "set_event_filter", "events": ["agent_start", "agent_end", "turn_start", "message_update", "message_end", "tool_execution_start", "tool_execution_end", "auto_retry_start", "model_changed"], "messageUpdates": "delta"})
                    elif frame.get("id") == "filter":
                        send({"id": "model", "type": "set_model", "provider": provider_id, "modelId": model_id})
                    elif frame.get("id") == "model":
                        send({"id": "thinking", "type": "set_thinking_level", "level": thinking})
                    elif frame.get("id") == "thinking":
                        send({"id": "compaction", "type": "set_auto_compaction", "enabled": False})
                    elif frame.get("id") == "compaction":
                        send({"id": "task", "type": "prompt", "message": task.prompt})
                        state = "running"
                elif typ == "turn_start":
                    turn_count += 1
                    if turn_count > max_turns:
                        send({"id": "abort-limit", "type": "abort"})
                        raise RuntimeError("Agent turn budget exceeded")
                elif typ == "tool_execution_start":
                    tools_started.append(frame)
                elif typ == "tool_execution_end":
                    tools_finished.append(frame)
                elif typ == "message_end" and frame.get("message", {}).get("role") == "assistant":
                    model_messages.append(frame["message"])
                elif typ == "model_changed":
                    changed = frame.get("model", {})
                    if isinstance(changed, dict) and changed.get("id") not in (None, model_id):
                        raise RuntimeError("Agent switched away from the requested model")
                elif typ == "prompt_result" and frame.get("id") == "task":
                    prompt_result = frame
                    if frame.get("sessionSettled"):
                        settled = True
                        break
                elif typ == "session_settled" and prompt_result:
                    settled = True
                    break
            if not settled:
                raise TimeoutError("OMP session did not settle within the wall-clock budget; state=" + state)
        except Exception as exc:
            error = str(exc)
        finally:
            try:
                if not settled:
                    send({"id": "abort-cleanup", "type": "abort"})
                proc.stdin.close()
                proc.wait(timeout=8)
            except (OSError, subprocess.SubprocessError):
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait(timeout=3)
            for reader in readers:
                reader.join(timeout=1)
    verification = _verify(task, workspace, case_dir, "candidate")
    tests_file, candidate_file = workspace / "test_public.py", workspace / task.filename
    unchanged_tests = tests_file.is_file() and tests_file.read_text() == task.public_tests
    names = [event.get("toolName", event.get("name")) for event in tools_started]
    observed_workflow = all(name in names for name in ("read", "edit", "bash"))
    tool_errors = sum(bool(event.get("isError")) for event in tools_finished)
    candidate_bytes = candidate_file.read_bytes() if candidate_file.is_file() else None
    changed_source = candidate_bytes is not None and candidate_bytes != task.source.encode()
    success = bool(not error and settled and prompt_result and prompt_result.get("status") == "completed" and
                   verification["passed"] and unchanged_tests and changed_source and observed_workflow)
    result = {"task_id": task.id, "task_kind": "local_coding_integration", "passed": success,
              "wall_seconds": time.monotonic() - started, "agent_turns": turn_count, "tool_names": names,
              "tool_errors": tool_errors, "observed_read_edit_test_workflow": observed_workflow,
              "prompt_result": prompt_result, "session_settled": settled, "error": error,
              "baseline_verification": baseline, "candidate_verification": verification,
              "public_tests_unchanged": unchanged_tests, "source_changed": changed_source,
              "model_messages": model_messages, "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest() if candidate_bytes is not None else None,
              "artifacts": {"events": str(case_dir / "events.jsonl"), "stderr": str(stderr_path), "workspace": str(workspace)}}
    (case_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run_evaluation(omp_path: str, base_url: str, model_id: str, output_dir: str, task_ids: list[str] | None = None,
                   max_seconds: float = 180, max_turns: int = 12, max_tokens: int = 2048, thinking: str = "off",
                   allow_scripted_backend: bool = False) -> dict:
    readiness = probe(omp_path, base_url)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if readiness["status"] != "ready":
        (output / "report.json").write_text(json.dumps(readiness, indent=2) + "\n")
        return readiness
    endpoint = readiness["endpoint"]
    real_model = endpoint.get("backend_kind") == "spark_loop72"
    if endpoint.get("model") != model_id:
        raise ValueError("Endpoint model identity differs from the requested model")
    if not real_model and not allow_scripted_backend:
        raise ValueError("Endpoint is not Spark Loop72; scripted backends are allowed only for explicit protocol testing")
    if real_model and endpoint.get("execution_manifest", {}).get("total_virtual_layers") != 72:
        raise ValueError("Endpoint does not report 72 virtual layers")
    selected = [task for task in TASKS if not task_ids or task.id in task_ids]
    if not selected or (task_ids and set(task_ids) != {t.id for t in selected}):
        raise ValueError("Unknown or empty task selection")
    rows = [run_task(task, readiness["omp_binary"], base_url, model_id, endpoint, output, max_seconds, max_turns, max_tokens, thinking) for task in selected]
    passed = sum(row["passed"] for row in rows)
    report = {"status": "completed", "evaluation_kind": "local_omp_coding_cases" if real_model else "real_omp_scripted_backend_protocol_check",
              "omp_version": OMP_VERSION, "omp_commit": OMP_COMMIT, "omp_binary_sha256": readiness["omp_binary_sha256"], "endpoint": endpoint, "thinking": thinking,
              "model_quality_evaluated": real_model, "official_benchmark_score": None,
              "local_task_success_rate": passed / len(rows) if real_model else None,
              "protocol_fixture_passed": passed == len(rows) if not real_model else None,
              "cases_run": len(rows), "cases": rows,
              "execution_isolation": "private workspace and agent config; run in a disposable container for OS isolation",
              "limitations": ["Local integration tasks are not an official coding benchmark.",
                              "Every model turn starts a fresh request KV cache; no cross-turn KV reuse is claimed."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", default="results/omp-coding")
    parser.add_argument("--task", action="append", choices=[task.id for task in TASKS])
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", choices=["off", "low", "high"], default="off")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        report = probe(args.omp, args.base_url)
    else:
        report = run_evaluation(args.omp, args.base_url, args.model_id, args.output, args.task, args.max_seconds, args.max_turns, args.max_tokens, args.thinking)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))
    if report["status"] not in {"ready", "completed"}:
        return 2
    return 0 if args.probe or all(row["passed"] for row in report.get("cases", [])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
