#!/usr/bin/env python3
"""Exercise the real pinned OMP executable with a SCRIPTED protocol fixture.

This is an HTTP/SSE/native-tool-parser/RPC/edit integration test. It does not
load Spark, measure model tokens per second, or evaluate model coding quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sparkx_omp_server import GenerationResult, PreparedPrompt, create_loop72_server
from omp_eval import run_evaluation


class ScriptedCodingBackend:
    backend_kind = "scripted_protocol_test"
    context_limit_tokens = 131072
    manifest = {"model_quality_evaluated": False, "fixture": "scripted native Spark tool markup; no weights loaded"}

    def __init__(self):
        self.requests = []

    def prepare(self, request):
        self.requests.append({"messages": request.messages, "tools": request.tools, "tool_names": [x["function"]["name"] for x in request.tools],
                              "template_kwargs": request.template_kwargs, "max_tokens": request.max_tokens})
        # Explicitly synthetic usage. This value is never treated as a measured
        # Spark tokenizer count or a supported long-context evaluation.
        return PreparedPrompt(list(range(16)), False)

    def generate(self, prepared, request, emit, should_stop):
        offered = {x["function"]["name"] for x in request.tools}
        if offered != {"read", "edit", "bash"}:
            raise RuntimeError("Unexpected real OMP tool set: " + repr(sorted(offered)))
        results = [m["content"] for m in request.messages if m["role"] == "tool"]
        if len(results) == 0:
            name, args = "read", {"path": "math_utils.py"}
        elif len(results) == 1:
            tag = re.search(r"math_utils\.py#([0-9a-fA-F]{4})\b", results[-1])
            if not tag:
                raise RuntimeError("Actual OMP read response did not contain its file snapshot tag: " + results[-1])
            patch = "*** Begin Patch\n[math_utils.py#" + tag.group(1) + "]\nPUT 4.=4:\n+    return min(upper, max(lower, value))\n*** End Patch"
            name, args = "edit", {"input": patch}
        elif len(results) == 2:
            name, args = "bash", {"command": "python -m unittest -v test_public.py", "timeout": 30}
        elif len(results) == 3:
            emit("The scripted transport fixture read the source, applied the bounded clamp edit, and ran the tests.")
            return GenerationResult(1, "stop", {"synthetic_usage": True})
        else:
            raise RuntimeError("Scripted fixture exceeded its expected four model turns")
        args["i"] = "Exercise the isolated protocol fixture"
        output = "<tool_call>" + name
        for key, value in args.items():
            text = value if isinstance(value, str) else json.dumps(value)
            output += "<arg_key>" + key + "</arg_key><arg_value>" + text + "</arg_value>"
        output += "</tool_call>"
        # Deliberately split control markers across packets to exercise buffering.
        for start in range(0, len(output), 7):
            if should_stop():
                raise InterruptedError("Protocol probe cancelled")
            emit(output[start:start + 7])
        return GenerationResult(1, "stop", {"synthetic_usage": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = ScriptedCodingBackend()
    server = create_loop72_server(backend=backend, port=0, model_id="protocol-fixture-only", max_output_tokens=2048)
    base_url = "http://127.0.0.1:" + str(server.server_port) + "/v1"
    try:
        report = run_evaluation(args.omp, base_url, "protocol-fixture-only", str(output), ["clamp_bounds"],
                                max_seconds=90, max_turns=6, max_tokens=2048, thinking="off", allow_scripted_backend=True)
    finally:
        server.shutdown()
        server.server_close()
        (output / "scripted_requests.json").write_text(json.dumps(backend.requests, ensure_ascii=False, indent=2) + "\n")
        (output / "server_records.json").write_text(json.dumps(server.request_records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    return 0 if report.get("protocol_fixture_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
