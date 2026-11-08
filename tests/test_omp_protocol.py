"""Protocol/negative-contract tests; scripted backends are not model evidence."""
import json
from pathlib import Path
import threading
import time
import tempfile
import types
import unittest
from unittest import mock
import urllib.error
import urllib.request

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

from sparkx_omp_server import (
    ChatRequest, GenerationResult, NativeSparkOutputParser, PreparedPrompt,
    ProtocolError, SparkGenerationBackend, create_loop72_server,
    generate_omp_models_yaml, normalize_messages,
)


MODEL = "protocol-fixture-only"
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"key": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["key", "count"], "additionalProperties": False,
}}}]


def request_body(**kwargs):
    return {"model": MODEL, "messages": [{"role": "user", "content": "Use the lookup tool."}], "max_tokens": 100, **kwargs}


class ScriptBackend:
    backend_kind = "scripted_protocol_test"
    context_limit_tokens = 4096
    manifest = {"model_quality_evaluated": False}

    def __init__(self, output="Done.", delay=0.0):
        self.output, self.delay = output, delay
        self.requests = []
        self.entered = threading.Event()
        self.cancelled = threading.Event()

    def prepare(self, request):
        self.requests.append(request)
        return PreparedPrompt(list(range(8)), request.template_kwargs.get("enable_thinking", False))

    def generate(self, prepared, request, emit, should_stop):
        self.entered.set()
        deadline = time.monotonic() + self.delay
        while time.monotonic() < deadline:
            if should_stop():
                self.cancelled.set()
                raise InterruptedError("test backend cancelled")
            time.sleep(0.005)
        text = self.output(request) if callable(self.output) else self.output
        for i in range(0, len(text), 3):
            if should_stop():
                self.cancelled.set()
                raise InterruptedError("test backend cancelled")
            emit(text[i:i + 3])
        return GenerationResult(len(text), "stop", {"synthetic_protocol_timing": True})


class HTTPFixture(unittest.TestCase):
    def serve(self, backend=None, **kwargs):
        backend = backend or ScriptBackend()
        server = create_loop72_server(backend=backend, port=0, model_id=MODEL, max_output_tokens=512, **kwargs)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_port}"

    @staticmethod
    def post(url, body):
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, dict(response.headers), response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read().decode()

    def test_native_tool_sse_then_matched_result_roundtrip(self):
        def output(request):
            if request.messages[-1]["role"] == "tool":
                self.assertEqual(request.messages[-2]["tool_calls"][0]["function"]["arguments"], {"key": "alpha", "count": 2})
                self.assertEqual(request.messages[-1]["content"], "two matches")
                return "Found two matches."
            return "<tool_call>lookup<arg_key>key</arg_key><arg_value>alpha</arg_value><arg_key>count</arg_key><arg_value>2</arg_value></tool_call>"
        backend = ScriptBackend(output)
        _, url = self.serve(backend)
        first = request_body(tools=TOOLS, stream=True, stream_options={"include_usage": True})
        status, headers, raw = self.post(url, first)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        frames = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        calls = [d["choices"][0]["delta"]["tool_calls"][0] for d in frames if d.get("choices") and "tool_calls" in d["choices"][0]["delta"]]
        self.assertEqual(len(calls), 1)
        call = {k: v for k, v in calls[0].items() if k != "index"}
        self.assertEqual(json.loads(call["function"]["arguments"]), {"key": "alpha", "count": 2})
        self.assertEqual(frames[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("completion_tokens", frames[-1]["usage"])
        second = request_body(tools=TOOLS, messages=first["messages"] + [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "two matches"},
        ])
        status, _, raw = self.post(url, second)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "Found two matches.")
        self.assertEqual(len(backend.requests), 2)

    def test_stream_arrives_before_generation_finishes(self):
        finished = threading.Event()
        class Backend(ScriptBackend):
            def generate(self, prepared, request, emit, should_stop):
                emit("First word ")
                finished.wait(2)
                emit("last.")
                return GenerationResult(4, "stop")
        _, url = self.serve(Backend())
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(request_body(stream=True)).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            while True:
                line = response.readline().decode()
                if '"content": "First word "' in line:
                    break
                self.assertTrue(line, "Stream closed before the early delta")
            self.assertFalse(finished.is_set())
            finished.set()
            self.assertIn("[DONE]", response.read().decode())

    def test_context_budget_and_unsupported_fields_fail_before_generation(self):
        backend = ScriptBackend()
        _, url = self.serve(backend, context_limit_tokens=520)
        status, _, raw = self.post(url, request_body(max_tokens=512, messages=[{"role": "user", "content": "x"}]))
        self.assertEqual(status, 200)  # 8 + 512 exactly fits.
        backend.context_limit_tokens = 519
        _, smaller = self.serve(backend, context_limit_tokens=519)
        status, _, raw = self.post(smaller, request_body(max_tokens=512))
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["code"], "context_length_exceeded")
        for field in ("response_format", "logit_bias", "stop"):
            status, _, raw = self.post(url, request_body(**{field: {}}))
            self.assertEqual(status, 400)
            self.assertIn(field, json.loads(raw)["error"]["message"])

    def test_single_model_is_serialized_and_timeout_cancels_worker(self):
        backend = ScriptBackend(delay=2)
        server, url = self.serve(backend, request_timeout_seconds=0.2)
        result = []
        thread = threading.Thread(target=lambda: result.append(self.post(url, request_body())))
        thread.start()
        self.assertTrue(backend.entered.wait(1))
        status, _, raw = self.post(url, request_body())
        self.assertEqual(status, 429)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0][0], 408)
        self.assertTrue(backend.cancelled.wait(1))
        for _ in range(50):
            if server.request_records:
                break
            time.sleep(0.01)
        self.assertEqual(server.request_records[-1]["status"], "cancelled")

    def test_invalid_tool_is_error_not_executable_text_success(self):
        backend = ScriptBackend("<tool_call>lookup<arg_key>key</arg_key><arg_value>x</arg_value><arg_key>count</arg_key><arg_value>not-an-integer</arg_value></tool_call>")
        _, url = self.serve(backend)
        status, _, raw = self.post(url, request_body(tools=TOOLS))
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(raw)["error"]["code"], "invalid_tool_arguments")
        backend.output = "No tool needed."
        status, _, raw = self.post(url, request_body(tools=TOOLS, tool_choice="required"))
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(raw)["error"]["code"], "tool_choice_not_satisfied")
        backend.output = "<tool_call>lookup<arg_key>key</arg_key><arg_value>x</arg_value>"
        status, _, raw = self.post(url, request_body(tools=TOOLS, stream=True))
        self.assertEqual(status, 200)
        frames = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertTrue(any(f.get("error", {}).get("code") == "incomplete_tool_call" for f in frames))
        self.assertFalse(any(f.get("choices") and f["choices"][0]["delta"].get("tool_calls") for f in frames))


class ParserContracts(unittest.TestCase):
    def test_rpc_unicode_chunks_and_broken_sequence(self):
        import base64
        from tools.omp_eval import RPCDecoder
        payload = json.dumps({"type": "message_end", "text": "東京 α"}, ensure_ascii=False).encode()
        pieces = [payload[:37], payload[37:40], payload[40:]]
        decoder = RPCDecoder()
        for index, part in enumerate(pieces):
            frame = {"type": "rpc_chunk", "chunkId": "a", "index": index, "count": len(pieces), "byteLength": len(payload),
                     "data": base64.b64encode(part).decode()}
            value = decoder.feed(json.dumps(frame))
        self.assertEqual(value["text"], "東京 α")
        decoder = RPCDecoder()
        frame.update(index=0, data=base64.b64encode(pieces[0]).decode())
        self.assertIsNone(decoder.feed(json.dumps(frame)))
        with self.assertRaisesRegex(ValueError, "interrupted"):
            decoder.feed('{"type":"ready"}')

    def test_thinking_markup_is_never_a_tool_and_unicode_strings_survive(self):
        req = ChatRequest.parse(request_body(tools=TOOLS), MODEL, 1000)
        parser = NativeSparkOutputParser(req, opens_thinking=True)
        raw = "Consider <tool_call>not_executable</tool_call></think><tool_call>lookup<arg_key>key</arg_key><arg_value>東京\nα</arg_value><arg_key>count</arg_key><arg_value>2</arg_value></tool_call>"
        deltas = []
        for char in raw:
            deltas.extend(parser.feed(char))
        deltas.extend(parser.feed("", final=True))
        message, reason = parser.finish("stop")
        self.assertEqual(reason, "tool_calls")
        self.assertIn("not_executable", message["reasoning_content"])
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"])["key"], "東京\nα")
        self.assertFalse(any("not_executable" in str(x.get("tool_calls", "")) for x in deltas))

    def test_parallel_tool_results_reordered_and_bad_ids_rejected(self):
        calls = [{"id": i, "type": "function", "function": {"name": "lookup", "arguments": '{"key":"x","count":1}'}} for i in ("a", "b")]
        messages = [{"role": "assistant", "tool_calls": calls}, {"role": "tool", "tool_call_id": "b", "content": "second"}, {"role": "tool", "tool_call_id": "a", "content": "first"}]
        result = normalize_messages(messages)
        self.assertEqual([m["content"] for m in result[1:]], ["first", "second"])
        self.assertIsInstance(result[0]["tool_calls"][0]["function"]["arguments"], dict)
        messages[2]["tool_call_id"] = "unknown"
        with self.assertRaisesRegex(ProtocolError, "tool_call_id"):
            normalize_messages(messages)

    def test_native_template_receives_schema_and_object_replay_no_fallback(self):
        class Tokenizer:
            chat_template = "native-fixture"
            seen = None
            def apply_chat_template(self, messages, **kwargs):
                self.seen = messages, kwargs
                if not kwargs.get("tools"):
                    raise ValueError("missing tool definitions")
                return [1, 2]
            def decode(self, *_args, **_kwargs):
                return "<|Bot|><think>"
        class Model:
            def execution_manifest(self):
                return {"total_virtual_layers": 72, "global_visits": 18}
            def generate_cached(self, input_ids, token_callback=None, should_stop=None):
                raise AssertionError("This test must not perform model generation")
        tokenizer = Tokenizer()
        backend = SparkGenerationBackend(Model(), tokenizer, 8192)
        req = ChatRequest.parse(request_body(tools=TOOLS), MODEL, 1000)
        prepared = backend.prepare(req)
        self.assertTrue(prepared.opens_thinking)
        self.assertEqual(tokenizer.seen[1]["tools"][0]["function"]["name"], "lookup")
        req.tools = []
        with self.assertRaisesRegex(ProtocolError, "no generic role-text fallback"):
            backend.prepare(req)

    def test_config_merge_does_not_erase_existing_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.yml"
            path.write_text('providers:\n  other:\n    baseUrl: http://example.invalid/v1\n    auth: none\n')
            generate_omp_models_yaml(str(path), provider_id="fixture", model_id=MODEL, context_window=8192, max_tokens=512)
            data = json.loads(path.read_text())
            self.assertEqual(data["providers"]["other"]["baseUrl"], "http://example.invalid/v1")
            self.assertIn("fixture", data["providers"])
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                generate_omp_models_yaml(str(path), provider_id="fixture")
            self.assertEqual(path.read_bytes(), before)


@unittest.skipIf(torch is None, "torch not installed")
class ActualGenerationBridge(unittest.TestCase):
    def test_core_generator_callback_to_http_sse_and_eos_usage(self):
        """Real generate_cached loop; deterministic fake logits, not Spark weights."""
        from sparkx_loop_v03_core import SparkXLoopV03

        class Model(torch.nn.Module):
            generate_cached = SparkXLoopV03.generate_cached
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(1, 1)
                self.settings = types.SimpleNamespace(use_paged_hybrid_cache=False, prefill_chunk_tokens=0)
                self.calls = 0
            def execution_manifest(self):
                return {"total_virtual_layers": 72, "global_visits": 18, "model_quality_evaluated": False}
            def forward_logits(self, input_ids, **kwargs):
                token = [6, 7, 2][min(self.calls, 2)]
                self.calls += 1
                logits = torch.full((1, 1, 16), -100.0)
                logits[0, 0, token] = 100.0
                return logits

        class Tokenizer:
            chat_template = "callback-fixture"
            eos_token_id = 2
            def apply_chat_template(self, *_args, **_kwargs): return [8, 9]
            def encode(self, text, **kwargs):
                return {"<think>": [3], "</think>": [4], "<｜end▁of▁sentence｜>": [2]}.get(text, [10, 11])
            def decode(self, ids, **kwargs):
                return "".join({6: "Hello ", 7: "world.", 2: "<｜end▁of▁sentence｜>"}.get(int(i), "x") for i in ids)

        model = Model().eval()
        cache_module = types.ModuleType("transformers.cache_utils")
        cache_module.DynamicCache = object
        with mock.patch.dict("sys.modules", {"transformers.cache_utils": cache_module}):
            server = create_loop72_server(model, Tokenizer(), port=0, model_id=MODEL, context_limit_tokens=1024, max_output_tokens=32)
            try:
                with mock.patch.object(model, "generate_cached", wraps=model.generate_cached) as generate:
                    status, _, raw = HTTPFixture.post(f"http://127.0.0.1:{server.server_port}", request_body(max_tokens=8, temperature=0, stream=True, stream_options={"include_usage": True}))
                self.assertIs(generate.call_args.kwargs["force_think_close"], False)
            finally:
                server.shutdown()
                server.server_close()
        self.assertEqual(status, 200)
        frames = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        text = "".join(frame["choices"][0]["delta"].get("content", "") for frame in frames if frame.get("choices"))
        self.assertEqual(text, "Hello world.")
        self.assertEqual(frames[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(frames[-1]["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.assertEqual(model.calls, 3)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
