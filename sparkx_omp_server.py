"""A local OpenAI Chat Completions bridge for the actual Spark Loop72 runtime.

No model weights are loaded on import. The HTTP/parser tests can inject a small
backend; their results are protocol checks, never model-quality measurements.
Oh My Pi is pinned to v18.18.1 in tools/omp_eval.py and docs/OMP_EVALUATION.md.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
import uuid


OMP_VERSION = "18.18.1"
OMP_COMMIT = "24439066040e006eb08da7b204f0ac41c79d2de9"
DEFAULT_MODEL_ID = "Spark-X2.6-4B-Recursive72"
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_END_MARKERS = ("<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>")


class ProtocolError(ValueError):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status, self.code = status, code

    def payload(self) -> dict:
        return {"error": {"message": str(self), "type": "invalid_request_error" if self.status < 500 else "server_error", "code": self.code}}


def _json_loads(text: str) -> Any:
    def bad_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(text, parse_constant=bad_constant)


def _content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict) or set(item) - {"type", "text"} or item.get("type") != "text" or not isinstance(item.get("text"), str):
                raise ProtocolError("Only text content is supported; images/audio are not silently discarded")
            parts.append(item["text"])
        return "".join(parts)
    raise ProtocolError("Message content must be text or an array of text blocks")


def normalize_messages(messages: Any) -> list[dict]:
    """Convert wire JSON arguments to native-template dictionaries, pair tools.

    Spark's template does not render tool_call_id. Parallel tool results are
    therefore reordered to their assistant call order before template rendering.
    """
    if not isinstance(messages, list) or not messages:
        raise ProtocolError("messages must be a nonempty array")
    out: list[dict] = []
    pending: dict[str, str] = {}
    results: dict[str, dict] = {}
    seen_ids: set[str] = set()
    for raw in messages:
        if not isinstance(raw, dict):
            raise ProtocolError("Each message must be an object")
        role = raw.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ProtocolError(f"Unsupported message role: {role!r}")
        if set(raw) - {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content"}:
            raise ProtocolError("Unsupported message field(s): " + ", ".join(sorted(set(raw) - {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content"})))
        if "name" in raw and role != "tool":
            raise ProtocolError("Named system/user/assistant messages are unsupported by the native template")
        if role != "assistant" and any(key in raw for key in ("tool_calls", "reasoning_content")):
            raise ProtocolError("tool_calls and reasoning_content are valid only on assistant messages")
        msg = {"role": role, "content": _content(raw.get("content"))}
        if role == "tool":
            cid = raw.get("tool_call_id")
            if not isinstance(cid, str) or cid not in pending or cid in results:
                raise ProtocolError("Tool result has an unknown, missing, or duplicate tool_call_id")
            if "name" in raw and raw["name"] != pending[cid]:
                raise ProtocolError("Tool result name does not match its assistant tool call")
            msg.update(tool_call_id=cid, name=pending[cid])
            results[cid] = msg
            if len(results) == len(pending):
                out.extend(results[c] for c in pending)
                pending, results = {}, {}
            continue
        if pending:
            raise ProtocolError("All assistant tool calls need matching tool results before the next message")
        if "tool_call_id" in raw:
            raise ProtocolError("tool_call_id is only valid on tool messages")
        if "reasoning_content" in raw:
            if role != "assistant" or not isinstance(raw["reasoning_content"], str):
                raise ProtocolError("reasoning_content must be a string on an assistant message")
            msg["reasoning_content"] = raw["reasoning_content"]
        if raw.get("tool_calls") is not None:
            if role != "assistant" or not isinstance(raw["tool_calls"], list):
                raise ProtocolError("tool_calls must be an array on an assistant message")
            calls = []
            for call in raw["tool_calls"]:
                if not isinstance(call, dict) or set(call) - {"id", "type", "function"} or call.get("type") != "function":
                    raise ProtocolError("Only function tool calls are supported")
                fn = call.get("function", {})
                cid = call.get("id")
                if not isinstance(fn, dict) or set(fn) - {"name", "arguments"} or not _NAME.fullmatch(str(fn.get("name", ""))):
                    raise ProtocolError("Malformed function tool call")
                if not isinstance(cid, str) or not cid or cid in seen_ids:
                    raise ProtocolError("Assistant tool call IDs must be nonempty and unique")
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = _json_loads(args)
                    except (ValueError, TypeError) as exc:
                        raise ProtocolError("Tool-call arguments are not valid JSON") from exc
                if not isinstance(args, dict):
                    raise ProtocolError("Tool-call arguments must encode an object")
                seen_ids.add(cid)
                pending[cid] = fn["name"]
                calls.append({"id": cid, "type": "function", "function": {"name": fn["name"], "arguments": args}})
            msg["tool_calls"] = calls
        out.append(msg)
    if pending:
        raise ProtocolError("Conversation ends with unresolved assistant tool calls")
    return out


def _validator(schema: dict):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ProtocolError("Install requirements-omp.txt to validate tool schemas", 503, "missing_dependency") from exc
    def check_refs(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef"} and (not isinstance(value, str) or not value.startswith("#")):
                    raise ProtocolError("External JSON Schema references are unsupported")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ProtocolError("Invalid tool parameter schema: " + str(exc).splitlines()[0]) from exc
    return Draft202012Validator(schema)


@dataclasses.dataclass
class ChatRequest:
    model: str
    messages: list[dict]
    tools: list[dict]
    tool_choice: Any
    stream: bool
    include_usage: bool
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    template_kwargs: dict
    parallel_tool_calls: bool = True

    @classmethod
    def parse(cls, body: Any, model_id: str, max_output_tokens: int) -> "ChatRequest":
        supported = {"model", "messages", "tools", "tool_choice", "stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "chat_template_kwargs", "reasoning_effort", "parallel_tool_calls", "n"}
        if not isinstance(body, dict) or set(body) - supported:
            extra = sorted(set(body) - supported) if isinstance(body, dict) else ["non-object body"]
            raise ProtocolError("Unsupported request field(s): " + ", ".join(extra))
        if body.get("model") != model_id:
            raise ProtocolError(f"Unknown model; expected {model_id!r}", 404, "model_not_found")
        if isinstance(body.get("n", 1), bool) or body.get("n", 1) != 1:
            raise ProtocolError("Only n=1 is supported")
        for name in ("stream", "parallel_tool_calls"):
            if name in body and not isinstance(body[name], bool):
                raise ProtocolError(f"{name} must be boolean")
        if "max_tokens" in body and "max_completion_tokens" in body and body["max_tokens"] != body["max_completion_tokens"]:
            raise ProtocolError("Conflicting max_tokens and max_completion_tokens")
        maximum = body.get("max_tokens", body.get("max_completion_tokens", min(2048, max_output_tokens)))
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= max_output_tokens:
            raise ProtocolError(f"Output budget must be an integer in [1, {max_output_tokens}]")
        def number(key: str, default: float, lo: float, hi: float) -> float:
            value = body.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ProtocolError(f"{key} must be a finite number in [{lo}, {hi}]")
            return float(value)
        top_k = body.get("top_k", 0)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ProtocolError("top_k must be a nonnegative integer")
        opts = body.get("stream_options", {})
        if not isinstance(opts, dict) or set(opts) - {"include_usage"} or not isinstance(opts.get("include_usage", False), bool):
            raise ProtocolError("Only stream_options.include_usage:boolean is supported")
        tk = copy.deepcopy(body.get("chat_template_kwargs", {}))
        if not isinstance(tk, dict) or set(tk) - {"enable_thinking", "reasoning_effort"}:
            raise ProtocolError("Only native enable_thinking/reasoning_effort template kwargs are supported")
        if "reasoning_effort" in body:
            if "reasoning_effort" in tk and tk["reasoning_effort"] != body["reasoning_effort"]:
                raise ProtocolError("Conflicting reasoning_effort fields")
            tk["reasoning_effort"] = body["reasoning_effort"]
        if "enable_thinking" in tk and not isinstance(tk["enable_thinking"], bool):
            raise ProtocolError("enable_thinking must be boolean")
        if "reasoning_effort" in tk and tk["reasoning_effort"] not in {"low", "high"}:
            raise ProtocolError("Spark supports reasoning_effort low or high; use enable_thinking:false for off")
        tools = copy.deepcopy(body.get("tools", []))
        if not isinstance(tools, list):
            raise ProtocolError("tools must be an array")
        names = set()
        for tool in tools:
            if not isinstance(tool, dict) or set(tool) - {"type", "function"} or tool.get("type") != "function":
                raise ProtocolError("Only function tools are supported")
            fn = tool.get("function")
            if not isinstance(fn, dict) or set(fn) - {"name", "description", "parameters", "strict"}:
                raise ProtocolError("Unsupported function schema field")
            name = fn.get("name", "")
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
                raise ProtocolError("Tool names must be valid and unique")
            names.add(name)
            if fn.get("strict") not in (None, False):
                raise ProtocolError("Constrained strict decoding is not implemented; strict:true is unsupported")
            if not isinstance(fn.get("parameters", {}), dict):
                raise ProtocolError("Function parameters must be a JSON Schema object")
            _validator(fn.get("parameters", {}))
        choice = body.get("tool_choice", "auto" if tools else "none")
        if isinstance(choice, dict):
            if set(choice) != {"type", "function"} or choice.get("type") != "function" or not isinstance(choice.get("function"), dict) or set(choice["function"]) != {"name"} or choice["function"]["name"] not in names:
                raise ProtocolError("Named tool_choice must name one of the provided function tools")
        elif choice not in ("auto", "none", "required"):
            raise ProtocolError("Unsupported tool_choice")
        if choice == "required" and not tools:
            raise ProtocolError("tool_choice required needs at least one tool")
        messages = normalize_messages(body.get("messages"))
        if choice == "none":
            tools = []
        elif choice == "required" or isinstance(choice, dict):
            name = choice["function"]["name"] if isinstance(choice, dict) else None
            if name:
                tools = [t for t in tools if t["function"]["name"] == name]
            instruction = "For this response, call " + (f"the function {name}." if name else "at least one of the available functions.")
            # This is prompt conditioning plus output validation, not a grammar/logit constraint.
            if messages[0]["role"] == "system":
                messages[0]["content"] += "\n\n" + instruction
            else:
                messages.insert(0, {"role": "system", "content": instruction})
        return cls(model_id, messages, tools, choice, body.get("stream", False), opts.get("include_usage", False), maximum,
                   number("temperature", 1.0, 0.0, 2.0), number("top_p", 0.95, 1e-9, 1.0), top_k,
                   number("repetition_penalty", 1.0, 1.0, 10.0), tk, body.get("parallel_tool_calls", True))


class NativeSparkOutputParser:
    """Incremental native Spark markup -> structured OpenAI deltas.

    Tool calls are exposed only after complete parsing/schema validation. Markup
    inside the thinking channel never becomes an executable tool call.
    """
    def __init__(self, request: ChatRequest, opens_thinking: bool):
        self.request = request
        self.state = "thinking" if opens_thinking else "text"
        self.buffer = ""
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: list[dict] = []
        self.schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in request.tools}

    @staticmethod
    def _hold_suffix(text: str, markers: tuple[str, ...]) -> int:
        return max([0] + [n for marker in markers for n in range(1, min(len(marker), len(text) + 1)) if text.endswith(marker[:n])])

    def _tool(self, payload: str) -> dict:
        split = payload.find("<arg_key>")
        name = (payload if split < 0 else payload[:split]).strip()
        if name not in self.schemas:
            raise ProtocolError(f"Model emitted unoffered tool {name!r}", 422, "invalid_tool_call")
        args: dict = {}
        rest = "" if split < 0 else payload[split:]
        schema = self.schemas[name]
        pattern = re.compile(r"\s*<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
        while rest.strip():
            match = pattern.match(rest)
            if not match:
                raise ProtocolError("Malformed native Spark tool arguments", 422, "invalid_tool_call")
            key, value = match.group(1).strip(), match.group(2)
            if key in args:
                raise ProtocolError("Duplicate tool argument", 422, "invalid_tool_call")
            prop = schema.get("properties", {}).get(key, {})
            typ = prop.get("type") if isinstance(prop, dict) else None
            if typ == "string":
                args[key] = value
            else:
                try:
                    args[key] = _json_loads(value)
                except ValueError:
                    args[key] = value
            rest = rest[match.end():]
        errors = list(_validator(schema).iter_errors(args))
        if errors:
            raise ProtocolError("Tool arguments failed schema validation: " + errors[0].message, 422, "invalid_tool_arguments")
        if not self.request.parallel_tool_calls and self.calls:
            raise ProtocolError("Model emitted multiple calls with parallel_tool_calls:false", 422, "invalid_tool_call")
        call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}}
        self.calls.append(call)
        return {"tool_calls": [{"index": len(self.calls) - 1, **call}]}

    def feed(self, text: str, final: bool = False) -> list[dict]:
        self.buffer += text
        out: list[dict] = []
        while self.buffer:
            if self.state == "tool":
                end = self.buffer.find("</tool_call>")
                if end < 0:
                    if final:
                        raise ProtocolError("Incomplete native tool call; no tool was emitted", 422, "incomplete_tool_call")
                    break
                out.append(self._tool(self.buffer[:end]))
                self.buffer = self.buffer[end + len("</tool_call>"):]
                self.state = "text"
                continue
            markers = ("</think>", "<think>") + _END_MARKERS if self.state == "thinking" else ("<tool_call>", "<think>", "</think>") + _END_MARKERS
            matches = [(self.buffer.find(m), m) for m in markers if self.buffer.find(m) >= 0]
            if matches:
                pos, marker = min(matches)
                chunk, self.buffer = self.buffer[:pos], self.buffer[pos + len(marker):]
            else:
                hold = 0 if final else self._hold_suffix(self.buffer, markers)
                end = len(self.buffer) - hold
                chunk, self.buffer = self.buffer[:end], self.buffer[end:]
                marker = None
            if chunk:
                if self.state == "thinking":
                    self.reasoning.append(chunk)
                    out.append({"reasoning_content": chunk})
                else:
                    self.content.append(chunk)
                    out.append({"content": chunk})
            if marker == "<tool_call>":
                self.state = "tool"
            elif marker == "<think>":
                self.state = "thinking"
            elif marker == "</think>":
                self.state = "text"
            if marker is None:
                break
        return out

    def finish(self, finish_reason: str) -> tuple[dict, str]:
        if self.state == "tool":
            raise ProtocolError("Incomplete native tool call", 422, "incomplete_tool_call")
        choice = self.request.tool_choice
        if (choice == "required" or isinstance(choice, dict)) and not self.calls:
            raise ProtocolError("Model did not satisfy required tool_choice", 422, "tool_choice_not_satisfied")
        if isinstance(choice, dict) and any(c["function"]["name"] != choice["function"]["name"] for c in self.calls):
            raise ProtocolError("Model did not satisfy named tool_choice", 422, "tool_choice_not_satisfied")
        msg: dict = {"role": "assistant", "content": "".join(self.content) or None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            msg["tool_calls"] = self.calls
        return msg, ("tool_calls" if self.calls and finish_reason == "stop" else finish_reason)


@dataclasses.dataclass
class PreparedPrompt:
    input_ids: list[int]
    opens_thinking: bool


@dataclasses.dataclass
class GenerationResult:
    completion_tokens: int
    finish_reason: str
    timing: dict = dataclasses.field(default_factory=dict)


class _TokenTextStreamer:
    """Bounded-window word/tag-boundary decoding, retaining incomplete UTF-8 until stable.

    Compacts the decoded token window (`max_window_tokens=48`, retaining a 4-token BPE
    context prefix) so long single-line JSON/patch tool arguments run in O(1) time per
    token instead of O(N^2) re-decoding (§11.7). Generated IDs are never skipped or changed.
    """
    def __init__(self, tokenizer: Any, emit: Callable[[str], None], max_window_tokens: int = 48, context_tokens: int = 4):
        self.tokenizer, self.emit = tokenizer, emit
        self.max_window_tokens = max(8, int(max_window_tokens))
        self.context_tokens = max(1, min(int(context_tokens), self.max_window_tokens // 2))
        self.cache: list[int] = []
        self.printed = 0

    def _compact_if_needed(self, text: str) -> None:
        if len(self.cache) <= self.max_window_tokens or self.printed < len(text):
            return
        ctx = self.cache[-self.context_tokens:]
        ctx_text = self.tokenizer.decode(ctx, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if text.endswith(ctx_text):
            self.cache = ctx
            self.printed = len(ctx_text)

    def put(self, token_id: int) -> None:
        self.cache.append(int(token_id))
        text = self.tokenizer.decode(self.cache, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        end = max(text.rfind(" ") + 1, text.rfind("\n") + 1, text.rfind(">") + 1)
        if len(self.cache) > self.max_window_tokens and end <= self.printed and not text.endswith("\ufffd"):
            # Long unbroken JSON/identifier token run without whitespace/newline: flush clean UTF-8
            end = len(text)
        if end > self.printed:
            self.emit(text[self.printed:end])
            self.printed = end
        if text.endswith("\n"):
            self.cache, self.printed = [], 0
        else:
            self._compact_if_needed(text)

    def finish(self) -> None:
        text = self.tokenizer.decode(self.cache, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if text[self.printed:]:
            self.emit(text[self.printed:])


class SparkGenerationBackend:
    """Calls the loaded Loop72 object, never a replacement base-model server."""
    backend_kind = "spark_loop72"

    def __init__(self, model: Any, tokenizer: Any, context_limit_tokens: Optional[int] = None, enable_prefix_cache: bool = False):
        self.model, self.tokenizer = model, tokenizer
        if getattr(model, "training", False):
            raise ValueError("Call model.eval() before serving so the manifest reports the executed inference recurrence")
        self.manifest = model.execution_manifest()
        if self.manifest.get("total_virtual_layers") != 72 or self.manifest.get("global_visits") != 18:
            raise ValueError("The OMP endpoint requires an actual 72-visit / 18-global runtime")
        signature = inspect.signature(model.generate_cached)
        if not {"token_callback", "should_stop"}.issubset(signature.parameters):
            raise ValueError("generate_cached needs the explicit token_callback/should_stop streaming seam; buffered output is not presented as token streaming")
        self._accepts_cache_kwarg = "cache" in signature.parameters
        configured = getattr(getattr(model, "paged_cache_cfg", None), "context_limit_tokens", None)
        if context_limit_tokens is None:
            if not isinstance(configured, int) or configured <= 0:
                raise ValueError("Provide the measured/configured context_limit_tokens explicitly")
            context_limit_tokens = configured
        self.context_limit_tokens = min(int(context_limit_tokens), int(configured)) if isinstance(configured, int) and configured > 0 else int(context_limit_tokens)
        self.enable_prefix_cache = bool(enable_prefix_cache and self._accepts_cache_kwarg and callable(getattr(model, "create_inference_cache", None)))
        self._prefix_state: Optional[dict] = None
        self._template_sha256 = hashlib.sha256(str(getattr(tokenizer, "chat_template", "")).encode()).hexdigest()
        self.manifest = {**self.manifest, "tokenizer_chat_template_sha256": self._template_sha256,
                         "cross_request_kv_reuse": self.enable_prefix_cache, "generation_policy": {"force_think_close": False}}

    def _runtime_fingerprint(self) -> str:
        cur_m = self.model.execution_manifest()
        sel_sha = (cur_m.get("selector_checkpoint_status") or {}).get("selector_sha256") or (cur_m.get("selector_checkpoint_status") or {}).get("checkpoint_sha256")
        return json.dumps({
            "template": self._template_sha256,
            "behavior": cur_m.get("behavior_identity"),
            "recurrence": cur_m.get("effective_recurrence_mode"),
            "selector": sel_sha,
        }, sort_keys=True)

    def reset_prefix_cache(self) -> None:
        self._prefix_state = None

    def prepare(self, request: ChatRequest) -> PreparedPrompt:
        if getattr(self.model, "training", False):
            raise ProtocolError("Model was switched into training mode after server startup", 503, "runtime_identity_changed")
        try:
            ids = self.tokenizer.apply_chat_template(request.messages, tools=request.tools or None, tokenize=True,
                                                     add_generation_prompt=True, return_dict=False, **request.template_kwargs)
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            elif hasattr(ids, "input_ids"):
                ids = ids.input_ids
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
                ids = ids[0]
            if not isinstance(ids, list) or not ids or any(isinstance(x, bool) or not isinstance(x, int) for x in ids):
                raise ValueError("Tokenizer did not return a single nonempty token sequence")
            tail = self.tokenizer.decode(ids[-16:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return PreparedPrompt(ids, tail.endswith("<think>"))
        except Exception as exc:
            raise ProtocolError("Native chat-template rendering failed; no generic role-text fallback: " + str(exc), 400, "chat_template_error") from exc

    def generate(self, prepared: PreparedPrompt, request: ChatRequest, emit: Callable[[str], None], should_stop: Callable[[], bool]) -> GenerationResult:
        import torch
        device = next(self.model.base.parameters()).device
        streamer = _TokenTextStreamer(self.tokenizer, emit)
        fp = self._runtime_fingerprint()
        reuse_cache = None
        prefix_hit_tokens = 0
        feed_ids = prepared.input_ids
        if self.enable_prefix_cache and self._prefix_state is not None:
            st = self._prefix_state
            committed: list[int] = st.get("committed_ids", [])
            cand_cache = st.get("cache")
            if (
                st.get("fingerprint") == fp
                and cand_cache is not None
                and hasattr(cand_cache, "get_seq_length")
                and int(cand_cache.get_seq_length()) == len(committed)
                and 0 < len(committed) < len(prepared.input_ids)
                and prepared.input_ids[:len(committed)] == committed
            ):
                reuse_cache = cand_cache
                prefix_hit_tokens = len(committed)
                feed_ids = prepared.input_ids[prefix_hit_tokens:]
            else:
                self._prefix_state = None
        if self.enable_prefix_cache and reuse_cache is None:
            reuse_cache = self.model.create_inference_cache(session_id="omp_prefix_session")

        inp = torch.tensor([feed_ids], dtype=torch.long, device=device)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=request.max_tokens,
            temperature=max(request.temperature, 1e-9),
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            tokenizer=self.tokenizer,
            stream_print=False,
            force_think_close=False,
            token_callback=streamer.put,
            should_stop=should_stop,
        )
        if reuse_cache is not None:
            gen_kwargs["cache"] = reuse_cache
        try:
            with torch.inference_mode():
                out = self.model.generate_cached(inp, **gen_kwargs)
        except Exception:
            self._prefix_state = None
            raise
        streamer.finish()
        timing = dict(getattr(self.model, "last_generation_timing", {}) or {})
        gen_ids = out[0, len(feed_ids):].tolist()
        count = len(gen_ids)
        reason = timing.get("finish_reason")
        if reason == "degenerate_loop":
            reason = "length"
        if reason not in {"stop", "length"}:
            self._prefix_state = None
            raise ProtocolError("Runtime did not report a valid generation finish_reason", 500, "runtime_contract_error")
        if self.enable_prefix_cache and reuse_cache is not None and hasattr(reuse_cache, "get_seq_length"):
            timing["prefix_cache_hit"] = bool(prefix_hit_tokens > 0)
            timing["prefix_cached_tokens"] = int(prefix_hit_tokens)
            expected_cached = prepared.input_ids + (gen_ids[:-1] if len(gen_ids) >= 1 else [])
            if int(reuse_cache.get_seq_length()) == len(expected_cached):
                self._prefix_state = {"fingerprint": fp, "committed_ids": expected_cached, "cache": reuse_cache}
            else:
                self._prefix_state = None
        return GenerationResult(count, reason, timing)


class Loop72HTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def create_loop72_server(model: Any = None, tokenizer: Any = None, host: str = "127.0.0.1", port: int = 8000,
                         model_id: str = DEFAULT_MODEL_ID, background: bool = True, *, backend: Any = None,
                         context_limit_tokens: Optional[int] = None, max_output_tokens: int = 4096,
                         api_key: Optional[str] = None, request_timeout_seconds: float = 900.0,
                         max_request_bytes: int = 16 * 1024 * 1024, enable_prefix_cache: bool = False) -> Loop72HTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ValueError("A non-loopback listener requires an explicit api_key")
    backend = backend if backend is not None else SparkGenerationBackend(model, tokenizer, context_limit_tokens, enable_prefix_cache=enable_prefix_cache)
    context_limit = min(int(context_limit_tokens or backend.context_limit_tokens), int(backend.context_limit_tokens))
    if context_limit < 2 or not 1 <= max_output_tokens < context_limit:
        raise ValueError("Require 1 <= max_output_tokens < context_limit_tokens")
    generation_lock = threading.Lock()
    records: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            pass

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def _authorized(self) -> bool:
            if api_key and self.headers.get("Authorization") != "Bearer " + api_key:
                self._json(401, ProtocolError("Invalid API key", 401, "authentication_error").payload())
                return False
            return True

        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0,
                           "owned_by": "spark-loop72", "context_length": context_limit, "max_model_len": context_limit,
                           "max_output_tokens": max_output_tokens}]})
            elif self.path.rstrip("/") in {"/health", "/v1/health"}:
                self._json(200, {"status": "ok", "model": model_id, "backend_kind": backend.backend_kind,
                           "context_limit_tokens": context_limit, "max_output_tokens": max_output_tokens,
                           "omp_version": OMP_VERSION, "execution_manifest": backend.manifest,
                           "serialized_requests": True, "generation_active": generation_lock.locked(),
                           "tool_choice_enforcement": "native prompt plus validated output; no constrained decoding"})
            else:
                self._json(404, ProtocolError("Unknown endpoint", 404).payload())

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._json(404, ProtocolError("Unknown endpoint", 404).payload())
                return
            self.connection.settimeout(min(30.0, request_timeout_seconds))
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ProtocolError("Chunked request bodies are unsupported")
                size = int(self.headers.get("Content-Length", "-1"))
                if not 0 < size <= max_request_bytes:
                    raise ProtocolError("Invalid or oversized Content-Length", 413, "request_too_large")
                body = _json_loads(self.rfile.read(size).decode("utf-8"))
                request = ChatRequest.parse(body, model_id, max_output_tokens)
            except (ValueError, UnicodeError, OSError) as exc:
                error = exc if isinstance(exc, ProtocolError) else ProtocolError(str(exc))
                self._json(error.status, error.payload())
                return
            if not generation_lock.acquire(blocking=False):
                self._json(429, ProtocolError("One generation is active; retry after it completes", 429, "server_busy").payload())
                return
            try:
                prepared = backend.prepare(request)
                if len(prepared.input_ids) + request.max_tokens > context_limit:
                    raise ProtocolError(f"Prompt ({len(prepared.input_ids)}) + output budget ({request.max_tokens}) exceeds configured context ({context_limit})", 400, "context_length_exceeded")
            except Exception as exc:
                generation_lock.release()
                error = exc if isinstance(exc, ProtocolError) else ProtocolError(str(exc), 500, "backend_error")
                self._json(error.status, error.payload())
                return
            request_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            events: queue.Queue = queue.Queue(maxsize=128)
            cancelled = threading.Event()
            deadline = time.monotonic() + request_timeout_seconds
            started = time.perf_counter()

            def stopped() -> bool:
                return cancelled.is_set() or time.monotonic() >= deadline

            def enqueue(event: tuple) -> None:
                while not stopped():
                    try:
                        events.put(event, timeout=0.1)
                        return
                    except queue.Full:
                        pass
                raise InterruptedError("HTTP generation cancelled or timed out")

            def worker() -> None:
                record = {"request_id": request_id, "backend_kind": backend.backend_kind, "prompt_tokens": len(prepared.input_ids),
                          "requested_max_tokens": request.max_tokens, "history_tool_results": sum(m["role"] == "tool" for m in request.messages),
                          "template_kwargs": request.template_kwargs, "status": "running"}
                parser = NativeSparkOutputParser(request, prepared.opens_thinking)
                try:
                    def emit(text: str) -> None:
                        for delta in parser.feed(text):
                            enqueue(("delta", delta))
                    result = backend.generate(prepared, request, emit, stopped)
                    for delta in parser.feed("", final=True):
                        enqueue(("delta", delta))
                    message, reason = parser.finish(result.finish_reason)
                    usage = {"prompt_tokens": len(prepared.input_ids), "completion_tokens": result.completion_tokens,
                             "total_tokens": len(prepared.input_ids) + result.completion_tokens}
                    record.update(status="completed", finish_reason=reason, usage=usage, loop72_timing=result.timing,
                                  tool_names=[c["function"]["name"] for c in parser.calls])
                    enqueue(("done", {"message": message, "finish_reason": reason, "usage": usage, "loop72_timing": result.timing}))
                except Exception as exc:
                    error = exc if isinstance(exc, ProtocolError) else ProtocolError(str(exc), 500, "generation_cancelled" if isinstance(exc, InterruptedError) else "backend_error")
                    record.update(status="cancelled" if isinstance(exc, InterruptedError) else "error", error=error.payload())
                    if not stopped():
                        enqueue(("error", error))
                finally:
                    record["server_wall_seconds"] = time.perf_counter() - started
                    records.append(record)
                    generation_lock.release()

            threading.Thread(target=worker, daemon=True, name="spark-loop72-generation").start()
            streaming_started = False
            completed = False
            def sse(data: Any) -> None:
                wire = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, allow_nan=False)
                self.wfile.write(("data: " + wire + "\n\n").encode())
                self.wfile.flush()
            def chunk(delta: dict, reason: Optional[str] = None) -> dict:
                return {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
            try:
                if request.stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    streaming_started = True
                    sse(chunk({"role": "assistant", "content": ""}))
                while not stopped():
                    try:
                        kind, data = events.get(timeout=1.0)
                    except queue.Empty:
                        if request.stream:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                        continue
                    if kind == "delta":
                        if request.stream:
                            sse(chunk(data))
                    elif kind == "error":
                        if request.stream:
                            sse(data.payload())
                            sse(chunk({}, "error"))
                            sse("[DONE]")
                        else:
                            self._json(data.status, data.payload())
                        completed = True
                        break
                    elif kind == "done":
                        if request.stream:
                            sse(chunk({}, data["finish_reason"]))
                            if request.include_usage:
                                sse({"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                                     "choices": [], "usage": data["usage"]})
                            sse("[DONE]")
                        else:
                            self._json(200, {"id": request_id, "object": "chat.completion", "created": created, "model": model_id,
                                       "choices": [{"index": 0, "message": data["message"], "finish_reason": data["finish_reason"]}],
                                       "usage": data["usage"], "loop72_timing": data["loop72_timing"]})
                        completed = True
                        break
                if not completed:
                    error = ProtocolError("Generation exceeded request deadline", 408, "request_timeout")
                    if streaming_started:
                        sse(error.payload())
                        sse(chunk({}, "error"))
                        sse("[DONE]")
                    else:
                        self._json(error.status, error.payload())
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                if not completed:
                    cancelled.set()
                self.close_connection = True

    server = Loop72HTTPServer((host, int(port)), Handler)
    server.backend, server.request_records = backend, records
    server.context_limit_tokens, server.model_id = context_limit, model_id
    if background:
        threading.Thread(target=server.serve_forever, daemon=True, name="spark-loop72-http").start()
    return server


def generate_omp_models_yaml(output_path: Optional[str] = None, port: int = 8000, model_id: str = DEFAULT_MODEL_ID,
                             context_window: int = 131072, max_tokens: int = 4096, *, base_url: Optional[str] = None,
                             provider_id: str = "spark_loop72_local", replace_provider: bool = False) -> dict:
    """Return an example; explicit writes merge providers and reject collisions.

    A model declaration is a client budget, not evidence of supported context.
    Pass the endpoint's measured/configured limits, not an aspirational maximum.
    """
    if not 1 <= int(max_tokens) < int(context_window):
        raise ValueError("max_tokens must be positive and below context_window")
    url = base_url or f"http://127.0.0.1:{int(port)}/v1"
    provider = {"baseUrl": url, "auth": "none", "api": "openai-completions", "models": [{
        "id": model_id, "name": "Spark X2.6 4B Loop72 D-LSA", "reasoning": True, "input": ["text"],
        "contextWindow": int(context_window), "maxTokens": int(max_tokens),
        "thinking": {"mode": "effort", "efforts": ["low", "high"], "defaultLevel": "low", "requiresEffort": False},
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsUsageInStreaming": True,
                   "supportsStrictMode": False, "toolStrictMode": "none", "supportsToolChoice": True,
                   "supportsForcedToolChoice": True, "maxTokensField": "max_tokens", "thinkingFormat": "qwen-chat-template",
                   "qwenTemplateReasoningEffort": True, "supportsReasoningEffort": True,
                   "reasoningContentField": "reasoning_content", "requiresReasoningContentForToolCalls": True}}]}
    document: dict = {"providers": {provider_id: provider}}
    path = Path(output_path).expanduser() if output_path is not None else None
    if path is not None and path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to merge an existing models.yml safely") from exc
        existing = yaml.safe_load(path.read_text())
        if not isinstance(existing, dict) or not isinstance(existing.get("providers", {}), dict):
            raise ValueError("Existing models file must contain a providers mapping")
        if provider_id in existing.get("providers", {}) and not replace_provider:
            raise ValueError(f"Provider {provider_id!r} already exists; explicit replace_provider=True is required")
        document = existing
        document.setdefault("providers", {})[provider_id] = provider
    # JSON is a YAML subset; this remains accepted as models.yml without PyYAML.
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
            handle.write(text)
            temp = handle.name
        os.replace(temp, path)
    return {"model_id": model_id, "provider_id": provider_id, "base_url": url, "context_window": int(context_window),
            "max_tokens": int(max_tokens), "written_path": str(path) if path else None, "yaml": text,
            "omp_version": OMP_VERSION, "omp_commit": OMP_COMMIT,
            "omp_cli_example": f"omp --model {provider_id}/{model_id} --thinking off"}
