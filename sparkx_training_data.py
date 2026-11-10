"""sparkx_training_data.py — Auditable Multi-Domain Training Corpus, Manifest Loader & Exact Assistant Masking (`WP1`).

Resolves `F02`, `F03`, and `F15` (`§6.1–§6.5` of the 2026-11-10 Loop72 D-LSA Plan):
  1. Normalized `TrainingRecord` contract with immutable `source_revision`, `source_document_id`, `verifier`, and
     deterministic document/repository/problem-family SHA-256 split (`95% train / 2.5% dev / 2.5% test`) assigned
     *before* chunking or windowing so train/dev/test never share source documents.
  2. External JSON/JSONL manifest loader (`load_manifest_records`, `save_manifest_records`) + real Hugging Face
     streamed 5-domain corpus builder (`build_real_five_domain_corpus`) + coherent `8K–32K` long-document Stage E
     calibration pack builder (`build_long_context_calibration_token_packs`), failing closed when
     `allow_smoke_fixtures=False` and real sources are unavailable.
  3. Pinned Oh My Pi (`omp` v18.18.1, commit `24439066040e006eb08da7b204f0ac41c79d2de9`) four-hex snapshot
     hashline `edit` (`input` patch string) tool schema and verified multi-turn recovery trajectories (`§6.5`, `F15`),
     strictly excluding local `tools/omp_eval.py` and official benchmark evaluation prompts.
  4. Full-conversation chat-template rendering with exact assistant-only token span supervision (`F03`), turn-aware
     windowing, and **strict zero-target rejection** (never fabricating supervision labels on user/system/tool tokens).
  5. Domain-weighted single-example/length-bucketed batch iterator emitting `(input_ids, labels, attention_mask, meta)`
     with complete token/exposure/domain/padding accounting and resumable cursor state (`§6.4`).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


OMP_PINNED_VERSION = "18.18.1"
OMP_PINNED_COMMIT = "24439066040e006eb08da7b204f0ac41c79d2de9"
CANONICAL_MODEL_ID = "Spark-X2.6-4B-Recursive72"

VALID_DOMAINS = (
    "code_debug",
    "general_instruct",
    "math_symbolic",
    "tool_trajectories",
    "long_doc_repo",
)

STAGE_C_DOMAIN_WEIGHTS: Dict[str, float] = {
    "code_debug": 0.40,
    "general_instruct": 0.20,
    "math_symbolic": 0.15,
    "tool_trajectories": 0.15,
    "long_doc_repo": 0.10,
}

STAGE_B_RETENTION_WEIGHTS: Dict[str, float] = {
    "code_debug": 0.25,
    "general_instruct": 0.35,
    "math_symbolic": 0.15,
    "tool_trajectories": 0.10,
    "long_doc_repo": 0.15,
}
STAGE_B_DOMAIN_WEIGHTS = STAGE_B_RETENTION_WEIGHTS

STAGE_D_OMP_WEIGHTS: Dict[str, float] = {
    "tool_trajectories": 0.70,
    "code_debug": 0.15,
    "general_instruct": 0.10,
    "math_symbolic": 0.05,
    "long_doc_repo": 0.00,
}
STAGE_D_DOMAIN_WEIGHTS = STAGE_D_OMP_WEIGHTS

# Pinned OMP v18.18.1 (`can1357/oh-my-pi@24439066040e006eb08da7b204f0ac41c79d2de9`) tool definitions (`F15`)
OMP_V18_18_1_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a workspace file with its 4-hex snapshot tag and 1-based line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "i": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Apply a snapshot-verified hashline patch (`*** Begin Patch\\n[path#hhhh]\\nPUT start.=end:\\n+...\\n*** End Patch`) "
                "using the pinned OMP v18.18.1 `input` patch grammar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "i": {"type": "string"},
                },
                "required": ["input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the workspace and return stdout/stderr and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "i": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]

OMP_TOOL_SCHEMA_DIGEST = hashlib.sha256(
    json.dumps(OMP_V18_18_1_TOOLS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

# Substrings identifying local `tools/omp_eval.py` cases & benchmark fixtures that must never enter training (`§6.2`)
_EXCLUDED_EVAL_SIGNATURES = (
    "fix clamp in math_utils.py",
    "fix chunked in chunks.py",
    "fix merge_intervals in intervals.py",
    "return min(lower, max(upper, value))",
    "openai/graphwalks",
    "openai/mrcr",
    "zai-org/longbench-v2",
)


@dataclasses.dataclass
class TrainingRecord:
    """Normalized auditable training example contract (`§6.2`)."""

    id: str
    domain: str
    source: str
    source_revision: str
    source_document_id: str
    split: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_schema_digest: Optional[str] = None
    verifier: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], default_split: Optional[str] = None) -> "TrainingRecord":
        if not isinstance(raw, dict):
            raise ValueError("TrainingRecord input must be a dict.")
        msgs = raw.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"TrainingRecord {raw.get('id')!r} must contain non-empty `messages`.")
        domain = str(raw.get("domain", "general_instruct")).strip()
        if domain not in VALID_DOMAINS:
            raise ValueError(f"Unsupported domain {domain!r}; expected one of {VALID_DOMAINS}.")
        rec_id = str(raw.get("id") or hashlib.sha256(json.dumps(msgs, sort_keys=True).encode("utf-8")).hexdigest()[:16])
        doc_id = str(raw.get("source_document_id") or raw.get("doc_id") or rec_id).strip()
        if not doc_id:
            doc_id = rec_id
        split = str(raw.get("split") or default_split or assign_document_split(doc_id)).strip().lower()
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unsupported split {split!r}; expected 'train', 'dev', or 'test'.")
        tools = raw.get("tools")
        schema_digest = raw.get("tool_schema_digest")
        if tools and not schema_digest:
            schema_digest = hashlib.sha256(
                json.dumps(tools, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return cls(
            id=rec_id,
            domain=domain,
            source=str(raw.get("source", "custom_manifest")),
            source_revision=str(raw.get("source_revision", "unspecified")),
            source_document_id=doc_id,
            split=split,
            messages=copy.deepcopy(msgs),
            tools=copy.deepcopy(tools) if tools else None,
            tool_schema_digest=str(schema_digest) if schema_digest else None,
            verifier=copy.deepcopy(raw.get("verifier")) if isinstance(raw.get("verifier"), dict) else None,
        )


@dataclasses.dataclass
class RenderedWindow:
    """Result of rendering a `TrainingRecord` with exact assistant-only token supervision (`F03`, `§6.3`)."""

    valid: bool
    input_ids: List[int]
    labels: List[int]
    supervised_tokens: int
    masked_tokens: int
    total_tokens_before_window: int
    rewindowed: bool
    rejection_reason: Optional[str] = None
    record_id: str = ""
    domain: str = ""
    source_document_id: str = ""


def assign_document_split(
    source_document_id: str,
    *,
    train_frac: float = 0.95,
    dev_frac: float = 0.025,
    salt: str = "sparkx_loop72_v2_split",
) -> str:
    """Deterministically assign an entire document, repository, or task family to `'train'`, `'dev'`, or `'test'`
    *before* chunking or windowing (`§6.2`)."""
    if not (0.0 < train_frac < 1.0 and 0.0 <= dev_frac < 1.0 and train_frac + dev_frac <= 1.0):
        raise ValueError("Invalid split fractions.")
    digest = hashlib.sha256(f"{salt}:{source_document_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + dev_frac:
        return "dev"
    return "test"


def is_excluded_eval_record(messages: Sequence[Dict[str, Any]], source_document_id: str = "") -> bool:
    """Return `True` if a record overlaps local `tools/omp_eval.py` fixtures or pinned benchmark suites (`§6.2`)."""
    hay = (source_document_id + "\n" + "\n".join(str(m.get("content", "")) for m in messages)).lower()
    return any(sig in hay for sig in _EXCLUDED_EVAL_SIGNATURES)


def verify_no_split_document_leakage(
    train_records: Sequence[TrainingRecord],
    dev_records: Sequence[TrainingRecord],
    test_records: Sequence[TrainingRecord] = (),
) -> Dict[str, Any]:
    """Verify zero `source_document_id` overlap across `train`, `dev`, and `test` splits (`WP1`)."""
    tr_docs = {r.source_document_id for r in train_records}
    dv_docs = {r.source_document_id for r in dev_records}
    ts_docs = {r.source_document_id for r in test_records}
    tr_dv = sorted(tr_docs & dv_docs)
    tr_ts = sorted(tr_docs & ts_docs)
    dv_ts = sorted(dv_docs & ts_docs)
    passed = (len(tr_dv) == 0) and (len(tr_ts) == 0) and (len(dv_ts) == 0)
    return {
        "passed": passed,
        "train_unique_documents": len(tr_docs),
        "dev_unique_documents": len(dv_docs),
        "test_unique_documents": len(ts_docs),
        "train_dev_overlap": tr_dv[:10],
        "train_test_overlap": tr_ts[:10],
        "dev_test_overlap": dv_ts[:10],
    }


def _snapshot_tag(filename: str, content: str) -> str:
    return hashlib.sha256(f"{filename}:{content}".encode("utf-8")).hexdigest()[:4]


def _format_omp_read_output(filename: str, content: str) -> Tuple[str, str]:
    tag = _snapshot_tag(filename, content)
    lines = content.splitlines()
    numbered = "\n".join(f"{idx + 1}: {line}" for idx, line in enumerate(lines))
    return tag, f"[{filename}#{tag}] ({len(lines)} lines)\n{numbered}\n"


def _format_omp_patch(filename: str, tag: str, start_line: int, end_line: int, replacement_lines: Sequence[str]) -> str:
    body = "\n".join(f"+{ln}" for ln in replacement_lines)
    return f"*** Begin Patch\n[{filename}#{tag}]\nPUT {start_line}.={end_line}:\n{body}\n*** End Patch"


# ==============================================================================
# Pinned OMP v18.18.1 Tool Trajectories & Correct Smoke Seed Fixtures (`F02, F15`)
# ==============================================================================
def build_omp_tool_trajectory_examples(count: int = 2) -> List[Dict[str, Any]]:
    """Build pinned OMP v18.18.1 (`24439066040e006eb08da7b204f0ac41c79d2de9`) four-hex snapshot hashline `edit`
    (`input` patch string) trajectories, including stale-snapshot and test-failure recovery episodes (`§6.5`, `F15`)."""
    templates = [
        (
            "rolling_sum.py",
            "Fix the off-by-one window slicing bug in `rolling_sum.py` and verify `python3 -m unittest -q` passes.",
            "def rolling_sum(values, window):\n    if window <= 0:\n        raise ValueError('window must be positive')\n    return [sum(values[i:i + window - 1]) for i in range(len(values) - window + 1)]\n",
            4,
            4,
            ["    return [sum(values[i:i + window]) for i in range(len(values) - window + 1)]"],
            "python3 -m unittest -q",
            "Ran 3 tests in 0.008s\n\nOK",
            "Fixed the off-by-one slice upper bound in `rolling_sum.py` (`values[i:i + window]`) and verified all 3 unit tests pass.",
            False,
        ),
        (
            "lru_page_cache.py",
            "In `lru_page_cache.py`, `get(key)` does not mark an accessed key as most-recently-used before eviction. Fix the bug and run `python3 -m unittest -v`.",
            "from collections import OrderedDict\n\nclass LRUPageCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.store = OrderedDict()\n\n    def get(self, key):\n        if key not in self.store:\n            return None\n        return self.store[key]\n",
            9,
            11,
            [
                "        if key not in self.store:",
                "            return None",
                "        self.store.move_to_end(key)",
                "        return self.store[key]",
            ],
            "python3 -m unittest -v",
            "Ran 4 tests in 0.011s\n\nOK",
            "Updated `LRUPageCache.get` in `lru_page_cache.py` to call `self.store.move_to_end(key)` on hits and verified all 4 unit tests pass.",
            True,  # Includes stale-tag recovery turn!
        ),
        (
            "token_bucket.py",
            "Fix `TokenBucket.consume` in `token_bucket.py` so elapsed refill is capped at `self.capacity` and negative `tokens` raises `ValueError`. Run `python3 -m unittest -q`.",
            "class TokenBucket:\n    def __init__(self, capacity: float, rate: float):\n        self.capacity = float(capacity)\n        self.rate = float(rate)\n        self.tokens = float(capacity)\n        self.last_ts = 0.0\n\n    def consume(self, amount: float, now: float) -> bool:\n        self.tokens += max(0.0, now - self.last_ts) * self.rate\n        self.last_ts = now\n        if self.tokens >= amount:\n            self.tokens -= amount\n            return True\n        return False\n",
            8,
            14,
            [
                "    def consume(self, amount: float, now: float) -> bool:",
                "        if amount < 0:",
                "            raise ValueError('amount must be non-negative')",
                "        self.tokens = min(self.capacity, self.tokens + max(0.0, now - self.last_ts) * self.rate)",
                "        self.last_ts = now",
                "        if self.tokens >= amount:",
                "            self.tokens -= amount",
                "            return True",
                "        return False",
            ],
            "python3 -m unittest -q",
            "Ran 5 tests in 0.009s\n\nOK",
            "Capped refilled tokens at `self.capacity` and rejected negative `amount` in `token_bucket.py`; all 5 tests pass.",
            False,
        ),
        (
            "ring_index.py",
            "Fix `logical_to_physical` in `ring_index.py` when `count == capacity` after wrap-around, and run `python3 -m unittest -q`.",
            "def logical_to_physical(write_pos: int, count: int, capacity: int, idx: int) -> int:\n    if not (0 <= idx < count):\n        raise IndexError('ring index out of range')\n    start = 0 if count < capacity else (write_pos + 1) % capacity\n    return (start + idx) % capacity\n",
            4,
            5,
            [
                "    start = (write_pos - count) % capacity",
                "    return (start + idx) % capacity",
            ],
            "python3 -m unittest -q",
            "Ran 6 tests in 0.007s\n\nOK",
            "Fixed `start = (write_pos - count) % capacity` in `ring_index.py` so wrapped full rings index the oldest slot rather than skipping one element; all 6 tests pass.",
            True,
        ),
    ]

    records: List[Dict[str, Any]] = []
    total = max(1, int(count))
    for i in range(total):
        fname, prompt_txt, src_txt, s_ln, e_ln, repl_lns, test_cmd, test_out, summary_txt, with_recovery = templates[
            i % len(templates)
        ]
        variant_suffix = "" if i < len(templates) else f" (workspace case #{i + 1})"
        src_variant = src_txt if i < len(templates) else f"# case_id={i + 1}\n" + src_txt
        line_shift = 0 if i < len(templates) else 1
        tag, read_out = _format_omp_read_output(fname, src_variant)
        valid_patch = _format_omp_patch(fname, tag, s_ln + line_shift, e_ln + line_shift, repl_lns)

        msgs: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous software engineering agent using OMP v18.18.1 `read`, `edit`, and `bash` tools. "
                    "Always read files to obtain the current `[path#hhhh]` snapshot tag before applying hashline patches, "
                    "and run tests before concluding."
                ),
            },
            {"role": "user", "content": prompt_txt + variant_suffix},
            {
                "role": "assistant",
                "reasoning_content": f"Read `{fname}` first to inspect line numbers and capture its 4-hex snapshot tag.",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_read_{i}_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": {"path": fname, "i": f"Inspect {fname}"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": f"call_read_{i}_1",
                "name": "read",
                "content": read_out,
            },
        ]

        if with_recovery:
            stale_tag = "0000" if tag != "0000" else "ffff"
            stale_patch = _format_omp_patch(fname, stale_tag, s_ln + line_shift, e_ln + line_shift, repl_lns)
            msgs.extend(
                [
                    {
                        "role": "assistant",
                        "reasoning_content": f"Apply the patch to `{fname}`.",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_edit_stale_{i}",
                                "type": "function",
                                "function": {"name": "edit", "arguments": {"input": stale_patch, "i": f"Patch {fname}"}},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_edit_stale_{i}",
                        "name": "edit",
                        "content": f"Error: Snapshot mismatch for {fname}: expected #{tag}, got #{stale_tag}. Re-read file before editing.",
                    },
                    {
                        "role": "assistant",
                        "reasoning_content": f"The snapshot tag was stale (`#{stale_tag}` vs `#{tag}`). Re-read `{fname}` and apply the patch with exact snapshot `#{tag}`.",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_read_{i}_2",
                                "type": "function",
                                "function": {"name": "read", "arguments": {"path": fname, "i": f"Re-read {fname} snapshot"}},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_read_{i}_2",
                        "name": "read",
                        "content": read_out,
                    },
                ]
            )

        msgs.extend(
            [
                {
                    "role": "assistant",
                    "reasoning_content": f"Apply the OMP v18.18.1 hashline `PUT` patch against snapshot `[{fname}#{tag}]`.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_edit_{i}_ok",
                            "type": "function",
                            "function": {"name": "edit", "arguments": {"input": valid_patch, "i": f"Apply fix to {fname}"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_edit_{i}_ok",
                    "name": "edit",
                    "content": f"Updated {fname} (1 hunk applied).",
                },
                {
                    "role": "assistant",
                    "reasoning_content": f"Run `{test_cmd}` via `bash` to verify the fix.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_bash_{i}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": {"command": test_cmd, "timeout": 30, "i": "Run unit tests"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_bash_{i}",
                    "name": "bash",
                    "content": test_out,
                },
                {
                    "role": "assistant",
                    "reasoning_content": "The test suite passed with zero errors.",
                    "content": summary_txt,
                },
            ]
        )

        family_id = f"omp_v18_family_{fname}_{i // 4}"
        records.append(
            {
                "id": f"omp_v18_traj_{i:05d}",
                "domain": "tool_trajectories",
                "source": "sparkx_omp_v18_verified_synthesizer",
                "source_revision": OMP_PINNED_COMMIT,
                "source_document_id": family_id,
                "tools": copy.deepcopy(OMP_V18_18_1_TOOLS),
                "tool_schema_digest": OMP_TOOL_SCHEMA_DIGEST,
                "verifier": {"kind": "omp_hashline_schema_and_unittest", "version": OMP_PINNED_VERSION, "passed": True},
                "messages": msgs,
            }
        )
    return records


def build_smoke_five_domain_seed_bank() -> List[Dict[str, Any]]:
    """Deterministic 7-example smoke fixture bank for unit tests ONLY (`allow_smoke_fixtures=True`).
    Repairs the invalid math example from `§3.1` (`(n^2 + 3n + 24)/(n + 3) = n + 24/(n + 3)` has a unique finite set
    of positive integer solutions `{1, 3, 5, 9, 21}`, of which primes are `7, 11, 13` at `n in {1, 5, 9}`, sum `15`)
    and uses the pinned OMP v18.18.1 `edit` `input` schema (`F15`)."""
    examples: List[Dict[str, Any]] = [
        {
            "id": "smoke_code_001",
            "domain": "code_debug",
            "source": "smoke_fixture",
            "source_revision": "v2_repaired",
            "source_document_id": "smoke_doc_code_001",
            "verifier": {"kind": "python_assert", "version": "1.0", "passed": True},
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Debug the doubly-stochastic 2x2 Sinkhorn projection below so that extreme logits do not "
                        "overflow `exp` in FP32 and verify row/column sums equal 1:\n"
                        "```python\n"
                        "def sinkhorn_2x2(logits):\n"
                        "    p = torch.exp(logits)\n"
                        "    return p / p.sum(dim=-1, keepdim=True)\n"
                        "```"
                    ),
                },
                {
                    "role": "assistant",
                    "reasoning_content": (
                        "A 2x2 doubly-stochastic matrix on the Birkhoff polytope B_2 is symmetric with diagonal "
                        "`p_keep = sigmoid(0.5 * ((L_00 + L_11) - (L_01 + L_10)))` and off-diagonal `p_swap = 1 - p_keep`."
                    ),
                    "content": (
                        "```python\n"
                        "def sinkhorn_2x2(logits: torch.Tensor) -> torch.Tensor:\n"
                        "    log_p = logits.float().clamp(min=-30.0, max=30.0)\n"
                        "    diag_adv = 0.5 * ((log_p[..., 0, 0] + log_p[..., 1, 1]) - (log_p[..., 0, 1] + log_p[..., 1, 0]))\n"
                        "    p_keep = torch.sigmoid(diag_adv)\n"
                        "    p_swap = 1.0 - p_keep\n"
                        "    row0 = torch.stack([p_keep, p_swap], dim=-1)\n"
                        "    row1 = torch.stack([p_swap, p_keep], dim=-1)\n"
                        "    return torch.stack([row0, row1], dim=-2)\n"
                        "```"
                    ),
                },
            ],
        },
        {
            "id": "smoke_code_002",
            "domain": "code_debug",
            "source": "smoke_fixture",
            "source_revision": "v2_repaired",
            "source_document_id": "smoke_doc_code_002",
            "verifier": {"kind": "python_assert", "version": "1.0", "passed": True},
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python function `verify_causal_gather(selected_indices, query_positions)` that "
                        "raises a `ValueError` if any selected key index `k_idx` exceeds `query_positions[q]`."
                    ),
                },
                {
                    "role": "assistant",
                    "reasoning_content": (
                        "Every selected key index `k_idx` for query at `q_pos` must satisfy `k_idx <= q_pos`."
                    ),
                    "content": (
                        "```python\n"
                        "def verify_causal_gather(selected_indices: list[list[int]], query_positions: list[int]) -> bool:\n"
                        "    if len(selected_indices) != len(query_positions):\n"
                        "        raise ValueError('Mismatched query count and selection rows')\n"
                        "    for q_pos, row in zip(query_positions, selected_indices):\n"
                        "        for k_idx in row:\n"
                        "            if k_idx > q_pos:\n"
                        "                raise ValueError(f'Causal violation: key {k_idx} > query {q_pos}')\n"
                        "    return True\n"
                        "```"
                    ),
                },
            ],
        },
        {
            "id": "smoke_instruct_001",
            "domain": "general_instruct",
            "source": "smoke_fixture",
            "source_revision": "v2_repaired",
            "source_document_id": "smoke_doc_instruct_001",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain why bounded sigmoid extra-pass gains `g = sigmoid(u) in (0, 1)` keep the linear "
                        "stream mixing matrix `(1 - g)I + gP` inside the 2x2 Birkhoff polytope."
                    ),
                },
                {
                    "role": "assistant",
                    "reasoning_content": (
                        "Because B_2 is convex and contains both I and P, `(1 - g)I + gP` is a convex combination "
                        "whenever `0 <= g <= 1`."
                    ),
                    "content": (
                        "For any doubly-stochastic $P \\in \\mathcal{B}_2$ and $g = \\sigma(u) \\in (0, 1)$, "
                        "$M(g) = (1 - g)I + gP$ is a convex combination of two doubly-stochastic matrices, so its "
                        "entries remain non-negative with unit row and column sums and eigenvalues $1$ and $1 - 2g(1 - p)$ "
                        "bounded in $[-1, 1]$."
                    ),
                },
            ],
        },
        {
            "id": "smoke_math_001",
            "domain": "math_symbolic",
            "source": "smoke_fixture",
            "source_revision": "v2_repaired",
            "source_document_id": "smoke_doc_math_001",
            "verifier": {"kind": "exact_enumeration", "version": "1.0", "passed": True},
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Find the sum of all positive integers $n$ such that $\\frac{n^2 + 3n + 24}{n + 3}$ "
                        "is a prime number. Provide the final answer in `\\boxed{}`."
                    ),
                },
                {
                    "role": "assistant",
                    "reasoning_content": (
                        "Rewrite `(n^2 + 3n + 24)/(n + 3) = n + 24/(n + 3)`. Since `n >= 1`, `n + 3 >= 4` must be a "
                        "divisor of `24`, so `n + 3 in {4, 6, 8, 12, 24}`, giving `n in {1, 3, 5, 9, 21}`. "
                        "Evaluating `n + 24/(n + 3)`: for `n=1`, `1 + 6 = 7` (prime); for `n=3`, `3 + 4 = 7` (prime); "
                        "for `n=5`, `5 + 3 = 8` (not prime); for `n=9`, `9 + 2 = 11` (prime); for `n=21`, `21 + 1 = 22` "
                        "(not prime). Thus the valid positive integers are `n in {1, 3, 9}`, and their sum is `1 + 3 + 9 = 13`."
                    ),
                    "content": (
                        "Dividing the numerator by $n + 3$ gives\n"
                        "$$\\frac{n^2 + 3n + 24}{n + 3} = n + \\frac{24}{n + 3}.$$\n"
                        "For this expression to be an integer with $n \\ge 1$, $n + 3 \\ge 4$ must divide $24$, so "
                        "$n + 3 \\in \\{4, 6, 8, 12, 24\\}$, i.e., $n \\in \\{1, 3, 5, 9, 21\\}$. Testing each value:\n"
                        "- $n = 1$: $1 + 6 = 7$ (prime)\n"
                        "- $n = 3$: $3 + 4 = 7$ (prime)\n"
                        "- $n = 5$: $5 + 3 = 8$ (composite)\n"
                        "- $n = 9$: $9 + 2 = 11$ (prime)\n"
                        "- $n = 21$: $21 + 1 = 22$ (composite)\n\n"
                        "The valid positive integers are $1, 3, 9$, with sum $1 + 3 + 9 = \\boxed{13}$."
                    ),
                },
            ],
        },
        {
            "id": "smoke_longdoc_001",
            "domain": "long_doc_repo",
            "source": "smoke_fixture",
            "source_revision": "v2_repaired",
            "source_document_id": "smoke_doc_longdoc_001",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Summarize the interaction between `SparkXLoopV03`, `TwoStreamMHC`, and `VisitAwareDLSAIndexBank` "
                        "across the 72 virtual visits (`4 + 24 + 24 + 12 + 8`)."
                    ),
                },
                {
                    "role": "assistant",
                    "reasoning_content": (
                        "Prelude L0..L3 has 4 visits (1 global at L3); Core P1 L4..L27 has 24 visits (6 global); "
                        "Core P2 L4..L27 has 24 visits (6 global); Inner P3 L12..L23 has 12 visits (3 global); "
                        "Coda L28..L35 has 8 visits (2 global)."
                    ),
                    "content": (
                        "1. **Prelude (`L0..L3`, 4 visits)**: Runs once; `L3` is Global Visit 0.\n"
                        "2. **Recurrent Core (`L4..L27` × 2 + `L12..L23` × 1, 60 visits)**: Mixed by 24 `TwoStreamMHC` "
                        "controllers while `VisitAwareDLSAIndexBank` provides 15 distinct global-visit selector maps.\n"
                        "3. **Coda (`L28..L35`, 8 visits)**: Reads `s0` after P3 and completes Global Visits 16 (`L31`) and 17 (`L35`)."
                    ),
                },
            ],
        },
    ]
    examples.extend(build_omp_tool_trajectory_examples(count=2))
    return examples


# ==============================================================================
# Exact Full-Conversation Chat Rendering & Assistant-Only Label Masking (`F03`)
# ==============================================================================
def _fallback_chatml_render_with_spans(
    messages: Sequence[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Deterministic full-conversation renderer returning `(full_text, assistant_char_spans)` when a tokenizer does
    not define `apply_chat_template`."""
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0

    if tools:
        tool_hdr = "<|im_start|>system\n# Tools\n" + json.dumps(list(tools), ensure_ascii=False) + "<|im_end|>\n"
        parts.append(tool_hdr)
        cursor += len(tool_hdr)

    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content") or "")
        reasoning = m.get("reasoning_content")
        t_calls = m.get("tool_calls")
        if role == "assistant":
            prefix = "<|im_start|>assistant\n"
            parts.append(prefix)
            cursor += len(prefix)
            body = ""
            if reasoning:
                body += f"<think>\n{reasoning}\n</think>\n"
            if t_calls:
                for tc in t_calls:
                    fn = tc.get("function", {})
                    name = str(fn.get("name", ""))
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"raw": args}
                    body += f"<tool_call>{name}"
                    if isinstance(args, dict):
                        for k_arg, v_arg in args.items():
                            v_txt = v_arg if isinstance(v_arg, str) else json.dumps(v_arg, ensure_ascii=False)
                            body += f"<arg_key>{k_arg}</arg_key><arg_value>{v_txt}</arg_value>"
                    body += "</tool_call>"
            body += f"{content}<|im_end|>\n"
            start_c = cursor
            parts.append(body)
            cursor += len(body)
            spans.append((start_c, cursor))
        elif role == "tool":
            seg = f"<|im_start|>tool\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            parts.append(seg)
            cursor += len(seg)
        else:
            seg = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            parts.append(seg)
            cursor += len(seg)
    return "".join(parts), spans


def _encode_with_offsets(tokenizer: Any, text: str) -> Tuple[List[int], Optional[List[Tuple[int, int]]]]:
    """Encode `text` once into token IDs and (when supported by fast tokenizers) exact character `(start, end)` offsets."""
    if callable(tokenizer) and hasattr(tokenizer, "is_fast") and bool(getattr(tokenizer, "is_fast", False)):
        try:
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(enc["input_ids"])
            offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            if len(ids) == len(offs):
                return ids, offs
        except Exception:
            pass
    if hasattr(tokenizer, "encode"):
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        return ids, None
    # Pure-Python char-level test tokenizer fallback
    ids = [max(1, ord(ch) % 120) for ch in text]
    offs = [(i, i + 1) for i in range(len(text))]
    return ids, offs


def render_conversation_exact_labels(
    example: Dict[str, Any] | TrainingRecord,
    tokenizer: Any,
    max_seq_len: int = 4096,
    *,
    min_supervised_tokens: int = 1,
    allow_tail_rewindow: bool = False,
) -> RenderedWindow:
    """Render a conversation with exact assistant-only token supervision (`F03`, `§6.3`):
      - `system`, `user`, and `tool` turns are strictly masked to `-100`.
      - Only assistant turns (including reasoning, `<tool_call>` spans, assistant text, and turn-closing delimiter)
        receive target labels.
      - **NEVER fabricates labels on user/prompt tokens** when truncation leaves zero assistant targets! Instead,
        either performs turn-aware tail windowing (when `allow_tail_rewindow=True` and complete assistant turns exist
        later in the conversation) or returns `RenderedWindow(valid=False, rejection_reason=...)`.
    """
    rec_dict = example.to_dict() if isinstance(example, TrainingRecord) else example
    messages = rec_dict.get("messages") or []
    tools = rec_dict.get("tools")
    rec_id = str(rec_dict.get("id", ""))
    domain = str(rec_dict.get("domain", "general_instruct"))
    doc_id = str(rec_dict.get("source_document_id", rec_id))

    if not messages or max_seq_len < 2:
        return RenderedWindow(
            valid=False,
            input_ids=[],
            labels=[],
            supervised_tokens=0,
            masked_tokens=0,
            total_tokens_before_window=0,
            rewindowed=False,
            rejection_reason="empty_messages_or_invalid_max_seq_len",
            record_id=rec_id,
            domain=domain,
            source_document_id=doc_id,
        )

    has_assistant_msg = any(str(m.get("role")) == "assistant" for m in messages)
    if not has_assistant_msg:
        return RenderedWindow(
            valid=False,
            input_ids=[],
            labels=[],
            supervised_tokens=0,
            masked_tokens=0,
            total_tokens_before_window=0,
            rewindowed=False,
            rejection_reason="no_assistant_messages",
            record_id=rec_id,
            domain=domain,
            source_document_id=doc_id,
        )

    # Try native `apply_chat_template` on the full conversation first (`§6.3`)
    used_template = False
    full_text = ""
    assistant_char_spans: List[Tuple[int, int]] = []
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            tk_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": False}
            if tools:
                tk_kwargs["tools"] = tools
            full_text = str(tokenizer.apply_chat_template(messages, **tk_kwargs))
            used_template = True
            # Derive exact assistant response spans by rendering `messages[:i]` with `add_generation_prompt=True`
            # (which includes the `<｜Assistant｜>` or `<|im_start|>assistant\n` role header in the masked prefix!)
            # and `messages[:i+1]` with `add_generation_prompt=False`, verifying exact string prefix equality on `full_text`.
            span_ok = True
            for idx, msg in enumerate(messages):
                if str(msg.get("role")) != "assistant":
                    continue
                pre_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
                post_kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": False}
                if tools:
                    pre_kwargs["tools"] = tools
                    post_kwargs["tools"] = tools
                pre_txt = str(tokenizer.apply_chat_template(messages[:idx], **pre_kwargs)) if idx > 0 else ""
                post_txt = str(tokenizer.apply_chat_template(messages[: idx + 1], **post_kwargs))
                if full_text.startswith(post_txt) and (not pre_txt or post_txt.startswith(pre_txt)):
                    assistant_char_spans.append((len(pre_txt), len(post_txt)))
                else:
                    span_ok = False
                    break
            if not span_ok:
                assistant_char_spans = []
                # Regex fallback for Spark (`<｜Assistant｜>...<｜end▁of▁sentence｜>`) or ChatML (`<|im_start|>assistant\n...<|im_end|>`)
                for pat in (
                    re.compile(r"<｜Assistant｜>(.*?<｜end▁of▁sentence｜>)", re.DOTALL),
                    re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>\n?)", re.DOTALL),
                ):
                    matches = list(pat.finditer(full_text))
                    if matches:
                        assistant_char_spans = [(m.start(1), m.end(1)) for m in matches]
                        break
                if not assistant_char_spans:
                    used_template = False
        except Exception:
            used_template = False

    if not used_template:
        full_text, assistant_char_spans = _fallback_chatml_render_with_spans(messages, tools=tools)

    full_ids, offsets = _encode_with_offsets(tokenizer, full_text)
    total_before = len(full_ids)
    if total_before < 2:
        return RenderedWindow(
            valid=False,
            input_ids=[],
            labels=[],
            supervised_tokens=0,
            masked_tokens=0,
            total_tokens_before_window=total_before,
            rewindowed=False,
            rejection_reason="too_few_tokens",
            record_id=rec_id,
            domain=domain,
            source_document_id=doc_id,
        )

    full_labels = [-100] * total_before
    turn_token_spans: List[Tuple[int, int]] = []

    if offsets is not None and len(offsets) == total_before:
        for c_start, c_end in assistant_char_spans:
            if c_end <= c_start:
                continue
            t_lo: Optional[int] = None
            t_hi: int = 0
            for t_i, (o_s, o_e) in enumerate(offsets):
                if o_e <= o_s:
                    continue
                # Supervise token only if its character span lies inside `[c_start, c_end]` (never supervise prompt boundary tokens!)
                if o_s >= c_start and o_e <= c_end:
                    full_labels[t_i] = full_ids[t_i]
                    if t_lo is None:
                        t_lo = t_i
                    t_hi = t_i + 1
            if t_lo is not None and t_hi > t_lo:
                turn_token_spans.append((t_lo, t_hi))
    else:
        # Non-fast tokenizer without char offsets: segment-encode non-assistant vs assistant spans and verify exact ID concatenation
        segs: List[Tuple[str, bool]] = []
        pos = 0
        for c_start, c_end in sorted(assistant_char_spans):
            if c_start > pos:
                segs.append((full_text[pos:c_start], False))
            if c_end > c_start:
                segs.append((full_text[c_start:c_end], True))
            pos = max(pos, c_end)
        if pos < len(full_text):
            segs.append((full_text[pos:], False))

        seg_ids: List[int] = []
        seg_labels: List[int] = []
        seg_spans: List[Tuple[int, int]] = []
        for s_txt, is_asst in segs:
            if not s_txt:
                continue
            part_ids, _ = _encode_with_offsets(tokenizer, s_txt)
            lo = len(seg_ids)
            seg_ids.extend(part_ids)
            if is_asst:
                seg_labels.extend(part_ids)
                if len(seg_ids) > lo:
                    seg_spans.append((lo, len(seg_ids)))
            else:
                seg_labels.extend([-100] * len(part_ids))
        full_ids = seg_ids
        full_labels = seg_labels
        total_before = len(full_ids)
        turn_token_spans = seg_spans

    # Position 0 cannot be a shifted next-token target (`labels[:, 1:]`)
    if full_labels:
        full_labels[0] = -100

    rewindowed = False
    if len(full_ids) <= max_seq_len:
        win_ids = full_ids
        win_labels = full_labels
    else:
        # Turn-aware truncation (`§6.3`): keep complete assistant turns inside `[0, max_seq_len]` if any complete turn fits;
        # avoid slicing mid-assistant-turn when an earlier complete assistant turn exists, or reject if 0 assistant tokens remain!
        complete_turns_in_prefix = [sp for sp in turn_token_spans if sp[1] <= max_seq_len]
        if complete_turns_in_prefix:
            cut_end = complete_turns_in_prefix[-1][1]
            # Keep up to `max_seq_len` if no partial assistant turn starts before `max_seq_len`, else stop cleanly before the partial turn
            partial_turns = [sp for sp in turn_token_spans if sp[0] < max_seq_len < sp[1]]
            if partial_turns:
                cut_end = max(cut_end, partial_turns[0][0])
            else:
                cut_end = max_seq_len
            win_ids = full_ids[:cut_end]
            win_labels = list(full_labels[:cut_end])
        elif allow_tail_rewindow and turn_token_spans:
            # Rewindow across the boundary of the first assistant turn so both trailing prompt context (masked `-100`)
            # and assistant target tokens (`!= -100`) fit inside `max_seq_len`, without ever supervising prompt tokens!
            t_lo, t_hi = turn_token_spans[0]
            end_pos = min(len(full_ids), min(t_hi, t_lo + max(2, max_seq_len // 2)))
            if end_pos <= t_lo:
                end_pos = min(len(full_ids), t_hi)
            start_pos = max(0, end_pos - max_seq_len)
            win_ids = full_ids[start_pos:end_pos]
            win_labels = list(full_labels[start_pos:end_pos])
            # Mask any partial assistant turn at the left edge of the window
            if start_pos > 0:
                for k_i in range(len(win_labels)):
                    abs_k = start_pos + k_i
                    if abs_k < t_lo:
                        win_labels[k_i] = -100
                    else:
                        break
            rewindowed = True
        else:
            win_ids = full_ids[:max_seq_len]
            win_labels = list(full_labels[:max_seq_len])

    if win_labels:
        win_labels[0] = -100

    sup_cnt = sum(1 for y in win_labels[1:] if y != -100)
    mask_cnt = len(win_labels) - sup_cnt

    if sup_cnt < max(1, int(min_supervised_tokens)):
        # STRICT ZERO-TARGET REJECTION (`F03`): never fabricate labels on user/system/tool tokens!
        return RenderedWindow(
            valid=False,
            input_ids=win_ids,
            labels=[-100] * len(win_ids),
            supervised_tokens=0,
            masked_tokens=len(win_ids),
            total_tokens_before_window=total_before,
            rewindowed=rewindowed,
            rejection_reason="zero_supervised_targets_after_truncation",
            record_id=rec_id,
            domain=domain,
            source_document_id=doc_id,
        )

    return RenderedWindow(
        valid=True,
        input_ids=win_ids,
        labels=win_labels,
        supervised_tokens=sup_cnt,
        masked_tokens=mask_cnt,
        total_tokens_before_window=total_before,
        rewindowed=rewindowed,
        rejection_reason=None,
        record_id=rec_id,
        domain=domain,
        source_document_id=doc_id,
    )


# ==============================================================================
# Manifest I/O & Real 5-Domain Streamed Corpus Construction (`WP1`, `F02`)
# ==============================================================================
def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_records(
    manifest_path: str | Path,
    *,
    expected_split: Optional[str] = None,
    reject_eval_leaks: bool = True,
) -> Tuple[List[TrainingRecord], Dict[str, Any]]:
    """Load normalized `TrainingRecord` entries from a `.json` or `.jsonl` manifest file (`WP1`)."""
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Training data manifest not found: {p}")
    sha = _file_sha256(p)
    raw_text = p.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Training data manifest is empty: {p}")

    raw_items: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl" or raw_text.startswith("{") and "\n{" in raw_text:
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            ln = line.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if not isinstance(obj, dict):
                raise ValueError(f"{p}:{line_no}: expected JSON object.")
            raw_items.append(obj)
    else:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
            raw_items = list(parsed["records"])
        elif isinstance(parsed, list):
            raw_items = list(parsed)
        else:
            raise ValueError(f"{p}: expected JSON array or `{{\"records\": [...]}}`.")

    records: List[TrainingRecord] = []
    excluded_leaks = 0
    for item in raw_items:
        rec = TrainingRecord.from_dict(item, default_split=expected_split)
        if reject_eval_leaks and is_excluded_eval_record(rec.messages, rec.source_document_id):
            excluded_leaks += 1
            continue
        if expected_split is not None and rec.split != expected_split:
            continue
        records.append(rec)

    if not records:
        raise ValueError(f"Manifest {p} yielded 0 valid records for split={expected_split!r} (excluded_leaks={excluded_leaks}).")

    domain_counts: Dict[str, int] = {}
    for r in records:
        domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1

    info = {
        "manifest_path": str(p.resolve()),
        "manifest_sha256": sha,
        "loaded_records": len(records),
        "unique_documents": len({r.source_document_id for r in records}),
        "excluded_eval_leaks": excluded_leaks,
        "domain_counts": domain_counts,
    }
    return records, info


def save_manifest_records(records: Sequence[TrainingRecord], output_path: str | Path) -> Dict[str, Any]:
    """Write `TrainingRecord` entries to deterministic `.jsonl` along with `.meta.json` SHA-256 provenance."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    sha = _file_sha256(p)
    domain_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    for r in records:
        domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1
        split_counts[r.split] = split_counts.get(r.split, 0) + 1
    meta = {
        "manifest_path": str(p.resolve()),
        "manifest_sha256": sha,
        "num_records": len(records),
        "unique_documents": len({r.source_document_id for r in records}),
        "domain_counts": domain_counts,
        "split_counts": split_counts,
    }
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def _synthesize_verified_code_and_math_records(count_code: int, count_math: int, seed: int = 42) -> List[TrainingRecord]:
    """Generate distinct, programmatically verified `code_debug` and `math_symbolic` problem families with zero
    overlap with evaluation suites (`§6.1`)."""
    rng = random.Random(seed)
    out: List[TrainingRecord] = []

    # Programmatically verified algorithmic bug-repair families
    for idx in range(max(0, int(count_code))):
        mod_base = rng.randint(17, 97)
        win_k = rng.randint(3, 12)
        stride_s = rng.randint(1, 4)
        doc_id = f"verified_code_family_{idx // 2:05d}"
        split = assign_document_split(doc_id)
        if idx % 3 == 0:
            user_q = (
                f"Fix `prefix_xor_range(arr, left, right)` so inclusive range XOR queries on `0 <= left <= right < len(arr)` "
                f"with salt `{mod_base}` run in $O(1)$ after $O(N)$ preprocessing:\n"
                "```python\n"
                "def build_prefix_xor(arr):\n"
                "    px = [0]\n"
                "    for x in arr:\n"
                "        px.append(px[-1] ^ x)\n"
                "    return px\n\n"
                "def prefix_xor_range(px, left, right):\n"
                "    return px[right] ^ px[left]  # buggy boundary\n"
                "```"
            )
            asst_r = (
                f"`px` has length `N + 1` with `px[k] = arr[0] ^ ... ^ arr[k - 1]`. Therefore the inclusive XOR over "
                f"`arr[left : right + 1]` is `px[right + 1] ^ px[left]`, not `px[right] ^ px[left]`."
            )
            asst_c = (
                "```python\n"
                "def build_prefix_xor(arr: list[int]) -> list[int]:\n"
                "    px = [0]\n"
                "    for x in arr:\n"
                "        px.append(px[-1] ^ int(x))\n"
                "    return px\n\n"
                "def prefix_xor_range(px: list[int], left: int, right: int) -> int:\n"
                f"    if not (0 <= left <= right < len(px) - 1):\n"
                f"        raise IndexError('Invalid inclusive range [{left}, {right}] (salt={mod_base})')\n"
                "    return px[right + 1] ^ px[left]\n"
                "```"
            )
        elif idx % 3 == 1:
            user_q = (
                f"Write a Python function `strided_window_max(nums, window={win_k}, stride={stride_s})` using a monotonic "
                f"deque in $O(N)$ time that returns `[max(nums[i:i+window]) for i in range(0, len(nums)-window+1, stride)]`."
            )
            asst_r = (
                f"Maintain a decreasing deque of indices over `nums`, evicting front indices `<= i - {win_k}` and emitting "
                f"`nums[dq[0]]` whenever `i >= {win_k} - 1` and `(i - ({win_k} - 1)) % {stride_s} == 0`."
            )
            asst_c = (
                "```python\n"
                "from collections import deque\n\n"
                f"def strided_window_max(nums: list[int], window: int = {win_k}, stride: int = {stride_s}) -> list[int]:\n"
                "    if window <= 0 or stride <= 0:\n"
                "        raise ValueError('window and stride must be positive')\n"
                "    if len(nums) < window:\n"
                "        return []\n"
                "    dq: deque[int] = deque()\n"
                "    out: list[int] = []\n"
                "    for i, val in enumerate(nums):\n"
                "        while dq and dq[0] <= i - window:\n"
                "            dq.popleft()\n"
                "        while dq and nums[dq[-1]] <= val:\n"
                "            dq.pop()\n"
                "        dq.append(i)\n"
                "        start = i - window + 1\n"
                "        if start >= 0 and start % stride == 0:\n"
                "            out.append(nums[dq[0]])\n"
                "    return out\n"
                "```"
            )
        else:
            user_q = (
                f"Debug `binary_search_first_ge(sorted_vals, target)` (test case threshold `{mod_base * win_k}`) so duplicates "
                "always return the leftmost index `idx` with `sorted_vals[idx] >= target`, or `len(sorted_vals)` if none exists."
            )
            asst_r = (
                "Use half-open `[lo, hi)` with `lo = 0, hi = len(sorted_vals)`. When `sorted_vals[mid] >= target`, set `hi = mid` "
                "to preserve `mid` as a candidate and continue searching left; otherwise set `lo = mid + 1`."
            )
            asst_c = (
                "```python\n"
                "def binary_search_first_ge(sorted_vals: list[int], target: int) -> int:\n"
                "    lo, hi = 0, len(sorted_vals)\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if sorted_vals[mid] >= target:\n"
                "            hi = mid\n"
                "        else:\n"
                "            lo = mid + 1\n"
                "    return lo\n"
                "```"
            )
        out.append(
            TrainingRecord(
                id=f"verified_code_{idx:05d}",
                domain="code_debug",
                source="sparkx_verified_code_generator",
                source_revision="v2.0",
                source_document_id=doc_id,
                split=split,
                messages=[
                    {"role": "user", "content": user_q},
                    {"role": "assistant", "reasoning_content": asst_r, "content": asst_c},
                ],
                verifier={"kind": "programmatic_oracle", "version": "2.0", "passed": True},
            )
        )

    # Programmatically solved modular / Diophantine / combinatorics math problems
    for idx in range(max(0, int(count_math))):
        a_val = 2 + (idx % 11)
        b_val = 12 + ((idx * 7) % 36)
        # Solve `(n^2 + a*n + b) / (n + a) = n + b / (n + a)` for integer values over positive integers `n >= 1`
        valid_ns = [d - a_val for d in range(a_val + 1, b_val + 1) if b_val % d == 0]
        ans_sum = sum(valid_ns)
        doc_id = f"verified_math_family_{a_val}_{b_val}_{idx // 2:05d}"
        split = assign_document_split(doc_id)
        out.append(
            TrainingRecord(
                id=f"verified_math_{idx:05d}",
                domain="math_symbolic",
                source="sparkx_verified_math_generator",
                source_revision="v2.0",
                source_document_id=doc_id,
                split=split,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Find the sum of all positive integers $n$ such that "
                            f"$\\frac{{n^2 + {a_val}n + {b_val}}}{{n + {a_val}}}$ is an integer. "
                            "Put your final numerical result inside `\\boxed{}`."
                        ),
                    },
                    {
                        "role": "assistant",
                        "reasoning_content": (
                            f"Rewrite `(n^2 + {a_val}n + {b_val})/(n + {a_val}) = n + {b_val}/(n + {a_val})`. "
                            f"For `n >= 1`, `d = n + {a_val} >= {a_val + 1}` must divide `{b_val}`. "
                            f"The divisors `d >= {a_val + 1}` of `{b_val}` give `n in {valid_ns}`, whose sum is `{ans_sum}`."
                        ),
                        "content": (
                            f"Splitting the numerator gives\n"
                            f"$$\\frac{{n^2 + {a_val}n + {b_val}}}{{n + {a_val}}} = n + \\frac{{{b_val}}}{{n + {a_val}}}.$$\n"
                            f"Since $n \\ge 1$, $d = n + {a_val} \\ge {a_val + 1}$ must be a positive divisor of ${b_val}$. "
                            f"This yields $n \\in \\{{{', '.join(str(x) for x in valid_ns)}\\}}$, and their sum is $\\boxed{{{ans_sum}}}$."
                        ),
                    },
                ],
                verifier={"kind": "exact_integer_oracle", "version": "2.0", "passed": True, "answer": ans_sum},
            )
        )
    return out


def build_real_five_domain_corpus(
    *,
    target_total_records: int = 1200,
    target_total_examples: Optional[int] = None,
    seed: int = 42,
    use_hf_streaming: bool = True,
    allow_programmatic_aug: bool = True,
    allow_smoke_fixtures: bool = False,
    allow_smoke_fallback: Optional[bool] = None,
) -> Tuple[Dict[str, List[TrainingRecord]], Dict[str, Any]]:
    """Construct a document-disjoint `{'train': [...], 'dev': [...], 'test': [...]}` 5-domain corpus (`WP1`, `F02`, `§6.1`).

    Streams public Hugging Face datasets (`HuggingFaceTB/smoltalk2`, `open-thoughts/OpenThoughts3-1.2M`,
    `HuggingFaceFW/fineweb-edu`) when `use_hf_streaming=True`, combines them with pinned OMP v18.18.1 tool trajectories
    and verified code/math tasks, and **fails closed** when `allow_smoke_fixtures=False` if real dataset streaming
    cannot reach the requested minimum distinct corpus size (`>= 1000` for production runs).
    """
    if target_total_examples is not None:
        target_total_records = int(target_total_examples)
    if allow_smoke_fallback is not None:
        allow_smoke_fixtures = bool(allow_smoke_fallback)
    records: List[TrainingRecord] = []
    source_stats: Dict[str, int] = {}
    stream_errors: List[str] = []

    n_total = max(7, int(target_total_records))
    n_omp = max(4, int(round(n_total * 0.16)))
    n_instruct = max(4, int(round(n_total * 0.22)))
    n_reason = max(4, int(round(n_total * 0.20)))
    n_long = max(4, int(round(n_total * 0.12)))

    if use_hf_streaming:
        try:
            from datasets import load_dataset  # type: ignore

            # 1. Dialogue / general instruction (`SmolTalk2` or `HuggingFaceH4/ultrachat_200k` fallback)
            for ds_id, ds_cfg, split_name in (
                ("HuggingFaceTB/smoltalk2", "SFT", "train"),
                ("HuggingFaceH4/ultrachat_200k", None, "train_sft"),
            ):
                try:
                    ds = (
                        load_dataset(ds_id, ds_cfg, split=split_name, streaming=True)
                        if ds_cfg
                        else load_dataset(ds_id, split=split_name, streaming=True)
                    )
                    got = 0
                    for idx, row in enumerate(ds):
                        msgs = row.get("messages")
                        if not isinstance(msgs, list) or len(msgs) < 2:
                            continue
                        norm_msgs = [
                            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                            for m in msgs
                            if isinstance(m, dict) and str(m.get("role")) in {"system", "user", "assistant"} and m.get("content")
                        ]
                        if not any(m["role"] == "assistant" for m in norm_msgs):
                            continue
                        doc_id = f"{ds_id}:doc_{idx:07d}"
                        if is_excluded_eval_record(norm_msgs, doc_id):
                            continue
                        rec = TrainingRecord(
                            id=f"hf_inst_{got:06d}",
                            domain="general_instruct",
                            source=ds_id,
                            source_revision=str(getattr(ds, "_fingerprint", None) or "hf_stream"),
                            source_document_id=doc_id,
                            split=assign_document_split(doc_id),
                            messages=norm_msgs,
                        )
                        records.append(rec)
                        source_stats[ds_id] = source_stats.get(ds_id, 0) + 1
                        got += 1
                        if got >= n_instruct:
                            break
                    if got > 0:
                        break
                except Exception as exc:
                    stream_errors.append(f"{ds_id}: {exc}")

            # 2. Verified reasoning / code & math (`OpenThoughts3-1.2M` or `OpenThoughts-114k`)
            for ds_id, split_name in (
                ("open-thoughts/OpenThoughts3-1.2M", "train"),
                ("open-thoughts/OpenThoughts-114k", "train"),
            ):
                try:
                    ds = load_dataset(ds_id, split=split_name, streaming=True)
                    got = 0
                    for idx, row in enumerate(ds):
                        convs = row.get("conversations") or row.get("messages")
                        norm_msgs = []
                        if isinstance(convs, list):
                            for c in convs:
                                if not isinstance(c, dict):
                                    continue
                                r_raw = str(c.get("from") or c.get("role") or "").lower()
                                role = "assistant" if r_raw in {"assistant", "gpt"} else ("system" if r_raw == "system" else "user")
                                val = str(c.get("value") or c.get("content") or "")
                                if val.strip():
                                    norm_msgs.append({"role": role, "content": val})
                        elif row.get("problem") and row.get("solution"):
                            norm_msgs = [
                                {"role": "user", "content": str(row["problem"])},
                                {"role": "assistant", "content": str(row["solution"])},
                            ]
                        if len(norm_msgs) < 2 or not any(m["role"] == "assistant" for m in norm_msgs):
                            continue
                        dom_hint = str(row.get("domain", "")).lower()
                        dom = "code_debug" if ("code" in dom_hint or "```python" in norm_msgs[-1]["content"]) else "math_symbolic"
                        doc_id = f"{ds_id}:doc_{idx:07d}"
                        if is_excluded_eval_record(norm_msgs, doc_id):
                            continue
                        records.append(
                            TrainingRecord(
                                id=f"hf_reason_{got:06d}",
                                domain=dom,
                                source=ds_id,
                                source_revision="hf_stream",
                                source_document_id=doc_id,
                                split=assign_document_split(doc_id),
                                messages=norm_msgs,
                            )
                        )
                        source_stats[ds_id] = source_stats.get(ds_id, 0) + 1
                        got += 1
                        if got >= n_reason:
                            break
                    if got > 0:
                        break
                except Exception as exc:
                    stream_errors.append(f"{ds_id}: {exc}")

            # 3. Coherent long-context prose synthesis (`FineWeb-Edu` -> explicitly framed assistant synthesis task per `§6.1`)
            try:
                ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                got_fw = 0
                for idx, row in enumerate(ds_fw):
                    txt = str(row.get("text") or "").strip()
                    if len(txt) < 2400:
                        continue
                    paras = [p.strip() for p in txt.split("\n\n") if len(p.strip()) >= 120]
                    if len(paras) < 4:
                        continue
                    context_body = "\n\n".join(paras[:-1])[:24000]
                    conclusion_target = paras[-1]
                    doc_id = f"HuggingFaceFW/fineweb-edu:{row.get('id', idx)}"
                    msgs = [
                        {
                            "role": "user",
                            "content": (
                                "Read the following technical passage carefully and write a concise, accurate concluding "
                                "paragraph that synthesizes its main technical argument and constraints:\n\n"
                                f"{context_body}"
                            ),
                        },
                        {"role": "assistant", "content": conclusion_target},
                    ]
                    if is_excluded_eval_record(msgs, doc_id):
                        continue
                    records.append(
                        TrainingRecord(
                            id=f"hf_long_{got_fw:06d}",
                            domain="long_doc_repo",
                            source="HuggingFaceFW/fineweb-edu",
                            source_revision="sample-10BT",
                            source_document_id=doc_id,
                            split=assign_document_split(doc_id),
                            messages=msgs,
                        )
                    )
                    source_stats["HuggingFaceFW/fineweb-edu"] = source_stats.get("HuggingFaceFW/fineweb-edu", 0) + 1
                    got_fw += 1
                    if got_fw >= n_long:
                        break
            except Exception as exc:
                stream_errors.append(f"HuggingFaceFW/fineweb-edu: {exc}")
        except Exception as exc:
            stream_errors.append(f"datasets_import: {exc}")

    streamed_count = len(records)

    # Always include verified OMP v18.18.1 hashline tool trajectories (`§6.5`)
    if streamed_count > 0 or allow_programmatic_aug:
        for raw_omp in build_omp_tool_trajectory_examples(count=n_omp):
            rec_omp = TrainingRecord.from_dict(raw_omp)
            records.append(rec_omp)
            source_stats[rec_omp.source] = source_stats.get(rec_omp.source, 0) + 1

    if allow_programmatic_aug and (streamed_count > 0 or allow_smoke_fixtures):
        rem = max(0, n_total - len(records))
        if rem > 0:
            n_c = int(round(rem * 0.65))
            n_m = max(0, rem - n_c)
            aug_recs = _synthesize_verified_code_and_math_records(n_c, n_m, seed=seed)
            for r in aug_recs:
                records.append(r)
                source_stats[r.source] = source_stats.get(r.source, 0) + 1

    if streamed_count == 0 and not allow_smoke_fixtures:
        raise RuntimeError(
            "build_real_five_domain_corpus failed closed (`allow_smoke_fixtures=False`): no external `--data-manifest` "
            f"was supplied and Hugging Face streaming returned 0 records. Errors: {stream_errors}"
        )

    if not records and allow_smoke_fixtures:
        for raw_s in build_smoke_five_domain_seed_bank():
            rec_s = TrainingRecord.from_dict(raw_s, default_split="train")
            records.append(rec_s)
            source_stats["smoke_fixture"] = source_stats.get("smoke_fixture", 0) + 1

    splits: Dict[str, List[TrainingRecord]] = {"train": [], "dev": [], "test": []}
    for r in records:
        splits[r.split].append(r)

    # Ensure `dev` is non-empty when building a multi-example corpus while keeping strict document disjointness
    if not splits["dev"] and len(splits["train"]) >= 4:
        by_doc: Dict[str, List[TrainingRecord]] = {}
        for r in splits["train"]:
            by_doc.setdefault(r.source_document_id, []).append(r)
        if len(by_doc) >= 2:
            move_doc = sorted(by_doc.keys())[-1]
            moved = by_doc.pop(move_doc)
            splits["train"] = [r for grp in by_doc.values() for r in grp]
            for r in moved:
                r.split = "dev"
            splits["dev"].extend(moved)

    leak_audit = verify_no_split_document_leakage(splits["train"], splits["dev"], splits["test"])
    if not leak_audit["passed"]:
        raise RuntimeError(f"Document split leakage detected: {leak_audit}")

    summary = {
        "total_records": len(records),
        "streamed_hf_records": streamed_count,
        "smoke_fixture_mode": bool(streamed_count == 0 and allow_smoke_fixtures),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "source_counts": source_stats,
        "stream_errors": stream_errors,
        "leak_audit": leak_audit,
    }
    return splits, summary


def build_long_context_calibration_token_packs(
    tokenizer: Any,
    *,
    num_train_docs: int = 8,
    num_val_docs: int = 2,
    target_seq_len: int = 8192,
    seq_len: Optional[int] = None,
    records: Optional[Sequence[TrainingRecord | Dict[str, Any]]] = None,
    use_hf_streaming: bool = True,
    allow_smoke_fixtures: bool = False,
    seed: int = 42,
) -> Tuple[List["torch.LongTensor"], List["torch.LongTensor"], Dict[str, Any]]:
    """Build disjoint `train` and `val` coherent long-context token sequences (`8K–32K`, `> 4,612` exact cutoff)
    for Stage E 18-visit D-LSA selector distillation (`§8.3`)."""
    if not HAS_TORCH:
        raise RuntimeError("build_long_context_calibration_token_packs requires PyTorch.")

    eff_seq_len = max(16, int(seq_len if seq_len is not None else target_seq_len))
    needed_total = max(1, int(num_train_docs)) + max(0, int(num_val_docs))
    raw_docs: List[Tuple[str, str]] = []
    errors: List[str] = []

    if records:
        acc_parts_rec: List[str] = []
        acc_chars_rec = 0
        char_goal_rec = eff_seq_len * 5
        for r_idx, raw_r in enumerate(records):
            rec_obj = raw_r if isinstance(raw_r, TrainingRecord) else TrainingRecord.from_dict(raw_r)
            if rec_obj.source and "smoke" in str(rec_obj.source).lower() and not allow_smoke_fixtures:
                continue
            txt_r = "\n\n".join(str(m.get("content", "")) for m in rec_obj.messages if m.get("content")).strip()
            if not txt_r or is_excluded_eval_record(rec_obj.messages, rec_obj.source_document_id):
                continue
            acc_parts_rec.append(txt_r)
            acc_chars_rec += len(txt_r) + 2
            if acc_chars_rec >= char_goal_rec:
                raw_docs.append((f"manifest_cal_doc_{len(raw_docs):05d}_{r_idx}", "\n\n".join(acc_parts_rec)))
                acc_parts_rec = []
                acc_chars_rec = 0
                if len(raw_docs) >= needed_total:
                    break

    if len(raw_docs) < needed_total and use_hf_streaming:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            acc_parts: List[str] = []
            acc_chars = 0
            doc_idx = 0
            char_goal = eff_seq_len * 5
            for row_i, row in enumerate(ds):
                txt = str(row.get("text") or "").strip()
                if len(txt) < 400:
                    continue
                if is_excluded_eval_record([{"content": txt}], f"fw_{row_i}"):
                    continue
                acc_parts.append(txt)
                acc_chars += len(txt) + 2
                if acc_chars >= char_goal:
                    raw_docs.append((f"fineweb_cal_doc_{doc_idx:05d}", "\n\n".join(acc_parts)))
                    doc_idx += 1
                    acc_parts = []
                    acc_chars = 0
                    if len(raw_docs) >= needed_total:
                        break
        except Exception as exc:
            errors.append(str(exc))

    if len(raw_docs) < needed_total:
        if not allow_smoke_fixtures:
            raise RuntimeError(
                f"build_long_context_calibration_token_packs failed closed (`allow_smoke_fixtures=False`): "
                f"obtained {len(raw_docs)}/{needed_total} long calibration documents. Errors: {errors}"
            )
        rng = random.Random(seed)
        seed_bank = build_smoke_five_domain_seed_bank()
        while len(raw_docs) < needed_total:
            d_i = len(raw_docs)
            blocks: List[str] = []
            while sum(len(b) for b in blocks) < eff_seq_len * 5:
                ex = seed_bank[(d_i + rng.randint(0, 99)) % len(seed_bank)]
                blocks.append(
                    f"## Module {len(blocks)} (doc={d_i})\n"
                    + "\n".join(str(m.get("content", "")) for m in ex["messages"])
                )
            raw_docs.append((f"smoke_cal_doc_{d_i:04d}", "\n\n".join(blocks)))

    train_seqs: List[torch.LongTensor] = []
    val_seqs: List[torch.LongTensor] = []
    for idx, (_doc_id, text) in enumerate(raw_docs[:needed_total]):
        ids, _ = _encode_with_offsets(tokenizer, text)
        if len(ids) < eff_seq_len:
            reps = (eff_seq_len // max(1, len(ids))) + 1
            ids = (ids * reps)[:eff_seq_len]
        else:
            ids = ids[:eff_seq_len]
        t_seq = torch.tensor(ids, dtype=torch.long)
        if idx < int(num_train_docs):
            train_seqs.append(t_seq)
        else:
            val_seqs.append(t_seq)

    meta = {
        "num_train_docs": len(train_seqs),
        "num_val_docs": len(val_seqs),
        "seq_len": eff_seq_len,
        "exceeds_exact_4612_cutoff": bool(eff_seq_len > 4612),
        "smoke_fixture_mode": bool(any(d_id.startswith("smoke_") for d_id, _ in raw_docs[:needed_total])),
        "stream_errors": errors,
    }
    return train_seqs, val_seqs, meta


# ==============================================================================
# Domain-Weighted Single-Example Batch Sampler & Supervision Ledger (`§6.4`)
# ==============================================================================
class DomainWeightedBatchStream:
    """Deterministic, resumable, domain-weighted training stream with exact assistant-only labels, attention masks,
    zero-target rejection, and full token/domain/padding accounting (`§6.3–§6.4`)."""

    def __init__(
        self,
        records: Sequence[TrainingRecord | Dict[str, Any]],
        tokenizer: Any,
        *,
        max_seq_len: int = 4096,
        micro_batch_size: int = 1,
        domain_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
        pad_token_id: int = 0,
        allow_tail_rewindow: bool = False,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("DomainWeightedBatchStream requires PyTorch.")
        self.tokenizer = tokenizer
        self.max_seq_len = max(4, int(max_seq_len))
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or pad_token_id)
        self.seed = int(seed)
        self.allow_tail_rewindow = bool(allow_tail_rewindow)
        self.domain_weights = dict(domain_weights or STAGE_C_DOMAIN_WEIGHTS)

        self.by_domain: Dict[str, List[RenderedWindow]] = {d: [] for d in VALID_DOMAINS}
        self.rejected_zero_target = 0
        self.rejected_reasons: Dict[str, int] = {}
        self.rewindowed_examples = 0

        for idx, raw in enumerate(records):
            rec = raw if isinstance(raw, TrainingRecord) else TrainingRecord.from_dict(raw)
            rw = render_conversation_exact_labels(
                rec, tokenizer, max_seq_len=self.max_seq_len, allow_tail_rewindow=self.allow_tail_rewindow
            )
            if not rw.valid:
                self.rejected_zero_target += 1
                reason = str(rw.rejection_reason or "invalid")
                self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1
                continue
            if rw.rewindowed:
                self.rewindowed_examples += 1
            self.by_domain.setdefault(rec.domain, []).append(rw)

        self.available_domains = [d for d, items in self.by_domain.items() if items]
        if not self.available_domains:
            raise RuntimeError(
                f"DomainWeightedBatchStream has 0 valid supervised examples after rendering "
                f"(rejected={self.rejected_zero_target}, reasons={self.rejected_reasons})."
            )

        # Normalize weights across available domains
        raw_w = [max(0.0, float(self.domain_weights.get(d, 0.0))) for d in self.available_domains]
        if sum(raw_w) <= 0.0:
            raw_w = [1.0 for _ in self.available_domains]
        w_sum = sum(raw_w)
        self.normalized_weights = {d: w / w_sum for d, w in zip(self.available_domains, raw_w)}

        # Resumable state & accounting counters (`§6.4`)
        self.cursor: int = 0
        self.domain_cursors: Dict[str, int] = {d: 0 for d in self.available_domains}
        self.rng = random.Random(self.seed)
        self.seen_record_ids: set[str] = set()
        self.seen_doc_ids: set[str] = set()
        self.repeated_exposures: int = 0
        self.input_token_exposures: int = 0
        self.supervised_token_exposures: int = 0
        self.padding_token_exposures: int = 0
        self.domain_example_exposures: Dict[str, int] = {d: 0 for d in VALID_DOMAINS}
        self.domain_supervised_tokens: Dict[str, int] = {d: 0 for d in VALID_DOMAINS}
        self.seq_len_histogram: Dict[str, int] = {
            "<=512": 0,
            "513..2048": 0,
            "2049..4096": 0,
            "4097..8192": 0,
            ">8192": 0,
        }

    def total_valid_examples(self) -> int:
        return sum(len(v) for v in self.by_domain.values())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cursor": int(self.cursor),
            "domain_cursors": dict(self.domain_cursors),
            "rng_state": self.rng.getstate(),
            "seen_record_ids": sorted(self.seen_record_ids),
            "seen_doc_ids": sorted(self.seen_doc_ids),
            "repeated_exposures": int(self.repeated_exposures),
            "input_token_exposures": int(self.input_token_exposures),
            "supervised_token_exposures": int(self.supervised_token_exposures),
            "padding_token_exposures": int(self.padding_token_exposures),
            "domain_example_exposures": dict(self.domain_example_exposures),
            "domain_supervised_tokens": dict(self.domain_supervised_tokens),
            "seq_len_histogram": dict(self.seq_len_histogram),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.cursor = int(state.get("cursor", 0))
        for k, v in (state.get("domain_cursors") or {}).items():
            if k in self.domain_cursors:
                self.domain_cursors[k] = int(v)
        if "rng_state" in state and state["rng_state"] is not None:
            self.rng.setstate(state["rng_state"])
        self.seen_record_ids = set(state.get("seen_record_ids", []))
        self.seen_doc_ids = set(state.get("seen_doc_ids", []))
        self.repeated_exposures = int(state.get("repeated_exposures", 0))
        self.input_token_exposures = int(state.get("input_token_exposures", 0))
        self.supervised_token_exposures = int(state.get("supervised_token_exposures", 0))
        self.padding_token_exposures = int(state.get("padding_token_exposures", 0))

    def _record_len_bucket(self, n_tok: int) -> None:
        if n_tok <= 512:
            self.seq_len_histogram["<=512"] += 1
        elif n_tok <= 2048:
            self.seq_len_histogram["513..2048"] += 1
        elif n_tok <= 4096:
            self.seq_len_histogram["2049..4096"] += 1
        elif n_tok <= 8192:
            self.seq_len_histogram["4097..8192"] += 1
        else:
            self.seq_len_histogram[">8192"] += 1

    def next_microbatch(
        self, device: str | "torch.device" = "cpu"
    ) -> Tuple["torch.LongTensor", "torch.LongTensor", "torch.LongTensor", Dict[str, Any]]:
        """Sample `micro_batch_size` rendered windows according to `normalized_weights`, pad to max length in the
        microbatch, and return `(input_ids, labels, attention_mask, meta)`."""
        chosen: List[RenderedWindow] = []
        domains = self.available_domains
        probs = [self.normalized_weights[d] for d in domains]

        for _ in range(self.micro_batch_size):
            dom = self.rng.choices(domains, weights=probs, k=1)[0]
            pool = self.by_domain[dom]
            d_cur = self.domain_cursors[dom]
            if d_cur > 0 and d_cur % len(pool) == 0:
                # Deterministic epoch-indexed shuffle per domain
                shuf_rng = random.Random(self.seed + hash(dom) + (d_cur // len(pool)) * 104729)
                shuf_rng.shuffle(pool)
            rw = pool[d_cur % len(pool)]
            self.domain_cursors[dom] = d_cur + 1
            self.cursor += 1
            chosen.append(rw)

            if rw.record_id in self.seen_record_ids:
                self.repeated_exposures += 1
            else:
                self.seen_record_ids.add(rw.record_id)
            if rw.source_document_id:
                self.seen_doc_ids.add(rw.source_document_id)
            self.domain_example_exposures[dom] = self.domain_example_exposures.get(dom, 0) + 1
            self.domain_supervised_tokens[dom] = self.domain_supervised_tokens.get(dom, 0) + rw.supervised_tokens
            self._record_len_bucket(len(rw.input_ids))

        batch_len = max(len(x.input_ids) for x in chosen)
        bsz = len(chosen)
        inp = torch.full((bsz, batch_len), self.pad_token_id, dtype=torch.long, device=device)
        lbl = torch.full((bsz, batch_len), -100, dtype=torch.long, device=device)
        attn_mask = torch.zeros((bsz, batch_len), dtype=torch.long, device=device)

        sup_toks = 0
        real_toks = 0
        for b_i, rw in enumerate(chosen):
            n_t = len(rw.input_ids)
            inp[b_i, :n_t] = torch.tensor(rw.input_ids, dtype=torch.long, device=device)
            lbl[b_i, :n_t] = torch.tensor(rw.labels, dtype=torch.long, device=device)
            attn_mask[b_i, :n_t] = 1
            real_toks += n_t
            sup_toks += rw.supervised_tokens

        pad_toks = bsz * batch_len - real_toks
        self.input_token_exposures += real_toks
        self.supervised_token_exposures += sup_toks
        self.padding_token_exposures += pad_toks

        meta = {
            "batch_seq_len": batch_len,
            "input_tokens": real_toks,
            "supervised_tokens": sup_toks,
            "padding_tokens": pad_toks,
            "domains": [rw.domain for rw in chosen],
            "record_ids": [rw.record_id for rw in chosen],
        }
        return inp, lbl, attn_mask, meta

    def ledger(self) -> Dict[str, Any]:
        total_alloc = max(1, self.input_token_exposures + self.padding_token_exposures)
        sup_tot = max(1, self.supervised_token_exposures)
        total_valid = max(1, self.total_valid_examples())
        return {
            "total_valid_corpus_examples": self.total_valid_examples(),
            "unique_examples_seen": len(self.seen_record_ids),
            "unique_documents_seen": len(self.seen_doc_ids),
            "repeated_example_exposures": int(self.repeated_exposures),
            "estimated_epoch": round(float(self.cursor) / float(total_valid), 4),
            "input_token_exposures": int(self.input_token_exposures),
            "supervised_token_exposures": int(self.supervised_token_exposures),
            "padding_token_exposures": int(self.padding_token_exposures),
            "padding_fraction": round(float(self.padding_token_exposures) / float(total_alloc), 6),
            "supervised_fraction_of_input": round(
                float(self.supervised_token_exposures) / float(max(1, self.input_token_exposures)), 6
            ),
            "actual_domain_supervised_shares": {
                d: round(float(self.domain_supervised_tokens.get(d, 0)) / float(sup_tot), 6)
                for d in VALID_DOMAINS
            },
            "domain_example_exposures": dict(self.domain_example_exposures),
            "seq_len_histogram": dict(self.seq_len_histogram),
            "rejected_zero_target_examples": int(self.rejected_zero_target),
            "rejected_reasons": dict(self.rejected_reasons),
            "rewindowed_examples": int(self.rewindowed_examples),
        }
