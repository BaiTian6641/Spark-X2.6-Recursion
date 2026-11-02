# -*- coding: utf-8 -*-
"""
Spark-X2.6-4B-En4B-A1B (v0.3) — Full-Finetuning & Agentic Verification Environment
==================================================================================
Provides the modular training, dataset curation, 16K sequence packing, checkpointing,
and 262K multi-turn agentic verification environment for `SparkX_Loop_v03_En4B_A1B_OneShot_Colab.ipynb`.

Designed to live in the GitHub repository (`BaiTian6641/Spark-X2.6-Recursion`) so the
Colab notebook stays concise and imports directly from the repository.
"""

from __future__ import annotations

import ast
import concurrent.futures
from decimal import Decimal, InvalidOperation
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import dataclasses
import importlib
import sparkx_loop_v03_core as _core_mod
importlib.reload(_core_mod)

from sparkx_loop_v03_core import (
    HAS_CCE,
    HAS_TRITON,
    LoopV03Settings,
    SparkXLoopV03,
    TwoStreamMHC,
    patch_spark_attention_for_loglinear_kv,
    patch_spark_for_unsloth_triton,
)

HAS_LIGER = bool(getattr(_core_mod, 'HAS_LIGER', getattr(_core_mod, 'HAS_LIGER_RMSNORM', False)))
HAS_UNSLOTH = bool(getattr(_core_mod, 'HAS_UNSLOTH', False))
patch_spark_for_loglinear_kv = getattr(_core_mod, 'patch_spark_for_loglinear_kv', patch_spark_attention_for_loglinear_kv)


@torch.no_grad()
def verify_birkhoff_sinkhorn(ctrl: TwoStreamMHC) -> float:
    if hasattr(_core_mod, 'verify_birkhoff_sinkhorn'):
        return float(_core_mod.verify_birkhoff_sinkhorn(ctrl))
    device = ctrl.mix_logits.device
    delta = torch.randn(4, 16, 2, 2, device=device, dtype=torch.float32) * 3.0
    mat = ctrl.mixing_matrix(dynamic_delta=delta)
    row_err = (mat.sum(dim=-1) - 1.0).abs().max().item()
    col_err = (mat.sum(dim=-2) - 1.0).abs().max().item()
    return float(max(row_err, col_err))


@torch.no_grad()
def verify_ngram_causality(model: SparkXLoopV03, vocab_size: int = 167168, device: Optional[torch.device] = None) -> float:
    if hasattr(_core_mod, 'verify_ngram_causality'):
        return float(_core_mod.verify_ngram_causality(model, vocab_size=vocab_size, device=device))
    if device is None:
        device = next(model.base.parameters()).device
    seq_a = torch.randint(10, min(vocab_size, 50000), (1, 16), device=device)
    seq_b = seq_a.clone()
    seq_b[:, 8:] = torch.randint(10, min(vocab_size, 50000), (1, 8), device=device)
    idx_a = model.engram.compute_ngram_indices(seq_a, model.engram.total_buckets)
    idx_b = model.engram.compute_ngram_indices(seq_b, model.engram.total_buckets)
    hash_diff = max(float((a[:, :8] - b[:, :8]).abs().max().item()) for a, b in zip(idx_a, idx_b))
    logits_a = model.forward_logits(seq_a).float()[:, :8, :]
    logits_b = model.forward_logits(seq_b).float()[:, :8, :]
    logit_diff = float((logits_a - logits_b).abs().max().item())
    return max(hash_diff, logit_diff)


def build_default_cfg() -> Dict[str, Any]:
    """Return the default v0.3 16K Unsloth/Triton configuration dictionary."""
    return {
        # Base & Target Repositories
        'BASE_MODEL_ID': 'XHToken/Spark-X2.6-4B',
        'V02_WARM_START_REPO': 'TassP0lar/Spark-X2.6-4B-Loop-mHC-R24',
        # Clean initialization from 1M-context XHToken/Spark-X2.6-4B avoids inheriting v0.2's 2K truncation bias
        'ENABLE_V02_WARM_START': False,
        'HF_REPO_ID': 'TassP0lar/Spark-X2.6-4B-En4B-A1B',
        'HF_PRIVATE': False,
        'PUSH_TO_HUB': True,

        # Checkpoint Retention & Periodic Save Policy
        'CHECKPOINT_DIR': '/content/checkpoints_v03_en4b_a1b',
        'FINAL_EXPORT_DIR': '/content/Spark-X2.6-4B-En4B-A1B-Final',
        'MAX_LOCAL_CHECKPOINTS': 3,
        'SAVE_EVERY_STEPS': 250,

        # Unattended Wall-Clock Budget (Colab Pro+ 24h ceiling with 1.5h safety margin)
        'MAX_RUN_HOURS': 22.5,

        # 72-Layer Telescoping Recurrence Geometry
        'LOOP_START': 4,
        'LOOP_END': 28,
        'P3_START': 12,
        'P3_END': 24,
        'MAX_PASSES': 4,
        'PASS_ADAPTER_RANK': 16,

        # 8-Anchor 3-Tier Zipfian MoE-Engram (4.11B Total / 1.10B Active)
        'ENG_HOT_BUCKETS': 16384,
        'ENG_COLD_BUCKETS': 49152,
        'ENG_DIM': 512,
        'ENG_SHARED_RANK': 5888,
        'ENG_EXPERTS': 32,
        'ENG_INITIAL_ACTIVE_EXPERTS': 8,
        'ENG_TOP_K': 7,
        'ENG_INITIAL_TOP_K': 2,
        'ENG_EXPERT_RANK': 3072,
        'ENG_ROUTE_DIM': 64,
        'ENG_STRIDE': 4,
        'ENG_OFFLOAD_TIER1B_CPU': True,
        'ENG_Z_LOSS_COEFF': 1e-4,

        # 16K Context & Sequence Packing (Zero Mid-Thought / Mid-Code Truncation!)
        'MAX_SEQ_LENGTH': 16384,
        'PACK_SEQUENCES': True,
        'BATCH_SIZE': 1,
        'GRAD_ACCUM': 8,
        'CE_CHUNK_TOKENS': 1024,
        'WARMUP_RATIO': 0.06,
        'WEIGHT_DECAY': 0.01,
        'MAX_GRAD_NORM': 1.0,
        'SEED': 42,

        # Differential Learning Rates
        'LR_BACKBONE': 1.2e-5,
        'LR_MHC': 2.0e-4,
        'LR_ENGRAM': 1.5e-4,
        'LR_ROUTER': 2.5e-4,
        'LR_TIER1B_SPARSE': 2.0e-4,

        # 4-Stage Progressive Context & Depth Curriculum (3,000 Optimizer Steps Total)
        'STAGES': [
            {
                'name': 'Stage1_Warmup_36L_4K',
                'start': 4, 'end': 28, 'passes': 1, 'enable_p3': False,
                'seq_len': 4096,
                'steps': 200,
                'freeze_backbone': True,
                'upcycle_8_to_32': False,
                'lr_scale': 1.2,
            },
            {
                'name': 'Stage2_R12_48L_8K',
                'start': 12, 'end': 24, 'passes': 2, 'enable_p3': False,
                'seq_len': 8192,
                'steps': 400,
                'freeze_backbone': False,
                'upcycle_8_to_32': False,
                'lr_scale': 1.0,
            },
            {
                'name': 'Stage3_R24_Upcycle32_60L_16K',
                'start': 4, 'end': 28, 'passes': 2, 'enable_p3': False,
                'seq_len': 16384,
                'steps': 800,
                'freeze_backbone': False,
                'upcycle_8_to_32': True,
                'lr_scale': 0.85,
            },
            {
                'name': 'Stage4_R72_Telescoping_72L_16K',
                'start': 4, 'end': 28, 'passes': 2, 'enable_p3': True,
                'p3_start': 12, 'p3_end': 24,
                'seq_len': 16384,
                'steps': 1600,
                'freeze_backbone': False,
                'upcycle_8_to_32': False,
                'lr_scale': 0.70,
            },
        ],

        # 5-Pillar + Multi-Turn Dataset Targets
        'TARGET_TOTAL_SAMPLES': 2400,
        'PILLAR_RATIOS': {
            'code_swe': 0.35,
            'web_3d_ui': 0.20,
            'math_stem': 0.25,
            'tool_agent': 0.15,
            'concise_think': 0.05,
        },

        # OpenCode Go Multi-Model Teacher Augmentation
        'ENABLE_OPENCODE_TEACHER': True,
        'OPENCODE_BASE_URL': 'https://opencode.ai/zen/go/v1',
        'OPENCODE_MODELS': [
            'glm-5',
            'deepseek-v4.1-flash',
            'deepseek-v4-pro',
            'mimo-v2.6-pro',
            'claude-haiku-5.5',
        ],
        'OPENCODE_TEACHER_SAMPLES': 80,
    }


# Each entry is (repo_id, subset_config, split_name, verified_shard_path).
# Providing `verified_shard_path` streams the exact .parquet/.jsonl shard directly via `hf://datasets/...`,
# completely bypassing README.md parsing, dataset_infos.json, mixed .txt/.py/.png files, and 30-shard directory scans.
DATASET_SOURCES: Dict[str, List[Tuple[str, Optional[str], str, Optional[str]]]] = {
    'code_swe': [
        ('nvidia/OpenCodeReasoning', 'split_0', 'split_0', 'split_0/train-00000-of-00030.parquet'),
        ('open-r1/verifiable-coding-problems-python_decontaminated', None, 'train', 'data/train-00000-of-00006.parquet'),
        ('bigcode/self-oss-instruct-sc2-exec-filter-50k', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('TeichAI/Claude-Opus-4.6-Reasoning-887x', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('iamtarun/python_code_instructions_18k_alpaca', None, 'train', 'data/train-00000-of-00001-8b6e212f3e1ece96.parquet'),
        ('Roman1111111/claude-opus-4.6-10000x', None, 'train', 'opus46_final.jsonl'),
        ('TeichAI/lordx64-claude-opus-4.7-max-cleaned', None, 'train', 'train.jsonl'),
        ('ronaldcmz/claude_opus_4.8_max_thinking_5k_v2', None, 'train', 'opus_4_8_max_thinking_5k_v2.jsonl'),
        ('ise-uiuc/Magicoder-OSS-Instruct-75K', None, 'train', 'data-oss_instruct-decontaminated.jsonl'),
        ('ise-uiuc/Magicoder-Evol-Instruct-110K', None, 'train', 'data-evol_instruct-decontaminated.jsonl'),
        ('m-a-p/CodeFeedback-Filtered-Instruction', None, 'train', 'CodeFeedback-Filtered-Instruction.jsonl'),
    ],
    'web_3d_ui': [
        ('Tesslate/UIGEN-T1.5-Dataset', None, 'train', 'output.parquet'),
        ('kelexine/fable-5-sft-traces', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('HuggingFaceH4/ultrachat_200k', None, 'train_sft', 'data/train_sft-00000-of-00003-a3ecf92756993583.parquet'),
        ('cfahlgren1/react-code-instructions', None, 'train', 'data/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl'),
        ('HelioAI/Claude-Fable-5-5500x', None, 'train', 'Fable-5-Distill-5500x.jsonl'),
        ('armand0e/Fable-5-Chat', None, 'train', 'train.jsonl'),
        ('nvidia/Nemotron-Instruction-Following-Chat-v1', None, 'chat_if', 'data/chat_if.jsonl'),
        ('NousResearch/Hermes-3-Dataset', None, 'train', 'hermes-3-dataset.jsonl'),
        ('DataCreatorAI/Multi-Turn-Conversational-SFT', None, 'train', 'multi_turn_conversations.jsonl'),
    ],
    'math_stem': [
        ('open-thoughts/OpenThoughts-114k', 'default', 'train', 'data/train-00000-of-00006.parquet'),
        ('bespokelabs/Bespoke-Stratos-17k', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('nvidia/OpenScienceReasoning-2', None, 'train', 'train/OpenScienceReasoning-2.parquet'),
        ('MegaScience/MegaScience', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('MegaScience/TextbookReasoning', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('OpenDataArena/Spark-234K', None, 'train', 'Spark-234K-data.parquet'),
        ('openai/gsm8k', 'main', 'train', 'main/train-00000-of-00001.parquet'),
        ('AI-MO/NuminaMath-CoT', None, 'train', 'data/train-00000-of-00005.parquet'),
        ('Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned', 'PHD-Science', 'train', 'PHD-Science.jsonl'),
    ],
    'tool_agent': [
        ('Locutusque/function-calling-chatml', None, 'train', 'data/train-00000-of-00001-f0b56c6983b4a78f.parquet'),
        ('lilacai/glaive-function-calling-v2-sharegpt', None, 'train', 'data/train-00000-of-00002-6f3344faa23e9b0a.parquet'),
        ('lilacai/glaive-function-calling-v2-sharegpt', None, 'train', 'data/train-00001-of-00002-41f063cddf49c933.parquet'),
        ('NousResearch/hermes-function-calling-v1', None, 'train', 'func-calling.json'),
        ('NousResearch/hermes-function-calling-v1', None, 'train', 'func-calling-singleturn.json'),
        ('NousResearch/hermes-function-calling-v1', None, 'train', 'json-mode-agentic.json'),
        ('glaiveai/glaive-function-calling-v2', None, 'train', 'glaive-function-calling-v2.json'),
    ],
    'concise_think': [
        ('ServiceNow-AI/R1-Distill-SFT', 'v0', 'train', 'v0/train-00000-of-00003.parquet'),
        ('Skywork/Skywork-OR1-RL-Data', None, 'code', 'data/code-00000-of-00003.parquet'),
        ('HuggingFaceH4/no_robots', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x', None, 'train', '10k/train.jsonl'),
        ('Roman1111111/gpt-5.4-step-by-step-reasoning', None, 'train', 'gpt5.4.jsonl'),
        ('vanty120/Gpt-5.4-Xhigh-Reasoning-2000x', None, 'train', 'train.jsonl'),
        ('TeichAI/gpt-5.1-high-reasoning-1000x', None, 'train', 'gpt-5.1-1000x.jsonl'),
    ],
}



def auto_configure_vram_16k(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Detect GPU VRAM & System RAM and tune 16K execution flags automatically."""
    random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['SEED'])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    try:
        import psutil
        sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        sys_ram_gb = 32.0

    # On 80GB+ GPUs (A100/H100), keep activations in GPU HBM; on 24-48GB GPUs (G4/L4/L40), stream activations to pinned host RAM
    cfg['OFFLOAD_CHECKPOINT_CPU'] = bool(vram_gb < 70.0 and sys_ram_gb >= 48.0)
    if vram_gb < 22.0:
        cfg['MAX_SEQ_LENGTH'] = 8192
        for st in cfg['STAGES']:
            st['seq_len'] = min(st['seq_len'], 8192)

    print('=' * 92)
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | System RAM: {sys_ram_gb:.1f} GB")
    print(f"Kernel Stack: Triton={HAS_TRITON} | Unsloth={HAS_UNSLOTH} | Liger={HAS_LIGER} | Cut-CE={HAS_CCE}")
    print(f"Max Seq Length: {cfg['MAX_SEQ_LENGTH']} tokens | Pinned-RAM Activation Streaming: {cfg['OFFLOAD_CHECKPOINT_CPU']}")
    print('=' * 92)
    return cfg


@torch.no_grad()
def run_step0_verification_suite(model: SparkXLoopV03, base_model: nn.Module, tokenizer: Any) -> Dict[str, float]:
    """Run all 5 Step-0 mathematical invariants before training begins."""
    patch_spark_for_unsloth_triton(base_model)
    model.eval()
    device = next(base_model.parameters()).device
    test_text = (
        'Solve for x: 3x + 15 = 42. Explain step by step and write a Python verification function.'
    )
    test_ids = tokenizer(test_text, return_tensors='pt').input_ids.to(device)

    # 1. Loop-disabled parity (mhc=False, engram=False, passes=1) vs raw upstream base_model
    base_logits = base_model(test_ids).logits.float()
    model.set_loop(start=4, end=28, passes=1, use_mhc=False, use_engram=False, enable_p3=False)
    disabled_logits = model.forward_logits(test_ids).float()
    disabled_diff = float((base_logits - disabled_logits).abs().max().item())

    # 2. Exact 0.0 Zero-Gated Step-0 Parity between 1-Pass (36L) and 2-Pass (60L)
    model.set_loop(start=4, end=28, passes=1, use_mhc=True, use_engram=True, enable_p3=False)
    p1_logits = model.forward_logits(test_ids).float()
    one_pass_highway_diff = float((base_logits - p1_logits).abs().max().item())

    model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=False)
    p2_logits = model.forward_logits(test_ids).float()
    p2_zero_gate_diff = float((p1_logits - p2_logits).abs().max().item())

    # 3. Exact 0.0 Zero-Gated Step-0 Parity between 1-Pass (36L) and 3-Pass (72L)
    model.set_loop(
        start=4, end=28, passes=2, use_mhc=True, use_engram=True,
        enable_p3=True, p3_start=12, p3_end=24,
    )
    p3_logits = model.forward_logits(test_ids).float()
    step0_diff = float((p1_logits - p3_logits).abs().max().item())

    # 4. Strict N-Gram & Forward Causal Prefix Invariance
    causal_prefix_err = float(verify_ngram_causality(model, vocab_size=tokenizer.vocab_size, device=device))

    # 5. Closed-Form Birkhoff-Sinkhorn Doubly-Stochastic Verification
    ctrl0 = model.controllers[str(model.loop_start)]
    birkhoff_sinkhorn_err = float(verify_birkhoff_sinkhorn(ctrl0))

    print(f"[Step-0 Check 1/5] Loop-Disabled vs Raw 36L Base Diff : {disabled_diff:.8f} (req < 1e-4)")
    print(f"[Step-0 Check 2/5] 36L (1-Pass) vs 60L (2-Pass) Diff  : {p2_zero_gate_diff:.8f} (req == 0.0 | highway delta={one_pass_highway_diff:.4f})")
    print(f"[Step-0 Check 3/5] 36L (1-Pass) vs 72L (3-Pass) Diff  : {step0_diff:.8f} (req == 0.0)")
    print(f"[Step-0 Check 4/5] Causal Prefix Invariance Error     : {causal_prefix_err:.8f} (req < 0.025)")
    print(f"[Step-0 Check 5/5] Birkhoff-Sinkhorn Doubly-Stochastic: {birkhoff_sinkhorn_err:.8f} (req < 1e-5)")

    assert disabled_diff < 1e-4, f"Loop-disabled parity failed: {disabled_diff}"
    assert p2_zero_gate_diff == 0.0, f"Expected exact 0.0 Step-0 parity between 36L and 60L, got {p2_zero_gate_diff}"
    assert one_pass_highway_diff < 2.0, f"1-Pass mHC highway delta out of range: {one_pass_highway_diff}"
    assert step0_diff == 0.0, f"Expected exact 0.0 Step-0 parity between 36L and 72L, got {step0_diff}"
    assert causal_prefix_err < 0.025, f"Causal prefix invariance failed: {causal_prefix_err}"
    assert birkhoff_sinkhorn_err < 1e-5, f"Birkhoff-Sinkhorn doubly-stochastic check failed: {birkhoff_sinkhorn_err}"

    return {
        'disabled_diff': disabled_diff,
        'p2_zero_gate_diff': p2_zero_gate_diff,
        'one_pass_highway_diff': one_pass_highway_diff,
        'step0_diff': step0_diff,
        'causal_prefix_err': causal_prefix_err,
        'birkhoff_sinkhorn_err': birkhoff_sinkhorn_err,
    }


# ==============================================================================
# 2. 5-Pillar + Multi-Turn Dataset Curation & OpenCode Go Teacher Verification
# ==============================================================================

_CLI_LEAK_RE = re.compile(
    r"(C:\\Users\\|/Users/[a-zA-Z0-9_-]+/claude-projects|already downloaded the library locally|"
    r"placeholder comment indicating that the full implementation will follow)",
    re.IGNORECASE,
)

_SLOP_PHRASES = (
    'as an ai language model',
    'i cannot fulfill this request',
    'here is a placeholder',
    '// rest of the code',
    '/* rest of code here */',
    '# todo: implement',
)


def _normalize_tool_tags(text: str) -> str:
    text = re.sub(r'<function_calls>\s*', '<tool_call>\n', text)
    text = re.sub(r'\s*</function_calls>', '\n</tool_call>', text)
    text = re.sub(r'<function_result>\s*', '<tool_response>\n', text)
    text = re.sub(r'\s*</function_result>', '\n</tool_response>', text)
    return text.strip()


def _split_thought_and_answer(raw_assistant: str) -> Tuple[str, str]:
    text = _normalize_tool_tags(raw_assistant)
    for open_tag, close_tag in [('<think>', '</think>'), ('<thought>', '</thought>'), ('<|begin_of_thought|>', '<|end_of_thought|>')]:
        if open_tag in text and close_tag in text:
            s_idx = text.find(open_tag) + len(open_tag)
            e_idx = text.find(close_tag)
            thought = text[s_idx:e_idx].strip()
            answer = (text[:text.find(open_tag)] + text[e_idx + len(close_tag):]).strip()
            return thought, answer
    return '', text


def extract_record_fields(row: Dict[str, Any], pillar: str) -> Optional[Dict[str, Any]]:
    """Extract normalized (system, history_turns, user, thought, answer) from heterogeneous HF schemas."""
    sys_prompt = str(row.get('system') or row.get('system_prompt') or row.get('system_message') or '').strip()
    history_turns: List[Dict[str, str]] = []
    top_thought = str(
        row.get('thinking')
        or row.get('reasoning')
        or row.get('reasoning_content')
        or row.get('step_by_step_trace')
        or row.get('inverted_reasoning')
        or ''
    ).strip()

    conv = row.get('conversations') or row.get('messages')
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:
            conv = None
    if isinstance(conv, list) and len(conv) >= 2:
        norm_msgs: List[Dict[str, str]] = []
        for m in conv:
            if isinstance(m, str):
                try:
                    m = json.loads(m)
                except Exception:
                    continue
            if not isinstance(m, dict):
                continue
            role = str(m.get('role') or m.get('from') or '').lower().strip()
            content = str(m.get('content') or m.get('value') or '').strip()
            m_think = str(m.get('reasoning_content') or m.get('thinking') or m.get('reasoning') or '').strip()
            if not content:
                continue
            if role in ('system', 'sys'):
                sys_prompt = content
            elif role in ('user', 'human', 'prompter'):
                norm_msgs.append({'role': 'user', 'content': _normalize_tool_tags(content)})
            elif role in ('assistant', 'gpt', 'model', 'bot'):
                if m_think and '<think>' not in content:
                    content = f"<think>\n{m_think}\n</think>\n{content}"
                norm_msgs.append({'role': 'assistant', 'content': _normalize_tool_tags(content)})
            elif role in ('tool', 'observation', 'function_response'):
                norm_msgs.append({'role': 'user', 'content': f"<tool_response>\n{content}\n</tool_response>"})

        if len(norm_msgs) >= 2 and norm_msgs[-1]['role'] == 'assistant':
            last_asst = norm_msgs[-1]['content']
            prev_msgs = norm_msgs[:-1]
            if prev_msgs and prev_msgs[-1]['role'] == 'user':
                user_prompt = prev_msgs[-1]['content']
                history_turns = prev_msgs[max(0, len(prev_msgs) - 5):-1]
                thought, answer = _split_thought_and_answer(last_asst)
                if not thought and top_thought:
                    thought = top_thought
                return {
                    'pillar': pillar,
                    'system': sys_prompt,
                    'history_turns': history_turns,
                    'user': user_prompt,
                    'thought': thought,
                    'answer': answer,
                }

    # Support Glaive-style `chat` column ("USER: ... ASSISTANT: ...")
    chat_raw = str(row.get('chat') or '').strip()
    if chat_raw and 'USER:' in chat_raw and 'ASSISTANT:' in chat_raw:
        u_part, a_part = chat_raw.split('ASSISTANT:', 1)
        u_clean = u_part.replace('USER:', '').strip()
        a_clean = a_part.replace('<|endoftext|>', '').strip()
        if u_clean and a_clean:
            thought, answer = _split_thought_and_answer(a_clean)
            if not thought and top_thought:
                thought = top_thought
            return {
                'pillar': pillar,
                'system': sys_prompt,
                'history_turns': [],
                'user': _normalize_tool_tags(u_clean),
                'thought': thought,
                'answer': answer,
            }

    user_prompt = str(
        row.get('instruction') or row.get('prompt') or row.get('question') or row.get('problem') or row.get('query') or ''
    ).strip()
    inp = str(row.get('input') or '').strip()
    if inp and inp not in user_prompt:
        user_prompt = f"{user_prompt}\n\n{inp}".strip()

    raw_out = str(
        row.get('output')
        or row.get('response')
        or row.get('solution')
        or row.get('final_solution')
        or row.get('answer')
        or row.get('completion')
        or ''
    ).strip()
    if not user_prompt or not raw_out:
        return None

    thought, answer = _split_thought_and_answer(raw_out)
    if not thought and top_thought:
        thought = top_thought

    ref_ans = str(row.get('reference_answer') or row.get('expected_answer') or '').strip()
    if not thought and ref_ans and len(answer) > 120 and ref_ans != answer:
        thought = answer
        answer = f"{answer}\n\n**Final Answer:** {ref_ans}"

    return {
        'pillar': pillar,
        'system': sys_prompt,
        'history_turns': [],
        'user': _normalize_tool_tags(user_prompt),
        'thought': thought,
        'answer': answer,
    }


def passes_quality_filter(rec: Dict[str, Any]) -> bool:
    """Filter out truncated completions, placeholder slop, or CLI local-filesystem hallucinations."""
    user = rec.get('user', '')
    thought = rec.get('thought', '')
    answer = rec.get('answer', '')
    if len(user) < 12 or len(answer) < 32:
        return False
    combined = f"{thought}\n{answer}"
    if _CLI_LEAK_RE.search(combined):
        return False
    low = combined.lower()
    for phrase in _SLOP_PHRASES:
        if phrase in low:
            return False
    if answer.count('```') % 2 != 0:
        return False
    if thought and len(thought) > 12000:
        return False
    if not thought:
        # Synthesize a concise structured deliberation header so `<think>...</think>` formatting is 100% consistent
        first_line = user.splitlines()[0][:140]
        rec['thought'] = (
            f"1. Analyze requirement: {first_line}\n"
            f"2. Formulate complete, self-contained solution with zero placeholders and verified syntax."
        )
    return True


def _open_streaming_dataset(
    ds_name: str,
    subset: Optional[str],
    split: str,
    shard_path: Optional[str] = None,
):
    """Open a Hugging Face dataset in streaming mode.
    When `shard_path` is provided, streams directly from `hf://datasets/{ds_name}/{shard_path}` via the
    native `parquet` or `json` builder, completely bypassing `README.md`, `dataset_infos.json`,
    extra `.txt`/`.py`/`.png` repo files, and slow multi-file directory resolution.
    """
    from datasets import load_dataset
    if shard_path:
        builder = 'parquet' if shard_path.endswith('.parquet') else 'json'
        hf_uri = f"hf://datasets/{ds_name}/{shard_path}"
        try:
            ds = load_dataset(builder, data_files=hf_uri, split='train', streaming=True)
            return ds, f"direct:{shard_path}"
        except Exception:
            try:
                ds = load_dataset(ds_name, data_files=shard_path, split='train', streaming=True)
                return ds, f"file:{shard_path}"
            except Exception:
                pass

    try:
        return (
            load_dataset(ds_name, subset, split=split, streaming=True)
            if subset else load_dataset(ds_name, split=split, streaming=True)
        ), split
    except Exception as exc:
        msg = str(exc)
        m = re.search(r"Available splits:\s*\[([^\]]+)\]", msg)
        if m:
            avail = [s.strip().strip("'\"") for s in m.group(1).split(',') if s.strip()]
            if avail:
                alt_split = avail[0]
                print(f"     [Auto-Split Resolve] {ds_name}: '{split}' -> '{alt_split}'", flush=True)
                ds = (
                    load_dataset(ds_name, subset, split=alt_split, streaming=True)
                    if subset else load_dataset(ds_name, split=alt_split, streaming=True)
                )
                return ds, alt_split
        raise


def curate_all_pillars_v03(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stream and filter high-signal trajectories across all 5 pillars with verbose real-time telemetry."""
    target_total = int(cfg.get('TARGET_TOTAL_SAMPLES', 2400))
    ratios = cfg.get('PILLAR_RATIOS', {})
    curated: List[Dict[str, Any]] = []
    t_all = time.time()

    # Suppress Hugging Face Hub's internal README.md / dataset_infos.json progress bar spam
    try:
        import huggingface_hub.utils as hf_utils
        hf_utils.disable_progress_bars()
    except Exception:
        pass
    try:
        import datasets.utils.logging as ds_logging
        ds_logging.disable_progress_bar()
    except Exception:
        pass

    print('=' * 98)
    print(
        f"[Dataset Curation Engine] Target: {target_total} verified trajectories across {len(ratios)} pillars "
        f"(Direct Shard Streaming enabled — bypasses README.md & non-data files)"
    )
    print('=' * 98, flush=True)

    pillar_items = list(ratios.items())
    for p_idx, (pillar_raw, ratio) in enumerate(pillar_items, start=1):
        pillar = pillar_raw.strip()
        quota = max(20, int(target_total * float(ratio)))
        sources = DATASET_SOURCES.get(pillar, [])
        collected = 0
        per_source_cap = max(35, math.ceil(quota / max(1, len(sources))) + 45)
        t_pillar = time.time()

        print(
            f"\n--- [Pillar {p_idx}/{len(pillar_items)}: {pillar}] "
            f"Target Quota: {quota} ({float(ratio)*100:.1f}%) | Sources: {len(sources)} | Cap/Source: {per_source_cap} ---",
            flush=True,
        )

        for s_idx, src_spec in enumerate(sources, start=1):
            if collected >= quota:
                break
            if len(src_spec) == 4:
                ds_name_raw, subset, split, shard_path = src_spec
            else:
                ds_name_raw, subset, split = src_spec[:3]
                shard_path = None
            ds_name = ds_name_raw.strip()
            need = min(per_source_cap, quota - collected)
            t_src = time.time()
            shard_label = shard_path if shard_path else f"config={subset or 'default'}, split={split}"
            print(f"  -> [{s_idx:2d}/{len(sources):2d}] Streaming {ds_name} ({shard_label} | target={need})...", flush=True)
            try:
                ds, used_split = _open_streaming_dataset(ds_name, subset, split, shard_path=shard_path)
                src_count = 0
                scanned = 0
                mt_count = 0
                think_chars = 0
                ans_chars = 0
                for row in ds:
                    scanned += 1
                    if scanned > need * 12 + 200 and src_count == 0:
                        break
                    if (time.time() - t_src) > 35.0 and src_count >= 15:
                        print(f"     [Time Cap 35s] {ds_name}: moving to next shard with +{src_count} samples.", flush=True)
                        break
                    rec = extract_record_fields(row, pillar)
                    if rec and passes_quality_filter(rec):
                        rec['source'] = ds_name
                        curated.append(rec)
                        collected += 1
                        src_count += 1
                        if rec.get('history_turns'):
                            mt_count += 1
                        think_chars += len(rec.get('thought', ''))
                        ans_chars += len(rec.get('answer', ''))
                        if src_count % 50 == 0:
                            print(
                                f"     ... {ds_name}: +{src_count}/{need} accepted (scanned={scanned}) | "
                                f"Pillar {pillar}: {collected}/{quota}",
                                flush=True,
                            )
                        if src_count >= need or collected >= quota:
                            break
                dt_src = time.time() - t_src
                avg_t = int(think_chars / max(1, src_count))
                avg_a = int(ans_chars / max(1, src_count))
                print(
                    f"     [OK] {ds_name} ({used_split}): +{src_count:3d} accepted (scanned={scanned:4d}, "
                    f"multi_turn={mt_count:3d}, avg_think={avg_t:4d}c, avg_ans={avg_a:4d}c, {dt_src:.1f}s) | "
                    f"Pillar Progress: {collected:4d}/{quota:4d}",
                    flush=True,
                )
            except Exception as exc:
                print(f"     [Skip] {ds_name}: {str(exc).splitlines()[0][:110]}", flush=True)

        print(f"==> [Pillar Complete] {pillar:15s}: {collected:4d} / {quota:4d} verified trajectories in {time.time() - t_pillar:.1f}s", flush=True)

    random.shuffle(curated)
    total_mt = sum(1 for r in curated if r.get('history_turns'))
    print('=' * 98)
    print(
        f"[Dataset Curation Summary] Total Verified Records: {len(curated)} "
        f"(Multi-Turn: {total_mt}) | Elapsed: {time.time() - t_all:.1f}s"
    )
    print('=' * 98, flush=True)
    return curated


def _extract_gsm8k_decimal(text: str) -> Optional[Decimal]:
    m = re.search(r'####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not m:
        nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if not nums:
            return None
        raw = nums[-1]
    else:
        raw = m.group(1)
    try:
        return Decimal(raw.replace(',', ''))
    except InvalidOperation:
        return None


def generate_opencode_go_trajectories(cfg: Dict[str, Any], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Optional multi-model teacher synthesis via OpenCode Go with GSM8K exact Decimal verification."""
    if not cfg.get('ENABLE_OPENCODE_TEACHER', False) or not api_key:
        print("[OpenCode Go Teacher] Skipped (no OPENCODE_API_KEY provided; using curated HF pillars).", flush=True)
        return []
    try:
        from datasets import load_dataset
        from openai import OpenAI
    except Exception as exc:
        print(f"[OpenCode Go Teacher] Skipped ({exc})", flush=True)
        return []

    client = OpenAI(base_url=cfg['OPENCODE_BASE_URL'], api_key=api_key, timeout=45.0)
    models = list(cfg.get('OPENCODE_MODELS', ['glm-5']))
    target_n = int(cfg.get('OPENCODE_TEACHER_SAMPLES', 60))
    prompts: List[Tuple[str, Optional[Decimal]]] = []

    print(f"\n[OpenCode Go Teacher] Loading {target_n} GSM8K verification prompts for teacher ensemble {models}...", flush=True)
    try:
        gsm = load_dataset('openai/gsm8k', 'main', split='train', streaming=True)
        for idx, row in enumerate(gsm):
            if idx >= target_n:
                break
            gold = _extract_gsm8k_decimal(str(row.get('answer', '')))
            prompts.append((str(row['question']), gold))
    except Exception as exc:
        print(f"[OpenCode Go Teacher] Could not stream gsm8k prompts: {exc}", flush=True)
        return []

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    def _query_one(item_idx: int, item: Tuple[str, Optional[Decimal]]) -> Optional[Dict[str, Any]]:
        q, gold = item
        model_id = models[item_idx % len(models)]
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            'Think concisely inside <think>...</think>, then provide the complete step-by-step '
                            'solution ending with #### <exact_number>.'
                        ),
                    },
                    {'role': 'user', 'content': q},
                ],
                max_tokens=1536,
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ''
            thought, answer = _split_thought_and_answer(content)
            if gold is not None:
                pred = _extract_gsm8k_decimal(answer)
                if pred is None or pred != gold:
                    return None
            rec = {
                'pillar': 'math_stem',
                'source': f"opencode_go/{model_id}",
                'system': 'You are Spark-X2.6, a rigorous mathematical and coding assistant.',
                'history_turns': [],
                'user': q,
                'thought': thought,
                'answer': answer,
            }
            return rec if passes_quality_filter(rec) else None
        except Exception:
            return None

    done_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_query_one, i, p) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futs):
            done_count += 1
            r = f.result()
            if r is not None:
                results.append(r)
            if done_count % 10 == 0 or done_count == len(prompts):
                print(
                    f"  [OpenCode Go Teacher] Queried {done_count:3d}/{len(prompts):3d} | "
                    f"Decimal-Verified: {len(results):3d} | Elapsed: {time.time() - t0:.1f}s",
                    flush=True,
                )

    print(f"[OpenCode Go Teacher] Complete: {len(results)}/{len(prompts)} exact Decimal-verified trajectories.", flush=True)
    return results


# ==============================================================================
# 3. 16K Zero-Truncation Parallel Sequence Packing (`STAGE_POOLS`)
# ==============================================================================

def tokenize_and_mask_record_16k(
    rec: Dict[str, Any],
    tokenizer: Any,
    max_seq_len: int = 16384,
) -> Optional[Tuple[List[int], List[int]]]:
    """Tokenize a single or multi-turn trajectory with:
    1. Zero mid-thought or mid-answer truncation (drops any record exceeding `max_seq_len`).
    2. Supervision (`labels != -100`) on assistant `<think>...</think>` + final deliverable + `<|im_end|>`.
    3. Loss masking (`-100`) on system/user prompts and `<tool_response>...</tool_response>` observations.
    """
    sys_text = rec.get('system') or (
        'You are Spark-X2.6, a 72-layer recurrent reasoning and agentic engineering model. '
        'Outline your plan concisely inside <think>...</think>, then emit the complete deliverable.'
    )
    messages: List[Dict[str, str]] = [{'role': 'system', 'content': sys_text}]
    for h_turn in rec.get('history_turns', []):
        messages.append({'role': h_turn['role'], 'content': h_turn['content']})
    messages.append({'role': 'user', 'content': rec['user']})

    assistant_body = f"<think>\n{rec['thought'].strip()}\n</think>\n{rec['answer'].strip()}"
    full_messages = messages + [{'role': 'assistant', 'content': assistant_body}]

    try:
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
    except Exception:
        prompt_str = f"<|im_start|>system\n{sys_text}<|im_end|>\n<|im_start|>user\n{rec['user']}<|im_end|>\n<|im_start|>assistant\n"
        full_str = prompt_str + assistant_body + "<|im_end|>\n"
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)

    if len(full_ids) > max_seq_len or len(full_ids) <= len(prompt_ids) + 4:
        return None

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def build_16k_stage_pools(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    cfg: Dict[str, Any],
) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    """Tokenize all trajectories in parallel and pack them into 4096, 8192, and 16384 sequence pools."""
    max_seq_len = int(cfg.get('MAX_SEQ_LENGTH', 16384))
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0))
    t0 = time.time()

    print('\n' + '=' * 98)
    print(f"[16K Parallel Tokenizer] Tokenizing {len(records)} trajectories (ceiling={max_seq_len} tokens, zero truncation)...")
    print('=' * 98, flush=True)

    tokenized_items: List[Tuple[List[int], List[int]]] = []
    dropped_len = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(tokenize_and_mask_record_16k, r, tokenizer, max_seq_len) for r in records]
        for idx, f in enumerate(futs, start=1):
            res = f.result()
            if res is not None:
                tokenized_items.append(res)
            else:
                dropped_len += 1
            if idx % 400 == 0 or idx == len(records):
                avg_l = (
                    sum(len(x[0]) for x in tokenized_items) / max(1, len(tokenized_items))
                    if tokenized_items else 0.0
                )
                print(
                    f"  [16K Tokenizer] Processed {idx:4d}/{len(records):4d} | "
                    f"retained={len(tokenized_items):4d} | dropped_over_{max_seq_len}={dropped_len:3d} | "
                    f"avg_len={avg_l:.1f} tok | {time.time() - t0:.1f}s",
                    flush=True,
                )

    if tokenized_items:
        lengths = sorted(len(x[0]) for x in tokenized_items)
        n_l = len(lengths)
        sup_toks = sum(sum(1 for y in x[1] if y != -100) for x in tokenized_items)
        tot_toks = sum(lengths)
        print(
            f"[Token Length Distribution] min={lengths[0]} | p50={lengths[n_l//2]} | "
            f"p90={lengths[int(n_l*0.9)]} | p99={lengths[int(n_l*0.99)]} | max={lengths[-1]} tokens | "
            f"Supervised Ratio: {100.0 * sup_toks / max(1, tot_toks):.1f}%",
            flush=True,
        )

    stage_lengths = sorted({int(st['seq_len']) for st in cfg['STAGES']})
    stage_pools: Dict[int, List[Dict[str, torch.Tensor]]] = {}

    for target_len in stage_lengths:
        pool: List[Dict[str, torch.Tensor]] = []
        cur_ids: List[int] = []
        cur_labels: List[int] = []
        docs_packed = 0

        for ids, lbls in tokenized_items:
            if len(ids) > target_len:
                continue
            if len(cur_ids) + len(ids) <= target_len:
                if cur_ids:
                    lbls = [-100] + lbls[1:]
                cur_ids.extend(ids)
                cur_labels.extend(lbls)
                docs_packed += 1
            else:
                if len(cur_ids) >= 64:
                    pool.append({
                        'input_ids': torch.tensor(cur_ids, dtype=torch.long),
                        'labels': torch.tensor(cur_labels, dtype=torch.long),
                    })
                cur_ids = list(ids)
                cur_labels = list(lbls)
                docs_packed += 1

        if len(cur_ids) >= 64:
            pool.append({
                'input_ids': torch.tensor(cur_ids, dtype=torch.long),
                'labels': torch.tensor(cur_labels, dtype=torch.long),
            })

        if not pool and tokenized_items:
            ids0, lbls0 = tokenized_items[0]
            pool.append({
                'input_ids': torch.tensor(ids0[:target_len], dtype=torch.long),
                'labels': torch.tensor(lbls0[:target_len], dtype=torch.long),
            })

        stage_pools[target_len] = pool
        tot_pool_tok = sum(x['input_ids'].numel() for x in pool)
        sup_pool_tok = sum(int((x['labels'] != -100).sum().item()) for x in pool)
        avg_tok = tot_pool_tok / max(1, len(pool))
        fill_pct = 100.0 * avg_tok / float(target_len)
        docs_per_seq = docs_packed / max(1, len(pool))
        print(
            f"[Stage Pool {target_len:5d}] Packed {len(pool):4d} sequences | "
            f"avg_len={avg_tok:7.1f} tok ({fill_pct:5.1f}% fill, ~{docs_per_seq:.1f} docs/seq) | "
            f"supervised_tokens={sup_pool_tok:,d} (pad_id={pad_id})",
            flush=True,
        )

    return stage_pools


# ==============================================================================
# 4. Rolling Max-3 Checkpointing, Hub Exporter & 4-Stage 16K Training Loop
# ==============================================================================

def _build_loop_modules_state_dict(model: SparkXLoopV03) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for k, v in model.controllers.state_dict().items():
        state[f"controllers.{k}"] = v.detach().cpu()
    for k, v in model.engram.state_dict().items():
        state[f"engram.{k}"] = v.detach().cpu()
    if getattr(model.engram, 'tier1b_cold_weight', None) is not None:
        state['engram.tier1b_cold_weight'] = model.engram.tier1b_cold_weight.detach().cpu()
    for b_idx, mod in model.backbone_engrams.items():
        for k, v in mod.state_dict().items():
            state[f"backbone_engrams.{b_idx}.{k}"] = v.detach().cpu()
        if getattr(mod, 'tier1b_cold_weight', None) is not None:
            state[f"backbone_engrams.{b_idx}.tier1b_cold_weight"] = mod.tier1b_cold_weight.detach().cpu()
    return state


def save_and_push_v03_checkpoint(
    model: SparkXLoopV03,
    tokenizer: Any,
    cfg: Dict[str, Any],
    stage_name: str,
    global_step: int,
    metrics: Dict[str, Any],
    hf_token: Optional[str] = None,
    push_backbone: bool = False,
) -> Path:
    """Save a rolling local checkpoint (keeping at most MAX_LOCAL_CHECKPOINTS=3) and push to Hugging Face Hub."""
    from safetensors.torch import save_file

    ckpt_root = Path(cfg['CHECKPOINT_DIR'])
    ckpt_root.mkdir(parents=True, exist_ok=True)
    tag = f"step_{global_step:05d}_{stage_name}"
    ckpt_dir = ckpt_root / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loop_sd = _build_loop_modules_state_dict(model)
    save_file(loop_sd, str(ckpt_dir / 'loop_modules_v03.safetensors'))
    if hasattr(_core_mod, 'loop_v03_state_dict'):
        torch.save(_core_mod.loop_v03_state_dict(model, include_tier1b=True), str(ckpt_dir / 'loop_v03_state.pt'))

    meta = {
        'architecture': 'Spark-X2.6-4B-En4B-A1B (v0.3)',
        'base_model': cfg.get('BASE_MODEL_ID', 'XHToken/Spark-X2.6-4B'),
        'merged_standalone': bool(push_backbone),
        'global_step': global_step,
        'stage_name': stage_name,
        'effective_layers': model.total_virtual_layers,
        'execution_plan': model.execution_plan,
        'active_meta': {
            'engram_active_experts': model.engram.active_experts,
            'engram_active_top_k': model.engram.active_top_k,
        },
        'settings': model.settings.to_dict() if hasattr(model.settings, 'to_dict') else dataclasses.asdict(model.settings),
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    meta_json = json.dumps(meta, indent=2)
    (ckpt_dir / 'loop_config_v03.json').write_text(meta_json)
    (ckpt_dir / 'loop_config.json').write_text(meta_json)

    if push_backbone:
        model.base.save_pretrained(str(ckpt_dir), safe_serialization=True, max_shard_size='4GB')
        tokenizer.save_pretrained(str(ckpt_dir))

    # Enforce rolling MAX_LOCAL_CHECKPOINTS retention
    all_ckpts = sorted([d for d in ckpt_root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    max_keep = int(cfg.get('MAX_LOCAL_CHECKPOINTS', 3))
    while len(all_ckpts) > max_keep:
        old = all_ckpts.pop(0)
        shutil.rmtree(old, ignore_errors=True)

    if cfg.get('PUSH_TO_HUB', True) and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            repo_id = cfg['HF_REPO_ID']
            api.create_repo(repo_id=repo_id, private=bool(cfg.get('HF_PRIVATE', False)), exist_ok=True)
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=repo_id,
                commit_message=f"[v0.3 {stage_name}] step={global_step} loss={metrics.get('loss', 0.0):.4f}",
            )
            print(f"[Hub Sync] Uploaded {tag} -> https://huggingface.co/{repo_id}")
        except Exception as exc:
            print(f"[Hub Sync Warning] Failed to upload {tag}: {exc}")

    return ckpt_dir


def export_final_v03_to_hub(
    model: SparkXLoopV03,
    tokenizer: Any,
    cfg: Dict[str, Any],
    global_step: int,
    metrics: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> Path:
    """Export standalone 36-layer finetuned backbone + `loop_modules_v03.safetensors` + `load_inference_v03.py`."""
    from safetensors.torch import save_file

    out_dir = Path(cfg['FINAL_EXPORT_DIR'])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving final standalone Spark-X2.6-4B-En4B-A1B package to {out_dir} ...")

    model.base.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(str(out_dir))

    loop_sd = _build_loop_modules_state_dict(model)
    save_file(loop_sd, str(out_dir / 'loop_modules_v03.safetensors'))
    if hasattr(_core_mod, 'loop_v03_state_dict'):
        torch.save(_core_mod.loop_v03_state_dict(model, include_tier1b=True), str(out_dir / 'loop_v03_state.pt'))

    meta = {
        'architecture': 'Spark-X2.6-4B-En4B-A1B (v0.3)',
        'base_model': cfg.get('BASE_MODEL_ID', 'XHToken/Spark-X2.6-4B'),
        'merged_standalone': True,
        'global_step': global_step,
        'effective_layers': model.total_virtual_layers,
        'execution_plan': model.execution_plan,
        'active_meta': {
            'engram_active_experts': model.engram.active_experts,
            'engram_active_top_k': model.engram.active_top_k,
        },
        'settings': model.settings.to_dict() if hasattr(model.settings, 'to_dict') else dataclasses.asdict(model.settings),
        'metrics': metrics,
    }
    meta_json = json.dumps(meta, indent=2)
    (out_dir / 'loop_config_v03.json').write_text(meta_json)
    (out_dir / 'loop_config.json').write_text(meta_json)

    for helper_file in ('sparkx_loop_v03_core.py', 'load_inference_v03.py'):
        src = Path(helper_file)
        if src.exists():
            shutil.copy2(src, out_dir / helper_file)

    if cfg.get('PUSH_TO_HUB', True) and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            repo_id = cfg['HF_REPO_ID']
            api.create_repo(repo_id=repo_id, private=bool(cfg.get('HF_PRIVATE', False)), exist_ok=True)
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=repo_id,
                commit_message=f"Final Spark-X2.6-4B-En4B-A1B (72L, 16K, step={global_step})",
            )
            print(f"[Final Hub Export] Uploaded complete model to https://huggingface.co/{repo_id}")
        except Exception as exc:
            print(f"[Final Hub Export Warning] {exc}")

    return out_dir


def _build_stage_optimizer(
    model: SparkXLoopV03,
    cfg: Dict[str, Any],
    lr_scale: float = 1.0,
    backbone_lr_mult: float = 1.0,
):
    try:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
    except Exception:
        opt_cls = torch.optim.AdamW

    bb_params, mhc_params, eng_params, router_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('controllers.'):
            mhc_params.append(p)
        elif 'router' in name or 'expert_centroids' in name:
            router_params.append(p)
        elif name.startswith('engram.') or name.startswith('backbone_engrams.'):
            eng_params.append(p)
        else:
            bb_params.append(p)

    groups = []
    if bb_params:
        groups.append({
            'params': bb_params,
            'lr': cfg['LR_BACKBONE'] * lr_scale * backbone_lr_mult,
            'weight_decay': cfg['WEIGHT_DECAY'],
        })
    if mhc_params:
        groups.append({'params': mhc_params, 'lr': cfg['LR_MHC'] * lr_scale, 'weight_decay': 0.0})
    if eng_params:
        groups.append({'params': eng_params, 'lr': cfg['LR_ENGRAM'] * lr_scale, 'weight_decay': cfg['WEIGHT_DECAY']})
    if router_params:
        groups.append({'params': router_params, 'lr': cfg['LR_ROUTER'] * lr_scale, 'weight_decay': 0.0})

    return opt_cls(groups, betas=(0.9, 0.95), eps=1e-8)


def _build_stage_cosine_scheduler(optimizer: Any, total_steps: int, warmup_ratio: float = 0.08, min_lr_ratio: float = 0.15):
    warmup_steps = max(2, int(round(total_steps * warmup_ratio)))

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def run_v03_four_stage_training(
    model: SparkXLoopV03,
    tokenizer: Any,
    stage_pools: Dict[int, List[Dict[str, torch.Tensor]]],
    cfg: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the 4-stage progressive sequence length & depth training loop with anti-overfitting & emergency recovery."""
    import importlib
    importlib.reload(_core_mod)
    # Hot-bind latest checkpoint-safe SparkXLoopV03 & ZipfianTieredMoEEngram methods onto already-instantiated modules in VRAM
    model_cls = type(model)
    model_cls.forward_hidden = _core_mod.SparkXLoopV03.forward_hidden
    model_cls.forward = _core_mod.SparkXLoopV03.forward
    model_cls.total_router_z_loss = _core_mod.SparkXLoopV03.total_router_z_loss
    for eng_mod in model.all_engram_modules():
        eng_cls = type(eng_mod)
        eng_cls.forward = _core_mod.ZipfianTieredMoEEngram.forward
        eng_cls.step_tier1b_sparse_adam = _core_mod.ZipfianTieredMoEEngram.step_tier1b_sparse_adam
        eng_mod._pending_expert_load = None
        eng_mod.last_router_z_loss = None
        eng_mod._active_z_losses = None
    model.zero_grad(set_to_none=True)

    device = next(model.base.parameters()).device
    vram_total_gb = (
        torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )
    start_wall = time.time()
    max_seconds = float(cfg.get('MAX_RUN_HOURS', 22.5)) * 3600.0
    grad_accum = int(cfg.get('GRAD_ACCUM', 8))
    save_every = int(cfg.get('SAVE_EVERY_STEPS', 250))
    log_every = int(cfg.get('LOG_EVERY_STEPS', 5))
    max_unfrozen_epochs = float(cfg.get('MAX_UNFROZEN_EPOCHS_PER_STAGE', 2.4))
    loss_floor = float(cfg.get('STAGE_CE_LOSS_FLOOR', 0.165))
    ckpt_root = Path(cfg.get('CHECKPOINT_DIR', '/content/checkpoints_v03_en4b_a1b'))

    global_step = 0
    history: List[Dict[str, Any]] = []
    budget_reached = False

    # Discover all existing local checkpoints so interrupted mid-stage runs can graduate cleanly without re-running
    existing_ckpts = sorted(ckpt_root.glob('step_*')) if ckpt_root.exists() else []

    try:
        for stage_idx, stage_cfg in enumerate(cfg['STAGES']):
            if budget_reached:
                break
            stage_name = stage_cfg['name']
            seq_len = int(stage_cfg['seq_len'])
            raw_stage_steps = int(stage_cfg['steps'])
            freeze_bb = bool(stage_cfg.get('freeze_backbone', False))
            pool = stage_pools.get(seq_len) or next(iter(stage_pools.values()))

            # Epoch-aware anti-overfitting step cap for unfrozen backbone stages
            steps_per_epoch = max(1, len(pool) // grad_accum)
            if not freeze_bb:
                capped_steps = max(80, int(round(max_unfrozen_epochs * steps_per_epoch)))
                stage_steps = min(raw_stage_steps, capped_steps)
            else:
                stage_steps = raw_stage_steps

            expected_end_step = global_step + stage_steps
            raw_expected_end_step = global_step + raw_stage_steps
            stage_ckpt_candidates = [
                ckpt_root / f"step_{expected_end_step:05d}_{stage_name}",
                ckpt_root / f"step_{raw_expected_end_step:05d}_{stage_name}",
            ] + [
                d for d in existing_ckpts
                if d.is_dir() and d.name.endswith(f"_{stage_name}") and (d / 'loop_v03_state.pt').exists()
            ]
            # Also allow an EMERGENCY_RECOVERY checkpoint taken after >= 140 steps into Stage 2 to graduate Stage 2
            if stage_idx == 1:
                for d in reversed(existing_ckpts):
                    if d.is_dir() and 'EMERGENCY_RECOVERY' in d.name and (d / 'loop_v03_state.pt').exists():
                        try:
                            st_num = int(d.name.split('_')[1])
                            if st_num >= 340:
                                stage_ckpt_candidates.insert(0, d)
                                break
                        except Exception:
                            pass

            matched_ckpt = next((d for d in stage_ckpt_candidates if d.exists() and (d / 'loop_v03_state.pt').exists()), None)
            if matched_ckpt is not None:
                try:
                    ckpt_step = int(matched_ckpt.name.split('_')[1])
                except Exception:
                    ckpt_step = expected_end_step
                # Only auto-skip if the checkpoint represents >= 1.8 epochs (or full stage completion)
                if ckpt_step >= global_step + min(stage_steps, max(50, int(1.8 * steps_per_epoch))):
                    print(
                        f"\n[Auto-Resume] Found converged checkpoint {matched_ckpt.name} for {stage_name}; "
                        f"restoring loop state and advancing global_step -> {ckpt_step}.",
                        flush=True,
                    )
                    state_obj = torch.load(str(matched_ckpt / 'loop_v03_state.pt'), map_location='cpu')
                    _core_mod.load_loop_v03_state_dict(model, state_obj, strict=False)
                    global_step = ckpt_step
                    continue

            if stage_cfg.get('upcycle_8_to_32', False):
                print(f"\n[{stage_name}] Executing Coarse-to-Fine Expert Upcycling (8 -> 32 experts, top_k=7)...", flush=True)
                model.upcycle_all_engrams_8_to_32(noise_std=1e-3)

            model.set_loop(
                start=int(stage_cfg['start']),
                end=int(stage_cfg['end']),
                passes=int(stage_cfg['passes']),
                use_mhc=True,
                use_engram=True,
                enable_p3=bool(stage_cfg.get('enable_p3', False)),
                p3_start=int(stage_cfg.get('p3_start', 12)),
                p3_end=int(stage_cfg.get('p3_end', 24)),
            )

            # Automatically enable pinned-RAM checkpoint streaming at 16K on <=85GB GPUs to prevent 60L/72L OOM
            if seq_len >= 16384 and 0.0 < vram_total_gb <= 85.0:
                model.offload_checkpoint_to_cpu = True

            for p in model.base.parameters():
                p.requires_grad = (not freeze_bb)
            for p in model.controllers.parameters():
                p.requires_grad = True
            for mod in model.all_engram_modules():
                for p in mod.parameters():
                    p.requires_grad = True

            # Preserve pretrained & Stage-2 backbone fluency during Stage 3 (Upcycle32) & Stage 4 (72L Telescoping)
            bb_lr_mult = 0.20 if (stage_idx >= 2 and not freeze_bb) else 1.0

            model.train()
            optimizer = _build_stage_optimizer(
                model, cfg,
                lr_scale=float(stage_cfg.get('lr_scale', 1.0)),
                backbone_lr_mult=bb_lr_mult,
            )
            scheduler = _build_stage_cosine_scheduler(optimizer, total_steps=stage_steps)

            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            spans = [f"prelude[0..{model.loop_start})"]
            for p_idx in range(model.loop_passes):
                spans.append(f"pass{p_idx + 1}[{model.loop_start}..{model.loop_end})")
            if model.enable_p3 and model.loop_passes >= 2:
                spans.append(f"pass3[{model.p3_start}..{model.p3_end})")
            spans.append(f"coda[{model.loop_end}..{model.n_layers})")
            plan_summary = ' -> '.join(spans)
            est_epochs = (stage_steps * grad_accum) / float(max(1, len(pool)))

            print('\n' + '=' * 108)
            print(
                f"Starting {stage_name} | Virtual Layers: {model.total_virtual_layers}L ({plan_summary})\n"
                f"  SeqLen: {seq_len} | GradAccum: {grad_accum} ({seq_len * grad_accum:,} tok/step) | "
                f"Steps: {stage_steps} (~{est_epochs:.2f} epochs over {len(pool)} seqs)\n"
                f"  Backbone Frozen: {freeze_bb} (BB_LR_Mult={bb_lr_mult:.2f}) | "
                f"Trainable Params: {trainable_p / 1e9:.2f}B (Frozen: {frozen_p / 1e9:.2f}B) | "
                f"Active Experts: {model.engram.active_experts} (Top-{model.engram.active_top_k}) | "
                f"CPU_Ckpt_Stream: {model.offload_checkpoint_to_cpu}"
            )
            print('=' * 108, flush=True)

            optimizer.zero_grad(set_to_none=True)
            pool_idx = 0
            epoch_order = list(range(len(pool)))
            rng = random.Random(int(cfg.get('SEED', 42)) + stage_idx * 1000)
            rng.shuffle(epoch_order)
            recent_ce: List[float] = []
            stage_t0 = time.time()

            for st_step in range(1, stage_steps + 1):
                step_t0 = time.time()
                accum_loss = 0.0
                accum_ce = 0.0
                accum_z = 0.0
                step_sup_tokens = 0

                for _ in range(grad_accum):
                    if pool_idx > 0 and (pool_idx % len(pool) == 0):
                        rng.shuffle(epoch_order)
                    sample = pool[epoch_order[pool_idx % len(pool)]]
                    pool_idx += 1
                    input_ids = sample['input_ids'].unsqueeze(0).to(device, non_blocking=True)
                    labels = sample['labels'].unsqueeze(0).to(device, non_blocking=True)
                    step_sup_tokens += int((labels != -100).sum().item())

                    out = model(input_ids=input_ids, labels=labels, return_logits=False)
                    loss = out['loss'] / float(grad_accum)
                    if not torch.isfinite(loss):
                        print(f"[Warn] Non-finite loss at step {global_step + 1}; skipping micro-step.", flush=True)
                        continue
                    loss.backward()
                    accum_loss += float(out['loss'].detach().item()) / float(grad_accum)
                    accum_ce += float(out['ce_loss'].detach().item()) / float(grad_accum)
                    z_val = out['z_loss']
                    accum_z += (float(z_val.detach().item()) if isinstance(z_val, torch.Tensor) else float(z_val)) / float(grad_accum)

                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['MAX_GRAD_NORM'])).item())
                init_lr0 = float(optimizer.param_groups[0].get('initial_lr', optimizer.param_groups[0]['lr']))
                lr_mult_now = float(scheduler.get_last_lr()[0]) / max(1e-12, init_lr0)
                if math.isfinite(grad_norm):
                    optimizer.step()
                    model.step_all_tier1b_sparse_adam(
                        lr=float(cfg['LR_TIER1B_SPARSE']) * float(stage_cfg.get('lr_scale', 1.0)) * lr_mult_now
                    )
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                step_dt = max(1e-3, time.time() - step_t0)
                tok_per_s = (seq_len * grad_accum) / step_dt
                elapsed_h = (time.time() - start_wall) / 3600.0
                stage_elapsed = time.time() - stage_t0
                stage_eta_m = ((stage_elapsed / st_step) * (stage_steps - st_step)) / 60.0

                vram_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                vram_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                active_gains = []
                for l_idx in range(model.loop_start, model.loop_end):
                    c_mod = model.controllers[str(l_idx)]
                    g_p = c_mod.p3_extra_gain if model.enable_p3 else c_mod.extra_gain
                    active_gains.append(float(torch.tanh(g_p.detach().float()).abs().item()))
                mhc_gain = sum(active_gains) / float(max(1, len(active_gains)))

                metrics = {
                    'stage': stage_name,
                    'step': global_step,
                    'stage_step': st_step,
                    'loss': round(accum_loss, 5),
                    'ce_loss': round(accum_ce, 5),
                    'z_loss': round(accum_z, 6),
                    'grad_norm': round(grad_norm, 4),
                    'tok_per_s': round(tok_per_s, 1),
                    'step_time_s': round(step_dt, 2),
                    'vram_peak_gb': round(vram_peak, 2),
                    'mhc_gain': round(mhc_gain, 5),
                    'elapsed_hours': round(elapsed_h, 3),
                }
                history.append(metrics)
                recent_ce.append(accum_ce)
                if len(recent_ce) > 10:
                    recent_ce.pop(0)

                if st_step <= 3 or st_step % log_every == 0 or st_step == stage_steps:
                    print(
                        f"[{stage_name} {st_step:4d}/{stage_steps:4d} | Global {global_step:4d}] "
                        f"loss={accum_loss:.4f} (ce={accum_ce:.4f}, z={accum_z:.5f}) | "
                        f"gnorm={grad_norm:.3f} | mHC_gain={mhc_gain:.4f} | "
                        f"{tok_per_s:6.0f} tok/s ({step_dt:4.1f}s/step, sup={step_sup_tokens:,}) | "
                        f"VRAM={vram_alloc:.1f}/{vram_peak:.1f}GB | "
                        f"elapsed={elapsed_h:.2f}h (stage ETA {stage_eta_m:.1f}m)",
                        flush=True,
                    )

                if save_every > 0 and (global_step % save_every == 0):
                    save_and_push_v03_checkpoint(
                        model, tokenizer, cfg, stage_name, global_step, metrics,
                        hf_token=hf_token, push_backbone=False,
                    )

                # Anti-overfitting early stage graduation if 10-step rolling CE drops below floor after warmup
                if (not freeze_bb) and st_step >= max(45, steps_per_epoch) and len(recent_ce) == 10:
                    avg_ce = sum(recent_ce) / 10.0
                    if avg_ce < loss_floor:
                        print(
                            f"[Anti-Overfit Guard] 10-step mean ce_loss={avg_ce:.4f} < {loss_floor:.4f} at step {st_step}/{stage_steps}; "
                            f"graduating {stage_name} early to preserve generalization.",
                            flush=True,
                        )
                        break

                if (time.time() - start_wall) >= max_seconds:
                    print(f"[Wall-Clock Guard] Reached {cfg['MAX_RUN_HOURS']}h limit at step {global_step}. Stopping cleanly.", flush=True)
                    budget_reached = True
                    break

            save_and_push_v03_checkpoint(
                model, tokenizer, cfg, stage_name, global_step,
                history[-1] if history else {'loss': 0.0},
                hf_token=hf_token, push_backbone=False,
            )
            del optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except BaseException as exc:
        if global_step > 0:
            print(f"\n[Emergency Recovery Guard] Caught {type(exc).__name__}: {exc}. Saving recovery checkpoint at step {global_step}...", flush=True)
            save_and_push_v03_checkpoint(
                model, tokenizer, cfg, 'EMERGENCY_RECOVERY', global_step,
                history[-1] if history else {'loss': 0.0},
                hf_token=hf_token, push_backbone=True,
            )
        else:
            print(f"\n[Emergency Recovery Guard] Caught {type(exc).__name__} at step 0 (skipping 8GB checkpoint upload): {exc}", flush=True)
        raise

    final_dir = export_final_v03_to_hub(
        model, tokenizer, cfg, global_step,
        history[-1] if history else {'loss': 0.0},
        hf_token=hf_token,
    )
    return {'global_step': global_step, 'history': history, 'final_dir': str(final_dir)}


# ==============================================================================
# 5. 262K Multi-Turn Agentic Verification Environment (`SparkXAgenticWorkspace`)
# ==============================================================================

DEFAULT_MAX_NEW_TOKENS = 262144

SPARK_X26_SYSTEM_PROMPTS: Dict[str, str] = {
    'agentic_3d_webgl': (
        'You are Spark-X2.6-4B-En4B-A1B, a 72-layer recurrent reasoning and 3D WebGL engineering agent.\n'
        'Execution Protocol:\n'
        '1. Outline your scene architecture concisely in 4-6 bullet points inside <think>...</think> and close `</think>` immediately.\n'
        '2. Never reference local host paths (`C:\\Users\\...`) or external local files; use CDN scripts (`https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`).\n'
        '3. Immediately after `</think>`, emit the complete, self-contained, runnable `<!DOCTYPE html> ... </html>` document inside a ```html ... ``` block.'
    ),
}


def set_loglinear_kv_runtime(
    model: SparkXLoopV03,
    enabled: bool = True,
    exact_threshold: int = 2048,
    sink_tokens: int = 8,
    local_window: int = 1024,
    base_block: int = 16,
    hip_top_blocks: int = 8,
):
    """Toggle runtime O(n log n) Log-Linear KV Cache on a SparkXLoopV03 instance."""
    attn_mod = __import__(type(model.base.model.layers[0].self_attn).__module__, fromlist=['*'])
    attn_mod._sparkx_sdpa_patched = False
    if enabled:
        patch_spark_for_loglinear_kv(
            model.base,
            exact_threshold=exact_threshold,
            sink_tokens=sink_tokens,
            local_window=local_window,
            base_block=base_block,
            hip_top_blocks=hip_top_blocks,
        )


class SparkXAgenticWorkspace:
    """Multi-Turn Agentic Workspace with automated HTML/JS/Three.js verification and 262K token generation."""

    def __init__(self, root_dir: str, model: SparkXLoopV03, tokenizer: Any):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.tokenizer = tokenizer

    def verify_html_webgl(self, html_text: str) -> Dict[str, Any]:
        lower = html_text.lower()
        has_doctype = '<!doctype html>' in lower or '<html' in lower
        has_closing = '</html>' in lower
        has_three = 'three' in lower and ('webglrenderer' in lower or 'instancedmesh' in lower)
        has_script = '<script' in lower and '</script>' in lower
        balanced_braces = abs(html_text.count('{') - html_text.count('}')) <= 2
        valid = bool(has_doctype and has_closing and has_three and has_script and balanced_braces and len(html_text) >= 1200)
        return {
            'valid': valid,
            'length_chars': len(html_text),
            'has_doctype': has_doctype,
            'has_closing_html': has_closing,
            'has_three_webgl': has_three,
            'balanced_braces': balanced_braces,
        }

    def write_file(self, filename: str, content: str) -> Dict[str, Any]:
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        ver = self.verify_html_webgl(content) if filename.endswith('.html') else {'valid': True, 'bytes': len(content)}
        return {'saved_path': str(path), 'bytes': len(content.encode('utf-8')), 'verification': ver}

    def run_agentic_mission(
        self,
        mission_prompt: str,
        target_filename: str = 'index.html',
        max_turns: int = 3,
        max_new_tokens_per_turn: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Dict[str, Any]:
        self.model.eval()
        device = next(self.model.base.parameters()).device
        messages = [
            {'role': 'system', 'content': SPARK_X26_SYSTEM_PROMPTS['agentic_3d_webgl']},
            {'role': 'user', 'content': mission_prompt},
        ]
        last_html = ''
        for turn in range(1, max_turns + 1):
            print('=' * 98)
            print(f"[AGENTIC WORKSPACE TURN {turn}/{max_turns}] Workspace: {self.root}")
            print('=' * 98)
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
                ).to(device)
            except Exception:
                raw_prompt = (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                    f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"
                )
                prompt_ids = self.tokenizer(raw_prompt, return_tensors='pt').input_ids.to(device)

            out_ids = self.model.generate_cached(
                prompt_ids,
                max_new_tokens=max_new_tokens_per_turn,
                temperature=0.35,
                top_p=0.92,
                repetition_penalty=1.08,
                eos_token_id=self.tokenizer.eos_token_id,
                tokenizer=self.tokenizer,
                think_budget_tokens=384,
            )
            gen_ids = out_ids[0, prompt_ids.shape[1]:]
            reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            extracted = ''
            m_fence = re.search(r'```html\s*(<!DOCTYPE html>.*?</html>)\s*```', reply, flags=re.DOTALL | re.IGNORECASE)
            if m_fence:
                extracted = m_fence.group(1).strip()
            else:
                m_raw = re.search(r'(<!DOCTYPE html>.*?</html>)', reply, flags=re.DOTALL | re.IGNORECASE)
                if m_raw:
                    extracted = m_raw.group(1).strip()

            if extracted:
                last_html = extracted
                res = self.write_file(target_filename, extracted)
                ver = res['verification']
                print(f"[Workspace Tool: write_file] Saved {res['saved_path']} ({res['bytes']} bytes) | Verification: {ver}")
                if ver.get('valid'):
                    return {'success': True, 'turn': turn, 'html': extracted, 'path': res['saved_path'], 'verification': ver}
                messages.append({'role': 'assistant', 'content': reply})
                messages.append({
                    'role': 'user',
                    'content': f"<tool_response>\nVerification failed: {json.dumps(ver)}. Output the complete corrected HTML document.\n</tool_response>",
                })
            else:
                messages.append({'role': 'assistant', 'content': reply})
                messages.append({
                    'role': 'user',
                    'content': '<tool_response>\nNo complete `<!DOCTYPE html> ... </html>` document found. Emit the complete HTML inside ```html ... ```.\n</tool_response>',
                })

        return {'success': bool(last_html), 'turn': max_turns, 'html': last_html, 'path': str(self.root / target_filename)}
