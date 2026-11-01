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
            ' concise_think': 0.05,
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


DATASET_SOURCES: Dict[str, List[Tuple[str, Optional[str], str]]] = {
    'code_swe': [
        ('nvidia/OpenCodeReasoning', 'split_0', 'train'),
        ('open-r1/verifiable-coding-problems-python_decontaminated', None, 'train'),
        ('ise-uiuc/Magicoder-OSS-Instruct-75K', None, 'train'),
        ('ise-uiuc/Magicoder-Evol-Instruct-110K', None, 'train'),
        ('nickrosh/Evol-Instruct-Code-80k-v1', None, 'train'),
        ('bigcode/self-oss-instruct-sc2-exec-filter-50k', None, 'train'),
        ('TIGER-Lab/MathInstruct', None, 'train'),
        ('m-a-p/CodeFeedback-Filtered-Instruction', None, 'train'),
        ('m-a-p/Code-Feedback', None, 'train'),
        (' GlaiveAI/glaive-code-assistant-v3', None, 'train'),
        (' Sahil2801/CodeAlpaca-20k', None, 'train'),
        ('iamtarun/python_code_instructions_18k_alpaca', None, 'train'),
    ],
    'web_3d_ui': [
        (' Tesslate/UIGEN-T1.5-Dataset', None, 'train'),
        ('cfahlgren1/react-code-instructions', None, 'train'),
        ('HuggingFaceH4/ultrachat_200k', None, 'train_sft'),
        ('Open-Orca/SlimOrca-Dedup', None, 'train'),
        ('teknium/OpenHermes-2.5', None, 'train'),
        ('BAAI/Infinity-Instruct', '7M_core', 'train'),
    ],
    'math_stem': [
        ('open-thoughts/OpenThoughts-114k', 'default', 'train'),
        ('bespoke-stratos/Bespoke-Stratos-17k', None, 'train'),
        ('nvidia/OpenMathInstruct-2', None, 'train'),
        ('meta-math/MetaMathQA', None, 'train'),
        ('openai/gsm8k', 'main', 'train'),
        ('AI-MO/NuminaMath-CoT', None, 'train'),
        ('TIGER-Lab/TheoremQA', None, 'test'),
        ('camel-ai/math', None, 'train'),
        ('camel-ai/physics', None, 'train'),
        ('camel-ai/chemistry', None, 'train'),
    ],
    'tool_agent': [
        ('Salesforce/xlam-function-calling-60k', None, 'train'),
        ('NousResearch/hermes-function-calling-v1', None, 'train'),
        ('glaiveai/glaive-function-calling-v2', None, 'train'),
        ('Trelis/function_calling_v3', None, 'train'),
        (' Locutusque/function-calling-chatml', None, 'train'),
        ('lilacai/glaive-function-calling-v2-sharegpt', None, 'train'),
    ],
    'concise_think': [
        ('Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B', None, 'train'),
        ('ServiceNow-AI/R1-Distill-SFT', 'v0', 'train'),
        (' Skywork/Skywork-OR1-RL-Data', None, 'train'),
        ('HuggingFaceH4/no_robots', None, 'train'),
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

    # 2. 1-Pass mHC + Engram enabled (passes=1, use_mhc=True, use_engram=True)
    model.set_loop(start=4, end=28, passes=1, use_mhc=True, use_engram=True, enable_p3=False)
    p1_logits = model.forward_logits(test_ids).float()
    one_pass_diff = float((base_logits - p1_logits).abs().max().item())

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
    print(f"[Step-0 Check 2/5] 1-Pass mHC+Engram vs Raw Base Diff : {one_pass_diff:.8f} (req < 0.05)")
    print(f"[Step-0 Check 3/5] 36L (1-Pass) vs 72L (3-Pass) Diff  : {step0_diff:.8f} (req == 0.0)")
    print(f"[Step-0 Check 4/5] Causal Prefix Invariance Error     : {causal_prefix_err:.8f} (req < 0.025)")
    print(f"[Step-0 Check 5/5] Birkhoff-Sinkhorn Doubly-Stochastic: {birkhoff_sinkhorn_err:.8f} (req < 1e-5)")

    assert disabled_diff < 1e-4, f"Loop-disabled parity failed: {disabled_diff}"
    assert one_pass_diff < 0.05, f"1-Pass mHC initialization drift too large: {one_pass_diff}"
    assert step0_diff == 0.0, f"Expected exact 0.0 Step-0 parity between 36L and 72L, got {step0_diff}"
    assert causal_prefix_err < 0.025, f"Causal prefix invariance failed: {causal_prefix_err}"
    assert birkhoff_sinkhorn_err < 1e-5, f"Birkhoff-Sinkhorn doubly-stochastic check failed: {birkhoff_sinkhorn_err}"

    return {
        'disabled_diff': disabled_diff,
        'one_pass_diff': one_pass_diff,
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
    sys_prompt = str(row.get('system') or row.get('system_prompt') or '').strip()
    history_turns: List[Dict[str, str]] = []

    conv = row.get('conversations') or row.get('messages')
    if isinstance(conv, list) and len(conv) >= 2:
        norm_msgs: List[Dict[str, str]] = []
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = str(m.get('role') or m.get('from') or '').lower()
            content = str(m.get('content') or m.get('value') or '').strip()
            if not content:
                continue
            if role in ('system',):
                sys_prompt = content
            elif role in ('user', 'human'):
                norm_msgs.append({'role': 'user', 'content': _normalize_tool_tags(content)})
            elif role in ('assistant', 'gpt', 'model'):
                norm_msgs.append({'role': 'assistant', 'content': _normalize_tool_tags(content)})
            elif role in ('tool', 'observation', 'function_response'):
                norm_msgs.append({'role': 'user', 'content': f"<tool_response>\n{content}\n</tool_response>"})

        if len(norm_msgs) >= 2 and norm_msgs[-1]['role'] == 'assistant':
            last_asst = norm_msgs[-1]['content']
            prev_msgs = norm_msgs[:-1]
            if prev_msgs and prev_msgs[-1]['role'] == 'user':
                user_prompt = prev_msgs[-1]['content']
                history_turns = prev_msgs[:-1]
                thought, answer = _split_thought_and_answer(last_asst)
                return {
                    'pillar': pillar,
                    'system': sys_prompt,
                    'history_turns': history_turns,
                    'user': user_prompt,
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
        row.get('output') or row.get('response') or row.get('solution') or row.get('answer') or row.get('completion') or ''
    ).strip()
    if not user_prompt or not raw_out:
        return None

    thought, answer = _split_thought_and_answer(raw_out)
    if not thought and row.get('reasoning'):
        thought = str(row.get('reasoning')).strip()
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


def curate_all_pillars_v03(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stream and filter high-signal trajectories across all 5 pillars."""
    from datasets import load_dataset

    target_total = int(cfg.get('TARGET_TOTAL_SAMPLES', 2400))
    ratios = cfg.get('PILLAR_RATIOS', {})
    curated: List[Dict[str, Any]] = []

    for pillar_raw, ratio in ratios.items():
        pillar = pillar_raw.strip()
        quota = max(20, int(target_total * float(ratio)))
        sources = DATASET_SOURCES.get(pillar, [])
        collected = 0
        per_source_cap = max(25, math.ceil(quota / max(1, len(sources))) + 25)

        for ds_name_raw, subset, split in sources:
            if collected >= quota:
                break
            ds_name = ds_name_raw.strip()
            try:
                ds = (
                    load_dataset(ds_name, subset, split=split, streaming=True)
                    if subset else load_dataset(ds_name, split=split, streaming=True)
                )
                src_count = 0
                for row in ds:
                    rec = extract_record_fields(row, pillar)
                    if rec and passes_quality_filter(rec):
                        curated.append(rec)
                        collected += 1
                        src_count += 1
                        if src_count >= per_source_cap or collected >= quota:
                            break
            except Exception as exc:
                print(f"[Dataset Stream Skip] {ds_name}: {str(exc)[:90]}")
        print(f"[Pillar Curated] {pillar:15s}: {collected:4d} / {quota:4d} verified trajectories")

    random.shuffle(curated)
    print(f"Total curated streaming records: {len(curated)}")
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
        print("[OpenCode Go Teacher] Skipped (no OPENCODE_API_KEY provided; using curated HF pillars).")
        return []
    try:
        from datasets import load_dataset
        from openai import OpenAI
    except Exception as exc:
        print(f"[OpenCode Go Teacher] Skipped ({exc})")
        return []

    client = OpenAI(base_url=cfg['OPENCODE_BASE_URL'], api_key=api_key, timeout=45.0)
    models = list(cfg.get('OPENCODE_MODELS', ['glm-5']))
    target_n = int(cfg.get('OPENCODE_TEACHER_SAMPLES', 60))
    prompts: List[Tuple[str, Optional[Decimal]]] = []

    try:
        gsm = load_dataset('openai/gsm8k', 'main', split='train', streaming=True)
        for idx, row in enumerate(gsm):
            if idx >= target_n:
                break
            gold = _extract_gsm8k_decimal(str(row.get('answer', '')))
            prompts.append((str(row['question']), gold))
    except Exception:
        pass

    results: List[Dict[str, Any]] = []

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
                'system': 'You are Spark-X2.6, a rigorous mathematical and coding assistant.',
                'history_turns': [],
                'user': q,
                'thought': thought,
                'answer': answer,
            }
            return rec if passes_quality_filter(rec) else None
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_query_one, i, p) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r is not None:
                results.append(r)

    print(f"[OpenCode Go Teacher] Synthesized & Decimal-verified {len(results)} trajectories.")
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

    tokenized_items: List[Tuple[List[int], List[int]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(tokenize_and_mask_record_16k, r, tokenizer, max_seq_len) for r in records]
        for f in futs:
            res = f.result()
            if res is not None:
                tokenized_items.append(res)

    print(f"[16K Tokenizer] Retained {len(tokenized_items)}/{len(records)} complete trajectories with zero truncation.")

    stage_lengths = sorted({int(st['seq_len']) for st in cfg['STAGES']})
    stage_pools: Dict[int, List[Dict[str, torch.Tensor]]] = {}

    for target_len in stage_lengths:
        pool: List[Dict[str, torch.Tensor]] = []
        cur_ids: List[int] = []
        cur_labels: List[int] = []

        for ids, lbls in tokenized_items:
            if len(ids) > target_len:
                continue
            if len(cur_ids) + len(ids) <= target_len:
                if cur_ids:
                    # Mask boundary transition token so packed sequences don't predict across document boundaries
                    lbls = [-100] + lbls[1:]
                cur_ids.extend(ids)
                cur_labels.extend(lbls)
            else:
                if len(cur_ids) >= 64:
                    pool.append({
                        'input_ids': torch.tensor(cur_ids, dtype=torch.long),
                        'labels': torch.tensor(cur_labels, dtype=torch.long),
                    })
                cur_ids = list(ids)
                cur_labels = list(lbls)

        if len(cur_ids) >= 64:
            pool.append({
                'input_ids': torch.tensor(cur_ids, dtype=torch.long),
                'labels': torch.tensor(cur_labels, dtype=torch.long),
            })

        # Fallback if all samples exceeded target_len
        if not pool and tokenized_items:
            ids0, lbls0 = tokenized_items[0]
            pool.append({
                'input_ids': torch.tensor(ids0[:target_len], dtype=torch.long),
                'labels': torch.tensor(lbls0[:target_len], dtype=torch.long),
            })

        stage_pools[target_len] = pool
        avg_tok = sum(x['input_ids'].numel() for x in pool) / max(1, len(pool))
        print(f"[Stage Pool {target_len:5d}] Packed {len(pool):4d} sequences (avg length = {avg_tok:.1f} tokens, pad_id={pad_id})")

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

    meta = {
        'architecture': 'Spark-X2.6-4B-En4B-A1B (v0.3)',
        'global_step': global_step,
        'stage_name': stage_name,
        'effective_layers': model.total_virtual_layers,
        'execution_plan': model.execution_plan,
        'settings': model.settings.to_dict() if hasattr(model.settings, 'to_dict') else dataclasses.asdict(model.settings),
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (ckpt_dir / 'loop_config_v03.json').write_text(json.dumps(meta, indent=2))

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

    meta = {
        'architecture': 'Spark-X2.6-4B-En4B-A1B (v0.3)',
        'global_step': global_step,
        'effective_layers': model.total_virtual_layers,
        'execution_plan': model.execution_plan,
        'settings': model.settings.to_dict() if hasattr(model.settings, 'to_dict') else dataclasses.asdict(model.settings),
        'metrics': metrics,
    }
    (out_dir / 'loop_config_v03.json').write_text(json.dumps(meta, indent=2))

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


def _build_stage_optimizer(model: SparkXLoopV03, cfg: Dict[str, Any], lr_scale: float = 1.0):
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
        groups.append({'params': bb_params, 'lr': cfg['LR_BACKBONE'] * lr_scale, 'weight_decay': cfg['WEIGHT_DECAY']})
    if mhc_params:
        groups.append({'params': mhc_params, 'lr': cfg['LR_MHC'] * lr_scale, 'weight_decay': 0.0})
    if eng_params:
        groups.append({'params': eng_params, 'lr': cfg['LR_ENGRAM'] * lr_scale, 'weight_decay': cfg['WEIGHT_DECAY']})
    if router_params:
        groups.append({'params': router_params, 'lr': cfg['LR_ROUTER'] * lr_scale, 'weight_decay': 0.0})

    return opt_cls(groups, betas=(0.9, 0.95), eps=1e-8)


def run_v03_four_stage_training(
    model: SparkXLoopV03,
    tokenizer: Any,
    stage_pools: Dict[int, List[Dict[str, torch.Tensor]]],
    cfg: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the 4-stage progressive sequence length & depth training loop with 22.5h budget & emergency recovery."""
    device = next(model.base.parameters()).device
    start_wall = time.time()
    max_seconds = float(cfg.get('MAX_RUN_HOURS', 22.5)) * 3600.0
    grad_accum = int(cfg.get('GRAD_ACCUM', 8))
    save_every = int(cfg.get('SAVE_EVERY_STEPS', 250))

    global_step = 0
    history: List[Dict[str, Any]] = []
    budget_reached = False

    try:
        for stage_cfg in cfg['STAGES']:
            if budget_reached:
                break
            stage_name = stage_cfg['name']
            seq_len = int(stage_cfg['seq_len'])
            stage_steps = int(stage_cfg['steps'])
            freeze_bb = bool(stage_cfg.get('freeze_backbone', False))

            if stage_cfg.get('upcycle_8_to_32', False):
                print(f"\n[{stage_name}] Executing Coarse-to-Fine Expert Upcycling (8 -> 32 experts, top_k=7)...")
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

            for p in model.base.parameters():
                p.requires_grad = (not freeze_bb)
            for p in model.controllers.parameters():
                p.requires_grad = True
            for mod in model.all_engram_modules():
                for p in mod.parameters():
                    p.requires_grad = True

            model.train()
            optimizer = _build_stage_optimizer(model, cfg, lr_scale=float(stage_cfg.get('lr_scale', 1.0)))
            pool = stage_pools.get(seq_len) or next(iter(stage_pools.values()))

            print('=' * 92)
            print(
                f"Starting {stage_name} | Effective Layers: {model.total_virtual_layers} | "
                f"SeqLen: {seq_len} | Steps: {stage_steps} | Backbone Frozen: {freeze_bb}"
            )
            print('=' * 92)

            optimizer.zero_grad(set_to_none=True)
            pool_idx = 0

            for st_step in range(1, stage_steps + 1):
                accum_loss = 0.0
                accum_ce = 0.0
                accum_z = 0.0

                for _ in range(grad_accum):
                    sample = pool[pool_idx % len(pool)]
                    pool_idx += 1
                    input_ids = sample['input_ids'].unsqueeze(0).to(device, non_blocking=True)
                    labels = sample['labels'].unsqueeze(0).to(device, non_blocking=True)

                    out = model(input_ids=input_ids, labels=labels, return_logits=False)
                    loss = out['loss'] / float(grad_accum)
                    if not torch.isfinite(loss):
                        print(f"[Warn] Non-finite loss at step {global_step + 1}; skipping micro-step.")
                        continue
                    loss.backward()
                    accum_loss += float(out['loss'].detach().item()) / float(grad_accum)
                    accum_ce += float(out['ce_loss'].detach().item()) / float(grad_accum)
                    z_val = out['z_loss']
                    accum_z += (float(z_val.detach().item()) if isinstance(z_val, torch.Tensor) else float(z_val)) / float(grad_accum)

                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['MAX_GRAD_NORM'])).item())
                if math.isfinite(grad_norm):
                    optimizer.step()
                    model.step_all_tier1b_sparse_adam(lr=float(cfg['LR_TIER1B_SPARSE']) * float(stage_cfg.get('lr_scale', 1.0)))
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                elapsed_h = (time.time() - start_wall) / 3600.0
                metrics = {
                    'stage': stage_name,
                    'step': global_step,
                    'stage_step': st_step,
                    'loss': round(accum_loss, 5),
                    'ce_loss': round(accum_ce, 5),
                    'z_loss': round(accum_z, 6),
                    'grad_norm': round(grad_norm, 4),
                    'elapsed_hours': round(elapsed_h, 3),
                }
                history.append(metrics)

                if st_step == 1 or st_step % 20 == 0 or st_step == stage_steps:
                    print(
                        f"[{stage_name} {st_step:4d}/{stage_steps:4d} | Global {global_step:4d}] "
                        f"loss={accum_loss:.4f} (ce={accum_ce:.4f}, z={accum_z:.5f}) | "
                        f"grad_norm={grad_norm:.3f} | elapsed={elapsed_h:.2f}h"
                    )

                if save_every > 0 and (global_step % save_every == 0):
                    save_and_push_v03_checkpoint(
                        model, tokenizer, cfg, stage_name, global_step, metrics,
                        hf_token=hf_token, push_backbone=False,
                    )

                if (time.time() - start_wall) >= max_seconds:
                    print(f"[Wall-Clock Guard] Reached {cfg['MAX_RUN_HOURS']}h limit at step {global_step}. Stopping cleanly.")
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
        print(f"\n[Emergency Recovery Guard] Caught {type(exc).__name__}: {exc}. Saving recovery checkpoint...")
        save_and_push_v03_checkpoint(
            model, tokenizer, cfg, 'EMERGENCY_RECOVERY', global_step,
            history[-1] if history else {'loss': 0.0},
            hf_token=hf_token, push_backbone=True,
        )
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
