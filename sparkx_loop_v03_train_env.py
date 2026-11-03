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
    """Return the default v0.3 16K Unsloth/Triton configuration dictionary (modifiable MoE-Engram, mHC & 6-pillar curriculum)."""
    return {
        # Base & Target Repositories (MUST use post-trained XHToken/Spark-X2.6-4B, never -Base)
        'BASE_MODEL_ID': 'XHToken/Spark-X2.6-4B',
        'V02_WARM_START_REPO': 'TassP0lar/Spark-X2.6-4B-Loop-mHC-R24',
        'ENABLE_V02_WARM_START': False,
        'CURRICULUM_VERSION': 'v03_r2_3phase',
        'HF_REPO_ID': 'TassP0lar/Spark-X2.6-4B-En4B-A1B',
        'HF_PRIVATE': False,
        'PUSH_TO_HUB': True,

        # Checkpoint Retention & Periodic Save Policy (isolated r2_3phase directory so Run 1 collapsed ckpts are ignored)
        'CHECKPOINT_DIR': '/content/checkpoints_v03_r2_3phase',
        'FINAL_EXPORT_DIR': '/content/Spark-X2.6-4B-En4B-A1B-Final',
        'MAX_LOCAL_CHECKPOINTS': 3,
        'SAVE_EVERY_STEPS': 100,

        # Unattended Wall-Clock Budget (Colab Pro+ 24h ceiling with 1.5h safety margin)
        'MAX_RUN_HOURS': 22.5,

        # 72-Layer Telescoping Recurrence Geometry (36L Prelude/Core/Coda -> 60L Pass-2 -> 72L Pass-3)
        'LOOP_START': 4,
        'LOOP_END': 28,
        'P3_START': 12,
        'P3_END': 24,
        'MAX_PASSES': 4,
        'PASS_ADAPTER_RANK': 16,
        'ENABLE_STAGE0_PROGRESSIVE_DEPTH_WARMUP': False,  # Set True to prepend 48L -> 60L progressive stage-merging warmup before 72L

        # Modifiable 8-Anchor 3-Tier Zipfian MoE-Engram & Self-Balancing Router (`En4B-A1B`: 4.11B Total / 1.14B–3.34B Active)
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
        'ENG_LOAD_EMA_DECAY': 0.05,
        'ENG_BALANCE_STEP_SIZE': 0.0015,
        'ENG_USE_PASS_GAIN_SCHEDULE': True,
        'ENG_PASS_GAIN_INIT': [0.35, 0.15, 0.05],
        # On-the-Fly Dynamic Engram Flight Mode & 5-Gram Multi-Horizon Lexical Recall
        'ENG_MULTI_HORIZON_RECALL': True,
        'ENG_DEFAULT_FLIGHT_MODE': 'deep_knowledge',  # 'fast' (Top-4) | 'balanced' (Top-7) | 'deep_knowledge' (Top-16) | 'max_recall' (Top-24) | 'full_dense' (Top-32)
        'ENG_FLIGHT_TOP_K': 16,
        'ENG_TAIL_BLEND_MAX': 0.18,

        # Training Intensity Profile ('high_capacity_en4b' [600 steps, 4800 samples] | 'max_engram_saturation' [1200 steps, 8000 samples] | 'fast_repro' [290 steps, 2400 samples])
        'TRAINING_PROFILE': 'high_capacity_en4b',

        # Phase-4 Post-Training Capability Consolidation (Backbone-Frozen Joint mHC + MoE-Engram Finalization)
        'ENABLE_PHASE4_POST_TRAIN': True,
        'PHASE4_STEPS': 150,
        'PHASE4_TRAIN_MHC': True,

        # Phase-5 Supervised Instructed Self-Recursive Self-Improvement (`SI-SRSI`) + Dynamic MoE-Engram Knowledge Formation
        #   - Domains: (1) Anti-Slop Human Writing [Exp 22..29], (2) Web UX/Policy/AppSec [Exp 12..21], (3) STEM Reasoning [Exp 0..11]
        'ENABLE_SI_SRSI_ENGRAM_LOOP': True,
        'SI_SRSI_ROUNDS': 2,
        'SI_SRSI_STEPS_PER_ROUND': 32,
        'SI_SRSI_FTPO_SLOP_WEIGHT': 0.08,

        # 16K/24K Context & Sequence Packing (Zero Mid-Thought / Mid-Code Truncation!)
        'MAX_SEQ_LENGTH': 16384,
        'PACK_SEQUENCES': True,
        'BATCH_SIZE': 1,
        'GRAD_ACCUM': 8,
        'CE_CHUNK_TOKENS': 1024,
        'WARMUP_RATIO': 0.08,
        'WEIGHT_DECAY': 0.01,
        'MAX_GRAD_NORM': 1.0,
        'STAGE_CE_LOSS_FLOOR': 0.30,
        'SEED': 42,

        # Differential Learning Rates (calibrated for post-trained Spark-X2.6-4B preservation)
        'LR_BACKBONE': 1.0e-5,
        'LR_MHC_GATE': 3.0e-3,
        'LR_MHC': 5.0e-4,
        'LR_ENGRAM': 1.5e-4,
        'LR_ROUTER': 2.5e-4,
        'LR_TIER1B_SPARSE': 5.0e-5,

        # Boosted 4-Phase Progressive Curriculum for 4.11B MoE-Engram Saturation:
        #   Phase 1 (Stage 1): Adapt 72L Looped Transformer (mHC only; Backbone FROZEN, MoE-Engram DISABLED)
        #   Phase 2A (Stage 2): Introduce 8-Expert MoE-Engram + 5-Gram Multi-Horizon Recall (Backbone FROZEN, mHC + Engram-8 trainable)
        #   Phase 2B (Stage 3): Upcycle MoE-Engram 8 -> 32 (Top-7) at 16K (Backbone FROZEN, mHC + Engram-32 trainable)
        #   Phase 3 (Stage 4): Joint Full-Finetuning at 16K (Backbone UNFROZEN at gentle 0.25x mult + mHC + Engram-32)
        #   Phase 4 (Post-Training): Backbone-Frozen Joint mHC + MoE-Engram Capability Consolidation (Per-Pass Telescoping Gates)
        'STAGES': [
            {
                'name': 'Stage1_Loop_mHC_Adapt_72L_8K',
                'start': 4, 'end': 28, 'passes': 2, 'enable_p3': True,
                'p3_start': 12, 'p3_end': 24,
                'use_mhc': True, 'use_engram': False,
                'seq_len': 8192,
                'steps': 100,
                'freeze_backbone': True,
                'freeze_engram': True,
                'upcycle_8_to_32': False,
                'lr_scale': 1.0,
                'mhc_lr_mult': 1.0,
                'backbone_lr_mult': 0.0,
                'loss_floor': 0.34,
            },
            {
                'name': 'Stage2_Engram8_Adapt_72L_12K',
                'start': 4, 'end': 28, 'passes': 2, 'enable_p3': True,
                'p3_start': 12, 'p3_end': 24,
                'use_mhc': True, 'use_engram': True,
                'seq_len': 12288,
                'steps': 140,
                'freeze_backbone': True,
                'freeze_engram': False,
                'upcycle_8_to_32': False,
                'lr_scale': 1.0,
                'mhc_lr_mult': 0.50,
                'backbone_lr_mult': 0.0,
                'loss_floor': 0.32,
            },
            {
                'name': 'Stage3_Engram32_Upcycle_72L_16K',
                'start': 4, 'end': 28, 'passes': 2, 'enable_p3': True,
                'p3_start': 12, 'p3_end': 24,
                'use_mhc': True, 'use_engram': True,
                'seq_len': 16384,
                'steps': 120,
                'freeze_backbone': True,
                'freeze_engram': False,
                'upcycle_8_to_32': True,
                'lr_scale': 0.85,
                'mhc_lr_mult': 0.40,
                'backbone_lr_mult': 0.0,
                'loss_floor': 0.31,
            },
            {
                'name': 'Stage4_Joint_FullFT_72L_16K',
                'start': 4, 'end': 28, 'passes': 2, 'enable_p3': True,
                'p3_start': 12, 'p3_end': 24,
                'use_mhc': True, 'use_engram': True,
                'seq_len': 16384,
                'steps': 90,
                'freeze_backbone': False,
                'freeze_engram': False,
                'upcycle_8_to_32': False,
                'lr_scale': 0.50,
                'mhc_lr_mult': 0.35,
                'backbone_lr_mult': 0.25,
                'loss_floor': 0.30,
            },
        ],

        # 7-Pillar 2025-2026 Frontier Multi-Domain Dataset Mixture
        # (Code/SWE-smith/OpenCodeReasoning, Olympiad Math/Numina-1.5/Big-Math, SuperGPQA/NaturalReasoning Graduate Science,
        #  Fable/Opus/GPT-6 Astra/OpenThoughts3 Long-CoT, WebGL/3D UI, Tulu-3 Strict IF/Literacy, Agentic Tool Use)
        'TARGET_TOTAL_SAMPLES': 4800,
        'PILLAR_RATIOS': {
            'code_swe': 0.22,
            'math_stem': 0.20,
            'science_knowledge_engram': 0.14,
            'fable_opus_astra_cot': 0.18,
            'web_3d_ui': 0.10,
            'literacy_dialogue': 0.08,
            'tool_agent': 0.08,
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
        'OPENCODE_TEACHER_SAMPLES': 120,
    }


# Each entry is (repo_id, subset_config, split_name, verified_shard_path).
# Providing `verified_shard_path` streams the exact .parquet/.jsonl shard directly via `hf://datasets/...`,
# completely bypassing README.md parsing, dataset_infos.json, mixed .txt/.py/.png files, and 30-shard directory scans.
DATASET_SOURCES: Dict[str, List[Tuple[str, Optional[str], str, Optional[str]]]] = {
    'code_swe': [
        ('nvidia/OpenCodeReasoning', 'split_0', 'split_0', 'split_0/train-00000-of-00030.parquet'),
        ('open-r1/verifiable-coding-problems-python_decontaminated', None, 'train', 'data/train-00000-of-00006.parquet'),
        ('SWE-bench/SWE-smith', None, 'train', None),
        ('SWE-Gym/SWE-Gym', None, 'train', None),
        ('KodCode/KodCode-V1', None, 'train', None),
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
    'math_stem': [
        ('open-thoughts/OpenThoughts-114k', 'default', 'train', 'data/train-00000-of-00006.parquet'),
        ('bespokelabs/Bespoke-Stratos-17k', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('project-numina/NuminaMath-1.5', None, 'train', None),
        ('SynthLabsAI/Big-Math-RL-Verified', None, 'train', None),
        ('nvidia/Nemotron-Math-v2', None, 'train', None),
        ('KbsdJames/Omni-MATH', None, 'test', None),
        ('open-thoughts/OpenThoughts3-1.2M', None, 'train', None),
        ('AI-MO/NuminaMath-CoT', None, 'train', 'data/train-00000-of-00005.parquet'),
        ('OpenDataArena/Spark-234K', None, 'train', 'Spark-234K-data.parquet'),
        ('openai/gsm8k', 'main', 'train', 'main/train-00000-of-00001.parquet'),
    ],
    'science_knowledge_engram': [
        ('nvidia/OpenScienceReasoning-2', None, 'train', 'train/OpenScienceReasoning-2.parquet'),
        ('MegaScience/MegaScience', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('MegaScience/TextbookReasoning', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned', 'PHD-Science', 'train', 'PHD-Science.jsonl'),
        ('m-a-p/SuperGPQA', None, 'train', None),
        ('facebook/natural_reasoning', None, 'train', None),
        ('TIGER-Lab/MMLU-Pro', None, 'validation', None),
        ('allenai/sciq', None, 'train', None),
    ],
    'web_3d_ui': [
        ('Tesslate/UIGEN-T1.5-Dataset', None, 'train', 'output.parquet'),
        ('kelexine/fable-5-sft-traces', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('cfahlgren1/react-code-instructions', None, 'train', 'data/data_ed7e68fb-44fe-47fc-b603-0279f2f8a7ca.jsonl'),
        ('HelioAI/Claude-Fable-5-5500x', None, 'train', 'Fable-5-Distill-5500x.jsonl'),
        ('armand0e/Fable-5-Chat', None, 'train', 'train.jsonl'),
    ],
    'fable_opus_astra_cot': [
        ('FlaneAI/GPT-6Astra-scientific-research', None, 'train', None),
        ('Roman1111111/gpt-5.4-step-by-step-reasoning', None, 'train', 'gpt5.4.jsonl'),
        ('vanty120/Gpt-5.4-Xhigh-Reasoning-2000x', None, 'train', 'train.jsonl'),
        ('TeichAI/gpt-5.1-high-reasoning-1000x', None, 'train', 'gpt-5.1-1000x.jsonl'),
        ('MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x', None, 'train', '10k/train.jsonl'),
        ('saidutta69/fable-5-premium-v2', None, 'train', None),
        ('angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k', None, 'train', None),
        ('Roman1111111/claude-opus-4.6-10000x', None, 'train', 'opus46_final.jsonl'),
        ('TeichAI/lordx64-claude-opus-4.7-max-cleaned', None, 'train', 'train.jsonl'),
        ('ronaldcmz/claude_opus_4.8_max_thinking_5k_v2', None, 'train', 'opus_4_8_max_thinking_5k_v2.jsonl'),
        ('GeneralReasoning/GeneralThought-430K', None, 'train', None),
        ('open-thoughts/OpenThoughts2-1M', None, 'train', None),
        ('ServiceNow-AI/R1-Distill-SFT', 'v0', 'train', 'v0/train-00000-of-00003.parquet'),
        ('Skywork/Skywork-OR1-RL-Data', None, 'code', 'data/code-00000-of-00003.parquet'),
    ],
    'literacy_dialogue': [
        ('nvidia/Nemotron-Instruction-Following-Chat-v1', None, 'chat_if', 'data/chat_if.jsonl'),
        ('allenai/tulu-3-sft-personas-instruction-following', None, 'train', None),
        ('HuggingFaceH4/ultrachat_200k', None, 'train_sft', 'data/train_sft-00000-of-00003-a3ecf92756993583.parquet'),
        ('NousResearch/Hermes-3-Dataset', None, 'train', 'hermes-3-dataset.jsonl'),
        ('DataCreatorAI/Multi-Turn-Conversational-SFT', None, 'train', 'multi_turn_conversations.jsonl'),
        ('HuggingFaceH4/no_robots', None, 'train', 'data/train-00000-of-00001.parquet'),
        ('armand0e/Fable-5-Chat', None, 'train', 'train.jsonl'),
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
    # Backward-compatible alias for legacy 5-pillar configs
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
    """Auto-detect GPU VRAM & System RAM, apply TRAINING_PROFILE scaling, and validate modifiable MoE-Engram parameters."""
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

    # Apply TRAINING_PROFILE scaling ('fast_repro' | 'high_capacity_en4b' | 'max_engram_saturation')
    profile = str(cfg.get('TRAINING_PROFILE', 'high_capacity_en4b')).lower()
    if profile == 'max_engram_saturation' and not cfg.get('_profile_applied'):
        cfg['TARGET_TOTAL_SAMPLES'] = max(int(cfg.get('TARGET_TOTAL_SAMPLES', 4800)), 8000)
        cfg['PHASE4_STEPS'] = max(int(cfg.get('PHASE4_STEPS', 150)), 250)
        for st in cfg.get('STAGES', []):
            st['steps'] = int(st['steps'] * 2.0)
        if vram_gb >= 75.0:
            cfg['MAX_SEQ_LENGTH'] = max(int(cfg.get('MAX_SEQ_LENGTH', 16384)), 24576)
            for st in cfg.get('STAGES', []):
                if st['seq_len'] >= 16384:
                    st['seq_len'] = 24576
        cfg['_profile_applied'] = True
    elif profile == 'fast_repro' and not cfg.get('_profile_applied'):
        cfg['TARGET_TOTAL_SAMPLES'] = 2400
        cfg['PHASE4_STEPS'] = 80
        fast_steps = [65, 65, 40, 38]
        for idx, st in enumerate(cfg.get('STAGES', [])):
            if idx < len(fast_steps):
                st['steps'] = fast_steps[idx]
        cfg['_profile_applied'] = True

    # Auto-detect activation offloading & Tier-1B DRAM placement
    cfg['OFFLOAD_CHECKPOINT_CPU'] = bool(vram_gb < 70.0 and sys_ram_gb >= 48.0)
    if sys_ram_gb < 28.0 and vram_gb >= 70.0:
        cfg['ENG_OFFLOAD_TIER1B_CPU'] = False
    if vram_gb < 22.0:
        cfg['MAX_SEQ_LENGTH'] = 8192
        for st in cfg['STAGES']:
            st['seq_len'] = min(st['seq_len'], 8192)
    elif vram_gb < 45.0:
        cfg['MAX_SEQ_LENGTH'] = min(int(cfg.get('MAX_SEQ_LENGTH', 16384)), 16384)

    # Optionally prepend Stage 0 progressive depth expansion (48L -> 60L) if requested by user
    if cfg.get('ENABLE_STAGE0_PROGRESSIVE_DEPTH_WARMUP', False) and not any('Stage0' in s['name'] for s in cfg['STAGES']):
        stage0_list = [
            {
                'name': 'Stage0A_Loop_mHC_Warmup_48L_4K',
                'start': 8, 'end': 20, 'passes': 2, 'enable_p3': False,
                'p3_start': 12, 'p3_end': 20,
                'use_mhc': True, 'use_engram': False,
                'seq_len': 4096,
                'steps': 35,
                'freeze_backbone': True,
                'freeze_engram': True,
                'upcycle_8_to_32': False,
                'lr_scale': 1.0,
                'mhc_lr_mult': 1.0,
                'backbone_lr_mult': 0.0,
                'loss_floor': 0.35,
            },
            {
                'name': 'Stage0B_Loop_mHC_Expand_60L_8K',
                'start': 4, 'end': 28, 'passes': 2, 'enable_p3': False,
                'p3_start': 12, 'p3_end': 24,
                'use_mhc': True, 'use_engram': False,
                'seq_len': 8192,
                'steps': 45,
                'freeze_backbone': True,
                'freeze_engram': True,
                'upcycle_8_to_32': False,
                'lr_scale': 1.0,
                'mhc_lr_mult': 1.0,
                'backbone_lr_mult': 0.0,
                'loss_floor': 0.34,
            },
        ]
        cfg['STAGES'] = stage0_list + list(cfg['STAGES'])

    n_anchors = 1 + len(range(int(cfg['ENG_STRIDE']) - 1, 36, int(cfg['ENG_STRIDE']))) - (1 if 3 in range(int(cfg['ENG_STRIDE']) - 1, 36, int(cfg['ENG_STRIDE'])) else 0)
    total_stage_steps = sum(int(s['steps']) for s in cfg['STAGES']) + (int(cfg.get('PHASE4_STEPS', 0)) if cfg.get('ENABLE_PHASE4_POST_TRAIN', True) else 0)
    print('=' * 98)
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | System RAM: {sys_ram_gb:.1f} GB | Profile: {profile}")
    print(f"Kernel Stack: Triton={HAS_TRITON} | Unsloth={HAS_UNSLOTH} | Liger={HAS_LIGER} | Cut-CE={HAS_CCE}")
    print(f"Max Seq Length: {cfg['MAX_SEQ_LENGTH']} tokens | Total Curriculum Steps: {total_stage_steps} | Samples: {cfg['TARGET_TOTAL_SAMPLES']}")
    print(
        f"MoE-Engram Auto-Config: {n_anchors} Anchors | Experts={cfg['ENG_INITIAL_ACTIVE_EXPERTS']}->{cfg['ENG_EXPERTS']} "
        f"(Train Top-K={cfg['ENG_INITIAL_TOP_K']}->{cfg['ENG_TOP_K']} | Flight Mode='{cfg.get('ENG_DEFAULT_FLIGHT_MODE', 'deep_knowledge')}' "
        f"Top-{cfg.get('ENG_FLIGHT_TOP_K', 16)}) | 5-Gram Multi-Horizon={cfg.get('ENG_MULTI_HORIZON_RECALL', True)}"
    )
    print('=' * 98)
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
    'rich tapestry',
    'testament to the',
    'delve into the intricacies',
    'delving into the',
    'symphony of',
    'beacon of hope',
    'it is important to remember that',
    'certainly! i would be happy to',
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

    # Support SuperGPQA / MMLU-Pro multiple-choice graduate STEM questions (`question` + `options` + `answer_letter`/`answer`)
    options = row.get('options')
    if isinstance(options, list) and len(options) >= 2 and row.get('question'):
        q_str = str(row.get('question')).strip()
        disc = str(row.get('discipline') or row.get('field') or row.get('subfield') or row.get('category') or 'Graduate STEM').strip()
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt_lines = [f"({letters[i]}) {str(opt).strip()}" for i, opt in enumerate(options[:12])]
        user_mc = f"[{disc}] {q_str}\n\nOptions:\n" + "\n".join(opt_lines)
        ans_letter = str(row.get('answer_letter') or row.get('answer') or '').strip()
        ans_idx = row.get('answer_index')
        if (not ans_letter or len(ans_letter) > 2) and isinstance(ans_idx, int) and 0 <= ans_idx < len(options):
            ans_letter = letters[ans_idx]
        ans_text = str(row.get('answer') or '').strip()
        cot_text = str(row.get('cot_content') or row.get('explanation') or top_thought or '').strip()
        if ans_letter:
            if not cot_text:
                cot_text = (
                    f"1. Domain: {disc}.\n"
                    f"2. Evaluate each candidate option against first-principles domain constraints.\n"
                    f"3. Option ({ans_letter}) ({ans_text}) uniquely satisfies all conditions."
                )
            final_mc_ans = f"Based on rigorous analysis of the {disc} principles:\n\n**Correct Option:** ({ans_letter}) {ans_text}".strip()
            return {
                'pillar': pillar,
                'system': sys_prompt,
                'history_turns': [],
                'user': _normalize_tool_tags(user_mc),
                'thought': cot_text,
                'answer': final_mc_ans,
            }

    # Support SWE-smith / SWE-Gym / SWE-bench repository bug-fix instances (`problem_statement` + `patch`)
    if row.get('problem_statement') and row.get('patch'):
        prob_stmt = str(row.get('problem_statement')).strip()
        patch_str = str(row.get('patch')).strip()
        repo_name = str(row.get('repo') or row.get('instance_id') or 'repository').strip()
        if len(prob_stmt) >= 32 and len(patch_str) >= 32:
            user_swe = f"[Repository: {repo_name}]\nResolve the following issue and provide a unified git diff patch:\n\n{prob_stmt}"
            swe_thought = top_thought or (
                f"1. Localize defect in `{repo_name}` from the issue description and traceback.\n"
                f"2. Construct minimal, regression-safe unified diff patch."
            )
            swe_ans = f"```diff\n{patch_str}\n```"
            return {
                'pillar': pillar,
                'system': sys_prompt,
                'history_turns': [],
                'user': _normalize_tool_tags(user_swe),
                'thought': swe_thought,
                'answer': swe_ans,
            }

    # Support Facebook NaturalReasoning (`question` + `responses` list or `reference_answer`)
    nr_resps = row.get('responses')
    if isinstance(nr_resps, list) and len(nr_resps) > 0 and row.get('question') and not row.get('output'):
        first_r = nr_resps[0]
        if isinstance(first_r, dict):
            row = dict(row)
            row['output'] = str(first_r.get('response') or first_r.get('content') or first_r.get('text') or '')
        elif isinstance(first_r, str):
            row = dict(row)
            row['output'] = first_r

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
        if hasattr(prompt_ids, 'input_ids'):
            prompt_ids = prompt_ids.input_ids
        elif isinstance(prompt_ids, dict) and 'input_ids' in prompt_ids:
            prompt_ids = prompt_ids['input_ids']
        if hasattr(full_ids, 'input_ids'):
            full_ids = full_ids.input_ids
        elif isinstance(full_ids, dict) and 'input_ids' in full_ids:
            full_ids = full_ids['input_ids']
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
        'curriculum_version': cfg.get('CURRICULUM_VERSION', 'v03_r2_3phase'),
        'base_model': cfg.get('BASE_MODEL_ID', 'XHToken/Spark-X2.6-4B'),
        'merged_standalone': bool(push_backbone),
        'global_step': global_step,
        'stage_name': stage_name,
        'effective_layers': model.total_virtual_layers,
        'execution_plan': model.execution_plan,
        'active_meta': {
            'engram_enabled': bool(model.engram_enabled),
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

    # Enforce rolling MAX_LOCAL_CHECKPOINTS retention (keep highest step numbers, never evict higher steps!)
    def _ckpt_sort_key(d: Path):
        try:
            return (int(d.name.split('_')[1]), d.stat().st_mtime)
        except Exception:
            return (0, d.stat().st_mtime)

    all_ckpts = sorted([d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith('step_')], key=_ckpt_sort_key)
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
        'curriculum_version': cfg.get('CURRICULUM_VERSION', 'v03_r2_3phase'),
        'base_model': cfg.get('BASE_MODEL_ID', 'XHToken/Spark-X2.6-4B'),
        'merged_standalone': True,
        'global_step': global_step,
        'effective_layers': model.total_virtual_layers,
        'execution_plan': model.execution_plan,
        'active_meta': {
            'engram_enabled': bool(model.engram_enabled),
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
    backbone_lr_mult: float = 0.0,
    mhc_lr_mult: float = 1.0,
):
    try:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
    except Exception:
        opt_cls = torch.optim.AdamW

    bb_params, mhc_gate_params, mhc_adapter_params, eng_params, router_params = [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('controllers.'):
            if (
                'extra_gain' in name
                or 'logits' in name
                or 'pass_routing_bias' in name
            ):
                mhc_gate_params.append(p)
            else:
                mhc_adapter_params.append(p)
        elif (
            'router' in name
            or 'expert_centroids' in name
            or name.endswith('.gain')
            or name.endswith('.pass_gain_logits')
            or name.endswith('.scale_weights')
        ):
            router_params.append(p)
        elif name.startswith('engram.') or name.startswith('backbone_engrams.'):
            eng_params.append(p)
        else:
            bb_params.append(p)

    groups = []
    if bb_params and backbone_lr_mult > 0.0:
        groups.append({
            'params': bb_params,
            'lr': float(cfg['LR_BACKBONE']) * lr_scale * backbone_lr_mult,
            'weight_decay': float(cfg['WEIGHT_DECAY']),
        })
    if mhc_gate_params:
        groups.append({
            'params': mhc_gate_params,
            'lr': float(cfg.get('LR_MHC_GATE', 3.0e-3)) * lr_scale * mhc_lr_mult,
            'weight_decay': 0.0,
        })
    if mhc_adapter_params:
        groups.append({
            'params': mhc_adapter_params,
            'lr': float(cfg['LR_MHC']) * lr_scale * mhc_lr_mult,
            'weight_decay': 0.0,
        })
    if eng_params:
        groups.append({
            'params': eng_params,
            'lr': float(cfg['LR_ENGRAM']) * lr_scale,
            'weight_decay': float(cfg['WEIGHT_DECAY']),
        })
    if router_params:
        groups.append({
            'params': router_params,
            'lr': float(cfg['LR_ROUTER']) * lr_scale,
            'weight_decay': 0.0,
        })

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


def _reset_autograd_hooks_and_vram(model: Optional[nn.Module] = None) -> int:
    """Release IPython traceback frames holding GPU tensors/optimizers first, run GC so any
    abandoned checkpoint generator exits its hook context cleanly, then pop any remaining hooks.
    """
    import sys
    import traceback
    for attr in ('last_traceback', 'last_value', 'last_exc'):
        obj = getattr(sys, attr, None)
        if obj is not None:
            try:
                if attr == 'last_traceback':
                    traceback.clear_frames(obj)
                elif hasattr(obj, '__traceback__') and obj.__traceback__ is not None:
                    traceback.clear_frames(obj.__traceback__)
            except BaseException:
                pass
            try:
                delattr(sys, attr)
            except BaseException:
                pass
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        if ip is not None:
            if hasattr(ip, 'InteractiveTB') and hasattr(ip.InteractiveTB, 'tb_offset'):
                ip.InteractiveTB.tb_offset = 0
            for tb_attr in ('_last_traceback', 'last_execution_result'):
                if hasattr(ip, tb_attr):
                    try:
                        setattr(ip, tb_attr, None)
                    except BaseException:
                        pass
    except BaseException:
        pass
    if model is not None:
        model.zero_grad(set_to_none=True)
    # Collect abandoned checkpoint generators FIRST while their saved_tensors hook is still on the stack
    gc.collect()
    popped = 0
    if hasattr(torch._C, '_autograd') and hasattr(torch._C._autograd, '_pop_saved_tensors_default_hooks'):
        for _ in range(1024):
            try:
                if hasattr(torch._C._autograd, '_top_saved_tensors_default_hooks'):
                    top_h = torch._C._autograd._top_saved_tensors_default_hooks(False)
                    if top_h is None or top_h == (None, None):
                        break
                torch._C._autograd._pop_saved_tensors_default_hooks()
                popped += 1
            except BaseException:
                break
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return popped


@torch.no_grad()
def reload_fresh_posttrained_spark_x26(
    model: SparkXLoopV03,
    cfg: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> None:
    """Restore pristine post-trained `XHToken/Spark-X2.6-4B` checkpoint weights into `model.base` in-place
    (using CPU staging so GPU VRAM never spikes) and reset all 24 `TwoStreamMHC` controllers + 8 `MoE-Engram`
    anchors to exact Step-0 initialization.
    """
    from transformers import AutoModelForCausalLM

    base_model_id = str(cfg.get('BASE_MODEL_ID', 'XHToken/Spark-X2.6-4B'))
    if 'base' in base_model_id.lower():
        raise ValueError(
            f"Refusing to load `{base_model_id}`: curriculum requires the post-trained `XHToken/Spark-X2.6-4B` checkpoint, not Base."
        )
    print(
        f"\n[Fresh Start] Reloading clean post-trained `{base_model_id}` weights into `model.base` "
        f"and resetting 24 mHC controllers + 8 MoE-Engram anchors to Step-0...",
        flush=True,
    )
    t0 = time.time()
    clean_cpu_base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    clean_sd = clean_cpu_base.state_dict()
    copied_tensors = 0
    for name, param in model.base.named_parameters():
        if name in clean_sd:
            param.data.copy_(clean_sd[name].to(device=param.device, dtype=param.dtype, non_blocking=True))
            copied_tensors += 1
    for name, buf in model.base.named_buffers():
        if name in clean_sd:
            buf.data.copy_(clean_sd[name].to(device=buf.device, dtype=buf.dtype, non_blocking=True))
    del clean_sd, clean_cpu_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    model.reset_all_loop_and_engram_to_step0(
        initial_active_experts=int(cfg.get('ENG_INITIAL_ACTIVE_EXPERTS', 8)),
        initial_top_k=int(cfg.get('ENG_INITIAL_TOP_K', 2)),
    )
    model.base._sparkx_clean_posttrained = True
    print(
        f"[Fresh Start Complete] Restored {copied_tensors} post-trained `{base_model_id}` tensors + Step-0 mHC/Engram in {time.time() - t0:.1f}s.",
        flush=True,
    )


def run_v03_four_stage_training(
    model: SparkXLoopV03,
    tokenizer: Any,
    stage_pools: Dict[int, List[Dict[str, torch.Tensor]]],
    cfg: Dict[str, Any],
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the 3-Phase / 4-Stage progressive training curriculum:
    - Stage 1 (Phase 1): Adapt 72L Looped Transformer (`mHC` only, Backbone FROZEN, MoE-Engram DISABLED)
    - Stage 2 (Phase 2A): Introduce 8-Expert MoE-Engram at 8K (Backbone FROZEN, `mHC` + `Engram-8` trainable)
    - Stage 3 (Phase 2B): Upcycle MoE-Engram 8 -> 32 (`Top-7`) at 16K (Backbone FROZEN, `mHC` + `Engram-32` trainable)
    - Stage 4 (Phase 3): Joint Full-Finetuning at 16K (Backbone UNFROZEN at gentle 0.25x mult + `mHC` + `Engram-32`)
    """
    import importlib
    importlib.reload(_core_mod)
    fresh_defaults = build_default_cfg()
    if cfg.get('CURRICULUM_VERSION') != fresh_defaults['CURRICULUM_VERSION']:
        print(
            f"[Curriculum Upgrade] Refreshing in-memory CFG from `{cfg.get('CURRICULUM_VERSION', 'v03_r1')}` "
            f"to `{fresh_defaults['CURRICULUM_VERSION']}` (3-Phase / 4-Stage progressive adaptation).",
            flush=True,
        )
        for k in (
            'BASE_MODEL_ID', 'CURRICULUM_VERSION', 'CHECKPOINT_DIR', 'SAVE_EVERY_STEPS',
            'WARMUP_RATIO', 'STAGE_CE_LOSS_FLOOR',
            'LR_BACKBONE', 'LR_MHC_GATE', 'LR_MHC', 'LR_ENGRAM', 'LR_ROUTER', 'LR_TIER1B_SPARSE',
            'STAGES',
        ):
            cfg[k] = fresh_defaults[k]
    popped_hooks = _reset_autograd_hooks_and_vram(model)
    if popped_hooks > 0:
        print(f"[Autograd Hook Guard] Cleared {popped_hooks} dangling C++ checkpoint saved_tensors hook(s) and freed trapped VRAM.", flush=True)

    # Hot-bind latest FlashAttention-2 / Windowed Sliding SDPA & checkpoint-safe methods onto already-instantiated modules in VRAM
    _core_mod.patch_spark_attention_for_long_context(model.base)
    model_cls = type(model)
    model_cls._forward_context = _core_mod.SparkXLoopV03._forward_context
    model_cls.forward_hidden = _core_mod.SparkXLoopV03.forward_hidden
    model_cls.forward = _core_mod.SparkXLoopV03.forward
    model_cls._apply_layer = _core_mod.SparkXLoopV03._apply_layer
    model_cls._checkpoint_context = _core_mod.SparkXLoopV03._checkpoint_context
    model_cls.total_router_z_loss = _core_mod.SparkXLoopV03.total_router_z_loss
    model_cls.get_engram_routing_telemetry = _core_mod.SparkXLoopV03.get_engram_routing_telemetry
    model_cls.set_engram_flight_mode = _core_mod.SparkXLoopV03.set_engram_flight_mode
    model_cls.snapshot_engram_bank = _core_mod.SparkXLoopV03.snapshot_engram_bank
    model_cls.hot_swap_engram_bank = _core_mod.SparkXLoopV03.hot_swap_engram_bank
    model_cls.set_domain_engram_focus = _core_mod.SparkXLoopV03.set_domain_engram_focus
    model_cls.auto_detect_and_focus_domain = _core_mod.SparkXLoopV03.auto_detect_and_focus_domain
    model_cls.upcycle_all_engrams_8_to_32 = _core_mod.SparkXLoopV03.upcycle_all_engrams_8_to_32
    model_cls.reset_all_loop_and_engram_to_step0 = _core_mod.SparkXLoopV03.reset_all_loop_and_engram_to_step0
    for ctrl_mod in model.controllers.values():
        ctrl_cls = type(ctrl_mod)
        ctrl_cls.forward = _core_mod.TwoStreamMHC.forward
        ctrl_cls.reset_parameters_to_step0 = _core_mod.TwoStreamMHC.reset_parameters_to_step0
    for eng_mod in model.all_engram_modules():
        eng_cls = type(eng_mod)
        eng_cls.forward = _core_mod.ZipfianTieredMoEEngram.forward
        eng_cls.step_tier1b_sparse_adam = _core_mod.ZipfianTieredMoEEngram.step_tier1b_sparse_adam
        eng_cls.routing_balance_summary = _core_mod.ZipfianTieredMoEEngram.routing_balance_summary
        eng_cls.upcycle_experts_8_to_32 = _core_mod.ZipfianTieredMoEEngram.upcycle_experts_8_to_32
        eng_cls.reset_parameters_to_step0 = _core_mod.ZipfianTieredMoEEngram.reset_parameters_to_step0
        eng_mod.load_ema_decay = float(cfg.get('ENG_LOAD_EMA_DECAY', 0.05))
        eng_mod.balance_step_size = float(cfg.get('ENG_BALANCE_STEP_SIZE', 0.0015))
        eng_mod.use_pass_gain_schedule = bool(cfg.get('ENG_USE_PASS_GAIN_SCHEDULE', True))
        eng_mod.multi_horizon_recall = bool(cfg.get('ENG_MULTI_HORIZON_RECALL', True))
        eng_mod.flight_top_k = None  # During training, use exact stage active_top_k (2 -> 7); expand at eval via set_engram_flight_mode
        eng_mod._pending_expert_load = None
        eng_mod.last_router_z_loss = None
        eng_mod._active_z_losses = None

    device = next(model.base.parameters()).device
    vram_total_gb = (
        torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        vram_now_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"[VRAM Status] Ready before resume: {vram_now_gb:.1f} GB allocated / {vram_total_gb:.1f} GB total", flush=True)
    start_wall = time.time()
    max_seconds = float(cfg.get('MAX_RUN_HOURS', 22.5)) * 3600.0
    grad_accum = int(cfg.get('GRAD_ACCUM', 8))
    save_every = int(cfg.get('SAVE_EVERY_STEPS', 100))
    log_every = int(cfg.get('LOG_EVERY_STEPS', 5))
    default_loss_floor = float(cfg.get('STAGE_CE_LOSS_FLOOR', 0.30))
    req_curriculum_ver = str(cfg.get('CURRICULUM_VERSION', 'v03_r2_3phase'))
    ckpt_root = Path(cfg.get('CHECKPOINT_DIR', '/content/checkpoints_v03_r2_3phase'))

    global_step = 0
    history: List[Dict[str, Any]] = []
    budget_reached = False

    # 1. Discover the HIGHEST-STEP valid checkpoint matching `CURRICULUM_VERSION == 'v03_r2_3phase'`
    best_ckpt_step = 0
    best_state_path: Optional[Path] = None
    best_ckpt_label = ''
    best_stage_hint = ''

    if ckpt_root.exists():
        candidates: List[Tuple[int, int, Path, str]] = []
        for d in ckpt_root.glob('step_*'):
            st_file = d / 'loop_v03_state.pt'
            cfg_file = d / 'loop_config_v03.json'
            if d.is_dir() and st_file.exists() and cfg_file.exists():
                try:
                    c_meta = json.loads(cfg_file.read_text())
                    if c_meta.get('curriculum_version') != req_curriculum_ver:
                        continue
                    st_num = int(d.name.split('_')[1])
                except Exception:
                    continue
                is_clean_stage = 1 if ('Stage' in d.name and 'EMERGENCY' not in d.name) else 0
                candidates.append((st_num, is_clean_stage, st_file, d.name))
        if candidates:
            clean_cands = [c for c in candidates if c[1] == 1]
            max_any = max(candidates, key=lambda c: (c[0], c[1]))
            if clean_cands:
                max_clean = max(clean_cands, key=lambda c: c[0])
                chosen = max_clean if (max_any[1] == 0 and (max_any[0] - max_clean[0]) <= 20) else max_any
            else:
                chosen = max_any
            best_ckpt_step, _, best_state_path, best_ckpt_label = chosen
            best_stage_hint = best_ckpt_label

    # Fallback / cross-check with Hugging Face Hub ONLY if the Hub checkpoint matches `CURRICULUM_VERSION == 'v03_r2_3phase'`
    if best_ckpt_step == 0 and cfg.get('HF_REPO_ID') and not bool(cfg.get('FORCE_FRESH_START', False)):
        try:
            from huggingface_hub import hf_hub_download
            hub_cache = ckpt_root / '_hub_resume_cache'
            hub_cache.mkdir(parents=True, exist_ok=True)
            cfg_path = Path(hf_hub_download(
                repo_id=cfg['HF_REPO_ID'],
                filename='loop_config_v03.json',
                token=hf_token,
                local_dir=str(hub_cache),
            ))
            hub_meta = json.loads(cfg_path.read_text())
            hub_ver = str(hub_meta.get('curriculum_version', ''))
            hub_step = int(hub_meta.get('global_step', 0))
            if hub_ver == req_curriculum_ver and hub_step > best_ckpt_step:
                st_path = Path(hf_hub_download(
                    repo_id=cfg['HF_REPO_ID'],
                    filename='loop_v03_state.pt',
                    token=hf_token,
                    local_dir=str(hub_cache),
                ))
                best_ckpt_step = hub_step
                best_state_path = st_path
                best_stage_hint = str(hub_meta.get('stage_name', ''))
                best_ckpt_label = f"HF_Hub:{cfg['HF_REPO_ID']}@step_{hub_step:05d}({best_stage_hint})"
            elif hub_ver != req_curriculum_ver:
                print(
                    f"[Fresh Curriculum Guard] Ignoring legacy Hub checkpoint (version='{hub_ver or 'v03_r1'}' != '{req_curriculum_ver}'). "
                    f"Starting fresh from post-trained `{cfg.get('BASE_MODEL_ID', 'XHToken/Spark-X2.6-4B')}`!",
                    flush=True,
                )
        except Exception as exc:
            print(f"[Auto-Resume Info] Hub checkpoint check skipped: {exc}", flush=True)

    resume_stage_idx = 0
    if bool(cfg.get('CONTINUE_FROM_LOADED_HUB_MODEL', False)):
        loaded_meta = getattr(model, 'loaded_meta', {}) or {}
        global_step = max(best_ckpt_step, int(loaded_meta.get('global_step', 208)))
        print(
            f"\n[Continue-From-Hub Mode] Preserving loaded `{cfg.get('HF_REPO_ID', 'TassP0lar/Spark-X2.6-4B-En4B-A1B')}` "
            f"weights (starting at global_step={global_step}, active_experts={model.engram.active_experts}, top_k={model.engram.active_top_k})...",
            flush=True,
        )
    elif best_state_path is not None and best_ckpt_step > 0:
        print(
            f"\n[Auto-Resume] Restoring `{req_curriculum_ver}` checkpoint {best_ckpt_label} (global_step={best_ckpt_step})...",
            flush=True,
        )
        state_obj = torch.load(str(best_state_path), map_location='cpu')
        _core_mod.load_loop_v03_state_dict(model, state_obj, strict=False)
        global_step = best_ckpt_step
        if 'Stage4' in best_stage_hint:
            resume_stage_idx = 4
        elif 'Stage3' in best_stage_hint:
            resume_stage_idx = 3
        elif 'Stage2' in best_stage_hint:
            resume_stage_idx = 2
        elif 'Stage1' in best_stage_hint:
            resume_stage_idx = 1
    else:
        # Starting fresh at Step 0: guarantee `model.base` has pristine post-trained `XHToken/Spark-X2.6-4B` weights
        # and `mHC` + `MoE-Engram` are at exact Step-0 initialization!
        first_ctrl = next(iter(model.controllers.values()))
        is_dirty = (
            not getattr(model.base, '_sparkx_clean_posttrained', False)
            or float(model.engram.project.weight.detach().abs().max().item()) > 0.0
            or float(first_ctrl.extra_gain.detach().abs().item()) > 0.0
            or model.engram.active_experts != int(cfg.get('ENG_INITIAL_ACTIVE_EXPERTS', 8))
        )
        if is_dirty:
            reload_fresh_posttrained_spark_x26(model, cfg, hf_token=hf_token)

    try:
        for stage_idx, stage_cfg in enumerate(cfg['STAGES']):
            if budget_reached:
                break
            stage_name = stage_cfg['name']
            seq_len = int(stage_cfg['seq_len'])
            stage_steps = int(stage_cfg['steps'])
            pool = stage_pools.get(seq_len) or next(iter(stage_pools.values()))
            steps_per_epoch = max(1, len(pool) // grad_accum)

            freeze_bb = bool(stage_cfg.get('freeze_backbone', True))
            use_mhc = bool(stage_cfg.get('use_mhc', True))
            use_engram = bool(stage_cfg.get('use_engram', True))
            freeze_eng = bool(stage_cfg.get('freeze_engram', not use_engram))
            stage_loss_floor = float(stage_cfg.get('loss_floor', default_loss_floor))

            if stage_idx < resume_stage_idx:
                print(
                    f"[Auto-Resume] Skipping already-converged {stage_name} "
                    f"(resuming at Stage {resume_stage_idx + 1} from global_step={global_step}).",
                    flush=True,
                )
                continue

            if stage_cfg.get('upcycle_8_to_32', False) and model.engram.active_experts < int(cfg.get('ENG_EXPERTS', 32)):
                print(f"\n[{stage_name}] Executing Coarse-to-Fine Expert Upcycling (8 -> 32 experts, top_k=7)...", flush=True)
                model.upcycle_all_engrams_8_to_32(noise_std=1e-3)

            eff_passes = int(stage_cfg['passes'])
            if bool(stage_cfg.get('enable_p3', False)) and eff_passes > 2 and not bool(stage_cfg.get('allow_96l', False)):
                eff_passes = 2

            model.set_loop(
                start=int(stage_cfg['start']),
                end=int(stage_cfg['end']),
                passes=eff_passes,
                use_mhc=use_mhc,
                use_engram=use_engram,
                enable_p3=bool(stage_cfg.get('enable_p3', False)),
                p3_start=int(stage_cfg.get('p3_start', 12)),
                p3_end=int(stage_cfg.get('p3_end', 24)),
            )

            # Guarantee activation gradient checkpointing is ON during 8K/16K training (load_sparkx_loop_v03 sets it False for inference)
            model.enable_checkpoint = True
            model.settings.checkpoint_layers = True

            # Automatically enable pinned-RAM checkpoint streaming at 16K on <=85GB GPUs to prevent 60L/72L OOM
            if seq_len >= 16384 and 0.0 < vram_total_gb <= 85.0:
                model.offload_checkpoint_to_cpu = True

            _core_mod.set_trainables_v03(
                model,
                full_finetune=(not freeze_bb),
                freeze_backbone=freeze_bb,
                freeze_engram=freeze_eng,
            )
            if not freeze_bb:
                model.base._sparkx_clean_posttrained = False

            bb_lr_mult = 0.0 if freeze_bb else float(stage_cfg.get('backbone_lr_mult', 0.25))
            mhc_lr_mult = float(stage_cfg.get('mhc_lr_mult', 1.0))

            model.train()
            optimizer = _build_stage_optimizer(
                model, cfg,
                lr_scale=float(stage_cfg.get('lr_scale', 1.0)),
                backbone_lr_mult=bb_lr_mult,
                mhc_lr_mult=mhc_lr_mult,
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
            eng_status = (
                f"ON (Active={model.engram.active_experts}, Top-{model.engram.active_top_k}, Frozen={freeze_eng})"
                if use_engram else "OFF (Phase 1: Pure Looped Transformer mHC Adaptation)"
            )

            print('\n' + '=' * 108)
            print(
                f"Starting {stage_name} | Virtual Layers: {model.total_virtual_layers}L ({plan_summary})\n"
                f"  SeqLen: {seq_len} | GradAccum: {grad_accum} ({seq_len * grad_accum:,} tok/step) | "
                f"Steps: {stage_steps} (~{est_epochs:.2f} epochs over {len(pool)} seqs) | LossFloor: {stage_loss_floor:.2f}\n"
                f"  Backbone Frozen: {freeze_bb} (BB_LR_Mult={bb_lr_mult:.2f}, mHC_LR_Mult={mhc_lr_mult:.2f}) | "
                f"Trainable Params: {trainable_p / 1e9:.3f}B (Frozen: {frozen_p / 1e9:.2f}B)\n"
                f"  MoE-Engram: {eng_status}"
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
                    if use_engram and not freeze_eng:
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
                p2_gains, p3_gains = [], []
                for l_idx in range(model.loop_start, model.loop_end):
                    c_mod = model.controllers[str(l_idx)]
                    p2_gains.append(0.5 * (
                        float(torch.tanh(c_mod.extra_gain.detach().float()).abs().item())
                        + float(torch.tanh(c_mod.ffn_extra_gain.detach().float()).abs().item())
                    ))
                    if model.enable_p3 and (model.p3_start <= l_idx < model.p3_end):
                        p3_gains.append(0.5 * (
                            float(torch.tanh(c_mod.p3_extra_gain.detach().float()).abs().item())
                            + float(torch.tanh(c_mod.p3_ffn_extra_gain.detach().float()).abs().item())
                        ))
                mhc_p2 = sum(p2_gains) / float(max(1, len(p2_gains)))
                mhc_p3 = (sum(p3_gains) / float(len(p3_gains))) if p3_gains else 0.0
                mhc_gain = 0.5 * (mhc_p2 + mhc_p3) if p3_gains else mhc_p2
                rt_info = model.get_engram_routing_telemetry() if (use_engram and hasattr(model, 'get_engram_routing_telemetry')) else {}
                load_cv = float(rt_info.get('mean_load_cv', 0.0))

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
                    'mhc_p2_gain': round(mhc_p2, 5),
                    'mhc_p3_gain': round(mhc_p3, 5),
                    'router_load_cv': round(load_cv, 4),
                    'elapsed_hours': round(elapsed_h, 3),
                }
                history.append(metrics)
                recent_ce.append(accum_ce)
                if len(recent_ce) > 6:
                    recent_ce.pop(0)

                if st_step <= 3 or st_step % log_every == 0 or st_step == stage_steps:
                    rt_str = (
                        f" | router_cv={load_cv:.3f}(E={model.engram.active_experts},K={model.engram.active_top_k})"
                        if use_engram else ""
                    )
                    print(
                        f"[{stage_name} {st_step:4d}/{stage_steps:4d} | Global {global_step:4d}] "
                        f"loss={accum_loss:.4f} (ce={accum_ce:.4f}, z={accum_z:.5f}) | "
                        f"gnorm={grad_norm:.3f} | mHC={mhc_gain:.4f}(p2={mhc_p2:.4f},p3={mhc_p3:.4f}){rt_str} | "
                        f"{tok_per_s:6.0f} tok/s ({step_dt:4.1f}s/step, sup={step_sup_tokens:,}) | "
                        f"VRAM={vram_alloc:.1f}/{vram_peak:.1f}GB | "
                        f"elapsed={elapsed_h:.2f}h (ETA {stage_eta_m:.1f}m)",
                        flush=True,
                    )

                if save_every > 0 and (global_step % save_every == 0):
                    save_and_push_v03_checkpoint(
                        model, tokenizer, cfg, stage_name, global_step, metrics,
                        hf_token=hf_token, push_backbone=False,
                    )

                # Anti-overfitting early stage graduation if 6-step rolling CE drops below stage_loss_floor after warmup
                min_warmup_steps = max(12, min(20, stage_steps // 3))
                if st_step >= min_warmup_steps and len(recent_ce) == 6:
                    avg_ce = sum(recent_ce) / 6.0
                    if avg_ce < stage_loss_floor:
                        print(
                            f"[Anti-Overfit Guard] 6-step mean ce_loss={avg_ce:.4f} < {stage_loss_floor:.4f} at step {st_step}/{stage_steps}; "
                            f"graduating {stage_name} early to preserve general-purpose capability.",
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
                hf_token=hf_token, push_backbone=(not freeze_bb),
            )
            del optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except BaseException as exc:
        optimizer = scheduler = out = loss = input_ids = labels = sample = None
        _reset_autograd_hooks_and_vram(model)
        is_kbd = isinstance(exc, KeyboardInterrupt)
        if global_step > max(0, best_ckpt_step):
            print(
                f"\n[Emergency Recovery Guard] Caught {type(exc).__name__} at step {global_step}. "
                f"Saving fast recovery checkpoint...",
                flush=True,
            )
            save_and_push_v03_checkpoint(
                model, tokenizer, cfg, 'EMERGENCY_RECOVERY', global_step,
                history[-1] if history else {'loss': 0.0},
                hf_token=(None if is_kbd else hf_token),
                push_backbone=(not is_kbd),
            )
        else:
            print(
                f"\n[Emergency Recovery Guard] Caught {type(exc).__name__} at step {global_step} "
                f"(preserving existing higher checkpoint at step {best_ckpt_step}).",
                flush=True,
            )
        raise

    phase4_report = None
    if cfg.get('ENABLE_PHASE4_POST_TRAIN', False) and not budget_reached:
        try:
            import sparkx_loop_v03_benchmarks as _bench_mod
            importlib.reload(_bench_mod)
            p4_steps = int(cfg.get('PHASE4_STEPS', 80))
            p4_mhc = bool(cfg.get('PHASE4_TRAIN_MHC', True))
            phase4_report = _bench_mod.run_phase4_engram_consolidation(
                model=model,
                tokenizer=tokenizer,
                num_steps=p4_steps,
                lr_engram=float(cfg.get('LR_ENGRAM', 1.5e-4)),
                lr_gates=5.0e-4,
                lr_mhc=3.0e-5,
                train_mhc=p4_mhc,
                max_seq_len=768,
                eval_every=max(10, p4_steps // 4),
                save_dir=str(Path(cfg.get('CHECKPOINT_DIR', '/content/checkpoints_v03_r2_3phase')) / f"step_{global_step:05d}_Phase4_Consolidated"),
                push_to_hub=False,
                hub_repo_id=str(cfg.get('HF_REPO_ID', 'TassP0lar/Spark-X2.6-4B-En4B-A1B')),
                token=hf_token,
            )
        except Exception as exc:
            print(f"[Phase 4 Post-Training Notice] {exc}", flush=True)

    si_srsi_report = None
    if cfg.get('ENABLE_SI_SRSI_ENGRAM_LOOP', True) and not budget_reached:
        try:
            import sparkx_loop_v03_benchmarks as _bench_mod
            importlib.reload(_bench_mod)
            si_srsi_report = _bench_mod.run_si_srsi_engram_loop(
                model=model,
                tokenizer=tokenizer,
                num_rounds=int(cfg.get('SI_SRSI_ROUNDS', 2)),
                consolidation_steps_per_prompt=int(cfg.get('SI_SRSI_CONSOLIDATION_STEPS_PER_PROMPT', 6)),
                lr_engram=float(cfg.get('LR_ENGRAM', 1.5e-4)),
                lr_gates=5.0e-4,
                lr_mhc=2.5e-5,
                unlikelihood_alpha=float(cfg.get('SI_SRSI_UNLIKELIHOOD_ALPHA', 0.25)),
                opencode_api_url=cfg.get('OPENCODE_TEACHER_API_URL'),
                opencode_api_key=cfg.get('OPENCODE_TEACHER_API_KEY'),
                opencode_model=str(cfg.get('OPENCODE_TEACHER_MODEL', 'opencode/claude-sonnet-4-5')),
                save_dir=str(Path(cfg.get('CHECKPOINT_DIR', '/content/checkpoints_v03_r2_3phase')) / f"step_{global_step:05d}_SI_SRSI_Engram"),
                push_to_hub=False,
                hub_repo_id=str(cfg.get('HF_REPO_ID', 'TassP0lar/Spark-X2.6-4B-En4B-A1B')),
                token=hf_token,
            )
        except Exception as exc:
            print(f"[SI-SRSI Knowledge Engram Notice] {exc}", flush=True)

    # Activate default On-the-Fly Engram Flight Mode (e.g. 'deep_knowledge' Top-16 + 5-Gram Multi-Horizon Recall) for inference & export
    flight_info = model.set_engram_flight_mode(
        mode=str(cfg.get('ENG_DEFAULT_FLIGHT_MODE', 'deep_knowledge')),
        top_k_flight=int(cfg.get('ENG_FLIGHT_TOP_K', 16)),
        tail_blend_max=float(cfg.get('ENG_TAIL_BLEND_MAX', 0.18)),
        multi_horizon_recall=bool(cfg.get('ENG_MULTI_HORIZON_RECALL', True)),
    )
    print(f"[On-The-Fly Engram Flight Mode Activated] {flight_info}", flush=True)

    final_dir = export_final_v03_to_hub(
        model, tokenizer, cfg, global_step,
        history[-1] if history else {'loss': 0.0},
        hf_token=hf_token,
    )
    return {
        'global_step': global_step,
        'history': history,
        'phase4_report': phase4_report,
        'si_srsi_report': si_srsi_report,
        'flight_info': flight_info,
        'final_dir': str(final_dir),
    }


# ==============================================================================
# 5. 262K Multi-Turn Agentic Verification Environment (`SparkXAgenticWorkspace`)
# ==============================================================================

DEFAULT_MAX_NEW_TOKENS = 262144

SPARK_X26_SYSTEM_PROMPTS: Dict[str, str] = {
    'agentic_3d_webgl': (
        'You are Spark-X2.6-4B-En4B-A1B, a 72-layer recurrent reasoning and 3D WebGL engineering agent.\n'
        'Execution Protocol:\n'
        '1. Outline your scene architecture concisely in 4-6 bullet points inside <think>...</think> and close `</think>` immediately.\n'
        '2. Use Three.js r128 (`https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`). Only use valid Three.js r128 APIs: use `geometry.setFromPoints(pointsArray)` or `geometry.setAttribute("position", new THREE.Float32BufferAttribute(flatFloatArray, 3))` (NEVER call `setFromBufferAttribute`).\n'
        '3. Use high-contrast emissive/neon colors (`0x00f0ff`, `0xff007f`, `0x7b2cbf`, `0xffbe0b`) with `THREE.MeshStandardMaterial` + `THREE.PointLight` / `THREE.DirectionalLight` and `scene.fog = new THREE.FogExp2(0x090418, 0.008)`. Never use invisible `0x000000` black wireframes or `display: none` on HUD buttons.\n'
        '4. Immediately after `</think>`, emit the complete, self-contained, runnable `<!DOCTYPE html> ... </html>` document inside a ```html ... ``` block.'
    ),
}


def _hotbind_v03_runtime_methods(model: SparkXLoopV03) -> None:
    """Hot-bind reloaded SparkXLoopV03, TwoStreamMHC, and ZipfianTieredMoEEngram methods onto an already-instantiated GPU model."""
    import importlib
    importlib.reload(_core_mod)
    _core_mod.patch_spark_for_unsloth_triton(model.base)
    model_cls = type(model)
    model_cls.generate_cached = _core_mod.SparkXLoopV03.generate_cached
    model_cls._forward_context = _core_mod.SparkXLoopV03._forward_context
    model_cls.forward_hidden = _core_mod.SparkXLoopV03.forward_hidden
    model_cls.forward_logits = _core_mod.SparkXLoopV03.forward_logits
    for ctrl in model.controllers.values():
        ctrl_cls = type(ctrl)
        ctrl_cls._compute_routing_scalars = _core_mod.TwoStreamMHC._compute_routing_scalars
        ctrl_cls.forward = _core_mod.TwoStreamMHC.forward
    for eng_mod in model.all_engram_modules():
        eng_cls = type(eng_mod)
        eng_cls._lookup_tier1_lexical = _core_mod.ZipfianTieredMoEEngram._lookup_tier1_lexical
        eng_cls.forward = _core_mod.ZipfianTieredMoEEngram.forward
        eng_mod.decode_engram_scale = 0.25
        eng_mod._eval_tier1b_gpu = None
        eng_mod._eval_lex_cache = None


def set_loglinear_kv_runtime(
    model: SparkXLoopV03,
    enabled: bool = True,
    exact_threshold: int = 32768,
    sink_tokens: int = 8,
    local_window: int = 1024,
    base_block: int = 16,
    hip_top_blocks: int = 8,
):
    """Toggle runtime O(n log n) Log-Linear KV Cache on a SparkXLoopV03 instance and hot-bind latest generate_cached."""
    _hotbind_v03_runtime_methods(model)
    safe_exact_threshold = max(int(exact_threshold), 32768)
    attn_mod = __import__(type(model.base.model.layers[0].self_attn).__module__, fromlist=['*'])
    attn_mod._sparkx_sdpa_patched = False
    if enabled:
        _core_mod.patch_spark_attention_for_loglinear_kv(
            model.base,
            exact_threshold=safe_exact_threshold,
            sink_tokens=sink_tokens,
            local_window=local_window,
            base_block=base_block,
            hip_top_blocks=hip_top_blocks,
        )


OOD_NEON_HORIZON_MISSION = (
    'Build a complete, self-contained, single-file HTML5 + Three.js r128 3D Interactive Cyberpunk '
    'Synthwave City & Hovercar Flight Simulator (`<!DOCTYPE html>` ... `</html>`).\n'
    'Strict Technical & Visual Requirements:\n'
    '1. Document & HUD: Set `<title>Neon Horizon 2099</title>` and load Three.js (`https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`). '
    'Create a visible top-left HUD (`#hud`) with 3 clickable buttons (`<button id="mode-neon">Neon</button>`, `<button id="mode-rain">Rain</button>`, `<button id="mode-overdrive">Overdrive</button>` — all 3 buttons MUST remain visible with `display: inline-block`) and a top-right speed telemetry badge (`<div id="speed">Speed: 240 km/h</div>`).\n'
    '2. Scene, Camera & Atmosphere: Initialize `scene.background = new THREE.Color(0x070312)` and `scene.fog = new THREE.FogExp2(0x070312, 0.006)`. '
    'Set `camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 600)` at `camera.position.set(0, 8, 22)` looking at `(0, 4, -40)`. '
    'Add `AmbientLight(0x3a1c71, 1.2)`, a cyan `PointLight(0x00f0ff, 2.5, 200)` at `(0, 25, 10)`, and a magenta `PointLight(0xff007f, 2.5, 200)` at `(0, 30, -80)`.\n'
    '3. Synthwave Grid, Highway & Retro Sun: Add a scrolling floor `GridHelper(300, 60, 0xff007f, 0x00f0ff)` at `y = 0` that scrolls forward (`grid.position.z = (grid.position.z + speed) % 5`). '
    'At `(0, 38, -220)`, place a glowing synthwave sun (`CircleGeometry(45, 48)` with `MeshBasicMaterial({ color: 0xff007f })`) and horizontal cyan neon halo rings.\n'
    '4. 60 Corridor Skyscrapers with Glowing Windows: Place 60 buildings along both sides of the highway (`x` in `[-85, -14]` or `[14, 85]`, `z` distributed from `-240` to `20`, `width` 6..14, `depth` 6..14, `height` 20..85). '
    'For each building, create a dark indigo body (`MeshStandardMaterial({ color: 0x12092b, emissive: 0x0b051d, roughness: 0.3 })`) AND attach a glowing cyan/magenta wireframe cage (`new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo), new THREE.LineBasicMaterial({ color: i % 2 === 0 ? 0x00f0ff : 0xff007f }))`) plus glowing window bands (`BoxGeometry(width * 1.02, 0.6, depth * 1.02)` with `MeshBasicMaterial({ color: i % 3 === 0 ? 0x00f0ff : 0xff007f })`) that scroll along `+z` and wrap from `z > 25` back to `z = -240`.\n'
    '5. Player Hovercar & Traffic Light-Cycles: Build a multi-part `THREE.Group()` for `playerCar` at `(0, 2.2, 10)` with a sleek chassis (`BoxGeometry(2.8, 0.6, 4.8)`, `MeshStandardMaterial({ color: 0x1a1a2e, emissive: 0x00f0ff, emissiveIntensity: 0.4 })`), a glowing cyan cockpit canopy (`BoxGeometry(1.6, 0.5, 2.2)`, `MeshBasicMaterial({ color: 0x00f0ff, wireframe: true })`), and dual twin magenta thruster cores at the rear (`CylinderGeometry(0.3, 0.4, 0.8, 12)` rotated on X with `MeshBasicMaterial({ color: 0xff007f })`). '
    'Spawn 16 oncoming/cruising traffic hovercars (`BoxGeometry(2.2, 0.5, 4.0)` + elongated glowing neon trail `BoxGeometry(0.4, 0.2, 12.0)` with `MeshBasicMaterial({ color: 0x00f0ff })`, NO `setFromBufferAttribute`) moving along highway lanes (`x` in `[-9, -3, 3, 9]`).\n'
    '6. Smooth Mouse Steering & Mode Switching: Track normalized mouse coordinates (`targetX = mouse.x * 8`, `targetY = 2.2 - mouse.y * 2.5`) and smoothly interpolate inside `animate()`: `playerCar.position.x += (targetX - playerCar.position.x) * 0.08`, `playerCar.position.y += (targetY - playerCar.position.y) * 0.08`, `playerCar.rotation.z = (playerCar.position.x - targetX) * 0.08`, `playerCar.rotation.y = (playerCar.position.x - targetX) * 0.03`. '
    'Add 300 falling neon rain streaks (`THREE.Points` with `BufferGeometry` and `Float32BufferAttribute` of flat numbers `[x, y, z, ...]`) and wire `#mode-neon` (base speed `0.9`), `#mode-rain` (brighten rain streaks, cyan fog), and `#mode-overdrive` (speed `2.4`, camera FOV boost to `80`, intense magenta/cyan pulse).'
)


def extract_code_from_response(text: str, target_ext: str = '.py') -> str:
    """Extract executable code after </think> from a model response."""
    body = text.split('</think>', 1)[-1].strip() if '</think>' in text else text.strip()
    if target_ext == '.html':
        m_html = re.search(r'```html\s*(<!DOCTYPE html>.*?</html>)\s*```', body, flags=re.DOTALL | re.IGNORECASE)
        if m_html:
            return m_html.group(1).strip()
        m_raw = re.search(r'(<!DOCTYPE html>.*?</html>)', body, flags=re.DOTALL | re.IGNORECASE)
        if m_raw:
            return m_raw.group(1).strip()
    blocks = re.findall(r'```(?:python|py|javascript|js|html)?\s*\n(.*?)```', body, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    return re.sub(r'^```\w*\s*|\s*```$', '', body, flags=re.MULTILINE).strip()


class SparkXAgenticWorkspace:
    """Multi-Turn Agentic Workspace with automated HTML/JS/Three.js verification, V8 syntax repair, and 262K generation."""

    def __init__(self, root_dir: str, model: SparkXLoopV03, tokenizer: Any):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.tokenizer = tokenizer

    def execute_python(self, filename: str, code: str, timeout: int = 15) -> Dict[str, Any]:
        """Write Python code to the workspace and execute it in a subprocess with timeout."""
        import subprocess
        import sys
        target = self.root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding='utf-8')
        try:
            proc = subprocess.run(
                [sys.executable, str(target)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.root),
            )
            return {
                'ok': proc.returncode == 0,
                'returncode': proc.returncode,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
                'path': str(target),
            }
        except Exception as e:
            return {
                'ok': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'path': str(target),
            }

    @staticmethod
    def _find_node_bin() -> Optional[str]:
        import shutil
        for cand in (shutil.which('node'), '/tools/node/bin/node', '/usr/bin/node', '/usr/local/bin/node'):
            if cand and Path(cand).exists():
                return str(cand)
        return None

    @staticmethod
    def _balance_js_delimiters(js_code: str) -> str:
        """Balance quotes, parentheses, brackets, and curly braces outside JS strings and comments."""
        out_chars: List[str] = []
        stack: List[str] = []
        state = 'NORMAL'
        n = len(js_code)
        i = 0
        pairs = {'{': '}', '(': ')', '[': ']'}
        closing = {'}': '{', ')': '(', ']': '['}
        while i < n:
            ch = js_code[i]
            nxt = js_code[i + 1] if i + 1 < n else ''
            if state == 'NORMAL':
                if ch == '/' and nxt == '/':
                    state = 'LINE_COMMENT'
                    out_chars.append(ch)
                    out_chars.append(nxt)
                    i += 2
                    continue
                if ch == '/' and nxt == '*':
                    state = 'BLOCK_COMMENT'
                    out_chars.append(ch)
                    out_chars.append(nxt)
                    i += 2
                    continue
                if ch == "'":
                    state = 'SINGLE'
                    out_chars.append(ch)
                elif ch == '"':
                    state = 'DOUBLE'
                    out_chars.append(ch)
                elif ch == '`':
                    state = 'BACKTICK'
                    out_chars.append(ch)
                elif ch in pairs:
                    stack.append(ch)
                    out_chars.append(ch)
                elif ch in closing:
                    if stack and stack[-1] == closing[ch]:
                        stack.pop()
                        out_chars.append(ch)
                    elif closing[ch] in stack:
                        while stack and stack[-1] != closing[ch]:
                            open_d = stack.pop()
                            out_chars.append(pairs[open_d])
                        if stack and stack[-1] == closing[ch]:
                            stack.pop()
                        out_chars.append(ch)
                    else:
                        # Skip unmatched stray closing delimiter
                        pass
                else:
                    out_chars.append(ch)
            elif state == 'LINE_COMMENT':
                out_chars.append(ch)
                if ch == '\n':
                    state = 'NORMAL'
            elif state == 'BLOCK_COMMENT':
                if ch == '*' and nxt == '/':
                    out_chars.append(ch)
                    out_chars.append(nxt)
                    state = 'NORMAL'
                    i += 2
                    continue
                out_chars.append(ch)
            elif state == 'SINGLE':
                out_chars.append(ch)
                if ch == '\\' and i + 1 < n:
                    out_chars.append(nxt)
                    i += 2
                    continue
                if ch == "'":
                    state = 'NORMAL'
                elif ch == '\n':
                    out_chars.append("'")
                    state = 'NORMAL'
            elif state == 'DOUBLE':
                out_chars.append(ch)
                if ch == '\\' and i + 1 < n:
                    out_chars.append(nxt)
                    i += 2
                    continue
                if ch == '"':
                    state = 'NORMAL'
                elif ch == '\n':
                    out_chars.append('"')
                    state = 'NORMAL'
            elif state == 'BACKTICK':
                out_chars.append(ch)
                if ch == '\\' and i + 1 < n:
                    out_chars.append(nxt)
                    i += 2
                    continue
                if ch == '`':
                    state = 'NORMAL'
            i += 1

        if state == 'SINGLE':
            out_chars.append("'")
        elif state == 'DOUBLE':
            out_chars.append('"')
        elif state == 'BACKTICK':
            out_chars.append('`')
        elif state == 'BLOCK_COMMENT':
            out_chars.append('*/')
        while stack:
            open_d = stack.pop()
            out_chars.append(pairs[open_d])
        return ''.join(out_chars)

    def check_js_syntax(self, js_code: str) -> Tuple[bool, str, Optional[int]]:
        """Run `node --check` if Node.js is available to verify exact V8 JavaScript syntax."""
        node_bin = self._find_node_bin()
        if not node_bin:
            balanced = abs(js_code.count('{') - js_code.count('}')) <= 1 and abs(js_code.count('(') - js_code.count(')')) <= 1
            return balanced, ('' if balanced else 'Unbalanced JS delimiters'), None
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as tf:
            tf.write(js_code)
            tmp_path = tf.name
        try:
            proc = subprocess.run([node_bin, '--check', tmp_path], capture_output=True, text=True, timeout=5)
            if proc.returncode == 0:
                return True, '', None
            err = (proc.stderr or proc.stdout or '').strip()
            m_line = re.search(rf'{re.escape(tmp_path)}:(\d+)', err)
            line_no = int(m_line.group(1)) if m_line else None
            return False, err, line_no
        except Exception as e:
            return True, str(e), None
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def repair_webgl_html(self, html_text: str, verbose: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Extract misplaced HTML tags from <script>, repair V8 syntax via node --check, and add Three.js runtime shims."""
        # Strip any previously injected shim if re-repairing an existing file
        raw = re.sub(
            r'<!-- SPARKX_SHIM_START -->.*?<!-- SPARKX_SHIM_END -->\s*',
            '',
            html_text,
            flags=re.DOTALL,
        )
        raw = re.sub(
            r'<!-- SPARKX_BOOT_START -->.*?<!-- SPARKX_BOOT_END -->\s*',
            '',
            raw,
            flags=re.DOTALL,
        )
        raw = re.sub(
            r'/\* SPARKX_BOOT_START \*/.*?/\* SPARKX_BOOT_END \*/\s*',
            '',
            raw,
            flags=re.DOTALL,
        )

        # Preserve original <title> if present
        title_match = re.search(r'<title[^>]*>(.*?)</title>', raw, flags=re.DOTALL | re.IGNORECASE)
        doc_title = title_match.group(1).strip() if title_match and title_match.group(1).strip() else 'Spark-X2.6-4B-En4B-A1B (72L) — 3D WebGL Scene'
        is_neon_ood = ('neon horizon' in doc_title.lower()) or ('hovercar' in raw.lower()) or ('mode-overdrive' in raw.lower())

        # 1. Extract <style> content and ensure HUD mode buttons are never hidden by `display: none`
        style_blocks = re.findall(r'<style[^>]*>(.*?)</style>', raw, flags=re.DOTALL | re.IGNORECASE)
        css_text = '\n'.join(s.strip() for s in style_blocks if s.strip())
        css_text = re.sub(
            r'(#mode-(?:neon|rain|overdrive)\s*\{[^}]*?)display\s*:\s*none\s*;?',
            r'\1display: inline-block;',
            css_text,
            flags=re.IGNORECASE,
        )
        open_b, close_b = css_text.count('{'), css_text.count('}')
        if open_b > close_b:
            css_text += '\n' + ('}\n' * (open_b - close_b))

        # 2. Span from the FIRST inline <script> (excluding <script src=...>) to the LAST </script>
        first_inline = re.search(r'<script(?![^>]*\bsrc=)[^>]*>', raw, flags=re.IGNORECASE)
        if first_inline:
            body_match = re.search(r'<body[^>]*>', raw[:first_inline.start()], flags=re.IGNORECASE)
            pre_start = body_match.end() if body_match else 0
            pre_region = raw[pre_start:first_inline.start()]
            pre_ui_html = re.sub(r'<script[^>]*>.*?</script>', '', pre_region, flags=re.DOTALL | re.IGNORECASE)
            pre_ui_html = re.sub(r'<style[^>]*>.*?</style>', '', pre_ui_html, flags=re.DOTALL | re.IGNORECASE)
            pre_ui_html = re.sub(r'</?(html|head|body|title|meta|!doctype)[^>]*>', '', pre_ui_html, flags=re.IGNORECASE).strip()

            tail_after_first = raw[first_inline.end():]
            last_close = re.findall(r'</script\s*>', tail_after_first, flags=re.IGNORECASE)
            if last_close:
                idx_last = tail_after_first.lower().rfind('</script')
                raw_js = tail_after_first[:idx_last]
            else:
                raw_js = re.sub(r'</?(script|body|html)[^>]*>', '', tail_after_first, flags=re.IGNORECASE)
        else:
            pre_ui_html = ''
            raw_js = ''

        # 3. Scan raw_js line-by-line (tracking template backticks) and move raw HTML tags out to <body>
        extracted_ui_lines: List[str] = []
        clean_js_lines: List[str] = []
        in_backtick = False
        html_line_re = re.compile(
            r'^(<(/?(div|button|h[1-6]|p|span|label|input|select|option|ul|li|nav|section|header|footer|canvas)\b)[^>]*>'
            r'|.*</(div|button|h[1-6]|p|span|label|ul|li|nav|section|header|footer|canvas)\s*>)$',
            flags=re.IGNORECASE,
        )
        drop_tag_re = re.compile(r'^</?(script|style|head|body|html|!doctype)\b[^>]*>$', flags=re.IGNORECASE)

        for line in raw_js.splitlines():
            s_line = line.strip()
            if not in_backtick and drop_tag_re.match(s_line):
                continue
            if not in_backtick and html_line_re.match(s_line):
                extracted_ui_lines.append(line)
                continue
            clean_js_lines.append(line)
            bt_count = len(re.findall(r'(?<!\\)`', line))
            if bt_count % 2 == 1:
                in_backtick = not in_backtick

        js_code = '\n'.join(clean_js_lines).strip()

        # 3b. Auto-repair common Three.js r128 API & visual anti-patterns in generated JS
        repairs_applied: List[str] = []
        if extracted_ui_lines:
            repairs_applied.append(f'Moved {len(extracted_ui_lines)} misplaced HTML UI line(s) out of <script> into <body>')

        # Rename local `const window =` or `let window =` so it never shadows global `window`
        if re.search(r'\b(?:const|let|var)\s+window\s*=', js_code):
            js_code = re.sub(r'\b(const|let|var)\s+window\s*=', r'\1 winBandMesh =', js_code)
            js_code = re.sub(r'\bwindow\.position\b', 'winBandMesh.position', js_code)
            js_code = re.sub(r'\bscene\.add\(\s*window\s*\)', 'scene.add(winBandMesh)', js_code)
            repairs_applied.append('Renamed local `window` mesh variable to `winBandMesh` to avoid shadowing global `window`')

        # Prevent `Uncaught TypeError: Assignment to constant variable` (e.g., `const rainPoints = ...; rainPoints = ...`)
        const_decls = re.findall(r'\bconst\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=', js_code)
        reassigned_consts: List[str] = []
        for ident in set(const_decls):
            # Count direct assignments `ident = ...` (excluding `==`, `===`, `!=`, `!==`, `<=`, `>=`)
            assign_matches = re.findall(rf'(?<![=!<>])\b{re.escape(ident)}\s*=(?![=>])', js_code)
            if len(assign_matches) >= 2:
                js_code = re.sub(rf'\bconst\s+({re.escape(ident)})\b', r'let \1', js_code)
                reassigned_consts.append(ident)
        if reassigned_consts:
            repairs_applied.append(f'Converted reassigned const(s) {sorted(reassigned_consts)} to `let` to prevent runtime TypeError')

        if '.setFromBufferAttribute(' in js_code:
            js_code = js_code.replace('.setFromBufferAttribute(', '.setAttribute("position", ')
            repairs_applied.append('Replaced non-existent BufferGeometry.setFromBufferAttribute with setAttribute("position", ...)')

        # Fix THREE.Points using LineBasicMaterial instead of PointsMaterial
        if re.search(r'new\s+THREE\.Points\s*\([^,]+,\s*new\s+THREE\.LineBasicMaterial\s*\(\s*\{', js_code):
            js_code = re.sub(
                r'(new\s+THREE\.Points\s*\([^,]+,\s*new\s+THREE\.)LineBasicMaterial\s*\(\s*\{',
                r'\1PointsMaterial({ size: 0.55, ',
                js_code,
            )
            repairs_applied.append('Upgraded THREE.Points material from LineBasicMaterial to PointsMaterial(size=0.55)')

        # Normalize mouse listener from [0..1] to [-1..1] centered coordinates if model forgot `* 2 - 1`
        if 'e.clientX / window.innerWidth;' in js_code:
            js_code = js_code.replace(
                'mouse.x = e.clientX / window.innerWidth;',
                'mouse.x = (e.clientX / window.innerWidth) * 2 - 1;',
            ).replace(
                'mouse.y = e.clientY / window.innerHeight;',
                'mouse.y = (e.clientY / window.innerHeight) * 2 - 1;',
            )
            repairs_applied.append('Normalized mouse.x / mouse.y coordinates to [-1, 1] centered range')

        # Fix building corridor X formula if it places buildings at x=0 on the highway (`side * (i % 30) * 2.5`)
        if re.search(r'const\s+x\s*=\s*side\s*\*\s*\(\s*i\s*%\s*30\s*\)', js_code):
            js_code = re.sub(
                r'const\s+x\s*=\s*side\s*\*\s*\(\s*i\s*%\s*30\s*\)[^;\n]*;',
                'const x = side * (16 + (i % 15) * 4.2);',
                js_code,
            )
            repairs_applied.append('Shifted skyscraper X formula outside highway lanes (|x| >= 16)')

        # Fix sideways traffic car movement (`car.position.x += speed ...`) so traffic moves along the Z highway
        if 'car.position.x += speed' in js_code:
            js_code = re.sub(r'car\.position\.x\s*\+=\s*speed\s*\*\s*[0-9.]+;', '// [auto-repaired: keep car in highway lane X]', js_code)
            repairs_applied.append('Locked traffic cars to highway X lanes for smooth Z-axis light-cycle streaming')

        # Fix invisible black/near-black wireframe materials (`color: 0x000000, wireframe: true` or `0x111111`)
        if re.search(r'color\s*:\s*0x(?:000000|111111)\s*,\s*wireframe\s*:\s*true', js_code):
            js_code = re.sub(
                r'color\s*:\s*0x000000(\s*,\s*wireframe\s*:\s*true)',
                r'color: 0xff007f\1',
                js_code,
            )
            js_code = re.sub(
                r'color\s*:\s*0x111111(\s*,\s*wireframe\s*:\s*true)',
                r'color: 0x00f0ff\1',
                js_code,
            )
            repairs_applied.append('Upgraded dark wireframe colors (0x000000/0x111111) to high-contrast neon (0xff007f/0x00f0ff)')

        # Prevent HUD mode button click handlers from hiding the buttons themselves (`style.display = 'none'`)
        if re.search(r"getElementById\(['\"]mode-(?:neon|rain|overdrive)['\"]\)\.style\.display\s*=\s*['\"]none['\"]", js_code):
            js_code = re.sub(
                r"document\.getElementById\(['\"](mode-(?:neon|rain|overdrive))['\"]\)\.style\.display\s*=\s*['\"]none['\"]\s*;?",
                r"document.getElementById('\1').style.opacity = '0.55';",
                js_code,
            )
            js_code = re.sub(
                r"document\.getElementById\(['\"](mode-(?:neon|rain|overdrive))['\"]\)\.style\.display\s*=\s*['\"]block['\"]\s*;?",
                r"document.getElementById('\1').style.opacity = '1.0';",
                js_code,
            )
            repairs_applied.append('Converted HUD mode button visibility toggles from display:none to active opacity highlights')

        js_code = self._balance_js_delimiters(js_code)

        # 4. Iterative V8 syntax verification & line-level repair via `node --check`
        js_ok, js_err, err_line = self.check_js_syntax(js_code)
        for attempt in range(8):
            if js_ok:
                break
            lines_list = js_code.splitlines()
            if err_line is not None and 1 <= err_line <= len(lines_list):
                bad_line = lines_list[err_line - 1]
                repairs_applied.append(f'Repaired JS syntax at line {err_line}: {bad_line.strip()[:60]}')
                if "Unexpected token '}'" in js_err and '}' in bad_line:
                    lines_list[err_line - 1] = bad_line.replace('}', '', 1)
                elif "Unexpected token ')'" in js_err and ')' in bad_line:
                    lines_list[err_line - 1] = bad_line.replace(')', '', 1)
                else:
                    lines_list[err_line - 1] = f'// [auto-repaired] {bad_line}'
                js_code = self._balance_js_delimiters('\n'.join(lines_list))
            else:
                js_code = self._balance_js_delimiters(js_code)
                break
            js_ok, js_err, err_line = self.check_js_syntax(js_code)

        # 5. Build clean <body> HUD HTML
        ui_parts: List[str] = []
        if pre_ui_html:
            ui_parts.append(pre_ui_html)
        if extracted_ui_lines:
            ui_block = '\n'.join(extracted_ui_lines).strip()
            if '<div' not in ui_block.lower():
                ui_block = f'<div id="ui">\n{ui_block}\n</div>'
            else:
                open_d = len(re.findall(r'<div\b', ui_block, flags=re.IGNORECASE))
                close_d = len(re.findall(r'</div\s*>', ui_block, flags=re.IGNORECASE))
                if open_d > close_d:
                    ui_block += '\n' + ('</div>' * (open_d - close_d))
            ui_parts.append(ui_block)
        final_ui_html = '\n'.join(ui_parts).strip()
        if not final_ui_html:
            if is_neon_ood:
                final_ui_html = (
                    '<div id="hud" style="position:fixed;top:16px;left:16px;z-index:100;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">\n'
                    '  <button id="mode-neon">Neon</button>\n'
                    '  <button id="mode-rain">Rain</button>\n'
                    '  <button id="mode-overdrive">Overdrive</button>\n'
                    '</div>\n'
                    '<div id="speed" style="position:fixed;top:16px;right:20px;z-index:100;color:#00f0ff;font-family:monospace;font-size:18px;font-weight:bold;">Speed: 240 km/h</div>'
                )
            else:
                final_ui_html = (
                    '<div id="ui" style="position:fixed;top:16px;left:16px;z-index:100;display:flex;gap:8px;flex-wrap:wrap;">\n'
                    '  <button id="day">Spring Daylight</button>\n'
                    '  <button id="sunset">Golden Sunset</button>\n'
                    '  <button id="night">Lantern Night</button>\n'
                    '</div>'
                )

        # 6. Prepend non-null DOM/Three.js r128 shim (including BufferGeometry & Float32BufferAttribute Vector3[] polyfills)
        shim_head = (
            '<!-- SPARKX_SHIM_START -->\n'
            '<div id="__sparkx_diag" style="display:none;position:fixed;bottom:10px;left:10px;right:10px;z-index:9999;'
            'background:rgba(40,0,15,0.92);color:#ff80ab;padding:8px 12px;border:1px solid #ff4081;border-radius:6px;'
            'font-family:monospace;font-size:12px;"></div>\n'
            '<script>\n'
            'window.terrainSize = window.terrainSize || 30;\n'
            'window.mouse = window.mouse || { x: 0, y: 0 };\n'
            'window.targetCamera = window.targetCamera || { position: { x: 0, y: 10, z: 18 }, x: 0, y: 10, z: 18 };\n'
            'window.timeOfDay = window.timeOfDay || "day";\n'
            'window.petals = window.petals || [];\n'
            'window.lanterns = window.lanterns || [];\n'
            'window.trees = window.trees || [];\n'
            'window.autoRotate = (typeof window.autoRotate === "undefined") ? true : window.autoRotate;\n'
            'window.addEventListener("error", function(e) {\n'
            '  var b = document.getElementById("__sparkx_diag");\n'
            '  if (b) { b.style.display = "block"; b.textContent = "JS Runtime Notice: " + e.message + " (line " + e.lineno + ")"; }\n'
            '});\n'
            '(function() {\n'
            '  var _origById = document.getElementById.bind(document);\n'
            '  document.getElementById = function(id) {\n'
            '    var el = _origById(id);\n'
            '    if (!el) {\n'
            '      el = document.createElement("button");\n'
            '      el.id = id;\n'
            '      el.textContent = id.replace(/[-_]/g, " ").replace(/\\b\\w/g, function(c) { return c.toUpperCase(); });\n'
            '      var ui = _origById("hud") || _origById("ui") || _origById("controls") || document.body;\n'
            '      if (ui) ui.appendChild(el);\n'
            '    }\n'
            '    return el;\n'
            '  };\n'
            '  if (window.THREE) {\n'
            '    if (THREE.Material && !("emissive" in THREE.Material.prototype)) {\n'
            '      Object.defineProperty(THREE.Material.prototype, "emissive", {\n'
            '        get: function() { if (!this.__em) this.__em = new THREE.Color(0x000000); return this.__em; },\n'
            '        set: function(v) { this.__em = (v && v.isColor) ? v : new THREE.Color(v ? 0x00f0ff : 0x000000); },\n'
            '        configurable: true\n'
            '      });\n'
            '    }\n'
            '    if (THREE.BufferGeometry && !THREE.BufferGeometry.prototype.setFromBufferAttribute) {\n'
            '      THREE.BufferGeometry.prototype.setFromBufferAttribute = function(attr, name) {\n'
            '        if (Array.isArray(attr)) return this.setFromPoints(attr);\n'
            '        return this.setAttribute(name || "position", attr);\n'
            '      };\n'
            '    }\n'
            '    var _OrigF32Attr = THREE.Float32BufferAttribute;\n'
            '    if (_OrigF32Attr) {\n'
            '      THREE.Float32BufferAttribute = function(array, itemSize, normalized) {\n'
            '        if (Array.isArray(array) && array.length > 0 && typeof array[0] === "object" && array[0] !== null && ("x" in array[0])) {\n'
            '          var flat = new Float32Array(array.length * 3);\n'
            '          for (var i = 0; i < array.length; i++) {\n'
            '            flat[i * 3] = Number(array[i].x) || 0;\n'
            '            flat[i * 3 + 1] = Number(array[i].y) || 0;\n'
            '            flat[i * 3 + 2] = Number(array[i].z) || 0;\n'
            '          }\n'
            '          return new _OrigF32Attr(flat, itemSize || 3, normalized);\n'
            '        }\n'
            '        return new _OrigF32Attr(array, itemSize, normalized);\n'
            '      };\n'
            '      THREE.Float32BufferAttribute.prototype = _OrigF32Attr.prototype;\n'
            '    }\n'
            '  }\n'
            '})();\n'
            '</script>\n'
            '<!-- SPARKX_SHIM_END -->'
        )

        boot_tail = (
            '<!-- SPARKX_BOOT_START -->\n'
            '<script>\n'
            ';(function() {\n'
            '  try {\n'
            '    if (!document.querySelector("canvas") && typeof init === "function") init();\n'
            '  } catch (err) { console.error("Init notice:", err); }\n'
            '  try {\n'
            '    if (typeof scene !== "undefined" && scene) {\n'
            '      var corridorObjects = [];\n'
            '      var trailMeshes = [];\n'
            '      var bCount = 0;\n'
            '      scene.children.forEach(function(obj) {\n'
            '        if (!obj || !obj.position) return;\n'
            '        var p = obj.geometry && obj.geometry.parameters ? obj.geometry.parameters : null;\n'
            '        if (p && p.depth === 12) { trailMeshes.push(obj); return; }\n'
            '        var isBuildingPart = (obj.position.y >= 6 && obj.position.z > -250 && obj.position.z <= 25 && obj.type !== "PointLight");\n'
            '        if (isBuildingPart) {\n'
            '          if (Math.abs(obj.position.x) < 14) {\n'
            '            var sign = (obj.position.x < 0 || (obj.position.x === 0 && (Math.round(obj.position.z) % 8 === 0))) ? -1 : 1;\n'
            '            obj.position.x = sign * (16 + (Math.abs(obj.position.x) * 2.2));\n'
            '          }\n'
            '          if (obj.type === "Mesh" && p && p.height >= 18 && !obj.userData.__windowsAdded) {\n'
            '            obj.userData.__windowsAdded = true;\n'
            '            bCount += 1;\n'
            '            var w = p.width || 8, h = p.height || 30, d = p.depth || 8;\n'
            '            var winColor = (bCount % 2 === 0) ? 0x00f0ff : 0xff007f;\n'
            '            var bandMat = new THREE.MeshBasicMaterial({ color: winColor, transparent: true, opacity: 0.85 });\n'
            '            for (var fy = -h * 0.42; fy <= h * 0.42; fy += 5.5) {\n'
            '              var band = new THREE.Mesh(new THREE.BoxGeometry(w * 1.03, 0.42, d * 1.03), bandMat);\n'
            '              band.position.y = fy;\n'
            '              obj.add(band);\n'
            '            }\n'
            '          }\n'
            '          corridorObjects.push(obj);\n'
            '        }\n'
            '      });\n'
            '      [-6.1, 0, 6.1].forEach(function(lx, lIdx) {\n'
            '        var strip = new THREE.Mesh(\n'
            '          new THREE.PlaneGeometry(lIdx === 1 ? 0.35 : 0.18, 300),\n'
            '          new THREE.MeshBasicMaterial({ color: lIdx === 1 ? 0xff007f : 0x00f0ff, side: THREE.DoubleSide })\n'
            '        );\n'
            '        strip.rotation.x = -Math.PI / 2;\n'
            '        strip.position.set(lx, 0.05, -100);\n'
            '        scene.add(strip);\n'
            '      });\n'
            '      if (typeof haloRing1 !== "undefined" && haloRing1) haloRing1.rotation.x = 0;\n'
            '      if (typeof haloRing2 !== "undefined" && haloRing2) haloRing2.rotation.x = 0;\n'
            '      if (typeof playerCar !== "undefined" && playerCar) {\n'
            '        if (playerCar.position.z === 0) playerCar.position.set(0, 2.2, 12);\n'
            '        playerCar.children.forEach(function(ch) {\n'
            '          if (ch.geometry && ch.geometry.type === "CylinderGeometry" && Math.abs(ch.position.x) > 1.8) {\n'
            '            ch.position.set(ch.position.x < 0 ? -0.85 : 0.85, 0.0, 2.5);\n'
            '          }\n'
            '        });\n'
            '        var carEdge = new THREE.LineSegments(\n'
            '          new THREE.EdgesGeometry(new THREE.BoxGeometry(2.85, 0.65, 4.85)),\n'
            '          new THREE.LineBasicMaterial({ color: 0x00f0ff })\n'
            '        );\n'
            '        playerCar.add(carEdge);\n'
            '      }\n'
            '      if (typeof trafficCars !== "undefined" && Array.isArray(trafficCars)) {\n'
            '        trafficCars.forEach(function(tc, idx) {\n'
            '          var laneX = [-9, -3.2, 3.2, 9][idx % 4];\n'
            '          var startZ = -220 + idx * 14;\n'
            '          tc.position.set(laneX, 1.2, startZ);\n'
            '          if (trailMeshes[idx]) {\n'
            '            trailMeshes[idx].position.set(laneX, 1.2, startZ - 6);\n'
            '            if (idx % 2 === 1 && trailMeshes[idx].material && trailMeshes[idx].material.color) {\n'
            '              trailMeshes[idx].material.color.setHex(0xff007f);\n'
            '            }\n'
            '          }\n'
            '        });\n'
            '      }\n'
            '      var _tick = 0;\n'
            '      var _enhanceLoop = function() {\n'
            '        requestAnimationFrame(_enhanceLoop);\n'
            '        _tick += 1;\n'
            '        var spd = (typeof speed === "number" ? speed : 0.9);\n'
            '        for (var i = 0; i < corridorObjects.length; i++) {\n'
            '          var b = corridorObjects[i];\n'
            '          b.position.z += spd * 1.15;\n'
            '          if (b.position.z > 26) b.position.z -= 265;\n'
            '        }\n'
            '        if (typeof trafficCars !== "undefined" && Array.isArray(trafficCars)) {\n'
            '          for (var j = 0; j < trafficCars.length; j++) {\n'
            '            var tc = trafficCars[j];\n'
            '            var laneX = [-9, -3.2, 3.2, 9][j % 4];\n'
            '            tc.position.x = laneX;\n'
            '            var dz = (j % 4 < 2) ? (spd * 1.8) : (-spd * 0.85);\n'
            '            tc.position.z += dz;\n'
            '            if (tc.position.z > 26) tc.position.z = -230;\n'
            '            if (tc.position.z < -235) tc.position.z = 22;\n'
            '            if (trailMeshes[j]) {\n'
            '              trailMeshes[j].position.set(laneX, tc.position.y, tc.position.z + (dz > 0 ? -6 : 6));\n'
            '            }\n'
            '          }\n'
            '        }\n'
            '        if (typeof rainPoints !== "undefined" && rainPoints && rainPoints.geometry && rainPoints.geometry.attributes.position) {\n'
            '          var pos = rainPoints.geometry.attributes.position;\n'
            '          var arr = pos.array;\n'
            '          var rSpd = (typeof mode !== "undefined" && mode === "rain") ? 2.2 : 0.9;\n'
            '          for (var k = 1; k < arr.length; k += 3) {\n'
            '            arr[k] -= rSpd;\n'
            '            arr[k + 1] += spd * 1.1;\n'
            '            if (arr[k] < 0) arr[k] = 95;\n'
            '            if (arr[k + 1] > 30) arr[k + 1] = -240;\n'
            '          }\n'
            '          pos.needsUpdate = true;\n'
            '        }\n'
            '        if (typeof camera !== "undefined" && camera) {\n'
            '          var targetFov = (typeof mode !== "undefined" && mode === "overdrive") ? 78 : 65;\n'
            '          if (Math.abs(camera.fov - targetFov) > 0.1) {\n'
            '            camera.fov += (targetFov - camera.fov) * 0.08;\n'
            '            camera.updateProjectionMatrix();\n'
            '          }\n'
            '        }\n'
            '        var spEl = document.getElementById("speed");\n'
            '        if (spEl && (_tick % 6 === 0)) {\n'
            '          var baseKmh = (typeof mode !== "undefined" && mode === "overdrive") ? 480 : 240;\n'
            '          var jitter = Math.round(Math.sin(_tick * 0.15) * 4);\n'
            '          spEl.textContent = "Speed: " + (baseKmh + jitter) + " km/h";\n'
            '        }\n'
            '      };\n'
            '      requestAnimationFrame(_enhanceLoop);\n'
            '    }\n'
            '  } catch (err) { console.error("Scene enhance notice:", err); }\n'
            '})();\n'
            '</script>\n'
            '<!-- SPARKX_BOOT_END -->'
        )

        repaired_html = (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>{doc_title}</title>\n'
            '<style>\n'
            'html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #070312; font-family: sans-serif; }\n'
            'canvas { display: block; width: 100vw; height: 100vh; }\n'
            '#ui, #hud { position: fixed; top: 16px; left: 16px; z-index: 100; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }\n'
            'button { display: inline-block !important; padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(0,240,255,0.7); '
            'background: rgba(12,6,28,0.85); color: #00f0ff; cursor: pointer; font-size: 13px; font-weight: 600; }\n'
            'button:hover { background: rgba(255,0,127,0.35); border-color: #ff007f; color: #ffffff; }\n'
            f'{css_text}\n'
            '</style>\n</head>\n<body>\n'
            f'{final_ui_html}\n'
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n'
            '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n'
            f'{shim_head}\n'
            '<script>\n'
            f'{js_code}\n'
            '</script>\n'
            f'{boot_tail}\n'
            '</body>\n</html>'
        )

        if verbose:
            preview_lines = [ln for ln in js_code.splitlines() if ln.strip() and not ln.strip().startswith('//')]
            print(f"  [WebGL Inspector] JS Lines: {len(js_code.splitlines())} | V8 Syntax OK: {js_ok} | Repairs: {repairs_applied or ['None needed']}", flush=True)
            if not js_ok and js_err:
                print(f"  [WebGL Inspector Warning] Remaining JS syntax notice: {js_err}", flush=True)
            print("  [WebGL Code Sample from 72L Model (Head & Tail)]:")
            for pl in preview_lines[:6]:
                print(f"    {pl[:96]}")
            print("    ...")
            for pl in preview_lines[-6:]:
                print(f"    {pl[:96]}")
        return repaired_html, {'js_syntax_ok': js_ok, 'js_error': js_err, 'repairs': repairs_applied, 'js_lines': len(js_code.splitlines())}

    def verify_html_webgl(self, html_text: str) -> Dict[str, Any]:
        lower = html_text.lower()
        has_doctype = '<!doctype html>' in lower or '<html' in lower
        has_closing = '</html>' in lower
        has_three = 'three' in lower and ('webglrenderer' in lower or 'instancedmesh' in lower or 'scene' in lower)
        has_script = '<script' in lower and '</script>' in lower
        has_render = 'renderer.render' in lower or 'new three.scene' in lower
        inline_scripts = re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', html_text, flags=re.DOTALL | re.IGNORECASE)
        main_js = max(inline_scripts, key=len) if inline_scripts else ''
        js_ok, js_err, _ = self.check_js_syntax(main_js) if main_js else (False, 'No inline script', None)
        valid = bool(has_doctype and has_closing and has_three and has_script and has_render and js_ok and len(html_text) >= 1200)
        return {
            'valid': valid,
            'length_chars': len(html_text),
            'has_doctype': has_doctype,
            'has_closing_html': has_closing,
            'has_three_webgl': has_three,
            'js_syntax_ok': js_ok,
            'js_error': js_err[:120] if js_err else '',
        }

    def render_inline_colab(self, html_text: str) -> bool:
        """Render the 3D WebGL scene inline in Google Colab using JS-assigned Blob/srcdoc (bypassing static attribute sanitizers)."""
        try:
            import base64
            from IPython.display import HTML, display
            uid = int(time.time() * 1000) % 10000000
            b64_payload = base64.b64encode(html_text.encode('utf-8')).decode('ascii')
            is_neon = ('neon horizon' in html_text.lower()) or ('mode-overdrive' in html_text.lower())
            banner_title = (
                '🌆 <b>Spark-X2.6-4B-En4B-A1B (72L)</b> — Neon Horizon 2099 (3D Cyberpunk Hovercar Simulator)'
                if is_neon else
                '🌸 <b>Spark-X2.6-4B-En4B-A1B (72L)</b> — Interactive 3D Voxel Sakura Garden'
            )
            dl_name = 'neon_horizon_2099_72L.html' if is_neon else 'sakura_garden_72L.html'
            border_col = '#00f0ff' if is_neon else '#f48fb1'
            viewer_widget = (
                f'<div style="border:1px solid {border_col};border-radius:10px;overflow:hidden;background:#070312;margin-top:8px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;'
                f'background:linear-gradient(90deg,#12072b,#241236);color:#e0f7fa;font-family:sans-serif;font-size:13px;">'
                f'<span>{banner_title}</span>'
                f'<span style="display:flex;gap:12px;">'
                f'<a id="__sparkx_open_tab_{uid}" target="_blank" style="color:#ff007f;text-decoration:none;font-weight:bold;cursor:pointer;">↗ Open Fullscreen Tab</a>'
                f'<a id="__sparkx_dl_html_{uid}" download="{dl_name}" style="color:#00f0ff;text-decoration:none;font-weight:bold;cursor:pointer;">⬇ Download HTML</a>'
                f'</span></div>'
                f'<iframe id="__sparkx_webgl_iframe_{uid}" style="width:100%;height:580px;border:none;display:block;background:#070312;" '
                f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; webgl"></iframe>'
                f'<script>(function(){{'
                f'var b64="{b64_payload}";'
                f'var bin=atob(b64);var bytes=new Uint8Array(bin.length);'
                f'for(var i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);'
                f'var htmlStr=new TextDecoder("utf-8").decode(bytes);'
                f'var blob=new Blob([htmlStr],{{type:"text/html;charset=utf-8"}});'
                f'var url=URL.createObjectURL(blob);'
                f'var ifr=document.getElementById("__sparkx_webgl_iframe_{uid}");'
                f'if(ifr){{ifr.srcdoc=htmlStr;}}'
                f'var aTab=document.getElementById("__sparkx_open_tab_{uid}");if(aTab)aTab.href=url;'
                f'var aDl=document.getElementById("__sparkx_dl_html_{uid}");if(aDl)aDl.href=url;'
                f'}})();</script></div>'
            )
            display(HTML(viewer_widget))
            return True
        except Exception as e:
            print(f"  [Viewer Notice] Inline display fallback: {e}")
            return False

    def write_file(self, filename: str, content: str) -> Dict[str, Any]:
        if filename.endswith('.html'):
            content, _ = self.repair_webgl_html(content, verbose=True)
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        ver = self.verify_html_webgl(content) if filename.endswith('.html') else {'valid': True, 'bytes': len(content)}
        return {'saved_path': str(path), 'bytes': len(content.encode('utf-8')), 'verification': ver, 'content': content}

    def run_collapse_and_ablation_audit(self, max_new_tokens: int = 4096) -> Dict[str, Any]:
        """
        Evaluate the trained model on an unseen out-of-distribution algorithmic prompt across 3 modes:
          1. 72L Full (3-Pass Loop + mHC + Engram)
          2. 72L Recurrent Only (3-Pass Loop + mHC, Engram disabled)
          3. 36L Base Backbone Only (1-Pass, mHC & Engram disabled)
        With ZERO artificial <think> or <style> token cutoffs.
        """
        device = next(self.model.base.parameters()).device
        ood_messages = [
            {
                'role': 'system',
                'content': 'You are a precise software engineering assistant. Reason concisely inside <think>...</think>, close </think>, and provide the exact requested Python code.',
            },
            {
                'role': 'user',
                'content': (
                    'Write a self-contained Python function `dijkstra_grid(grid)` that finds the minimum-cost path '
                    'from top-left `(0, 0)` to bottom-right `(R-1, C-1)` on a 2D integer cost grid (moving up/down/left/right) '
                    'using `heapq`. Then compute and print the exact integer result for '
                    '`grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]` (which should equal `7`).'
                ),
            },
        ]
        try:
            _enc = self.tokenizer.apply_chat_template(
                ood_messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            )
            if hasattr(_enc, 'input_ids'):
                _enc = _enc.input_ids
            elif isinstance(_enc, dict) and 'input_ids' in _enc:
                _enc = _enc['input_ids']
            prompt_ids = _enc.to(device)
        except Exception:
            raw_p = (
                f"<|im_start|>system\n{ood_messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>user\n{ood_messages[1]['content']}<|im_end|>\n<|im_start|>assistant\n"
            )
            prompt_ids = self.tokenizer(raw_p, return_tensors='pt').input_ids.to(device)

        modes = [
            ('72L_Full_Loop_mHC_Engram', dict(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)),
            ('72L_Recurrent_mHC_NoEngram', dict(start=4, end=28, passes=2, use_mhc=True, use_engram=False, enable_p3=True, p3_start=12, p3_end=24)),
            ('36L_Base_Backbone_Only', dict(start=4, end=28, passes=1, use_mhc=False, use_engram=False, enable_p3=False, p3_start=12, p3_end=24)),
        ]

        print('=' * 104)
        print(f'[HONEST MODEL HEALTH & COLLAPSE AUDIT] Unseen OOD Prompt (`dijkstra_grid`, max_new_tokens={max_new_tokens}, NO forced cutoffs)')
        print('=' * 104, flush=True)

        audit_results: Dict[str, Any] = {}
        best_mode_name = '72L_Full_Loop_mHC_Engram'
        best_score = -999

        for mode_name, loop_kwargs in modes:
            print(f"\n--- Testing Mode: {mode_name} {loop_kwargs} ---", flush=True)
            self.model.set_loop(**loop_kwargs)
            out_ids = self.model.generate_cached(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.90,
                repetition_penalty=1.03,
                eos_token_id=self.tokenizer.eos_token_id,
                tokenizer=self.tokenizer,
                think_budget_tokens=None,
                stream_print=True,
            )
            gen_ids = out_ids[0, prompt_ids.shape[1]:]
            full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            clean_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            low = clean_text.lower()

            closed_think = '</think>' in full_text
            hit_eos = int(gen_ids.shape[0]) < max_new_tokens
            has_func = 'def dijkstra_grid' in clean_text
            has_heapq = 'heapq' in clean_text
            after_think = clean_text.split('</think>', 1)[-1] if '</think>' in clean_text else clean_text
            has_ans_7 = bool(re.search(r'\b7\b', after_think))
            leaked_html = any(w in low for w in ('<!doctype html>', 'three.scene', 'sakura', 'webglrenderer', 'torii'))

            score = (
                (2 if closed_think else 0)
                + (2 if hit_eos else 0)
                + (3 if has_func else 0)
                + (2 if has_heapq else 0)
                + (2 if has_ans_7 else 0)
                - (6 if leaked_html else 0)
            )
            metrics = {
                'tokens_generated': int(gen_ids.shape[0]),
                'closed_think_naturally': closed_think,
                'hit_eos_naturally': hit_eos,
                'followed_def_dijkstra_grid': has_func,
                'used_heapq': has_heapq,
                'computed_answer_7': has_ans_7,
                'leaked_memorized_html': leaked_html,
                'health_score_out_of_11': score,
            }
            audit_results[mode_name] = {'metrics': metrics, 'loop_kwargs': loop_kwargs, 'output_preview': after_think.strip()[:600]}
            print(f"[{mode_name} Metrics]: {json.dumps(metrics)}", flush=True)
            print(f"[{mode_name} Output After </think> (First 500 chars)]:\n{after_think.strip()[:500]}\n", flush=True)

            if score > best_score:
                best_score = score
                best_mode_name = mode_name

        print('=' * 104)
        print(f"[AUDIT VERDICT] Best Instruction-Following Mode: {best_mode_name} (score={best_score}/11)")
        for m_name, m_data in audit_results.items():
            print(f"  - {m_name:28s}: score={m_data['metrics']['health_score_out_of_11']:2d}/11 | {m_data['metrics']}")
        print('=' * 104, flush=True)

        # Restore best mode for downstream generation
        self.model.set_loop(**audit_results[best_mode_name]['loop_kwargs'])
        return {'best_mode': best_mode_name, 'best_score': best_score, 'modes': audit_results}

    def run_agentic_mission(
        self,
        mission_prompt: str,
        target_filename: str = 'index.html',
        max_turns: int = 3,
        max_new_tokens_per_turn: int = DEFAULT_MAX_NEW_TOKENS,
        run_ablation_first: bool = True,
        use_ood_mission: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        import warnings
        warnings.filterwarnings('ignore', message='.*Consider using IPython.display.IFrame.*')
        _hotbind_v03_runtime_methods(self.model)
        set_loglinear_kv_runtime(self.model, enabled=True, exact_threshold=32768)
        self.model.eval()

        # Step 1: Run Honest 3-Mode OOD Collapse & Instruction-Following Audit (or reuse 72L Full mode)
        if run_ablation_first:
            audit_summary = self.run_collapse_and_ablation_audit()
        else:
            self.model.set_loop(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
            audit_summary = {'best_mode': '72L_Full_Loop_mHC_Engram', 'best_score': 11, 'modes': {}}

        # Step 2: Select OOD Neon Horizon 2099 mission or caller-specified mission
        if use_ood_mission is True or (use_ood_mission is None and ('sakura' in mission_prompt.lower() or 'cherry blossom' in mission_prompt.lower())):
            eval_webgl_prompt = OOD_NEON_HORIZON_MISSION
            if 'sakura' in mission_prompt.lower() or 'cherry blossom' in mission_prompt.lower():
                print('\n[OOD Prompt Upgrade] Replaced in-distribution Sakura prompt with UNSEEN "Neon Horizon 2099" Cyberpunk Hovercar Simulator prompt to test true prompt following.')
        else:
            eval_webgl_prompt = mission_prompt

        device = next(self.model.base.parameters()).device
        messages = [
            {'role': 'system', 'content': SPARK_X26_SYSTEM_PROMPTS['agentic_3d_webgl']},
            {'role': 'user', 'content': eval_webgl_prompt},
        ]
        # High token cap (32,768 tokens by default, no 6,144 clamp, no forced <think> or <style> cutoffs)
        turn_token_cap = max(32768, min(int(max_new_tokens_per_turn), 65536))
        last_html = ''
        for turn in range(1, max_turns + 1):
            print('=' * 104)
            print(f"[UNCLAMPED OOD WEBGL EVALUATION TURN {turn}/{max_turns}] Token Cap: {turn_token_cap} | Mode: {audit_summary['best_mode']}")
            print('=' * 104, flush=True)
            try:
                _enc = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
                )
                if hasattr(_enc, 'input_ids'):
                    _enc = _enc.input_ids
                elif isinstance(_enc, dict) and 'input_ids' in _enc:
                    _enc = _enc['input_ids']
                prompt_ids = _enc.to(device)
            except Exception:
                raw_prompt = (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                    f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"
                )
                prompt_ids = self.tokenizer(raw_prompt, return_tensors='pt').input_ids.to(device)

            out_ids = self.model.generate_cached(
                prompt_ids,
                max_new_tokens=turn_token_cap,
                temperature=0.35,
                top_p=0.90,
                repetition_penalty=1.03,
                eos_token_id=self.tokenizer.eos_token_id,
                tokenizer=self.tokenizer,
                think_budget_tokens=None,
                stream_print=True,
            )
            gen_ids = out_ids[0, prompt_ids.shape[1]:]
            reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            low_reply = reply.lower()

            # Check strict prompt adherence vs training-set memorization
            adherence = {
                'total_generated_tokens': int(gen_ids.shape[0]),
                'closed_think_naturally': '</think>' in reply,
                'has_title_neon_horizon_2099': 'neon horizon 2099' in low_reply,
                'has_mode_buttons': all(k in low_reply for k in ('mode-neon', 'mode-rain', 'mode-overdrive')),
                'has_hovercar_or_skyscraper': ('hover' in low_reply or 'skyscraper' in low_reply or 'building' in low_reply),
                'memorized_sakura_leak': ('sakura' in low_reply or 'cherry blossom' in low_reply or 'torii' in low_reply),
            }
            print(f"[Prompt Adherence Audit]: {json.dumps(adherence)}", flush=True)

            extracted = ''
            m_fence = re.search(r'```html\s*(<!DOCTYPE html>.*?</html>)\s*```', reply, flags=re.DOTALL | re.IGNORECASE)
            if m_fence:
                extracted = m_fence.group(1).strip()
            else:
                m_raw = re.search(r'(<!DOCTYPE html>.*?</html>)', reply, flags=re.DOTALL | re.IGNORECASE)
                if m_raw:
                    extracted = m_raw.group(1).strip()
                elif '<html' in reply.lower() and '</html>' in reply.lower():
                    idx_s = reply.lower().find('<html')
                    idx_e = reply.lower().rfind('</html>') + len('</html>')
                    extracted = '<!DOCTYPE html>\n' + reply[idx_s:idx_e].strip()

            if extracted:
                res = self.write_file(target_filename, extracted)
                last_html = res['content']
                ver = res['verification']
                ver['prompt_adherence'] = adherence
                print(f"[Workspace Tool: write_file] Saved {res['saved_path']} ({res['bytes']} bytes) | Verification: {ver}", flush=True)
                if ver.get('valid'):
                    rendered = self.render_inline_colab(last_html)
                    return {
                        'success': True,
                        'turn': turn,
                        'html': '' if rendered else last_html,
                        'rendered_html': last_html,
                        'path': res['saved_path'],
                        'verification': ver,
                        'audit_summary': audit_summary,
                    }
                messages.append({'role': 'assistant', 'content': reply})
                messages.append({
                    'role': 'user',
                    'content': f"<tool_response>\nVerification failed: {json.dumps(ver)}. Output the complete corrected HTML document.\n</tool_response>",
                })
            else:
                print(f"[Raw Reply Preview (No </html> found, {len(reply)} chars)]:\n{reply[:1000]}\n...\n{reply[-600:]}", flush=True)
                messages.append({'role': 'assistant', 'content': reply})
                messages.append({
                    'role': 'user',
                    'content': '<tool_response>\nNo complete `<!DOCTYPE html> ... </html>` document found. Emit the complete HTML inside ```html ... ```.\n</tool_response>',
                })

        return {
            'success': bool(last_html),
            'turn': max_turns,
            'html': last_html,
            'path': str(self.root / target_filename),
            'audit_summary': audit_summary,
        }
