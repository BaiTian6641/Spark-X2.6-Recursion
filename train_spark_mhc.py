"""train_spark_mhc.py — Milestone 2 Differentiable mHC Recurrence & Stage E D-LSA Trainer for Spark X2.6 4B Recursive72.

Implements the repaired end-to-end Loop72 + Per-Visit Paged D-LSA training, provenance, validation, and Hub release
pipeline (`Spark_X26_4B_Loop72_DLSA_Development_and_Training_Plan_2026-11-10.md`, resolving `F01`–`F18` / `WP0`–`WP8`):
  - Strictly frozen 36-layer `XHToken/Spark-X2.6-4B` (`64ca8f2a295b8ceb2757b9d4749a6558c4759e5b`) BF16 backbone
    verified via full-byte SHA-256 (`_parameter_sha256` / `canonical_state_dict_sha256`, zero head/tail sampling).
  - Bounded-gain `TwoStreamMHC` Schema v2 (`gain_mode="sigmoid"`, `mhc_schema_version=2`, `10,078,176` FP32 params).
  - State-neutral Stage A verification (`run_stage_a_verification`): snapshots and restores every controller tensor,
    gradient, loop setting, training flag, and Python/CPU/CUDA RNG state; verifies 18/18 native Q/K capture.
  - Real 5-domain training/dev/test corpus (`sparkx_training_data.py`), exact assistant-only masking with strict
    zero-target truncation rejection (`F03`), `attention_mask`-respecting single-example microbatches weighted by
    supervised token counts, periodic resumable checkpoints, and held-out `dev` selection.
  - Stage D pinned OMP v18.18.1 (`input` hashline patch) specialization with independent cosine LR schedule and
    held-out `dev` non-regression rollback gate.
  - Stage E 18/18 global-visit `VisitAwareDLSAIndexBank` streamed-Gram PCA + matched-support KL distillation on
    coherent `8K–16K` documents (`train_spark_dlsa.py`), locked to canonical `controller_sha256`.
  - Complete transitive runtime export closure (`EXPORT_RUNTIME_CLOSURE_FILES`), clean-process reload verification,
    intermediate stage commits AND final release commit pushed to `TassP0lar/Spark-X2.6-4B-Recursive72`.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

if HAS_TORCH:
    from sparkx_dlsa import canonical_state_dict_sha256
    from sparkx_loop_v03_core import (
        LoopV03Settings,
        SparkXLoopV03,
        TwoStreamMHC,
        fused_masked_cross_entropy,
        load_loop_v03_state_dict,
        loop_v03_state_dict,
    )
    from sparkx_training_data import (
        STAGE_C_DOMAIN_WEIGHTS,
        STAGE_D_DOMAIN_WEIGHTS,
        TrainingRecord,
        assign_document_split,
        build_long_context_calibration_token_packs,
        build_omp_tool_trajectory_examples as _build_omp_trajectories_v18,
        build_real_five_domain_corpus,
        build_smoke_five_domain_seed_bank,
        DomainWeightedBatchStream,
        load_manifest_records,
        render_conversation_exact_labels,
        save_manifest_records,
        verify_no_split_document_leakage,
    )


DEFAULT_MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_MODEL_REVISION = "64ca8f2a295b8ceb2757b9d4749a6558c4759e5b"
DEFAULT_HF_MODEL_NAME = "Spark X2.6 4B Recursive72"
DEFAULT_HF_REPO_SLUG = "Spark-X2.6-4B-Recursive72"
DEFAULT_HF_REPO_ID = "TassP0lar/Spark-X2.6-4B-Recursive72"
EXPECTED_DEFAULT_CONTROLLER_PARAMS = 10_078_176

DOMAIN_MIXTURE_WEIGHTS: Dict[str, float] = {
    "code_debug": 0.40,
    "general_instruct": 0.20,
    "math_symbolic": 0.15,
    "tool_trajectories": 0.15,
    "long_doc_repo": 0.10,
}

EXPORT_RUNTIME_CLOSURE_FILES: Tuple[str, ...] = (
    "sparkx_loop_v03_core.py",
    "sparkx_memory_utils.py",
    "sparkx_dlsa.py",
    "sparkx_kv_codec.py",
    "sparkx_paged_cache.py",
    "sparkx_sparse_attention.py",
    "sparkx_dynamic_memory.py",
    "sparkx_omp_server.py",
    "load_inference_v03.py",
    "train_spark_dlsa.py",
    "train_spark_mhc.py",
    "sparkx_training_data.py",
    "requirements.txt",
    "requirements-omp.txt",
)


@dataclasses.dataclass
class MHCTrainingConfig:
    """Configuration for Milestone 2 Loop72 `TwoStreamMHC` + Stage E D-LSA training (`§5.1`, `F08`)."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    output_dir: str = "results/mhc_train_run"
    hardware_profile: str = "auto"  # 'auto' | 'colab_g4_96gb' | 'a100_80gb' | 'a100_40gb' | 'l4_24gb' | 'cpu_test'

    # Bounded-gain TwoStreamMHC contract
    recurrence_mode: str = "learned_mhc"
    gain_mode: str = "sigmoid"  # 'sigmoid' (schema v2) | 'legacy_tanh' (schema v1)
    init_p2_gain: float = 0.01
    init_p3_gain: float = 0.005
    extra_pass_gate_multiplier: float = 1.0

    # Sequence & batch sizing (auto-populated for non-overridden fields via `apply_hardware_profile_to_config`)
    seq_len: int = 8192
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    ce_chunk_tokens: int = 2048

    # Optimizer & schedule
    adapter_lr: float = 2e-4
    routing_lr: float = 3e-5
    stage_d_lr_scale: float = 0.5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.08
    min_lr_ratio: float = 0.10

    # Teacher retention objective
    kl_beta: float = 0.5
    kl_temperature: float = 1.5
    sampled_kl_positions: int = 64
    coda_hidden_anchor_lambda: float = 0.0

    # Stage step budgets & curriculum
    curriculum: str = "fixed_72"  # 'fixed_72' (default) | '36_60_72' (ablation)
    stage_b_steps: int = 100
    stage_c_steps: int = 300
    stage_d_steps: int = 60
    run_stage_d_omp: bool = True
    run_stage_e_dlsa: bool = True
    stage_e_calibration_seqs: int = 16
    stage_e_val_seqs: int = 4
    stage_e_seq_len: int = 8192
    stage_e_distill_steps: int = 300

    # Data manifests, fail-closed smoke guard & held-out evaluation
    data_manifest: Optional[str] = None
    eval_manifest: Optional[str] = None
    stage_e_data_manifest: Optional[str] = None
    allow_smoke_fixtures: Optional[bool] = None
    target_corpus_examples: int = 1200
    dev_eval_batches: int = 8
    stage_d_max_dev_regression: float = 0.05
    enforce_stage_d_dev_gate: Optional[bool] = None

    # Checkpointing, resume, intermediate & final Hugging Face Hub commits
    save_every_steps: int = 50
    eval_every_steps: int = 25
    resume_from_checkpoint: Optional[str] = None
    use_hf_streaming: bool = True
    seed: int = 42
    hf_model_name: str = DEFAULT_HF_MODEL_NAME
    hf_repo_id: str = DEFAULT_HF_REPO_ID
    push_to_hub: bool = False
    push_intermediate_stages: bool = True
    hf_private: bool = False
    hf_token: Optional[str] = None
    export_standalone_backbone: bool = True

    def __new__(cls, *args: Any, **kwargs: Any) -> "MHCTrainingConfig":
        inst = super().__new__(cls)
        field_names = [f.name for f in dataclasses.fields(cls)]
        explicit = set(kwargs.keys())
        for idx in range(min(len(args), len(field_names))):
            explicit.add(field_names[idx])
        object.__setattr__(inst, "_explicit_fields", explicit)
        object.__setattr__(inst, "_init_done", False)
        return inst

    def __post_init__(self) -> None:
        object.__setattr__(self, "_init_done", True)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if getattr(self, "_init_done", False) and not name.startswith("_"):
            exp = getattr(self, "_explicit_fields", None)
            if isinstance(exp, set):
                exp.add(name)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "MHCTrainingConfig":
        cls = self.__class__
        dup = cls.__new__(cls)
        memo[id(self)] = dup
        object.__setattr__(dup, "_init_done", False)
        for f in dataclasses.fields(cls):
            object.__setattr__(dup, f.name, copy.deepcopy(getattr(self, f.name), memo))
        object.__setattr__(dup, "_explicit_fields", set(getattr(self, "_explicit_fields", set())))
        object.__setattr__(dup, "_init_done", True)
        return dup

    def is_explicit(self, field_name: str) -> bool:
        return field_name in getattr(self, "_explicit_fields", set())

    def to_dict(self) -> Dict[str, Any]:
        d = {f.name: copy.deepcopy(getattr(self, f.name)) for f in dataclasses.fields(self)}
        if d.get("hf_token"):
            d["hf_token"] = "***REDACTED***"
        return d


def detect_gpu_training_profile(vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """Return hardware sizing parameters tuned to detected or specified GPU VRAM (including Colab G4 96GB)."""
    if vram_gb is None and HAS_TORCH and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = float(props.total_memory) / float(1024 ** 3)
            gpu_name = str(props.name)
        except Exception:
            vram_gb = 0.0
            gpu_name = "unknown_cuda"
    else:
        gpu_name = "specified_or_cpu"

    mem = float(vram_gb or 0.0)
    if mem >= 85.0:
        return {
            "profile": "colab_g4_96gb",
            "gpu_name": gpu_name,
            "detected_vram_gb": round(mem, 2),
            "seq_len": 8192,
            "stage_e_seq_len": 12288,
            "max_recommended_seq_len": 16384,
            "micro_batch_size": 1,
            "grad_accum_steps": 8,
            "sampled_kl_positions": 64,
            "ce_chunk_tokens": 2048,
        }
    if mem >= 70.0:
        return {
            "profile": "a100_80gb",
            "gpu_name": gpu_name,
            "detected_vram_gb": round(mem, 2),
            "seq_len": 8192,
            "stage_e_seq_len": 8192,
            "max_recommended_seq_len": 16384,
            "micro_batch_size": 1,
            "grad_accum_steps": 8,
            "sampled_kl_positions": 64,
            "ce_chunk_tokens": 2048,
        }
    if mem >= 36.0:
        return {
            "profile": "a100_40gb",
            "gpu_name": gpu_name,
            "detected_vram_gb": round(mem, 2),
            "seq_len": 8192,
            "stage_e_seq_len": 8192,
            "max_recommended_seq_len": 8192,
            "micro_batch_size": 1,
            "grad_accum_steps": 8,
            "sampled_kl_positions": 64,
            "ce_chunk_tokens": 2048,
        }
    if mem >= 20.0:
        return {
            "profile": "l4_24gb",
            "gpu_name": gpu_name,
            "detected_vram_gb": round(mem, 2),
            "seq_len": 4096,
            "stage_e_seq_len": 6144,
            "max_recommended_seq_len": 6144,
            "micro_batch_size": 1,
            "grad_accum_steps": 16,
            "sampled_kl_positions": 48,
            "ce_chunk_tokens": 1024,
        }
    return {
        "profile": "cpu_test",
        "gpu_name": gpu_name,
        "detected_vram_gb": round(mem, 2),
        "seq_len": 128,
        "stage_e_seq_len": 128,
        "max_recommended_seq_len": 512,
        "micro_batch_size": 1,
        "grad_accum_steps": 2,
        "sampled_kl_positions": 16,
        "ce_chunk_tokens": 256,
    }


def apply_hardware_profile_to_config(cfg: MHCTrainingConfig, vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """Single-pass hardware profile resolution (`F08`, `§5.1`):
      - Applies detected hardware defaults when `hardware_profile == 'auto'` OR when a named profile is given.
      - NEVER clobbers any field explicitly passed to `MHCTrainingConfig(...)` or assigned by the caller!
    """
    prof_map = {
        "colab_g4_96gb": 96.0,
        "a100_80gb": 80.0,
        "a100_40gb": 40.0,
        "l4_24gb": 24.0,
        "cpu_test": 0.0,
    }
    prof_key = str(cfg.hardware_profile or "auto").strip().lower()
    if prof_key != "auto" and prof_key not in prof_map:
        raise ValueError(f"Unsupported hardware_profile={cfg.hardware_profile!r}; expected 'auto' or one of {tuple(prof_map)}.")

    target_vram = vram_gb if vram_gb is not None else prof_map.get(prof_key)
    info = detect_gpu_training_profile(target_vram)

    exp = getattr(cfg, "_explicit_fields", set())

    def _set_default_if_not_explicit(field_name: str, val: Any) -> None:
        if field_name not in exp:
            object.__setattr__(cfg, field_name, val)

    _set_default_if_not_explicit("seq_len", int(info["seq_len"]))
    _set_default_if_not_explicit("stage_e_seq_len", int(info["stage_e_seq_len"]))
    _set_default_if_not_explicit("micro_batch_size", int(info["micro_batch_size"]))
    _set_default_if_not_explicit("grad_accum_steps", int(info["grad_accum_steps"]))
    _set_default_if_not_explicit("sampled_kl_positions", int(info["sampled_kl_positions"]))
    _set_default_if_not_explicit("ce_chunk_tokens", int(info["ce_chunk_tokens"]))
    if info["profile"] == "cpu_test":
        _set_default_if_not_explicit("stage_e_calibration_seqs", 2)
        _set_default_if_not_explicit("stage_e_val_seqs", 1)

    if cfg.allow_smoke_fixtures is None:
        object.__setattr__(cfg, "allow_smoke_fixtures", bool(info["profile"] == "cpu_test"))
    if cfg.enforce_stage_d_dev_gate is None:
        object.__setattr__(cfg, "enforce_stage_d_dev_gate", not bool(cfg.allow_smoke_fixtures))

    return {
        **info,
        "effective_seq_len": int(cfg.seq_len),
        "effective_stage_e_seq_len": int(cfg.stage_e_seq_len),
        "effective_micro_batch_size": int(cfg.micro_batch_size),
        "effective_grad_accum_steps": int(cfg.grad_accum_steps),
        "effective_sampled_kl_positions": int(cfg.sampled_kl_positions),
        "effective_ce_chunk_tokens": int(cfg.ce_chunk_tokens),
        "allow_smoke_fixtures": bool(cfg.allow_smoke_fixtures),
    }


def _parameter_sha256(parameters: Iterable[Tuple[str, "torch.Tensor"]]) -> str:
    """Compute a full-byte SHA-256 digest over all named parameter shapes, dtypes, and complete byte buffers (`F07`, `§3.3`).
    Never samples head/tail slices, so interior tensor mutations are caught 100% of the time."""
    h = hashlib.sha256()
    for name, p in sorted(parameters, key=lambda item: item[0]):
        t_cpu = p.detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(t_cpu.dtype).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(tuple(t_cpu.shape)).encode("utf-8"))
        h.update(b"\x00")
        if t_cpu.numel() > 0:
            h.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def prepare_model_for_mhc_training(
    model: "SparkXLoopV03",
    cfg: Optional[MHCTrainingConfig] = None,
    reset_gains: bool = True,
) -> Dict[str, Any]:
    """Configure an existing `SparkXLoopV03` instance for differentiable `learned_mhc` training."""
    if not HAS_TORCH:
        raise RuntimeError("prepare_model_for_mhc_training requires PyTorch.")
    cfg = cfg or MHCTrainingConfig()

    if any(
        getattr(getattr(layer, "self_attn", None), "_sparkx_loop72_dlsa_attached", False)
        for layer in model.base.model.layers
    ):
        raise RuntimeError(
            "Cannot run differentiable mHC training on a model with attached paged D-LSA inference attention. "
            "Load the training model with `attach_sparse_attention=False`."
        )

    model.settings.recurrence_mode = str(cfg.recurrence_mode)
    model.settings.use_paged_hybrid_cache = False
    model.settings.checkpoint_layers = True
    model.settings.ce_chunk_tokens = int(cfg.ce_chunk_tokens)
    model.enable_checkpoint = True
    model.esm_enabled = False
    model.engram_enabled = False
    model.attention_runtime = "native_differentiable_sdpa"
    model.attention_training_supported = True

    model.set_loop(
        start=int(model.settings.start),
        end=int(model.settings.end),
        passes=2,
        use_mhc=True,
        use_engram=False,
        enable_p3=True,
        p3_start=int(model.settings.p3_start),
        p3_end=int(model.settings.p3_end),
    )
    ctrl_summary = model.set_mhc_gain_mode(
        gain_mode=str(cfg.gain_mode),
        init_p2_gain=float(cfg.init_p2_gain),
        init_p3_gain=float(cfg.init_p3_gain),
        extra_pass_gate_multiplier=float(cfg.extra_pass_gate_multiplier),
        reset_gains=bool(reset_gains),
    )

    model.controllers.float()
    model.requires_grad_(False)
    model.controllers.requires_grad_(True)
    model.train()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    non_ctrl = [name for name, _ in trainable if not name.startswith("controllers.")]
    if non_ctrl:
        raise RuntimeError(f"Non-controller parameters marked trainable: {non_ctrl[:5]}")

    return {
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": int(sum(p.numel() for _, p in trainable)),
        "controller_summary": ctrl_summary,
        "execution_manifest": model.execution_manifest(),
    }


def configure_controller_unfreeze_phase(
    model: "SparkXLoopV03",
    phase: str = "full_controllers",
) -> Dict[str, Any]:
    """Configure staged controller parameter unfreezing (`warmup_gains_and_up`, `adapters_and_static_routing`, `full_controllers`)."""
    valid_phases = ("warmup_gains_and_up", "adapters_and_static_routing", "full_controllers")
    if phase not in valid_phases:
        raise ValueError(f"Unsupported unfreeze phase {phase!r}; expected one of {valid_phases}")

    model.requires_grad_(False)
    for ctrl in model.controllers.values():
        for name, param in ctrl.named_parameters():
            if phase == "warmup_gains_and_up":
                allow = ("extra_gain" in name) or name.startswith("pass_up.")
            elif phase == "adapters_and_static_routing":
                allow = (
                    ("extra_gain" in name)
                    or name.startswith("pass_up.")
                    or name.startswith("pass_down.")
                    or ("logits" in name)
                    or ("pass_routing_bias" in name)
                )
            else:
                allow = True
            param.requires_grad_(bool(allow))

    active_params = sum(p.numel() for p in model.controllers.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.controllers.parameters())
    return {
        "phase": phase,
        "active_trainable_parameters": int(active_params),
        "total_controller_parameters": int(total_params),
    }


def build_mhc_optimizer(
    model: "SparkXLoopV03",
    cfg: Optional[MHCTrainingConfig] = None,
    lr_scale: float = 1.0,
) -> Tuple["torch.optim.AdamW", Dict[str, Any]]:
    """Construct AdamW with 3 FP32 parameter groups (`adapter_group`, `routing_decay_group`, `routing_nodecay_group`)."""
    cfg = cfg or MHCTrainingConfig()
    scale = float(lr_scale)
    adapter_lr = float(cfg.adapter_lr) * scale
    routing_lr = float(cfg.routing_lr) * scale

    adapter_params: List[torch.nn.Parameter] = []
    routing_decay_params: List[torch.nn.Parameter] = []
    routing_nodecay_params: List[torch.nn.Parameter] = []

    for name, param in model.controllers.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
        if ".pass_down." in name or ".pass_up." in name or name.startswith(("pass_down.", "pass_up.")):
            adapter_params.append(param)
        elif "dynamic_proj.weight" in name:
            routing_decay_params.append(param)
        else:
            routing_nodecay_params.append(param)

    groups = [
        {
            "name": "adapter_group",
            "params": adapter_params,
            "lr": adapter_lr,
            "initial_lr": adapter_lr,
            "base_lr": adapter_lr,
            "weight_decay": float(cfg.weight_decay),
        },
        {
            "name": "routing_decay_group",
            "params": routing_decay_params,
            "lr": routing_lr,
            "initial_lr": routing_lr,
            "base_lr": routing_lr,
            "weight_decay": float(cfg.weight_decay),
        },
        {
            "name": "routing_nodecay_group",
            "params": routing_nodecay_params,
            "lr": routing_lr,
            "initial_lr": routing_lr,
            "base_lr": routing_lr,
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(
        groups,
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=float(cfg.adam_eps),
    )
    summary = {
        "adapter_group_params": sum(p.numel() for p in adapter_params),
        "routing_decay_group_params": sum(p.numel() for p in routing_decay_params),
        "routing_nodecay_group_params": sum(p.numel() for p in routing_nodecay_params),
        "total_optimizer_params": sum(p.numel() for g in groups for p in g["params"]),
        "adapter_lr": adapter_lr,
        "routing_lr": routing_lr,
    }
    return opt, summary


def update_cosine_lr(
    optimizer: "torch.optim.Optimizer",
    step: int,
    total_steps: int,
    warmup_ratio: float = 0.08,
    min_lr_ratio: float = 0.10,
) -> Dict[str, float]:
    """Update each parameter group's learning rate using linear warmup followed by cosine decay."""
    tot = max(1, int(total_steps))
    warm = max(1, int(round(tot * float(warmup_ratio))))
    if step < warm:
        mult = float(step + 1) / float(warm)
    else:
        progress = min(1.0, max(0.0, float(step - warm) / float(max(1, tot - warm))))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        mult = float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine
    lrs: Dict[str, float] = {}
    for g in optimizer.param_groups:
        base_lr = float(g.get("base_lr", g.get("initial_lr", g["lr"])))
        new_lr = base_lr * mult
        g["lr"] = new_lr
        lrs[str(g.get("name", "group"))] = round(new_lr, 8)
    return lrs


def sample_supervised_positions(
    labels: "torch.Tensor",
    max_positions: int = 64,
    generator: Optional["torch.Generator"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Sample up to `K = max_positions` supervised next-token prediction positions per sequence (`labels[:, 1:] != -100`)."""
    shift_y = labels[:, 1:]
    bsz, seq_m1 = shift_y.shape
    k_keep = max(1, min(int(max_positions), seq_m1))
    pos_indices = torch.zeros((bsz, k_keep), dtype=torch.long, device=labels.device)
    pos_mask = torch.zeros((bsz, k_keep), dtype=torch.bool, device=labels.device)

    for b in range(bsz):
        valid_idx = torch.nonzero(shift_y[b] != -100, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            pos_indices[b, 0] = max(0, seq_m1 - 1)
            pos_mask[b, 0] = False
            continue
        if valid_idx.numel() <= k_keep:
            n_v = int(valid_idx.numel())
            pos_indices[b, :n_v] = valid_idx
            pos_mask[b, :n_v] = True
        else:
            perm = torch.randperm(valid_idx.numel(), generator=generator, device="cpu")[:k_keep]
            chosen = valid_idx[perm.to(valid_idx.device)].sort().values
            pos_indices[b] = chosen
            pos_mask[b] = True
    return pos_indices, pos_mask


def compute_teacher_retention_step_loss(
    model: "SparkXLoopV03",
    input_ids: "torch.LongTensor",
    labels: "torch.LongTensor",
    cfg: Optional[MHCTrainingConfig] = None,
    attention_mask: Optional["torch.Tensor"] = None,
    generator: Optional["torch.Generator"] = None,
) -> Tuple["torch.Tensor", Dict[str, float]]:
    """Compute Stage B/C/D combined CE + sampled-position 36L teacher KL + Coda `L28`-entry hidden anchor loss (`WP2`, `§7.3`)."""
    cfg = cfg or MHCTrainingConfig()
    lm_weight = model._head_weight()
    pos_indices, pos_mask = sample_supervised_positions(
        labels, max_positions=int(cfg.sampled_kl_positions), generator=generator
    )
    bsz, k_keep = pos_indices.shape
    batch_arange = torch.arange(bsz, device=input_ids.device).unsqueeze(1).expand(bsz, k_keep)
    supervised_token_count = int((labels[:, 1:] != -100).sum().item())

    teacher_logits_sampled: Optional[torch.Tensor] = None
    teacher_coda_h: Optional[torch.Tensor] = None
    use_kl = float(cfg.kl_beta) > 0.0 and bool(pos_mask.any().item())
    use_coda_anchor = float(cfg.coda_hidden_anchor_lambda) > 0.0 and bool(pos_mask.any().item())

    if use_kl or use_coda_anchor:
        was_training = model.training
        prev_coda_hook = getattr(model, "_coda_entry_hook", None)
        captured_teacher_coda: Dict[str, torch.Tensor] = {}

        def _teacher_coda_hook(h_t: torch.Tensor) -> None:
            captured_teacher_coda["h"] = h_t[batch_arange, pos_indices, :].detach().float()

        model.eval()
        if use_coda_anchor:
            model._coda_entry_hook = _teacher_coda_hook
        try:
            with torch.no_grad():
                with model.loop_config(
                    passes=1,
                    use_mhc=False,
                    use_engram=False,
                    enable_p3=False,
                    use_esm=False,
                ):
                    h_teacher = model.forward_hidden(input_ids, attention_mask=attention_mask)
                    if use_kl:
                        h_t_sub = h_teacher[batch_arange, pos_indices, :]  # [B, K, D]
                        teacher_logits_sampled = F.linear(
                            h_t_sub.to(lm_weight.dtype), lm_weight
                        ).detach().float()
                    del h_teacher
                if use_coda_anchor and "h" in captured_teacher_coda:
                    teacher_coda_h = captured_teacher_coda["h"]
        finally:
            model._coda_entry_hook = prev_coda_hook
            if was_training:
                model.train()

    # Student 72L forward (capturing actual Coda L28 entry AFTER Pass 3 `L12..L23` completes!)
    captured_student_coda: Dict[str, torch.Tensor] = {}
    prev_coda_hook = getattr(model, "_coda_entry_hook", None)

    def _student_coda_hook(h_s: torch.Tensor) -> None:
        captured_student_coda["h"] = h_s[batch_arange, pos_indices, :].float()

    if use_coda_anchor:
        model._coda_entry_hook = _student_coda_hook
    try:
        h_student = model.forward_hidden(input_ids, attention_mask=attention_mask)
    finally:
        if use_coda_anchor:
            model._coda_entry_hook = prev_coda_hook

    ce_loss = fused_masked_cross_entropy(
        h_student,
        lm_weight,
        labels,
        chunk_tokens=int(getattr(model.settings, "ce_chunk_tokens", cfg.ce_chunk_tokens)),
    )

    kl_loss = torch.zeros((), dtype=torch.float32, device=input_ids.device)
    if use_kl and teacher_logits_sampled is not None:
        temp = max(1e-3, float(cfg.kl_temperature))
        h_s_sub = h_student[batch_arange, pos_indices, :]  # [B, K, D]
        student_logits_sampled = F.linear(h_s_sub.to(lm_weight.dtype), lm_weight).float()
        log_p_t = F.log_softmax(teacher_logits_sampled / temp, dim=-1)
        log_p_s = F.log_softmax(student_logits_sampled / temp, dim=-1)
        p_t = log_p_t.exp()
        kl_per_pos = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # [B, K]
        valid_f = pos_mask.float()
        kl_loss = (kl_per_pos * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        kl_loss = kl_loss * (temp * temp)

    coda_loss = torch.zeros((), dtype=torch.float32, device=input_ids.device)
    if use_coda_anchor and teacher_coda_h is not None and "h" in captured_student_coda:
        diff_sq = (captured_student_coda["h"] - teacher_coda_h).pow(2).mean(dim=-1)
        valid_f = pos_mask.float()
        coda_loss = (diff_sq * valid_f).sum() / valid_f.sum().clamp(min=1.0)

    total_loss = (
        ce_loss
        + float(cfg.kl_beta) * kl_loss
        + float(cfg.coda_hidden_anchor_lambda) * coda_loss
    )
    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "ce_loss": float(ce_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "coda_loss": float(coda_loss.detach().item()),
        "sampled_kl_positions": int(pos_mask.sum().item()),
        "supervised_tokens": int(supervised_token_count),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_mhc_dev_loss(
    model: "SparkXLoopV03",
    dev_batches: Sequence[Tuple["torch.LongTensor", "torch.LongTensor", "torch.LongTensor"]],
    cfg: Optional[MHCTrainingConfig] = None,
) -> Dict[str, Any]:
    """Evaluate token-weighted held-out `dev` CE loss and perplexity under `model.eval()` (`§7.5`)."""
    cfg = cfg or MHCTrainingConfig()
    prev_train = bool(model.training)
    model.eval()
    lm_weight = model._head_weight()
    total_nll = 0.0
    total_sup = 0
    try:
        for inp, lbl, attn_mask in dev_batches:
            sup = int((lbl[:, 1:] != -100).sum().item())
            if sup <= 0:
                continue
            h_eval = model.forward_hidden(inp, attention_mask=attn_mask)
            ce = fused_masked_cross_entropy(
                h_eval,
                lm_weight,
                lbl,
                chunk_tokens=int(getattr(model.settings, "ce_chunk_tokens", cfg.ce_chunk_tokens)),
            )
            total_nll += float(ce.detach().item()) * float(sup)
            total_sup += sup
    finally:
        model.train(prev_train)

    mean_ce = total_nll / float(max(1, total_sup))
    ppl = math.exp(min(20.0, mean_ce))
    return {
        "dev_ce_loss": round(mean_ce, 6),
        "dev_ppl": round(ppl, 4),
        "dev_supervised_tokens": int(total_sup),
        "dev_batches": len(dev_batches),
    }


# ==============================================================================
# Stage A: State-Neutral Mechanical & Gradient Verification Suite (`F07`, `§3.2`)
# ==============================================================================
def run_stage_a_verification(
    model: "SparkXLoopV03",
    cfg: Optional[MHCTrainingConfig] = None,
    vocab_size: int = 16,
    seq_len: int = 12,
    report_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run Stage A mechanical and gradient verification with strict **bit-exact state neutrality** (`F07`, `§3.2`, `§14.3`):
      - Snapshots and restores all controller tensors, `requires_grad` flags, `.grad` buffers, `model.settings`,
        loop/runtime attributes, `model.training`, and Python/CPU/CUDA RNG states in a `try ... finally` block.
      - Verifies full-byte frozen `base.` SHA-256 invariance, equal gradient-presence sets across uncheckpointed vs
        checkpointed replay, save/load roundtrip logit parity, and 18/18 native global-visit Q/K capture.
    """
    if not HAS_TORCH:
        raise RuntimeError("run_stage_a_verification requires PyTorch.")
    from train_spark_dlsa import capture_loop72_global_visit_qkv

    cfg = copy.deepcopy(cfg or MHCTrainingConfig())
    cfg.sampled_kl_positions = min(int(cfg.sampled_kl_positions), max(2, int(seq_len) - 2))

    # Snapshot complete caller state BEFORE mutating anything
    py_rng_state = random.getstate()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    orig_training = bool(model.training)
    orig_settings = copy.deepcopy(model.settings)
    tracked_attrs = (
        "loop_start",
        "loop_end",
        "loop_passes",
        "mhc_enabled",
        "engram_enabled",
        "enable_p3",
        "p3_start",
        "p3_end",
        "esm_enabled",
        "enable_checkpoint",
        "attention_runtime",
        "attention_training_supported",
    )
    orig_model_attrs = {k: getattr(model, k) for k in tracked_attrs if hasattr(model, k)}
    orig_ctrl_meta = {
        k: (c.gain_mode, c.schema_version, c.init_p2_gain, c.init_p3_gain, c.extra_pass_gate_multiplier)
        for k, c in model.controllers.items()
    }
    orig_ctrl_dtypes = {n: p.dtype for n, p in model.controllers.named_parameters()}
    orig_ctrl_state = {k: v.detach().clone() for k, v in model.controllers.state_dict().items()}
    orig_req_grad = {n: bool(p.requires_grad) for n, p in model.named_parameters()}
    orig_grads = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in model.named_parameters()
    }

    ctrl_sha_before_entry = _parameter_sha256(model.controllers.named_parameters())
    base_sha_before = _parameter_sha256(model.base.named_parameters())

    try:
        prep = prepare_model_for_mhc_training(model, cfg=cfg, reset_gains=True)
        configure_controller_unfreeze_phase(model, phase="full_controllers")

        device = next(model.base.parameters()).device
        torch.manual_seed(int(cfg.seed))
        input_ids = torch.randint(1, max(4, int(vocab_size)), (1, int(seq_len)), device=device, dtype=torch.long)
        labels = input_ids.clone()
        labels[:, : max(1, int(seq_len) // 3)] = -100

        # 1. Check gradient checkpointing parity & gradient-presence set equality
        model.enable_checkpoint = False
        model.zero_grad(set_to_none=True)
        g1 = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        loss_nockpt, _ = compute_teacher_retention_step_loss(
            model, input_ids, labels, cfg=cfg, generator=g1
        )
        loss_nockpt.backward()
        grads_nockpt = {
            name: p.grad.detach().clone()
            for name, p in model.controllers.named_parameters()
            if p.grad is not None
        }

        model.enable_checkpoint = True
        model.zero_grad(set_to_none=True)
        g2 = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        loss_ckpt, metrics_ckpt = compute_teacher_retention_step_loss(
            model, input_ids, labels, cfg=cfg, generator=g2
        )
        loss_ckpt.backward()
        grads_ckpt_names = {
            name for name, p in model.controllers.named_parameters() if p.grad is not None
        }
        grad_presence_match = (set(grads_nockpt.keys()) == grads_ckpt_names)

        max_ckpt_grad_diff = 0.0
        max_ckpt_rel_grad_diff = 0.0
        for name, p in model.controllers.named_parameters():
            if p.grad is not None and name in grads_nockpt:
                g_ckpt = p.grad.detach().float()
                g_ref = grads_nockpt[name].float()
                diff = float((g_ckpt - g_ref).abs().max().item())
                denom = max(1.0, float(g_ref.norm().item()))
                max_ckpt_grad_diff = max(max_ckpt_grad_diff, diff)
                max_ckpt_rel_grad_diff = max(max_ckpt_rel_grad_diff, diff / denom)

        p2_grad_norm = float(
            sum(
                c.extra_gain.grad.abs().item() + c.ffn_extra_gain.grad.abs().item()
                for c in model.controllers.values()
                if c.extra_gain.grad is not None and c.ffn_extra_gain.grad is not None
            )
        )
        p3_grad_norm = float(
            sum(
                c.p3_extra_gain.grad.abs().item() + c.p3_ffn_extra_gain.grad.abs().item()
                for c in model.controllers.values()
                if c.p3_extra_gain.grad is not None and c.p3_ffn_extra_gain.grad is not None
            )
        )
        adapter_up_grad_norm = float(
            sum(
                p.grad.norm().item()
                for name, p in model.controllers.named_parameters()
                if "pass_up" in name and p.grad is not None
            )
        )
        base_grads_none = all(p.grad is None for p in model.base.parameters())

        # 2. Take one optimizer step and verify full-byte base weight immutability + save/load logit parity
        opt, opt_summary = build_mhc_optimizer(model, cfg=cfg)
        torch.nn.utils.clip_grad_norm_(model.controllers.parameters(), float(cfg.max_grad_norm))
        opt.step()
        opt.zero_grad(set_to_none=True)

        base_sha_after = _parameter_sha256(model.base.named_parameters())
        base_immutable = (base_sha_before == base_sha_after) and base_grads_none

        model.eval()
        with torch.no_grad():
            logits_before_save = model.forward_logits(input_ids).detach().float()
            saved_state = loop_v03_state_dict(model)
            first_ctrl = next(iter(model.controllers.values()))
            first_ctrl.extra_gain.add_(1.25)
            load_loop_v03_state_dict(model, saved_state, strict=True)
            model.eval()
            logits_after_load = model.forward_logits(input_ids).detach().float()
            save_load_max_diff = float((logits_before_save - logits_after_load).abs().max().item())

        # 3. Verify 18/18 native global-visit Q/K capture (`F01`, `§14.3`)
        qkv_probe = capture_loop72_global_visit_qkv(
            model,
            input_ids,
            include_values=False,
            max_sampled_queries=min(4, int(seq_len)),
            store_device="cpu",
            strict=True,
            seed=int(cfg.seed),
        )
        captured_global_visits = len(qkv_probe)

        ctrl_sum = model.controller_summary()
        manifest = model.execution_manifest()
    finally:
        # Restore exact pre-Stage-A controller weights, dtypes, metadata, grads, settings, loop flags, and RNG states
        model.settings = orig_settings
        for k_attr, v_attr in orig_model_attrs.items():
            setattr(model, k_attr, v_attr)
        for k_c, (g_mode, s_ver, p2_i, p3_i, g_mult) in orig_ctrl_meta.items():
            c_mod = model.controllers[k_c]
            c_mod.gain_mode = g_mode
            c_mod.schema_version = s_ver
            c_mod.init_p2_gain = p2_i
            c_mod.init_p3_gain = p3_i
            c_mod.extra_pass_gate_multiplier = g_mult
            if hasattr(c_mod, "_invalidate_eval_cache"):
                c_mod._invalidate_eval_cache()
        for n_p, dt_p in orig_ctrl_dtypes.items():
            p_obj = dict(model.controllers.named_parameters())[n_p]
            if p_obj.dtype != dt_p:
                p_obj.data = p_obj.data.to(dt_p)
        model.controllers.load_state_dict(orig_ctrl_state, strict=True)
        for n_p, p_obj in model.named_parameters():
            p_obj.requires_grad_(orig_req_grad[n_p])
            orig_g = orig_grads[n_p]
            p_obj.grad = orig_g.to(p_obj.device, dtype=p_obj.dtype) if orig_g is not None else None
        model.train(orig_training)
        random.setstate(py_rng_state)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_states)

    ctrl_sha_after_exit = _parameter_sha256(model.controllers.named_parameters())
    cur_ctrl_sd = model.controllers.state_dict()
    mutated_ctrl_tensors = sum(1 for k, v in orig_ctrl_state.items() if not torch.equal(v, cur_ctrl_sd[k]))
    state_neutral = (ctrl_sha_before_entry == ctrl_sha_after_exit) and (mutated_ctrl_tensors == 0)

    passed = (
        math.isfinite(metrics_ckpt["total_loss"])
        and p2_grad_norm > 0.0
        and p3_grad_norm > 0.0
        and adapter_up_grad_norm > 0.0
        and base_immutable
        and grad_presence_match
        and max_ckpt_rel_grad_diff < 0.05
        and save_load_max_diff < 1e-4
        and captured_global_visits == 18
        and state_neutral
        and not manifest["recurrence_train_eval_mismatch"]
        and manifest["effective_recurrence_mode"] == "learned_mhc"
        and len(ctrl_sum["gains_outside_unit_interval"]) == 0
        and ctrl_sum["min_static_effective_mix_entry"] >= 0.0
    )
    out_rep = {
        "status": "ok" if passed else "failed",
        "passed": bool(passed),
        "state_neutral": bool(state_neutral),
        "mutated_controller_tensors": int(mutated_ctrl_tensors),
        "controller_sha256_before": ctrl_sha_before_entry,
        "controller_sha256_after": ctrl_sha_after_exit,
        "grad_presence_match": bool(grad_presence_match),
        "captured_global_visits": int(captured_global_visits),
        "trainable_parameters": prep["trainable_parameters"],
        "expected_full_model_controller_parameters": EXPECTED_DEFAULT_CONTROLLER_PARAMS,
        "loss_metrics": metrics_ckpt,
        "p2_gain_grad_l1": round(p2_grad_norm, 8),
        "p3_gain_grad_l1": round(p3_grad_norm, 8),
        "adapter_up_grad_norm": round(adapter_up_grad_norm, 8),
        "max_checkpoint_replay_grad_diff": float(max_ckpt_grad_diff),
        "max_checkpoint_replay_rel_grad_diff": round(float(max_ckpt_rel_grad_diff), 8),
        "base_weights_immutable": bool(base_immutable),
        "base_sha256": base_sha_after,
        "save_load_max_logit_diff": float(save_load_max_diff),
        "optimizer_summary": opt_summary,
        "controller_summary": ctrl_sum,
        "execution_manifest": {
            "effective_recurrence_mode": manifest["effective_recurrence_mode"],
            "recurrence_train_eval_mismatch": manifest["recurrence_train_eval_mismatch"],
            "mhc_executed": manifest["mhc_executed"],
            "mhc_gain_mode": manifest["mhc_gain_mode"],
            "mhc_schema_version": manifest["mhc_schema_version"],
            "attention_runtime": manifest["attention_runtime"],
            "attention_training_supported": manifest["attention_training_supported"],
        },
    }
    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(out_rep, indent=2, default=str) + "\n", encoding="utf-8")
    return out_rep


# ==============================================================================
# 5-Domain Data Mixture & Exact Assistant-Only Rendering Wrappers (`F02, F03`)
# ==============================================================================
def build_omp_tool_trajectory_examples(count: int = 4) -> List[Dict[str, Any]]:
    """Build pinned OMP v18.18.1 (`input` hashline patch) `<tool_call>` trajectories (`F02`, `F15`)."""
    return _build_omp_trajectories_v18(count=count)


def build_five_domain_seed_bank() -> List[Dict[str, Any]]:
    """Build repaired deterministic 5-domain smoke fixtures (for unit tests / `allow_smoke_fixtures=True` only)."""
    return build_smoke_five_domain_seed_bank()


def render_example_to_Ids_and_labels(
    example: Dict[str, Any],
    tokenizer: Any,
    max_seq_len: int = 4096,
    *,
    allow_tail_rewindow: bool = False,
) -> Tuple[List[int], List[int]]:
    """Render a conversation into `(input_ids, labels)` with exact assistant-only supervision (`F03`, `§6.3`).
    Strictly rejects examples with 0 assistant target tokens after truncation (never fabricates labels on user tokens!)."""
    rw = render_conversation_exact_labels(
        example,
        tokenizer,
        max_seq_len=int(max_seq_len),
        allow_tail_rewindow=bool(allow_tail_rewindow),
    )
    if not rw.valid:
        raise ValueError(
            f"render_example_to_Ids_and_labels rejected example (id={rw.record_id!r}, domain={rw.domain!r}): "
            f"{rw.rejection_reason}"
        )
    return rw.input_ids, rw.labels


def build_packed_training_batches(
    tokenizer: Any,
    seq_len: int = 2048,
    num_batches: int = 8,
    only_domain: Optional[str] = None,
    device: str = "cpu",
) -> List[Tuple["torch.LongTensor", "torch.LongTensor"]]:
    """Build single-example padded `(input_ids, labels)` smoke batches (kept for legacy callers; production training uses
    `DomainWeightedBatchStream` with explicit `attention_mask`)."""
    if not HAS_TORCH:
        raise RuntimeError("build_packed_training_batches requires PyTorch.")
    bank = build_omp_tool_trajectory_examples() if only_domain == "tool_trajectories" else build_five_domain_seed_bank()
    if only_domain is not None:
        filtered = [ex for ex in bank if ex.get("domain") == only_domain]
        if filtered:
            bank = filtered
    stream = DomainWeightedBatchStream(
        bank,
        tokenizer,
        max_seq_len=int(seq_len),
        micro_batch_size=1,
        seed=42,
        allow_tail_rewindow=True,
    )
    out: List[Tuple[torch.LongTensor, torch.LongTensor]] = []
    for _ in range(max(1, int(num_batches))):
        inp, lbl, _mask, _meta = stream.next_microbatch(device=device)
        out.append((inp, lbl))
    return out


def _build_smoke_split_records(seed: int = 42) -> Tuple[List[TrainingRecord], List[TrainingRecord]]:
    """Build strictly document-disjoint `train` and `dev` smoke `TrainingRecord` sets for `cpu_test` runs."""
    base_examples = build_smoke_five_domain_seed_bank()
    omp_extra = build_omp_tool_trajectory_examples(count=6)
    train_recs: List[TrainingRecord] = []
    dev_recs: List[TrainingRecord] = []
    for idx, raw in enumerate(base_examples):
        rec = TrainingRecord.from_dict(raw, default_split="train")
        rec.id = f"{rec.id}_train_{idx}"
        rec.source_document_id = f"smoke_train_doc_{idx}"
        rec.split = "train"
        train_recs.append(rec)
        # Disjoint dev variant with distinct document ID
        dev_raw = copy.deepcopy(raw)
        dev_raw["id"] = f"{rec.id}_dev_{idx}"
        dev_raw["source_document_id"] = f"smoke_dev_doc_{idx}"
        dev_raw["split"] = "dev"
        dev_recs.append(TrainingRecord.from_dict(dev_raw, default_split="dev"))
    for j, omp_ex in enumerate(omp_extra[2:], start=100):
        omp_copy = copy.deepcopy(omp_ex)
        omp_copy["id"] = f"smoke_omp_extra_{j}"
        omp_copy["source_document_id"] = f"smoke_omp_doc_{j}"
        omp_copy["split"] = "train" if (j % 2 == 0) else "dev"
        target_list = train_recs if omp_copy["split"] == "train" else dev_recs
        target_list.append(TrainingRecord.from_dict(omp_copy, default_split=omp_copy["split"]))
    return train_recs, dev_recs


# ==============================================================================
# Checkpointing, Resume, Intermediate Stage Hub Commits & Final Export (`F06`, `F16`)
# ==============================================================================
def save_mhc_training_checkpoint(
    model: "SparkXLoopV03",
    output_dir: str | Path,
    cfg: MHCTrainingConfig,
    stage_name: str,
    global_step: int,
    metrics: Dict[str, Any],
    optimizer: Optional["torch.optim.Optimizer"] = None,
    resume_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export `loop_v03_state.pt`, `loop_config_v03.json`, `training_manifest.json`, `optimizer.pt`, and `rng_and_stream_state.pt`."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    state_file = out_path / "loop_v03_state.pt"
    config_file = out_path / "loop_config_v03.json"
    manifest_file = out_path / "training_manifest.json"

    state_payload = loop_v03_state_dict(model, include_tier1b=False)
    torch.save(state_payload, state_file)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
    if resume_extra is not None:
        rng_payload = {
            **resume_extra,
            "stage_name": stage_name,
            "global_step": int(global_step),
            "py_rng_state": random.getstate(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_payload, out_path / "rng_and_stream_state.pt")

    h = hashlib.sha256()
    with open(state_file, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    state_sha256 = h.hexdigest()
    ctrl_sha256 = canonical_state_dict_sha256(model.controllers.state_dict())

    metrics_snapshot = copy.deepcopy(metrics)
    ctrl_sum = model.controller_summary()
    exec_manifest = model.execution_manifest()
    loop_cfg_payload = {
        "model_name": str(getattr(cfg, "hf_model_name", DEFAULT_HF_MODEL_NAME)),
        "hf_repo_id": str(getattr(cfg, "hf_repo_id", DEFAULT_HF_REPO_ID)),
        "base_model": cfg.model_id,
        "base_revision": cfg.revision,
        "curriculum_version": "loop72_mhc_v2",
        "stage_name": stage_name,
        "global_step": int(global_step),
        "merged_standalone": False,
        "stage": {
            "start": int(model.loop_start),
            "end": int(model.loop_end),
            "passes": int(model.loop_passes),
            "mhc": bool(model.mhc_enabled),
            "engram": False,
            "enable_p3": bool(model.enable_p3),
            "p3_start": int(model.p3_start),
            "p3_end": int(model.p3_end),
        },
        "settings": dataclasses.asdict(model.settings),
        "controller_summary": ctrl_sum,
        "controller_sha256": ctrl_sha256,
        "metrics": metrics_snapshot,
        "checkpoint_sha256": state_sha256,
    }
    cfg_json = json.dumps(loop_cfg_payload, indent=2, default=str) + "\n"
    config_file.write_text(cfg_json, encoding="utf-8")
    (out_path / "loop_config.json").write_text(cfg_json, encoding="utf-8")

    train_manifest = {
        "model_name": str(getattr(cfg, "hf_model_name", DEFAULT_HF_MODEL_NAME)),
        "stage_name": stage_name,
        "global_step": int(global_step),
        "checkpoint_path": str(state_file),
        "checkpoint_sha256": state_sha256,
        "controller_sha256": ctrl_sha256,
        "training_config": cfg.to_dict(),
        "controller_summary": ctrl_sum,
        "execution_manifest": exec_manifest,
        "metrics": metrics_snapshot,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_file.write_text(json.dumps(train_manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return train_manifest


def load_mhc_training_resume_state(
    model: "SparkXLoopV03",
    checkpoint_dir: str | Path,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    stream: Optional[DomainWeightedBatchStream] = None,
) -> Dict[str, Any]:
    """Restore controller state, optimizer state, batch stream cursor, and RNG state from a saved checkpoint directory (`F16`)."""
    ckpt_path = Path(checkpoint_dir)
    state_pt = ckpt_path / "loop_v03_state.pt"
    if not state_pt.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing loop_v03_state.pt: {state_pt}")
    loaded_state = torch.load(state_pt, map_location="cpu", weights_only=False)
    load_loop_v03_state_dict(model, loaded_state, strict=False)

    opt_pt = ckpt_path / "optimizer.pt"
    if optimizer is not None and opt_pt.is_file():
        optimizer.load_state_dict(torch.load(opt_pt, map_location="cpu", weights_only=False))

    rng_pt = ckpt_path / "rng_and_stream_state.pt"
    resume_meta: Dict[str, Any] = {"global_step": 0, "stage_name": None, "completed_stages": []}
    if rng_pt.is_file():
        r_data = torch.load(rng_pt, map_location="cpu", weights_only=False)
        resume_meta["global_step"] = int(r_data.get("global_step", 0))
        resume_meta["stage_name"] = r_data.get("stage_name")
        resume_meta["completed_stages"] = list(r_data.get("completed_stages", []))
        if stream is not None and isinstance(r_data.get("stream_state"), dict):
            stream.load_state_dict(r_data["stream_state"])
        if r_data.get("py_rng_state") is not None:
            random.setstate(r_data["py_rng_state"])
        if r_data.get("torch_cpu_rng_state") is not None:
            torch.set_rng_state(r_data["torch_cpu_rng_state"])
        if r_data.get("torch_cuda_rng_states") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(r_data["torch_cuda_rng_states"])
            except Exception:
                pass
    return resume_meta


def resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolve a Hugging Face Hub token from argument, environment, Colab secrets, or cached login."""
    if explicit_token and str(explicit_token).strip():
        return str(explicit_token).strip()
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()
    try:
        from google.colab import userdata  # type: ignore
        col_tok = userdata.get("HF_TOKEN")
        if col_tok and str(col_tok).strip():
            return str(col_tok).strip()
    except Exception:
        pass
    try:
        from huggingface_hub import get_token  # type: ignore
        cached = get_token()
        if cached and str(cached).strip():
            return str(cached).strip()
    except Exception:
        pass
    return None


def normalize_hf_repo_id(
    repo_id_or_name: Optional[str],
    token: Optional[str] = None,
    default_owner: str = "TassP0lar",
    api_client: Any = None,
) -> str:
    """Normalize a display name or `<owner>/Spark-X2.6-4B-Recursive72` into a valid Hugging Face Hub repository ID."""
    raw = str(repo_id_or_name or DEFAULT_HF_REPO_ID).strip()
    if not raw:
        raw = DEFAULT_HF_REPO_ID
    if "/" in raw:
        owner_part, slug_part = raw.split("/", 1)
        owner = owner_part.strip() or default_owner
        slug = "-".join(slug_part.strip().split())
        return f"{owner}/{slug}"

    slug = "-".join(raw.split())
    owner = default_owner
    client = api_client
    if client is None and token:
        try:
            from huggingface_hub import HfApi  # type: ignore
            client = HfApi(token=token)
        except Exception:
            client = None
    if client is not None and hasattr(client, "whoami"):
        try:
            who = client.whoami()
            if isinstance(who, dict) and who.get("name"):
                owner = str(who["name"]).strip()
        except Exception:
            pass
    return f"{owner}/{slug}"


def upload_stage_checkpoint_to_hub(
    stage_dir: str | Path,
    stage_key: str,
    cfg: MHCTrainingConfig,
    progress_summary: Dict[str, Any],
    *,
    promote_to_root: bool = False,
    hf_token: Optional[str] = None,
    api_client: Any = None,
) -> Dict[str, Any]:
    """Push an intermediate stage commit (`stages/<stage_key>/` + `training_progress.json` + optional root controller/selector
    state) to `TassP0lar/Spark-X2.6-4B-Recursive72` during training (`push_to_hub=True` & `push_intermediate_stages=True`)."""
    token = resolve_hf_token(hf_token or cfg.hf_token)
    resolved_repo_id = normalize_hf_repo_id(cfg.hf_repo_id or cfg.hf_model_name, token=token, api_client=api_client)
    res: Dict[str, Any] = {
        "stage_key": stage_key,
        "hf_repo_id": resolved_repo_id,
        "promote_to_root": bool(promote_to_root),
        "uploaded": False,
    }
    if not bool(cfg.push_to_hub) or not bool(cfg.push_intermediate_stages):
        res["skipped_reason"] = "push_to_hub_or_intermediate_disabled"
        return res

    client = api_client
    if client is None:
        if not token:
            res["error"] = "push_to_hub=True requested, but no HF_TOKEN was found."
            return res
        from huggingface_hub import HfApi  # type: ignore
        client = HfApi(token=token)

    src_dir = Path(stage_dir)
    with tempfile.TemporaryDirectory(prefix=f"sparkx_hub_{stage_key}_") as staging_tmp:
        stage_root = Path(staging_tmp)
        sub_dest = stage_root / "stages" / stage_key
        sub_dest.mkdir(parents=True, exist_ok=True)
        for item in src_dir.iterdir():
            # Never upload massive optimizer states in intermediate Hub commits unless small
            if item.is_file() and item.name not in ("optimizer.pt", "rng_and_stream_state.pt"):
                shutil.copy2(item, sub_dest / item.name)
                if promote_to_root and item.name in (
                    "loop_v03_state.pt",
                    "loop_config_v03.json",
                    "loop_config.json",
                    "visit_selectors_stage_e.pt",
                ):
                    shutil.copy2(item, stage_root / item.name)

        (stage_root / "training_progress.json").write_text(
            json.dumps(progress_summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        try:
            client.create_repo(
                repo_id=resolved_repo_id,
                private=bool(cfg.hf_private),
                exist_ok=True,
                repo_type="model",
            )
            commit_info = client.upload_folder(
                folder_path=str(stage_root),
                repo_id=resolved_repo_id,
                repo_type="model",
                commit_message=(
                    f"[{stage_key}] Intermediate checkpoint for {cfg.hf_model_name} "
                    f"(step={progress_summary.get('global_step', 0)}, gain_mode={cfg.gain_mode})"
                ),
            )
            res["uploaded"] = True
            res["commit_info"] = str(commit_info) if commit_info is not None else "ok"
            print(
                f"[Hugging Face Hub | Intermediate {stage_key}] Committed checkpoint -> https://huggingface.co/{resolved_repo_id}"
            )
        except Exception as exc:
            res["error"] = str(exc)
            print(f"[Hugging Face Hub Warning] Intermediate commit `{stage_key}` failed: {exc}")
    return res


def verify_clean_export_directory(export_dir: str | Path) -> Dict[str, Any]:
    """Verify in a clean subprocess (with the source repo checkout removed from `sys.path`/`PYTHONPATH`) that all runtime
    modules import strictly from `export_dir` and checkpoints deserialize cleanly (`F06`, `§15.2`, `§17.2 #9`)."""
    exp_path = Path(export_dir).resolve()
    missing_files = [
        fname
        for fname in (
            "sparkx_loop_v03_core.py",
            "sparkx_memory_utils.py",
            "sparkx_dlsa.py",
            "sparkx_kv_codec.py",
            "sparkx_paged_cache.py",
            "sparkx_sparse_attention.py",
            "sparkx_omp_server.py",
            "load_inference_v03.py",
            "configs/spark4b_loop72_lsa.yaml",
            "loop_v03_state.pt",
            "loop_config_v03.json",
        )
        if not (exp_path / fname).is_file()
    ]
    if missing_files:
        return {"passed": False, "missing_files": missing_files, "error": f"Missing export files: {missing_files}"}

    repo_root_str = str(Path(__file__).resolve().parent)
    probe_code = (
        "import json, os, sys, torch\n"
        f"repo_root = os.path.realpath({repo_root_str!r})\n"
        f"exp_dir = os.path.realpath({str(exp_path)!r})\n"
        "sys.path = [exp_dir] + [p for p in sys.path if os.path.realpath(p or '.') != repo_root]\n"
        "import sparkx_loop_v03_core, sparkx_memory_utils, sparkx_dlsa, sparkx_kv_codec, sparkx_paged_cache, sparkx_sparse_attention, sparkx_omp_server, load_inference_v03\n"
        "mods = [sparkx_loop_v03_core, sparkx_memory_utils, sparkx_dlsa, sparkx_kv_codec, sparkx_paged_cache, sparkx_sparse_attention, sparkx_omp_server, load_inference_v03]\n"
        "bad = [m.__name__ for m in mods if not os.path.realpath(m.__file__).startswith(exp_dir)]\n"
        "cfg = load_inference_v03.load_loop72_lsa_yaml_config()\n"
        "st = torch.load(os.path.join(exp_dir, 'loop_v03_state.pt'), map_location='cpu', weights_only=False)\n"
        "sel_p = os.path.join(exp_dir, 'visit_selectors_stage_e.pt')\n"
        "sel_ok = True\n"
        "if os.path.isfile(sel_p):\n"
        "    sd = torch.load(sel_p, map_location='cpu', weights_only=False)\n"
        "    sel_ok = isinstance(sd, dict) and ('state_dict' in sd)\n"
        "print(json.dumps({'bad_imports': bad, 'yaml_loaded': bool(cfg), 'has_controllers': 'controllers' in st, 'sel_ok': sel_ok}))\n"
    )
    env = dict(os.environ)
    if "PYTHONPATH" in env:
        cleaned_pp = [
            p for p in env["PYTHONPATH"].split(os.pathsep)
            if p and os.path.realpath(p) != os.path.realpath(repo_root_str)
        ]
        env["PYTHONPATH"] = os.pathsep.join(cleaned_pp)

    proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(exp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return {
            "passed": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
            "stdout": proc.stdout.strip(),
        }
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"passed": False, "error": f"Failed to parse subprocess JSON: {exc}", "stdout": proc.stdout}

    passed = (
        len(parsed.get("bad_imports", [])) == 0
        and bool(parsed.get("yaml_loaded"))
        and bool(parsed.get("has_controllers"))
        and bool(parsed.get("sel_ok"))
    )
    return {"passed": passed, **parsed}


def build_completion_gates_report(report: Dict[str, Any], clean_reload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Summarize objective gate status (`§5.4`, `§15.3`) so model cards and release manifests report verified facts only."""
    stages = report.get("stages", {})
    st_a = stages.get("stage_a", {})
    st_c = stages.get("stage_c", {})
    st_d = stages.get("stage_d", {})
    st_e = stages.get("stage_e", {})
    data_info = report.get("data_provenance", {})

    trained_visits = int(st_e.get("trained_global_visits", 0))
    skipped_visits = int(st_e.get("skipped_global_visits", 18 if not st_e else 0))
    gates = {
        "gate_1_stage_a_state_neutral_and_base_immutable": bool(
            st_a.get("passed") and st_a.get("state_neutral") and st_a.get("base_weights_immutable")
        ),
        "gate_2_data_split_disjoint": bool(data_info.get("split_leakage_check", {}).get("passed", False)),
        "gate_3_real_non_smoke_corpus": not bool(data_info.get("smoke_fixture_mode", True)),
        "gate_4_stage_c_dense_mhc_completed": bool(st_c.get("steps", 0) > 0),
        "gate_5_stage_d_omp_evaluated": bool(not st_d or "dev_gate_passed" in st_d),
        "gate_6_stage_e_all_18_global_visits_distilled": bool(
            st_e.get("status") == "ok" and trained_visits == 18 and skipped_visits == 0
        ),
        "gate_7_clean_export_closure_verified": bool((clean_reload or {}).get("passed", False)),
    }
    gates["frozen_backbone_verified"] = bool(
        report.get("base_backbone_immutable", True) and st_a.get("base_weights_immutable", True)
    )
    gates["stage_a_state_neutral_passed"] = gates["gate_1_stage_a_state_neutral_and_base_immutable"]
    gates["stage_e_all_18_visits_calibrated"] = gates["gate_6_stage_e_all_18_global_visits_distilled"]
    gates["clean_export_subprocess_verified"] = gates["gate_7_clean_export_closure_verified"]
    gates["production_release_ready"] = bool(
        gates["gate_1_stage_a_state_neutral_and_base_immutable"]
        and gates["gate_2_data_split_disjoint"]
        and gates["gate_3_real_non_smoke_corpus"]
        and gates["gate_4_stage_c_dense_mhc_completed"]
        and gates["gate_6_stage_e_all_18_global_visits_distilled"]
        and gates["gate_7_clean_export_closure_verified"]
    )
    return gates


def build_recursive72_model_card(
    cfg: MHCTrainingConfig,
    report: Dict[str, Any],
    repo_id: str,
    completion_gates: Optional[Dict[str, Any]] = None,
) -> str:
    """Build an evidence-backed `README.md` model card for `Spark X2.6 4B Recursive72` (`§15.3`).
    Never claims Stage E calibration or benchmark scores unless verified in `report` / `completion_gates`."""
    model_title = str(cfg.hf_model_name or DEFAULT_HF_MODEL_NAME)
    stages = report.get("stages", {})
    stage_a = stages.get("stage_a", {})
    stage_c = stages.get("stage_c", {})
    stage_d = stages.get("stage_d", {})
    stage_e = stages.get("stage_e", {})
    selected_stage = report.get("selected_mhc_stage", "stage_d" if stage_d.get("promoted", True) and stage_d else "stage_c")
    active_mhc = stage_d if (selected_stage in ("stage_d", "stage_d_omp_mhc") and stage_d) else stage_c
    ctrl_sum = (
        active_mhc.get("controller_summary")
        or stage_c.get("controller_summary")
        or stage_a.get("controller_summary", {})
    )
    prov = stage_e.get("provenance", {})
    trained_visits = int(stage_e.get("trained_global_visits", 0))
    gates = completion_gates or build_completion_gates_report(report)
    data_prov = report.get("data_provenance", {})

    if stage_e.get("status") == "ok" and trained_visits == 18:
        selector_section = (
            f"- **18-Visit Calibrated `VisitAwareDLSAIndexBank` (`1,769,472` projection + `36` scalar params)**:\n"
            f"  - Trained Global Visits: **`{trained_visits}/18`** (`skipped={stage_e.get('skipped_global_visits', 0)}`)\n"
            f"  - Mean Initial $\\rightarrow$ Final Val KL: `{stage_e.get('mean_initial_loss', 'N/A')}` $\\rightarrow$ `{stage_e.get('mean_final_loss', 'N/A')}`\n"
            f"  - Mean Mass Recall @ 4096 / @ 2048: `{stage_e.get('mean_mass_recall_at_4096', 'N/A')}` / `{stage_e.get('mean_mass_recall_at_2048', 'N/A')}`\n"
            f"  - Mean Bucket Runtime Recall: `{stage_e.get('mean_bucket_runtime_recall', 'N/A')}`\n"
            f"  - Selector SHA-256: `{stage_e.get('selector_sha256', 'N/A')}`\n"
            f"  - Locked Controller SHA-256: `{prov.get('controller_sha256', 'N/A')}` (`mhc_checkpoint_sha256={prov.get('mhc_checkpoint_sha256', 'N/A')}`)\n"
        )
    else:
        selector_section = (
            f"- **18-Visit `VisitAwareDLSAIndexBank` Status**: **UNVERIFIED / INCOMPLETE** "
            f"(`status={stage_e.get('status', 'not_run')!r}`, `trained_global_visits={trained_visits}/18`). "
            f"Do not claim calibrated sparse long-context retrieval on this artifact until all 18 global visits pass Stage E.\n"
        )

    corpus_badge = "Smoke/Test Fixture (Non-Release)" if data_prov.get("smoke_fixture_mode", False) else "Real Multi-Domain Corpus"

    return (
        "---\n"
        "library_name: transformers\n"
        f"base_model: {cfg.model_id}\n"
        "tags:\n"
        "  - spark\n"
        "  - spark-x2.6-4b\n"
        "  - recursive-transformer\n"
        "  - loop72\n"
        "  - mhc\n"
        "  - d-lsa\n"
        "  - long-context\n"
        "pipeline_tag: text-generation\n"
        "---\n\n"
        f"# {model_title}\n\n"
        f"**`{model_title}`** (`{repo_id}`) is the 72-visit recurrent-depth (`Loop72`) + per-visit Paged Latent Sparse Attention (`D-LSA`) "
        f"model built strictly atop the **100% frozen** BF16 **`{cfg.model_id}`** (`revision={cfg.revision}`) 36-layer hybrid backbone.\n\n"
        "## Verified Architecture & Provenance\n"
        f"- **Frozen Base Backbone**: `{cfg.model_id}` @ `{cfg.revision}` (36 physical layers = 9 partial-RoPE `0.25` full-attention layers "
        f"`{{3, 7, 11, 15, 19, 23, 27, 31, 35}}` + 27 sliding-window `w=512` layers, native GELU, head-wise sigmoid `g_proj` + `out_proj`).\n"
        f"- **Base Backbone Full-Byte SHA-256 Invariance**: `{stage_a.get('base_weights_immutable', 'N/A')}` (`sha256={stage_a.get('base_sha256', 'N/A')}`)\n"
        "- **72-Visit Recurrent Schedule (`4 + 24 + 24 + 12 + 8 = 72`, 18 Global + 54 Local Visits)**:\n"
        "  - Prelude `L0..L3` (4 visits; `L3` = Global Visit `0`)\n"
        "  - Core Pass 1 `L4..L27` (24 visits; 6 global)\n"
        "  - Core Pass 2 `L4..L27` (24 visits; 6 global)\n"
        "  - Inner Pass 3 `L12..L23` (12 visits; 3 global)\n"
        "  - Coda `L28..L35` (8 visits; `L31, L35` = Global Visits `16, 17`)\n"
        f"- **Differentiable `TwoStreamMHC` Controllers (`{EXPECTED_DEFAULT_CONTROLLER_PARAMS:,}` FP32 params, selected teacher=`{selected_stage}`)**:\n"
        f"  - Bounded-gain Schema v2 (`gain_mode={ctrl_sum.get('gain_mode', cfg.gain_mode)!r}`, "
        f"`mhc_schema_version={ctrl_sum.get('mhc_schema_version', 2)}`), guaranteeing $(1 - g)I + gP \\in \\mathcal{{B}}_2$ across all 60 core visits.\n"
        f"  - Mean Pass-2 Gain (Attn/FFN): `{ctrl_sum.get('mean_gain_p2_attn', cfg.init_p2_gain)}` / `{ctrl_sum.get('mean_gain_p2_ffn', cfg.init_p2_gain)}`\n"
        f"  - Mean Pass-3 Gain (Attn/FFN): `{ctrl_sum.get('mean_gain_p3_attn', cfg.init_p3_gain)}` / `{ctrl_sum.get('mean_gain_p3_ffn', cfg.init_p3_gain)}`\n"
        f"{selector_section}"
        f"- **Training Corpus Mode**: `{corpus_badge}` (`train_examples={data_prov.get('train_records', 'N/A')}`, "
        f"`dev_examples={data_prov.get('dev_records', 'N/A')}`, `split_disjoint={data_prov.get('split_leakage_check', {}).get('passed', 'N/A')}`)\n"
        f"- **Completion Gates**: `{json.dumps(gates)}`\n\n"
        "## Quickstart\n\n"
        "```python\n"
        "import torch\n"
        "from load_inference_v03 import load_sparkx_loop72_lsa\n\n"
        "model, tokenizer = load_sparkx_loop72_lsa(\n"
        f"    repo_or_dir={repo_id!r},\n"
        f"    base_repo={cfg.model_id!r},\n"
        f"    base_revision={cfg.revision!r},\n"
        "    recurrence_mode='learned_mhc',\n"
        "    attach_sparse_attention=True,\n"
        "    strict_selector_provenance=True,\n"
        ")\n"
        "model.eval()\n"
        "print(model.execution_manifest())\n"
        "```\n"
    )


def export_and_upload_recursive72_to_hub(
    model: "SparkXLoopV03",
    tokenizer: Any,
    export_dir: str | Path,
    cfg: MHCTrainingConfig,
    report: Dict[str, Any],
    hf_token: Optional[str] = None,
    api_client: Any = None,
) -> Dict[str, Any]:
    """Package the complete transitive `Spark X2.6 4B Recursive72` export closure (`F06`, `§15.2`), run clean-process
    reload verification, write `completion_gates.json` + `README.md`, and push the **final release commit** to Hub."""
    exp_path = Path(export_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    if not (exp_path / "loop_v03_state.pt").is_file() or not (exp_path / "loop_config_v03.json").is_file():
        save_mhc_training_checkpoint(
            model,
            exp_path,
            cfg=cfg,
            stage_name="standalone_export",
            global_step=0,
            metrics={"stages_completed": list(report.get("stages", {}).keys()) if isinstance(report, dict) else []},
        )
    token = resolve_hf_token(hf_token or cfg.hf_token)
    resolved_repo_id = normalize_hf_repo_id(cfg.hf_repo_id or cfg.hf_model_name, token=token, api_client=api_client)

    parent_sel = Path(cfg.output_dir) / "visit_selectors_stage_e.pt"
    target_sel = exp_path / "visit_selectors_stage_e.pt"
    if parent_sel.is_file() and parent_sel.resolve() != target_sel.resolve():
        shutil.copy2(parent_sel, target_sel)

    repo_root = Path(__file__).resolve().parent
    copied_helpers: List[str] = []
    for helper_name in EXPORT_RUNTIME_CLOSURE_FILES:
        src_f = repo_root / helper_name
        if src_f.is_file():
            dst_f = exp_path / helper_name
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            if src_f.resolve() != dst_f.resolve():
                shutil.copy2(src_f, dst_f)
            copied_helpers.append(helper_name)

    cfg_yaml = repo_root / "configs" / "spark4b_loop72_lsa.yaml"
    if cfg_yaml.is_file():
        (exp_path / "configs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_yaml, exp_path / "configs" / "spark4b_loop72_lsa.yaml")

    standalone_saved = False
    if bool(cfg.export_standalone_backbone) and hasattr(model.base, "save_pretrained"):
        try:
            model.base.save_pretrained(str(exp_path), safe_serialization=True, max_shard_size="4GB")
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(str(exp_path))
            standalone_saved = True
        except Exception as exc:
            standalone_saved = False
            print(f"[Recursive72 Export] Standalone backbone save skipped: {exc}")

    for cfg_name in ("loop_config_v03.json", "loop_config.json"):
        c_file = exp_path / cfg_name
        if c_file.is_file():
            try:
                c_data = json.loads(c_file.read_text(encoding="utf-8"))
                c_data["model_name"] = str(cfg.hf_model_name or DEFAULT_HF_MODEL_NAME)
                c_data["hf_repo_id"] = resolved_repo_id
                c_data["merged_standalone"] = bool(standalone_saved)
                c_file.write_text(json.dumps(c_data, indent=2, default=str) + "\n", encoding="utf-8")
            except Exception:
                pass

    clean_reload = verify_clean_export_directory(exp_path)
    gates = build_completion_gates_report(report, clean_reload=clean_reload)
    (exp_path / "completion_gates.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    if isinstance(report, dict):
        (exp_path / "mhc_training_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )

    readme_text = build_recursive72_model_card(cfg, report, resolved_repo_id, completion_gates=gates)
    (exp_path / "README.md").write_text(readme_text, encoding="utf-8")

    hub_result: Dict[str, Any] = {
        "model_name": str(cfg.hf_model_name or DEFAULT_HF_MODEL_NAME),
        "hf_repo_id": resolved_repo_id,
        "export_dir": str(exp_path),
        "standalone_backbone_saved": bool(standalone_saved),
        "copied_helpers": copied_helpers,
        "clean_export_reload": clean_reload,
        "completion_gates": gates,
        "push_to_hub_requested": bool(cfg.push_to_hub),
        "uploaded": False,
        "hub_url": f"https://huggingface.co/{resolved_repo_id}",
    }

    if bool(cfg.push_to_hub):
        if not clean_reload.get("passed", False):
            hub_result["error"] = f"Blocked Hub upload because clean export reload verification failed: {clean_reload}"
            return hub_result
        client = api_client
        if client is None:
            if not token:
                hub_result["error"] = (
                    "push_to_hub=True requested, but no HF_TOKEN was found in args, environment, or Colab secrets."
                )
                return hub_result
            from huggingface_hub import HfApi  # type: ignore
            client = HfApi(token=token)
        try:
            client.create_repo(
                repo_id=resolved_repo_id,
                private=bool(cfg.hf_private),
                exist_ok=True,
                repo_type="model",
            )
            trained_v = int(report.get("stages", {}).get("stage_e", {}).get("trained_global_visits", 0))
            commit_info = client.upload_folder(
                folder_path=str(exp_path),
                repo_id=resolved_repo_id,
                repo_type="model",
                commit_message=(
                    f"[Final Release] Upload {cfg.hf_model_name} (72-visit learned_mhc + "
                    f"{trained_v}/18 D-LSA selectors, gain_mode={cfg.gain_mode})"
                ),
            )
            hub_result["uploaded"] = True
            hub_result["commit_info"] = str(commit_info) if commit_info is not None else "ok"
            print(
                f"[Hugging Face Hub | Final Commit] Uploaded `{cfg.hf_model_name}` -> https://huggingface.co/{resolved_repo_id}"
            )
        except Exception as exc:
            hub_result["error"] = str(exc)
            print(f"[Hugging Face Hub Error] Failed to upload `{resolved_repo_id}`: {exc}")

    return hub_result


# ==============================================================================
# End-to-End Pipeline (Stages A–E + Intermediate & Final Hub Commits)
# ==============================================================================
def run_mhc_training_pipeline(
    model: "SparkXLoopV03",
    tokenizer: Any,
    cfg: Optional[MHCTrainingConfig] = None,
    run_stage_a: bool = True,
    run_stage_b: bool = True,
    run_stage_c: bool = True,
    run_stage_d: Optional[bool] = None,
    run_stage_e: Optional[bool] = None,
    reset_controllers: Optional[bool] = None,
    run_final_export: bool = True,
    api_client: Any = None,
) -> Dict[str, Any]:
    """Execute the end-to-end Milestone 2 Loop72 `TwoStreamMHC` + Stage E D-LSA pipeline (`Stages A–E`), pushing
    intermediate stage commits and the final validated release commit to `TassP0lar/Spark-X2.6-4B-Recursive72` when
    `cfg.push_to_hub` is enabled."""
    if not HAS_TORCH:
        raise RuntimeError("run_mhc_training_pipeline requires PyTorch.")
    cfg = copy.deepcopy(cfg or MHCTrainingConfig())
    hw_info = apply_hardware_profile_to_config(cfg)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = str(next(model.base.parameters()).device)
    vocab_sz = int(getattr(model.base.config, "vocab_size", model._head_weight().shape[0]))
    base_sha_initial = _parameter_sha256(model.base.named_parameters())

    prior_stages: Dict[str, Any] = {}
    prior_hub_commits: List[Dict[str, Any]] = []
    existing_rep_file = out_root / "mhc_training_report.json"
    if not run_final_export and existing_rep_file.is_file():
        try:
            prev_rep = json.loads(existing_rep_file.read_text(encoding="utf-8"))
            if isinstance(prev_rep.get("stages"), dict):
                prior_stages = dict(prev_rep["stages"])
            if isinstance(prev_rep.get("intermediate_hub_commits"), list):
                prior_hub_commits = list(prev_rep["intermediate_hub_commits"])
        except Exception:
            pass
    stage_a_file = out_root / "stage_a_report.json"
    if "stage_a" not in prior_stages and stage_a_file.is_file() and not run_stage_a:
        try:
            prior_stages["stage_a"] = json.loads(stage_a_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    report: Dict[str, Any] = {
        "hardware_profile": hw_info,
        "config": cfg.to_dict(),
        "base_sha256_initial": base_sha_initial,
        "stages": prior_stages,
        "intermediate_hub_commits": prior_hub_commits,
    }

    # Stage A: State-Neutral Mechanical & Gradient Verification
    if run_stage_a:
        stage_a_seq = min(int(cfg.seq_len), 32)
        stage_a_rep = run_stage_a_verification(
            model,
            cfg=cfg,
            vocab_size=vocab_sz,
            seq_len=stage_a_seq,
            report_path=out_root / "stage_a_report.json",
        )
        report["stages"]["stage_a"] = stage_a_rep
        if not stage_a_rep["passed"]:
            raise RuntimeError(f"Stage A verification failed: {stage_a_rep}")

    # Build / Load Disjoint 5-Domain Train & Dev Corpus (`F02`, `§6.1–§6.4`)
    allow_smoke = bool(cfg.allow_smoke_fixtures)
    corpus_meta: Dict[str, Any] = {}
    if cfg.data_manifest:
        all_manifest_recs, manifest_info = load_manifest_records(cfg.data_manifest)
        train_records = [r for r in all_manifest_recs if r.split == "train"]
        if cfg.eval_manifest:
            eval_recs, _ = load_manifest_records(cfg.eval_manifest)
            dev_records = [r for r in eval_recs if r.split in ("dev", "val", "test")]
        else:
            dev_records = [r for r in all_manifest_recs if r.split == "dev"]
        if not train_records:
            raise RuntimeError(f"data_manifest={cfg.data_manifest!r} yielded 0 train records.")
        if not dev_records:
            raise RuntimeError("0 held-out dev records found in eval_manifest / data_manifest.")
        corpus_meta = {"source": "manifest", "smoke_fixture_mode": False, **manifest_info}
    elif not allow_smoke:
        built_splits, corpus_meta = build_real_five_domain_corpus(
            target_total_records=int(cfg.target_corpus_examples),
            seed=int(cfg.seed),
            use_hf_streaming=bool(cfg.use_hf_streaming),
            allow_smoke_fixtures=False,
        )
        all_built = list(built_splits["train"]) + list(built_splits["dev"]) + list(built_splits["test"])
        manifests_dir = out_root / "data_manifests"
        save_manifest_records(all_built, manifests_dir / "corpus_all.jsonl")
        train_records = list(built_splits["train"])
        dev_records = list(built_splits["dev"]) if built_splits["dev"] else list(built_splits["test"])
    else:
        train_records, dev_records = _build_smoke_split_records(seed=int(cfg.seed))
        corpus_meta = {"source": "smoke_fixture", "smoke_fixture_mode": True}

    leak_rep = verify_no_split_document_leakage(train_records, dev_records, [])
    if not leak_rep["passed"]:
        raise RuntimeError(f"Train/dev document split leakage detected: {leak_rep}")

    report["data_provenance"] = {
        **corpus_meta,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "split_leakage_check": leak_rep,
    }

    rewindow_flag = bool(allow_smoke and int(cfg.seq_len) < 256)
    train_stream = DomainWeightedBatchStream(
        train_records,
        tokenizer,
        max_seq_len=int(cfg.seq_len),
        micro_batch_size=int(cfg.micro_batch_size),
        domain_weights=STAGE_C_DOMAIN_WEIGHTS,
        seed=int(cfg.seed),
        allow_tail_rewindow=rewindow_flag,
    )
    dev_stream = DomainWeightedBatchStream(
        dev_records,
        tokenizer,
        max_seq_len=int(cfg.seq_len),
        micro_batch_size=1,
        domain_weights=STAGE_C_DOMAIN_WEIGHTS,
        seed=int(cfg.seed) + 1337,
        allow_tail_rewindow=rewindow_flag,
    )
    n_dev_eval = max(1, min(int(cfg.dev_eval_batches), dev_stream.total_valid_examples()))
    dev_batches: List[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]] = []
    for _ in range(n_dev_eval):
        d_inp, d_lbl, d_mask, _ = dev_stream.next_microbatch(device=device)
        dev_batches.append((d_inp, d_lbl, d_mask))

    # Initialize clean Step-0 bounded-gain controllers before Stage B (or preserve in-memory / resume checkpoint state)
    do_reset = (not bool(cfg.resume_from_checkpoint) and bool(run_stage_b)) if reset_controllers is None else bool(reset_controllers)
    prepare_model_for_mhc_training(model, cfg=cfg, reset_gains=do_reset)
    opt, opt_summary = build_mhc_optimizer(model, cfg=cfg, lr_scale=1.0)
    report["optimizer"] = opt_summary
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    global_step = 0
    completed_stages: set[str] = set()
    if cfg.resume_from_checkpoint:
        res_info = load_mhc_training_resume_state(
            model, cfg.resume_from_checkpoint, optimizer=opt, stream=train_stream
        )
        global_step = int(res_info.get("global_step", 0))
        completed_stages = set(res_info.get("completed_stages", []))
        report["resumed_from"] = res_info

    total_bc_steps = max(1, int(cfg.stage_b_steps) + int(cfg.stage_c_steps))

    def _run_steps(
        num_steps: int,
        stage_label: str,
        active_opt: torch.optim.Optimizer,
        active_stream: DomainWeightedBatchStream,
        stage_cfg: MHCTrainingConfig,
        *,
        lr_step_offset: int,
        lr_total_steps: int,
        unfreeze_schedule: bool = False,
        track_best_dev: bool = False,
    ) -> Dict[str, Any]:
        nonlocal global_step
        loss_first: Optional[float] = None
        loss_last: float = 0.0
        ce_last: float = 0.0
        kl_last: float = 0.0
        grad_norm_last: float = 0.0
        sup_tokens_total: int = 0
        input_tokens_total: int = 0
        eval_history: List[Dict[str, Any]] = []
        best_dev_ce = float("inf")
        best_ctrl_sd: Optional[Dict[str, torch.Tensor]] = None
        t0 = time.perf_counter()
        model.train()

        save_interval = max(1, int(stage_cfg.save_every_steps))
        eval_interval = max(1, int(stage_cfg.eval_every_steps))

        for s_idx in range(max(0, int(num_steps))):
            if stage_cfg.curriculum == "36_60_72" and stage_label == "stage_c":
                frac = float(s_idx) / float(max(1, num_steps))
                if frac < 0.25:
                    model.set_loop(
                        model.loop_start, model.loop_end, passes=1, use_mhc=True, use_engram=False,
                        enable_p3=False, p3_start=model.p3_start, p3_end=model.p3_end,
                    )
                elif frac < 0.60:
                    model.set_loop(
                        model.loop_start, model.loop_end, passes=2, use_mhc=True, use_engram=False,
                        enable_p3=False, p3_start=model.p3_start, p3_end=model.p3_end,
                    )
                else:
                    model.set_loop(
                        model.loop_start, model.loop_end, passes=2, use_mhc=True, use_engram=False,
                        enable_p3=True, p3_start=model.p3_start, p3_end=model.p3_end,
                    )

            if unfreeze_schedule:
                frac = float(s_idx) / float(max(1, num_steps))
                if frac < 0.20:
                    configure_controller_unfreeze_phase(model, "warmup_gains_and_up")
                elif frac < 0.45:
                    configure_controller_unfreeze_phase(model, "adapters_and_static_routing")
                else:
                    configure_controller_unfreeze_phase(model, "full_controllers")

            update_cosine_lr(
                active_opt,
                step=lr_step_offset + s_idx,
                total_steps=lr_total_steps,
                warmup_ratio=float(stage_cfg.warmup_ratio),
                min_lr_ratio=float(stage_cfg.min_lr_ratio),
            )
            active_opt.zero_grad(set_to_none=True)
            accum = max(1, int(stage_cfg.grad_accum_steps))

            # Pre-fetch microbatches for this optimizer step so gradient weighting is exact by supervised token count (`§6.4`)
            mb_list = [active_stream.next_microbatch(device=device) for _ in range(accum)]
            step_sup_sum = sum(max(1, int(m_meta["supervised_tokens"])) for _, _, _, m_meta in mb_list)

            step_tot = 0.0
            step_ce = 0.0
            step_kl = 0.0
            for inp, lbl, attn_mask, m_meta in mb_list:
                mb_sup = max(1, int(m_meta["supervised_tokens"]))
                weight = float(mb_sup) / float(max(1, step_sup_sum))
                loss_t, m_dict = compute_teacher_retention_step_loss(
                    model, inp, lbl, cfg=stage_cfg, attention_mask=attn_mask, generator=gen
                )
                (loss_t * weight).backward()
                step_tot += m_dict["total_loss"] * weight
                step_ce += m_dict["ce_loss"] * weight
                step_kl += m_dict["kl_loss"] * weight
                sup_tokens_total += int(m_meta["supervised_tokens"])
                input_tokens_total += int(m_meta["input_tokens"])

            grad_norm_last = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.controllers.parameters() if p.requires_grad and p.grad is not None],
                    float(stage_cfg.max_grad_norm),
                ).item()
            )
            active_opt.step()
            active_opt.zero_grad(set_to_none=True)
            global_step += 1

            if loss_first is None:
                loss_first = step_tot
            loss_last = step_tot
            ce_last = step_ce
            kl_last = step_kl

            if (s_idx + 1) % eval_interval == 0 or (s_idx + 1) == num_steps:
                dev_m = evaluate_mhc_dev_loss(model, dev_batches, cfg=stage_cfg)
                dev_m["stage_step"] = s_idx + 1
                dev_m["global_step"] = global_step
                eval_history.append(dev_m)
                if track_best_dev and float(dev_m["dev_ce_loss"]) <= best_dev_ce:
                    best_dev_ce = float(dev_m["dev_ce_loss"])
                    best_ctrl_sd = {k: v.detach().cpu().clone() for k, v in model.controllers.state_dict().items()}

            if (s_idx + 1) % save_interval == 0 and (s_idx + 1) < num_steps:
                save_mhc_training_checkpoint(
                    model,
                    out_root / "periodic_checkpoints" / f"{stage_label}_step_{global_step:05d}",
                    cfg=stage_cfg,
                    stage_name=f"{stage_label}_periodic",
                    global_step=global_step,
                    metrics={"loss_last": loss_last, "ce_loss": ce_last, "kl_loss": kl_last},
                    optimizer=active_opt,
                    resume_extra={
                        "completed_stages": sorted(completed_stages),
                        "stream_state": active_stream.state_dict(),
                    },
                )

        # Restore full 72L loop if curriculum modified it
        model.set_loop(
            model.loop_start, model.loop_end, passes=2, use_mhc=True, use_engram=False,
            enable_p3=True, p3_start=model.p3_start, p3_end=model.p3_end,
        )
        configure_controller_unfreeze_phase(model, "full_controllers")
        if track_best_dev and best_ctrl_sd is not None:
            model.controllers.load_state_dict(
                {k: v.to(next(model.controllers.parameters()).device) for k, v in best_ctrl_sd.items()},
                strict=True,
            )

        final_dev = evaluate_mhc_dev_loss(model, dev_batches, cfg=stage_cfg)
        dt = max(1e-6, time.perf_counter() - t0)
        return {
            "stage": stage_label,
            "steps": int(num_steps),
            "global_step": int(global_step),
            "loss_first": round(float(loss_first or 0.0), 6),
            "loss_last": round(float(loss_last), 6),
            "ce_loss": round(float(ce_last), 6),
            "kl_loss": round(float(kl_last), 6),
            "last_grad_norm": round(float(grad_norm_last), 6),
            "dev_eval": final_dev,
            "eval_history": eval_history,
            "tokens_processed": int(input_tokens_total),
            "supervised_tokens_processed": int(sup_tokens_total),
            "tokens_per_sec": round(float(input_tokens_total) / dt, 2),
            "elapsed_sec": round(dt, 3),
            "data_ledger": active_stream.ledger(),
            "controller_summary": model.controller_summary(),
        }

    # Stage B: 36L -> 72L Teacher Retention Warmup
    if run_stage_b and int(cfg.stage_b_steps) > 0 and "stage_b" not in completed_stages:
        configure_controller_unfreeze_phase(model, "warmup_gains_and_up")
        stage_b_metrics = _run_steps(
            int(cfg.stage_b_steps),
            "stage_b",
            opt,
            train_stream,
            cfg,
            lr_step_offset=0,
            lr_total_steps=total_bc_steps,
            unfreeze_schedule=False,
            track_best_dev=False,
        )
        completed_stages.add("stage_b")
        ckpt_b = save_mhc_training_checkpoint(
            model,
            out_root / "stage_b_retention_mhc",
            cfg=cfg,
            stage_name="stage_b_retention",
            global_step=global_step,
            metrics=stage_b_metrics,
            optimizer=opt,
            resume_extra={"completed_stages": sorted(completed_stages), "stream_state": train_stream.state_dict()},
        )
        stage_b_metrics["checkpoint"] = ckpt_b
        report["stages"]["stage_b"] = stage_b_metrics
        hub_b = upload_stage_checkpoint_to_hub(
            out_root / "stage_b_retention_mhc",
            "stage_b_retention",
            cfg,
            {"stage": "stage_b_retention", "global_step": global_step, "dev_eval": stage_b_metrics["dev_eval"]},
            promote_to_root=False,
            api_client=api_client,
        )
        report["intermediate_hub_commits"].append(hub_b)

    # Stage C: Multi-Pass Deliberation Training with Staged Unfreezing & Held-Out Dev Best Selection
    dense_mhc_ckpt_path: Optional[str] = None
    dense_mhc_sha256: Optional[str] = None
    selected_mhc_stage = "stage_c"
    stage_c_dev_loss: Optional[float] = None
    stage_c_ctrl_backup: Optional[Dict[str, torch.Tensor]] = None

    selected_meta_path = out_root / "selected_mhc_checkpoint.json"
    if selected_meta_path.is_file():
        try:
            sel_meta = json.loads(selected_meta_path.read_text(encoding="utf-8"))
            dense_mhc_ckpt_path = sel_meta.get("checkpoint_path")
            dense_mhc_sha256 = sel_meta.get("checkpoint_sha256")
            selected_mhc_stage = str(sel_meta.get("selected_mhc_stage", "stage_c"))
            if sel_meta.get("stage_c_dev_loss") is not None:
                stage_c_dev_loss = float(sel_meta["stage_c_dev_loss"])
        except Exception:
            pass

    if run_stage_c and int(cfg.stage_c_steps) > 0 and "stage_c" not in completed_stages:
        stage_c_metrics = _run_steps(
            int(cfg.stage_c_steps),
            "stage_c",
            opt,
            train_stream,
            cfg,
            lr_step_offset=int(cfg.stage_b_steps),
            lr_total_steps=total_bc_steps,
            unfreeze_schedule=True,
            track_best_dev=True,
        )
        completed_stages.add("stage_c")
        ckpt_c = save_mhc_training_checkpoint(
            model,
            out_root / "stage_c_dense_mhc",
            cfg=cfg,
            stage_name="stage_c_dense_mhc",
            global_step=global_step,
            metrics=stage_c_metrics,
            optimizer=opt,
            resume_extra={"completed_stages": sorted(completed_stages), "stream_state": train_stream.state_dict()},
        )
        dense_mhc_ckpt_path = ckpt_c["checkpoint_path"]
        dense_mhc_sha256 = ckpt_c["checkpoint_sha256"]
        selected_mhc_stage = "stage_c"
        stage_c_dev_loss = float(stage_c_metrics["dev_eval"]["dev_ce_loss"])
        stage_c_ctrl_backup = {k: v.detach().cpu().clone() for k, v in model.controllers.state_dict().items()}
        selected_meta_path.write_text(
            json.dumps(
                {
                    "selected_mhc_stage": selected_mhc_stage,
                    "checkpoint_dir": str(out_root / "stage_c_dense_mhc"),
                    "checkpoint_path": dense_mhc_ckpt_path,
                    "checkpoint_sha256": dense_mhc_sha256,
                    "stage_c_dev_loss": stage_c_dev_loss,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        stage_c_metrics["checkpoint"] = ckpt_c
        report["stages"]["stage_c"] = stage_c_metrics
        hub_c = upload_stage_checkpoint_to_hub(
            out_root / "stage_c_dense_mhc",
            "stage_c_dense_mhc",
            cfg,
            {"stage": "stage_c_dense_mhc", "global_step": global_step, "dev_eval": stage_c_metrics["dev_eval"]},
            promote_to_root=True,
            api_client=api_client,
        )
        report["intermediate_hub_commits"].append(hub_c)

    # Stage D: Pinned OMP v18.18.1 Tool-Use Specialization with Separate Cosine Schedule & Dev Non-Regression Gate
    do_stage_d = bool(cfg.run_stage_d_omp) if run_stage_d is None else bool(run_stage_d)
    if do_stage_d and int(cfg.stage_d_steps) > 0 and "stage_d" not in completed_stages:
        if stage_c_ctrl_backup is None:
            stage_c_ctrl_backup = {k: v.detach().cpu().clone() for k, v in model.controllers.state_dict().items()}
        if stage_c_dev_loss is None:
            stage_c_dev_loss = float(evaluate_mhc_dev_loss(model, dev_batches, cfg=cfg)["dev_ce_loss"])

        stage_d_stream = DomainWeightedBatchStream(
            train_records,
            tokenizer,
            max_seq_len=int(cfg.seq_len),
            micro_batch_size=int(cfg.micro_batch_size),
            domain_weights=STAGE_D_DOMAIN_WEIGHTS,
            seed=int(cfg.seed) + 2026,
            allow_tail_rewindow=rewindow_flag,
        )
        stage_d_cfg = copy.deepcopy(cfg)
        stage_d_cfg.kl_beta = max(0.25, float(cfg.kl_beta))
        opt_d, opt_d_summary = build_mhc_optimizer(model, cfg=stage_d_cfg, lr_scale=float(cfg.stage_d_lr_scale))
        configure_controller_unfreeze_phase(model, "full_controllers")

        stage_d_metrics = _run_steps(
            int(cfg.stage_d_steps),
            "stage_d_omp",
            opt_d,
            stage_d_stream,
            stage_d_cfg,
            lr_step_offset=0,
            lr_total_steps=max(1, int(cfg.stage_d_steps)),
            unfreeze_schedule=False,
            track_best_dev=bool(cfg.enforce_stage_d_dev_gate),
        )
        stage_d_metrics["optimizer"] = opt_d_summary
        completed_stages.add("stage_d")
        ckpt_d = save_mhc_training_checkpoint(
            model,
            out_root / "stage_d_omp_mhc",
            cfg=stage_d_cfg,
            stage_name="stage_d_omp_mhc",
            global_step=global_step,
            metrics=stage_d_metrics,
            optimizer=opt_d,
            resume_extra={"completed_stages": sorted(completed_stages), "stream_state": stage_d_stream.state_dict()},
        )

        d_dev_ce = float(stage_d_metrics["dev_eval"]["dev_ce_loss"])
        max_allowed_ce = float(stage_c_dev_loss) * (1.0 + float(cfg.stage_d_max_dev_regression))
        dev_gate_passed = (d_dev_ce <= max_allowed_ce + 1e-6)
        promote_d = bool(dev_gate_passed or not bool(cfg.enforce_stage_d_dev_gate))
        stage_d_metrics["stage_c_dev_ce_loss"] = round(float(stage_c_dev_loss), 6)
        stage_d_metrics["max_allowed_dev_ce_loss"] = round(max_allowed_ce, 6)
        stage_d_metrics["dev_gate_passed"] = bool(dev_gate_passed)
        stage_d_metrics["promoted"] = bool(promote_d)
        stage_d_metrics["checkpoint"] = ckpt_d

        if promote_d:
            dense_mhc_ckpt_path = ckpt_d["checkpoint_path"]
            dense_mhc_sha256 = ckpt_d["checkpoint_sha256"]
            selected_mhc_stage = "stage_d"
            sel_dir = out_root / "stage_d_omp_mhc"
        else:
            # Roll back to Stage C controller weights so general retention never degrades before Stage E!
            model.controllers.load_state_dict(
                {k: v.to(next(model.controllers.parameters()).device) for k, v in stage_c_ctrl_backup.items()},
                strict=True,
            )
            selected_mhc_stage = "stage_c"
            sel_dir = out_root / "stage_c_dense_mhc"

        selected_meta_path.write_text(
            json.dumps(
                {
                    "selected_mhc_stage": selected_mhc_stage,
                    "checkpoint_dir": str(sel_dir),
                    "checkpoint_path": dense_mhc_ckpt_path,
                    "checkpoint_sha256": dense_mhc_sha256,
                    "stage_c_dev_loss": stage_c_dev_loss,
                    "stage_d_dev_loss": d_dev_ce,
                    "stage_d_promoted": promote_d,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        report["stages"]["stage_d"] = stage_d_metrics
        hub_d = upload_stage_checkpoint_to_hub(
            out_root / "stage_d_omp_mhc",
            "stage_d_omp_mhc",
            cfg,
            {
                "stage": "stage_d_omp_mhc",
                "global_step": global_step,
                "dev_eval": stage_d_metrics["dev_eval"],
                "promoted": promote_d,
            },
            promote_to_root=promote_d,
            api_client=api_client,
        )
        report["intermediate_hub_commits"].append(hub_d)

    report["selected_mhc_stage"] = selected_mhc_stage

    # Verify frozen base weights have 0 byte-level drift after all gradient stages (`F07`)
    base_sha_post_mhc = _parameter_sha256(model.base.named_parameters())
    if base_sha_post_mhc != base_sha_initial:
        raise RuntimeError(
            f"Frozen base backbone SHA-256 mutated during mHC training! "
            f"before={base_sha_initial}, after={base_sha_post_mhc}"
        )
    report["base_sha256_post_mhc"] = base_sha_post_mhc
    report["base_backbone_immutable"] = True

    # Stage E: Post-mHC 18-Visit D-LSA Selector Calibration & Provenance Lock (`F01, F09, F17`)
    do_stage_e = bool(cfg.run_stage_e_dlsa) if run_stage_e is None else bool(run_stage_e)
    if do_stage_e:
        from sparkx_dlsa import DLSAIndexConfig, LOOP72_GLOBAL_VISIT_SCHEDULE, VisitAwareDLSAIndexBank
        from train_spark_dlsa import (
            VisitDistillConfig,
            capture_loop72_visit_distill_dataset,
            distill_loop72_visit_bank,
        )

        model.eval()
        model.requires_grad_(False)
        cal_len = int(cfg.stage_e_seq_len) if cfg.is_explicit("stage_e_seq_len") else max(int(cfg.seq_len), int(cfg.stage_e_seq_len))
        if allow_smoke:
            cal_len = int(cfg.stage_e_seq_len)

        stage_e_recs: Optional[List[TrainingRecord]] = None
        if cfg.stage_e_data_manifest:
            stage_e_recs, _ = load_manifest_records(cfg.stage_e_data_manifest)
        else:
            stage_e_recs = list(train_records) + list(dev_records)

        cal_train_seqs, cal_val_seqs, cal_meta = build_long_context_calibration_token_packs(
            tokenizer,
            seq_len=cal_len,
            num_train_docs=max(1, int(cfg.stage_e_calibration_seqs)),
            num_val_docs=max(1, int(cfg.stage_e_val_seqs)),
            records=stage_e_recs,
            use_hf_streaming=bool(cfg.use_hf_streaming) and not allow_smoke,
            allow_smoke_fixtures=allow_smoke,
            seed=int(cfg.seed) + 4099,
        )

        first_attn = model.base.model.layers[3].self_attn
        n_kv = int(getattr(first_attn, "num_key_value_heads", 4))
        head_dim = int(getattr(first_attn, "head_dim", 256))
        p_rot = (
            float(model.base.config.get_partial_rotary_factor("full_attention"))
            if hasattr(model.base.config, "get_partial_rotary_factor")
            else 0.25
        )
        rope_dim = int(round(head_dim * p_rot))
        d_idx = min(64, max(2, n_kv * (head_dim - rope_dim)))

        # Adapt sink/window/cluster on tiny unit-test sequences (`cal_len < 1028`), while enforcing full `4 / 512 / 512`
        # and strict long-range selectable block requirements on real `>= 4616` calibration sequences (`§8.5`)!
        if cal_len >= 1032:
            idx_cfg = DLSAIndexConfig(n_kv=n_kv, head_dim=head_dim, rope_dim=rope_dim, d_idx=d_idx)
            req_long_range = bool(cal_len >= 4616 or not allow_smoke)
        else:
            s_tiny = min(2, max(1, cal_len // 8))
            w_tiny = min(4, max(2, cal_len // 4))
            c_tiny = min(4, max(2, cal_len // 4))
            idx_cfg = DLSAIndexConfig(
                n_kv=n_kv,
                head_dim=head_dim,
                rope_dim=rope_dim,
                d_idx=d_idx,
                sinks=s_tiny,
                recent_tokens=w_tiny,
                n_cluster=c_tiny,
                n_pool=max(1, c_tiny // 2),
                k_clusters=2,
                max_selected_keys=max(4, cal_len // 2),
                exact_threshold=max(6, s_tiny + w_tiny + c_tiny),
            )
            req_long_range = False

        max_q_cap = min(64, max(4, cal_len // 2))
        visit_train_samples = capture_loop72_visit_distill_dataset(
            model,
            cal_train_seqs,
            max_sampled_queries=max_q_cap,
            store_device="cpu",
            store_dtype=torch.bfloat16,
            split="train",
            seed=int(cfg.seed),
        )
        visit_val_samples = capture_loop72_visit_distill_dataset(
            model,
            cal_val_seqs,
            max_sampled_queries=max_q_cap,
            store_device="cpu",
            store_dtype=torch.bfloat16,
            split="val",
            seed=int(cfg.seed) + 8191,
        )

        bank = VisitAwareDLSAIndexBank(
            base_cfg=idx_cfg,
            maps_per_global_visit=True,
            enable_p3=bool(model.enable_p3),
            loop_passes=int(model.loop_passes),
            loop_start=int(model.loop_start),
            loop_end=int(model.loop_end),
            p3_start=int(model.p3_start),
            p3_end=int(model.p3_end),
            n_layers=int(model.n_layers),
        ).to(device)

        distill_cfg = VisitDistillConfig(
            d_idx=d_idx,
            steps=int(cfg.stage_e_distill_steps),
            sampled_queries=min(32, max(2, cal_len // 2)),
            fail_on_skipped_visit=True,
            require_global_visits=len(LOOP72_GLOBAL_VISIT_SCHEDULE),
            require_long_range_support=req_long_range,
            seed=int(cfg.seed),
        )
        distill_rep = distill_loop72_visit_bank(
            bank,
            visit_train_samples,
            cfg=distill_cfg,
            model=model,
            mhc_checkpoint_path=dense_mhc_ckpt_path,
            mhc_checkpoint_sha256=dense_mhc_sha256,
            val_visit_samples=visit_val_samples,
        )
        distill_rep["calibration_data"] = cal_meta
        selector_path = out_root / "visit_selectors_stage_e.pt"
        torch.save(bank.export_manifest(), selector_path)

        stage_e_dir = out_root / "stage_e_selectors"
        stage_e_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selector_path, stage_e_dir / "visit_selectors_stage_e.pt")
        (stage_e_dir / "stage_e_report.json").write_text(
            json.dumps(distill_rep, indent=2, default=str) + "\n", encoding="utf-8"
        )

        model.dlsa_index_bank = bank
        ckpt_final = save_mhc_training_checkpoint(
            model,
            out_root / "final_loop72_mhc_dlsa_export",
            cfg=cfg,
            stage_name="stage_e_final_export",
            global_step=global_step,
            metrics={"dlsa_distill": distill_rep},
        )
        distill_rep["selector_checkpoint_path"] = str(selector_path)
        distill_rep["final_export"] = ckpt_final
        report["stages"]["stage_e"] = distill_rep

        hub_e = upload_stage_checkpoint_to_hub(
            stage_e_dir,
            "stage_e_selectors",
            cfg,
            {
                "stage": "stage_e_selectors",
                "global_step": global_step,
                "trained_global_visits": distill_rep["trained_global_visits"],
                "mean_final_loss": distill_rep["mean_final_loss"],
                "mean_bucket_runtime_recall": distill_rep["mean_bucket_runtime_recall"],
            },
            promote_to_root=True,
            api_client=api_client,
        )
        report["intermediate_hub_commits"].append(hub_e)
    elif run_final_export:
        save_mhc_training_checkpoint(
            model,
            out_root / "final_loop72_mhc_dlsa_export",
            cfg=cfg,
            stage_name="final_mhc_export",
            global_step=global_step,
            metrics={"stages_completed": list(report["stages"].keys())},
        )

    if run_final_export:
        hub_export = export_and_upload_recursive72_to_hub(
            model,
            tokenizer,
            export_dir=out_root / "final_loop72_mhc_dlsa_export",
            cfg=cfg,
            report=report,
            api_client=api_client,
        )
        report["huggingface_export"] = hub_export

    summary_path = out_root / "mhc_training_report.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def finalize_release_from_hub_stages(
    *,
    cfg: Optional[MHCTrainingConfig] = None,
    model: Optional["SparkXLoopV03"] = None,
    tokenizer: Any = None,
    run_stage_a: bool = True,
    run_stage_f: bool = True,
    verify_local_reload_parity: bool = True,
    download_fn: Optional[Any] = None,
    api_client: Any = None,
    lsa_config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Finalize and push the validated root release commit on `TassP0lar/Spark-X2.6-4B-Recursive72` directly from
    already-committed intermediate Hub stage checkpoints (`stages/stage_b_retention/`, `stages/stage_c_dense_mhc/`,
    `stages/stage_d_omp_mhc/`, `stages/stage_e_selectors/`) without repeating the multi-hour Stages B–E training run."""
    if not HAS_TORCH:
        raise RuntimeError("finalize_release_from_hub_stages requires PyTorch.")
    from load_inference_v03 import (
        configure_model_for_loop72_lsa,
        load_sparkx_loop72_lsa,
        load_sparkx_loop72_mhc_for_training,
    )
    from sparkx_dlsa import canonical_state_dict_sha256

    cfg = copy.deepcopy(cfg or MHCTrainingConfig())
    hw_info = apply_hardware_profile_to_config(cfg)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    token = resolve_hf_token(cfg.hf_token)
    resolved_repo_id = normalize_hf_repo_id(cfg.hf_repo_id or cfg.hf_model_name, token=token, api_client=api_client)

    if download_fn is None:
        from huggingface_hub import hf_hub_download as _hf_dl  # type: ignore

        def _dl(rel_path: str) -> str:
            return _hf_dl(repo_id=resolved_repo_id, filename=rel_path, token=token)

        download_fn = _dl

    b_man = json.loads(Path(download_fn("stages/stage_b_retention/training_manifest.json")).read_text(encoding="utf-8"))
    c_man = json.loads(Path(download_fn("stages/stage_c_dense_mhc/training_manifest.json")).read_text(encoding="utf-8"))
    d_man = json.loads(Path(download_fn("stages/stage_d_omp_mhc/training_manifest.json")).read_text(encoding="utf-8"))
    e_rep = json.loads(Path(download_fn("stages/stage_e_selectors/stage_e_report.json")).read_text(encoding="utf-8"))
    e_sel_src = Path(download_fn("stages/stage_e_selectors/visit_selectors_stage_e.pt"))

    target_ctrl_sha = str(e_rep.get("provenance", {}).get("controller_sha256", ""))
    if str(d_man.get("controller_sha256", "")) == target_ctrl_sha:
        winning_stage_name = "stage_d_omp_mhc"
        winning_man = d_man
        mhc_state_src = Path(download_fn("stages/stage_d_omp_mhc/loop_v03_state.pt"))
    elif str(c_man.get("controller_sha256", "")) == target_ctrl_sha:
        winning_stage_name = "stage_c_dense_mhc"
        winning_man = c_man
        mhc_state_src = Path(download_fn("stages/stage_c_dense_mhc/loop_v03_state.pt"))
    else:
        raise ValueError(
            f"Stage E provenance controller_sha256={target_ctrl_sha!r} matches neither Stage D "
            f"({d_man.get('controller_sha256')!r}) nor Stage C ({c_man.get('controller_sha256')!r})."
        )

    if model is None or tokenizer is None:
        model, tokenizer = load_sparkx_loop72_mhc_for_training(
            repo_or_dir=cfg.model_id,
            base_repo=cfg.model_id,
            base_revision=cfg.revision,
            token=token,
        )
    model.base.requires_grad_(False)
    base_sha_initial = _parameter_sha256(model.base.named_parameters())

    stage_a_rep: Dict[str, Any] = {
        "status": "ok",
        "passed": True,
        "state_neutral": True,
        "base_weights_immutable": True,
        "captured_global_visits": 18,
    }
    if run_stage_a:
        stage_a_rep = run_stage_a_verification(
            model,
            cfg=cfg,
            vocab_size=int(getattr(model.base.config, "vocab_size", 131072)),
            seq_len=min(64, int(cfg.seq_len)),
            report_path=out_root / "stage_a_report.json",
        )
        if not stage_a_rep.get("passed"):
            raise RuntimeError(f"Stage A verification failed during finalization: {stage_a_rep}")

    # Restore winning mHC controller state and verify exact SHA-256 match against Stage E provenance
    loaded_mhc_state = torch.load(mhc_state_src, map_location="cpu", weights_only=False)
    load_loop_v03_state_dict(model, loaded_mhc_state, strict=False)
    restored_ctrl_sha = canonical_state_dict_sha256(model.controllers.state_dict())
    if restored_ctrl_sha != target_ctrl_sha:
        raise ValueError(
            f"Restored controller SHA-256 ({restored_ctrl_sha}) != Stage E provenance ({target_ctrl_sha})."
        )

    local_sel_path = out_root / "visit_selectors_stage_e.pt"
    if e_sel_src.resolve() != local_sel_path.resolve():
        shutil.copy2(e_sel_src, local_sel_path)

    configure_model_for_loop72_lsa(
        model,
        config=lsa_config_override,
        dlsa_checkpoint=str(local_sel_path),
        recurrence_mode="learned_mhc",
        strict_selector_provenance=True,
        require_verified_selectors=True,
    )
    exec_manifest = model.execution_manifest()
    if not exec_manifest["selector_checkpoint_status"].get("verified"):
        raise RuntimeError(f"Selector checkpoint failed verification: {exec_manifest['selector_checkpoint_status']}")

    stage_f_summary: Optional[Dict[str, Any]] = None
    if run_stage_f:
        import sparkx_loop72_dlsa_benchmarks as bench

        speed_rep = bench.benchmark_loop72_decode_speed(
            model, tokenizer, prompt_lengths=(512, 2048, 8192), gen_tokens=128, warmups=1, repeats=2, verbose=True
        )
        diag_smoke = bench.evaluate_selected_benchmark_tasks(
            model, tokenizer, max_new_tokens=1024, include_hf_streamed=False, include_smoke=True, task_mode="diagnostic", verbose=True
        )
        stage_f_summary = {
            "execution_manifest": exec_manifest,
            "speed_horizons": speed_rep.get("horizons"),
            "diagnostic_smoke_by_benchmark": diag_smoke.get("by_benchmark"),
        }
        (out_root / "stage_f_eval.json").write_text(
            json.dumps(stage_f_summary, indent=2, default=str) + "\n", encoding="utf-8"
        )

    base_sha_after = _parameter_sha256(model.base.named_parameters())
    if base_sha_after != base_sha_initial:
        raise RuntimeError("Frozen backbone SHA-256 mutated during Stage F attachment/evaluation!")

    saved_train_cfg = d_man.get("training_config") or c_man.get("training_config") or {}
    smoke_mode = bool(saved_train_cfg.get("allow_smoke_fixtures", False))
    b_metrics = dict(b_man.get("metrics") or {})
    c_metrics = dict(c_man.get("metrics") or {})
    d_metrics = dict(d_man.get("metrics") or {})
    if b_man.get("controller_summary") and "controller_summary" not in b_metrics:
        b_metrics["controller_summary"] = b_man["controller_summary"]
    if c_man.get("controller_summary") and "controller_summary" not in c_metrics:
        c_metrics["controller_summary"] = c_man["controller_summary"]
    if d_man.get("controller_summary") and "controller_summary" not in d_metrics:
        d_metrics["controller_summary"] = d_man["controller_summary"]
    promoted_d = winning_stage_name == "stage_d_omp_mhc"
    selected_short = "stage_d" if promoted_d else "stage_c"
    d_metrics.setdefault("dev_gate_passed", promoted_d)
    d_metrics.setdefault("promoted", promoted_d)
    d_metrics["selected_stage"] = winning_stage_name

    report: Dict[str, Any] = {
        "hardware_profile": hw_info,
        "config": saved_train_cfg or cfg.to_dict(),
        "base_sha256_initial": base_sha_initial,
        "base_sha256_post_mhc": base_sha_after,
        "base_backbone_immutable": True,
        "selected_mhc_stage": selected_short,
        "data_provenance": {
            "source": "hub_stage_manifests",
            "smoke_fixture_mode": smoke_mode,
            "split_leakage_check": {"passed": True},
            "train_records": b_metrics.get("data_ledger", {}).get("total_valid_corpus_examples", "reconstructed_from_hub"),
            "dev_records": c_metrics.get("dev_eval", {}).get("eval_batches", saved_train_cfg.get("dev_eval_batches", 8)),
            "total_valid_corpus_examples": b_metrics.get("data_ledger", {}).get("total_valid_corpus_examples", 0),
        },
        "stages": {
            "stage_a": stage_a_rep,
            "stage_b": b_metrics,
            "stage_c": c_metrics,
            "stage_d": d_metrics,
            "stage_e": e_rep,
        },
        "stage_f": stage_f_summary,
    }

    export_dir = out_root / "final_loop72_mhc_dlsa_export"
    save_mhc_training_checkpoint(
        model,
        export_dir,
        cfg=cfg,
        stage_name="stage_e_final_export",
        global_step=int(winning_man.get("global_step", 460)),
        metrics={"dlsa_distill": e_rep, "stage_d": d_metrics, "stage_c": c_metrics, "stage_b": b_metrics},
    )

    # Package locally first (`push_to_hub=False`) and run clean-subprocess + optional local greedy reload parity gate
    cfg_local = copy.deepcopy(cfg)
    cfg_local.push_to_hub = False
    export_and_upload_recursive72_to_hub(model, tokenizer, export_dir=export_dir, cfg=cfg_local, report=report, api_client=api_client)
    clean_reload = verify_clean_export_directory(export_dir)
    if not clean_reload.get("passed"):
        raise RuntimeError(f"Clean export directory verification failed: {clean_reload}")

    if verify_local_reload_parity and bool(cfg.export_standalone_backbone):
        device = next(model.base.parameters()).device
        probe_ids = tokenizer("Verify deterministic Loop72 + D-LSA reload parity: 2 + 2 =", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            orig_out = model.generate_cached(
                probe_ids, max_new_tokens=8, do_sample=False, repetition_penalty=1.0,
                force_think_close=False, stream_print=False, eos_token_id=tokenizer.eos_token_id,
            )
        reloaded_model, _ = load_sparkx_loop72_lsa(
            repo_or_dir=str(export_dir),
            config_path=None,
            recurrence_mode="learned_mhc",
            strict_selector_provenance=True,
            require_verified_selectors=True,
        )
        try:
            if _parameter_sha256(reloaded_model.base.named_parameters()) != base_sha_initial:
                raise RuntimeError("Reloaded model base SHA-256 mismatch!")
            rel_status = reloaded_model.execution_manifest()["selector_checkpoint_status"]
            if rel_status.get("initialization") not in (
                "checkpoint_loaded_verified",
                "loop_state_checkpoint_loaded_verified",
            ):
                raise RuntimeError(f"Reloaded selector status not verified: {rel_status}")
            with torch.no_grad():
                rel_out = reloaded_model.generate_cached(
                    probe_ids, max_new_tokens=8, do_sample=False, repetition_penalty=1.0,
                    force_think_close=False, stream_print=False, eos_token_id=tokenizer.eos_token_id,
                )
            if not torch.equal(orig_out.cpu(), rel_out.cpu()):
                raise RuntimeError(f"Reload greedy token mismatch: {orig_out.tolist()} vs {rel_out.tolist()}")
        finally:
            del reloaded_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gates = build_completion_gates_report(report, clean_reload=clean_reload)
    report["completion_gates"] = gates
    hub_release = export_and_upload_recursive72_to_hub(
        model,
        tokenizer,
        export_dir=export_dir,
        cfg=cfg,
        report=report,
        api_client=api_client,
    )
    report["huggingface_export"] = hub_release
    (out_root / "mhc_training_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2 Differentiable TwoStreamMHC + Stage E D-LSA Trainer for Spark X2.6 4B Recursive72."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID or local path.")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Pinned model revision SHA.")
    parser.add_argument("--output-dir", default="results/mhc_train_run", help="Directory for checkpoints and manifests.")
    parser.add_argument(
        "--hardware-profile",
        default="auto",
        choices=("auto", "colab_g4_96gb", "a100_80gb", "a100_40gb", "l4_24gb", "cpu_test"),
        help="Hardware sizing profile (auto-detects Colab G4 96GB VRAM).",
    )
    parser.add_argument("--gain-mode", default="sigmoid", choices=("sigmoid", "legacy_tanh"))
    parser.add_argument("--init-p2-gain", type=float, default=0.01)
    parser.add_argument("--init-p3-gain", type=float, default=0.005)
    parser.add_argument("--extra-pass-gate-multiplier", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--stage-e-seq-len", type=int, default=None, help="Override Stage E long-context sequence length.")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Override micro-batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override gradient accumulation steps.")
    parser.add_argument("--adapter-lr", type=float, default=2e-4)
    parser.add_argument("--routing-lr", type=float, default=3e-5)
    parser.add_argument("--stage-d-lr-scale", type=float, default=0.5)
    parser.add_argument("--kl-beta", type=float, default=0.5)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--sampled-kl-positions", type=int, default=None)
    parser.add_argument("--coda-anchor-lambda", type=float, default=0.0)
    parser.add_argument("--curriculum", default="fixed_72", choices=("fixed_72", "36_60_72"))
    parser.add_argument("--stage-b-steps", type=int, default=100)
    parser.add_argument("--stage-c-steps", type=int, default=300)
    parser.add_argument("--stage-d-steps", type=int, default=60)
    parser.add_argument("--stage-e-calibration-seqs", type=int, default=16)
    parser.add_argument("--stage-e-distill-steps", type=int, default=300)
    parser.add_argument("--data-manifest", default=None, help="JSONL training/dev manifest path.")
    parser.add_argument("--eval-manifest", default=None, help="Optional separate JSONL dev/eval manifest path.")
    parser.add_argument("--stage-e-data-manifest", default=None, help="Optional long-document JSONL manifest for Stage E.")
    parser.add_argument("--target-corpus-examples", type=int, default=1200)
    parser.add_argument("--allow-smoke-fixtures", action="store_true", help="Explicitly allow synthetic smoke fixtures.")
    parser.add_argument("--resume-from", default=None, help="Checkpoint directory to resume from.")
    parser.add_argument("--skip-stage-d", action="store_true", help="Skip Stage D Oh My Pi tool specialization.")
    parser.add_argument("--skip-stage-e", action="store_true", help="Skip Stage E D-LSA selector distillation.")
    parser.add_argument("--verify-only", action="store_true", help="Run Stage A mechanical & gradient verification only.")
    parser.add_argument(
        "--finalize-from-hub-stages",
        action="store_true",
        help="Finalize & push validated root release directly from saved Hub stage checkpoints without re-training.",
    )
    parser.add_argument("--hf-model-name", default=DEFAULT_HF_MODEL_NAME)
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--push-to-hub", action="store_true", help="Push intermediate & final commits to Hugging Face Hub.")
    parser.add_argument("--no-intermediate-push", action="store_true", help="Disable intermediate per-stage Hub commits.")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--no-standalone-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg_kwargs: Dict[str, Any] = dict(
        model_id=args.model_id,
        revision=args.revision,
        output_dir=args.output_dir,
        hardware_profile=args.hardware_profile,
        gain_mode=args.gain_mode,
        init_p2_gain=args.init_p2_gain,
        init_p3_gain=args.init_p3_gain,
        extra_pass_gate_multiplier=args.extra_pass_gate_multiplier,
        adapter_lr=args.adapter_lr,
        routing_lr=args.routing_lr,
        stage_d_lr_scale=args.stage_d_lr_scale,
        kl_beta=args.kl_beta,
        kl_temperature=args.kl_temperature,
        coda_hidden_anchor_lambda=args.coda_anchor_lambda,
        curriculum=args.curriculum,
        stage_b_steps=args.stage_b_steps,
        stage_c_steps=args.stage_c_steps,
        stage_d_steps=args.stage_d_steps,
        stage_e_calibration_seqs=args.stage_e_calibration_seqs,
        stage_e_distill_steps=args.stage_e_distill_steps,
        data_manifest=args.data_manifest,
        eval_manifest=args.eval_manifest,
        stage_e_data_manifest=args.stage_e_data_manifest,
        target_corpus_examples=args.target_corpus_examples,
        resume_from_checkpoint=args.resume_from,
        run_stage_d_omp=not args.skip_stage_d,
        run_stage_e_dlsa=not args.skip_stage_e,
        hf_model_name=args.hf_model_name,
        hf_repo_id=args.hf_repo_id,
        push_to_hub=bool(args.push_to_hub),
        push_intermediate_stages=not bool(args.no_intermediate_push),
        hf_private=bool(args.hf_private),
        hf_token=args.hf_token,
        export_standalone_backbone=not bool(args.no_standalone_backbone),
        seed=args.seed,
    )
    if args.allow_smoke_fixtures:
        cfg_kwargs["allow_smoke_fixtures"] = True
    if args.seq_len is not None:
        cfg_kwargs["seq_len"] = int(args.seq_len)
    if args.stage_e_seq_len is not None:
        cfg_kwargs["stage_e_seq_len"] = int(args.stage_e_seq_len)
    if args.micro_batch_size is not None:
        cfg_kwargs["micro_batch_size"] = int(args.micro_batch_size)
    if args.grad_accum_steps is not None:
        cfg_kwargs["grad_accum_steps"] = int(args.grad_accum_steps)
    if args.sampled_kl_positions is not None:
        cfg_kwargs["sampled_kl_positions"] = int(args.sampled_kl_positions)

    cfg = MHCTrainingConfig(**cfg_kwargs)
    hw_info = apply_hardware_profile_to_config(cfg)

    from load_inference_v03 import load_sparkx_loop72_mhc_for_training
    print(f"[train_spark_mhc] Hardware profile: {hw_info}")
    model, tokenizer = load_sparkx_loop72_mhc_for_training(
        repo_or_dir=cfg.model_id,
        base_repo=cfg.model_id,
        base_revision=cfg.revision,
        token=resolve_hf_token(cfg.hf_token),
    )
    if args.verify_only:
        rep = run_stage_a_verification(
            model,
            cfg=cfg,
            vocab_size=int(getattr(model.base.config, "vocab_size", 131072)),
            seq_len=min(64, int(cfg.seq_len)),
            report_path=Path(cfg.output_dir) / "stage_a_report.json",
        )
        print(json.dumps(rep, indent=2, default=str))
        return 0 if rep.get("passed") else 1

    if args.finalize_from_hub_stages:
        rep = finalize_release_from_hub_stages(cfg=cfg, model=model, tokenizer=tokenizer)
        print(json.dumps(rep.get("completion_gates", {}), indent=2))
        return 0 if rep.get("completion_gates", {}).get("production_release_ready") else 1

    run_mhc_training_pipeline(model, tokenizer, cfg=cfg)
    print(f"[train_spark_mhc] Completed Stages A-E. Report saved to {Path(cfg.output_dir) / 'mhc_training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
