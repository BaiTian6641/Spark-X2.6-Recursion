"""train_spark_mhc.py — Milestone 2 Differentiable mHC Recurrence Trainer for Spark X2.6 4B Loop72-LSA.

Implements Sections 4, 5 (Stages A–E), and 6 of `docs/roadmap/09_LOOP72_MHC_TRAINING_AND_VALIDATION.md`:
  - Versioned bounded-gain `TwoStreamMHC` parameterization (`gain_mode="sigmoid"`, `schema_version=2`,
    `g = gate_mult * sigmoid(u) in [0, 1]`, initialized to `g_P2=0.01`, `g_P3=0.005`, `10,078,176` params).
  - Stage A: Mechanical & gradient verification (trainable mask, frozen backbone SHA-256 invariance,
    Birkhoff doubly-stochastic bounds, P2/P3/adapter/routing gradients, checkpoint replay parity, save/load parity).
  - Stage B: 36L-to-72L teacher retention warmup with memory-safe sampled-position full-vocabulary KL
    (`K=64` supervised positions per sequence, `beta=0.5`, `T=1.5`, `lambda_h=0.0`, no-grad 36L teacher first).
  - Stage C: Multi-pass deliberation training (`fixed_72` default or `36_60_72` curriculum) with staged
    controller unfreezing (`warmup_gains_and_up` -> `adapters_and_static_routing` -> `full_controllers`),
    separate AdamW parameter groups (`adapter_lr=2e-4`, `routing_lr=3e-5`), and 5-domain mixture
    (40% code/debugging, 20% general instruction, 15% math, 15% tool-use, 10% long-doc/repo).
  - Stage D: Optional `Oh My Pi` (`omp`) native `<tool_call>` (`read`/`edit`/`bash`) trajectory specialization
    after saving the pre-tool Stage C dense mHC checkpoint.
  - Stage E: Post-mHC 18-visit `VisitAwareDLSAIndexBank` PCA + sampled-query KL distillation with
    `mhc_checkpoint_sha256` and `recurrence_mode` provenance lock.
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
    from sparkx_loop_v03_core import (
        LoopV03Settings,
        SparkXLoopV03,
        TwoStreamMHC,
        fused_masked_cross_entropy,
        load_loop_v03_state_dict,
        loop_v03_state_dict,
    )


DEFAULT_MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_MODEL_REVISION = "642661dc6dc74bee0fb40cb7efbe3fb2ecc96bcd"
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


@dataclasses.dataclass
class MHCTrainingConfig:
    """Configuration for Milestone 2 Loop72 `TwoStreamMHC` training (Stages A–E)."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    output_dir: str = "results/mhc_train_run"
    hardware_profile: str = "auto"  # 'auto' | 'colab_g4_96gb' | 'a100_80gb' | 'a100_40gb' | 'l4_24gb' | 'cpu_test'

    # Bounded-gain TwoStreamMHC contract (Section 4)
    recurrence_mode: str = "learned_mhc"
    gain_mode: str = "sigmoid"  # 'sigmoid' (schema v2) | 'legacy_tanh' (schema v1)
    init_p2_gain: float = 0.01
    init_p3_gain: float = 0.005
    extra_pass_gate_multiplier: float = 1.0

    # Sequence & batch sizing (auto-tuned for Colab G4 96GB when hardware_profile='auto')
    seq_len: int = 8192
    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    ce_chunk_tokens: int = 2048

    # Optimizer & schedule (Section 5 Stage C & Section 6)
    adapter_lr: float = 2e-4
    routing_lr: float = 3e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.08
    min_lr_ratio: float = 0.10

    # Teacher retention objective (Section 5 Stage B & Section 6.2)
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
    stage_e_calibration_seqs: int = 4
    stage_e_seq_len: int = 2048
    stage_e_distill_steps: int = 150

    # Checkpointing, Hugging Face Hub upload & reproducibility
    save_every_steps: int = 50
    eval_every_steps: int = 25
    use_hf_streaming: bool = True
    seed: int = 42
    hf_model_name: str = DEFAULT_HF_MODEL_NAME
    hf_repo_id: str = DEFAULT_HF_REPO_ID
    push_to_hub: bool = False
    hf_private: bool = False
    hf_token: Optional[str] = None
    export_standalone_backbone: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
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
            "max_recommended_seq_len": 4096,
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
        "max_recommended_seq_len": 512,
        "micro_batch_size": 1,
        "grad_accum_steps": 2,
        "sampled_kl_positions": 16,
        "ce_chunk_tokens": 256,
    }


def apply_hardware_profile_to_config(cfg: MHCTrainingConfig, vram_gb: Optional[float] = None) -> Dict[str, Any]:
    """Populate `cfg` sequence length and accumulation settings when `hardware_profile` is set or `'auto'`."""
    prof_map = {
        "colab_g4_96gb": 96.0,
        "a100_80gb": 80.0,
        "a100_40gb": 40.0,
        "l4_24gb": 24.0,
        "cpu_test": 0.0,
    }
    target_vram = vram_gb if vram_gb is not None else prof_map.get(str(cfg.hardware_profile).lower())
    info = detect_gpu_training_profile(target_vram)
    if str(cfg.hardware_profile).lower() in prof_map:
        cfg.seq_len = int(info["seq_len"])
        cfg.micro_batch_size = int(info["micro_batch_size"])
        cfg.grad_accum_steps = int(info["grad_accum_steps"])
        cfg.sampled_kl_positions = int(info["sampled_kl_positions"])
        cfg.ce_chunk_tokens = int(info["ce_chunk_tokens"])
    return info


def _parameter_sha256(parameters: Iterable[Tuple[str, "torch.Tensor"]]) -> str:
    """Compute a deterministic SHA-256 digest over named parameter shapes, dtypes, and sample bytes."""
    h = hashlib.sha256()
    for name, p in sorted(parameters, key=lambda item: item[0]):
        t = p.detach()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        flat = t.reshape(-1)
        if flat.numel() <= 4096:
            h.update(flat.cpu().float().numpy().tobytes())
        else:
            head = flat[:2048].cpu().float().numpy().tobytes()
            tail = flat[-2048:].cpu().float().numpy().tobytes()
            h.update(head)
            h.update(tail)
    return h.hexdigest()


def prepare_model_for_mhc_training(
    model: "SparkXLoopV03",
    cfg: Optional[MHCTrainingConfig] = None,
    reset_gains: bool = True,
) -> Dict[str, Any]:
    """Configure an existing `SparkXLoopV03` instance for differentiable `learned_mhc` training.

    Enforces:
      - `recurrence_mode = "learned_mhc"`
      - `attention_training_supported == True` (rejects attached detached inference KV archive)
      - Frozen `base.` weights (`requires_grad = False`), trainable `controllers.` (`requires_grad = True`)
      - Versioned bounded-gain schema (`gain_mode="sigmoid"`, `mhc_schema_version=2`)
      - `esm_enabled = False`, `enable_checkpoint = True`
    """
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
    """Configure staged controller parameter unfreezing (Section 5 Stage C):
      1. `'warmup_gains_and_up'`: unfreeze extra-pass gains (`extra_gain`, `ffn_extra_gain`, `p3_extra_gain`,
         `p3_ffn_extra_gain`) and `pass_up` projections (which have non-zero step-0 gradient because `pass_down`
         is Kaiming-initialized).
      2. `'adapters_and_static_routing'`: unfreeze gains, `pass_down`, `pass_up`, static `read/write/mix_logits`,
         and `pass_routing_bias` (keeping `dynamic_proj` frozen).
      3. `'full_controllers'`: unfreeze all `10,078,176` controller parameters including `dynamic_proj` and RMSNorm weights.
    """
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
) -> Tuple["torch.optim.AdamW", Dict[str, Any]]:
    """Construct AdamW with separate parameter groups for pass adapters vs. routing/gains (Section 5 Stage C):
      - `adapter_group` (`pass_down`, `pass_up`): `lr = cfg.adapter_lr` (`2e-4`), `weight_decay = cfg.weight_decay` (`0.01`)
      - `routing_decay_group` (`dynamic_proj.weight`, `ffn_dynamic_proj.weight`): `lr = cfg.routing_lr` (`3e-5`), `weight_decay = cfg.weight_decay`
      - `routing_nodecay_group` (gains, static read/write/mix logits, `pass_routing_bias`, RMSNorm weights):
        `lr = cfg.routing_lr` (`3e-5`), `weight_decay = 0.0`
    All controller parameters are kept in FP32 so `lr * grad` updates never underflow BF16 ULP (`2^-7`).
    """
    cfg = cfg or MHCTrainingConfig()
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
            "lr": float(cfg.adapter_lr),
            "initial_lr": float(cfg.adapter_lr),
            "base_lr": float(cfg.adapter_lr),
            "weight_decay": float(cfg.weight_decay),
        },
        {
            "name": "routing_decay_group",
            "params": routing_decay_params,
            "lr": float(cfg.routing_lr),
            "initial_lr": float(cfg.routing_lr),
            "base_lr": float(cfg.routing_lr),
            "weight_decay": float(cfg.weight_decay),
        },
        {
            "name": "routing_nodecay_group",
            "params": routing_nodecay_params,
            "lr": float(cfg.routing_lr),
            "initial_lr": float(cfg.routing_lr),
            "base_lr": float(cfg.routing_lr),
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
        "adapter_lr": float(cfg.adapter_lr),
        "routing_lr": float(cfg.routing_lr),
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
    """Sample up to `K = max_positions` supervised next-token prediction positions per sequence (`labels[:, 1:] != -100`).

    Returns:
      - `pos_indices`: `[B, K]` long tensor of hidden-state positions in `[0, S-2]`
      - `pos_mask`: `[B, K]` boolean tensor indicating valid sampled positions
    """
    shift_y = labels[:, 1:]
    bsz, seq_m1 = shift_y.shape
    k_keep = max(1, min(int(max_positions), seq_m1))
    pos_indices = torch.zeros((bsz, k_keep), dtype=torch.long, device=labels.device)
    pos_mask = torch.zeros((bsz, k_keep), dtype=torch.bool, device=labels.device)

    for b in range(bsz):
        valid_idx = torch.nonzero(shift_y[b] != -100, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            # Fallback to final token position if a sequence has no unmasked labels
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
    """Compute the Stage B/C combined objective without materializing a `[B, S, V]` 16 GiB logit tensor:
      1. Samples `K = cfg.sampled_kl_positions` supervised positions per sequence.
      2. If `cfg.kl_beta > 0`, runs the native 36-layer teacher (`passes=1, use_mhc=False, enable_p3=False`)
         under `torch.no_grad()` FIRST, extracts only the `K` sampled hidden vectors `[B, K, D]`, computes
         teacher logits `[B, K, V]`, and frees all 36L teacher activations before the 72L student graph is built.
      3. Runs the 72L `learned_mhc` student forward (`forward_hidden`), computes `fused_masked_cross_entropy`
         over all supervised tokens, and computes exact full-vocabulary temperature-scaled KL on the `K` sampled
         positions:
           `L = L_CE + beta * T^2 * KL(p_teacher_T || p_student_T) + lambda_h * ||h_student_27 - h_teacher_27||^2`
    """
    cfg = cfg or MHCTrainingConfig()
    lm_weight = model._head_weight()
    pos_indices, pos_mask = sample_supervised_positions(
        labels, max_positions=int(cfg.sampled_kl_positions), generator=generator
    )
    bsz, k_keep = pos_indices.shape
    batch_arange = torch.arange(bsz, device=input_ids.device).unsqueeze(1).expand(bsz, k_keep)

    teacher_logits_sampled: Optional[torch.Tensor] = None
    teacher_coda_h: Optional[torch.Tensor] = None
    use_kl = float(cfg.kl_beta) > 0.0 and bool(pos_mask.any().item())
    use_coda_anchor = float(cfg.coda_hidden_anchor_lambda) > 0.0

    if use_kl or use_coda_anchor:
        was_training = model.training
        prev_hook = getattr(model, "_residual_capture_hook", None)
        captured_teacher_coda: Dict[str, torch.Tensor] = {}

        def _teacher_hook(idx: int, v_idx: int, p_idx: int, stage: str, h_t: torch.Tensor) -> None:
            if use_coda_anchor and stage == "post_block" and idx == (model.loop_end - 1):
                captured_teacher_coda["h"] = h_t[batch_arange, pos_indices, :].detach().float()

        model.eval()
        if use_coda_anchor:
            model._residual_capture_hook = _teacher_hook
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
            model._residual_capture_hook = prev_hook
            if was_training:
                model.train()

    # Student 72L forward (with optional coda anchor capture at entry to Layer 28)
    captured_student_coda: Dict[str, torch.Tensor] = {}
    prev_hook = getattr(model, "_residual_capture_hook", None)

    def _student_hook(idx: int, v_idx: int, p_idx: int, stage: str, h_s: torch.Tensor) -> None:
        if use_coda_anchor and stage == "post_block" and idx == (model.loop_end - 1):
            captured_student_coda["h"] = h_s[batch_arange, pos_indices, :].float()

    if use_coda_anchor:
        model._residual_capture_hook = _student_hook
    try:
        h_student = model.forward_hidden(input_ids, attention_mask=attention_mask)
    finally:
        if use_coda_anchor:
            model._residual_capture_hook = prev_hook

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
    }
    return total_loss, metrics


# ==============================================================================
# Stage A: Mechanical & Gradient Verification Suite
# ==============================================================================
def run_stage_a_verification(
    model: "SparkXLoopV03",
    cfg: Optional[MHCTrainingConfig] = None,
    vocab_size: int = 16,
    seq_len: int = 12,
) -> Dict[str, Any]:
    """Run Stage A mechanical and gradient verification (Section 5 Stage A):
      1. Verifies trainable parameter mask (`controllers.` only) and parameter count.
      2. Verifies `effective_recurrence_mode(True) == effective_recurrence_mode(False) == 'learned_mhc'`.
      3. Verifies bounded-gain Birkhoff doubly-stochastic invariants (`gains_outside_unit_interval == []`).
      4. Verifies deterministic forward + backward with teacher retention KL (`K` sampled positions).
      5. Verifies non-zero gradients on Pass-2 (`extra_gain`), Pass-3 (`p3_extra_gain`), `pass_up` adapters,
         and routing parameters.
      6. Verifies frozen `base.` parameters have `grad is None` and identical SHA-256 before/after an optimizer step.
      7. Verifies checkpoint recomputation gradient parity and save/load roundtrip logit equality.
    """
    if not HAS_TORCH:
        raise RuntimeError("run_stage_a_verification requires PyTorch.")
    cfg = copy.deepcopy(cfg or MHCTrainingConfig())
    cfg.sampled_kl_positions = min(int(cfg.sampled_kl_positions), max(2, seq_len - 2))

    prep = prepare_model_for_mhc_training(model, cfg=cfg, reset_gains=True)
    configure_controller_unfreeze_phase(model, phase="full_controllers")

    device = next(model.base.parameters()).device
    base_sha_before = _parameter_sha256(model.base.named_parameters())

    torch.manual_seed(int(cfg.seed))
    input_ids = torch.randint(1, max(4, int(vocab_size)), (1, int(seq_len)), device=device, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, : max(1, seq_len // 3)] = -100  # Mask prompt prefix

    # 1. Check gradient checkpointing parity vs uncheckpointed forward/backward
    model.enable_checkpoint = False
    model.zero_grad(set_to_none=True)
    g1 = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    loss_nockpt, metrics_nockpt = compute_teacher_retention_step_loss(
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

    max_ckpt_grad_diff = 0.0
    for name, p in model.controllers.named_parameters():
        if p.grad is not None and name in grads_nockpt:
            diff = float((p.grad.detach() - grads_nockpt[name]).abs().max().item())
            max_ckpt_grad_diff = max(max_ckpt_grad_diff, diff)

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

    # 2. Take one optimizer step and verify base weight immutability + save/load logit parity
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
        # Perturb controller gain in-place, then restore from saved_state
        first_ctrl = next(iter(model.controllers.values()))
        first_ctrl.extra_gain.add_(1.25)
        load_loop_v03_state_dict(model, saved_state, strict=True)
        model.eval()
        logits_after_load = model.forward_logits(input_ids).detach().float()
        save_load_max_diff = float((logits_before_save - logits_after_load).abs().max().item())

    ctrl_sum = model.controller_summary()
    manifest = model.execution_manifest()
    passed = (
        math.isfinite(metrics_ckpt["total_loss"])
        and p2_grad_norm > 0.0
        and p3_grad_norm > 0.0
        and adapter_up_grad_norm > 0.0
        and base_immutable
        and max_ckpt_grad_diff < 1e-4
        and save_load_max_diff == 0.0
        and not manifest["recurrence_train_eval_mismatch"]
        and manifest["effective_recurrence_mode"] == "learned_mhc"
        and len(ctrl_sum["gains_outside_unit_interval"]) == 0
        and ctrl_sum["min_static_effective_mix_entry"] >= 0.0
    )
    return {
        "status": "ok" if passed else "failed",
        "passed": bool(passed),
        "trainable_parameters": prep["trainable_parameters"],
        "expected_full_model_controller_parameters": EXPECTED_DEFAULT_CONTROLLER_PARAMS,
        "loss_metrics": metrics_ckpt,
        "p2_gain_grad_l1": round(p2_grad_norm, 8),
        "p3_gain_grad_l1": round(p3_grad_norm, 8),
        "adapter_up_grad_norm": round(adapter_up_grad_norm, 8),
        "max_checkpoint_replay_grad_diff": float(max_ckpt_grad_diff),
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


# ==============================================================================
# 5-Domain Data Mixture & Native Spark Tool-Trajectory Formatting (Stages B, C, D)
# ==============================================================================
def build_omp_tool_trajectory_examples() -> List[Dict[str, Any]]:
    """Build native Spark `<tool_call>` (`read`/`edit`/`bash`) multi-turn trajectories aligned with `sparkx_omp_server.py`."""
    omp_tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the contents of a file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Replace an exact string in a workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]
    return [
        {
            "domain": "tool_trajectories",
            "tools": omp_tools,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an autonomous software engineering agent using read, edit, and bash tools. Inspect files before editing and run tests before concluding.",
                },
                {
                    "role": "user",
                    "content": "Fix the off-by-one window slicing bug in `rolling_sum.py` and verify that `pytest -q` passes.",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "First, read `rolling_sum.py` and `test_rolling_sum.py` to inspect the window boundary calculation.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_read_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": {"path": "rolling_sum.py"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_read_1",
                    "name": "read",
                    "content": "def rolling_sum(values, window):\n    if window <= 0:\n        raise ValueError('window must be positive')\n    return [sum(values[i:i + window - 1]) for i in range(len(values) - window + 1)]\n",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "The slice uses `i + window - 1` instead of `i + window`, dropping the last element of each window. Let's patch `rolling_sum.py` with `edit`.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_edit_1",
                            "type": "function",
                            "function": {
                                "name": "edit",
                                "arguments": {
                                    "path": "rolling_sum.py",
                                    "old_text": "sum(values[i:i + window - 1])",
                                    "new_text": "sum(values[i:i + window])",
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_edit_1",
                    "name": "edit",
                    "content": "Applied edit to rolling_sum.py.",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "Now run the test suite with `bash` to confirm the fix passes.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bash_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": {"command": "pytest -q"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bash_1",
                    "name": "bash",
                    "content": "3 passed in 0.04s",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "All 3 unit tests pass after correcting the slice upper bound.",
                    "content": "Fixed the off-by-one slice in `rolling_sum.py` (`values[i:i + window]`) and verified that all 3 tests pass under `pytest -q`.",
                },
            ],
        },
        {
            "domain": "tool_trajectories",
            "tools": omp_tools,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an autonomous software engineering agent using read, edit, and bash tools.",
                },
                {
                    "role": "user",
                    "content": "In `lru_page_cache.py`, accessing an existing key does not mark it as most-recently-used before eviction. Fix the bug and run `python3 -m unittest`.",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "Let's read `lru_page_cache.py` first to see how `get` and `put` manage `OrderedDict` order.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_read_lru",
                            "type": "function",
                            "function": {"name": "read", "arguments": {"path": "lru_page_cache.py"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_read_lru",
                    "name": "read",
                    "content": "from collections import OrderedDict\n\nclass LRUPageCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.store = OrderedDict()\n\n    def get(self, key):\n        if key not in self.store:\n            return None\n        return self.store[key]\n",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "`get` returns `self.store[key]` without calling `self.store.move_to_end(key)`. Let's apply the fix using `edit`.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_edit_lru",
                            "type": "function",
                            "function": {
                                "name": "edit",
                                "arguments": {
                                    "path": "lru_page_cache.py",
                                    "old_text": "        if key not in self.store:\n            return None\n        return self.store[key]",
                                    "new_text": "        if key not in self.store:\n            return None\n        self.store.move_to_end(key)\n        return self.store[key]",
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_edit_lru",
                    "name": "edit",
                    "content": "Applied edit to lru_page_cache.py.",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "Let's run the unit test suite via `bash`.",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bash_lru",
                            "type": "function",
                            "function": {"name": "bash", "arguments": {"command": "python3 -m unittest -v"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bash_lru",
                    "name": "bash",
                    "content": "Ran 4 tests in 0.012s\n\nOK",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "The unit tests pass.",
                    "content": "Updated `LRUPageCache.get` in `lru_page_cache.py` to call `self.store.move_to_end(key)` on cache hits and verified all 4 unit tests pass.",
                },
            ],
        },
    ]


def build_five_domain_seed_bank() -> List[Dict[str, Any]]:
    """Build deterministic multi-domain training examples covering the 5-domain Stage B/C mixture:
      - 40% `code_debug` (multi-file debugging, patch synthesis, invariant repair)
      - 20% `general_instruct` (technical exposition, instruction following)
      - 15% `math_symbolic` (multi-step competition math & symbolic proofs)
      - 15% `tool_trajectories` (native Spark `<tool_call>` trajectories)
      - 10% `long_doc_repo` (multi-module repository architecture & long-context synthesis)
    """
    examples: List[Dict[str, Any]] = [
        {
            "domain": "code_debug",
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
                        "`p_keep = sigmoid(0.5 * ((L_00 + L_11) - (L_01 + L_10)))` and off-diagonal `p_swap = 1 - p_keep`. "
                        "Clamping logits to `[-30, 30]` and using `torch.sigmoid(diag_adv)` gives the exact closed-form "
                        "infinite-iteration Sinkhorn limit without `exp` overflow."
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
            "domain": "code_debug",
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
                        "For each query position `q_pos` and its selected key index list, every `k_idx >= 0` must "
                        "satisfy `0 <= k_idx <= q_pos` to prevent future-token leakage in causal sparse attention."
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
            "domain": "general_instruct",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain why bounded sigmoid extra-pass gains `g = sigmoid(u) in (0, 1)` preserve the "
                        "Birkhoff polytope guarantee `(1 - g)I + gP in B_2` whereas `g = tanh(u) in (-1, 1)` does not."
                    ),
                },
                {
                    "role": "assistant",
                    "reasoning_content": (
                        "The Birkhoff polytope B_2 is the convex hull of 2x2 permutation matrices (`I` and `Swap`). "
                        "Because B_2 is convex and `I in B_2`, `(1 - g)I + gP` remains non-negative with unit row/column "
                        "sums for every `P in B_2` if and only if `g in [0, 1]`. When `g < 0`, off-diagonal entries `g * p_swap` "
                        "become negative whenever `p_swap > 0`."
                    ),
                    "content": (
                        "A matrix $M(g) = (1 - g)I + gP$ with $P \\in \\mathcal{B}_2$ has diagonal entries "
                        "$1 - g p_{\\text{swap}}$ and off-diagonal entries $g p_{\\text{swap}}$. While row and column "
                        "sums equal $1$ for any scalar $g$, non-negativity of all entries requires $0 \\le g p_{\\text{swap}} \\le 1$. "
                        "Using $g = \\sigma(u) \\in (0, 1)$ guarantees $M(g) \\in \\mathcal{B}_2$ for all $u \\in \\mathbb{R}$, "
                        "preventing negative mixing weights and amplification across 36 recurrent Pass-2/Pass-3 visits."
                    ),
                },
            ],
        },
        {
            "domain": "math_symbolic",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Find the sum of all positive integers $n$ such that $\\frac{n^2 + 11n + 24}{n + 3}$ "
                        "is a prime number. Provide the final answer in `\\boxed{}`."
                    ),
                },
                {
                    "role": "assistant",
                    "reasoning_content": (
                        "Factor the numerator: `n^2 + 11n + 24 = (n + 3)(n + 8)`. For `n > 0`, `n + 3 != 0`, so the "
                        "expression simplifies to `n + 8`. Wait — if the problem means `(n^2 + 11n + 30)/(n + 3)`, "
                        "let's check `(n^2 + 11n + 24)/(n + 3) = n + 8`, which has infinitely many primes. Let's solve "
                        "for when `n` divides `24` in `(24 / (n + 3))`: if the expression is `(n^2 + 3n + 24)/(n + 3) = n + 24/(n + 3)`, "
                        "let's verify the exact algebraic identity."
                    ),
                    "content": (
                        "Since $n^2 + 11n + 24 = (n + 3)(n + 8)$, for any positive integer $n$, the quotient simplifies "
                        "identically to $n + 8$. For the lowest positive integer $n = 3$, $n + 8 = \\boxed{11}$."
                    ),
                },
            ],
        },
        {
            "domain": "long_doc_repo",
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
                        "Trace the 72 virtual visits: Prelude L0..L3 (4 visits, 1 global at L3), Pass 1 L4..L27 "
                        "(24 visits, 6 global at L7,11,15,19,23,27), Pass 2 L4..L27 (24 visits, 6 global), Pass 3 L12..L23 "
                        "(12 visits, 3 global at L15,19,23), and Coda L28..L35 (8 visits, 2 global at L31,35)."
                    ),
                    "content": (
                        "1. **Prelude (`L0..L3`, visits `0..3`)**: Executes once on the raw token embeddings; `L3` is Global Visit 0.\n"
                        "2. **Recurrent Core (`L4..L27` × 2 + `L12..L23` × 1, visits `4..63`)**: Wrapped by 24 `TwoStreamMHC` "
                        "controllers that route the two residual streams `(s0, s1)` through 60 controller visits while `VisitAwareDLSAIndexBank` "
                        "assigns distinct `(W_k, W_q)` projections to each of the 15 core global visits.\n"
                        "3. **Coda (`L28..L35`, visits `64..71`)**: Reads `s0` from the recurrent core and completes Global Visits 16 (`L31`) and 17 (`L35`)."
                    ),
                },
            ],
        },
    ]
    examples.extend(build_omp_tool_trajectory_examples())
    return examples


def render_example_to_Ids_and_labels(
    example: Dict[str, Any],
    tokenizer: Any,
    max_seq_len: int = 4096,
) -> Tuple[List[int], List[int]]:
    """Render a chat/tool conversation into `(input_ids, labels)` where user/system/tool-result tokens are masked to `-100`
    and assistant reasoning, `<tool_call>` JSON, and final answer tokens are supervised."""
    messages = example.get("messages", [])
    tools = example.get("tools")

    def _encode_text(txt: str) -> List[int]:
        if hasattr(tokenizer, "encode"):
            return list(tokenizer.encode(txt, add_special_tokens=False))
        return [max(1, ord(ch) % 120) for ch in txt]

    def _render_prefix(msgs: List[Dict[str, Any]], add_gen: bool = False) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                kwargs: Dict[str, Any] = {
                    "tokenize": False,
                    "add_generation_prompt": add_gen,
                }
                if tools:
                    kwargs["tools"] = tools
                return str(tokenizer.apply_chat_template(msgs, **kwargs))
            except Exception:
                pass
        parts: List[str] = []
        for m in msgs:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            reasoning = m.get("reasoning_content")
            t_calls = m.get("tool_calls")
            if role == "assistant":
                seg = "<|im_start|>assistant\n"
                if reasoning:
                    seg += f"<think>\n{reasoning}\n</think>\n"
                if t_calls:
                    for tc in t_calls:
                        fn = tc.get("function", {})
                        payload = json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments", {})}, ensure_ascii=False)
                        seg += f"<tool_call>\n{payload}\n</tool_call>\n"
                seg += f"{content}<|im_end|>\n"
                parts.append(seg)
            elif role == "tool":
                parts.append(f"<|im_start|>tool\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_gen:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    input_ids: List[int] = []
    labels: List[int] = []
    prev_text = ""
    for i, msg in enumerate(messages):
        cur_text = _render_prefix(messages[: i + 1], add_gen=False)
        delta_text = cur_text[len(prev_text) :] if cur_text.startswith(prev_text) else _render_prefix([msg], add_gen=False)
        prev_text = cur_text
        delta_ids = _encode_text(delta_text)
        if str(msg.get("role")) == "assistant":
            input_ids.extend(delta_ids)
            labels.extend(delta_ids)
        else:
            input_ids.extend(delta_ids)
            labels.extend([-100] * len(delta_ids))
        if len(input_ids) >= max_seq_len:
            break

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    # Ensure at least 2 supervised tokens exist
    if sum(1 for y in labels[1:] if y != -100) == 0 and len(input_ids) >= 2:
        half = max(1, len(input_ids) // 2)
        for j in range(half, len(input_ids)):
            labels[j] = input_ids[j]

    return input_ids, labels


def build_packed_training_batches(
    tokenizer: Any,
    seq_len: int = 2048,
    num_batches: int = 8,
    only_domain: Optional[str] = None,
    device: str = "cpu",
) -> List[Tuple["torch.LongTensor", "torch.LongTensor"]]:
    """Pack rendered 5-domain (or Stage-D `tool_trajectories`) examples into `[1, seq_len]` `(input_ids, labels)` tensors."""
    if not HAS_TORCH:
        raise RuntimeError("build_packed_training_batches requires PyTorch.")
    bank = build_omp_tool_trajectory_examples() if only_domain == "tool_trajectories" else build_five_domain_seed_bank()
    if only_domain is not None:
        filtered = [ex for ex in bank if ex.get("domain") == only_domain]
        if filtered:
            bank = filtered

    rendered = [render_example_to_Ids_and_labels(ex, tokenizer, max_seq_len=seq_len) for ex in bank]
    batches: List[Tuple[torch.LongTensor, torch.LongTensor]] = []
    tok_buf: List[int] = []
    lbl_buf: List[int] = []
    idx = 0
    while len(batches) < max(1, int(num_batches)):
        ids_i, lbl_i = rendered[idx % len(rendered)]
        idx += 1
        # Mask the first token of each packed document segment so cross-document boundary is not predicted
        if ids_i:
            lbl_i = list(lbl_i)
            lbl_i[0] = -100
        tok_buf.extend(ids_i)
        lbl_buf.extend(lbl_i)
        while len(tok_buf) >= int(seq_len) and len(batches) < max(1, int(num_batches)):
            t_slice = torch.tensor([tok_buf[:seq_len]], dtype=torch.long, device=device)
            y_slice = torch.tensor([lbl_buf[:seq_len]], dtype=torch.long, device=device)
            batches.append((t_slice, y_slice))
            tok_buf = tok_buf[seq_len:]
            lbl_buf = lbl_buf[seq_len:]
    return batches


def save_mhc_training_checkpoint(
    model: "SparkXLoopV03",
    output_dir: str | Path,
    cfg: MHCTrainingConfig,
    stage_name: str,
    global_step: int,
    metrics: Dict[str, Any],
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> Dict[str, Any]:
    """Export `loop_v03_state.pt`, `loop_config_v03.json`, `training_manifest.json`, and optional `optimizer.pt`."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    state_file = out_path / "loop_v03_state.pt"
    config_file = out_path / "loop_config_v03.json"
    manifest_file = out_path / "training_manifest.json"

    state_payload = loop_v03_state_dict(model, include_tier1b=False)
    torch.save(state_payload, state_file)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out_path / "optimizer.pt")

    h = hashlib.sha256()
    with open(state_file, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    state_sha256 = h.hexdigest()

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
        "training_config": cfg.to_dict(),
        "controller_summary": ctrl_sum,
        "execution_manifest": exec_manifest,
        "metrics": metrics_snapshot,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_file.write_text(json.dumps(train_manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return train_manifest


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
    """Normalize a display name like `'Spark X2.6 4B Recursive72'` or `'owner/Spark X2.6 4B Recursive72'`
    into a valid Hugging Face Hub `<owner>/Spark-X2.6-4B-Recursive72` repository ID."""
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


def build_recursive72_model_card(
    cfg: MHCTrainingConfig,
    report: Dict[str, Any],
    repo_id: str,
) -> str:
    """Build the `README.md` Hugging Face model card for `Spark X2.6 4B Recursive72`."""
    model_title = str(cfg.hf_model_name or DEFAULT_HF_MODEL_NAME)
    stage_c = report.get("stages", {}).get("stage_c", {})
    stage_d = report.get("stages", {}).get("stage_d", {})
    stage_e = report.get("stages", {}).get("stage_e", {})
    ctrl_sum = (
        stage_d.get("controller_summary")
        or stage_c.get("controller_summary")
        or report.get("stages", {}).get("stage_a", {}).get("controller_summary", {})
    )
    prov = stage_e.get("provenance", {})
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
        f"model built strictly atop the frozen BF16 **`{cfg.model_id}`** (`revision={cfg.revision}`) 36-layer hybrid backbone.\n\n"
        "## Architecture Summary\n"
        "- **Base Backbone**: `XHToken/Spark-X2.6-4B` (36 physical layers = 9 full-attention + 27 sliding-window layers, native GELU, frozen BF16).\n"
        "- **72-Visit Recurrent Schedule (`4 + 24 + 24 + 12 + 8 = 72`)**:\n"
        "  - Prelude `L0..L3` (1 pass = 4 visits)\n"
        "  - Core Pass 1 `L4..L27` (24 visits)\n"
        "  - Core Pass 2 `L4..L27` (24 visits)\n"
        "  - Inner Pass 3 `L12..L23` (12 visits)\n"
        "  - Coda `L28..L35` (1 pass = 8 visits)\n"
        f"- **Differentiable `TwoStreamMHC` Controllers (`{EXPECTED_DEFAULT_CONTROLLER_PARAMS:,}` params)**:\n"
        f"  - Bounded-gain Schema v2 (`gain_mode={ctrl_sum.get('gain_mode', cfg.gain_mode)!r}`, "
        f"`mhc_schema_version={ctrl_sum.get('mhc_schema_version', 2)}`), guaranteeing $(1 - g)I + gP \\in \\mathcal{{B}}_2$ across all 60 core visits.\n"
        f"  - Mean Pass-2 Gain (Attn/FFN): `{ctrl_sum.get('mean_gain_p2_attn', cfg.init_p2_gain)}` / `{ctrl_sum.get('mean_gain_p2_ffn', cfg.init_p2_gain)}`\n"
        f"  - Mean Pass-3 Gain (Attn/FFN): `{ctrl_sum.get('mean_gain_p3_attn', cfg.init_p3_gain)}` / `{ctrl_sum.get('mean_gain_p3_ffn', cfg.init_p3_gain)}`\n"
        "- **18-Visit `VisitAwareDLSAIndexBank`**:\n"
        f"  - Per-visit `(W_k, W_q)` $\\mathbb{{S}}^{{63}}$ projections distilled on the trained `learned_mhc` residual stream.\n"
        f"  - Locked `mhc_checkpoint_sha256`: `{prov.get('mhc_checkpoint_sha256', 'N/A')}`\n\n"
        "## Quickstart\n\n"
        "```python\n"
        "import torch\n"
        "from load_inference_v03 import load_sparkx_loop72_lsa\n\n"
        "model, tokenizer = load_sparkx_loop72_lsa(\n"
        f"    repo_or_dir={repo_id!r},\n"
        f"    base_repo={cfg.model_id!r},\n"
        f"    revision={cfg.revision!r},\n"
        "    recurrence_mode='learned_mhc',\n"
        "    attach_sparse_attention=True,\n"
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
    """Package the final `Spark X2.6 4B Recursive72` export directory (including `README.md`, `loop_v03_state.pt`,
    `visit_selectors_stage_e.pt`, `loop_config_v03.json`, runtime helper modules, and optional standalone backbone)
    and upload to Hugging Face Hub when `cfg.push_to_hub` is enabled."""
    import shutil

    exp_path = Path(export_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    token = resolve_hf_token(hf_token or cfg.hf_token)
    resolved_repo_id = normalize_hf_repo_id(cfg.hf_repo_id or cfg.hf_model_name, token=token, api_client=api_client)

    # Copy Stage E selector checkpoint into the export folder if present in parent output_dir
    parent_sel = Path(cfg.output_dir) / "visit_selectors_stage_e.pt"
    target_sel = exp_path / "visit_selectors_stage_e.pt"
    if parent_sel.is_file() and parent_sel.resolve() != target_sel.resolve():
        shutil.copy2(parent_sel, target_sel)

    # Copy helper runtime files so the Hugging Face repository is self-contained
    repo_root = Path(__file__).resolve().parent
    copied_helpers: List[str] = []
    for helper_name in (
        "sparkx_loop_v03_core.py",
        "sparkx_dlsa.py",
        "sparkx_paged_cache.py",
        "sparkx_sparse_attention.py",
        "load_inference_v03.py",
        "train_spark_dlsa.py",
        "train_spark_mhc.py",
    ):
        src_f = repo_root / helper_name
        if src_f.is_file():
            shutil.copy2(src_f, exp_path / helper_name)
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

    # Update loop_config_v03.json / loop_config.json with final model_name, hf_repo_id, and merged_standalone flag
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

    readme_text = build_recursive72_model_card(cfg, report, resolved_repo_id)
    (exp_path / "README.md").write_text(readme_text, encoding="utf-8")

    hub_result: Dict[str, Any] = {
        "model_name": str(cfg.hf_model_name or DEFAULT_HF_MODEL_NAME),
        "hf_repo_id": resolved_repo_id,
        "export_dir": str(exp_path),
        "standalone_backbone_saved": bool(standalone_saved),
        "copied_helpers": copied_helpers,
        "push_to_hub_requested": bool(cfg.push_to_hub),
        "uploaded": False,
        "hub_url": f"https://huggingface.co/{resolved_repo_id}",
    }

    if bool(cfg.push_to_hub):
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
            client.upload_folder(
                folder_path=str(exp_path),
                repo_id=resolved_repo_id,
                repo_type="model",
                commit_message=(
                    f"Upload {cfg.hf_model_name} (72-visit learned_mhc + 18-visit D-LSA, "
                    f"gain_mode={cfg.gain_mode})"
                ),
            )
            hub_result["uploaded"] = True
            print(
                f"[Hugging Face Hub] Uploaded `{cfg.hf_model_name}` -> https://huggingface.co/{resolved_repo_id}"
            )
        except Exception as exc:
            hub_result["error"] = str(exc)
            print(f"[Hugging Face Hub Error] Failed to upload `{resolved_repo_id}`: {exc}")

    return hub_result


def run_mhc_training_pipeline(
    model: "SparkXLoopV03",
    tokenizer: Any,
    cfg: Optional[MHCTrainingConfig] = None,
    run_stage_a: bool = True,
    run_stage_b: bool = True,
    run_stage_c: bool = True,
    run_stage_d: Optional[bool] = None,
    run_stage_e: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end Milestone 2 Loop72 `TwoStreamMHC` training & calibration pipeline (Stages A–E)."""
    if not HAS_TORCH:
        raise RuntimeError("run_mhc_training_pipeline requires PyTorch.")
    cfg = copy.deepcopy(cfg or MHCTrainingConfig())
    hw_info = apply_hardware_profile_to_config(cfg)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = str(next(model.base.parameters()).device)
    vocab_sz = int(getattr(model.base.config, "vocab_size", model._head_weight().shape[0]))
    report: Dict[str, Any] = {
        "hardware_profile": hw_info,
        "config": cfg.to_dict(),
        "stages": {},
    }

    # Stage A: Mechanical & Gradient Verification
    if run_stage_a:
        stage_a_seq = min(int(cfg.seq_len), 32)
        stage_a_rep = run_stage_a_verification(
            model, cfg=cfg, vocab_size=vocab_sz, seq_len=stage_a_seq
        )
        report["stages"]["stage_a"] = stage_a_rep
        if not stage_a_rep["passed"]:
            raise RuntimeError(f"Stage A verification failed: {stage_a_rep}")

    # Re-initialize clean Step-0 bounded-gain controllers before Stage B
    prepare_model_for_mhc_training(model, cfg=cfg, reset_gains=True)
    opt, opt_summary = build_mhc_optimizer(model, cfg=cfg)
    report["optimizer"] = opt_summary
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    batches_general = build_packed_training_batches(
        tokenizer,
        seq_len=int(cfg.seq_len),
        num_batches=max(4, min(32, int(cfg.stage_b_steps) + int(cfg.stage_c_steps))),
        only_domain=None,
        device=device,
    )

    global_step = 0
    total_bc_steps = max(1, int(cfg.stage_b_steps) + int(cfg.stage_c_steps))

    def _run_steps(
        num_steps: int,
        stage_label: str,
        batch_pool: Sequence[Tuple[torch.LongTensor, torch.LongTensor]],
        stage_cfg: MHCTrainingConfig,
        unfreeze_schedule: bool = False,
    ) -> Dict[str, Any]:
        nonlocal global_step
        loss_first: Optional[float] = None
        loss_last: float = 0.0
        ce_last: float = 0.0
        kl_last: float = 0.0
        t0 = time.perf_counter()
        model.train()

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
                opt,
                step=global_step,
                total_steps=total_bc_steps,
                warmup_ratio=float(stage_cfg.warmup_ratio),
                min_lr_ratio=float(stage_cfg.min_lr_ratio),
            )
            opt.zero_grad(set_to_none=True)
            accum = max(1, int(stage_cfg.grad_accum_steps))
            step_tot = 0.0
            step_ce = 0.0
            step_kl = 0.0
            for a_i in range(accum):
                inp, lbl = batch_pool[(global_step * accum + a_i) % len(batch_pool)]
                loss_t, m_dict = compute_teacher_retention_step_loss(
                    model, inp, lbl, cfg=stage_cfg, generator=gen
                )
                (loss_t / float(accum)).backward()
                step_tot += m_dict["total_loss"] / float(accum)
                step_ce += m_dict["ce_loss"] / float(accum)
                step_kl += m_dict["kl_loss"] / float(accum)

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.controllers.parameters() if p.requires_grad and p.grad is not None],
                    float(stage_cfg.max_grad_norm),
                ).item()
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if loss_first is None:
                loss_first = step_tot
            loss_last = step_tot
            ce_last = step_ce
            kl_last = step_kl

        # Restore full 72L loop if curriculum modified it
        model.set_loop(
            model.loop_start, model.loop_end, passes=2, use_mhc=True, use_engram=False,
            enable_p3=True, p3_start=model.p3_start, p3_end=model.p3_end,
        )
        configure_controller_unfreeze_phase(model, "full_controllers")
        dt = max(1e-6, time.perf_counter() - t0)
        tokens_processed = int(num_steps) * int(stage_cfg.grad_accum_steps) * int(stage_cfg.seq_len)
        return {
            "stage": stage_label,
            "steps": int(num_steps),
            "global_step": int(global_step),
            "loss_first": round(float(loss_first or 0.0), 6),
            "loss_last": round(float(loss_last), 6),
            "ce_loss": round(float(ce_last), 6),
            "kl_loss": round(float(kl_last), 6),
            "tokens_processed": tokens_processed,
            "tokens_per_sec": round(tokens_processed / dt, 2),
            "elapsed_sec": round(dt, 3),
            "controller_summary": model.controller_summary(),
        }

    # Stage B: 36L -> 72L Teacher Retention Warmup (gains + adapter up-projections unfrozen first)
    if run_stage_b and int(cfg.stage_b_steps) > 0:
        configure_controller_unfreeze_phase(model, "warmup_gains_and_up")
        stage_b_metrics = _run_steps(
            int(cfg.stage_b_steps),
            "stage_b",
            batches_general,
            cfg,
            unfreeze_schedule=False,
        )
        ckpt_b = save_mhc_training_checkpoint(
            model,
            out_root / "stage_b_retention_mhc",
            cfg=cfg,
            stage_name="stage_b_retention",
            global_step=global_step,
            metrics=stage_b_metrics,
            optimizer=opt,
        )
        stage_b_metrics["checkpoint"] = ckpt_b
        report["stages"]["stage_b"] = stage_b_metrics

    # Stage C: Multi-Pass Deliberation Training with Staged Unfreezing
    dense_mhc_ckpt_path: Optional[str] = None
    dense_mhc_sha256: Optional[str] = None
    if run_stage_c and int(cfg.stage_c_steps) > 0:
        stage_c_metrics = _run_steps(
            int(cfg.stage_c_steps),
            "stage_c",
            batches_general,
            cfg,
            unfreeze_schedule=True,
        )
        ckpt_c = save_mhc_training_checkpoint(
            model,
            out_root / "stage_c_dense_mhc",
            cfg=cfg,
            stage_name="stage_c_dense_mhc",
            global_step=global_step,
            metrics=stage_c_metrics,
            optimizer=opt,
        )
        dense_mhc_ckpt_path = ckpt_c["checkpoint_path"]
        dense_mhc_sha256 = ckpt_c["checkpoint_sha256"]
        stage_c_metrics["checkpoint"] = ckpt_c
        report["stages"]["stage_c"] = stage_c_metrics

    # Stage D: Optional Oh My Pi (`omp`) Native Tool-Use Specialization
    do_stage_d = bool(cfg.run_stage_d_omp) if run_stage_d is None else bool(run_stage_d)
    if do_stage_d and int(cfg.stage_d_steps) > 0:
        batches_omp = build_packed_training_batches(
            tokenizer,
            seq_len=int(cfg.seq_len),
            num_batches=max(2, min(16, int(cfg.stage_d_steps))),
            only_domain="tool_trajectories",
            device=device,
        )
        stage_d_cfg = copy.deepcopy(cfg)
        stage_d_cfg.kl_beta = max(0.25, float(cfg.kl_beta))
        stage_d_metrics = _run_steps(
            int(cfg.stage_d_steps),
            "stage_d_omp",
            batches_omp,
            stage_d_cfg,
            unfreeze_schedule=False,
        )
        ckpt_d = save_mhc_training_checkpoint(
            model,
            out_root / "stage_d_omp_mhc",
            cfg=stage_d_cfg,
            stage_name="stage_d_omp_mhc",
            global_step=global_step,
            metrics=stage_d_metrics,
            optimizer=opt,
        )
        dense_mhc_ckpt_path = ckpt_d["checkpoint_path"]
        dense_mhc_sha256 = ckpt_d["checkpoint_sha256"]
        stage_d_metrics["checkpoint"] = ckpt_d
        report["stages"]["stage_d"] = stage_d_metrics

    # Stage E: Post-mHC 18-Visit D-LSA Selector Calibration & Provenance Lock
    do_stage_e = bool(cfg.run_stage_e_dlsa) if run_stage_e is None else bool(run_stage_e)
    if do_stage_e:
        from sparkx_dlsa import DLSAIndexConfig, LOOP72_GLOBAL_VISIT_SCHEDULE, VisitAwareDLSAIndexBank
        from train_spark_dlsa import (
            VisitDistillConfig,
            capture_loop72_global_visit_qkv,
            distill_loop72_visit_bank,
        )

        model.eval()
        model.requires_grad_(False)
        cal_len = min(int(cfg.stage_e_seq_len), int(cfg.seq_len))
        cal_batches = build_packed_training_batches(
            tokenizer,
            seq_len=cal_len,
            num_batches=max(1, int(cfg.stage_e_calibration_seqs)),
            only_domain=None,
            device=device,
        )
        visit_samples: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {
            gv: [] for gv in range(len(LOOP72_GLOBAL_VISIT_SCHEDULE))
        }
        for inp_cal, _ in cal_batches:
            captured = capture_loop72_global_visit_qkv(model, inp_cal)
            for gv_idx, (q_t, k_t, _v_t) in captured.items():
                visit_samples[gv_idx].append((q_t, k_t))

        first_attn = model.base.model.layers[3].self_attn
        n_kv = int(getattr(first_attn, "num_key_value_heads", 4))
        head_dim = int(getattr(first_attn, "head_dim", 256))
        p_rot = float(model.base.config.get_partial_rotary_factor("full_attention")) if hasattr(model.base.config, "get_partial_rotary_factor") else 0.25
        rope_dim = int(round(head_dim * p_rot))
        d_idx = min(64, max(2, n_kv * (head_dim - rope_dim)))

        bank = VisitAwareDLSAIndexBank(
            base_cfg=DLSAIndexConfig(
                n_kv=n_kv,
                head_dim=head_dim,
                rope_dim=rope_dim,
                d_idx=d_idx,
            ),
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
            seed=int(cfg.seed),
        )
        distill_rep = distill_loop72_visit_bank(
            bank,
            visit_samples,
            cfg=distill_cfg,
            model=model,
            mhc_checkpoint_path=dense_mhc_ckpt_path,
            mhc_checkpoint_sha256=dense_mhc_sha256,
        )
        selector_path = out_root / "visit_selectors_stage_e.pt"
        torch.save(bank.export_manifest(), selector_path)
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
    else:
        save_mhc_training_checkpoint(
            model,
            out_root / "final_loop72_mhc_dlsa_export",
            cfg=cfg,
            stage_name="final_mhc_export",
            global_step=global_step,
            metrics={"stages_completed": list(report["stages"].keys())},
        )

    hub_export = export_and_upload_recursive72_to_hub(
        model,
        tokenizer,
        export_dir=out_root / "final_loop72_mhc_dlsa_export",
        cfg=cfg,
        report=report,
    )
    report["huggingface_export"] = hub_export

    summary_path = out_root / "mhc_training_report.json"
    summary_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2 Differentiable TwoStreamMHC Trainer for Spark X2.6 4B Recursive72."
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
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override gradient accumulation steps.")
    parser.add_argument("--adapter-lr", type=float, default=2e-4)
    parser.add_argument("--routing-lr", type=float, default=3e-5)
    parser.add_argument("--kl-beta", type=float, default=0.5)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--sampled-kl-positions", type=int, default=64)
    parser.add_argument("--curriculum", default="fixed_72", choices=("fixed_72", "36_60_72"))
    parser.add_argument("--stage-b-steps", type=int, default=100)
    parser.add_argument("--stage-c-steps", type=int, default=300)
    parser.add_argument("--stage-d-steps", type=int, default=60)
    parser.add_argument("--stage-e-distill-steps", type=int, default=150)
    parser.add_argument("--skip-stage-d", action="store_true", help="Skip Stage D Oh My Pi tool specialization.")
    parser.add_argument("--skip-stage-e", action="store_true", help="Skip Stage E D-LSA selector distillation.")
    parser.add_argument("--verify-only", action="store_true", help="Run Stage A mechanical & gradient verification only.")
    parser.add_argument(
        "--hf-model-name",
        default=DEFAULT_HF_MODEL_NAME,
        help="Model display name for Hugging Face model card and manifest (default: 'Spark X2.6 4B Recursive72').",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=DEFAULT_HF_REPO_ID,
        help="Target Hugging Face Hub repository ID or model name (default: 'TassP0lar/Spark-X2.6-4B-Recursive72').",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the trained 'Spark X2.6 4B Recursive72' package to Hugging Face Hub after training.",
    )
    parser.add_argument("--hf-private", action="store_true", help="Create private Hugging Face repository when pushing.")
    parser.add_argument("--hf-token", default=None, help="Explicit Hugging Face write token (or use HF_TOKEN env/Colab secret).")
    parser.add_argument(
        "--no-standalone-backbone",
        action="store_true",
        help="Export only loop controllers + D-LSA selectors without duplicating the 36L base weights.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = MHCTrainingConfig(
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
        kl_beta=args.kl_beta,
        kl_temperature=args.kl_temperature,
        sampled_kl_positions=args.sampled_kl_positions,
        curriculum=args.curriculum,
        stage_b_steps=args.stage_b_steps,
        stage_c_steps=args.stage_c_steps,
        stage_d_steps=args.stage_d_steps,
        stage_e_distill_steps=args.stage_e_distill_steps,
        run_stage_d_omp=not args.skip_stage_d,
        run_stage_e_dlsa=not args.skip_stage_e,
        hf_model_name=args.hf_model_name,
        hf_repo_id=args.hf_repo_id,
        push_to_hub=bool(args.push_to_hub),
        hf_private=bool(args.hf_private),
        hf_token=args.hf_token,
        export_standalone_backbone=not bool(args.no_standalone_backbone),
        seed=args.seed,
    )
    hw_info = apply_hardware_profile_to_config(cfg)
    if args.seq_len is not None:
        cfg.seq_len = int(args.seq_len)
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = int(args.grad_accum_steps)

    from load_inference_v03 import load_sparkx_loop72_lsa
    print(f"[train_spark_mhc] Hardware profile: {hw_info}")
    model, tokenizer = load_sparkx_loop72_lsa(
        repo_or_dir=cfg.model_id,
        base_repo=cfg.model_id,
        revision=cfg.revision,
        recurrence_mode="learned_mhc",
        attach_sparse_attention=False,
    )
    if args.verify_only:
        rep = run_stage_a_verification(
            model,
            cfg=cfg,
            vocab_size=int(getattr(model.base.config, "vocab_size", 131072)),
            seq_len=min(64, int(cfg.seq_len)),
        )
        print(json.dumps(rep, indent=2, default=str))
        return 0 if rep.get("passed") else 1

    rep = run_mhc_training_pipeline(model, tokenizer, cfg=cfg)
    print(f"[train_spark_mhc] Completed Stages A-E. Report saved to {Path(cfg.output_dir) / 'mhc_training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
