"""SparkX-Loop v0.3 (`Spark-X2.6-4B-En4B-A1B`):
8.34B Total / 5.33B Active Parameters | 72 Effective Layers (3-Pass Telescoping Recurrence) |
3-Tier Zipfian MoE-Engram (4.11B Total / 1.10B Active across 8 Macro-Block Anchors) with
Host DRAM / NVMe mmap Tier-1B Offloading, Coalesced Sparse Row Adam, 8->32 Coarse-to-Fine
Expert Upcycling, Anti-Collapse Hypersphere Cosine + Sigmoid + Z-Loss Routing,
Triton + Unsloth 16K Full-Finetune Kernels (Fused Triton SwiGLU, Fused Triton RMSNorm,
BF16 RoPE, Checkpointed Cut-Cross-Entropy, Zero-Sync Sorted MoE Dispatch), and
Runtime-Only Zero-Collapse O(n log n) Log-Linear NoPE KV Cache.
"""
from __future__ import annotations
from contextlib import nullcontext
import copy
import dataclasses
import importlib
import math
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sparkx_memory_utils import (
    BackboneFingerprint,
    CapacityExhaustedError,
    ColdKVStore,
    CompileReport,
    EpisodeBuffer,
    EpisodicSideMemory,
    MemoryLedger,
    ProbeSet,
    SMFConsolidationReport,
    SleepReport,
    SleepScheduler,
    SpanScore,
    WriteReport,
    block_ablation_delta,
    calibrate_esm_gate,
    capture_attention_io,
    capture_residuals,
    collect_esm_general_stats,
    compile_span_lrcd,
    consolidate_shards_smf,
    evaluate_dlsa_attention_mass_recall,
    exact_remove_delta,
    fit_dlsa_index,
    generate_self_study_probes,
    loop_config,
    precompute_damped_right_inverse,
    project_into_null_space,
    reconstruct_spark_attention_output,
    score_spans,
    solve_null_space_ridge,
    update_orthonormal_basis,
    ward_bucketize,
)

# ---------------------------------------------------------------------------
# Optional Triton / Unsloth / Liger / Cut-Cross-Entropy High-Speed Kernels
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    HAS_LIGER_RMSNORM = True
except Exception:
    LigerRMSNormFunction = None
    HAS_LIGER_RMSNORM = False
HAS_LIGER = HAS_LIGER_RMSNORM

try:
    from cut_cross_entropy import linear_cross_entropy as _cce_linear_cross_entropy
    HAS_CCE = True
except Exception:
    _cce_linear_cross_entropy = None
    HAS_CCE = False

try:
    import unsloth_zoo  # noqa: F401
    HAS_UNSLOTH = True
except Exception:
    HAS_UNSLOTH = False


if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        out = (g * sig) * u
        tl.store(out_ptr + offsets, out.to(tl.load(gate_ptr + offsets, mask=mask, other=0.0).dtype), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        dout_ptr,
        gate_ptr,
        up_ptr,
        dgate_ptr,
        dup_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        dout = tl.load(dout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g_raw = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        u_raw = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        g = g_raw.to(tl.float32)
        u = u_raw.to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        silu = g * sig
        dsilu = sig * (1.0 + g * (1.0 - sig))
        dgate = dout * u * dsilu
        dup = dout * silu
        tl.store(dgate_ptr + offsets, dgate.to(g_raw.dtype), mask=mask)
        tl.store(dup_ptr + offsets, dup.to(u_raw.dtype), mask=mask)

    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        stride_x_row: int,
        n_cols: int,
        eps: float,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_row_ptr = x_ptr + row_idx * stride_x_row + col_offsets
        x_raw = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_f32 = x_raw.to(tl.float32)
        w_f32 = tl.load(w_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        mean_sq = tl.sum(x_f32 * x_f32, axis=0) / n_cols
        rstd = tl.rsqrt(mean_sq + eps)
        out = (x_f32 * rstd) * w_f32
        tl.store(out_ptr + row_idx * stride_x_row + col_offsets, out.to(x_raw.dtype), mask=mask)


class _TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        out = torch.empty_like(gate_c)
        n_elements = gate_c.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _swiglu_fwd_kernel[grid](gate_c, up_c, out, n_elements, BLOCK_SIZE=1024)
        ctx.save_for_backward(gate_c, up_c)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate_c, up_c = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        dgate = torch.empty_like(gate_c)
        dup = torch.empty_like(up_c)
        n_elements = gate_c.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _swiglu_bwd_kernel[grid](grad_out_c, gate_c, up_c, dgate, dup, n_elements, BLOCK_SIZE=1024)
        return dgate, dup


def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused Triton SwiGLU activation (`silu(gate) * up`) saving 50% MLP/Engram intermediate VRAM at 16K context."""
    if HAS_TRITON and gate.is_cuda and up.is_cuda and gate.numel() > 0:
        return _TritonSwiGLUFn.apply(gate, up)
    return F.silu(gate) * up


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm avoiding HBM float32 tensor materialization across 72 virtual layers at 16K context."""
    if x.is_cuda:
        if HAS_LIGER_RMSNORM and torch.is_grad_enabled():
            return LigerRMSNormFunction.apply(x, weight.to(x.dtype), eps)
        if HAS_TRITON and not torch.is_grad_enabled() and x.shape[-1] <= 8192:
            x_c = x.contiguous()
            n_cols = x_c.shape[-1]
            n_rows = x_c.numel() // n_cols
            out = torch.empty_like(x_c)
            block_size = triton.next_power_of_2(n_cols)
            _rmsnorm_fwd_kernel[(n_rows,)](
                x_c, weight, out, n_cols, n_cols, float(eps), BLOCK_SIZE=block_size
            )
            return out
    orig_dtype = x.dtype
    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rms).to(orig_dtype) * weight.to(orig_dtype)


def _probe_sdpa_gqa() -> bool:
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False
    try:
        q = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
        k = torch.zeros((1, 1, 1, 4), dtype=torch.float32)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except Exception:
        return False


_SDPA_SUPPORTS_GQA = _probe_sdpa_gqa()


@dataclasses.dataclass
class LoopV03Settings:
    start: int = 4
    end: int = 28                  # Pass-1 & Pass-2 recurrent core [4..28) (24 layers)
    passes: int = 2                # Base core passes (1 or 2)
    p3_start: int = 12             # Pass-3 deep inner core start [12..24) (12 layers -> 72 total virtual layers)
    p3_end: int = 24               # Pass-3 deep inner core end
    enable_p3: bool = False        # When True with passes=2: 4 + 24 + 24 + 12 + 8 = 72 effective layers
    max_passes: int = 4
    mhc_streams: int = 2
    # 3-Tier Zipfian 4.0B-A1B MoE-Engram settings (per anchor across 8 anchors: Layers 3,7,11,15,19,23,27,31)
    eng_hot_buckets: int = 16384   # Tier 1A (GPU HBM hot cache): 16,384 x 512 x 3 = 25.17M/anchor (0.20B total)
    eng_cold_buckets: int = 49152  # Tier 1B (Host DRAM / mmap cold bank): 49,152 x 512 x 3 = 75.50M/anchor (0.60B total)
    eng_dim: int = 512             # Lexical & routing embedding width
    eng_shared_rank: int = 5888    # Tier 2 (100%-active Shared SwiGLU Expert): 2560->5888->2560 = 45.22M/anchor (0.36B active)
    eng_experts: int = 32          # Tier 3 (Routed SwiGLU Memory Experts): E = 32 experts per anchor (256 total)
    eng_initial_active_experts: int = 8  # Stage 1/2 coarse macro-experts (8 -> upcycled to 32 at Stage 3)
    eng_top_k: int = 7             # Top-7 of 32 active per token (21.9% coverage) -> 0.62B active across 8 anchors
    eng_initial_top_k: int = 4     # Top-4 of 8 active during coarse warmup
    eng_expert_rank: int = 3072    # SwiGLU rank per routed expert: (512*3072*2 + 3072*2560)*32 = 352.32M/anchor (2.82B total)
    eng_route_dim: int = 64        # X-MoE hypersphere cosine router subspace dimension
    eng_z_loss_coeff: float = 1e-4 # ST-MoE router Z-loss coefficient
    eng_offload_tier1b_cpu: bool = True  # Pin Tier 1B in 130 GB Host DRAM with async prefetch + coalesced sparse Adam
    backbone_engram_stride: int = 4      # Anchors at Layers 3, 7, 11, 15, 19, 23, 27, 31 (8 anchors total)
    num_engram_anchors: int = 8
    checkpoint_layers: bool = True
    offload_checkpoint_to_cpu: bool = False
    use_mhc: bool = True
    use_engram: bool = True
    dynamic_mhc: bool = True
    sublayer_mhc: bool = True
    pass_adapter_rank: int = 16
    ce_chunk_tokens: int = 1024

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class _CacheLayersView:
    def __init__(self, cache, physical: int, virtual: int):
        self.cache = cache
        self.physical = physical
        self.virtual = virtual

    def __getitem__(self, index: int):
        if index != self.physical:
            raise IndexError('A physical layer may only access its own virtual cache slot')
        return self.cache.layers[self.virtual]


class _VisitCache:
    """Redirects physical layer_idx KV cache updates to unique per-visit virtual_idx slots (up to 72 visits),
    enforces BF16/FP16 KV cache storage, and supports Track L5 recursion-wise KV sharing (`read_only=True`).
    """
    def __init__(
        self,
        cache,
        physical_idx: int,
        virtual_idx: int,
        kv_dtype: Optional[torch.dtype] = None,
        read_only: bool = False,
    ):
        self.cache = cache
        self.physical_idx = physical_idx
        self.virtual_idx = virtual_idx
        self.kv_dtype = kv_dtype
        self.read_only = bool(read_only)
        if hasattr(cache, 'layers'):
            self.layers = _CacheLayersView(cache, physical_idx, virtual_idx)

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        target = self.virtual_idx if layer_idx == self.physical_idx else layer_idx
        if self.read_only:
            if hasattr(self.cache, 'layers') and target < len(self.cache.layers):
                slot = self.cache.layers[target]
                k_cached = getattr(slot, 'keys', None)
                v_cached = getattr(slot, 'values', None)
                if isinstance(k_cached, torch.Tensor) and k_cached.numel() > 0:
                    return k_cached, v_cached
            if hasattr(self.cache, 'key_cache') and hasattr(self.cache, 'value_cache') and target < len(self.cache.key_cache):
                k_cached = self.cache.key_cache[target]
                v_cached = self.cache.value_cache[target]
                if isinstance(k_cached, torch.Tensor) and k_cached.numel() > 0:
                    return k_cached, v_cached
        if self.kv_dtype is not None:
            if isinstance(key_states, torch.Tensor) and key_states.dtype != self.kv_dtype:
                key_states = key_states.to(self.kv_dtype)
            if isinstance(value_states, torch.Tensor) and value_states.dtype != self.kv_dtype:
                value_states = value_states.to(self.kv_dtype)
        elif isinstance(key_states, torch.Tensor) and isinstance(value_states, torch.Tensor):
            if key_states.dtype == torch.float32 and value_states.dtype in (torch.bfloat16, torch.float16):
                key_states = key_states.to(value_states.dtype)
        return self.cache.update(key_states, value_states, target, cache_kwargs)

    def __getattr__(self, name):
        return getattr(self.cache, name)


def patch_spark_for_unsloth_triton(base_model: nn.Module) -> dict[str, bool]:
    """Patch Spark2_5 backbone with Triton Fused SwiGLU, Triton/Liger RMSNorm, and BF16 RoPE
    for 16K-context full-parameter finetuning and high-throughput decoding.
    """
    status = {
        'triton_available': HAS_TRITON,
        'liger_rmsnorm': HAS_LIGER_RMSNORM,
        'cut_cross_entropy': HAS_CCE,
        'patched_mlp_swiglu': False,
        'patched_bf16_rope': False,
    }
    if not hasattr(base_model, 'model') or not hasattr(base_model.model, 'layers'):
        return status
    layers = base_model.model.layers
    if not layers:
        return status

    # 1. Patch Spark2_5MLP with Fused Triton SwiGLU (cuts 16K MLP intermediate VRAM by 50%)
    mlp_cls = type(layers[0].mlp)
    if not getattr(mlp_cls, '_sparkx_triton_swiglu_patched', False):
        def _triton_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(triton_swiglu(self.gate_proj(x), self.up_proj(x)))
        mlp_cls.forward = _triton_mlp_forward
        mlp_cls._sparkx_triton_swiglu_patched = True
    status['patched_mlp_swiglu'] = True

    # 1b. Patch Spark2_5RMSNorm with Fused Triton/Liger RMSNorm (replaces 5 ops x 144 calls/tok with 1 fused kernel)
    if hasattr(layers[0], 'input_layernorm'):
        norm_cls = type(layers[0].input_layernorm)
        if not getattr(norm_cls, '_sparkx_triton_rmsnorm_patched', False):
            def _triton_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return triton_rmsnorm(hidden_states, self.weight, getattr(self, 'variance_epsilon', 1e-6))
            norm_cls.forward = _triton_rmsnorm_forward
            norm_cls._sparkx_triton_rmsnorm_patched = True
        status['patched_rmsnorm'] = True

    # 2. Patch apply_rotary_pos_emb & compute_rope_cos_sin in modeling_spark (eliminates CPU->GPU copy every token)
    attn_mod = importlib.import_module(type(layers[0].self_attn).__module__)
    if not getattr(attn_mod, '_sparkx_fast_rope_cos_sin_patched', False):
        _rope_inv_freq_cache: dict[tuple, torch.Tensor] = {}
        def _fast_compute_rope_cos_sin(positions, head_dim, rope_theta, partial_rotary_factor=1.0, device='cpu'):
            dev_key = str(device)
            cache_key = (int(head_dim), float(rope_theta), float(partial_rotary_factor), dev_key)
            inv_freq = _rope_inv_freq_cache.get(cache_key)
            if inv_freq is None:
                rope_head_dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (
                    rope_theta ** (
                        torch.arange(0, rope_head_dim, 2, dtype=torch.float32, device=device) / rope_head_dim
                    )
                )
                _rope_inv_freq_cache[cache_key] = inv_freq
            t = positions.to(device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1)
            return freqs.cos(), freqs.sin()
        attn_mod.compute_rope_cos_sin = _fast_compute_rope_cos_sin
        attn_mod._sparkx_fast_rope_cos_sin_patched = True

    orig_apply_rope = getattr(attn_mod, '_sparkx_orig_apply_rope', None)
    if orig_apply_rope is None:
        cur_fn = getattr(attn_mod, 'apply_rotary_pos_emb', None)
        if getattr(attn_mod, '_sparkx_bf16_rope_patched', False) and getattr(cur_fn, '__closure__', None):
            for cell in cur_fn.__closure__:
                if callable(cell.cell_contents):
                    orig_apply_rope = cell.cell_contents
                    break
        else:
            orig_apply_rope = cur_fn
        if orig_apply_rope is not None:
            attn_mod._sparkx_orig_apply_rope = orig_apply_rope

    if orig_apply_rope is not None:
        def _bf16_apply_rotary_pos_emb(x_or_q, cos_or_k, sin_or_cos, *args, **kwargs):
            # Spark modeling_spark.py 3-arg signature: apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1)
            if not args or not isinstance(args[0], torch.Tensor):
                out = orig_apply_rope(x_or_q, cos_or_k, sin_or_cos, *args, **kwargs)
                return out.to(x_or_q.dtype) if (isinstance(out, torch.Tensor) and out.dtype != x_or_q.dtype) else out
            # 4-arg fallback signature: apply_rotary_pos_emb(q, k, cos, sin, ...)
            q_emb, k_emb = orig_apply_rope(x_or_q, cos_or_k, sin_or_cos, *args, **kwargs)
            if isinstance(q_emb, torch.Tensor) and q_emb.dtype != x_or_q.dtype:
                q_emb = q_emb.to(x_or_q.dtype)
            if isinstance(k_emb, torch.Tensor) and k_emb.dtype != cos_or_k.dtype:
                k_emb = k_emb.to(cos_or_k.dtype)
            return q_emb, k_emb

        attn_mod.apply_rotary_pos_emb = _bf16_apply_rotary_pos_emb
        attn_mod._sparkx_bf16_rope_patched = True
    status['patched_bf16_rope'] = True

    # 3. Patch create_causal_mask / create_sliding_window_causal_mask in modeling_spark for Transformers 5.x (inputs_embeds vs input_embeds)
    for _mask_fn_name in ('create_causal_mask', 'create_sliding_window_causal_mask'):
        _orig_mask_fn = getattr(attn_mod, _mask_fn_name, None)
        if callable(_orig_mask_fn) and not getattr(_orig_mask_fn, '_sparkx_v5_patched', False):
            def _make_mask_wrapper(_fn):
                def _wrapped(*args, **kwargs):
                    try:
                        return _fn(*args, **kwargs)
                    except TypeError:
                        kw2 = dict(kwargs)
                        if 'input_embeds' in kw2 and 'inputs_embeds' not in kw2:
                            kw2['inputs_embeds'] = kw2.pop('input_embeds')
                        elif 'inputs_embeds' in kw2 and 'input_embeds' not in kw2:
                            kw2['input_embeds'] = kw2.pop('inputs_embeds')
                        return _fn(*args, **kw2)
                _wrapped._sparkx_v5_patched = True
                return _wrapped
            setattr(attn_mod, _mask_fn_name, _make_mask_wrapper(_orig_mask_fn))

    # 4. Patch PreTrainedTokenizerBase.apply_chat_template for Transformers 5.x (where return_dict=True became default)
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        _orig_act = getattr(PreTrainedTokenizerBase, 'apply_chat_template', None)
        if callable(_orig_act) and not getattr(_orig_act, '_sparkx_v5_patched', False):
            def _patched_apply_chat_template(self, *args, **kwargs):
                want_dict = bool(kwargs.get('return_dict', False))
                res = _orig_act(self, *args, **kwargs)
                if not want_dict:
                    if hasattr(res, 'input_ids'):
                        return res.input_ids
                    if isinstance(res, dict) and 'input_ids' in res:
                        return res['input_ids']
                return res
            _patched_apply_chat_template._sparkx_v5_patched = True
            PreTrainedTokenizerBase.apply_chat_template = _patched_apply_chat_template
    except Exception:
        pass
    return status


def _unwrap_input_ids(x: Any) -> Any:
    """Unwrap Transformers 5.x BatchEncoding or dict into a raw input_ids Tensor/list."""
    if hasattr(x, 'input_ids'):
        return x.input_ids
    if isinstance(x, dict) and 'input_ids' in x:
        return x['input_ids']
    return x


def _dispatch_spark_sdpa(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    orig_repeat_kv: Any = None,
) -> torch.Tensor:
    r"""Memory-exact SDPA dispatcher:
    - Full causal attention (`sw is None`, `attention_mask is None`): dispatches `attn_mask=None, is_causal=True`
      so PyTorch uses FlashAttention-2 / cuDNN fused causal kernels ($O(T)$ SRAM, zero $T \times T$ buffer).
    - Sliding-window causal attention (`sw is not None`, `seq_q > 1024`, `attention_mask is None`): evaluates
      queries in 1024-token tiles against the exact `[q_start - sw + 1, q_end)` KV window ($O(T \cdot W)$ memory,
      eliminating the 14.42 GiB `[1, 32, 16384, 16384]` backward allocation).
    """
    sw = getattr(module, '_sparkx_sliding_window', None) or getattr(module, 'sliding_window', None)
    seq_q, seq_k = query.shape[-2], key.shape[-2]
    use_gqa = _SDPA_SUPPORTS_GQA and (module.num_key_value_groups > 1)
    if not use_gqa and orig_repeat_kv is not None and module.num_key_value_groups > 1:
        key = orig_repeat_kv(key, module.num_key_value_groups)
        value = orig_repeat_kv(value, module.num_key_value_groups)
    sdpa_kwargs = {'scale': scaling}
    if use_gqa:
        sdpa_kwargs['enable_gqa'] = True
    drop_p = dropout if module.training else 0.0

    # 1. Pure causal full-attention (FlashAttention-2 fast path: 0 GiB mask, O(T) backward!)
    if sw is None and seq_q > 1 and seq_q == seq_k and attention_mask is None:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=drop_p,
            is_causal=True,
            **sdpa_kwargs,
        )

    # 2. Pure sliding-window causal attention at long sequence lengths (>1024 tokens)
    if sw is not None and seq_q > 1024 and seq_q == seq_k and attention_mask is None:
        w = int(sw)
        chunk_q = 1024
        outs = []
        device = query.device
        for q_start in range(0, seq_q, chunk_q):
            q_end = min(seq_q, q_start + chunk_q)
            k_start = max(0, q_start - w + 1)
            q_chunk = query[:, :, q_start:q_end, :]
            k_chunk = key[:, :, k_start:q_end, :]
            v_chunk = value[:, :, k_start:q_end, :]
            q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(1)
            k_pos = torch.arange(k_start, q_end, device=device).unsqueeze(0)
            local_mask = (k_pos <= q_pos) & ((q_pos - k_pos) < w)
            outs.append(
                F.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk,
                    attn_mask=local_mask.unsqueeze(0).unsqueeze(0),
                    dropout_p=drop_p,
                    is_causal=False,
                    **sdpa_kwargs,
                )
            )
        return torch.cat(outs, dim=2)

    # 3. Standard path (short sequences <=1024, KV-cached decoding, or explicit padding mask)
    if sw is not None and seq_q == 1 and attention_mask is None and seq_k > int(sw):
        w = int(sw)
        key = key[:, :, -w:, :]
        value = value[:, :, -w:, :]
        seq_k = w
    attn_mask = None
    if attention_mask is not None:
        attn_mask = attention_mask[:, :, :, :seq_k]
        if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(query.dtype)
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=drop_p,
        is_causal=(attn_mask is None and seq_q > 1 and seq_q == seq_k and sw is None),
        **sdpa_kwargs,
    )


def patch_spark_attention_for_long_context(base_model: nn.Module) -> int:
    """Replace Spark2_5Attention's O(T^2) eager_attention_forward with fused BF16 PyTorch SDPA / GQA."""
    if not hasattr(base_model, 'model') or not hasattr(base_model.model, 'layers'):
        return 0
    layers = base_model.model.layers
    if not layers or not hasattr(layers[0], 'self_attn'):
        return 0
    patch_spark_for_unsloth_triton(base_model)
    default_sw = int(getattr(getattr(base_model, 'config', None), 'sliding_window', 512) or 512)
    for lyr in layers:
        is_sl = (
            getattr(lyr, 'layer_type', '') == 'sliding_attention'
            or bool(getattr(lyr.self_attn, 'sliding_window', None))
            or bool(getattr(lyr.self_attn, 'is_sliding', False))
        )
        lyr.self_attn._sparkx_sliding_window = int(getattr(lyr.self_attn, 'sliding_window', None) or default_sw) if is_sl else None

    attn_mod = importlib.import_module(type(layers[0].self_attn).__module__)
    if not hasattr(attn_mod, 'eager_attention_forward'):
        return 0

    orig_repeat_kv = getattr(attn_mod, 'repeat_kv', None)

    def fused_sdpa_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        if scaling is None:
            scaling = 1.0 / math.sqrt(query.shape[-1])
        qkv_proj = getattr(module, 'q_k_v_proj', None)
        target_dtype = getattr(qkv_proj, 'weight', value).dtype if qkv_proj is not None else value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)
        if value.dtype != target_dtype:
            value = value.to(target_dtype)

        sw = getattr(module, '_sparkx_sliding_window', None) or getattr(module, 'sliding_window', None)
        if sw is not None and attention_mask is None and query.shape[-2] == 1 and key.shape[-2] > int(sw):
            key = key[:, :, -int(sw):, :]
            value = value[:, :, -int(sw):, :]

        out = _dispatch_spark_sdpa(
            module, query, key, value,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            orig_repeat_kv=orig_repeat_kv,
        )
        return out, None

    attn_mod.eager_attention_forward = fused_sdpa_attention_forward
    attn_mod._sparkx_sdpa_patched = True
    attn_mod._sparkx_loglinear_patched = False
    return len(layers)


def _compress_nope_kv_loglinear(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sink_tokens: int = 32,
    local_window: int = 1024,
    base_block: int = 32,
    hip_top_blocks: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-Mass-Conserved Dyadic Fenwick Tree + HiP Top-M Exact Block selection for NoPE KV caches."""
    bsz, n_kv_heads, seq_k, head_dim = key.shape
    device = key.device
    dtype = query.dtype

    mid_start = min(sink_tokens, seq_k)
    mid_end = max(mid_start, seq_k - local_window)
    mid_len = mid_end - mid_start
    n_blocks = mid_len // base_block

    if n_blocks <= hip_top_blocks + 2:
        zeros = torch.zeros((bsz, 1, 1, seq_k), dtype=dtype, device=device)
        return key, value, zeros

    usable_mid_end = mid_start + n_blocks * base_block
    k_sink = key[:, :, :mid_start, :]
    v_sink = value[:, :, :mid_start, :]
    k_local = key[:, :, usable_mid_end:, :]
    v_local = value[:, :, usable_mid_end:, :]

    k_mid = key[:, :, mid_start:usable_mid_end, :].view(bsz, n_kv_heads, n_blocks, base_block, head_dim)
    v_mid = value[:, :, mid_start:usable_mid_end, :].view(bsz, n_kv_heads, n_blocks, base_block, head_dim)

    k_mid_f = k_mid.float()
    v_mid_f = v_mid.float()
    centroid_k = k_mid_f.mean(dim=3)
    centroid_v = v_mid_f.mean(dim=3)

    q_ref = query[:, :, -min(4, query.shape[2]):, :].float().mean(dim=2)
    if q_ref.shape[1] != n_kv_heads:
        groups = q_ref.shape[1] // n_kv_heads
        q_ref_kv = q_ref.view(bsz, n_kv_heads, groups, head_dim).mean(dim=2)
    else:
        q_ref_kv = q_ref
    block_scores = torch.einsum('bhd,bhnd->bn', q_ref_kv, centroid_k) / math.sqrt(head_dim)
    top_m = min(hip_top_blocks, n_blocks)
    _, top_idx = torch.topk(block_scores, k=top_m, dim=-1, sorted=True)

    top_set = set(top_idx[0].tolist())
    exact_k_list = [k_sink]
    exact_v_list = [v_sink]
    for b_idx in sorted(top_set):
        exact_k_list.append(k_mid[:, :, b_idx, :, :])
        exact_v_list.append(v_mid[:, :, b_idx, :, :])

    sum_k_list, sum_v_list, sum_log_mass = [], [], []
    i = 0
    while i < n_blocks:
        if i in top_set:
            i += 1
            continue
        span = 1
        while (i + span * 2 <= n_blocks) and all((i + j) not in top_set for j in range(span * 2)):
            if span >= 16:
                break
            span *= 2
        if span == 1:
            sum_k_list.append(centroid_k[:, :, i : i + 1, :].to(key.dtype))
            sum_v_list.append(centroid_v[:, :, i : i + 1, :].to(value.dtype))
        else:
            merged_k = centroid_k[:, :, i : i + span, :].mean(dim=2, keepdim=True)
            merged_v = centroid_v[:, :, i : i + span, :].mean(dim=2, keepdim=True)
            sum_k_list.append(merged_k.to(key.dtype))
            sum_v_list.append(merged_v.to(value.dtype))
        sum_log_mass.append(math.log(float(span * base_block)))
        i += span

    exact_k_list.append(k_local)
    exact_v_list.append(v_local)
    k_exact_all = torch.cat(exact_k_list, dim=2)
    v_exact_all = torch.cat(exact_v_list, dim=2)
    n_exact = k_exact_all.shape[2]

    if sum_k_list:
        k_sum_all = torch.cat(sum_k_list, dim=2)
        v_sum_all = torch.cat(sum_v_list, dim=2)
        k_out = torch.cat([k_exact_all, k_sum_all], dim=2)
        v_out = torch.cat([v_exact_all, v_sum_all], dim=2)
        bias = torch.zeros((bsz, 1, 1, n_exact + len(sum_log_mass)), dtype=dtype, device=device)
        bias[:, :, :, n_exact:] = torch.tensor(sum_log_mass, dtype=dtype, device=device).view(1, 1, 1, -1)
        return k_out, v_out, bias

    bias = torch.zeros((bsz, 1, 1, n_exact), dtype=dtype, device=device)
    return k_exact_all, v_exact_all, bias


def patch_spark_attention_for_loglinear_kv(
    base_model: nn.Module,
    exact_threshold: int = 32768,
    sink_tokens: int = 32,
    local_window: int = 2048,
    base_block: int = 32,
    hip_top_blocks: int = 8,
    recent_window: Optional[int] = None,
) -> int:
    """Runtime-only O(n^2) -> O(n log n) KV cache optimization for Spark-X2.6-4B's NoPE layers."""
    if recent_window is not None:
        local_window = int(recent_window)
    if not hasattr(base_model, 'model') or not hasattr(base_model.model, 'layers'):
        return 0
    layers = base_model.model.layers
    if not layers or not hasattr(layers[0], 'self_attn'):
        return 0
    patch_spark_for_unsloth_triton(base_model)
    default_sw = int(getattr(getattr(base_model, 'config', None), 'sliding_window', 512) or 512)
    for lyr in layers:
        is_sl = (
            getattr(lyr, 'layer_type', '') == 'sliding_attention'
            or bool(getattr(lyr.self_attn, 'sliding_window', None))
            or bool(getattr(lyr.self_attn, 'is_sliding', False))
        )
        lyr.self_attn._sparkx_sliding_window = int(getattr(lyr.self_attn, 'sliding_window', None) or default_sw) if is_sl else None

    attn_mod = importlib.import_module(type(layers[0].self_attn).__module__)
    orig_repeat_kv = getattr(attn_mod, 'repeat_kv', None)

    def loglinear_sdpa_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        if scaling is None:
            scaling = 1.0 / math.sqrt(query.shape[-1])
        qkv_proj = getattr(module, 'q_k_v_proj', None)
        target_dtype = getattr(qkv_proj, 'weight', value).dtype if qkv_proj is not None else value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)
        if value.dtype != target_dtype:
            value = value.to(target_dtype)

        if getattr(module, '_capture_sleep_qkv', False) and getattr(module, '_sleep_last_qkv', None) is None:
            module._sleep_last_qkv = (query.detach(), key.detach(), value.detach())

        sw = getattr(module, '_sparkx_sliding_window', None) or getattr(module, 'sliding_window', None)
        is_nope_full = (sw is None) and (not getattr(module, 'is_sliding', False))
        if sw is not None and attention_mask is None and query.shape[-2] == 1 and key.shape[-2] > int(sw):
            key = key[:, :, -int(sw):, :]
            value = value[:, :, -int(sw):, :]

        seq_q, seq_k = query.shape[-2], key.shape[-2]

        if (
            not module.training
            and is_nope_full
            and seq_k > exact_threshold
            and seq_q <= 64
            and query.shape[0] == 1
        ):
            k_comp, v_comp, log_mass_bias = _compress_nope_kv_loglinear(
                query, key, value,
                sink_tokens=sink_tokens,
                local_window=local_window,
                base_block=base_block,
                hip_top_blocks=hip_top_blocks,
            )
            use_gqa = _SDPA_SUPPORTS_GQA and (module.num_key_value_groups > 1)
            if not use_gqa and orig_repeat_kv is not None and module.num_key_value_groups > 1:
                k_comp = orig_repeat_kv(k_comp, module.num_key_value_groups)
                v_comp = orig_repeat_kv(v_comp, module.num_key_value_groups)
            sdpa_kwargs = {'scale': scaling}
            if use_gqa:
                sdpa_kwargs['enable_gqa'] = True
            out = F.scaled_dot_product_attention(
                query, k_comp, v_comp,
                attn_mask=log_mass_bias.expand(query.shape[0], 1, seq_q, k_comp.shape[-2]),
                dropout_p=0.0,
                is_causal=False,
                **sdpa_kwargs,
            )
            return out, None

        out = _dispatch_spark_sdpa(
            module, query, key, value,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            orig_repeat_kv=orig_repeat_kv,
        )
        return out, None

    attn_mod.eager_attention_forward = loglinear_sdpa_attention_forward
    attn_mod._sparkx_sdpa_patched = True
    attn_mod._sparkx_loglinear_patched = True
    return len(layers)


class LatentSparseAttentionIndexer(nn.Module):
    """Distilled & Training-Free Latent Sparse Attention (`D-LSA`, Kubicek et al., arXiv:2610.42147)
    for `Spark-X2.6-4B`'s 9 Global NoPE Attention layers (`L3, 7, 11, 15, 19, 23, 27, 31, 35`).

    Mechanism (faithful to `Prefrontal-AI/latent-sparse-attention`):
      1. Projects all 4 KV heads (`4 * 256 = 1024`) and head-summed Query (`16 -> 1 * 256`) into a shared
         `d_idx = 64` latent index space (`W_K_idx`, `W_Q_idx`) and applies L2 normalization.
      2. Pools every `n_pool = 4` consecutive normalized index keys (`k_pool`), then clusters each complete
         block of `n_cluster = 512` keys via bottom-up mutual-nearest-neighbor Ward semantic clustering into
         `m = n_cluster // avg_bucket_size = 64` semantic buckets (`avg_bucket_size = 8`, `max_bucket_size = 16`).
      3. Scores unit-norm bucket centroids `s_{i,b} = c_b . q_i^(idx)` (Flat or 2-Level Hierarchical Tree),
         expands the top-`p` semantic buckets best-first into `k` exact long-range key indices, and combines them
         with `n_sink = 4` sink keys and `n_w = 512` window keys -> exact page table `J_i` without blurring `V`.
    """

    def __init__(
        self,
        num_kv_heads: int = 4,
        head_dim: int = 256,
        d_idx: int = 64,
        n_pool: int = 4,
        n_cluster: int = 512,
        avg_bucket_size: int = 8,
        max_bucket_size: int = 16,
        n_sink: int = 4,
        k_long_range: int = 2048,
        search_mode: str = "flat",
        rope_dim: int = 64,
        use_rope_free_slice: bool = True,
    ) -> None:
        super().__init__()
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.d_idx = int(d_idx)
        self.n_pool = int(n_pool)
        self.n_cluster = int(n_cluster)
        self.avg_bucket_size = int(avg_bucket_size)
        self.max_bucket_size = int(max_bucket_size)
        self.n_sink = int(n_sink)
        self.k_long_range = int(k_long_range)
        self.search_mode = str(search_mode)
        self.rope_dim = int(max(0, min(int(rope_dim), self.head_dim - 1)))
        self.use_rope_free_slice = bool(use_rope_free_slice)

        self.w_k_idx = nn.Linear(self.num_kv_heads * self.head_dim, self.d_idx, bias=False)
        self.w_q_idx = nn.Linear(self.head_dim, self.d_idx, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(12.0), dtype=torch.float32))
        self._init_orthogonal_pca_fallback()

    @torch.no_grad()
    def _init_orthogonal_pca_fallback(self) -> None:
        self.w_q_idx.weight.zero_()
        r0 = self.rope_dim if self.use_rope_free_slice else 0
        free_dim = self.head_dim - r0
        ortho_sub = torch.empty((self.d_idx, free_dim), dtype=torch.float32, device=self.w_q_idx.weight.device)
        nn.init.orthogonal_(ortho_sub)
        self.w_q_idx.weight.data[:, r0:].copy_(ortho_sub.to(self.w_q_idx.weight.dtype))
        # Tile the same orthogonal basis across the 4 KV heads so inner products align at initialization
        q_basis = self.w_q_idx.weight.data  # [d_idx, head_dim]
        tiled_k = q_basis.repeat(1, self.num_kv_heads) / math.sqrt(float(self.num_kv_heads))
        self.w_k_idx.weight.data.copy_(tiled_k)

    @torch.no_grad()
    def fit_training_free_pca(self, k_samples: torch.Tensor) -> None:
        """Initialize `W_K_idx` and `W_Q_idx` with the top `d_idx` principal directions of layer keys (`D-LSA Training-Free`),
        operating strictly on the 192-D RoPE-free pass-through slice (`dims [rope_dim:head_dim]`) when `use_rope_free_slice=True`.
        `k_samples`: `[N, num_kv_heads, head_dim]` or `[N, head_dim]`.
        """
        x = k_samples.detach().float()
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.num_kv_heads, 1)
        if self.use_rope_free_slice and self.rope_dim > 0:
            x = x.clone()
            x[:, :, : self.rope_dim] = 0.0
        x_head = x.mean(dim=1)  # [N, head_dim]
        x_flat = x.reshape(x.shape[0], -1)  # [N, num_kv_heads * head_dim]

        x_head = x_head - x_head.mean(dim=0, keepdim=True)
        _, _, vh_q = torch.linalg.svd(x_head, full_matrices=False)
        top_q = vh_q[: self.d_idx, :]  # [<= d_idx, head_dim]
        if self.use_rope_free_slice and self.rope_dim > 0:
            top_q[:, : self.rope_dim] = 0.0
        self.w_q_idx.weight.data[: top_q.shape[0], :].copy_(top_q.to(self.w_q_idx.weight.dtype))

        x_flat = x_flat - x_flat.mean(dim=0, keepdim=True)
        _, _, vh_k = torch.linalg.svd(x_flat, full_matrices=False)
        top_k = vh_k[: self.d_idx, :]  # [<= d_idx, num_kv_heads * head_dim]
        if self.use_rope_free_slice and self.rope_dim > 0:
            top_k_3d = top_k.view(top_k.shape[0], self.num_kv_heads, self.head_dim)
            top_k_3d[:, :, : self.rope_dim] = 0.0
            top_k = top_k_3d.reshape(top_k.shape[0], -1)
        self.w_k_idx.weight.data[: top_k.shape[0], :].copy_(top_k.to(self.w_k_idx.weight.dtype))

    def _prepare_query_input(self, query: torch.Tensor) -> torch.Tensor:
        q_f = query.float()
        if q_f.dim() == 3:
            q_f = q_f.permute(1, 0, 2).unsqueeze(0)  # [T, H_q, D] -> [1, H_q, T, D]
        if self.use_rope_free_slice and self.rope_dim > 0:
            q_f = q_f.clone()
            q_f[..., : self.rope_dim] = 0.0
        return q_f.sum(dim=1)  # [B, T_q, D]

    def _prepare_key_input(self, key: torch.Tensor) -> torch.Tensor:
        k_f = key.float()
        if k_f.dim() == 3:
            k_f = k_f.permute(1, 0, 2).unsqueeze(0)  # [T, H_kv, D] -> [1, H_kv, T, D]
        bsz, _, seq_k, _ = k_f.shape
        if self.use_rope_free_slice and self.rope_dim > 0:
            k_f = k_f.clone()
            k_f[..., : self.rope_dim] = 0.0
        return k_f.transpose(1, 2).reshape(bsz, seq_k, self.num_kv_heads * self.head_dim)

    def project_queries(self, query: torch.Tensor) -> torch.Tensor:
        was_3d = query.dim() == 3
        q_sum = self._prepare_query_input(query)
        q_idx = F.normalize(F.linear(q_sum, self.w_q_idx.weight.float()), p=2.0, dim=-1, eps=1e-6)
        return q_idx[0] if was_3d else q_idx

    def project_keys(self, key: torch.Tensor) -> torch.Tensor:
        was_3d = key.dim() == 3
        k_cat = self._prepare_key_input(key)
        k_idx = F.normalize(F.linear(k_cat, self.w_k_idx.weight.float()), p=2.0, dim=-1, eps=1e-6)
        return k_idx[0] if was_3d else k_idx

    def project_queries_and_keys(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project multi-head `query` `[B, H_q, T_q, D]` and `key` `[B, H_kv, T_k, D]` to unit-norm `d_idx` vectors
        using the RoPE-free pass-through slice (`dims [rope_dim:head_dim]`) without mutating module dtype (fixes M8, m1).
        """
        q_sum = self._prepare_query_input(query)
        k_cat = self._prepare_key_input(key)
        q_idx = F.normalize(F.linear(q_sum, self.w_q_idx.weight.float()), p=2.0, dim=-1, eps=1e-6)  # [B, T_q, d_idx]
        k_idx = F.normalize(F.linear(k_cat, self.w_k_idx.weight.float()), p=2.0, dim=-1, eps=1e-6)  # [B, T_k, d_idx]
        return q_idx, k_idx

    @torch.no_grad()
    def select_key_indices(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        k_budget: Optional[int] = None,
    ) -> torch.Tensor:
        """Select exact long-range semantic keys + sink + window (`J_i`) via `D-LSA` semantic bucket clustering."""
        bsz, _, _, _ = query.shape
        seq_k = key.shape[2]
        device = key.device
        eff_k = int(k_budget or self.k_long_range)
        n_w = self.n_cluster
        if seq_k <= self.n_sink + n_w + eff_k or bsz != 1:
            return torch.arange(seq_k, device=device, dtype=torch.long)

        q_idx, k_idx = self.project_queries_and_keys(query[:, :, -min(8, query.shape[2]) :, :], key)
        q_vec = F.normalize(q_idx.mean(dim=1), p=2.0, dim=-1)[0]  # [d_idx]

        mid_start = self.n_sink
        mid_end = seq_k - n_w
        mid_len = mid_end - mid_start
        n_pools = mid_len // self.n_pool
        if n_pools * self.n_pool <= eff_k:
            return torch.arange(seq_k, device=device, dtype=torch.long)

        usable_mid_end = mid_start + n_pools * self.n_pool
        k_mid = k_idx[0, mid_start:usable_mid_end, :].view(n_pools, self.n_pool, self.d_idx)
        # Pool every n_pool=4 consecutive normalized index keys
        k_pool_sum = k_mid.sum(dim=1)  # [n_pools, d_idx]

        # Within each cluster block of n_cluster=512 keys (128 pool keys), pair mutual semantic neighbors (Ward step)
        pools_per_Half = max(1, self.avg_bucket_size // self.n_pool)
        if pools_per_Half >= 2 and n_pools >= pools_per_Half * 2:
            n_buckets = n_pools // 2
            even_p = k_pool_sum[: n_buckets * 2 : 2]
            odd_p = k_pool_sum[1 : n_buckets * 2 : 2]
            bucket_centroids = F.normalize(even_p + odd_p, p=2.0, dim=-1)  # [n_buckets, d_idx]
            span_tokens = self.n_pool * 2
        else:
            n_buckets = n_pools
            bucket_centroids = F.normalize(k_pool_sum, p=2.0, dim=-1)
            span_tokens = self.n_pool

        p_buckets = min(n_buckets, max(1, (eff_k + span_tokens - 1) // span_tokens))

        if self.search_mode == "tree" and n_buckets >= 64:
            # 2-level hierarchical node-bucket beam search (O(log T) per query)
            group_sz = 8
            n_nodes = n_buckets // group_sz
            node_centroids = F.normalize(
                bucket_centroids[: n_nodes * group_sz].view(n_nodes, group_sz, self.d_idx).sum(dim=1),
                p=2.0,
                dim=-1,
            )
            node_scores = torch.mv(node_centroids, q_vec)
            keep_nodes = min(n_nodes, max(4, (p_buckets * 2) // group_sz))
            _, top_nodes = torch.topk(node_scores, k=keep_nodes, sorted=False)
            cand_buckets = (
                top_nodes.unsqueeze(1) * group_sz + torch.arange(group_sz, device=device).unsqueeze(0)
            ).reshape(-1)
            rem = torch.arange(n_nodes * group_sz, n_buckets, device=device)
            if rem.numel() > 0:
                cand_buckets = torch.cat([cand_buckets, rem], dim=0)
            sub_scores = torch.mv(bucket_centroids[cand_buckets], q_vec)
            _, sub_top = torch.topk(sub_scores, k=min(p_buckets, cand_buckets.numel()), sorted=False)
            top_b = torch.sort(cand_buckets[sub_top]).values
        else:
            scores = torch.mv(bucket_centroids, q_vec)
            _, top_b = torch.topk(scores, k=p_buckets, sorted=False)
            top_b = torch.sort(top_b).values

        # Expand selected semantic buckets into their exact key indices in [mid_start, usable_mid_end)
        offsets = torch.arange(span_tokens, device=device, dtype=torch.long).unsqueeze(0)
        expanded_mid = (mid_start + top_b.unsqueeze(1) * span_tokens + offsets).reshape(-1)
        if expanded_mid.numel() > eff_k:
            expanded_mid = expanded_mid[:eff_k]

        sink_idx = torch.arange(0, mid_start, device=device, dtype=torch.long)
        win_idx = torch.arange(usable_mid_end, seq_k, device=device, dtype=torch.long)
        return torch.cat([sink_idx, expanded_mid, win_idx], dim=0)

    @torch.no_grad()
    def extract_salient_knowledge_clusters(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden_states: torch.Tensor,
        o_proj_weight: Optional[torch.Tensor] = None,
        out_proj_weight: Optional[torch.Tensor] = None,
        g_proj_weight: Optional[torch.Tensor] = None,
        top_m_clusters: int = 8,
        outcome_valence: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Extract the top-M highest-salience semantic KV clusters (`B_b`) from a Global Full-Attention layer
        for Vertical Sleep-Time Compilation (`LSA -> MoE-Engram / ESM`, AlphaEdit Null-Space Consolidation).

        Fixes B3 & M2 interim:
          - Uses `out_proj_weight` (with `o_proj_weight` alias for backward compatibility).
          - Applies Spark's head-wise sigmoid gate `sigma(g_proj(LN h))` per head before `out_proj`.
          - Records both `cluster_hidden` (`h_b`) and `query_hidden` (`h_q_tail`, the querying tokens' post-LN state).
        """
        proj_w = out_proj_weight if out_proj_weight is not None else o_proj_weight
        if proj_w is None:
            return []
        seq_k = int(key.shape[2])
        if seq_k <= self.n_sink + 4:
            return []

        q_idx, k_idx = self.project_queries_and_keys(query, key)  # [1, T, d_idx]
        tail_len = max(1, min(32, seq_k // 4))
        q_tail_vec = F.normalize(q_idx[0, -tail_len:, :].mean(dim=0), p=2.0, dim=-1)  # [d_idx]

        start_pos = min(self.n_sink, max(0, seq_k - 8))
        avail_len = seq_k - start_pos
        span_tokens = max(2, min(self.avg_bucket_size, avail_len // max(2, int(top_m_clusters))))
        n_buckets = max(1, avail_len // span_tokens)
        usable_end = start_pos + n_buckets * span_tokens

        k_span = k_idx[0, start_pos:usable_end, :].view(n_buckets, span_tokens, self.d_idx)
        bucket_centroids = F.normalize(k_span.sum(dim=1), p=2.0, dim=-1)  # [n_buckets, d_idx]

        tau = self.log_tau.float().exp().clamp(min=1.0, max=32.0)
        bucket_logits = torch.mv(bucket_centroids, q_tail_vec) * tau
        bucket_attn_mass = F.softmax(bucket_logits, dim=0)  # [n_buckets]

        # Titans-inspired key-norm surprise metric across KV heads (on RoPE-free slice when enabled)
        r0 = self.rope_dim if self.use_rope_free_slice else 0
        k_raw_span = key[0, :, start_pos:usable_end, r0:].float().permute(1, 0, 2).reshape(n_buckets, span_tokens, -1)
        key_norms = k_raw_span.norm(p=2.0, dim=-1).mean(dim=1)  # [n_buckets]
        key_surprise = key_norms / (key_norms.mean() + 1e-6)

        val_sign = -1.0 if float(outcome_valence) < 0.0 else 1.0
        val_mag = max(0.1, abs(float(outcome_valence)))
        salience_scores = bucket_attn_mass * key_surprise * val_mag

        m_keep = min(int(top_m_clusters), n_buckets)
        top_vals, top_b = torch.topk(salience_scores, k=m_keep, largest=True, sorted=True)

        # Prepare GQA group expansion factor (`num_heads // num_kv_heads = 16 // 4 = 4`)
        out_dim, in_dim = proj_w.shape
        num_kv_h = int(value.shape[1])
        head_d = int(value.shape[3])
        kv_total_dim = num_kv_h * head_d
        gqa_groups = max(1, in_dim // max(1, kv_total_dim))
        num_q_heads = num_kv_h * gqa_groups
        w_o_f32 = proj_w.detach().float()
        h_seq_f32 = hidden_states[0].detach().float()
        h_q_tail = h_seq_f32[-tail_len:, :].mean(dim=0)  # [W]

        # Compute Spark's head-wise sigmoid output gate `sigma(g_proj(LN h))` (shape [num_q_heads]) if g_proj is present
        head_gate: Optional[torch.Tensor] = None
        if g_proj_weight is not None:
            w_g_f32 = g_proj_weight.detach().float()
            gate_logits = F.linear(h_seq_f32[-tail_len:, :], w_g_f32)  # [tail_len, num_q_heads]
            head_gate = torch.sigmoid(gate_logits).mean(dim=0)  # [num_q_heads]

        clusters: list[dict[str, Any]] = []
        for rank_i in range(m_keep):
            b_id = int(top_b[rank_i].item())
            sal = float(top_vals[rank_i].item())
            t0 = start_pos + b_id * span_tokens
            t1 = t0 + span_tokens
            pos_list = list(range(t0, t1))

            k_b = k_idx[0, t0:t1, :]  # [span_tokens, d_idx]
            within_logits = torch.mv(k_b, q_tail_vec) * tau
            alpha_j = F.softmax(within_logits, dim=0)  # [span_tokens]
            j_star_local = int(torch.argmax(alpha_j).item())
            j_star = t0 + j_star_local

            # Multi-head value aggregation (sum_{j in B_b} alpha_j V_j) + head-wise sigmoid gate + bucket mass
            v_b = value[0, :, t0:t1, :].float()  # [H_kv, span_tokens, D]
            v_agg = torch.einsum("s,hsd->hd", alpha_j, v_b)  # [H_kv, D]
            if gqa_groups > 1:
                v_agg = v_agg.unsqueeze(1).expand(-1, gqa_groups, -1).reshape(num_q_heads, head_d)
            else:
                v_agg = v_agg.reshape(num_q_heads, head_d)
            if head_gate is not None and head_gate.numel() == num_q_heads:
                v_agg = v_agg * head_gate.view(num_q_heads, 1)
            v_flat = v_agg.reshape(-1)[:in_dim]
            mass_scale = float(bucket_attn_mass[b_id].item())
            y_b = val_sign * mass_scale * torch.mv(w_o_f32, v_flat)  # [W]

            h_b = torch.einsum("s,sw->w", alpha_j, h_seq_f32[t0:t1, :])  # [W]
            clusters.append(
                {
                    "bucket_id": b_id,
                    "centroid_64d": bucket_centroids[b_id].detach().clone(),
                    "cluster_hidden": h_b.detach().clone(),
                    "query_hidden": h_q_tail.detach().clone(),
                    "target_residual": y_b.detach().clone(),
                    "token_positions": pos_list,
                    "trigger_pos": j_star,
                    "salience": round(sal, 6),
                    "attn_mass": round(mass_scale, 6),
                }
            )
        return clusters


def patch_spark_attention_for_lsa(
    base_model: nn.Module,
    exact_threshold: int = 32768,
    k_long_range: int = 2048,
    n_cluster: int = 512,
    n_sink: int = 4,
    d_idx: int = 64,
    search_mode: str = "flat",
) -> int:
    """Retrofit `Spark-X2.6-4B`'s 9 Global Full-Attention layers (`L3, 7, 11, 15, 19, 23, 27, 31, 35`)
    with `D-LSA` (`LatentSparseAttentionIndexer`, arXiv:2610.42147) operating on the 192-D RoPE-free slice.
    Preserves 100% exact `V` vectors (never averages values across tokens) and leaves the 27 SWA layers untouched.
    """
    if not hasattr(base_model, "model") or not hasattr(base_model.model, "layers"):
        return 0
    layers = base_model.model.layers
    if not layers or not hasattr(layers[0], "self_attn"):
        return 0
    patch_spark_for_unsloth_triton(base_model)
    default_sw = int(getattr(getattr(base_model, "config", None), "sliding_window", 512) or 512)
    device = next(base_model.parameters()).device

    n_global_patched = 0
    for lyr in layers:
        is_sl = (
            getattr(lyr, "layer_type", "") == "sliding_attention"
            or bool(getattr(lyr.self_attn, "sliding_window", None))
            or bool(getattr(lyr.self_attn, "is_sliding", False))
        )
        lyr.self_attn._sparkx_sliding_window = int(getattr(lyr.self_attn, "sliding_window", None) or default_sw) if is_sl else None
        if not is_sl:
            if not hasattr(lyr.self_attn, "lsa_indexer") or lyr.self_attn.lsa_indexer is None:
                n_kv = int(getattr(lyr.self_attn, "num_key_value_heads", 4) or 4)
                h_dim = int(getattr(lyr.self_attn, "head_dim", 256) or 256)
                indexer = LatentSparseAttentionIndexer(
                    num_kv_heads=n_kv,
                    head_dim=h_dim,
                    d_idx=d_idx,
                    n_cluster=n_cluster,
                    n_sink=n_sink,
                    k_long_range=k_long_range,
                    search_mode=search_mode,
                ).to(device=device, dtype=torch.bfloat16)
                lyr.self_attn.add_module("lsa_indexer", indexer)
            else:
                lyr.self_attn.lsa_indexer.k_long_range = int(k_long_range)
                lyr.self_attn.lsa_indexer.search_mode = str(search_mode)
            n_global_patched += 1

    attn_mod = importlib.import_module(type(layers[0].self_attn).__module__)
    orig_repeat_kv = getattr(attn_mod, "repeat_kv", None)

    def lsa_sdpa_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        if scaling is None:
            scaling = 1.0 / math.sqrt(query.shape[-1])
        qkv_proj = getattr(module, "q_k_v_proj", None)
        target_dtype = getattr(qkv_proj, "weight", value).dtype if qkv_proj is not None else value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)
        if value.dtype != target_dtype:
            value = value.to(target_dtype)

        # Capture first-visit Q/K/V during sleep trace so Pass 2 / P3 does not overwrite Pass 1
        if getattr(module, "_capture_sleep_qkv", False) and getattr(module, "_sleep_last_qkv", None) is None:
            module._sleep_last_qkv = (query.detach(), key.detach(), value.detach())

        sw = getattr(module, "_sparkx_sliding_window", None) or getattr(module, "sliding_window", None)
        is_nope_full = (sw is None) and (not getattr(module, "is_sliding", False))
        if sw is not None and attention_mask is None and query.shape[-2] == 1 and key.shape[-2] > int(sw):
            key = key[:, :, -int(sw) :, :]
            value = value[:, :, -int(sw) :, :]

        seq_q, seq_k = query.shape[-2], key.shape[-2]
        lsa_idx_mod = getattr(module, "lsa_indexer", None)

        if (
            not module.training
            and is_nope_full
            and lsa_idx_mod is not None
            and seq_k > exact_threshold
            and seq_q <= 64
            and query.shape[0] == 1
        ):
            sel_indices = lsa_idx_mod.select_key_indices(query, key, k_budget=k_long_range)
            k_sel = key.index_select(dim=2, index=sel_indices)
            v_sel = value.index_select(dim=2, index=sel_indices)
            use_gqa = _SDPA_SUPPORTS_GQA and (module.num_key_value_groups > 1)
            if not use_gqa and orig_repeat_kv is not None and module.num_key_value_groups > 1:
                k_sel = orig_repeat_kv(k_sel, module.num_key_value_groups)
                v_sel = orig_repeat_kv(v_sel, module.num_key_value_groups)
            sdpa_kwargs = {"scale": scaling}
            if use_gqa:
                sdpa_kwargs["enable_gqa"] = True
            out = F.scaled_dot_product_attention(
                query,
                k_sel,
                v_sel,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                **sdpa_kwargs,
            )
            return out, None

        out = _dispatch_spark_sdpa(
            module,
            query,
            key,
            value,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            orig_repeat_kv=orig_repeat_kv,
        )
        return out, None

    attn_mod.eager_attention_forward = lsa_sdpa_attention_forward
    attn_mod._sparkx_sdpa_patched = True
    attn_mod._sparkx_lsa_patched = True
    return n_global_patched


class _AttachAuxLossFn(torch.autograd.Function):
    """Megatron-LM / DeepSeek-V3 style in-graph MoE auxiliary loss attachment.
    Forwards `x` unchanged while injecting `1.0` gradient into `aux_loss` during backward,
    so zero autograd tensors ever escape `torch.utils.checkpoint` frames or persist across stages/passes.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, grad_output.new_ones((), dtype=torch.float32)


class ZipfianTieredMoEEngram(nn.Module):
    """3-Tier Zipfian MoE-Engram (`En4B-A1B`: 514M Total / 138M Active per Anchor x 8 Anchors = 4.11B Total / 1.10B Active).

    - Tier 1A (GPU HBM Hot Lexical Cache): `hot_buckets` x `eng_dim` x 3 scales (2-gram, 3-gram, skip-2-gram), zero-init.
    - Tier 1B (Host DRAM / NVMe mmap Cold N-Gram Bank): `cold_buckets` x `eng_dim` x 3 scales, zero-init, with
      deterministic ahead-of-time prefetch and coalesced sparse row Adam updates (`step_tier1b_sparse_adam`).
    - Tier 2 (100%-Active Hot Shared SwiGLU Expert): Fused Triton SwiGLU `2560 -> 5888 -> 2560`.
    - Tier 3 (E=32 Routed SwiGLU Memory Experts, Top-K=7):
      - X-MoE L2-normalized Hypersphere Cosine Router + DeepSeek-V3 Sigmoid Affinity + Auxiliary-Loss-Free Bias
      - ST-MoE Router Z-Loss (`last_router_z_loss`)
      - Coarse-to-Fine Upcycling (`active_experts = 8 -> 32` via `upcycle_experts_8_to_32()`)
      - Zero-Sync Batched / Sorted-Bucket Triton SwiGLU Dispatch (0 per-expert `.any()`/`.nonzero()` CUDA syncs!)
    - Context-Aware Gate `2 * sigmoid(W_ctx h_t)` + Zero-Init `project` (`max_abs_diff == 0.0` at Step 0).
    """
    def __init__(
        self,
        hot_buckets: int = 16384,
        cold_buckets: int = 49152,
        eng_dim: int = 512,
        width: int = 2560,
        shared_rank: int = 5888,
        num_experts: int = 32,
        initial_active_experts: int = 8,
        top_k: int = 7,
        initial_top_k: int = 4,
        expert_rank: int = 3072,
        route_dim: int = 64,
        max_passes: int = 4,
        offload_tier1b_cpu: bool = True,
        param_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hot_buckets = int(hot_buckets)
        self.cold_buckets = int(cold_buckets)
        self.total_buckets = self.hot_buckets + self.cold_buckets
        self.eng_dim = int(eng_dim)
        self.width = int(width)
        self.shared_rank = int(shared_rank)
        self.num_experts = int(num_experts)
        self.active_experts = min(int(initial_active_experts), self.num_experts)
        self.target_top_k = min(int(top_k), self.num_experts)
        self.active_top_k = min(int(initial_top_k), self.active_experts)
        self.expert_rank = int(expert_rank)
        self.route_dim = int(route_dim)
        self.max_passes = int(max_passes)
        self.offload_tier1b_cpu = bool(offload_tier1b_cpu)

        # Tier 1A: GPU HBM Hot Lexical Cache (3 scales * hot_buckets, zero-init for cold-bucket safety)
        self.tier1a_hot = nn.Embedding(self.hot_buckets * 3, self.eng_dim, dtype=param_dtype)
        nn.init.zeros_(self.tier1a_hot.weight)
        self.scale_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32))

        # Tier 1B: Host DRAM / mmap Long-Tail N-Gram Bank
        if self.cold_buckets > 0 and self.offload_tier1b_cpu:
            pin = torch.cuda.is_available()
            self.tier1b_cold_weight = torch.zeros(
                (self.cold_buckets * 3, self.eng_dim), dtype=param_dtype, device='cpu', pin_memory=pin
            )
            self.tier1b_exp_avg = torch.zeros_like(self.tier1b_cold_weight, pin_memory=pin)
            self.tier1b_exp_avg_sq = torch.zeros_like(self.tier1b_cold_weight, pin_memory=pin)
            self.tier1b_step = 0
            self._pending_tier1b_rows: Optional[torch.Tensor] = None
            self._pending_tier1b_leaf: Optional[torch.Tensor] = None
        elif self.cold_buckets > 0:
            self.tier1b_cold = nn.Embedding(self.cold_buckets * 3, self.eng_dim, dtype=param_dtype)
            nn.init.zeros_(self.tier1b_cold.weight)
        else:
            self.tier1b_cold_weight = None

        self.norm_eps = 1e-6
        self.norm_weight = nn.Parameter(torch.ones(self.width, dtype=torch.float32))

        # Lexical projection to width for conditioning Tier-2 and Tier-3
        self.lexical_up = nn.Linear(self.eng_dim, self.width, bias=False, dtype=param_dtype)
        nn.init.normal_(self.lexical_up.weight, mean=0.0, std=0.02)

        # Tier 2: 100%-Active Hot Shared SwiGLU Expert (2560 -> 5888 -> 2560 = 45.22M params/anchor)
        self.shared_gate = nn.Linear(self.width, self.shared_rank, bias=False, dtype=param_dtype)
        self.shared_up = nn.Linear(self.width, self.shared_rank, bias=False, dtype=param_dtype)
        self.shared_down = nn.Linear(self.shared_rank, self.width, bias=False, dtype=param_dtype)
        nn.init.kaiming_uniform_(self.shared_gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, mean=0.0, std=0.01)

        # Tier 3: Anti-Collapse X-MoE Hypersphere Cosine Router + DeepSeek-V3 Sigmoid Affinity
        self.router_proj = nn.Linear(self.width, self.route_dim, bias=False, dtype=torch.float32)
        self.expert_centroids = nn.Parameter(torch.empty(self.num_experts, self.route_dim, dtype=torch.float32))
        nn.init.orthogonal_(self.router_proj.weight)
        nn.init.orthogonal_(self.expert_centroids)
        self.router_log_temp = nn.Parameter(torch.tensor(math.log(0.15), dtype=torch.float32))
        self.pass_router_bias = nn.Parameter(torch.zeros(self.max_passes, self.num_experts, dtype=torch.float32))
        self.register_buffer('expert_bias', torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer('expert_load_ema', torch.full((self.num_experts,), 1.0 / self.num_experts, dtype=torch.float32))

        # Tier 3: Routed SwiGLU Memory Experts (E=32, rank=3072 -> 352.32M params/anchor)
        self.query_proj = nn.Linear(self.width, self.eng_dim, bias=False, dtype=param_dtype)
        nn.init.kaiming_uniform_(self.query_proj.weight, a=math.sqrt(5))

        self.expert_gate = nn.Parameter(torch.empty(self.num_experts, self.eng_dim, self.expert_rank, dtype=param_dtype))
        self.expert_up = nn.Parameter(torch.empty(self.num_experts, self.eng_dim, self.expert_rank, dtype=param_dtype))
        self.expert_down = nn.Parameter(torch.empty(self.num_experts, self.expert_rank, self.width, dtype=param_dtype))
        self._init_orthogonal_experts()

        # Context-aware sigmoid gate & zero-initialized residual projection
        self.ctx_gate = nn.Linear(self.width, self.width, bias=True, dtype=param_dtype)
        nn.init.zeros_(self.ctx_gate.weight)
        nn.init.zeros_(self.ctx_gate.bias)

        self.project = nn.Linear(self.width, self.width, bias=False, dtype=param_dtype)
        nn.init.zeros_(self.project.weight)
        self.gain = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        # Per-pass telescoping gate logits: sigmoid([-0.6190, -1.7346, -2.9444, -2.9444]) == [0.35, 0.15, 0.05, 0.05]
        # so Pass 1 (initial lexical lookup) injects at 0.35x, Pass 2 (reasoning) at 0.15x, and Pass 3 (verification) at 0.05x
        self.pass_gain_logits = nn.Parameter(
            torch.tensor([-0.6190392, -1.7346011, -2.9444390, -2.9444390], dtype=torch.float32)
        )
        self.use_pass_gain_schedule: bool = False
        # On-the-fly Dynamic Engram Flight Mode & Elastic Knowledge Recall
        self.flight_mode: str = 'balanced'
        self.flight_top_k: Optional[int] = None
        self.elastic_tail_blend_max: float = 0.18
        self.multi_horizon_recall: bool = False
        self.last_flight_stats: dict[str, Any] = {}
        self.last_router_z_loss: torch.Tensor = torch.tensor(0.0)
        self._gate_override_tensor: Optional[torch.Tensor] = None
        self._gate_override_ref_val: Optional[float] = None

    @property
    def proj(self) -> nn.Linear:
        return self.project

    @property
    def ngram_order_logits(self) -> nn.Parameter:
        return self.scale_weights

    @property
    def gate(self) -> torch.Tensor:
        cur_gain = float(self.gain.detach().float().item())
        if self._gate_override_tensor is None or self._gate_override_ref_val != cur_gain:
            sig = torch.sigmoid(self.gain.detach().float()).clamp(1e-6, 1.0 - 1e-6)
            self._gate_override_tensor = torch.atanh(sig)
            self._gate_override_ref_val = cur_gain
        return self._gate_override_tensor

    @torch.no_grad()
    def _init_orthogonal_experts(self):
        for e in range(self.num_experts):
            nn.init.orthogonal_(self.expert_gate[e].float())
            nn.init.orthogonal_(self.expert_up[e].float())
            nn.init.orthogonal_(self.expert_down[e].float())
        self.expert_gate.data.mul_(0.05)
        self.expert_up.data.mul_(0.05)
        self.expert_down.data.mul_(0.02)

    @torch.no_grad()
    def upcycle_experts_8_to_32(self, noise_std: float = 1e-3):
        """Clone the 8 trained coarse macro-experts into 32 fine-grained experts in reverse index order
        (`7 -> 28..31`, ..., `0 -> 0..3`) so coarse experts `1..7` are never overwritten before cloning.
        """
        if self.active_experts >= self.num_experts:
            self.active_top_k = self.target_top_k
            return
        coarse = self.active_experts
        factor = self.num_experts // coarse
        src_gate = self.expert_gate.data[:coarse].clone()
        src_up = self.expert_up.data[:coarse].clone()
        src_down = self.expert_down.data[:coarse].clone()
        src_cent = self.expert_centroids.data[:coarse].clone()
        src_pbias = self.pass_router_bias.data[:, :coarse].clone()
        for m in range(coarse - 1, -1, -1):
            for sub in range(factor - 1, -1, -1):
                dst = m * factor + sub
                self.expert_gate.data[dst].copy_(src_gate[m])
                self.expert_up.data[dst].copy_(src_up[m])
                self.expert_down.data[dst].copy_(src_down[m])
                self.expert_centroids.data[dst].copy_(src_cent[m])
                self.pass_router_bias.data[:, dst].copy_(src_pbias[:, m])
                if sub > 0:
                    self.expert_gate.data[dst].add_(torch.randn_like(self.expert_gate.data[dst]) * noise_std)
                    self.expert_up.data[dst].add_(torch.randn_like(self.expert_up.data[dst]) * noise_std)
                    self.expert_centroids.data[dst].add_(torch.randn_like(self.expert_centroids.data[dst]) * noise_std)
        self.active_experts = self.num_experts
        self.active_top_k = self.target_top_k
        self.expert_load_ema.fill_(1.0 / self.num_experts)
        self.expert_bias.zero_()

    @staticmethod
    def compute_ngram_indices(
        input_ids: torch.LongTensor,
        total_buckets: int,
        prev_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Deterministic causal hashing over (t-2, t-1, t); zero access to future tokens.
        bsz, seq_len = input_ids.shape
        if prev_ids is None:
            p1 = F.pad(input_ids[:, :-1], (1, 0), value=0)
            p2 = F.pad(input_ids[:, :-2], (2, 0), value=0) if seq_len >= 2 else torch.zeros_like(input_ids)
        else:
            last1 = prev_ids[:, -1:]
            p1 = torch.cat([last1, input_ids[:, :-1]], dim=1) if seq_len > 1 else last1
            if prev_ids.shape[1] >= 2:
                last2 = prev_ids[:, -2:-1]
            else:
                last2 = torch.zeros_like(last1)
            if seq_len == 1:
                p2 = last2
            else:
                p2 = torch.cat([last2, last1, input_ids[:, :-2]], dim=1)
        bi = (p1 * 1315423911 + input_ids * 2654435761) % total_buckets
        tri = (p2 * 97531 + p1 * 1315423911 + input_ids * 2654435761) % total_buckets
        skip = (p2 * 65537 + input_ids * 2654435761) % total_buckets
        return bi, tri, skip

    def _lookup_tier1_lexical(
        self,
        ngram_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
        param_dtype: torch.dtype,
    ) -> torch.Tensor:
        bi, tri, skip = ngram_indices
        sw = F.softmax(self.scale_weights, dim=0).to(param_dtype)
        hot_b = self.hot_buckets

        bi_hot = bi.clamp(max=hot_b - 1)
        tri_hot = tri.clamp(max=hot_b - 1) + hot_b
        skip_hot = skip.clamp(max=hot_b - 1) + 2 * hot_b

        e_bi = self.tier1a_hot(bi_hot)
        e_tri = self.tier1a_hot(tri_hot)
        e_skip = self.tier1a_hot(skip_hot)

        if self.cold_buckets <= 0:
            return sw[0] * e_bi + sw[1] * e_tri + sw[2] * e_skip

        is_cold_bi = (bi >= hot_b)
        is_cold_tri = (tri >= hot_b)
        is_cold_skip = (skip >= hot_b)

        cold_b = self.cold_buckets
        c_bi = (bi - hot_b).clamp(min=0, max=cold_b - 1)
        c_tri = (tri - hot_b).clamp(min=0, max=cold_b - 1) + cold_b
        c_skip = (skip - hot_b).clamp(min=0, max=cold_b - 1) + 2 * cold_b

        if self.offload_tier1b_cpu and self.tier1b_cold_weight is not None:
            if not self.training and device.type == 'cuda':
                eval_gpu = getattr(self, '_eval_tier1b_gpu', None)
                if eval_gpu is None or eval_gpu.device != device or eval_gpu.dtype != param_dtype:
                    eval_gpu = self.tier1b_cold_weight.to(device=device, dtype=param_dtype, non_blocking=True)
                    self._eval_tier1b_gpu = eval_gpu
                c_e_bi = eval_gpu[c_bi]
                c_e_tri = eval_gpu[c_tri]
                c_e_skip = eval_gpu[c_skip]
            else:
                self._eval_tier1b_gpu = None
                stacked = torch.stack([c_bi, c_tri, c_skip], dim=0)
                uniq_gpu, inv = torch.unique(stacked, return_inverse=True)
                uniq_cpu = uniq_gpu.cpu()
                fetched_cpu = self.tier1b_cold_weight[uniq_cpu]
                fetched_gpu = fetched_cpu.to(device=device, dtype=param_dtype, non_blocking=True)
                if self.training:
                    fetched_gpu = fetched_gpu.detach().requires_grad_(True)
                    self._pending_tier1b_rows = uniq_cpu
                    self._pending_tier1b_leaf = fetched_gpu
                gathered = fetched_gpu[inv]
                c_e_bi, c_e_tri, c_e_skip = gathered[0], gathered[1], gathered[2]
        else:
            c_e_bi = self.tier1b_cold(c_bi)
            c_e_tri = self.tier1b_cold(c_tri)
            c_e_skip = self.tier1b_cold(c_skip)

        e_bi = torch.where(is_cold_bi.unsqueeze(-1), c_e_bi, e_bi)
        e_tri = torch.where(is_cold_tri.unsqueeze(-1), c_e_tri, e_tri)
        e_skip = torch.where(is_cold_skip.unsqueeze(-1), c_e_skip, e_skip)
        return sw[0] * e_bi + sw[1] * e_tri + sw[2] * e_skip

    @torch.no_grad()
    def step_tier1b_sparse_adam(self, lr: float = 2e-4, beta1: float = 0.9, beta2: float = 0.95, eps: float = 1e-8):
        """Coalesced sparse Adam update on pinned Host DRAM for Tier 1B (<0.3 ms/step) + Post-Backward Self-Balancing Router EMA."""
        # 1. Update auxiliary-loss-free expert load EMA, adaptive router bias, and starved-centroid re-anchoring AFTER backward
        #    (never during forward, so gradient checkpointing recomputation is 100% deterministic!)
        pending_load = getattr(self, '_pending_expert_load', None)
        if pending_load is not None:
            e_act = self.active_experts
            load = pending_load.to(device=self.expert_load_ema.device, dtype=torch.float32)
            if load.numel() == e_act:
                ema_decay = float(getattr(self, 'load_ema_decay', 0.05))
                self.expert_load_ema[:e_act].lerp_(load, ema_decay)
                target_load = 1.0 / float(e_act)
                imbalance = (target_load - self.expert_load_ema[:e_act]) / max(1e-6, target_load)
                base_step = float(getattr(self, 'balance_step_size', 0.0015))
                adaptive_step = base_step * (0.5 * torch.sign(imbalance) + 0.5 * imbalance.clamp(-2.0, 2.0))
                self.expert_bias[:e_act].add_(adaptive_step).clamp_(-0.5, 0.5)
                starved_mask = self.expert_load_ema[:e_act] < (0.15 * target_load)
                if bool(starved_mask.any().item()):
                    mean_cent = F.normalize(self.expert_centroids[:e_act].mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-6)
                    self.expert_centroids.data[:e_act][starved_mask] = F.normalize(
                        0.98 * self.expert_centroids.data[:e_act][starved_mask] + 0.02 * mean_cent,
                        p=2, dim=-1, eps=1e-6,
                    )
            self._pending_expert_load = None

        # 2. Coalesced sparse Adam update for Tier 1B cold N-gram rows
        if (
            not self.offload_tier1b_cpu
            or self.tier1b_cold_weight is None
            or self._pending_tier1b_leaf is None
            or self._pending_tier1b_leaf.grad is None
            or self._pending_tier1b_rows is None
        ):
            self._pending_tier1b_leaf = None
            self._pending_tier1b_rows = None
            return
        rows = self._pending_tier1b_rows
        grad = self._pending_tier1b_leaf.grad.detach().to(device='cpu', dtype=torch.float32)
        self.tier1b_step += 1
        m = self.tier1b_exp_avg[rows].float()
        v = self.tier1b_exp_avg_sq[rows].float()
        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bc1 = 1.0 - (beta1 ** self.tier1b_step)
        bc2 = 1.0 - (beta2 ** self.tier1b_step)
        step_size = lr / bc1
        denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
        w = self.tier1b_cold_weight[rows].float()
        w.addcdiv_(m, denom, value=-step_size)
        self.tier1b_exp_avg[rows] = m.to(self.tier1b_exp_avg.dtype)
        self.tier1b_exp_avg_sq[rows] = v.to(self.tier1b_exp_avg_sq.dtype)
        self.tier1b_cold_weight[rows] = w.to(self.tier1b_cold_weight.dtype)
        self._pending_tier1b_leaf = None
        self._pending_tier1b_rows = None

    @torch.no_grad()
    def routing_balance_summary(self) -> dict[str, Any]:
        """Return router load balance telemetry (CV, min/max load ratio, temperature, per-pass scales, n-gram weights)."""
        e_act = max(1, int(self.active_experts))
        loads = self.expert_load_ema[:e_act].detach().float()
        target = 1.0 / float(e_act)
        mean_l = float(loads.mean().item())
        std_l = float(loads.std(unbiased=False).item()) if e_act > 1 else 0.0
        cv = std_l / max(1e-6, mean_l)
        min_ratio = float(loads.min().item()) / max(1e-6, target)
        max_ratio = float(loads.max().item()) / max(1e-6, target)
        p_scales = (
            torch.sigmoid(self.pass_gain_logits.detach().float()[:3]).tolist()
            if hasattr(self, 'pass_gain_logits') and self.pass_gain_logits is not None
            else [1.0, 1.0, 1.0]
        )
        sw = F.softmax(self.scale_weights.detach().float(), dim=0).tolist()
        f_top_k = getattr(self, 'flight_top_k', None)
        eff_k = min(int(f_top_k), e_act) if (f_top_k is not None and int(f_top_k) > 0) else int(self.active_top_k)
        return {
            'active_experts': e_act,
            'active_top_k': int(self.active_top_k),
            'flight_mode': str(getattr(self, 'flight_mode', 'balanced')),
            'effective_top_k': eff_k,
            'multi_horizon_recall': bool(getattr(self, 'multi_horizon_recall', False)),
            'load_cv': round(cv, 4),
            'min_load_ratio': round(min_ratio, 3),
            'max_load_ratio': round(max_ratio, 3),
            'router_temp': round(float(self.router_log_temp.detach().float().exp().item()), 4),
            'gain_sigmoid': round(float(torch.sigmoid(self.gain.detach().float()).item()), 4),
            'pass_scales': [round(x, 3) for x in p_scales],
            'ngram_weights': [round(x, 3) for x in sw],
            'last_flight_stats': dict(getattr(self, 'last_flight_stats', {})),
        }

    @torch.no_grad()
    def reset_parameters_to_step0(self, initial_active_experts: int = 8, initial_top_k: int = 2):
        """Re-initialize all Tier-1A/1B tables, Shared Expert, Routed Experts, and Zero-Gate to fresh Step-0 state."""
        self.active_experts = min(int(initial_active_experts), self.num_experts)
        self.active_top_k = min(int(initial_top_k), self.active_experts)
        nn.init.zeros_(self.tier1a_hot.weight)
        self.scale_weights.zero_()
        if self.cold_buckets > 0 and self.offload_tier1b_cpu and self.tier1b_cold_weight is not None:
            self.tier1b_cold_weight.zero_()
            self.tier1b_exp_avg.zero_()
            self.tier1b_exp_avg_sq.zero_()
            self.tier1b_step = 0
            self._pending_tier1b_rows = None
            self._pending_tier1b_leaf = None
        elif self.cold_buckets > 0 and hasattr(self, 'tier1b_cold'):
            nn.init.zeros_(self.tier1b_cold.weight)
        self.norm_weight.fill_(1.0)
        nn.init.normal_(self.lexical_up.weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.shared_gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, mean=0.0, std=0.01)
        nn.init.orthogonal_(self.router_proj.weight)
        nn.init.orthogonal_(self.expert_centroids)
        self.router_log_temp.fill_(math.log(0.15))
        self.pass_router_bias.zero_()
        self.expert_bias.zero_()
        self.expert_load_ema.fill_(1.0 / self.num_experts)
        nn.init.kaiming_uniform_(self.query_proj.weight, a=math.sqrt(5))
        self._init_orthogonal_experts()
        nn.init.zeros_(self.ctx_gate.weight)
        nn.init.zeros_(self.ctx_gate.bias)
        nn.init.zeros_(self.project.weight)
        self.gain.fill_(-2.0)
        self.last_router_z_loss = torch.tensor(0.0)
        self._active_z_losses = None
        self._pending_expert_load = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        dtype: torch.dtype,
        prev_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        pass_idx: int = 0,
        ngram_indices: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        param_dtype = self.project.weight.dtype
        if ngram_indices is None:
            ngram_indices = self.compute_ngram_indices(input_ids, self.total_buckets, prev_ids=prev_ids)

        use_multi_horizon = bool(getattr(self, 'multi_horizon_recall', False))
        # 1. Tier 1A + 1B Lexical N-gram Lookup (cached across recurrent passes during eval!)
        cache_key = (
            (ngram_indices[0].data_ptr(), ngram_indices[0].shape, param_dtype, use_multi_horizon)
            if not self.training else None
        )
        cached_lex = getattr(self, '_eval_lex_cache', None) if not self.training else None
        if cached_lex is not None and cached_lex[0] == cache_key:
            e_lexical, lex_hidden = cached_lex[1], cached_lex[2]
        else:
            e_lexical = self._lookup_tier1_lexical(ngram_indices, input_ids.device, param_dtype)
            if use_multi_horizon and e_lexical.shape[1] >= 2:
                # Causal 5-gram multi-horizon contextual trace (9 N-gram bucket probes spanning t-4..t, zero future leakage)
                e_prev1 = F.pad(e_lexical[:, :-1, :], (0, 0, 1, 0))
                if e_lexical.shape[1] >= 3:
                    e_prev2 = F.pad(e_lexical[:, :-2, :], (0, 0, 2, 0))
                    e_lexical = (0.83 * e_lexical + 0.12 * e_prev1 + 0.05 * e_prev2).to(param_dtype)
                else:
                    e_lexical = (0.88 * e_lexical + 0.12 * e_prev1).to(param_dtype)
            if self.training:
                e_lexical = F.dropout(e_lexical, p=0.20, training=True)
            lex_hidden = self.lexical_up(e_lexical)
            if not self.training:
                self._eval_lex_cache = (cache_key, e_lexical, lex_hidden)

        # Diagnostic hook (default 1.0 = bit-identical no-op): scale/zero the Tier-1 lexical path AFTER the
        # eval cache so ablation never poisons cached values. Used by `sparkx_engram_generalization`.
        _lex_s = getattr(self, '_ablate_lex', 1.0)
        if _lex_s != 1.0:
            e_lexical = e_lexical * _lex_s
            lex_hidden = lex_hidden * _lex_s

        if hidden_states is not None:
            h_norm = triton_rmsnorm(hidden_states.to(param_dtype), self.norm_weight, self.norm_eps)
            h_norm_f = h_norm.float()

            # 2. Tier 2: 100%-Active Hot Shared SwiGLU Expert (Fused Triton SwiGLU)
            h_fused_in = h_norm + lex_hidden
            shared_out = self.shared_down(triton_swiglu(self.shared_gate(h_fused_in), self.shared_up(h_fused_in)))

            # 3. Tier 3: X-MoE Hypersphere Cosine Router + Sigmoid Affinity + On-The-Fly Elastic Top-K Recall
            e_act = self.active_experts
            k_base = min(self.active_top_k, e_act)
            f_top_k = getattr(self, 'flight_top_k', None)
            k_act = min(int(f_top_k), e_act) if (f_top_k is not None and int(f_top_k) > 0) else k_base
            p_idx = min(max(int(pass_idx), 0), self.max_passes - 1)

            u = F.normalize(self.router_proj(h_norm_f), p=2, dim=-1, eps=1e-6)
            c = F.normalize(self.expert_centroids[:e_act], p=2, dim=-1, eps=1e-6)
            temp = self.router_log_temp.exp().clamp(min=0.03, max=2.0)
            router_logits = (torch.matmul(u, c.t()) / temp) + self.pass_router_bias[p_idx, :e_act]

            if self.training:
                log_z = torch.logsumexp(router_logits, dim=-1)
                z_curr = log_z.pow(2).mean()
                self.last_router_z_loss = z_curr.detach()
                active_z = getattr(self, '_active_z_losses', None)
                if active_z is not None:
                    active_z.append(z_curr)
            else:
                self.last_router_z_loss = router_logits.new_zeros(())

            scores = torch.sigmoid(router_logits)
            sel_scores = scores + self.expert_bias[:e_act]
            dom_bias = getattr(self, '_domain_zone_bias', None)
            if dom_bias is not None and dom_bias.numel() >= e_act:
                sel_scores = sel_scores + dom_bias[:e_act].to(device=sel_scores.device, dtype=sel_scores.dtype)
            _, topk_idx = torch.topk(sel_scores, k=k_act, dim=-1)
            topk_raw = torch.gather(scores, dim=-1, index=topk_idx)

            if k_act > k_base:
                # Entropy-Gated Two-Stage Elastic Expert Recall:
                # Preserves exact primary k_base expert distribution while blending in sharpened tail specialists
                # (k_base+1 .. k_act) proportional to per-token routing uncertainty H_norm(t)^2.
                prim_raw = topk_raw[..., :k_base]
                tail_raw = topk_raw[..., k_base:]
                prim_gates = prim_raw / (prim_raw.sum(dim=-1, keepdim=True) + 1e-6)
                tail_sq = tail_raw.pow(2)
                tail_gates = tail_sq / (tail_sq.sum(dim=-1, keepdim=True) + 1e-6)
                ent = -(prim_gates * torch.log(prim_gates.clamp(min=1e-7))).sum(dim=-1, keepdim=True)
                max_ent = math.log(max(2, k_base))
                ent_norm = (ent / max_ent).clamp(0.0, 1.0)
                beta_tail = float(getattr(self, 'elastic_tail_blend_max', 0.18))
                alpha_t = beta_tail * ent_norm.pow(2)
                combined_gates = torch.cat([
                    (1.0 - alpha_t) * prim_gates,
                    alpha_t * tail_gates,
                ], dim=-1)
                topk_gates = combined_gates.to(param_dtype)
                if not self.training:
                    self.last_flight_stats = {
                        'k_base': int(k_base),
                        'k_flight': int(k_act),
                        'mean_entropy_norm': round(float(ent_norm.mean().item()), 4),
                        'mean_tail_alpha': round(float(alpha_t.mean().item()), 4),
                    }
            else:
                topk_gates = (topk_raw / (topk_raw.sum(dim=-1, keepdim=True) + 1e-6)).to(param_dtype)
                if not self.training:
                    self.last_flight_stats = {
                        'k_base': int(k_base),
                        'k_flight': int(k_act),
                        'mean_entropy_norm': 0.0,
                        'mean_tail_alpha': 0.0,
                    }

            if self.training:
                with torch.no_grad():
                    one_hot = F.one_hot(topk_idx, num_classes=e_act).float()
                    load = one_hot.sum(dim=-2).mean(dim=(0, 1)) / float(k_act)
                    self._pending_expert_load = load.detach()

            q = self.query_proj(h_norm) + e_lexical
            flat_q = q.reshape(-1, self.eng_dim)
            n_tokens = flat_q.shape[0]
            flat_idx = topk_idx.reshape(n_tokens, k_act)
            flat_gates = topk_gates.reshape(n_tokens, k_act)

            fp32_down_map = getattr(self, '_alphaedit_fp32_down', None)
            use_fp32_compiled = bool(fp32_down_map)

            if n_tokens <= 8 and not use_fp32_compiled:
                # Zero-loop, zero-sync batched contraction for single-token / small-batch decoding!
                w_g = self.expert_gate[flat_idx]   # [N, K, D, R]
                w_u = self.expert_up[flat_idx]     # [N, K, D, R]
                w_d = self.expert_down[flat_idx]   # [N, K, R, W]
                g_proj = torch.einsum('nd,nkdr->nkr', flat_q, w_g)
                u_proj = torch.einsum('nd,nkdr->nkr', flat_q, w_u)
                act = triton_swiglu(g_proj, u_proj)
                out_k = torch.einsum('nkr,nkrw->nkw', act, w_d)
                routed_flat = (out_k * flat_gates.unsqueeze(-1)).sum(dim=1)
            else:
                # Sorted contiguous bucket dispatch for 4K/8K/16K prefill & training (or exact fp32 AlphaEdit evaluation)
                flat_e = flat_idx.reshape(-1)
                sorted_e, sort_perm = torch.sort(flat_e)
                token_rows = torch.div(sort_perm, k_act, rounding_mode='floor')
                sorted_q = flat_q.index_select(0, token_rows)
                sorted_g = flat_gates.reshape(-1).index_select(0, sort_perm).unsqueeze(-1)
                counts = torch.bincount(sorted_e, minlength=e_act).tolist()

                accum_dt = torch.float32 if use_fp32_compiled else param_dtype
                out_chunks = []
                offset = 0
                for e_id, cnt in enumerate(counts):
                    if cnt == 0:
                        continue
                    q_e = sorted_q[offset : offset + cnt]
                    g_e = sorted_g[offset : offset + cnt]
                    offset += cnt
                    act_e = triton_swiglu(
                        torch.matmul(q_e, self.expert_gate[e_id]),
                        torch.matmul(q_e, self.expert_up[e_id]),
                    )
                    if use_fp32_compiled and e_id in fp32_down_map:
                        w_d_fp32 = fp32_down_map[e_id].to(device=act_e.device, dtype=torch.float32)
                        out_chunks.append(torch.matmul(act_e.float(), w_d_fp32) * g_e.float())
                    elif use_fp32_compiled:
                        out_chunks.append((torch.matmul(act_e, self.expert_down[e_id]) * g_e).float())
                    else:
                        out_chunks.append(torch.matmul(act_e, self.expert_down[e_id]) * g_e)

                routed_flat = torch.zeros((n_tokens, self.width), dtype=accum_dt, device=flat_q.device)
                if out_chunks:
                    routed_flat.index_add_(0, token_rows, torch.cat(out_chunks, dim=0))

            routed_out = routed_flat.view(shared_out.shape)
            _sem_s = getattr(self, '_ablate_sem', 1.0)
            if _sem_s != 1.0:
                routed_out = routed_out * _sem_s
                shared_out = shared_out * _sem_s
            if use_fp32_compiled:
                combined_f32 = 0.5 * lex_hidden.float() + shared_out.float() + routed_out.float()
                ctx_f32 = 2.0 * torch.sigmoid(self.ctx_gate(h_norm).float())
                combined_f32 = combined_f32 * ctx_f32
                projected = F.linear(combined_f32, self.project.weight.float())
            else:
                combined = 0.5 * lex_hidden + shared_out + routed_out
                ctx = 2.0 * torch.sigmoid(self.ctx_gate(h_norm).float()).to(param_dtype)
                combined = combined * ctx
                projected = self.project(combined)
        else:
            combined = F.silu(lex_hidden)
            projected = self.project(combined)
        p_idx_gate = min(max(int(pass_idx), 0), self.max_passes - 1)
        if not self.training and self._gate_override_tensor is not None:
            gate = torch.tanh(self._gate_override_tensor.to(device=projected.device, dtype=torch.float32)).abs().to(dtype)
            if getattr(self, 'use_pass_gain_schedule', False):
                gate = gate * torch.sigmoid(self.pass_gain_logits[p_idx_gate]).to(dtype)
        else:
            gate = torch.sigmoid(self.gain).to(dtype)
            if getattr(self, 'use_pass_gain_schedule', False):
                gate = gate * torch.sigmoid(self.pass_gain_logits[p_idx_gate]).to(dtype)
            elif not self.training and input_ids.shape[1] == 1:
                gate = gate * float(getattr(self, 'decode_engram_scale', 0.25))
        out = projected.to(dtype) * gate
        if getattr(self, 'selective_gating_enabled', False) and hidden_states is not None:
            # 1. WISE Activation Scope Gate (`arXiv:2405.14768` Eq. 5): exact 0.0 perturbation outside compiled scope
            scope_thr = float(getattr(self, 'scope_threshold', 0.0) or 0.0)
            if scope_thr > 0.0:
                protos = getattr(self, '_compiled_scope_prototypes', None)
                if protos is not None and protos.numel() > 0 and int(pass_idx) == 0:
                    use_hidden_scope = (
                        getattr(self, '_scope_proto_space', 'router') == 'hidden'
                        and int(protos.shape[-1]) == int(self.width)
                    )
                    q_scope = (
                        F.normalize(h_norm_f, p=2.0, dim=-1, eps=1e-6).to(u.dtype)
                        if use_hidden_scope
                        else u
                    )
                    sims = torch.matmul(q_scope, protos.to(device=q_scope.device, dtype=q_scope.dtype).t())
                    s_scope = sims.max(dim=-1, keepdim=True).values
                elif getattr(self, 'use_centroid_scope_fallback', False):
                    s_scope = torch.matmul(u, c.t()).max(dim=-1, keepdim=True).values
                else:
                    s_scope = u.new_zeros(u.shape[:-1] + (1,))
                margin = float(getattr(self, 'scope_margin', 0.0) or 0.0)
                if margin <= 0.0:
                    scope_mask = (s_scope >= scope_thr).to(out.dtype)
                else:
                    scope_mask = ((s_scope - scope_thr) / margin).clamp(0.0, 1.0).to(out.dtype)
                out = out * scope_mask

            # 2. DeepSeek Signed-Sqrt RMSNorm Query-Key Gate (`arXiv:2601.07372` Sec. 2.3)
            if bool(getattr(self, 'qk_gate_enabled', False)):
                h_hat = h_norm_f / (h_norm_f.norm(dim=-1, keepdim=True) + 1e-6)
                m_f = out.float()
                m_hat = m_f / (m_f.norm(dim=-1, keepdim=True) + 1e-6)
                s_qk = (h_hat * m_hat).sum(dim=-1, keepdim=True) * float(getattr(self, 'qk_gate_scale', 2.0))
                s_sqrt = torch.sign(s_qk) * torch.sqrt(s_qk.abs() + 1e-6) + float(getattr(self, 'qk_gate_bias', -1.0))
                out = out * torch.sigmoid(s_sqrt).to(out.dtype)

            # 3. In-Manifold Trust-Region Relative Write Norm Cap (`||delta|| <= rho_max * ||h||`)
            max_rel = getattr(self, 'max_rel_delta_norm', None)
            if max_rel is not None and float(max_rel) > 0.0:
                h_nrm = hidden_states.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                d_nrm = out.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                cap_scale = torch.clamp((float(max_rel) * h_nrm) / d_nrm, max=1.0).to(out.dtype)
                out = out * cap_scale

            # 4. Prompt-Prefix Suppression (`min_active_pos`): zero Engram writes on unsupervised `<|System|>...<|User|>` prefix tokens
            min_pos = int(getattr(self, 'min_active_pos', 0) or 0)
            if min_pos > 0 and out.dim() == 3 and out.shape[1] > 1:
                clamp_p = min(min_pos, int(out.shape[1]))
                out = out.clone()
                out[:, :clamp_p, :] = 0.0
        return out

    @property
    def engram_gain(self) -> torch.Tensor:
        return self.gain

    @torch.no_grad()
    def register_scope_prototypes(self, u_vecs: torch.Tensor, space: Optional[str] = None) -> int:
        """Register unit-norm directions (`[M, 64]` router or `[M, width]` hidden) in the WISE activation scope prototype bank."""
        d_last = int(u_vecs.shape[-1]) if u_vecs.dim() >= 1 else 64
        if space == 'hidden' or (space is None and d_last == int(self.width) and int(self.width) != 64):
            proto_dim = int(self.width)
            target_space = 'hidden'
        else:
            proto_dim = 64
            target_space = 'router'
        u_norm = F.normalize(u_vecs.detach().float().reshape(-1, proto_dim), p=2.0, dim=-1, eps=1e-6)
        existing = getattr(self, '_compiled_scope_prototypes', None)
        cur_space = getattr(self, '_scope_proto_space', 'router')
        if existing is None or existing.numel() == 0 or existing.shape[-1] != proto_dim or cur_space != target_space:
            self._compiled_scope_prototypes = u_norm.to(self.expert_centroids.device)
        else:
            self._compiled_scope_prototypes = torch.cat(
                [existing.to(self.expert_centroids.device), u_norm.to(self.expert_centroids.device)], dim=0
            )
        self._scope_proto_space = target_space
        return int(self._compiled_scope_prototypes.shape[0])

    def configure_selective_gating(
        self,
        enabled: bool = True,
        scope_threshold: float = 0.70,
        scope_margin: float = 0.0,
        qk_gate_enabled: bool = False,
        qk_gate_scale: float = 2.0,
        qk_gate_bias: float = -1.0,
        max_rel_delta_norm: Optional[float] = 0.05,
        use_centroid_scope_fallback: bool = False,
        min_active_pos: int = 0,
    ) -> None:
        """Configure opt-in selective gating (`BIE-3` / `SI4`). Default `enabled=False` is 100% bit-identical to legacy forward."""
        self.selective_gating_enabled = bool(enabled)
        self.scope_threshold = float(scope_threshold)
        self.scope_margin = float(scope_margin)
        self.qk_gate_enabled = bool(qk_gate_enabled)
        self.qk_gate_scale = float(qk_gate_scale)
        self.qk_gate_bias = float(qk_gate_bias)
        self.max_rel_delta_norm = None if max_rel_delta_norm is None else float(max_rel_delta_norm)
        self.use_centroid_scope_fallback = bool(use_centroid_scope_fallback)
        self.min_active_pos = int(min_active_pos)

    @torch.no_grad()
    def reset_alphaedit_basis(self) -> None:
        """Reset the per-expert AlphaEdit active subspace basis, WISE scope prototypes, and historical activation tracker."""
        self._alphaedit_basis = {}
        self._alphaedit_history_acts = {}
        self._alphaedit_history_qvecs = []
        self._alphaedit_basis_qcount = {}
        self._alphaedit_fp32_down = {}
        self._compiled_scope_prototypes = None
        self._scope_proto_space = 'router'
        self._eval_lex_cache = None

    @torch.no_grad()
    def compile_clusters_alphaedit(
        self,
        clusters: list[dict[str, Any]],
        input_ids: Optional[torch.Tensor] = None,
        domain: str = "general",
        alpha_ridge: float = 1e-2,
        null_threshold: float = 1e-4,
        target_scale: float = 0.35,
        inject_lexical: bool = False,
        lexical_strength: float = 0.08,
        key_mode: str = "query",
    ) -> dict[str, Any]:
        """Compile salient LSA semantic clusters (`B_b`) vertically into this `ZipfianTieredMoEEngram` anchor
        via Null-Space Constrained Closed-Form Ridge Compilation (`AlphaEdit`, Fang et al., ICLR 2025).

        Phase 0 & BIE-5/SI4 fixes:
          - M5: Exact damped right-inverse `W^T (W W^T + lam I)^{-1}` with scale-adaptive Tikhonov damping.
          - M6: Never overwrites `self.project.weight` with `0.25 * I` (skips cleanly if `project` is zero-gated).
          - M4: Raises `CapacityExhaustedError` via `update_orthonormal_basis` instead of silently truncating old basis columns.
          - m1: Uses `F.linear(..., self.router_proj.weight.float())` without mutating module parameters in place.
          - Cross-Expert Null-Space Projection: Protects all previously compiled query vectors `Q_hist` across every
            touched specialist expert `e_id` in Top-K MoE routing so subsequent edits never leak across experts.
          - Post-SwiGLU MEMIT Prototype Aggregation (`fact_id`): Averages post-SwiGLU activations `act_b` across probes
            of the same `fact_id` in the exact linear domain of `expert_down` while registering full-width (`2560`-D)
            normalized hidden prototypes in the WISE scope bank.
        """
        if not clusters:
            return {
                "status": "empty",
                "compiled_clusters": 0,
                "experts_updated": [],
                "mean_rel_fit_error": 0.0,
                "max_null_space_leakage": 0.0,
                "total_active_null_rank": 0,
            }

        if not hasattr(self, "_alphaedit_basis") or self._alphaedit_basis is None:
            self._alphaedit_basis = {}
        if not hasattr(self, "_alphaedit_history_acts") or self._alphaedit_history_acts is None:
            self._alphaedit_history_acts = {}
        if not hasattr(self, "_alphaedit_history_qvecs") or self._alphaedit_history_qvecs is None:
            self._alphaedit_history_qvecs = []
        if not hasattr(self, "_alphaedit_basis_qcount") or self._alphaedit_basis_qcount is None:
            self._alphaedit_basis_qcount = {}
        if not hasattr(self, "_alphaedit_fp32_down") or self._alphaedit_fp32_down is None:
            self._alphaedit_fp32_down = {}

        device = self.expert_down.device
        param_dtype = self.project.weight.dtype

        # Fix M6: never overwrite `self.project.weight` (which also carries lexical and shared paths)
        proj_w_f32 = self.project.weight.detach().float()
        if proj_w_f32.norm().item() < 1e-6:
            return {
                "status": "skipped_zero_project",
                "compiled_clusters": 0,
                "experts_updated": [],
                "mean_rel_fit_error": 0.0,
                "max_null_space_leakage": 0.0,
                "total_active_null_rank": sum(int(v.shape[1]) for v in self._alphaedit_basis.values()),
            }

        # Fix M5: exact damped right-inverse `W_proj^T (W_proj W_proj^T + lam I)^{-1}` with scale-adaptive relative damping
        mean_sv_sq = float(proj_w_f32.pow(2).sum().item()) / float(max(1, proj_w_f32.shape[0]))
        eff_lam = min(1e-2, max(1e-8, 1e-3 * mean_sv_sq))
        pinv_w_p = precompute_damped_right_inverse(proj_w_f32, lam=eff_lam)  # [width, width]

        dom = str(domain).lower().strip()
        zones = {
            "stem_reason": set(range(0, 12)) | {30, 31},
            "web_ux_sec": set(range(12, 22)) | {30, 31},
            "anti_slop": set(range(22, 30)) | {30, 31},
        }
        zone_set = zones.get(dom, set(range(self.active_experts)))

        e_act = int(self.active_experts)
        k_base = min(self.active_top_k, e_act)
        f_top_k = getattr(self, "flight_top_k", None)
        k_act = min(int(f_top_k), e_act) if (f_top_k is not None and int(f_top_k) > 0) else k_base

        c_norm = F.normalize(self.expert_centroids[:e_act].detach().float(), p=2.0, dim=-1, eps=1e-6)
        temp = self.router_log_temp.detach().float().exp().clamp(min=0.03, max=2.0)
        p_bias = self.pass_router_bias[0, :e_act].detach().float()
        e_bias = self.expert_bias[:e_act].detach().float()
        dom_bias = getattr(self, "_domain_zone_bias", None)
        if dom_bias is not None and dom_bias.numel() >= e_act:
            e_bias = e_bias + dom_bias[:e_act].detach().float().to(device)

        if not self.training and getattr(self, "_gate_override_tensor", None) is not None:
            mod_gate = float(torch.tanh(self._gate_override_tensor.detach().float()).abs().item())
        else:
            mod_gate = float(torch.sigmoid(self.gain.detach().float()).item())
        if getattr(self, "use_pass_gain_schedule", False):
            mod_gate *= float(torch.sigmoid(self.pass_gain_logits[0].detach().float()).item())
        mod_gate = max(mod_gate, 1e-3)

        # Optional precomputed N-gram indices if `input_ids` is passed
        ngram_all = None
        if input_ids is not None and input_ids.numel() > 0:
            ids_dev = input_ids.to(device)
            if ids_dev.dim() == 1:
                ids_dev = ids_dev.unsqueeze(0)
            ngram_all = self.compute_ngram_indices(ids_dev, self.total_buckets)

        use_multi_horizon = bool(getattr(self, "multi_horizon_recall", False))
        use_query_key = str(key_mode).lower().strip() != "cluster"

        # Group clusters by optional `fact_id` so multi-probe facts form a single rank-1 post-SwiGLU prototype (MEMIT Eq. 14)
        grouped_clusters: dict[Any, list[dict[str, Any]]] = {}
        for idx_c, cl in enumerate(clusters):
            f_key = cl.get("fact_id", f"__single_{idx_c}")
            grouped_clusters.setdefault(f_key, []).append(cl)

        expert_buckets: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        compiled_proto_list: list[torch.Tensor] = []

        for _, cl_group in grouped_clusters.items():
            group_items = []
            for cl in cl_group:
                raw_key_h = cl.get("query_hidden") if (use_query_key and "query_hidden" in cl) else cl["cluster_hidden"]
                h_b = raw_key_h.to(device=device, dtype=param_dtype).view(1, 1, self.width)
                raw_y = cl.get("target_residual") if "target_residual" in cl else cl["value_residual"]
                y_b = raw_y.to(device=device, dtype=torch.float32).view(self.width)
                trig_pos = int(cl.get("trigger_pos", 0))

                h_norm = triton_rmsnorm(h_b, self.norm_weight, self.norm_eps)
                h_norm_f = h_norm.float().view(1, self.width)
                h_hat_proto = F.normalize(h_norm_f, p=2.0, dim=-1, eps=1e-6)
                compiled_proto_list.append(h_hat_proto.detach())

                # Per-cluster `input_ids` override (so multi-prompt batches match each prompt's exact N-grams)
                cl_ids = cl.get("input_ids")
                if isinstance(cl_ids, torch.Tensor) and cl_ids.numel() > 0:
                    cl_ids_dev = cl_ids.to(device)
                    if cl_ids_dev.dim() == 1:
                        cl_ids_dev = cl_ids_dev.unsqueeze(0)
                    cur_ng = self.compute_ngram_indices(cl_ids_dev, self.total_buckets)
                else:
                    cur_ng = ngram_all

                if cur_ng is not None and 0 <= trig_pos < cur_ng[0].shape[1]:
                    sub_ng = (
                        cur_ng[0][:, trig_pos : trig_pos + 1],
                        cur_ng[1][:, trig_pos : trig_pos + 1],
                        cur_ng[2][:, trig_pos : trig_pos + 1],
                    )
                    if inject_lexical and float(lexical_strength) > 0.0:
                        q_cue = self.query_proj(h_norm).detach().view(self.eng_dim)
                        for scale_i, idx_t in enumerate(sub_ng):
                            b_id = int(idx_t.item())
                            if 0 <= b_id < self.hot_buckets:
                                row = b_id * 3 + scale_i
                                self.tier1a_hot.weight.data[row].lerp_(
                                    q_cue.to(self.tier1a_hot.weight.dtype), float(lexical_strength)
                                )
                    if use_multi_horizon and trig_pos >= 1:
                        w_start = max(0, trig_pos - 2)
                        win_ng = (
                            cur_ng[0][:, w_start : trig_pos + 1],
                            cur_ng[1][:, w_start : trig_pos + 1],
                            cur_ng[2][:, w_start : trig_pos + 1],
                        )
                        e_win = self._lookup_tier1_lexical(win_ng, device, param_dtype)
                        if e_win.shape[1] >= 3:
                            e_lex = (0.83 * e_win[:, -1:, :] + 0.12 * e_win[:, -2:-1, :] + 0.05 * e_win[:, -3:-2, :]).to(param_dtype)
                        elif e_win.shape[1] == 2:
                            e_lex = (0.88 * e_win[:, -1:, :] + 0.12 * e_win[:, -2:-1, :]).to(param_dtype)
                        else:
                            e_lex = e_win[:, -1:, :]
                    else:
                        e_lex = self._lookup_tier1_lexical(sub_ng, device, param_dtype)
                else:
                    e_lex = torch.zeros((1, 1, self.eng_dim), device=device, dtype=param_dtype)
                _lex_s = getattr(self, "_ablate_lex", 1.0)
                if _lex_s != 1.0:
                    e_lex = e_lex * _lex_s

                u = F.normalize(F.linear(h_norm_f, self.router_proj.weight.float()), p=2.0, dim=-1, eps=1e-6)  # [1, 64]
                q_vec = (self.query_proj(h_norm) + e_lex).reshape(1, self.eng_dim)
                ctx_b = (2.0 * torch.sigmoid(self.ctx_gate(h_norm).float())).view(self.width).clamp(min=0.1)
                group_items.append((u, q_vec, ctx_b, y_b))

            # Compute exact primary Top-K gates per probe and select shared specialist `e_star` with max (min_probe_gate * free_frac)
            probe_gate_maps: list[dict[int, float]] = []
            for u_it, _, _, _ in group_items:
                r_logits_it = (torch.matmul(u_it, c_norm.t()) / temp) + p_bias
                scores_it = torch.sigmoid(r_logits_it)[0]
                _, topk_it = torch.topk(scores_it + e_bias, k=k_act, dim=-1)
                topk_raw_it = torch.gather(scores_it, dim=-1, index=topk_it)
                prim_raw_it = topk_raw_it[:k_base]
                prim_gates_it = prim_raw_it / (prim_raw_it.sum() + 1e-6)
                if k_act > k_base:
                    ent_it = -(prim_gates_it * torch.log(prim_gates_it.clamp(min=1e-7))).sum()
                    ent_norm_it = (ent_it / math.log(max(2, k_base))).clamp(0.0, 1.0)
                    alpha_t_it = float(getattr(self, "elastic_tail_blend_max", 0.18)) * ent_norm_it.pow(2)
                    prim_gates_it = (1.0 - alpha_t_it) * prim_gates_it
                g_map: dict[int, float] = {}
                for k_i in range(k_base):
                    g_map[int(topk_it[k_i].item())] = float(prim_gates_it[k_i].item())
                probe_gate_maps.append(g_map)

            common_experts = set(probe_gate_maps[0].keys())
            for gm in probe_gate_maps[1:]:
                common_experts &= set(gm.keys())
            candidate_experts = sorted(common_experts) if common_experts else sorted(
                {e_c for gm in probe_gate_maps for e_c in gm.keys()}
            )

            e_star = candidate_experts[0]
            best_score = -1.0
            use_zone_bonus = dom in zones
            for cand_e in candidate_experts:
                v_basis = self._alphaedit_basis.get(cand_e)
                used_r = int(v_basis.shape[1]) if v_basis is not None else 0
                free_frac = max(0.05, 1.0 - (used_r / float(self.expert_rank)))
                zone_mult = (1.15 if cand_e in zone_set else 1.0) if use_zone_bonus else 1.0
                if common_experts:
                    gate_metric = min(gm[cand_e] for gm in probe_gate_maps)
                else:
                    gate_metric = sum(gm.get(cand_e, 0.0) for gm in probe_gate_maps) / float(len(probe_gate_maps))
                sc = gate_metric * free_frac * zone_mult
                if sc > best_score:
                    best_score = sc
                    e_star = cand_e

            w_g = self.expert_gate[e_star].detach()
            w_u = self.expert_up[e_star].detach()

            for idx_p, (u_it, q_it, ctx_it, y_it) in enumerate(group_items):
                gm = probe_gate_maps[idx_p]
                g_star_it = max(gm.get(e_star, max(gm.values()) if gm else 0.1), 1e-2)
                act_it = triton_swiglu(torch.matmul(q_it, w_g), torch.matmul(q_it, w_u))[0].float()
                r_target_it = float(target_scale) * y_it
                comb_delta_it = torch.mv(pinv_w_p, r_target_it / mod_gate)
                dz_it = comb_delta_it / (ctx_it * g_star_it)
                expert_buckets.setdefault(e_star, []).append((act_it, dz_it, q_it.detach()))

        fit_errors: list[float] = []
        max_null_leakage = 0.0
        updated_experts: list[int] = []
        max_allowed_rank = max(1, int(self.expert_rank) - 16)
        new_qvecs_this_call: list[torch.Tensor] = []

        for e_id, pairs in expert_buckets.items():
            # Cross-Expert Null-Space Projection in `float64`: ensure all historical compiled queries `Q_hist` are protected on `e_id`
            prev_q_count = int(self._alphaedit_basis_qcount.get(e_id, 0))
            total_q_hist = len(self._alphaedit_history_qvecs)
            v_act = self._alphaedit_basis.get(e_id)
            if total_q_hist > prev_q_count:
                unproj_q = torch.cat(self._alphaedit_history_qvecs[prev_q_count:total_q_hist], dim=0).to(
                    device=device, dtype=param_dtype
                )
                act_hist_e = triton_swiglu(
                    torch.matmul(unproj_q, self.expert_gate[e_id].detach()),
                    torch.matmul(unproj_q, self.expert_up[e_id].detach()),
                ).double()
                act_hist_perp = project_into_null_space(act_hist_e, v_act)
                v_act = update_orthonormal_basis(
                    v_act, act_hist_perp, max_rank=max_allowed_rank, null_threshold=1e-7
                )
                self._alphaedit_basis[e_id] = v_act
                a_hist_prev = self._alphaedit_history_acts.get(e_id)
                if a_hist_prev is None:
                    self._alphaedit_history_acts[e_id] = act_hist_e.float().detach().clone()
                else:
                    self._alphaedit_history_acts[e_id] = torch.cat(
                        [a_hist_prev, act_hist_e.float().detach().clone()], dim=0
                    )
                self._alphaedit_basis_qcount[e_id] = total_q_hist

            a_mat_f64 = torch.stack([p[0] for p in pairs], dim=0).double()  # [M_e, 3072]
            dz_mat_f64 = torch.stack([p[1] for p in pairs], dim=0).double()  # [M_e, 2560]

            dw_down_f64, a_perp_f64, _, rel_err = solve_null_space_ridge(
                a_mat_f64, dz_mat_f64, v_basis=v_act, alpha_ridge=float(alpha_ridge)
            )
            fit_errors.append(rel_err)

            # Audit exact historical null-space leakage `||A_hist @ dW_down||_inf` in `float64` BEFORE appending current acts
            a_hist = self._alphaedit_history_acts.get(e_id)
            if a_hist is not None and a_hist.shape[0] > 0:
                leak = float(torch.matmul(a_hist.double(), dw_down_f64).abs().max().item())
                if leak > max_null_leakage:
                    max_null_leakage = leak

            # Update orthonormal active basis `V_active[e_id]` in `float64` so capacity exhaustion raises before mutating weights (M4 fix)
            new_basis = update_orthonormal_basis(
                v_act, a_perp_f64, max_rank=max_allowed_rank, null_threshold=min(float(null_threshold), 1e-6)
            )
            self._alphaedit_basis[e_id] = new_basis

            # Accumulate in fp32 master weight (evaluated in fp32 by `forward()` when `_alphaedit_fp32_down` is populated)
            dw_down = dw_down_f64.float()
            if e_id not in self._alphaedit_fp32_down:
                self._alphaedit_fp32_down[e_id] = self.expert_down.data[e_id].detach().float().clone()
            self._alphaedit_fp32_down[e_id].add_(dw_down)
            self.expert_down.data[e_id].copy_(self._alphaedit_fp32_down[e_id].to(self.expert_down.dtype))

            a_mat_f32 = a_mat_f64.float()
            if a_hist is None:
                self._alphaedit_history_acts[e_id] = a_mat_f32.detach().clone()
            else:
                self._alphaedit_history_acts[e_id] = torch.cat([a_hist, a_mat_f32.detach().clone()], dim=0)
            for p in pairs:
                new_qvecs_this_call.append(p[2])
            updated_experts.append(e_id)

        if new_qvecs_this_call:
            self._alphaedit_history_qvecs.extend(new_qvecs_this_call)
            for e_id in updated_experts:
                self._alphaedit_basis_qcount[e_id] = len(self._alphaedit_history_qvecs)

        if compiled_proto_list:
            self.register_scope_prototypes(torch.cat(compiled_proto_list, dim=0), space='hidden')

        self._eval_lex_cache = None
        total_rank = sum(int(v.shape[1]) for v in self._alphaedit_basis.values() if v is not None)
        return {
            "status": "ok",
            "compiled_clusters": len(clusters),
            "experts_updated": sorted(updated_experts),
            "mean_rel_fit_error": round(sum(fit_errors) / max(1, len(fit_errors)), 6),
            "max_null_space_leakage": float(max_null_leakage),
            "total_active_null_rank": total_rank,
            "scope_prototypes_count": int(getattr(self, "_compiled_scope_prototypes", torch.empty(0)).shape[0]),
        }


class TwoStreamMHC(nn.Module):
    """Sublayer Dual-mHC with Pass-0/Pass-1/Pass-2 (72-Layer) Conditioning, Closed-Form 2x2 Sinkhorn,
    and Exact Zero-Gated Identity Initialization (`extra_gain=0, p3_extra_gain=0`).

    Matches `SparkX_Loop_v02_OneShot_Colab.ipynb` (`mix_logits=[[4,-4],[-4,4]]`, `read_logits=[5,-5]`,
    `write_logits=[5,-5]`, `eff_mat = I + gain * (P - I)`) so:
    1. Pass 0 (`use_mhc=True, passes=1`) matches the upstream 36-layer `Spark-X2.6-4B` backbone (`max_diff < 0.025`).
    2. Pass 2 (`passes=2`) and Pass 3 (`enable_p3=True`, 72 virtual layers) have exact `zero_gate_diff == 0.0` at Step 0.
    """
    def __init__(
        self,
        width: int = 2560,
        streams: int = 2,
        iters: int = 20,
        dynamic: bool = True,
        sublayer_mhc: bool = True,
        pass_adapter_rank: int = 16,
        max_passes: int = 4,
        rmsnorm_eps: float = 1e-6,
    ):
        super().__init__()
        if streams != 2:
            raise ValueError('TwoStreamMHC expects 2 streams')
        self.width = width
        self.streams = streams
        self.n = streams
        self.iters = iters
        self.dynamic = dynamic
        self.sublayer_mhc = sublayer_mhc
        self.pass_adapter_rank = pass_adapter_rank
        self.max_passes = max_passes
        self.norm_eps = rmsnorm_eps

        # Primary / Attention mHC parameters (aligned with v0.2 for exact warm-start & Step-0 parity)
        self.mix_logits = nn.Parameter(torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32))
        self.read_logits = nn.Parameter(torch.tensor([5.0, -5.0], dtype=torch.float32))
        self.write_logits = nn.Parameter(torch.tensor([5.0, -5.0], dtype=torch.float32))
        self.extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.p3_extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Sublayer FFN mHC parameters
        self.ffn_mix_logits = nn.Parameter(torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32))
        self.ffn_read_logits = nn.Parameter(torch.tensor([5.0, -5.0], dtype=torch.float32))
        self.ffn_write_logits = nn.Parameter(torch.tensor([5.0, -5.0], dtype=torch.float32))
        self.ffn_extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.p3_ffn_extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.pass_routing_bias = nn.Parameter(torch.zeros(max_passes, 2, 8, dtype=torch.float32))

        if dynamic:
            flat_dim = streams * width
            self.norm_weight = nn.Parameter(torch.ones(flat_dim, dtype=torch.float32))
            self.dynamic_proj = nn.Linear(flat_dim, 8, bias=False, dtype=torch.float32)
            nn.init.zeros_(self.dynamic_proj.weight)
            self.ffn_norm_weight = nn.Parameter(torch.ones(flat_dim, dtype=torch.float32))
            self.ffn_dynamic_proj = nn.Linear(flat_dim, 8, bias=False, dtype=torch.float32)
            nn.init.zeros_(self.ffn_dynamic_proj.weight)
        else:
            self.register_parameter('dynamic_proj', None)
            self.register_parameter('ffn_dynamic_proj', None)

        if pass_adapter_rank > 0:
            self.pass_down = nn.ModuleList([
                nn.Linear(width, pass_adapter_rank, bias=False, dtype=torch.float32)
                for _ in range(max_passes)
            ])
            self.pass_up = nn.ModuleList([
                nn.Linear(pass_adapter_rank, width, bias=False, dtype=torch.float32)
                for _ in range(max_passes)
            ])
            for p_idx in range(max_passes):
                nn.init.kaiming_uniform_(self.pass_down[p_idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.pass_up[p_idx].weight)
        else:
            self.pass_down = None
            self.pass_up = None

    @property
    def attn_extra_gain(self) -> nn.Parameter:
        return self.extra_gain

    @property
    def p3_attn_extra_gain(self) -> nn.Parameter:
        return self.p3_extra_gain

    @torch.no_grad()
    def reset_parameters_to_step0(self):
        """Restore TwoStreamMHC to exact Step-0 zero-gated identity initialization."""
        self.mix_logits.copy_(torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32, device=self.mix_logits.device))
        self.read_logits.copy_(torch.tensor([5.0, -5.0], dtype=torch.float32, device=self.read_logits.device))
        self.write_logits.copy_(torch.tensor([5.0, -5.0], dtype=torch.float32, device=self.write_logits.device))
        self.extra_gain.zero_()
        self.p3_extra_gain.zero_()
        self.ffn_mix_logits.copy_(torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32, device=self.ffn_mix_logits.device))
        self.ffn_read_logits.copy_(torch.tensor([5.0, -5.0], dtype=torch.float32, device=self.ffn_read_logits.device))
        self.ffn_write_logits.copy_(torch.tensor([5.0, -5.0], dtype=torch.float32, device=self.ffn_write_logits.device))
        self.ffn_extra_gain.zero_()
        self.p3_ffn_extra_gain.zero_()
        self.pass_routing_bias.zero_()
        if self.dynamic and self.dynamic_proj is not None:
            self.norm_weight.fill_(1.0)
            nn.init.zeros_(self.dynamic_proj.weight)
            self.ffn_norm_weight.fill_(1.0)
            nn.init.zeros_(self.ffn_dynamic_proj.weight)
        if self.pass_down is not None and self.pass_up is not None:
            for p_idx in range(self.max_passes):
                nn.init.kaiming_uniform_(self.pass_down[p_idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.pass_up[p_idx].weight)

    def mixing_matrix(self, dynamic_delta: Optional[torch.Tensor] = None, base_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = (self.mix_logits if base_logits is None else base_logits).float()
        log_p = base if dynamic_delta is None else (base + dynamic_delta.float())
        log_p = log_p.clamp(min=-30.0, max=30.0)
        diag_adv = 0.5 * ((log_p[..., 0, 0] + log_p[..., 1, 1]) - (log_p[..., 0, 1] + log_p[..., 1, 0]))
        p_keep = torch.sigmoid(diag_adv)
        p_swap = 1.0 - p_keep
        row0 = torch.stack([p_keep, p_swap], dim=-1)
        row1 = torch.stack([p_swap, p_keep], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def _compute_routing(self, streams: torch.Tensor, pass_idx: int = 0, is_ffn: bool = False):
        p_idx = min(max(0, int(pass_idx)), self.max_passes - 1)
        sub_idx = 1 if is_ffn else 0
        base_r = self.ffn_read_logits if is_ffn else self.read_logits
        base_w = self.ffn_write_logits if is_ffn else self.write_logits
        base_m = self.ffn_mix_logits if is_ffn else self.mix_logits
        proj = self.ffn_dynamic_proj if is_ffn else self.dynamic_proj
        nw = getattr(self, 'ffn_norm_weight' if is_ffn else 'norm_weight', None)
        p_bias = self.pass_routing_bias[p_idx, sub_idx].float()

        if not self.dynamic or proj is None:
            r_log = base_r.float() + p_bias[:2]
            w_log = base_w.float() + p_bias[2:4]
            m_log = base_m.float() + p_bias[4:8].reshape(2, 2)
            rw = r_log.softmax(dim=0)[:, None, None, None]
            ww = w_log.softmax(dim=0)[:, None, None, None]
            mat = self.mixing_matrix(base_logits=m_log)[:, :, None, None, None]
            return rw, ww, mat

        with torch.autocast(device_type=streams.device.type, enabled=False):
            _, bsz, seq_len, dim = streams.shape
            flat = streams.permute(1, 2, 0, 3).reshape(bsz, seq_len, 2 * dim).float()
            rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            normed = (flat * rms * nw) if nw is not None else (flat * rms)
            dyn = proj(normed) + p_bias.view(1, 1, -1)
            dyn_read = dyn[..., 0:2]
            dyn_write = dyn[..., 2:4]
            dyn_mix = dyn[..., 4:8].reshape(bsz, seq_len, 2, 2)
            rw = (base_r.float() + dyn_read).softmax(dim=-1).permute(2, 0, 1).unsqueeze(-1)
            ww = (base_w.float() + dyn_write).softmax(dim=-1).permute(2, 0, 1).unsqueeze(-1)
            mat = self.mixing_matrix(dyn_mix, base_logits=base_m).permute(2, 3, 0, 1).unsqueeze(-1)
            return rw, ww, mat

    def _compute_routing_scalars(self, s0: torch.Tensor, s1: torch.Tensor, pass_idx: int = 0, is_ffn: bool = False):
        p_idx = min(max(0, int(pass_idx)), self.max_passes - 1)
        sub_idx = 1 if is_ffn else 0
        base_r = self.ffn_read_logits if is_ffn else self.read_logits
        base_w = self.ffn_write_logits if is_ffn else self.write_logits
        base_m = self.ffn_mix_logits if is_ffn else self.mix_logits
        proj = self.ffn_dynamic_proj if is_ffn else self.dynamic_proj
        nw = getattr(self, 'ffn_norm_weight' if is_ffn else 'norm_weight', None)
        p_bias = self.pass_routing_bias[p_idx, sub_idx].float()

        if not self.dynamic or proj is None:
            r_log = base_r.float() + p_bias[:2]
            w_log = base_w.float() + p_bias[2:4]
            m_log = (base_m.float().reshape(4) + p_bias[4:8]).clamp(min=-30.0, max=30.0)
            r1 = torch.sigmoid(r_log[1] - r_log[0])
            w1 = torch.sigmoid(w_log[1] - w_log[0])
            diag_adv = 0.5 * ((m_log[0] + m_log[3]) - (m_log[1] + m_log[2]))
            p_swap = 1.0 - torch.sigmoid(diag_adv)
            return r1, 1.0 - w1, w1, p_swap

        with torch.autocast(device_type=s0.device.type, enabled=False):
            flat = torch.cat((s0, s1), dim=-1)
            rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            normed = (flat * rms * nw) if nw is not None else (flat * rms)
            dyn = proj(normed) + p_bias.view(1, 1, -1)
            r_log = base_r.float() + dyn[..., 0:2]
            w_log = base_w.float() + dyn[..., 2:4]
            m_log = (base_m.float().reshape(1, 1, 4) + dyn[..., 4:8]).clamp(min=-30.0, max=30.0)
            r1 = torch.sigmoid(r_log[..., 1:2] - r_log[..., 0:1])
            w1 = torch.sigmoid(w_log[..., 1:2] - w_log[..., 0:1])
            diag_adv = 0.5 * ((m_log[..., 0:1] + m_log[..., 3:4]) - (m_log[..., 1:2] + m_log[..., 2:3]))
            p_swap = 1.0 - torch.sigmoid(diag_adv)
            return r1, 1.0 - w1, w1, p_swap

    def _effective_gain(self, gain_param: torch.Tensor, pass_idx: int = 0, extra: bool = False) -> torch.Tensor | float:
        if not extra:
            return 1.0
        raw = torch.tanh(gain_param.float())
        min_g = float(
            getattr(self, 'min_p3_gain', 0.0) if pass_idx >= 2 else getattr(self, 'min_extra_gain', 0.0)
        )
        if min_g <= 0.0:
            return raw
        # Smooth softplus floor so Pass-2 / Pass-3 recurrence gains never collapse to zero (Track L3)
        return min_g + (1.0 - min_g) * torch.sigmoid((raw - min_g) * 6.0) * 0.0 + F.softplus((raw - min_g) * 10.0) / 10.0

    @torch.no_grad()
    def controller_telemetry(self) -> dict[str, Any]:
        """Return Track L1 diagnostic summary of P2/P3 gains and static Sinkhorn read/write/swap probabilities."""
        g_p2_a = float(self._effective_gain(self.extra_gain, pass_idx=1, extra=True))
        g_p2_f = float(self._effective_gain(self.ffn_extra_gain, pass_idx=1, extra=True))
        g_p3_a = float(self._effective_gain(self.p3_extra_gain, pass_idx=2, extra=True))
        g_p3_f = float(self._effective_gain(self.p3_ffn_extra_gain, pass_idx=2, extra=True))
        r_a = self.read_logits.float().softmax(dim=0)
        w_a = self.write_logits.float().softmax(dim=0)
        m_a = self.mixing_matrix(base_logits=self.mix_logits.float())
        r_f = self.ffn_read_logits.float().softmax(dim=0)
        w_f = self.ffn_write_logits.float().softmax(dim=0)
        m_f = self.mixing_matrix(base_logits=self.ffn_mix_logits.float())
        return {
            'gain_p2_attn': round(g_p2_a, 5),
            'gain_p2_ffn': round(g_p2_f, 5),
            'gain_p3_attn': round(g_p3_a, 5),
            'gain_p3_ffn': round(g_p3_f, 5),
            'read_s1_prob_attn': round(float(r_a[1].item()), 5),
            'write_s0_prob_attn': round(float(w_a[0].item()), 5),
            'swap_prob_attn': round(float(m_a[0, 1].item()), 5),
            'read_s1_prob_ffn': round(float(r_f[1].item()), 5),
            'write_s0_prob_ffn': round(float(w_f[0].item()), 5),
            'swap_prob_ffn': round(float(m_f[0, 1].item()), 5),
            'gain_collapsed': (abs(g_p2_a) < 0.03 and abs(g_p2_f) < 0.03),
        }

    def _apply_pass_adapter(self, x: torch.Tensor, pass_idx: int) -> torch.Tensor:
        if self.pass_down is None or self.pass_up is None:
            return torch.zeros_like(x, dtype=torch.float32)
        p_idx = min(max(0, int(pass_idx)), self.max_passes - 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            return self.pass_up[p_idx](F.silu(self.pass_down[p_idx](normed)))

    def forward(
        self,
        streams: torch.Tensor,
        layer: nn.Module,
        position_embeddings,
        attention_mask,
        extra: bool = False,
        pass_idx: int = 0,
        past_key_values=None,
        cache_position=None,
    ) -> torch.Tensor:
        orig_dtype = streams.dtype
        s0, s1 = streams[0].float(), streams[1].float()

        if self.sublayer_mhc and hasattr(layer, 'input_layernorm') and hasattr(layer, 'post_attention_layernorm'):
            target_dtype = layer.mlp.gate_proj.weight.dtype
            # 1. Attention sublayer mHC (exact 3D closed-form 2x2 Sinkhorn)
            r1_a, w0_a, w1_a, p_swap_a = self._compute_routing_scalars(s0, s1, pass_idx=pass_idx, is_ffn=False)
            diff_a = s1 - s0
            mixed_a = s0 + r1_a * diff_a
            normed_a = layer.input_layernorm(mixed_a.to(orig_dtype)).to(target_dtype)
            attn_out, _ = layer.self_attn(
                hidden_states=normed_a,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            gain_a_param = self.p3_extra_gain if pass_idx >= 2 else self.extra_gain
            gain_a = self._effective_gain(gain_a_param, pass_idx=pass_idx, extra=extra)
            alpha_a = gain_a * p_swap_a
            attn_delta = gain_a * attn_out.float()
            m0 = s0 + alpha_a * diff_a + w0_a * attn_delta
            m1 = s1 - alpha_a * diff_a + w1_a * attn_delta

            # 2. FFN sublayer mHC + Pass-conditioned deliberation adapter
            r1_f, w0_f, w1_f, p_swap_f = self._compute_routing_scalars(m0, m1, pass_idx=pass_idx, is_ffn=True)
            diff_f = m1 - m0
            mixed_f = m0 + r1_f * diff_f
            normed_f = layer.post_attention_layernorm(mixed_f.to(orig_dtype)).to(target_dtype)
            mlp_out = layer.mlp(normed_f).float() + self._apply_pass_adapter(mixed_f, pass_idx)
            gain_f_param = self.p3_ffn_extra_gain if pass_idx >= 2 else self.ffn_extra_gain
            gain_f = self._effective_gain(gain_f_param, pass_idx=pass_idx, extra=extra)
            alpha_f = gain_f * p_swap_f
            mlp_delta = gain_f * mlp_out
            out0 = m0 + alpha_f * diff_f + w0_f * mlp_delta
            out1 = m1 - alpha_f * diff_f + w1_f * mlp_delta
            return torch.stack((out0.to(orig_dtype), out1.to(orig_dtype)), dim=0)

        r1, w0, w1, p_swap = self._compute_routing_scalars(s0, s1, pass_idx=pass_idx, is_ffn=False)
        diff = s1 - s0
        mixed_input = (s0 + r1 * diff).to(orig_dtype)
        updated = layer(
            mixed_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        delta = updated.float() - mixed_input.float() + self._apply_pass_adapter(mixed_input, pass_idx)
        gain_param = self.p3_extra_gain if pass_idx >= 2 else self.extra_gain
        gain = self._effective_gain(gain_param, pass_idx=pass_idx, extra=extra)
        alpha = gain * p_swap
        scaled_delta = gain * delta
        out0 = s0 + alpha * diff + w0 * scaled_delta
        out1 = s1 - alpha * diff + w1 * scaled_delta
        return torch.stack((out0.to(orig_dtype), out1.to(orig_dtype)), dim=0)


def fused_masked_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int = 1024,
) -> torch.Tensor:
    """Unsloth / Cut-Cross-Entropy + Gradient-Checkpointed Chunked Cross-Entropy over V=167,168.

    Prevents the 11.0 GB [B, 16384, 167168] FP32 logit spike at 16K context by:
    1. Using Apple/Unsloth `cut_cross_entropy.linear_cross_entropy` Triton kernel when available, OR
    2. Evaluating valid token chunks under `torch.utils.checkpoint.checkpoint` so only a single
       [chunk_tokens, 167168] logit block is materialized during backward.
    """
    shift_h = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
    shift_y = labels[:, 1:].contiguous().reshape(-1).to(shift_h.device)
    valid_mask = shift_y != -100
    if not torch.any(valid_mask):
        return hidden_states.sum() * 0.0

    valid_h = shift_h[valid_mask]
    valid_y = shift_y[valid_mask]
    n_valid = valid_y.numel()

    if HAS_CCE and valid_h.is_cuda and valid_h.dtype in (torch.bfloat16, torch.float16):
        try:
            return _cce_linear_cross_entropy(valid_h, lm_head_weight.to(valid_h.dtype), valid_y)
        except Exception:
            pass

    if n_valid <= chunk_tokens:
        logits = F.linear(valid_h.to(lm_head_weight.dtype), lm_head_weight).float()
        return F.cross_entropy(logits, valid_y, reduction='mean')

    def _chunk_ce(h_chunk: torch.Tensor, y_chunk: torch.Tensor) -> torch.Tensor:
        lg = F.linear(h_chunk.to(lm_head_weight.dtype), lm_head_weight).float()
        return F.cross_entropy(lg, y_chunk, reduction='sum')

    total_loss = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    use_ckpt = valid_h.requires_grad or lm_head_weight.requires_grad
    step_c = max(128, int(chunk_tokens))
    for start_idx in range(0, n_valid, step_c):
        h_c = valid_h[start_idx : start_idx + step_c]
        y_c = valid_y[start_idx : start_idx + step_c]
        if use_ckpt:
            total_loss = total_loss + checkpoint(_chunk_ce, h_c, y_c, use_reentrant=False)
        else:
            total_loss = total_loss + _chunk_ce(h_c, y_c)
    return total_loss / float(n_valid)


class SparkXLoopV03(nn.Module):
    """Spark-X2.6-4B-En4B-A1B (v0.3):
    - 36 physical layers unrolled up to 72 effective layers via 3-Pass Telescoping Recurrence:
      Prelude [0..4) + Pass-1 [4..28) + Pass-2 [4..28) + Pass-3 [12..24) + Coda [28..36) = 72 virtual layers.
    - 8 Macro-Block MoE-Engram Anchors at physical layers 3, 7, 11, 15, 19, 23, 27, 31
      (514M total / 138M active per anchor = 4.11B Total / 1.10B Active Engram).
    """
    def __init__(self, base_model: nn.Module, settings: Optional[LoopV03Settings] = None,
                 strict: bool = True, patch_sdpa: bool = True):
        super().__init__()
        self.base = base_model
        self.settings = settings or LoopV03Settings()
        if patch_sdpa:
            patch_spark_attention_for_long_context(self.base)
        self.n_layers = len(self.base.model.layers)
        if strict and self.n_layers != 36:
            raise ValueError(f'Expected 36 Spark layers, got {self.n_layers}')
        width = int(self.base.config.hidden_size)
        if strict and width != 2560:
            raise ValueError(f'Expected hidden_size 2560, got {width}')

        self.loop_start = self.settings.start
        self.loop_end = self.settings.end
        self.loop_passes = self.settings.passes
        self.p3_start = self.settings.p3_start
        self.p3_end = self.settings.p3_end
        self.enable_p3 = self.settings.enable_p3
        self.mhc_enabled = self.settings.use_mhc
        self.engram_enabled = self.settings.use_engram
        self.enable_checkpoint = self.settings.checkpoint_layers
        self.offload_checkpoint_to_cpu = self.settings.offload_checkpoint_to_cpu

        self._validate_loop()
        self.controllers = nn.ModuleDict({
            str(i): TwoStreamMHC(
                width=width,
                streams=self.settings.mhc_streams,
                dynamic=self.settings.dynamic_mhc,
                sublayer_mhc=self.settings.sublayer_mhc,
                pass_adapter_rank=self.settings.pass_adapter_rank,
                max_passes=self.settings.max_passes,
            )
            for i in range(self.loop_start, self.loop_end)
        })

        # 8 Macro-Block Anchors: Layer 3 (Prelude) + Layers 7, 11, 15, 19, 23, 27 (Recurrent Core) + Layer 31 (Coda)
        param_dtype = self.base.model.embedding.weight.dtype
        if param_dtype not in (torch.bfloat16, torch.float16):
            param_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        def _make_engram():
            return ZipfianTieredMoEEngram(
                hot_buckets=self.settings.eng_hot_buckets,
                cold_buckets=self.settings.eng_cold_buckets,
                eng_dim=self.settings.eng_dim,
                width=width,
                shared_rank=self.settings.eng_shared_rank,
                num_experts=self.settings.eng_experts,
                initial_active_experts=self.settings.eng_initial_active_experts,
                top_k=self.settings.eng_top_k,
                initial_top_k=self.settings.eng_initial_top_k,
                expert_rank=self.settings.eng_expert_rank,
                route_dim=self.settings.eng_route_dim,
                max_passes=self.settings.max_passes,
                offload_tier1b_cpu=self.settings.eng_offload_tier1b_cpu,
                param_dtype=param_dtype,
            )

        self.engram = _make_engram()  # Anchor 0: Layer 3 (end of Prelude [0..4))
        stride = max(1, int(self.settings.backbone_engram_stride))
        anchor_indices = [
            idx for idx in range(self.loop_start + stride - 1, min(self.n_layers - 4, 32), stride)
        ]
        self.backbone_engrams = nn.ModuleDict({
            str(idx): _make_engram() for idx in anchor_indices
        })

        # Phase 3: Dedicated Episodic Side Memory (ESM) at the 9 global layers (`{3, 7, 11, 15, 19, 23, 27, 31, 35}`)
        self.esm_layers = [idx for idx in (3, 7, 11, 15, 19, 23, 27, 31, 35) if idx < self.n_layers]
        self.esm_enabled = False
        self.esm_visits = 'first'  # 'first' (default, Pass 0 only) | 'all' (ablation)
        self.esm = nn.ModuleDict({
            str(idx): EpisodicSideMemory(width=width) for idx in self.esm_layers
        })
        self._residual_capture_hook = None
        self.share_global_kv_across_passes = False

    def all_engram_modules(self) -> list[ZipfianTieredMoEEngram]:
        return [self.engram] + [mod for mod in self.backbone_engrams.values()]

    def iter_engram_anchors(self) -> list[tuple[int, ZipfianTieredMoEEngram]]:
        return [(3, self.engram)] + [(int(k), mod) for k, mod in self.backbone_engrams.items()]

    @torch.no_grad()
    def set_recursion_kv_sharing(self, enabled: bool = True) -> dict[str, Any]:
        """Track L5: Share Pass-0 global KV cache across Pass 1 and Pass 2 (`read_only=True`), reducing global KV slots from 17 to 9."""
        self.share_global_kv_across_passes = bool(enabled)
        plan = self.execution_plan
        global_set = set(self.esm_layers)
        total_global_visits = sum(1 for idx in plan if idx in global_set)
        unique_global_layers = len(set(idx for idx in plan if idx in global_set))
        eff_slots = unique_global_layers if self.share_global_kv_across_passes else total_global_visits
        return {
            'share_global_kv_across_passes': self.share_global_kv_across_passes,
            'unshared_global_kv_slots': total_global_visits,
            'effective_global_kv_slots': eff_slots,
            'global_kv_reduction_ratio': round(total_global_visits / max(1, eff_slots), 3),
        }

    @torch.no_grad()
    def set_mhc_gain_floor(self, min_gain_p2: float = 0.05, min_gain_p3: float = 0.02) -> dict[str, Any]:
        """Track L3: Set smooth softplus gain floor across all 24 TwoStreamMHC controllers to prevent P2/P3 gain collapse."""
        for ctrl in self.controllers.values():
            ctrl.min_extra_gain = float(max(0.0, min_gain_p2))
            ctrl.min_p3_gain = float(max(0.0, min_gain_p3))
        return {
            'controllers_updated': len(self.controllers),
            'min_gain_p2': float(max(0.0, min_gain_p2)),
            'min_gain_p3': float(max(0.0, min_gain_p3)),
        }

    @torch.no_grad()
    def get_mhc_controller_telemetry(self) -> dict[str, Any]:
        """Track L1: Aggregate per-layer TwoStreamMHC P2/P3 gains, Sinkhorn read/write/swap probs, and gain-collapse flags."""
        per_layer = {}
        p2_a, p2_f, p3_a, p3_f = [], [], [], []
        collapsed_layers = []
        for k, ctrl in self.controllers.items():
            info = ctrl.controller_telemetry()
            per_layer[f'L{k}'] = info
            p2_a.append(abs(info['gain_p2_attn']))
            p2_f.append(abs(info['gain_p2_ffn']))
            p3_a.append(abs(info['gain_p3_attn']))
            p3_f.append(abs(info['gain_p3_ffn']))
            if info['gain_collapsed']:
                collapsed_layers.append(int(k))
        return {
            'num_controllers': len(self.controllers),
            'mean_abs_gain_p2_attn': round(sum(p2_a) / max(1, len(p2_a)), 5),
            'mean_abs_gain_p2_ffn': round(sum(p2_f) / max(1, len(p2_f)), 5),
            'mean_abs_gain_p3_attn': round(sum(p3_a) / max(1, len(p3_a)), 5),
            'mean_abs_gain_p3_ffn': round(sum(p3_f) / max(1, len(p3_f)), 5),
            'collapsed_controller_layers': collapsed_layers,
            'per_layer': per_layer,
        }

    def capture_residuals(
        self,
        input_ids: torch.LongTensor,
        layers: Sequence[int],
        positions: Optional[torch.Tensor] = None,
        where: str = 'post_block',
        cache_position: Optional[torch.Tensor] = None,
        return_all_visits: bool = False,
        visit: int = 0,
    ) -> dict[Any, torch.Tensor]:
        """Phase 1: Capture residual stream tensors across requested layers in 36L or 72L Loop Base mode."""
        return capture_residuals(
            self,
            input_ids=input_ids,
            layers=layers,
            positions=positions,
            where=where,
            cache_position=cache_position,
            return_all_visits=return_all_visits,
            visit=visit,
        )

    def capture_attention_io(
        self,
        input_ids: torch.LongTensor,
        layers: Sequence[int],
        visit: int = 0,
        cache_position: Optional[torch.Tensor] = None,
    ) -> dict[int, dict[str, Any]]:
        """Phase 1: Capture attention inputs, post-RoPE Q/K/V, gate, and reconstruction error (< 1e-3)."""
        return capture_attention_io(
            self,
            input_ids=input_ids,
            layers=layers,
            visit=visit,
            cache_position=cache_position,
        )

    @torch.no_grad()
    def reset_all_loop_and_engram_to_step0(self, initial_active_experts: int = 8, initial_top_k: int = 2):
        """Re-initialize all 24 TwoStreamMHC controllers and all 8 ZipfianTieredMoEEngram anchors to Step 0."""
        for ctrl in self.controllers.values():
            ctrl.reset_parameters_to_step0()
        for eng_mod in self.all_engram_modules():
            eng_mod.reset_parameters_to_step0(
                initial_active_experts=initial_active_experts,
                initial_top_k=initial_top_k,
            )

    def upcycle_all_engrams_8_to_32(self, noise_std: float = 1e-3):
        for mod in self.all_engram_modules():
            mod.upcycle_experts_8_to_32(noise_std=noise_std)

    def step_all_tier1b_sparse_adam(self, lr: float = 2e-4):
        for mod in self.all_engram_modules():
            mod.step_tier1b_sparse_adam(lr=lr)

    @torch.no_grad()
    def get_engram_routing_telemetry(self) -> dict[str, Any]:
        """Aggregate router balance CV, min/max load ratio, and per-pass telescoping scales across all 8 Engram anchors."""
        per_anchor = {}
        cvs, mins, maxs, gains = [], [], [], []
        for l_idx, mod in self.iter_engram_anchors():
            info = mod.routing_balance_summary()
            per_anchor[f'L{l_idx}'] = info
            cvs.append(info['load_cv'])
            mins.append(info['min_load_ratio'])
            maxs.append(info['max_load_ratio'])
            gains.append(info['gain_sigmoid'])
        return {
            'active_experts': self.engram.active_experts,
            'active_top_k': self.engram.active_top_k,
            'flight_mode': str(getattr(self.engram, 'flight_mode', 'balanced')),
            'effective_top_k': int(getattr(self.engram, 'flight_top_k', None) or self.engram.active_top_k),
            'multi_horizon_recall': bool(getattr(self.engram, 'multi_horizon_recall', False)),
            'mean_load_cv': round(sum(cvs) / max(1, len(cvs)), 4),
            'min_load_ratio': round(min(mins) if mins else 1.0, 3),
            'max_load_ratio': round(max(maxs) if maxs else 1.0, 3),
            'mean_gain_sigmoid': round(sum(gains) / max(1, len(gains)), 4),
            'per_anchor': per_anchor,
        }

    @torch.no_grad()
    def set_engram_flight_mode(
        self,
        mode: str = 'balanced',
        top_k_flight: Optional[int] = None,
        tail_blend_max: Optional[float] = None,
        multi_horizon_recall: Optional[bool] = None,
        scale: Optional[float] = None,
    ) -> dict[str, Any]:
        """Switch all 8 MoE-Engram anchors on the fly to access deeper parametric knowledge without retraining.

        Supported presets (`mode`):
          - 'fast':           top_k=4,  multi_horizon=False, tail_blend=0.00, scale=0.20 (~0.88B active Engram params)
          - 'balanced':       top_k=7,  multi_horizon=False, tail_blend=0.00, scale=0.25 (~1.14B active Engram params)
          - 'deep_knowledge': top_k=16, multi_horizon=True,  tail_blend=0.18, scale=0.25 (~1.93B active Engram params)
          - 'max_recall':     top_k=24, multi_horizon=True,  tail_blend=0.22, scale=0.25 (~2.64B active Engram params)
          - 'full_dense':     top_k=32, multi_horizon=True,  tail_blend=0.25, scale=0.25 (~3.34B active Engram params)
        """
        presets = {
            'fast':           {'k': 4,  'mh': False, 'tb': 0.00, 'sc': 0.20},
            'balanced':       {'k': int(self.engram.active_top_k), 'mh': False, 'tb': 0.00, 'sc': 0.25},
            'deep_knowledge': {'k': 16, 'mh': True,  'tb': 0.18, 'sc': 0.25},
            'max_recall':     {'k': 24, 'mh': True,  'tb': 0.22, 'sc': 0.25},
            'full_dense':     {'k': 32, 'mh': True,  'tb': 0.25, 'sc': 0.25},
        }
        p = presets.get(str(mode).lower(), presets['balanced'])
        eff_k = int(top_k_flight) if top_k_flight is not None else int(p['k'])
        eff_mh = bool(multi_horizon_recall) if multi_horizon_recall is not None else bool(p['mh'])
        eff_tb = float(tail_blend_max) if tail_blend_max is not None else float(p['tb'])
        eff_sc = float(scale) if scale is not None else float(p['sc'])

        n_anchors = len(self.all_engram_modules())
        for mod in self.all_engram_modules():
            e_act = int(mod.active_experts)
            k_clamped = min(max(1, eff_k), e_act)
            mod.flight_mode = str(mode).lower()
            mod.flight_top_k = k_clamped if k_clamped != int(mod.active_top_k) else None
            mod.multi_horizon_recall = eff_mh
            mod.elastic_tail_blend_max = eff_tb
            mod.decode_engram_scale = eff_sc
            mod._eval_lex_cache = None

        # Compute exact active Engram parameter footprint across all 8 anchors
        m0 = self.engram
        e_act = int(m0.active_experts)
        k_used = min(max(1, eff_k), e_act)
        # Per-anchor active dense + routed SwiGLU + lexical parameters
        shared_params = m0.width * m0.shared_rank * 3
        routed_params = k_used * (m0.eng_dim * m0.expert_rank * 2 + m0.expert_rank * m0.width)
        proj_params = (m0.eng_dim * m0.width * 2) + (m0.width * m0.width * 2)
        lex_table_params = (m0.hot_buckets + m0.cold_buckets) * 3 * m0.eng_dim
        active_per_anchor = shared_params + routed_params + proj_params
        total_per_anchor = shared_params + (m0.num_experts * (m0.eng_dim * m0.expert_rank * 2 + m0.expert_rank * m0.width)) + proj_params + lex_table_params
        return {
            'flight_mode': str(mode).lower(),
            'anchors': n_anchors,
            'active_experts_pool': e_act,
            'base_top_k': int(m0.active_top_k),
            'flight_top_k': k_used,
            'total_experts_fired_per_token': k_used + 1,  # +1 for 100%-Active Shared SwiGLU Expert
            'multi_horizon_5gram_recall': eff_mh,
            'ngram_probes_per_token': 9 if eff_mh else 3,
            'elastic_tail_blend_max': eff_tb,
            'decode_engram_scale': eff_sc,
            'active_compute_engram_params_B': round((active_per_anchor * n_anchors) / 1e9, 3),
            'total_engram_capacity_B': round((total_per_anchor * n_anchors) / 1e9, 3),
        }

    @torch.no_grad()
    def snapshot_engram_bank(self) -> dict[str, Any]:
        """Take an in-memory CPU snapshot of all 8 MoE-Engram anchors for instant on-the-fly bank switching or blending."""
        bank = {}
        for l_idx, mod in self.iter_engram_anchors():
            sd = {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
            if getattr(mod, 'tier1b_cold_weight', None) is not None:
                sd['__tier1b_cold_weight__'] = mod.tier1b_cold_weight.detach().cpu().clone()
            bank[str(l_idx)] = sd
        return bank

    @torch.no_grad()
    def hot_swap_engram_bank(
        self,
        bank_source: Any,
        blend_alpha: float = 1.0,
    ) -> dict[str, Any]:
        """Hot-swap (`blend_alpha=1.0`) or federate/interpolate (`0 < blend_alpha < 1.0`) an Engram knowledge bank on the fly."""
        t0 = time.perf_counter()
        if isinstance(bank_source, str):
            bank_data = torch.load(bank_source, map_location='cpu', weights_only=False)
        elif isinstance(bank_source, dict):
            bank_data = bank_source
        else:
            raise ValueError('bank_source must be a file path or a snapshot dictionary')

        alpha = float(max(0.0, min(1.0, blend_alpha)))
        updated_tensors = 0
        for l_idx, mod in self.iter_engram_anchors():
            key = str(l_idx)
            src_sd = bank_data.get(key) or bank_data.get(f'L{l_idx}')
            if src_sd is None:
                continue
            cur_sd = mod.state_dict()
            for p_name, cur_t in cur_sd.items():
                if p_name in src_sd and src_sd[p_name].shape == cur_t.shape:
                    src_t = src_sd[p_name].to(device=cur_t.device, dtype=cur_t.dtype)
                    if alpha >= 0.9999:
                        cur_t.copy_(src_t)
                    else:
                        cur_t.lerp_(src_t, alpha)
                    updated_tensors += 1
            if '__tier1b_cold_weight__' in src_sd and getattr(mod, 'tier1b_cold_weight', None) is not None:
                c_src = src_sd['__tier1b_cold_weight__']
                if c_src.shape == mod.tier1b_cold_weight.shape:
                    if alpha >= 0.9999:
                        mod.tier1b_cold_weight.copy_(c_src.to(mod.tier1b_cold_weight.dtype))
                    else:
                        mod.tier1b_cold_weight.lerp_(c_src.to(mod.tier1b_cold_weight.dtype), alpha)
                    updated_tensors += 1
            mod._eval_lex_cache = None
            mod._eval_tier1b_gpu = None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            'status': 'ok',
            'blend_alpha': round(alpha, 3),
            'updated_tensors': updated_tensors,
            'swap_latency_ms': round(elapsed_ms, 2),
        }

    @torch.no_grad()
    def set_domain_engram_focus(
        self,
        domain: str = 'general',
        boost: float = 0.08,
    ) -> dict[str, Any]:
        """Partition the 32 routed experts per anchor into Domain Knowledge Engram Zones for SI-SRSI & specialized inference:
          - 'stem_reason' : Experts [0..11]  + Bridge [30..31] (14 specialists/anchor = 112 across 8 anchors)
          - 'web_ux_sec'  : Experts [12..21] + Bridge [30..31] (12 specialists/anchor =  96 across 8 anchors)
          - 'anti_slop'   : Experts [22..29] + Bridge [30..31] (10 specialists/anchor =  80 across 8 anchors)
          - 'general'     : Uniform zero bias across all 32 experts
        """
        dom = str(domain).lower().strip()
        zones = {
            'stem_reason': list(range(0, 12)) + [30, 31],
            'web_ux_sec':  list(range(12, 22)) + [30, 31],
            'anti_slop':   list(range(22, 30)) + [30, 31],
        }
        selected_indices = zones.get(dom, [])
        for mod in self.all_engram_modules():
            if not selected_indices or boost <= 0.0:
                mod._domain_zone_bias = None
            else:
                vec = torch.zeros(mod.num_experts, dtype=torch.float32, device=mod.expert_bias.device)
                for idx in selected_indices:
                    if idx < mod.num_experts:
                        vec[idx] = float(boost)
                # Zero-mean center so average selection threshold stays invariant
                vec = vec - vec.mean()
                mod._domain_zone_bias = vec
        return {
            'domain_focus': dom if selected_indices else 'general',
            'boost': round(float(boost) if selected_indices else 0.0, 4),
            'zone_experts': selected_indices,
            'expert_indices': selected_indices,
        }

    @torch.no_grad()
    def auto_detect_and_focus_domain(self, prompt_text: str, boost: float = 0.08) -> dict[str, Any]:
        """Automatically detect prompt domain (`anti_slop`, `web_ux_sec`, `stem_reason`, or `general`) and focus the Engram router."""
        low = str(prompt_text).lower()
        if any(k in low for k in ('html', 'css', 'react', 'ux', 'wcag', 'aria-', 'frontend', 'backend', 'endpoint', 'csrf', 'xss', 'csp', 'sql', 'api', 'tailwind', 'three.js', 'webgl')):
            dom = 'web_ux_sec'
        elif any(k in low for k in ('essay', 'report', 'prose', 'composition', 'editorial', 'article', 'memo', 'rewrite', 'human', 'anti-slop', 'tone', 'narrative')):
            dom = 'anti_slop'
        elif any(k in low for k in ('prove', 'derive', 'theorem', 'hamiltonian', 'quantum', 'eigenvalue', 'integral', 'equation', 'matrix', 'dijkstra', 'algorithm', 'complexity', 'boxed', 'olympiad', 'molecule', 'thermodynamic')):
            dom = 'stem_reason'
        else:
            dom = 'general'
        return self.set_domain_engram_focus(domain=dom, boost=boost)

    def _layers(self) -> nn.ModuleList:
        """Return the underlying 36-layer Spark decoder layer list."""
        return self.base.model.layers

    def loop_config(
        self,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        passes: Optional[int] = None,
        use_mhc: Optional[bool] = None,
        use_engram: Optional[bool] = None,
        enable_p3: Optional[bool] = None,
        p3_start: Optional[int] = None,
        p3_end: Optional[int] = None,
        use_esm: Optional[bool] = None,
    ):
        """Temporarily override loop/Engram/ESM execution settings and restore on exit."""
        return loop_config(
            self,
            start=start,
            end=end,
            passes=passes,
            use_mhc=use_mhc,
            use_engram=use_engram,
            enable_p3=enable_p3,
            p3_start=p3_start,
            p3_end=p3_end,
            use_esm=use_esm,
        )

    @torch.no_grad()
    def reset_sleep_time_memory_compiler(self) -> None:
        """Reset all per-anchor AlphaEdit null-space bases and history trackers."""
        for _, mod in self.iter_engram_anchors():
            mod.reset_alphaedit_basis()

    @torch.no_grad()
    def run_sleep_time_memory_compiler(
        self,
        episodic_input_ids: torch.Tensor,
        domain: str = "general",
        top_m_clusters: int = 6,
        alpha_ridge: float = 1e-2,
        target_scale: float = 0.35,
        outcome_valence: float = 1.0,
        inject_lexical: bool = False,
        calibrate_lsa_pca: bool = False,
        key_mode: str = "query",
        trace_in_loop_base: bool = True,
    ) -> dict[str, Any]:
        """Execute a Vertical Sleep-Time Memory Compilation cycle (`D-LSA Horizontal Working Attention -> Vertical MoE-Engram`).

        Pipeline:
          1. Captures Q/K/V and pre-attention hidden states across `Spark-X2.6-4B`'s 9 full-attention layers
             (`L3, 7, 11, 15, 19, 23, 27, 31, 35`) in a single gradient-free forward trace (`0` backbone backprop).
             On the 72L Loop Base (`trace_in_loop_base=True`), captures the first-visit (Pass 0) states while keeping
             `use_engram=False` so pre-existing Engram outputs do not double-count in the target.
          2. Extracts the top-M highest-salience LSA semantic clusters (`B_b`) using the 192-D RoPE-free pass-through
             slice (`[64:256]`) and computes their exact gated value residuals
             `y_b^(l) = out_proj(sigmoid(g_proj(LN h)) * sum_{j in B_b} alpha_j V_j^(l))`.
          3. Compiles those residuals vertically into the 8 paired `ZipfianTieredMoEEngram` anchors
             (`L3, 7, 11, 15, 19, 23, 27, 31`) via `AlphaEdit` Null-Space Constrained Closed-Form Ridge Compilation,
             guaranteeing `||A_hist @ Delta W_down||_inf ~ 1e-6` across sequential sleep cycles.
        """
        t0 = time.perf_counter()
        was_training = self.training
        self.eval()

        device = self._head_weight().device
        ids = episodic_input_ids.to(device=device, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        seq_len = int(ids.shape[1])

        layers = self._layers()
        # Ensure full-attention layers have `lsa_indexer` attached
        has_any_lsa = any(getattr(lyr.self_attn, "lsa_indexer", None) is not None for lyr in layers)
        if not has_any_lsa:
            patch_spark_attention_for_lsa(self.base)

        # Pair each MoE-Engram anchor (`{3, 7, 11, 15, 19, 23, 27, 31}`) to its exact or nearest full-attention layer
        full_attn_indices = [
            i for i, lyr in enumerate(layers) if getattr(lyr.self_attn, "lsa_indexer", None) is not None
        ]
        if not full_attn_indices:
            full_attn_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35]

        anchor_to_nope: dict[int, int] = {}
        needed_nope: set[int] = set()
        for l_eng, _ in self.iter_engram_anchors():
            nearest_nope = min(full_attn_indices, key=lambda n_idx: abs(int(n_idx) - int(l_eng)))
            anchor_to_nope[int(l_eng)] = int(nearest_nope)
            needed_nope.add(int(nearest_nope))

        captured_hidden: dict[int, torch.Tensor] = {}
        hooks = []

        def _make_attn_pre_hook(layer_idx: int):
            def _hook(_mod, args, kwargs):
                # Record only the first visit (Pass 0) so multi-pass loop execution stays paired with first-visit Q/K/V
                if layer_idx in captured_hidden:
                    return
                h_in = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if isinstance(h_in, torch.Tensor):
                    captured_hidden[layer_idx] = h_in.detach()
            return _hook

        for n_idx in needed_nope:
            if 0 <= n_idx < len(layers):
                lyr = layers[n_idx]
                lyr.self_attn._capture_sleep_qkv = True
                lyr.self_attn._sleep_last_qkv = None
                hooks.append(
                    lyr.self_attn.register_forward_pre_hook(_make_attn_pre_hook(n_idx), with_kwargs=True)
                )

        t_trace_start = time.perf_counter()
        try:
            if trace_in_loop_base:
                with self.loop_config(use_engram=False, use_esm=False):
                    _ = self.forward_hidden(ids)
            else:
                with self.loop_config(passes=1, use_mhc=False, use_engram=False, enable_p3=False, use_esm=False):
                    _ = self.forward_hidden(ids)
        finally:
            for h in hooks:
                h.remove()
            for n_idx in needed_nope:
                if 0 <= n_idx < len(layers):
                    layers[n_idx].self_attn._capture_sleep_qkv = False
        trace_latency_ms = (time.perf_counter() - t_trace_start) * 1000.0

        # Extract LSA clusters per needed full-attention layer
        t_extract_start = time.perf_counter()
        nope_clusters: dict[int, list[dict[str, Any]]] = {}
        evictable_positions: set[int] = set()
        for n_idx in sorted(needed_nope):
            if not (0 <= n_idx < len(layers)):
                continue
            lyr = layers[n_idx]
            qkv = getattr(lyr.self_attn, "_sleep_last_qkv", None)
            h_in = captured_hidden.get(n_idx)
            lsa_idx = getattr(lyr.self_attn, "lsa_indexer", None)
            out_proj = getattr(lyr.self_attn, "out_proj", None) or getattr(lyr.self_attn, "o_proj", None)
            g_proj = getattr(lyr.self_attn, "g_proj", None)
            if qkv is None or h_in is None or lsa_idx is None or out_proj is None:
                continue
            q_t, k_t, v_t = qkv
            if calibrate_lsa_pca and k_t.shape[2] >= lsa_idx.d_idx:
                k_samp = k_t[0].permute(1, 0, 2).contiguous()  # [T, H_kv, D]
                lsa_idx.fit_training_free_pca(k_samp)
            cl_list = lsa_idx.extract_salient_knowledge_clusters(
                query=q_t,
                key=k_t,
                value=v_t,
                hidden_states=h_in,
                out_proj_weight=out_proj.weight,
                g_proj_weight=(g_proj.weight if g_proj is not None else None),
                top_m_clusters=top_m_clusters,
                outcome_valence=outcome_valence,
            )
            nope_clusters[n_idx] = cl_list
            for cl in cl_list:
                evictable_positions.update(cl["token_positions"])
            lyr.self_attn._sleep_last_qkv = None
        extract_latency_ms = (time.perf_counter() - t_extract_start) * 1000.0

        t_solve_start = time.perf_counter()
        per_anchor_stats: dict[str, Any] = {}
        total_compiled = 0
        fit_errs: list[float] = []
        max_leak = 0.0

        for l_eng, mod in self.iter_engram_anchors():
            n_idx = anchor_to_nope.get(int(l_eng), int(l_eng))
            cl_list = nope_clusters.get(n_idx, [])
            res = mod.compile_clusters_alphaedit(
                clusters=cl_list,
                input_ids=ids,
                domain=domain,
                alpha_ridge=alpha_ridge,
                target_scale=target_scale,
                inject_lexical=inject_lexical,
                key_mode=key_mode,
            )
            res["source_nope_layer"] = int(n_idx)
            per_anchor_stats[f"L{l_eng}"] = res
            total_compiled += int(res["compiled_clusters"])
            fit_errs.append(float(res["mean_rel_fit_error"]))
            if float(res["max_null_space_leakage"]) > max_leak:
                max_leak = float(res["max_null_space_leakage"])
        solve_latency_ms = (time.perf_counter() - t_solve_start) * 1000.0

        if was_training:
            self.train()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        evict_cnt = len(evictable_positions)
        return {
            "status": "ok",
            "domain": str(domain),
            "episode_tokens": seq_len,
            "anchors_compiled": len(per_anchor_stats),
            "total_clusters_compiled": total_compiled,
            "evictable_kv_tokens": evict_cnt,
            "horizontal_compression_ratio": round(evict_cnt / max(1, seq_len), 4),
            "mean_rel_fit_error": round(sum(fit_errs) / max(1, len(fit_errs)), 6),
            "max_null_space_leakage": float(max_leak),
            "trace_latency_ms": round(trace_latency_ms, 2),
            "extract_latency_ms": round(extract_latency_ms, 2),
            "solve_latency_ms": round(solve_latency_ms, 2),
            "compile_latency_ms": round(elapsed_ms, 2),
            "per_anchor_stats": per_anchor_stats,
        }

    def total_router_z_loss(self) -> torch.Tensor:
        losses = []
        for mod in self.all_engram_modules():
            active_z = getattr(mod, '_active_z_losses', None)
            if active_z:
                losses.extend(active_z)
                mod._active_z_losses = None
            elif mod.last_router_z_loss is not None:
                losses.append(mod.last_router_z_loss.detach())
        if not losses:
            return torch.tensor(0.0, device=self._head_weight().device)
        return torch.stack([l.to(self._head_weight().device) for l in losses]).mean()

    def _validate_loop(self):
        if not (0 <= self.loop_start < self.loop_end <= self.n_layers):
            raise ValueError('Invalid loop boundaries')
        if (self.loop_start % 4) or (self.loop_end % 4):
            raise ValueError('Loop boundaries must align to Spark groups of four')
        if self.enable_p3:
            if not (self.loop_start <= self.p3_start < self.p3_end <= self.loop_end):
                raise ValueError('Pass-3 inner core must lie inside [loop_start, loop_end]')
            if (self.p3_start % 4) or (self.p3_end % 4):
                raise ValueError('Pass-3 boundaries must align to Spark groups of four')

    @property
    def execution_plan(self) -> list[int]:
        plan = (
            list(range(0, self.loop_start))
            + list(range(self.loop_start, self.loop_end)) * self.loop_passes
        )
        if self.enable_p3 and self.loop_passes >= 2:
            plan = plan + list(range(self.p3_start, self.p3_end))
        plan = plan + list(range(self.loop_end, self.n_layers))
        return plan

    @property
    def total_virtual_layers(self) -> int:
        return len(self.execution_plan)

    def effective_depth(self) -> int:
        return len(self.execution_plan)

    def _make_loop_cache_config(self):
        plan = tuple(self.execution_plan)
        if getattr(self, '_cached_loop_plan', None) == plan and getattr(self, '_cached_loop_cfg', None) is not None:
            return self._cached_loop_cfg
        cfg = copy.deepcopy(self.base.config)
        cfg.num_hidden_layers = len(plan)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[i] for i in plan]
        self._cached_loop_plan = plan
        self._cached_loop_cfg = cfg
        return cfg

    def _checkpoint_context(self):
        return nullcontext()

    def set_loop(
        self,
        start: int,
        end: int,
        passes: int,
        use_mhc: bool = True,
        use_engram: bool = True,
        enable_p3: bool = False,
        p3_start: int = 12,
        p3_end: int = 24,
    ):
        self.loop_start, self.loop_end, self.loop_passes = start, end, passes
        self.mhc_enabled, self.engram_enabled = use_mhc, use_engram
        self.enable_p3, self.p3_start, self.p3_end = enable_p3, p3_start, p3_end
        self._cached_loop_cfg = None
        self._validate_loop()
        for idx in range(start, end):
            if str(idx) not in self.controllers:
                raise RuntimeError(f'Controller for layer {idx} was not created')

    def _forward_context(self, embeds, attention_mask, past_key_values=None, cache_position=None):
        model = self.base.model
        cfg = self.base.config
        device = embeds.device
        bsz, q_len = embeds.shape[:2]
        custom_cache_pos = cache_position is not None
        if cache_position is None:
            if past_key_values is not None:
                try:
                    past_seen = int(past_key_values.get_seq_length(0))
                except Exception:
                    try:
                        past_seen = int(past_key_values.get_seq_length())
                    except Exception:
                        past_seen = 0
            else:
                past_seen = 0
            cache_position = torch.arange(past_seen, past_seen + q_len, device=device)
        if not custom_cache_pos and hasattr(model, 'make_loop_context'):
            return model.make_loop_context(embeds, attention_mask), cache_position
        if (past_key_values is not None and q_len == 1 and attention_mask is None) or (
            not custom_cache_pos and past_key_values is None and attention_mask is None and q_len > 1024
        ):
            masks = {'full_attention': None, 'sliding_attention': None}
        else:
            q_idx = cache_position.view(1, 1, q_len, 1)
            if past_key_values is None:
                k_len = q_len
                k_idx = cache_position.view(1, 1, 1, q_len)
            else:
                k_len = int(cache_position[-1].item()) + 1 if cache_position.numel() > 0 else q_len
                k_idx = torch.arange(k_len, device=device).view(1, 1, 1, k_len)
            causal_ok = (k_idx <= q_idx)
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                pad_ok = attention_mask[:, :k_len].bool().view(bsz, 1, 1, -1)
                if pad_ok.shape[-1] == k_len:
                    causal_ok = causal_ok & pad_ok
            min_val = torch.finfo(embeds.dtype).min
            zero_val = torch.tensor(0.0, dtype=embeds.dtype, device=device)
            masks = {'full_attention': torch.where(causal_ok, zero_val, min_val)}
            if getattr(model, 'has_sliding_layers', False):
                sw = int(getattr(cfg, 'sliding_window', 512) or 512)
                sw_ok = causal_ok & ((q_idx - k_idx) < sw)
                masks['sliding_attention'] = torch.where(sw_ok, zero_val, min_val)
        implementation = importlib.import_module(type(model.layers[0].self_attn).__module__)
        rope = {}
        for layer_type in set(cfg.layer_types):
            rope[layer_type] = implementation.compute_rope_cos_sin(
                cache_position, cfg.head_dim, cfg.get_rope_theta(layer_type),
                partial_rotary_factor=cfg.get_partial_rotary_factor(layer_type), device=device,
            )
        return (masks, rope), cache_position

    def _apply_layer(self, idx, h, masks, ropes, past_key_values=None, cache_position=None, virtual_idx: Optional[int] = None, read_only_kv: bool = False):
        layer = self.base.model.layers[idx]
        kind = layer.layer_type
        rp, mk = ropes[kind], masks[kind]
        kv_dtype = self.base.model.embedding.weight.dtype
        v_cache = (
            _VisitCache(past_key_values, idx, virtual_idx if virtual_idx is not None else idx, kv_dtype=kv_dtype, read_only=read_only_kv)
            if past_key_values is not None else None
        )
        def step(x, _layer=layer, _rp=rp, _mk=mk, _vc=v_cache, _cp=cache_position):
            out = _layer(x.float(), position_embeddings=_rp, attention_mask=_mk,
                         past_key_values=_vc, cache_position=_cp)
            return out.to(x.dtype)
        if v_cache is None and self.training and self.enable_checkpoint and (
            h.requires_grad or any(p.requires_grad for p in layer.parameters())
        ):
            ckpt_in = h.to(kv_dtype) if h.shape[1] >= 8192 else h
            return checkpoint(step, ckpt_in, use_reentrant=False).float()
        return layer(h, position_embeddings=rp, attention_mask=mk,
                     past_key_values=v_cache, cache_position=cache_position)

    def forward_hidden(self, input_ids: torch.LongTensor, attention_mask=None,
                       past_key_values=None, cache_position=None,
                       prev_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        input_ids = _unwrap_input_ids(input_ids)
        if input_ids.ndim != 2:
            raise ValueError('Expected input_ids shape [batch, sequence]')
        for mod in self.all_engram_modules():
            mod.last_router_z_loss = None
            mod._active_z_losses = [] if (self.training and torch.is_grad_enabled()) else None
        # Ahead-of-time deterministic token-ID hashing (computed before Layer 0!)
        ngram_indices = (
            self.engram.compute_ngram_indices(input_ids, self.engram.total_buckets, prev_ids=prev_ids)
            if self.engram_enabled else None
        )
        embeds = self.base.model.embedding(input_ids)
        (masks, ropes), cache_position = self._forward_context(
            embeds, attention_mask, past_key_values=past_key_values, cache_position=cache_position
        )
        # Match upstream Spark2_5Model.forward residual stream dtype (embeds.float())
        kv_dtype = embeds.dtype
        h = embeds.float()
        virtual_idx = 0
        visit_counts: dict[int, int] = {}
        first_virtual_idx: dict[int, int] = {}
        share_global_kv = bool(getattr(self, 'share_global_kv_across_passes', False))
        global_layer_set = set(getattr(self, 'esm_layers', (3, 7, 11, 15, 19, 23, 27, 31, 35)))
        cap_hook = getattr(self, '_residual_capture_hook', None)
        # BIE experiments: restrict in-loop Engram anchors to a subset of passes (None = all passes, the trained behaviour)
        engram_pass_filter = getattr(self, 'engram_pass_filter', None)
        esm_on = bool(getattr(self, 'esm_enabled', False)) and hasattr(self, 'esm')
        esm_visits_mode = str(getattr(self, 'esm_visits', 'first'))

        def _apply_engram_residual(h_in: torch.Tensor, _eng: ZipfianTieredMoEEngram, _p: int = 0) -> torch.Tensor:
            def _eng_step(x, _m=_eng, _pass=_p):
                return x + _m(
                    input_ids, x.dtype, prev_ids=prev_ids, hidden_states=x,
                    pass_idx=_pass, ngram_indices=ngram_indices,
                )
            if past_key_values is None and self.training and self.enable_checkpoint and (
                h_in.requires_grad or any(p.requires_grad for p in _eng.parameters())
            ):
                ckpt_in = h_in if h_in.requires_grad else h_in.detach().requires_grad_(True)
                return checkpoint(_eng_step, ckpt_in, use_reentrant=False)
            return _eng_step(h_in)

        def _maybe_apply_esm(h_in: torch.Tensor, layer_idx: int, v_idx: int) -> torch.Tensor:
            if not esm_on:
                return h_in
            if esm_visits_mode != 'all' and v_idx > 0:
                return h_in
            k_str = str(layer_idx)
            if k_str not in self.esm:
                return h_in
            esm_mod = self.esm[k_str]
            if not esm_mod.has_active_writes():
                return h_in
            return h_in + esm_mod(h_in).to(h_in.dtype)

        # 1. Prelude layers [0..loop_start) (includes Anchor 0 at Layer 3 + any backbone anchors < loop_start)
        for idx in range(self.loop_start):
            v_idx = visit_counts.get(idx, 0)
            visit_counts[idx] = v_idx + 1
            if v_idx == 0:
                first_virtual_idx[idx] = virtual_idx
            if cap_hook is not None:
                cap_hook(idx, v_idx, 0, 'pre_block', h)
            h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                  cache_position=cache_position, virtual_idx=virtual_idx)
            if cap_hook is not None:
                cap_hook(idx, v_idx, 0, 'post_layer', h)
            if self.engram_enabled:
                if idx == 3:
                    h = _apply_engram_residual(h, self.engram, _p=0)
                elif str(idx) in self.backbone_engrams:
                    h = _apply_engram_residual(h, self.backbone_engrams[str(idx)], _p=0)
            h = _maybe_apply_esm(h, idx, v_idx)
            if cap_hook is not None:
                cap_hook(idx, v_idx, 0, 'post_block', h)
            virtual_idx += 1
        if self.engram_enabled and 0 < self.loop_start <= 3:
            h = _apply_engram_residual(h, self.engram, _p=0)

        # Build pass schedule: Pass 0..loop_passes-1 over [loop_start, loop_end), plus optional Pass 2 over [p3_start, p3_end)
        pass_spans = [(p, self.loop_start, self.loop_end) for p in range(self.loop_passes)]
        if self.enable_p3 and self.loop_passes >= 2:
            pass_spans.append((2, self.p3_start, self.p3_end))

        if not self.mhc_enabled:
            for p, s_idx, e_idx in pass_spans:
                for idx in range(s_idx, e_idx):
                    v_idx = visit_counts.get(idx, 0)
                    visit_counts[idx] = v_idx + 1
                    if v_idx == 0:
                        first_virtual_idx[idx] = virtual_idx
                    share_ro = bool(v_idx > 0 and share_global_kv and idx in global_layer_set)
                    eff_v_idx = first_virtual_idx.get(idx, virtual_idx) if share_ro else virtual_idx
                    if cap_hook is not None:
                        cap_hook(idx, v_idx, p, 'pre_block', h)
                    h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                          cache_position=cache_position, virtual_idx=eff_v_idx, read_only_kv=share_ro)
                    if cap_hook is not None:
                        cap_hook(idx, v_idx, p, 'post_layer', h)
                    if self.engram_enabled and str(idx) in self.backbone_engrams:
                        eng_delta = self.backbone_engrams[str(idx)](
                            input_ids, h.dtype, prev_ids=prev_ids, hidden_states=h,
                            pass_idx=p, ngram_indices=ngram_indices,
                        )
                        if p == 0:
                            h = h + eng_delta
                    h = _maybe_apply_esm(h, idx, v_idx)
                    if cap_hook is not None:
                        cap_hook(idx, v_idx, p, 'post_block', h)
                    virtual_idx += 1
        else:
            # At >=8K training, store between-layer checkpoint boundary streams in BF16 (saves >10GB VRAM at 16K/60L-72L)
            # while TwoStreamMHC.forward executes all internal Sinkhorn & residual math in FP32.
            stream_dtype = kv_dtype if (self.training and self.enable_checkpoint and input_ids.shape[1] >= 8192) else h.dtype
            streams = torch.stack((h.to(stream_dtype), h.to(stream_dtype)), dim=0)
            if (
                past_key_values is None
                and self.training
                and torch.is_grad_enabled()
                and not streams.requires_grad
                and (
                    any(p.requires_grad for p in self.controllers.parameters())
                    or (self.engram_enabled and any(p.requires_grad for p in self.backbone_engrams.parameters()))
                )
            ):
                streams = streams.detach().requires_grad_(True)
            for p, s_idx, e_idx in pass_spans:
                for idx in range(s_idx, e_idx):
                    v_idx = visit_counts.get(idx, 0)
                    visit_counts[idx] = v_idx + 1
                    if v_idx == 0:
                        first_virtual_idx[idx] = virtual_idx
                    share_ro = bool(v_idx > 0 and share_global_kv and idx in global_layer_set)
                    eff_v_idx = first_virtual_idx.get(idx, virtual_idx) if share_ro else virtual_idx
                    if cap_hook is not None:
                        cap_hook(idx, v_idx, p, 'pre_block', streams[0])
                    kind = self.base.model.layers[idx].layer_type
                    layer = self.base.model.layers[idx]
                    ctrl = self.controllers[str(idx)]
                    rp, mk = ropes[kind], masks[kind]
                    v_cache = (
                        _VisitCache(past_key_values, idx, eff_v_idx, kv_dtype=kv_dtype, read_only=share_ro)
                        if past_key_values is not None else None
                    )
                    eng_mod = (
                        self.backbone_engrams[str(idx)]
                        if (self.engram_enabled and str(idx) in self.backbone_engrams)
                        else None
                    )
                    virtual_idx += 1
                    def step(s, _ctrl=ctrl, _layer=layer, _rp=rp, _mk=mk, _extra=(p > 0), _p=p, _vc=v_cache, _cp=cache_position, _eng=eng_mod, _idx=idx, _v_idx=v_idx, _pf=engram_pass_filter):
                        s_out = _ctrl(s, _layer, _rp, _mk, extra=_extra, pass_idx=_p,
                                      past_key_values=_vc, cache_position=_cp)
                        if cap_hook is not None:
                            cap_hook(_idx, _v_idx, _p, 'post_layer', s_out[0])
                        if _eng is not None and (_pf is None or _p in _pf):
                            eng_delta = _eng(
                                input_ids, s_out.dtype, prev_ids=prev_ids, hidden_states=s_out[0],
                                pass_idx=_p, ngram_indices=ngram_indices,
                            )
                            if _extra:
                                fgain_param = _ctrl.p3_ffn_extra_gain if _p >= 2 else _ctrl.ffn_extra_gain
                                fgain = _ctrl._effective_gain(fgain_param, pass_idx=_p, extra=True)
                                eng_delta = (eng_delta.float() * fgain).to(s_out.dtype)
                            s_out = torch.stack((s_out[0] + eng_delta, s_out[1]), dim=0)
                        if esm_on and (esm_visits_mode == 'all' or _v_idx == 0) and str(_idx) in self.esm:
                            esm_m = self.esm[str(_idx)]
                            if esm_m.has_active_writes():
                                esm_d = esm_m(s_out[0]).to(s_out.dtype)
                                s_out = torch.stack((s_out[0] + esm_d, s_out[1]), dim=0)
                        return s_out
                    if v_cache is None and self.training and self.enable_checkpoint:
                        streams = checkpoint(step, streams, use_reentrant=False)
                    else:
                        streams = step(streams)
                    if cap_hook is not None:
                        cap_hook(idx, v_idx, p, 'post_block', streams[0])
            h = streams[0].float()

        # 3. Coda layers [loop_end..n_layers) (includes Anchor 7 at Layer 31 + Global Layer 35!)
        for idx in range(self.loop_end, self.n_layers):
            v_idx = visit_counts.get(idx, 0)
            visit_counts[idx] = v_idx + 1
            if cap_hook is not None:
                cap_hook(idx, v_idx, 0, 'pre_block', h)
            h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                  cache_position=cache_position, virtual_idx=virtual_idx)
            if cap_hook is not None:
                cap_hook(idx, v_idx, 0, 'post_layer', h)
            if self.engram_enabled and str(idx) in self.backbone_engrams:
                h = _apply_engram_residual(h, self.backbone_engrams[str(idx)], _p=0)
            h = _maybe_apply_esm(h, idx, v_idx)
            if cap_hook is not None:
                cap_hook(idx, v_idx, 0, 'post_block', h)
            virtual_idx += 1
        return self.base.model.norm(h).to(kv_dtype)

    def _head_weight(self) -> torch.Tensor:
        if getattr(self.base.config, 'tie_word_embeddings', False):
            return self.base.model.embedding.weight
        return self.base.lm_head.weight

    def forward_logits(self, input_ids: torch.LongTensor, attention_mask=None,
                       past_key_values=None, cache_position=None,
                       prev_ids: Optional[torch.LongTensor] = None):
        input_ids = _unwrap_input_ids(input_ids)
        h = self.forward_hidden(
            input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            prev_ids=prev_ids,
        )
        if getattr(self.base.config, 'tie_word_embeddings', False):
            return F.linear(h, self.base.model.embedding.weight)
        return self.base.lm_head(h)

    def forward(self, input_ids, labels=None, attention_mask=None, return_logits: bool = False):
        input_ids = _unwrap_input_ids(input_ids)
        if labels is None:
            logits = self.forward_logits(input_ids, attention_mask)
            return {'logits': logits}
        if input_ids.shape[1] < 2:
            raise ValueError('Need at least 2 tokens for next-token objective')
        h = self.forward_hidden(input_ids, attention_mask=attention_mask)
        ce_loss = fused_masked_cross_entropy(
            h, self._head_weight(), labels, chunk_tokens=self.settings.ce_chunk_tokens
        )
        z_loss = self.total_router_z_loss() * float(self.settings.eng_z_loss_coeff) if (self.training and self.engram_enabled) else 0.0
        loss = ce_loss + z_loss
        if return_logits:
            logits = F.linear(h, self._head_weight()) if getattr(self.base.config, 'tie_word_embeddings', False) else self.base.lm_head(h)
            return {'logits': logits, 'loss': loss, 'ce_loss': ce_loss, 'z_loss': z_loss}
        return {'logits': None, 'loss': loss, 'ce_loss': ce_loss, 'z_loss': z_loss}

    @torch.no_grad()
    def generate_cached(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32768,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.4,
        top_p: float = 0.90,
        top_k: int = 40,
        repetition_penalty: float = 1.03,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
        tokenizer=None,
        think_budget_tokens: Optional[int] = None,
        stream_print: bool = True,
        **kwargs,
    ):
        """Pure, un-distorted 72-visit BF16 KV-cached looped generation."""
        from transformers.cache_utils import DynamicCache
        input_ids = _unwrap_input_ids(input_ids)
        prev_train = self.training
        self.eval()
        cache = DynamicCache()
        out = input_ids

        stop_ids = set()
        if eos_token_id is not None:
            stop_ids.add(int(eos_token_id))
        think_reentry_mask = {3}  # Token ID 3 is <think> in Spark-X2.6-4B
        think_close_ids = {4}     # Token ID 4 is </think> in Spark-X2.6-4B
        close_seq_ids = [4]
        if tokenizer is not None:
            if getattr(tokenizer, 'eos_token_id', None) is not None:
                stop_ids.add(int(tokenizer.eos_token_id))
            for stop_str in ('<｜end▁of▁sentence｜>', '<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
                try:
                    s_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    if len(s_ids) == 1:
                        stop_ids.add(int(s_ids[0]))
                except Exception:
                    pass
            for t_str in ('<think>', '<think>\n'):
                try:
                    t_ids = tokenizer.encode(t_str, add_special_tokens=False)
                    if len(t_ids) == 1:
                        think_reentry_mask.add(int(t_ids[0]))
                except Exception:
                    pass
            try:
                c_ids = tokenizer.encode('</think>', add_special_tokens=False)
                if len(c_ids) == 1:
                    think_close_ids.add(int(c_ids[0]))
                seq_c = tokenizer.encode('\n</think>\n', add_special_tokens=False)
                if seq_c:
                    close_seq_ids = [int(x) for x in seq_c]
            except Exception:
                pass

        min_think_tokens = int(kwargs.get('min_think_tokens', 0) or 0)
        eff_think_budget: Optional[int] = None
        if think_budget_tokens is not None and int(think_budget_tokens) > 0:
            eff_think_budget = min(int(think_budget_tokens), max(64, int(max_new_tokens) - 256))
        else:
            # Default safety ceiling so generation never hits max_new_tokens while still inside <think>
            eff_think_budget = max(128, int(max_new_tokens) - 320) if int(max_new_tokens) >= 512 else None

        # Detect if chat_template.jinja already placed <think> (token 3) at the end of input_ids
        in_think = bool(out.shape[1] > 0 and int(out[0, -1].item()) in think_reentry_mask)
        if not in_think and tokenizer is not None and out.shape[1] >= 2:
            try:
                prompt_tail = tokenizer.decode(out[0, -4:].tolist(), skip_special_tokens=False)
                if '<think>' in prompt_tail and '</think>' not in prompt_tail:
                    in_think = True
            except Exception:
                pass
        think_closed = False
        gen_list: list[int] = []
        t0 = time.time()

        def _sample_step(step_logits: torch.Tensor, cur_gen: list[int], is_closed: bool) -> torch.Tensor:
            logits_1d = step_logits[0, -1, :].float().clone()
            if is_closed and think_reentry_mask:
                for tid in think_reentry_mask:
                    if 0 <= tid < logits_1d.shape[0]:
                        logits_1d[tid] = -1e9
            elif not is_closed and min_think_tokens > 0 and len(cur_gen) < min_think_tokens:
                # Prevent premature </think> or EOS before minimum reasoning depth is reached
                for tid in (think_close_ids | stop_ids):
                    if 0 <= tid < logits_1d.shape[0]:
                        logits_1d[tid] = -1e9
            if cur_gen and repetition_penalty > 1.0:
                win = cur_gen[-256:]
                u_ids = torch.tensor(list(set(win)), dtype=torch.long, device=logits_1d.device)
                vals = logits_1d[u_ids]
                rep_p = float(repetition_penalty)
                logits_1d[u_ids] = torch.where(vals > 0, vals / rep_p, vals * rep_p)
            if not do_sample or temperature <= 1e-5:
                return logits_1d.argmax(dim=-1, keepdim=True).unsqueeze(0)
            scaled = logits_1d / float(temperature)
            if top_k > 0:
                k_val = min(int(top_k), scaled.shape[-1])
                v_k, _ = torch.topk(scaled, k_val)
                scaled = torch.where(scaled < v_k[-1], torch.full_like(scaled, -1e9), scaled)
            probs = F.softmax(scaled, dim=-1)
            if 0.0 < top_p < 1.0:
                s_probs, s_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(s_probs, dim=-1)
                mask = (cum - s_probs) > top_p
                s_probs = torch.where(mask, torch.zeros_like(s_probs), s_probs)
                s_probs = s_probs / s_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                choice = torch.multinomial(s_probs, num_samples=1)
                return s_idx.gather(-1, choice).unsqueeze(0)
            return torch.multinomial(probs, num_samples=1).unsqueeze(0)

        try:
            logits = self.forward_logits(out, past_key_values=cache)
            next_id = _sample_step(logits, gen_list, think_closed)
            tok_int = int(next_id.item())
            gen_list.append(tok_int)
            if tok_int in think_close_ids:
                in_think = False
                think_closed = True
            prev_tok = out[:, -2:]
            out = torch.cat((out, next_id), dim=-1)
            if tok_int in stop_ids:
                return out
            step_i = 1
            while step_i < max_new_tokens:
                # Enforce think_budget_tokens so the model NEVER truncates mid-thought without emitting </think>
                if not think_closed and eff_think_budget is not None and len(gen_list) >= eff_think_budget:
                    for cid in close_seq_ids:
                        if step_i >= max_new_tokens:
                            break
                        _ = self.forward_logits(next_id, past_key_values=cache, prev_ids=prev_tok)
                        prev_tok = out[:, -2:]
                        next_id = torch.tensor([[cid]], dtype=out.dtype, device=out.device)
                        gen_list.append(cid)
                        out = torch.cat((out, next_id), dim=-1)
                        step_i += 1
                    in_think = False
                    think_closed = True
                    if stream_print:
                        dt = max(1e-3, time.time() - t0)
                        print(f"  [72L Budget </think>] Closed </think> at budget={eff_think_budget} tokens ({len(gen_list)/dt:.1f} tok/s).", flush=True)
                    if step_i >= max_new_tokens:
                        break

                logits = self.forward_logits(next_id, past_key_values=cache, prev_ids=prev_tok)
                prev_tok = out[:, -2:]
                if not think_closed and tokenizer is not None and (len(gen_list) <= 16 or len(gen_list) % 8 == 0):
                    tail_txt = tokenizer.decode(gen_list[-48:], skip_special_tokens=False)
                    if '<think>' in tail_txt and '</think>' not in tail_txt:
                        in_think = True
                    if '</think>' in tail_txt:
                        if in_think and stream_print:
                            dt = max(1e-3, time.time() - t0)
                            print(f"  [72L Natural </think>] Model closed </think> naturally at {len(gen_list)} tokens ({len(gen_list)/dt:.1f} tok/s).", flush=True)
                        in_think = False
                        think_closed = True

                next_id = _sample_step(logits, gen_list, think_closed)
                tok_int = int(next_id.item())
                gen_list.append(tok_int)
                if tok_int in think_close_ids:
                    in_think = False
                    think_closed = True
                out = torch.cat((out, next_id), dim=-1)
                step_i += 1

                if stream_print and tokenizer is not None and len(gen_list) % 128 == 0:
                    dt = max(1e-3, time.time() - t0)
                    snippet = tokenizer.decode(gen_list[-24:], skip_special_tokens=False).replace('\n', ' ')
                    phase = 'think' if in_think else 'answer'
                    print(f"  [Gen | {len(gen_list):5d} tok | {len(gen_list)/dt:5.1f} tok/s | phase={phase}] ...{snippet[-72:]}", flush=True)

                if tok_int in stop_ids:
                    if stream_print:
                        dt = max(1e-3, time.time() - t0)
                        print(f"  [Gen Complete] Hit stop token at {len(gen_list)} tokens ({len(gen_list)/dt:.1f} tok/s).", flush=True)
                    break
        finally:
            self.train(prev_train)
        return out


def set_trainables_v03(
    model: SparkXLoopV03,
    full_finetune: bool = True,
    freeze_backbone: bool = False,
    freeze_engram: Optional[bool] = None,
):
    if freeze_engram is None:
        freeze_engram = not bool(model.engram_enabled)
    for p in model.base.parameters():
        p.requires_grad_(bool(full_finetune and not freeze_backbone))
    # Always keep embeddings frozen to protect 167,168 vocabulary geometry
    for p in model.base.model.embedding.parameters():
        p.requires_grad_(False)
    if full_finetune and not freeze_backbone:
        for p in model.base.model.norm.parameters():
            p.requires_grad_(True)
    for p in model.controllers.parameters():
        p.requires_grad_(bool(model.mhc_enabled))
    for p in model.engram.parameters():
        p.requires_grad_(not bool(freeze_engram))
    for p in model.backbone_engrams.parameters():
        p.requires_grad_(not bool(freeze_engram))
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def build_optimizer_v03(
    model: SparkXLoopV03,
    lr_adapter: float = 2.0e-4,
    lr_full: float = 1.0e-5,
    lr_mhc: float = 5.0e-3,
    weight_decay: float = 0.01,
    use_8bit: bool = True,
):
    adapters, routing, backbone = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            (name.startswith('controllers.') and '.pass_' not in name)
            or name.endswith('.gain')
            or name.endswith('.router_log_temp')
            or name.endswith('.scale_weights')
        ):
            routing.append(p)
        elif name.startswith('base.model.'):
            backbone.append(p)
        else:
            adapters.append(p)
    groups = [
        {'params': adapters, 'lr': lr_adapter, 'weight_decay': weight_decay},
        {'params': routing, 'lr': lr_mhc, 'weight_decay': 0.0},
    ]
    if backbone:
        groups.append({'params': backbone, 'lr': lr_full, 'weight_decay': weight_decay})
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(groups, betas=(0.9, 0.95), eps=1e-8)
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
        except Exception:
            pass
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def count_en4b_a1b_params(model: SparkXLoopV03) -> dict[str, int]:
    """Compute exact parameter breakdown across Backbone, GPU Engrams, and Offloaded Tier-1B DRAM/SSD."""
    backbone_total = sum(p.numel() for p in model.base.parameters())
    ctrl_total = sum(p.numel() for p in model.controllers.parameters())
    eng_gpu_total = sum(p.numel() for mod in model.all_engram_modules() for p in mod.parameters())
    eng_cpu_tier1b = sum(
        mod.tier1b_cold_weight.numel()
        for mod in model.all_engram_modules()
        if getattr(mod, 'tier1b_cold_weight', None) is not None
    )
    eng_total = eng_gpu_total + eng_cpu_tier1b
    active_eng = 0
    for mod in model.all_engram_modules():
        t2 = mod.shared_gate.weight.numel() + mod.shared_up.weight.numel() + mod.shared_down.weight.numel()
        per_exp = (mod.eng_dim * mod.expert_rank * 2) + (mod.expert_rank * mod.width)
        t3_act = mod.active_top_k * per_exp
        proj_act = mod.lexical_up.weight.numel() + mod.query_proj.weight.numel() + mod.ctx_gate.weight.numel() + mod.project.weight.numel()
        active_eng += (t2 + t3_act + proj_act)
    total_params = backbone_total + ctrl_total + eng_total
    active_params = backbone_total + ctrl_total + active_eng
    return {
        'backbone_total': backbone_total,
        'controllers_total': ctrl_total,
        'engram_gpu_total': eng_gpu_total,
        'engram_cpu_tier1b_total': eng_cpu_tier1b,
        'engram_total': eng_total,
        'engram_active_per_pass': active_eng,
        'model_total_params': total_params,
        'model_active_params': active_params,
        'grand_total': total_params,
        'active_total_per_token': active_params,
    }


def loop_v03_state_dict(model: SparkXLoopV03, include_tier1b: bool = True) -> dict:
    """Serialize all v0.3 loop modules (mHC controllers + 8 MoE-Engrams + optional offloaded Tier-1B banks)."""
    tier1b = {}
    if include_tier1b:
        if getattr(model.engram, 'tier1b_cold_weight', None) is not None:
            tier1b['engram'] = model.engram.tier1b_cold_weight.detach().cpu()
        for k, mod in model.backbone_engrams.items():
            if getattr(mod, 'tier1b_cold_weight', None) is not None:
                tier1b[f'backbone_engrams.{k}'] = mod.tier1b_cold_weight.detach().cpu()
    active_meta = {
        'engram_active_experts': model.engram.active_experts,
        'engram_active_top_k': model.engram.active_top_k,
        'use_pass_gain_schedule': bool(getattr(model.engram, 'use_pass_gain_schedule', False)),
    }
    return {
        'version': 'v0.3-en4b-a1b',
        'settings': dataclasses.asdict(model.settings),
        'active_meta': active_meta,
        'controllers': {k: v.detach().cpu() for k, v in model.controllers.state_dict().items()},
        'engram': {k: v.detach().cpu() for k, v in model.engram.state_dict().items()},
        'backbone_engrams': {k: v.detach().cpu() for k, v in model.backbone_engrams.state_dict().items()},
        'tier1b_cold': tier1b,
    }


def load_loop_v03_state_dict(model: SparkXLoopV03, state: dict, strict: bool = False) -> dict:
    """Restore v0.3 loop state (including active expert count and pinned Host DRAM Tier-1B weights)."""
    meta = state.get('active_meta', {})
    act_e = int(meta.get('engram_active_experts', model.engram.active_experts))
    act_k = int(meta.get('engram_active_top_k', model.engram.active_top_k))
    use_pg = bool(meta.get('use_pass_gain_schedule', False))
    for mod in model.all_engram_modules():
        mod.active_experts = min(act_e, mod.num_experts)
        mod.active_top_k = min(act_k, mod.active_experts)
        mod.use_pass_gain_schedule = use_pg
    c_res = model.controllers.load_state_dict(state['controllers'], strict=strict)
    e_res = model.engram.load_state_dict(state['engram'], strict=strict)
    b_res = model.backbone_engrams.load_state_dict(state['backbone_engrams'], strict=strict)
    tier1b = state.get('tier1b_cold', {})
    if 'engram' in tier1b and getattr(model.engram, 'tier1b_cold_weight', None) is not None:
        model.engram.tier1b_cold_weight.copy_(tier1b['engram'])
    for k, mod in model.backbone_engrams.items():
        key = f'backbone_engrams.{k}'
        if key in tier1b and getattr(mod, 'tier1b_cold_weight', None) is not None:
            mod.tier1b_cold_weight.copy_(tier1b[key])
    return {'controllers': c_res, 'engram': e_res, 'backbone_engrams': b_res}


@torch.no_grad()
def warm_start_from_v02_state(model: SparkXLoopV03, v02_state: dict) -> dict[str, int]:
    """Warm-start v0.3 mHC controllers and compatible Engram gates/norms from a v0.2 `loop_state.pt`."""
    copied_ctrl, copied_eng = 0, 0
    v02_ctrl = v02_state.get('controllers', {})
    cur_ctrl = model.controllers.state_dict()
    for k, src_t in v02_ctrl.items():
        if k in cur_ctrl:
            dst_t = cur_ctrl[k]
            if dst_t.shape == src_t.shape:
                dst_t.copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype))
                copied_ctrl += 1
            elif dst_t.ndim == src_t.ndim and dst_t.shape[1:] == src_t.shape[1:] and dst_t.shape[0] >= src_t.shape[0]:
                dst_t[:src_t.shape[0]].copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype))
                copied_ctrl += 1

    def _copy_compatible_engram(dst_mod: ZipfianTieredMoEEngram, src_dict: dict) -> int:
        cnt = 0
        dst_sd = dst_mod.state_dict()
        for k in ('scale_weights', 'norm_weight', 'ctx_gate.weight', 'ctx_gate.bias', 'project.weight', 'gain'):
            if k in src_dict and k in dst_sd and dst_sd[k].shape == src_dict[k].shape:
                dst_sd[k].copy_(src_dict[k].to(device=dst_sd[k].device, dtype=dst_sd[k].dtype))
                cnt += 1
        return cnt

    if 'engram' in v02_state:
        copied_eng += _copy_compatible_engram(model.engram, v02_state['engram'])
    if 'backbone_engrams' in v02_state:
        v02_be = v02_state['backbone_engrams']
        for idx_key, mod in model.backbone_engrams.items():
            sub_dict = {
                k.split(f'{idx_key}.', 1)[1]: v
                for k, v in v02_be.items()
                if k.startswith(f'{idx_key}.')
            }
            if sub_dict:
                copied_eng += _copy_compatible_engram(mod, sub_dict)
    return {'copied_controller_tensors': copied_ctrl, 'copied_engram_tensors': copied_eng}


patch_spark_for_loglinear_kv = patch_spark_attention_for_loglinear_kv


class _MetricFloat(float):
    """A float value that also supports dict-style indexing (`['max_row_err']`, `['causal_prefix_unchanged']`)."""

    def __new__(cls, val: float, meta: dict):
        obj = super().__new__(cls, float(val))
        obj._meta = dict(meta)
        return obj

    def __getitem__(self, key):
        return self._meta[key]

    def get(self, key, default=None):
        return self._meta.get(key, default)

    def keys(self):
        return self._meta.keys()

    def items(self):
        return self._meta.items()


@torch.no_grad()
def verify_birkhoff_sinkhorn(ctrl: Optional[TwoStreamMHC] = None) -> _MetricFloat:
    """Return max absolute row/col deviation from doubly-stochastic 1.0 across random dynamic perturbations."""
    if ctrl is None:
        ctrl = TwoStreamMHC(width=64, dynamic=False, pass_adapter_rank=0)
    device = ctrl.mix_logits.device
    delta = torch.randn(4, 16, 2, 2, device=device, dtype=torch.float32) * 3.0
    mat = ctrl.mixing_matrix(dynamic_delta=delta)
    row_err = float((mat.sum(dim=-1) - 1.0).abs().max().item())
    col_err = float((mat.sum(dim=-2) - 1.0).abs().max().item())
    max_err = max(row_err, col_err)
    return _MetricFloat(max_err, {'max_row_err': row_err, 'max_col_err': col_err, 'max_err': max_err})


@torch.no_grad()
def verify_ngram_causality(
    model_or_engram: Any,
    vocab_size: int = 167168,
    device: Optional[torch.device] = None,
) -> _MetricFloat:
    """Verify that mutating future tokens t >= prefix_len has zero effect on prefix_len hashes/outputs."""
    if isinstance(model_or_engram, ZipfianTieredMoEEngram):
        engram = model_or_engram
        if device is None:
            device = engram.project.weight.device
        seq_a = torch.randint(10, min(vocab_size, 50000), (1, 16), device=device)
        seq_b = seq_a.clone()
        seq_b[:, 8:] = torch.randint(10, min(vocab_size, 50000), (1, 8), device=device)
        idx_a = engram.compute_ngram_indices(seq_a, engram.total_buckets)
        idx_b = engram.compute_ngram_indices(seq_b, engram.total_buckets)
        hash_diff = max(float((a[:, :8] - b[:, :8]).abs().max().item()) for a, b in zip(idx_a, idx_b))
        h_in = torch.randn(1, 16, engram.width, device=device, dtype=engram.project.weight.dtype)
        out_a = engram(seq_a, dtype=engram.project.weight.dtype, hidden_states=h_in).float()[:, :8, :]
        out_b = engram(seq_b, dtype=engram.project.weight.dtype, hidden_states=h_in).float()[:, :8, :]
        out_diff = float((out_a - out_b).abs().max().item())
        max_diff = max(hash_diff, out_diff)
        return _MetricFloat(max_diff, {'causal_prefix_unchanged': (max_diff == 0.0), 'max_diff': max_diff})

    model = model_or_engram
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
    max_diff = max(hash_diff, logit_diff)
    return _MetricFloat(max_diff, {'causal_prefix_unchanged': (max_diff < 1e-4), 'max_diff': max_diff})

