"""SparkX-Loop v0.3 (`Spark-X2.6-4B-En4B-A1B`):
8.2B Total / 5.2B Active Parameters | 72 Effective Layers (3-Pass Telescoping Recurrence) |
3-Tier Zipfian MoE-Engram (4.0B Total / 1.0B Active across 8 Macro-Block Anchors) with
Host DRAM / NVMe mmap Tier-1B Offloading, Coalesced Sparse Row Adam, 8->32 Coarse-to-Fine
Expert Upcycling, Anti-Collapse Hypersphere Cosine + Sigmoid + Z-Loss Routing, and
Runtime-Only Zero-Collapse O(n log n) Log-Linear NoPE KV Cache.
"""
from __future__ import annotations
from contextlib import nullcontext
import copy
import dataclasses
import importlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
    ce_chunk_tokens: int = 512


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
    and enforces BF16/FP16 KV cache storage so RoPE float32 intermediates never bloat KV memory.
    """
    def __init__(self, cache, physical_idx: int, virtual_idx: int, kv_dtype: Optional[torch.dtype] = None):
        self.cache = cache
        self.physical_idx = physical_idx
        self.virtual_idx = virtual_idx
        self.kv_dtype = kv_dtype
        if hasattr(cache, 'layers'):
            self.layers = _CacheLayersView(cache, physical_idx, virtual_idx)

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        target = self.virtual_idx if layer_idx == self.physical_idx else layer_idx
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


def patch_spark_attention_for_long_context(base_model: nn.Module) -> int:
    """Replace Spark2_5Attention's O(T^2) eager_attention_forward with fused BF16 PyTorch SDPA / GQA."""
    if not hasattr(base_model, 'model') or not hasattr(base_model.model, 'layers'):
        return 0
    layers = base_model.model.layers
    if not layers or not hasattr(layers[0], 'self_attn'):
        return 0
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
    if getattr(attn_mod, '_sparkx_sdpa_patched', False) and not getattr(attn_mod, '_sparkx_loglinear_patched', False):
        return len(layers)

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

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, : key.shape[-2]]
            if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
                attn_mask = attn_mask.to(query.dtype)
        use_gqa = _SDPA_SUPPORTS_GQA and (module.num_key_value_groups > 1)
        if not use_gqa and orig_repeat_kv is not None and module.num_key_value_groups > 1:
            key = orig_repeat_kv(key, module.num_key_value_groups)
            value = orig_repeat_kv(value, module.num_key_value_groups)
        sdpa_kwargs = {'scale': scaling}
        if use_gqa:
            sdpa_kwargs['enable_gqa'] = True
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=False,
            **sdpa_kwargs,
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
    block_mean_k = k_mid_f.mean(dim=3)
    block_mean_norm = k_mid_f.norm(dim=-1, keepdim=True).mean(dim=3)
    centroid_k = block_mean_k * (block_mean_norm / (block_mean_k.norm(dim=-1, keepdim=True) + 1e-6))
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
            sub_k = centroid_k[:, :, i : i + span, :]
            sub_norm = sub_k.norm(dim=-1, keepdim=True).mean(dim=2, keepdim=True)
            sub_mean = sub_k.mean(dim=2, keepdim=True)
            merged_k = sub_mean * (sub_norm / (sub_mean.norm(dim=-1, keepdim=True) + 1e-6))
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
    exact_threshold: int = 2048,
    sink_tokens: int = 32,
    local_window: int = 1024,
    base_block: int = 32,
    hip_top_blocks: int = 8,
) -> int:
    """Runtime-only O(n^2) -> O(n log n) KV cache optimization for Spark-X2.6-4B's NoPE layers."""
    if not hasattr(base_model, 'model') or not hasattr(base_model.model, 'layers'):
        return 0
    layers = base_model.model.layers
    if not layers or not hasattr(layers[0], 'self_attn'):
        return 0
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

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, : key.shape[-2]]
            if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
                attn_mask = attn_mask.to(query.dtype)
        use_gqa = _SDPA_SUPPORTS_GQA and (module.num_key_value_groups > 1)
        if not use_gqa and orig_repeat_kv is not None and module.num_key_value_groups > 1:
            key = orig_repeat_kv(key, module.num_key_value_groups)
            value = orig_repeat_kv(value, module.num_key_value_groups)
        sdpa_kwargs = {'scale': scaling}
        if use_gqa:
            sdpa_kwargs['enable_gqa'] = True
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=False,
            **sdpa_kwargs,
        )
        return out, None

    attn_mod.eager_attention_forward = loglinear_sdpa_attention_forward
    attn_mod._sparkx_sdpa_patched = True
    attn_mod._sparkx_loglinear_patched = True
    return len(layers)


class ZipfianTieredMoEEngram(nn.Module):
    """3-Tier Zipfian MoE-Engram (`En4B-A1B`: 500M Total / 125M Active per Anchor x 8 Anchors = 4.0B Total / 1.0B Active).

    - Tier 1A (GPU HBM Hot Lexical Cache): `hot_buckets` x `eng_dim` x 3 scales (2-gram, 3-gram, skip-2-gram), zero-init.
    - Tier 1B (Host DRAM / NVMe mmap Cold N-Gram Bank): `cold_buckets` x `eng_dim` x 3 scales, zero-init, with
      deterministic ahead-of-time prefetch and coalesced sparse row Adam updates (`step_tier1b_sparse_adam`).
    - Tier 2 (100%-Active Hot Shared SwiGLU Expert): `shared_gate_up` [width -> 2*shared_rank] + `shared_down` [shared_rank -> width].
    - Tier 3 (E=32 Routed SwiGLU Memory Experts, Top-K=7):
      - X-MoE L2-normalized Hypersphere Cosine Router + DeepSeek-V3 Sigmoid Affinity + Auxiliary-Loss-Free Bias
      - ST-MoE Router Z-Loss (`last_router_z_loss`)
      - Coarse-to-Fine Upcycling (`active_experts = 8 -> 32` via `upcycle_experts_8_to_32()`)
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

        # Tier 1B: Host DRAM / mmap Long-Tail N-Gram Bank (not registered as a dense GPU Parameter when offloaded)
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

        # Tier 3: Routed SwiGLU Memory Experts (E=32, rank=1440 -> 353.89M params/anchor)
        # Parameterized as SwiGLU over query_proj(h_t) + e_lex in R^eng_dim -> expert_rank -> width
        # Wait: to reach exact 353.89M per anchor (or configurable rank), we use:
        # w_gate: [E, eng_dim, expert_rank], w_up: [E, eng_dim, expert_rank], w_down: [E, expert_rank, width]
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
        self.gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.last_router_z_loss: torch.Tensor = torch.tensor(0.0)

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
        """Clone the 8 trained coarse macro-experts into 32 fine-grained experts with orthogonal perturbations."""
        if self.active_experts >= self.num_experts:
            self.active_top_k = self.target_top_k
            return
        coarse = self.active_experts
        factor = self.num_experts // coarse
        for m in range(coarse):
            for sub in range(factor):
                dst = m * factor + sub
                if dst == m:
                    continue
                self.expert_gate.data[dst].copy_(self.expert_gate.data[m])
                self.expert_up.data[dst].copy_(self.expert_up.data[m])
                self.expert_down.data[dst].copy_(self.expert_down.data[m])
                self.expert_centroids.data[dst].copy_(self.expert_centroids.data[m])
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
        if prev_ids is None:
            p1 = F.pad(input_ids[:, :-1], (1, 0), value=0)
            p2 = F.pad(input_ids[:, :-2], (2, 0), value=0)
        else:
            cat1 = torch.cat([prev_ids[:, -1:], input_ids[:, :-1]], dim=1)
            if prev_ids.shape[1] >= 2:
                cat2 = torch.cat([prev_ids[:, -2:], input_ids[:, :-2]], dim=1)
            else:
                pad2 = F.pad(prev_ids[:, -1:], (1, 0), value=0)
                cat2 = torch.cat([pad2, input_ids[:, :-2]], dim=1)[:, : input_ids.shape[1]]
            p1, p2 = cat1, cat2
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

        # Hot Tier 1A lookup on GPU HBM
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

        if not (is_cold_bi.any() or is_cold_tri.any() or is_cold_skip.any()):
            return sw[0] * e_bi + sw[1] * e_tri + sw[2] * e_skip

        cold_b = self.cold_buckets
        c_bi = (bi - hot_b).clamp(min=0, max=cold_b - 1)
        c_tri = (tri - hot_b).clamp(min=0, max=cold_b - 1) + cold_b
        c_skip = (skip - hot_b).clamp(min=0, max=cold_b - 1) + 2 * cold_b

        if self.offload_tier1b_cpu and self.tier1b_cold_weight is not None:
            # Coalesced unique row prefetch from pinned Host DRAM -> GPU HBM
            stacked = torch.stack([c_bi, c_tri, c_skip], dim=0)
            uniq_gpu, inv = torch.unique(stacked, return_inverse=True)
            uniq_cpu = uniq_gpu.cpu()
            fetched_cpu = self.tier1b_cold_weight[uniq_cpu]
            fetched_gpu = fetched_cpu.to(device=device, dtype=param_dtype, non_blocking=True)
            if self.training:
                fetched_gpu = fetched_gpu.detach().requires_grad_(True)
                self._pending_tier1b_rows = uniq_cpu
                self._pending_tier1b_leaf = fetched_gpu
            gathered = fetched_gpu[inv]  # [3, B, T, eng_dim]
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
        """Coalesced sparse Adam update on pinned Host DRAM for Tier 1B (<0.3 ms/step)."""
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

        # 1. Tier 1A + 1B Lexical N-gram Lookup
        e_lexical = self._lookup_tier1_lexical(ngram_indices, input_ids.device, param_dtype)
        lex_hidden = self.lexical_up(e_lexical)

        if hidden_states is not None:
            h_f = hidden_states.float()
            rms = torch.rsqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            h_norm_f = h_f * rms * self.norm_weight
            h_norm = h_norm_f.to(param_dtype)

            # 2. Tier 2: 100%-Active Hot Shared SwiGLU Expert
            h_fused_in = h_norm + lex_hidden
            shared_out = self.shared_down(F.silu(self.shared_gate(h_fused_in)) * self.shared_up(h_fused_in))

            # 3. Tier 3: X-MoE Hypersphere Cosine Router + Sigmoid Affinity + ST-MoE Router Z-Loss
            e_act = self.active_experts
            k_act = min(self.active_top_k, e_act)
            p_idx = min(max(int(pass_idx), 0), self.max_passes - 1)

            u = F.normalize(self.router_proj(h_norm_f), p=2, dim=-1, eps=1e-6)
            c = F.normalize(self.expert_centroids[:e_act], p=2, dim=-1, eps=1e-6)
            temp = self.router_log_temp.exp().clamp(min=0.03, max=2.0)
            router_logits = (torch.matmul(u, c.t()) / temp) + self.pass_router_bias[p_idx, :e_act]

            if self.training:
                log_z = torch.logsumexp(router_logits, dim=-1)
                self.last_router_z_loss = log_z.pow(2).mean()
            else:
                self.last_router_z_loss = router_logits.new_zeros(())

            scores = torch.sigmoid(router_logits)
            sel_scores = scores + self.expert_bias[:e_act]
            _, topk_idx = torch.topk(sel_scores, k=k_act, dim=-1)
            topk_raw = torch.gather(scores, dim=-1, index=topk_idx)
            topk_gates = (topk_raw / (topk_raw.sum(dim=-1, keepdim=True) + 1e-6)).to(param_dtype)

            if self.training:
                with torch.no_grad():
                    one_hot = F.one_hot(topk_idx, num_classes=e_act).float()
                    load = one_hot.sum(dim=-2).mean(dim=(0, 1)) / float(k_act)
                    self.expert_load_ema[:e_act].lerp_(load, 0.05)
                    target_load = 1.0 / float(e_act)
                    self.expert_bias[:e_act].add_(0.001 * torch.sign(target_load - self.expert_load_ema[:e_act])).clamp_(-0.5, 0.5)

            # Query vector q_t in R^eng_dim conditioned on both semantic hidden state and lexical N-gram
            q = self.query_proj(h_norm) + e_lexical
            flat_q = q.reshape(-1, self.eng_dim)
            flat_idx = topk_idx.reshape(-1, k_act)
            flat_gates = topk_gates.reshape(-1, k_act)

            # Memory-efficient grouped expert evaluation (iterates over active experts instead of materializing [N, K, D, R])
            routed_flat = torch.zeros((flat_q.shape[0], self.width), dtype=param_dtype, device=flat_q.device)
            for e_id in range(e_act):
                mask_e = (flat_idx == e_id)
                if not mask_e.any():
                    continue
                token_rows, slot_cols = mask_e.nonzero(as_tuple=True)
                q_e = flat_q.index_select(0, token_rows)
                g_e = flat_gates[token_rows, slot_cols].unsqueeze(-1)
                act_e = F.silu(torch.matmul(q_e, self.expert_gate[e_id])) * torch.matmul(q_e, self.expert_up[e_id])
                out_e = torch.matmul(act_e, self.expert_down[e_id]) * g_e
                routed_flat.index_add_(0, token_rows, out_e)

            routed_out = routed_flat.view_as(shared_out)
            combined = lex_hidden + shared_out + routed_out
            ctx = 2.0 * torch.sigmoid(self.ctx_gate(h_norm).float()).to(param_dtype)
            combined = combined * ctx
        else:
            combined = F.silu(lex_hidden)

        projected = self.project(combined)
        gate = torch.sigmoid(self.gain).to(dtype)
        return projected.to(dtype) * gate


class TwoStreamMHC(nn.Module):
    """Sublayer Dual-mHC with Pass-0/Pass-1/Pass-2 (72-Layer) Conditioning and Closed-Form 2x2 Sinkhorn."""
    def __init__(
        self,
        width: int = 2560,
        streams: int = 2,
        iters: int = 20,
        dynamic: bool = True,
        sublayer_mhc: bool = True,
        pass_adapter_rank: int = 16,
        max_passes: int = 4,
    ):
        super().__init__()
        if streams != 2:
            raise ValueError('TwoStreamMHC expects 2 streams')
        self.width = width
        self.streams = streams
        self.iters = iters
        self.dynamic = dynamic
        self.sublayer_mhc = sublayer_mhc
        self.pass_adapter_rank = pass_adapter_rank
        self.max_passes = max_passes
        self.norm_eps = 1e-6

        self.read_logits = nn.Parameter(torch.tensor([2.0, -2.0], dtype=torch.float32))
        self.write_logits = nn.Parameter(torch.tensor([2.0, -2.0], dtype=torch.float32))
        self.mix_logits = nn.Parameter(torch.tensor([[2.0, -2.0], [-2.0, 2.0]], dtype=torch.float32))
        self.extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.p3_extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.ffn_read_logits = nn.Parameter(torch.tensor([2.0, -2.0], dtype=torch.float32))
        self.ffn_write_logits = nn.Parameter(torch.tensor([2.0, -2.0], dtype=torch.float32))
        self.ffn_mix_logits = nn.Parameter(torch.tensor([[2.0, -2.0], [-2.0, 2.0]], dtype=torch.float32))
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

    def mixing_matrix(self, dynamic_delta: Optional[torch.Tensor] = None, base_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.mix_logits if base_logits is None else base_logits
        log_p = base if dynamic_delta is None else (base + dynamic_delta)
        log_p = log_p.clamp(min=-30.0, max=30.0)
        # Closed-form exact 2x2 symmetric Birkhoff projection:
        # For a 2x2 matrix [[a, b], [c, d]], Sinkhorn converges to a symmetric doubly-stochastic
        # matrix parameterized by the cross-ratio (a + d) - (b + c):
        diag_adv = 0.5 * ((log_p[..., 0, 0] + log_p[..., 1, 1]) - (log_p[..., 0, 1] + log_p[..., 1, 0]))
        p_keep = torch.sigmoid(diag_adv)
        p_swap = 1.0 - p_keep
        row0 = torch.stack([p_keep, p_swap], dim=-1)
        row1 = torch.stack([p_swap, p_keep], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def _compute_routing(self, streams: torch.Tensor, is_ffn: bool, pass_idx: int = 0):
        p_idx = min(max(pass_idx, 0), self.max_passes - 1)
        r_base = self.ffn_read_logits if is_ffn else self.read_logits
        w_base = self.ffn_write_logits if is_ffn else self.write_logits
        m_base = self.ffn_mix_logits if is_ffn else self.mix_logits
        p_bias = self.pass_routing_bias[p_idx, 1 if is_ffn else 0]
        r_base = r_base + p_bias[0:2]
        w_base = w_base + p_bias[2:4]
        m_base = m_base + p_bias[4:8].view(2, 2)

        if not self.dynamic:
            p = self.mixing_matrix(base_logits=m_base).to(streams.dtype)
            r = F.softmax(r_base, dim=0).to(streams.dtype)
            w = (2.0 * torch.sigmoid(w_base)).to(streams.dtype)
            return p, r, w, True

        flat = streams.permute(1, 2, 0, 3).reshape(streams.shape[1], streams.shape[2], -1).float()
        rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        nw = self.ffn_norm_weight if is_ffn else self.norm_weight
        proj = self.ffn_dynamic_proj if is_ffn else self.dynamic_proj
        raw = proj(flat * rms * nw)
        r = F.softmax(r_base + raw[..., 0:2], dim=-1).to(streams.dtype)
        w = (2.0 * torch.sigmoid(w_base + raw[..., 2:4])).to(streams.dtype)
        p = self.mixing_matrix(raw[..., 4:8].view(*raw.shape[:-1], 2, 2), base_logits=m_base).to(streams.dtype)
        return p, r, w, False

    @staticmethod
    def _mix_and_read(streams: torch.Tensor, p: torch.Tensor, r: torch.Tensor, is_static: bool):
        if is_static:
            mixed = torch.einsum('ij,jbtd->ibtd', p, streams)
            layer_in = torch.einsum('i,ibtd->btd', r, mixed)
        else:
            mixed = torch.einsum('btij,jbtd->ibtd', p, streams)
            layer_in = torch.einsum('bti,ibtd->btd', r, mixed)
        return mixed, layer_in

    @staticmethod
    def _write(mixed: torch.Tensor, delta: torch.Tensor, w: torch.Tensor, is_static: bool):
        if is_static:
            return mixed + w.view(2, 1, 1, 1) * delta.unsqueeze(0)
        return mixed + w.permute(2, 0, 1).unsqueeze(-1) * delta.unsqueeze(0)

    def _apply_pass_adapter(self, h: torch.Tensor, pass_idx: int) -> torch.Tensor:
        if self.pass_adapter_rank <= 0 or pass_idx <= 0:
            return h
        p_idx = min(pass_idx, self.max_passes - 1)
        delta = self.pass_up[p_idx](F.silu(self.pass_down[p_idx](h.float())))
        return h + delta.to(h.dtype)

    def forward(self, streams: torch.Tensor, layer: nn.Module, position_embeddings, attention_mask,
                extra: bool = False, pass_idx: int = 0, past_key_values=None, cache_position=None) -> torch.Tensor:
        p_attn, r_attn, w_attn, static_attn = self._compute_routing(streams, is_ffn=False, pass_idx=pass_idx)
        mixed_attn, attn_in = self._mix_and_read(streams, p_attn, r_attn, static_attn)

        if self.sublayer_mhc and hasattr(layer, 'self_attn') and hasattr(layer, 'mlp'):
            target_dtype = layer.mlp.gate_proj.weight.dtype
            normed_attn = layer.input_layernorm(attn_in).to(target_dtype)
            attn_out, _ = layer.self_attn(
                hidden_states=normed_attn,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            attn_delta = attn_out.to(streams.dtype)
            if extra:
                gain_param = self.p3_extra_gain if pass_idx >= 2 else self.extra_gain
                attn_delta = attn_delta * torch.tanh(gain_param.float()).to(streams.dtype)
            mid_streams = self._write(mixed_attn, attn_delta, w_attn, static_attn)

            p_ffn, r_ffn, w_ffn, static_ffn = self._compute_routing(mid_streams, is_ffn=True, pass_idx=pass_idx)
            mixed_ffn, ffn_in = self._mix_and_read(mid_streams, p_ffn, r_ffn, static_ffn)
            ffn_in = self._apply_pass_adapter(ffn_in, pass_idx)
            normed_ffn = layer.post_attention_layernorm(ffn_in).to(target_dtype)
            ffn_out = layer.mlp(normed_ffn)
            ffn_delta = ffn_out.to(streams.dtype)
            if extra:
                fgain_param = self.p3_ffn_extra_gain if pass_idx >= 2 else self.ffn_extra_gain
                ffn_delta = ffn_delta * torch.tanh(fgain_param.float()).to(streams.dtype)
            return self._write(mixed_ffn, ffn_delta, w_ffn, static_ffn)

        layer_in = self._apply_pass_adapter(attn_in, pass_idx)
        layer_out = layer(
            layer_in,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        delta = (layer_out - attn_in).to(streams.dtype)
        if extra:
            gain_param = self.p3_extra_gain if pass_idx >= 2 else self.extra_gain
            delta = delta * torch.tanh(gain_param.float()).to(streams.dtype)
        return self._write(mixed_attn, delta, w_attn, static_attn)


def fused_masked_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int = 512,
) -> torch.Tensor:
    shift_h = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_h = shift_h.view(-1, shift_h.shape[-1])
    flat_labels = shift_labels.view(-1)
    valid_mask = flat_labels != -100
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return (flat_h[:1].float().sum() * 0.0)
    valid_h = flat_h.index_select(0, valid_indices)
    valid_t = flat_labels.index_select(0, valid_indices)
    w = lm_head_weight.to(valid_h.dtype)
    total_loss = valid_h.new_zeros((), dtype=torch.float32)
    n_valid = valid_t.numel()
    for start in range(0, n_valid, max(64, chunk_tokens)):
        end = min(start + max(64, chunk_tokens), n_valid)
        logits_chunk = F.linear(valid_h[start:end], w).float()
        total_loss = total_loss + F.cross_entropy(logits_chunk, valid_t[start:end], reduction='sum')
    return total_loss / float(n_valid)


class SparkXLoopV03(nn.Module):
    """Spark-X2.6-4B-En4B-A1B (v0.3):
    - 36 physical layers unrolled up to 72 effective layers via 3-Pass Telescoping Recurrence:
      Prelude [0..4) + Pass-1 [4..28) + Pass-2 [4..28) + Pass-3 [12..24) + Coda [28..36) = 72 virtual layers.
    - 8 Macro-Block MoE-Engram Anchors at physical layers 3, 7, 11, 15, 19, 23, 27, 31
      (500M total / 125M active per anchor = 4.0B Total / 1.0B Active Engram).
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
        # Core + Coda macro-block anchors: Layers 7, 11, 15, 19, 23, 27, 31 (7 anchors + 1 Prelude = 8 anchors total)
        anchor_indices = [
            idx for idx in range(self.loop_start + stride - 1, min(self.n_layers - 4, 32), stride)
        ]
        self.backbone_engrams = nn.ModuleDict({
            str(idx): _make_engram() for idx in anchor_indices
        })

    def all_engram_modules(self) -> list[ZipfianTieredMoEEngram]:
        return [self.engram] + [mod for mod in self.backbone_engrams.values()]

    def upcycle_all_engrams_8_to_32(self, noise_std: float = 1e-3):
        for mod in self.all_engram_modules():
            mod.upcycle_experts_8_to_32(noise_std=noise_std)

    def step_all_tier1b_sparse_adam(self, lr: float = 2e-4):
        for mod in self.all_engram_modules():
            mod.step_tier1b_sparse_adam(lr=lr)

    def total_router_z_loss(self) -> torch.Tensor:
        losses = [mod.last_router_z_loss for mod in self.all_engram_modules() if mod.last_router_z_loss is not None]
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
        if self.training and self.enable_checkpoint and self.offload_checkpoint_to_cpu:
            return torch.autograd.graph.save_on_cpu(pin_memory=True)
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
        if cache_position is None:
            past_seen = past_key_values.get_seq_length(0) if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + embeds.shape[1], device=device)
        position_ids = cache_position.unsqueeze(0)
        if hasattr(model, 'make_loop_context'):
            return model.make_loop_context(embeds, attention_mask), cache_position
        if past_key_values is not None and embeds.shape[1] == 1 and attention_mask is None:
            masks = {'full_attention': None, 'sliding_attention': None}
        else:
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
            mask_cfg = self._make_loop_cache_config() if past_key_values is not None else cfg
            kwargs = dict(config=mask_cfg, input_embeds=embeds, attention_mask=attention_mask,
                          cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids)
            masks = {'full_attention': create_causal_mask(**kwargs)}
            if getattr(model, 'has_sliding_layers', False):
                masks['sliding_attention'] = create_sliding_window_causal_mask(**kwargs)
        implementation = importlib.import_module(type(model.layers[0].self_attn).__module__)
        rope = {}
        rope_dtype = embeds.dtype if (past_key_values is not None and embeds.dtype in (torch.bfloat16, torch.float16)) else None
        for layer_type in set(cfg.layer_types):
            cos, sin = implementation.compute_rope_cos_sin(
                cache_position, cfg.head_dim, cfg.get_rope_theta(layer_type),
                partial_rotary_factor=cfg.get_partial_rotary_factor(layer_type), device=device,
            )
            if rope_dtype is not None:
                cos = cos.to(rope_dtype)
                sin = sin.to(rope_dtype)
            rope[layer_type] = (cos, sin)
        return (masks, rope), cache_position

    def _apply_layer(self, idx, h, masks, ropes, past_key_values=None, cache_position=None, virtual_idx: Optional[int] = None):
        layer = self.base.model.layers[idx]
        kind = layer.layer_type
        rp, mk = ropes[kind], masks[kind]
        kv_dtype = self.base.model.embedding.weight.dtype
        v_cache = (
            _VisitCache(past_key_values, idx, virtual_idx if virtual_idx is not None else idx, kv_dtype=kv_dtype)
            if past_key_values is not None else None
        )
        target_dtype = layer.mlp.gate_proj.weight.dtype
        def step(x, _layer=layer, _rp=rp, _mk=mk, _vc=v_cache, _cp=cache_position, _dt=target_dtype):
            x_in = x.to(_dt) if x.dtype != _dt else x
            out = _layer(x_in, position_embeddings=_rp, attention_mask=_mk,
                         past_key_values=_vc, cache_position=_cp)
            return out.to(x.dtype) if out.dtype != x.dtype else out
        if v_cache is None and self.training and self.enable_checkpoint and (
            h.requires_grad or any(p.requires_grad for p in layer.parameters())
        ):
            with self._checkpoint_context():
                return checkpoint(step, h, use_reentrant=False)
        return step(h)

    def forward_hidden(self, input_ids: torch.LongTensor, attention_mask=None,
                       past_key_values=None, cache_position=None,
                       prev_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError('Expected input_ids shape [batch, sequence]')
        # Ahead-of-time deterministic token-ID hashing (computed before Layer 0!)
        ngram_indices = (
            self.engram.compute_ngram_indices(input_ids, self.engram.total_buckets, prev_ids=prev_ids)
            if self.engram_enabled else None
        )
        embeds = self.base.model.embedding(input_ids)
        (masks, ropes), cache_position = self._forward_context(
            embeds, attention_mask, past_key_values=past_key_values, cache_position=cache_position
        )
        h = embeds.float()
        kv_dtype = embeds.dtype
        virtual_idx = 0

        # 1. Prelude layers [0..loop_start)
        for idx in range(self.loop_start):
            h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                  cache_position=cache_position, virtual_idx=virtual_idx)
            virtual_idx += 1
        if self.engram_enabled:
            h = h + self.engram(
                input_ids, h.dtype, prev_ids=prev_ids, hidden_states=h,
                pass_idx=0, ngram_indices=ngram_indices,
            )

        # Build pass schedule: Pass 0..loop_passes-1 over [loop_start, loop_end), plus optional Pass 2 over [p3_start, p3_end)
        pass_spans = [(p, self.loop_start, self.loop_end) for p in range(self.loop_passes)]
        if self.enable_p3 and self.loop_passes >= 2:
            pass_spans.append((2, self.p3_start, self.p3_end))

        if not self.mhc_enabled:
            for p, s_idx, e_idx in pass_spans:
                for idx in range(s_idx, e_idx):
                    h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                          cache_position=cache_position, virtual_idx=virtual_idx)
                    if self.engram_enabled and str(idx) in self.backbone_engrams:
                        eng_delta = self.backbone_engrams[str(idx)](
                            input_ids, h.dtype, prev_ids=prev_ids, hidden_states=h,
                            pass_idx=p, ngram_indices=ngram_indices,
                        )
                        if p == 0:
                            h = h + eng_delta
                    virtual_idx += 1
        else:
            streams = torch.stack((h, h), dim=0)
            for p, s_idx, e_idx in pass_spans:
                for idx in range(s_idx, e_idx):
                    kind = self.base.model.layers[idx].layer_type
                    layer = self.base.model.layers[idx]
                    ctrl = self.controllers[str(idx)]
                    rp, mk = ropes[kind], masks[kind]
                    v_cache = _VisitCache(past_key_values, idx, virtual_idx, kv_dtype=kv_dtype) if past_key_values is not None else None
                    eng_mod = (
                        self.backbone_engrams[str(idx)]
                        if (self.engram_enabled and str(idx) in self.backbone_engrams)
                        else None
                    )
                    virtual_idx += 1
                    def step(s, _ctrl=ctrl, _layer=layer, _rp=rp, _mk=mk, _extra=(p > 0), _p=p, _vc=v_cache, _cp=cache_position, _eng=eng_mod):
                        s_out = _ctrl(s, _layer, _rp, _mk, extra=_extra, pass_idx=_p,
                                      past_key_values=_vc, cache_position=_cp)
                        if _eng is not None:
                            eng_delta = _eng(
                                input_ids, s_out.dtype, prev_ids=prev_ids, hidden_states=s_out[0],
                                pass_idx=_p, ngram_indices=ngram_indices,
                            )
                            if _extra:
                                fgain = _ctrl.p3_ffn_extra_gain if _p >= 2 else _ctrl.ffn_extra_gain
                                eng_delta = eng_delta * torch.tanh(fgain.float()).to(s_out.dtype)
                            s_out = torch.stack((s_out[0] + eng_delta, s_out[1]), dim=0)
                        return s_out
                    if v_cache is None and self.training and self.enable_checkpoint:
                        with self._checkpoint_context():
                            streams = checkpoint(step, streams, use_reentrant=False)
                    else:
                        streams = step(streams)
            h = streams[0]

        # 3. Coda layers [loop_end..n_layers) (includes Anchor 7 at Layer 31!)
        for idx in range(self.loop_end, self.n_layers):
            h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                  cache_position=cache_position, virtual_idx=virtual_idx)
            if self.engram_enabled and str(idx) in self.backbone_engrams:
                h = h + self.backbone_engrams[str(idx)](
                    input_ids, h.dtype, prev_ids=prev_ids, hidden_states=h,
                    pass_idx=0, ngram_indices=ngram_indices,
                )
            virtual_idx += 1
        norm_in = h.to(kv_dtype) if (not self.training and kv_dtype in (torch.bfloat16, torch.float16)) else h
        return self.base.model.norm(norm_in).to(kv_dtype)

    def _head_weight(self) -> torch.Tensor:
        if getattr(self.base.config, 'tie_word_embeddings', False):
            return self.base.model.embedding.weight
        return self.base.lm_head.weight

    def forward_logits(self, input_ids: torch.LongTensor, attention_mask=None,
                       past_key_values=None, cache_position=None,
                       prev_ids: Optional[torch.LongTensor] = None):
        h = self.forward_hidden(
            input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            prev_ids=prev_ids,
        )
        if getattr(self.base.config, 'tie_word_embeddings', False):
            return F.linear(h, self.base.model.embedding.weight)
        return self.base.lm_head(h)

    def forward(self, input_ids, labels=None, attention_mask=None, return_logits: bool = False):
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
    def generate_cached(self, input_ids, max_new_tokens=256, eos_token_id=None):
        from transformers.cache_utils import DynamicCache
        prev_train = self.training
        self.eval()
        try:
            cache = DynamicCache(config=self._make_loop_cache_config())
        except TypeError:
            cache = DynamicCache()
        out = input_ids
        try:
            logits = self.forward_logits(out, past_key_values=cache)
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            prev_tok = out[:, -2:]
            out = torch.cat((out, next_id), dim=-1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                return out
            for _ in range(max_new_tokens - 1):
                logits = self.forward_logits(next_id, past_key_values=cache, prev_ids=prev_tok)
                prev_tok = out[:, -2:]
                next_id = logits[:, -1].argmax(-1, keepdim=True)
                out = torch.cat((out, next_id), dim=-1)
                if eos_token_id is not None and torch.all(next_id == eos_token_id):
                    break
        finally:
            self.train(prev_train)
        return out


def set_trainables_v03(model: SparkXLoopV03, full_finetune: bool = True, freeze_backbone: bool = False):
    for p in model.base.parameters():
        p.requires_grad_(bool(full_finetune and not freeze_backbone))
    # Always keep embeddings frozen to protect 167,168 vocabulary geometry
    for p in model.base.model.embedding.parameters():
        p.requires_grad_(False)
    if full_finetune and not freeze_backbone:
        for p in model.base.model.norm.parameters():
            p.requires_grad_(True)
    for p in model.controllers.parameters():
        p.requires_grad_(True)
    for p in model.engram.parameters():
        p.requires_grad_(True)
    for p in model.backbone_engrams.parameters():
        p.requires_grad_(True)
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def build_optimizer_v03(
    model: SparkXLoopV03,
    lr_adapter: float = 2.0e-4,
    lr_full: float = 2.0e-6,
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
    # Active Engram per pass across 8 anchors: Tier-2 (100%) + Tier-3 (active_top_k / num_experts) + ctx_gate/project/query
    active_eng = 0
    for mod in model.all_engram_modules():
        t2 = mod.shared_gate.weight.numel() + mod.shared_up.weight.numel() + mod.shared_down.weight.numel()
        per_exp = (mod.eng_dim * mod.expert_rank * 2) + (mod.expert_rank * mod.width)
        t3_act = mod.active_top_k * per_exp
        proj_act = mod.lexical_up.weight.numel() + mod.query_proj.weight.numel() + mod.ctx_gate.weight.numel() + mod.project.weight.numel()
        active_eng += (t2 + t3_act + proj_act)
    return {
        'backbone_total': backbone_total,
        'controllers_total': ctrl_total,
        'engram_gpu_total': eng_gpu_total,
        'engram_cpu_tier1b_total': eng_cpu_tier1b,
        'engram_total': eng_total,
        'engram_active_per_pass': active_eng,
        'model_total_params': backbone_total + ctrl_total + eng_total,
        'model_active_params': backbone_total + ctrl_total + active_eng,
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
    for mod in model.all_engram_modules():
        mod.active_experts = min(act_e, mod.num_experts)
        mod.active_top_k = min(act_k, mod.active_experts)
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

