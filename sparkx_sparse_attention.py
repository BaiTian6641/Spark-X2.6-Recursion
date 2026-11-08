"""sparkx_sparse_attention.py — Offset-Aware Local Sliding Attention, Causal Paged D-LSA Global Attention,
and YaRN Partial-RoPE Position Scaling for Spark X2.6 4B Loop72 LSA (Work Packages P2, P4, P8).

Implements §3.1, §5.3, §5.4, §6.5, and §8 of the Loop72 LSA Review and Development Plan:
  1. Offset-aware local sliding attention (`local_sliding_attention_forward`): uses exact absolute query/key positions
     from `LocalSlidingRingSlot` so chunked prefill (`511/512/513` boundaries) and decode never misalign masks after ring eviction.
  2. Strictly causal paged D-LSA global attention (`paged_dlsa_global_attention_forward`):
     - Single-query sparse decode (`q_len == 1`) with per-visit `VisitAwareDLSAIndexBank` and `GlobalPagedVisitSlot.gather_selected`.
     - Tiled multi-query causal sparse prefill (`sparse_query_tile_tokens = 16`): each query at absolute position `p_i`
       selects strictly from `upto = p_i + 1` (never seeing future blocks or averaging chunk queries), coalesces page gathers
       over the tile union, and applies an exact per-query causal selection mask `S_union[j] in S_i and S_union[j] <= p_i`.
  3. YaRN-style frequency ramp scaling (`compute_yarn_partial_rope_cos_sin`, P8) on the 64 rotated dimensions of the
     global layers (`theta=5e6`), constructed on the fly in FP32 without pre-allocating multi-million-token tables.
"""
from __future__ import annotations

import dataclasses
import math
import types
from typing import Any, Dict, List, Optional, Tuple

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
    from sparkx_dlsa import FaithfulDLSAIndexer, PagedLatentIndex, VisitAwareDLSAIndexBank, _dummy_q
    from sparkx_paged_cache import GlobalPagedVisitSlot, LocalSlidingRingSlot, Loop72HybridPagedCache


@dataclasses.dataclass
class SparseAttentionConfig:
    index_dim: int = 64
    maps_per_global_visit: bool = True
    search: str = "flat"
    pool_size: int = 4
    cluster_block_tokens: int = 512
    average_bucket_tokens: int = 8
    max_bucket_tokens: int = 16
    sinks: int = 4
    recent_tokens: int = 512
    long_range_tokens: int = 4096
    rerank: int = 1
    sparse_query_tile_tokens: int = 16
    selection_reuse: bool = False
    residual_compensation: bool = False
    unbounded_dense_fallback: bool = False
    yarn_scale_factor: float = 1.0
    yarn_orig_max_position: int = 1048576
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0


@dataclasses.dataclass
class SparseRuntimeTelemetry:
    sparse_prefill_calls: int = 0
    sparse_decode_calls: int = 0
    exact_short_calls: int = 0
    local_sliding_calls: int = 0
    dense_fallback_calls: int = 0
    total_selected_keys: int = 0
    total_sparse_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        mean_sel = self.total_selected_keys / max(1, self.total_sparse_queries)
        return {
            "sparse_prefill_calls": self.sparse_prefill_calls,
            "sparse_decode_calls": self.sparse_decode_calls,
            "exact_short_calls": self.exact_short_calls,
            "local_sliding_calls": self.local_sliding_calls,
            "dense_fallback_calls": self.dense_fallback_calls,
            "total_selected_keys": self.total_selected_keys,
            "total_sparse_queries": self.total_sparse_queries,
            "mean_selected_keys_per_query": round(mean_sel, 2),
        }


def _expand_gqa_kv(
    key: "torch.Tensor", value: "torch.Tensor", num_q_heads: int
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    num_kv_heads = int(key.shape[1])
    if num_q_heads == num_kv_heads:
        return key, value
    groups = num_q_heads // num_kv_heads
    return key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)


def local_sliding_attention_forward(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    q_positions: "torch.Tensor",
    k_positions: "torch.Tensor",
    sliding_window: int = 512,
    scaling: Optional[float] = None,
    tile_q: int = 256,
) -> "torch.Tensor":
    """Offset-aware local sliding-window attention (`[B, H_q, Q, D]`).

    Uses explicit absolute `q_positions` (`[Q]`) and `k_positions` (`[K]`) so ring-buffer eviction on `LocalSlidingRingSlot`
    never misaligns columns, and chunked prefill produces bit-identical results across `511/512/513` boundaries."""
    bsz, h_q, q_len, head_dim = query.shape
    scale = float(scaling) if scaling is not None else (1.0 / math.sqrt(head_dim))
    w = max(1, int(sliding_window))
    dev = query.device
    q_pos = q_positions.to(device=dev, dtype=torch.long).reshape(-1)
    k_pos = k_positions.to(device=dev, dtype=torch.long).reshape(-1)
    k_exp, v_exp = _expand_gqa_kv(key, value, h_q)

    outs: List["torch.Tensor"] = []
    step = max(1, int(tile_q))
    for s_i in range(0, q_len, step):
        e_i = min(q_len, s_i + step)
        q_tile = query[:, :, s_i:e_i, :]
        qp_tile = q_pos[s_i:e_i].unsqueeze(1)  # [Tq, 1]
        kp_row = k_pos.unsqueeze(0)             # [1, Tk]
        diff = qp_tile - kp_row
        valid = (diff >= 0) & (diff < w)        # [Tq, Tk]
        outs.append(
            F.scaled_dot_product_attention(
                q_tile,
                k_exp,
                v_exp,
                attn_mask=valid.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
        )
    return torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]


@torch.no_grad()
def _select_for_single_query(
    indexer: "FaithfulDLSAIndexer",
    idx_obj: "PagedLatentIndex",
    q_single: "torch.Tensor",
    q_abs_pos: int,
    budget: int,
    slot: Optional["GlobalPagedVisitSlot"] = None,
) -> "torch.Tensor":
    """Select exact causal key positions `<= q_abs_pos` for a single query `q_single` (`[H_q, 1, D]`)."""
    upto = int(q_abs_pos) + 1
    if upto <= indexer.n_sink + indexer.window + budget:
        return torch.arange(0, upto, device=q_single.device, dtype=torch.long)

    dummy_k = torch.zeros((indexer.n_kv, 1, indexer.head_dim), device=q_single.device, dtype=q_single.dtype)
    q_in, _ = indexer.inputs(q_single, dummy_k)
    q_proj = indexer.project_q(q_in)

    def _cand_fn(mid_indices: "torch.Tensor") -> "torch.Tensor":
        if slot is not None:
            k_cand, _ = slot.gather_selected(mid_indices, device=q_single.device, dtype=q_single.dtype)
            k0_c = k_cand[0]
            _, k_in_c = indexer.inputs(_dummy_q(k0_c), k0_c)
            return indexer.project_k(k_in_c)
        raise RuntimeError("candidate_key_fn requires a GlobalPagedVisitSlot when persistent_token_latents=False")

    return idx_obj.select(
        q_proj,
        budget=budget,
        upto=upto,
        dist_alpha_over_tau=indexer.dist_alpha_over_tau(),
        search_mode=indexer.search_mode,
        candidate_key_fn=_cand_fn if (slot is not None and not idx_obj.persistent_token_latents and indexer.rerank > 1) else None,
    )


def paged_dlsa_global_attention_forward(
    query: "torch.Tensor",
    slot: "GlobalPagedVisitSlot",
    indexer: "FaithfulDLSAIndexer",
    q_positions: "torch.Tensor",
    budget: Optional[int] = None,
    scaling: Optional[float] = None,
    query_tile_tokens: int = 16,
    telemetry: Optional[SparseRuntimeTelemetry] = None,
) -> "torch.Tensor":
    """Execute strictly causal D-LSA sparse attention against a `GlobalPagedVisitSlot` (both prefill and decode).

    - Short sequences (`upto <= n_sink + window + budget`): gathers all causal keys `[0..upto)` from `slot` and runs
      exact causal attention (Step 1 of §5.4).
    - Single-query decode (`q_len == 1`, Step 2–3 of §5.4): selects `sel` for `upto = q_pos + 1`, gathers `(k_sel, v_sel)`
      from `slot`, and computes attention over `sel`.
    - Multi-query prefill (`q_len > 1`, Step 4 of §5.4): tiles queries in blocks of `query_tile_tokens = 16`. Each query `i`
      at absolute position `p_i` computes its own strictly causal selection `S_i` from `upto = p_i + 1`. The tile union
      `S_union` is gathered once from `slot` and masked per query by `(k_pos in S_i) & (k_pos <= p_i)` so future tokens
      can never leak into earlier queries' selection or attention."""
    bsz, h_q, q_len, head_dim = query.shape
    if bsz != 1:
        raise ValueError(f"paged_dlsa_global_attention_forward currently expects batch_size=1, got {bsz}")
    scale = float(scaling) if scaling is not None else (1.0 / math.sqrt(head_dim))
    eff_budget = int(budget if budget is not None else indexer.k_long_range)
    dev = query.device
    q_pos = q_positions.to(device=dev, dtype=torch.long).reshape(-1)
    max_upto = int(q_pos[-1].item()) + 1 if q_pos.numel() > 0 else slot.logical_seq_length

    idx_obj: Optional[PagedLatentIndex] = slot.dlsa_state_slot.get("index")

    # Fast path 1: Sequence is within the exact budget threshold -> gather all `[0..max_upto)` and apply exact causal mask
    if max_upto <= indexer.n_sink + indexer.window + eff_budget or idx_obj is None:
        if telemetry is not None:
            telemetry.exact_short_calls += 1
        all_pos = torch.arange(0, max_upto, device=dev, dtype=torch.long)
        k_all, v_all = slot.gather_selected(all_pos, device=dev, dtype=query.dtype)
        k_exp, v_exp = _expand_gqa_kv(k_all, v_all, h_q)
        if q_len == 1:
            return F.scaled_dot_product_attention(
                query, k_exp, v_exp, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
            )
        causal_mask = all_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [q_len, max_upto]
        return F.scaled_dot_product_attention(
            query,
            k_exp,
            v_exp,
            attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )

    # Path 2: Single-query sparse decode (`q_len == 1`)
    if q_len == 1:
        p0 = int(q_pos[0].item())
        sel = _select_for_single_query(indexer, idx_obj, query[0], p0, eff_budget, slot=slot)
        slot.sparse_Step_calls += 1
        if telemetry is not None:
            telemetry.sparse_decode_calls += 1
            telemetry.total_selected_keys += int(sel.numel())
            telemetry.total_sparse_queries += 1
        k_sel, v_sel = slot.gather_selected(sel, device=dev, dtype=query.dtype)
        k_exp, v_exp = _expand_gqa_kv(k_sel, v_sel, h_q)
        return F.scaled_dot_product_attention(
            query, k_exp, v_exp, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
        )

    # Path 3: Tiled multi-query strictly causal sparse prefill (`q_len > 1`)
    if telemetry is not None:
        telemetry.sparse_prefill_calls += 1
    slot.sparse_Step_calls += 1
    tile_sz = max(1, int(query_tile_tokens))
    outs: List["torch.Tensor"] = []

    for s_i in range(0, q_len, tile_sz):
        e_i = min(q_len, s_i + tile_sz)
        t_len = e_i - s_i
        q_tile = query[:, :, s_i:e_i, :]
        sels_tile: List["torch.Tensor"] = []
        for local_i in range(t_len):
            p_abs = int(q_pos[s_i + local_i].item())
            sel_i = _select_for_single_query(
                indexer, idx_obj, query[0, :, s_i + local_i : s_i + local_i + 1, :], p_abs, eff_budget, slot=slot
            )
            sels_tile.append(sel_i)
            if telemetry is not None:
                telemetry.total_selected_keys += int(sel_i.numel())
                telemetry.total_sparse_queries += 1

        union_pos = torch.cat(sels_tile, dim=0).unique()  # sorted ascending `[U]`
        u_len = int(union_pos.numel())
        # Build exact `[t_len, u_len]` boolean mask where row `local_i` is True iff `union_pos[j] in sel_i` and `<= p_abs`
        allow_mask = torch.zeros((t_len, u_len), dtype=torch.bool, device=dev)
        for local_i, sel_i in enumerate(sels_tile):
            idx_in_u = torch.searchsorted(union_pos, sel_i)
            idx_in_u = idx_in_u.clamp(max=max(0, u_len - 1))
            matched = union_pos.index_select(0, idx_in_u) == sel_i
            allow_mask[local_i, idx_in_u[matched]] = True

        k_u, v_u = slot.gather_selected(union_pos, device=dev, dtype=query.dtype)
        k_exp, v_exp = _expand_gqa_kv(k_u, v_u, h_q)
        outs.append(
            F.scaled_dot_product_attention(
                q_tile,
                k_exp,
                v_exp,
                attn_mask=allow_mask.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
        )
    return torch.cat(outs, dim=2)


def compute_yarn_partial_rope_cos_sin(
    positions: "torch.Tensor",
    head_dim: int = 256,
    rope_theta: float = 5_000_000.0,
    partial_rotary_factor: float = 0.25,
    scale_factor: float = 1.0,
    orig_max_position: int = 1_048_576,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: Any = "cpu",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """YaRN-style frequency-ramp rotary embeddings (`arXiv:2309.00071`, Work Package P8) for Spark's 64-D rotated global slice.

    When `scale_factor <= 1.0`, matches standard FP32 RoPE bit-for-bit.
    When `scale_factor > 1.0` (e.g., `5_000_000 / 1_048_576 = 4.76837`), interpolates low-frequency dimensions while
    keeping high-frequency dimensions unstretched and applies the YaRN amplitude temperature `mscale = 0.1 * ln(s) + 1.0`."""
    rope_dim = int(round(head_dim * partial_rotary_factor))
    idx = torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
    inv_freq_extrap = 1.0 / (float(rope_theta) ** (idx / float(rope_dim)))

    if float(scale_factor) <= 1.0:
        inv_freq = inv_freq_extrap
        mscale = 1.0
    else:
        s = float(scale_factor)
        inv_freq_interp = inv_freq_extrap / s
        # Compute wavelength ramp boundaries in dimension index space
        low = max(
            0.0,
            float(rope_dim)
            * math.log(float(orig_max_position) / (float(beta_fast) * 2.0 * math.pi))
            / (2.0 * math.log(float(rope_theta))),
        )
        high = min(
            float(rope_dim // 2 - 1),
            float(rope_dim)
            * math.log(float(orig_max_position) / (float(beta_slow) * 2.0 * math.pi))
            / (2.0 * math.log(float(rope_theta))),
        )
        dim_idx = torch.arange(0, rope_dim // 2, dtype=torch.float32, device=device)
        ramp = ((dim_idx - low) / max(1e-5, high - low)).clamp(0.0, 1.0)
        smooth = 1.0 - ramp  # 1 = high-freq (keep extrapolated), 0 = low-freq (interpolate by 1/s)
        inv_freq = inv_freq_interp * (1.0 - smooth) + inv_freq_extrap * smooth
        mscale = 0.1 * math.log(s) + 1.0

    t = positions.to(device=device, dtype=torch.float32).reshape(-1)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * mscale, emb.sin() * mscale


def attach_loop72_dlsa_attention(
    loop_model: Any,
    index_bank: Optional["VisitAwareDLSAIndexBank"] = None,
    sparse_cfg: Optional[SparseAttentionConfig] = None,
) -> Dict[str, Any]:
    """Attach a per-visit `VisitAwareDLSAIndexBank` and per-instance paged D-LSA attention dispatch to `loop_model`.

    Does NOT mutate global class methods when wrapping each physical layer's `self_attn`, preserving process isolation."""
    cfg = sparse_cfg or SparseAttentionConfig()
    base = getattr(loop_model, "base", loop_model)
    layers = base.model.layers
    device = next(base.parameters()).device

    if index_bank is None:
        schedule = loop_model.visit_schedule_summary()["schedule"] if hasattr(loop_model, "visit_schedule_summary") else None
        index_bank = VisitAwareDLSAIndexBank(
            maps_per_global_visit=cfg.maps_per_global_visit,
            d_idx=cfg.index_dim,
            n_sink=cfg.sinks,
            n_pool=cfg.pool_size,
            n_cluster=cfg.cluster_block_tokens,
            avg_bucket_size=cfg.average_bucket_tokens,
            max_bucket_size=cfg.max_bucket_tokens,
            window=cfg.recent_tokens,
            k_long_range=cfg.long_range_tokens,
            rerank=cfg.rerank,
            search_mode=cfg.search,
            visit_schedule=schedule,
        ).to(device=device)

    loop_model.dlsa_index_bank = index_bank
    loop_model.sparse_attention_cfg = cfg
    telemetry = SparseRuntimeTelemetry()
    loop_model.sparse_telemetry = telemetry

    attn_mod = __import__(type(layers[0].self_attn).__module__, fromlist=["apply_rotary_pos_emb"])
    apply_rope_fn = getattr(attn_mod, "apply_rotary_pos_emb", None)
    default_sw = int(getattr(getattr(base, "config", None), "sliding_window", 512) or 512)

    def _make_instance_attn_forward(phys_idx: int, is_sliding: bool):
        def _instance_forward(
            self_attn: nn.Module,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Any] = None,
            cache_position: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            bsz, q_len = int(input_shape[0]), int(input_shape[1])
            h_dim = int(getattr(self_attn, "head_dim", 256) or 256)
            n_q = int(getattr(self_attn, "num_heads", 16) or 16)
            n_kv = int(getattr(self_attn, "num_key_value_heads", 4) or 4)
            scaling = float(getattr(self_attn, "scaling", 1.0 / math.sqrt(h_dim)))

            qkv = self_attn.q_k_v_proj(hidden_states)
            q_dim = n_q * h_dim
            kv_dim = n_kv * h_dim
            query_states, key_states, value_states = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
            query_states = query_states.view(bsz, q_len, n_q, h_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, n_kv, h_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, n_kv, h_dim).transpose(1, 2)

            cos, sin = position_embeddings
            rot_dim = int(cos.shape[-1])
            if apply_rope_fn is not None:
                q_rot = apply_rope_fn(query_states[..., :rot_dim], cos, sin).to(query_states.dtype)
                k_rot = apply_rope_fn(key_states[..., :rot_dim], cos, sin).to(key_states.dtype)
                if rot_dim < h_dim:
                    query_states = torch.cat([q_rot, query_states[..., rot_dim:]], dim=-1)
                    key_states = torch.cat([k_rot, key_states[..., rot_dim:]], dim=-1)
                else:
                    query_states, key_states = q_rot, k_rot

            # Check if `past_key_values` is a `_VisitCache` wrapping `Loop72HybridPagedCache`
            raw_cache = getattr(past_key_values, "cache", past_key_values)
            v_idx = int(getattr(past_key_values, "virtual_idx", phys_idx)) if past_key_values is not None else phys_idx
            read_only = bool(getattr(past_key_values, "read_only", False))

            if isinstance(raw_cache, Loop72HybridPagedCache):
                slot = raw_cache.get_visit_slot(v_idx)
                if cache_position is not None and cache_position.numel() == q_len:
                    q_pos = cache_position.to(device=hidden_states.device, dtype=torch.long)
                else:
                    s_p = slot.logical_seq_length
                    q_pos = torch.arange(s_p, s_p + q_len, device=hidden_states.device, dtype=torch.long)

                if isinstance(slot, LocalSlidingRingSlot):
                    telemetry.local_sliding_calls += 1
                    if not read_only:
                        k_win, v_win = slot.update(key_states, value_states, cache_position=q_pos)
                    else:
                        k_win, v_win = slot.keys, slot.values
                    k_pos = slot.last_returned_positions if slot.last_returned_positions is not None else q_pos
                    sw = int(getattr(self_attn, "_sparkx_sliding_window", None) or getattr(self_attn, "sliding_window", None) or default_sw)
                    attn_out = local_sliding_attention_forward(
                        query_states, k_win, v_win, q_positions=q_pos, k_positions=k_pos, sliding_window=sw, scaling=scaling
                    )
                else:
                    assert isinstance(slot, GlobalPagedVisitSlot)
                    ix = index_bank.get_indexer(
                        phys_idx,
                        virtual_visit=v_idx,
                        layer_visit_ordinal=slot.layer_visit_ordinal,
                        global_visit_id=slot.global_visit_id,
                    )
                    visit_budget = index_bank.get_visit_budget(
                        phys_idx,
                        virtual_visit=v_idx,
                        layer_visit_ordinal=slot.layer_visit_ordinal,
                        global_visit_id=slot.global_visit_id,
                    )
                    if not read_only:
                        slot.append(key_states, value_states, cache_position=q_pos, indexer=ix)
                    attn_out = paged_dlsa_global_attention_forward(
                        query_states,
                        slot=slot,
                        indexer=ix,
                        q_positions=q_pos,
                        budget=visit_budget,
                        scaling=scaling,
                        query_tile_tokens=cfg.sparse_query_tile_tokens,
                        telemetry=telemetry,
                    )
            else:
                # Uncached or standard cache path
                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, phys_idx, {"cache_position": cache_position}
                    )
                seq_k = int(key_states.shape[-2])
                if cache_position is not None and cache_position.numel() == q_len:
                    q_pos = cache_position.to(device=hidden_states.device, dtype=torch.long)
                else:
                    q_pos = torch.arange(seq_k - q_len, seq_k, device=hidden_states.device, dtype=torch.long)

                if is_sliding:
                    telemetry.local_sliding_calls += 1
                    sw = int(getattr(self_attn, "_sparkx_sliding_window", None) or getattr(self_attn, "sliding_window", None) or default_sw)
                    k_pos = torch.arange(max(0, seq_k - int(key_states.shape[-2])), seq_k, device=hidden_states.device, dtype=torch.long)
                    attn_out = local_sliding_attention_forward(
                        query_states, key_states, value_states, q_positions=q_pos, k_positions=k_pos, sliding_window=sw, scaling=scaling
                    )
                else:
                    # Uncached global visit: route through a transient `GlobalPagedVisitSlot` so sparse causality & per-visit D-LSA still run!
                    v_ord = int(getattr(self_attn, "_active_visit_ordinal", 0))
                    ix = index_bank.get_indexer(phys_idx, virtual_visit=v_idx, layer_visit_ordinal=v_ord)
                    visit_budget = index_bank.get_visit_budget(phys_idx, virtual_visit=v_idx, layer_visit_ordinal=v_ord)
                    gid = index_bank.resolve_global_visit_id(phys_idx, virtual_visit=v_idx, layer_visit_ordinal=v_ord)
                    from sparkx_paged_cache import PagedCacheConfig
                    tmp_slot = GlobalPagedVisitSlot(
                        virtual_visit=v_idx,
                        physical_layer=phys_idx,
                        pass_idx=0,
                        global_visit_id=gid,
                        layer_visit_ordinal=v_ord,
                        cfg=PagedCacheConfig(
                            sinks=cfg.sinks,
                            recent_tokens=cfg.recent_tokens,
                            global_hot_tokens_per_visit=max(8192, seq_k),
                        ),
                    )
                    tmp_slot.append(key_states, value_states, cache_position=torch.arange(seq_k, device=hidden_states.device), indexer=ix)
                    attn_out = paged_dlsa_global_attention_forward(
                        query_states,
                        slot=tmp_slot,
                        indexer=ix,
                        q_positions=q_pos,
                        budget=visit_budget,
                        scaling=scaling,
                        query_tile_tokens=cfg.sparse_query_tile_tokens,
                        telemetry=telemetry,
                    )

            # Apply Spark's native head-wise sigmoid output gate (`sigmoid(g_proj(LN h))`) before `out_proj`
            attn_out = attn_out.transpose(1, 2).contiguous()  # [B, Q, H_q, D]
            if hasattr(self_attn, "g_proj") and self_attn.g_proj is not None:
                gate = torch.sigmoid(self_attn.g_proj(hidden_states)).unsqueeze(-1)  # [B, Q, H_q, 1]
                attn_out = attn_out * gate.to(attn_out.dtype)
            attn_out = attn_out.reshape(bsz, q_len, n_q * h_dim)
            out = self_attn.out_proj(attn_out)
            return out, None

        return _instance_forward

    for idx_l, lyr in enumerate(layers):
        is_sl = (
            getattr(lyr, "layer_type", "") == "sliding_attention"
            or bool(getattr(lyr.self_attn, "sliding_window", None))
            or bool(getattr(lyr.self_attn, "is_sliding", False))
        )
        if not hasattr(lyr.self_attn, "_sparkx_orig_forward"):
            lyr.self_attn._sparkx_orig_forward = lyr.self_attn.forward
        lyr.self_attn.forward = types.MethodType(_make_instance_attn_forward(idx_l, is_sl), lyr.self_attn)
        lyr.self_attn._sparkx_loop72_dlsa_attached = True

    return {
        "status": "ok",
        "attached_layers": len(layers),
        "index_bank": index_bank.parameter_manifest(),
    }
