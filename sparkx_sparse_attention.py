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
import inspect
import math
import types
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    try:
        _SDPA_SUPPORTS_ENABLE_GQA = "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
    except Exception:
        _SDPA_SUPPORTS_ENABLE_GQA = False
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False
    _SDPA_SUPPORTS_ENABLE_GQA = False

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
    exact_threshold: int = 4612
    gpu_resident_max_tokens: int = 8192
    local_window: int = 512
    rerank: int = 1
    sparse_query_tile_tokens: int = 16
    selection_reuse: bool = False
    enable_native_gqa: bool = True
    residual_compensation: bool = False
    residual_block_compensation: Optional[bool] = None
    unbounded_dense_fallback: bool = False
    yarn_scale_factor: float = 1.0
    yarn_orig_max_position: int = 1048576
    yarn_original_max_position: Optional[int] = None
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    def __post_init__(self) -> None:
        if self.residual_block_compensation is not None:
            self.residual_compensation = bool(self.residual_block_compensation)
        else:
            self.residual_block_compensation = bool(self.residual_compensation)
        if self.yarn_original_max_position is not None:
            self.yarn_orig_max_position = int(self.yarn_original_max_position)
        else:
            self.yarn_original_max_position = int(self.yarn_orig_max_position)


@dataclasses.dataclass
class SparseRuntimeTelemetry:
    sparse_prefill_calls: int = 0
    sparse_decode_calls: int = 0
    exact_short_calls: int = 0
    local_sliding_calls: int = 0
    dense_fallback_calls: int = 0
    selection_reuse_hits: int = 0
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
            "selection_reuse_hits": self.selection_reuse_hits,
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


def _sdpa_gqa(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    attn_mask: Optional["torch.Tensor"] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_native_gqa: bool = True,
) -> "torch.Tensor":
    """Execute `F.scaled_dot_product_attention` using native `enable_gqa=True` when supported by PyTorch, avoiding
    4x KV head `repeat_interleave` allocations (`~1.55 GiB/token` saved across 72 visits)."""
    h_q = int(query.shape[1])
    h_kv = int(key.shape[1])
    if h_q != h_kv and enable_native_gqa and _SDPA_SUPPORTS_ENABLE_GQA:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=True,
        )
    k_exp, v_exp = _expand_gqa_kv(key, value, h_q)
    return F.scaled_dot_product_attention(
        query,
        k_exp,
        v_exp,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )


def local_sliding_attention_forward(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    q_positions: "torch.Tensor",
    k_positions: "torch.Tensor",
    sliding_window: int = 512,
    scaling: Optional[float] = None,
    tile_q: int = 256,
    window_already_bounded: bool = False,
    enable_native_gqa: bool = True,
) -> "torch.Tensor":
    """Offset-aware local sliding-window attention (`[B, H_q, Q, D]`).

    Uses explicit absolute `q_positions` (`[Q]`) and `k_positions` (`[K]`) so ring-buffer eviction on `LocalSlidingRingSlot`
    never misaligns columns, and chunked prefill produces bit-identical results across `511/512/513` boundaries.
    During single-token decode (`q_len == 1`) when `window_already_bounded=True` and `K <= W`, skips mask construction
    and dispatches directly to `_sdpa_gqa(..., attn_mask=None)`."""
    bsz, h_q, q_len, head_dim = query.shape
    scale = float(scaling) if scaling is not None else (1.0 / math.sqrt(head_dim))
    w = max(1, int(sliding_window))
    if q_len == 1 and window_already_bounded and int(key.shape[-2]) <= w:
        return _sdpa_gqa(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=False,
            scale=scale,
            enable_native_gqa=enable_native_gqa,
        )

    dev = query.device
    q_pos = q_positions.to(device=dev, dtype=torch.long).reshape(-1)
    k_pos = k_positions.to(device=dev, dtype=torch.long).reshape(-1)

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
            _sdpa_gqa(
                q_tile,
                key,
                value,
                attn_mask=valid.unsqueeze(0).unsqueeze(0),
                is_causal=False,
                scale=scale,
                enable_native_gqa=enable_native_gqa,
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
    selection_reuse: bool = False,
    telemetry: Optional[SparseRuntimeTelemetry] = None,
) -> "torch.Tensor":
    """Select exact causal key positions `<= q_abs_pos` for a single query `q_single` (`[H_q, 1, D]`)."""
    upto = int(q_abs_pos) + 1
    if upto <= indexer.n_sink + indexer.window + budget:
        return torch.arange(0, upto, device=q_single.device, dtype=torch.long)

    q_in = indexer._group_free_query_features(q_single)
    q_proj = indexer.project_q(q_in)

    # Optional query-coherence selection reuse across consecutive single-token decode steps
    max_reuse = max(2 if selection_reuse else 1, int(getattr(indexer, "selection_reuse_steps", 1)))
    if slot is not None and max_reuse > 1:
        last_q = slot.dlsa_state_slot.get("last_q_proj")
        last_sel = slot.dlsa_state_slot.get("last_sel")
        reuse_cnt = int(slot.dlsa_state_slot.get("reuse_count", 0))
        if (
            isinstance(last_q, torch.Tensor)
            and isinstance(last_sel, torch.Tensor)
            and reuse_cnt + 1 < max_reuse
        ):
            cos_sim = float((q_proj.reshape(-1) * last_q.reshape(-1)).sum().item())
            if cos_sim >= float(getattr(indexer, "reuse_cos_threshold", 0.96)):
                slot.dlsa_state_slot["reuse_count"] = reuse_cnt + 1
                new_tail = torch.tensor([upto - 1], device=last_sel.device, dtype=torch.long)
                reused = torch.unique(torch.cat([last_sel[last_sel < upto], new_tail], dim=0), sorted=True)
                slot.dlsa_state_slot["last_sel"] = reused
                if telemetry is not None:
                    telemetry.selection_reuse_hits += 1
                return reused

    def _cand_fn(mid_indices: "torch.Tensor") -> "torch.Tensor":
        if slot is not None:
            k_cand, _ = slot.gather_selected(mid_indices, device=q_single.device, dtype=q_single.dtype, return_positions=False)
            k0_c = k_cand[0]
            _, k_in_c = indexer.inputs(_dummy_q(k0_c), k0_c)
            return indexer.project_k(k_in_c)
        raise RuntimeError("candidate_key_fn requires a GlobalPagedVisitSlot when persistent_token_latents=False")

    sel = idx_obj.select(
        q_proj,
        budget=budget,
        upto=upto,
        dist_alpha_over_tau=indexer.dist_alpha_over_tau(),
        search_mode=indexer.search_mode,
        candidate_key_fn=_cand_fn if (slot is not None and not idx_obj.persistent_token_latents and indexer.rerank > 1) else None,
    )
    if slot is not None and max_reuse > 1:
        slot.dlsa_state_slot["last_q_proj"] = q_proj.detach()
        slot.dlsa_state_slot["last_sel"] = sel.detach()
        slot.dlsa_state_slot["reuse_count"] = 0
    return sel


def paged_dlsa_global_attention_forward(
    query: Optional["torch.Tensor"] = None,
    slot: Optional["GlobalPagedVisitSlot"] = None,
    indexer: Optional["FaithfulDLSAIndexer"] = None,
    q_positions: Optional["torch.Tensor"] = None,
    budget: Optional[int] = None,
    scaling: Optional[float] = None,
    query_tile_tokens: int = 16,
    telemetry: Optional[SparseRuntimeTelemetry] = None,
    *,
    module: Any = None,
    sparse_cfg: Optional[SparseAttentionConfig] = None,
    known_upto: Optional[int] = None,
) -> "torch.Tensor":
    """Execute strictly causal D-LSA sparse attention against a `GlobalPagedVisitSlot` (both prefill and decode).

    - Short sequences (`upto <= eff_exact_threshold`): uses zero-copy GPU resident slice when `slot.is_fully_gpu_resident(max_upto)`
      or gathers `[0..upto)` from `slot`, and runs exact causal attention with native GQA (Step 1 of §5.4).
    - Single-query decode (`q_len == 1`, Step 2–3 of §5.4): selects `sel` for `upto = q_pos + 1`, gathers `(k_sel, v_sel)`
      from `slot`, and computes attention over `sel`.
    - Multi-query prefill (`q_len > 1`, Step 4 of §5.4): tiles queries in blocks of `query_tile_tokens = 16`. Each query `i`
      at absolute position `p_i` computes its own strictly causal selection `S_i` from `upto = p_i + 1`. The tile union
      `S_union` is gathered once from `slot` and masked per query by `(k_pos in S_i) & (k_pos <= p_i)` so future tokens
      can never leak into earlier queries' selection or attention."""
    if query is None or slot is None or indexer is None:
        raise ValueError("paged_dlsa_global_attention_forward requires `query`, `slot`, and `indexer`")
    bsz, h_q, q_len, head_dim = query.shape
    if bsz != 1:
        raise ValueError(f"paged_dlsa_global_attention_forward currently expects batch_size=1, got {bsz}")
    scale = float(scaling) if scaling is not None else (1.0 / math.sqrt(head_dim))
    eff_budget = int(budget if budget is not None else indexer.k_long_range)
    if sparse_cfg is not None:
        query_tile_tokens = int(sparse_cfg.sparse_query_tile_tokens)
        enable_native_gqa = bool(sparse_cfg.enable_native_gqa)
        selection_reuse = bool(sparse_cfg.selection_reuse)
        eff_exact_threshold = (
            int(sparse_cfg.exact_threshold)
            if int(sparse_cfg.exact_threshold) > 0
            else (indexer.n_sink + indexer.window + eff_budget)
        )
    else:
        enable_native_gqa = True
        selection_reuse = False
        idx_cfg = getattr(indexer, "cfg", None)
        cfg_thresh = int(getattr(idx_cfg, "exact_threshold", 0) or 0)
        eff_exact_threshold = (
            min(cfg_thresh, indexer.n_sink + indexer.window + eff_budget)
            if cfg_thresh > 0
            else (indexer.n_sink + indexer.window + eff_budget)
        )

    dev = query.device
    if known_upto is not None and q_len == 1:
        max_upto = int(known_upto)
        q_pos = None
    elif q_positions is not None and q_positions.numel() > 0:
        q_pos = q_positions.to(device=dev, dtype=torch.long).reshape(-1)
        max_upto = int(q_pos[-1].item()) + 1
    else:
        max_upto = int(slot.logical_seq_length)
        q_pos = torch.arange(max(0, max_upto - q_len), max_upto, device=dev, dtype=torch.long)

    idx_obj: Optional[PagedLatentIndex] = slot.dlsa_state_slot.get("index")

    # Fast path 1: Sequence is within the exact budget threshold -> zero-copy GPU resident slice or gather `[0..max_upto)`
    if max_upto <= eff_exact_threshold or idx_obj is None:
        if telemetry is not None:
            telemetry.exact_short_calls += 1
        if slot.is_fully_gpu_resident(max_upto):
            k_all, v_all = slot.get_resident_prefix(max_upto)
        else:
            all_pos = torch.arange(0, max_upto, device=dev, dtype=torch.long)
            k_all, v_all = slot.gather_selected(all_pos, device=dev, dtype=query.dtype, return_positions=False)

        if q_len == 1:
            return _sdpa_gqa(
                query,
                k_all,
                v_all,
                attn_mask=None,
                is_causal=False,
                scale=scale,
                enable_native_gqa=enable_native_gqa,
            )
        if max_upto == q_len and q_pos is not None and int(q_pos[0].item()) == 0:
            return _sdpa_gqa(
                query,
                k_all,
                v_all,
                attn_mask=None,
                is_causal=True,
                scale=scale,
                enable_native_gqa=enable_native_gqa,
            )
        if q_pos is None:
            q_pos = torch.arange(max(0, max_upto - q_len), max_upto, device=dev, dtype=torch.long)
        all_pos = torch.arange(0, max_upto, device=dev, dtype=torch.long)
        causal_mask = all_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [q_len, max_upto]
        return _sdpa_gqa(
            query,
            k_all,
            v_all,
            attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),
            is_causal=False,
            scale=scale,
            enable_native_gqa=enable_native_gqa,
        )

    # Path 2: Single-query sparse decode (`q_len == 1`)
    if q_len == 1:
        p0 = max_upto - 1
        sel = _select_for_single_query(
            indexer,
            idx_obj,
            query[0],
            p0,
            eff_budget,
            slot=slot,
            selection_reuse=selection_reuse,
            telemetry=telemetry,
        )
        slot.sparse_Step_calls += 1
        if telemetry is not None:
            telemetry.sparse_decode_calls += 1
            telemetry.total_selected_keys += int(sel.numel())
            telemetry.total_sparse_queries += 1
        k_sel, v_sel = slot.gather_selected(sel, device=dev, dtype=query.dtype, return_positions=False)
        return _sdpa_gqa(
            query,
            k_sel,
            v_sel,
            attn_mask=None,
            is_causal=False,
            scale=scale,
            enable_native_gqa=enable_native_gqa,
        )

    # Path 3: Tiled multi-query strictly causal sparse prefill (`q_len > 1`)
    if telemetry is not None:
        telemetry.sparse_prefill_calls += 1
    slot.sparse_Step_calls += 1
    tile_sz = max(1, int(query_tile_tokens))
    outs: List["torch.Tensor"] = []
    if q_pos is None:
        q_pos = torch.arange(max(0, max_upto - q_len), max_upto, device=dev, dtype=torch.long)
    q_pos_list: List[int] = [int(x) for x in q_pos.tolist()]

    idx_obj._ensure_materialized()
    can_vectorize = (
        int(getattr(indexer, "rerank", 1)) == 1
        and str(getattr(indexer, "search_mode", "flat")).lower().strip() == "flat"
        and (idx_obj.n_blocks == 0 or (idx_obj._mat_bpools.numel() > 0 and int(idx_obj._mat_bpools.shape[0]) >= idx_obj.n_blocks))
        and all(q_pos_list[i] <= q_pos_list[i + 1] for i in range(len(q_pos_list) - 1))
    )

    if can_vectorize:
        vec_tile_sz = max(tile_sz, 64) if q_len >= 64 else tile_sz
        q_proj_all = indexer.project_q(indexer._group_free_query_features(query[0]))  # [q_len, d_idx]
        dist_alpha_tau = float(indexer.dist_alpha_over_tau())
        exact_cutoff = indexer.n_sink + indexer.window + eff_budget

        def _query_regime(pos_val: int) -> Tuple[bool, int]:
            u_val = min(pos_val + 1, idx_obj.t)
            if u_val <= exact_cutoff or idx_obj.n_blocks == 0:
                return True, 0
            lim = max(0, min(idx_obj.n_blocks, (u_val - idx_obj.window - idx_obj.mid_start) // idx_obj.n_cluster))
            if lim == 0:
                return True, 0
            return False, lim

        s_i = 0
        while s_i < q_len:
            is_short_s, limit_s = _query_regime(q_pos_list[s_i])
            if is_short_s:
                # Coalesce the entire contiguous exact-prefix span into a single causal SDPA call
                e_i = s_i + 1
                while e_i < q_len and _query_regime(q_pos_list[e_i])[0]:
                    e_i += 1
                t_len = e_i - s_i
                q_tile = query[:, :, s_i:e_i, :]
                q_pos_tile = q_pos[s_i:e_i]
                span_upto = min(q_pos_list[e_i - 1] + 1, idx_obj.t)
                if slot.is_fully_gpu_resident(span_upto):
                    k_span, v_span = slot.get_resident_prefix(span_upto)
                else:
                    all_span_pos = torch.arange(0, span_upto, device=dev, dtype=torch.long)
                    k_span, v_span = slot.gather_selected(all_span_pos, device=dev, dtype=query.dtype, return_positions=False)
                k_span_pos = torch.arange(0, span_upto, device=dev, dtype=torch.long)
                allow_mask = k_span_pos.unsqueeze(0) <= q_pos_tile.unsqueeze(1)  # [t_len, span_upto]
                if telemetry is not None:
                    telemetry.total_selected_keys += int(allow_mask.sum().item())
                    telemetry.total_sparse_queries += t_len
                outs.append(
                    _sdpa_gqa(
                        q_tile,
                        k_span,
                        v_span,
                        attn_mask=allow_mask.unsqueeze(0).unsqueeze(0),
                        is_causal=False,
                        scale=scale,
                        enable_native_gqa=enable_native_gqa,
                    )
                )
                s_i = e_i
                continue

            # Vectorized strictly-causal sparse tile with constant Ward block horizon `limit_s > 0`
            e_i = s_i + 1
            max_e = min(q_len, s_i + vec_tile_sz)
            while e_i < max_e and _query_regime(q_pos_list[e_i]) == (False, limit_s):
                e_i += 1
            t_len = e_i - s_i
            q_tile = query[:, :, s_i:e_i, :]
            q_pos_tile = q_pos[s_i:e_i]
            q_proj_tile = q_proj_all[s_i:e_i]
            max_tile_upto = min(q_pos_list[e_i - 1] + 1, idx_obj.t)
            clustered_end = idx_obj.mid_start + limit_s * idx_obj.n_cluster

            cents_lim = idx_obj.centroids[:limit_s].to(dev)
            toks_lim = idx_obj.bucket_tokens[:limit_s].to(dev)
            c_slots = int(cents_lim.shape[1])
            b_tot = limit_s * c_slots

            scores = torch.matmul(q_proj_tile, cents_lim.reshape(b_tot, -1).t())  # [t_len, b_tot]
            if dist_alpha_tau > 0.0:
                blk_centers = idx_obj.mid_start + (torch.arange(limit_s, device=dev, dtype=torch.float32) + 0.5) * float(idx_obj.n_cluster)
                blk_pen = dist_alpha_tau * torch.log1p(
                    (q_pos_tile.float().unsqueeze(1) - blk_centers.unsqueeze(0)).clamp(min=0.0) / max(1.0, float(idx_obj.window))
                )
                scores = (scores.view(t_len, limit_s, c_slots) - blk_pen.unsqueeze(2)).reshape(t_len, b_tot)

            toks_flat = toks_lim.reshape(b_tot)
            scores = scores.masked_fill(toks_flat.unsqueeze(0) == 0, float("-inf"))
            stage1_budget = int(eff_budget)
            max_cand_buckets = min(b_tot, max(16, stage1_budget // max(1, idx_obj.n_pool) + 8))
            flat_s, flat_i = torch.topk(scores, k=max_cand_buckets, dim=-1, largest=True, sorted=True)
            flat_t = toks_flat.unsqueeze(0).expand(t_len, b_tot).gather(1, flat_i)
            csum = torch.cumsum(flat_t, dim=-1)
            take = ((csum - flat_t) < stage1_budget) & torch.isfinite(flat_s)

            b_selected_mask = torch.zeros((t_len, b_tot), dtype=torch.bool, device=dev)
            b_selected_mask.scatter_(1, flat_i, take)
            union_buckets = torch.nonzero(b_selected_mask.any(dim=0)).squeeze(1)

            if union_buckets.numel() > 0:
                bpools_flat = idx_obj._mat_bpools[:limit_s].reshape(b_tot, idx_obj.max_pools_per_bucket).to(dev)
                u_pool_bases = bpools_flat.index_select(0, union_buckets)
                ub_idx, slot_idx = torch.nonzero(u_pool_bases >= 0, as_tuple=True)
                valid_bases = u_pool_bases[ub_idx, slot_idx]
                pool_bucket_id = union_buckets.index_select(0, ub_idx)
                valid_bases, perm = torch.sort(valid_bases)
                pool_bucket_id = pool_bucket_id.index_select(0, perm)
                mid_tokens = (
                    valid_bases.unsqueeze(1) + torch.arange(idx_obj.n_pool, device=dev, dtype=torch.long).unsqueeze(0)
                ).reshape(-1)
                allow_mid_pools = b_selected_mask.index_select(1, pool_bucket_id)
                allow_mid = allow_mid_pools.unsqueeze(2).expand(t_len, -1, idx_obj.n_pool).reshape(t_len, -1)
            else:
                mid_tokens = torch.zeros(0, dtype=torch.long, device=dev)
                allow_mid = torch.zeros((t_len, 0), dtype=torch.bool, device=dev)

            sink_tokens = torch.arange(0, min(idx_obj.n_sink, max_tile_upto), device=dev, dtype=torch.long)
            tail_tokens = torch.arange(clustered_end, max_tile_upto, device=dev, dtype=torch.long)
            allow_sink = sink_tokens.unsqueeze(0) <= q_pos_tile.unsqueeze(1)
            allow_tail = tail_tokens.unsqueeze(0) <= q_pos_tile.unsqueeze(1)

            union_pos = torch.cat([sink_tokens, mid_tokens, tail_tokens], dim=0)
            allow_mask = torch.cat([allow_sink, allow_mid, allow_tail], dim=1)

            if telemetry is not None:
                telemetry.total_selected_keys += int(allow_mask.sum().item())
                telemetry.total_sparse_queries += t_len

            k_u, v_u = slot.gather_selected(union_pos, device=dev, dtype=query.dtype, return_positions=False)
            outs.append(
                _sdpa_gqa(
                    q_tile,
                    k_u,
                    v_u,
                    attn_mask=allow_mask.unsqueeze(0).unsqueeze(0),
                    is_causal=False,
                    scale=scale,
                    enable_native_gqa=enable_native_gqa,
                )
            )
            s_i = e_i

        return torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]

    for s_i in range(0, q_len, tile_sz):
        e_i = min(q_len, s_i + tile_sz)
        t_len = e_i - s_i
        q_tile = query[:, :, s_i:e_i, :]
        sels_tile: List["torch.Tensor"] = []
        for local_i in range(t_len):
            p_abs = q_pos_list[s_i + local_i]
            sel_i = _select_for_single_query(
                indexer,
                idx_obj,
                query[0, :, s_i + local_i : s_i + local_i + 1, :],
                p_abs,
                eff_budget,
                slot=slot,
                selection_reuse=False,
                telemetry=telemetry,
            )
            sels_tile.append(sel_i)
            if telemetry is not None:
                telemetry.total_selected_keys += int(sel_i.numel())
                telemetry.total_sparse_queries += 1

        union_pos = torch.cat(sels_tile, dim=0).unique()  # sorted ascending `[U]`
        u_len = int(union_pos.numel())
        allow_mask = torch.zeros((t_len, u_len), dtype=torch.bool, device=dev)
        for local_i, sel_i in enumerate(sels_tile):
            idx_in_u = torch.searchsorted(union_pos, sel_i)
            idx_in_u = idx_in_u.clamp(max=max(0, u_len - 1))
            matched = union_pos.index_select(0, idx_in_u) == sel_i
            allow_mask[local_i, idx_in_u[matched]] = True

        k_u, v_u = slot.gather_selected(union_pos, device=dev, dtype=query.dtype, return_positions=False)
        outs.append(
            _sdpa_gqa(
                q_tile,
                k_u,
                v_u,
                attn_mask=allow_mask.unsqueeze(0).unsqueeze(0),
                is_causal=False,
                scale=scale,
                enable_native_gqa=enable_native_gqa,
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
    loop_model.attention_runtime = 'paged_dlsa_inference'
    loop_model.attention_training_supported = False
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
            if torch.is_grad_enabled() and (self_attn.training or hidden_states.requires_grad):
                raise RuntimeError(
                    'Paged D-LSA attention uses detached inference KV storage and cannot train mHC. '
                    'Load with attach_sparse_attention=False and recurrence_mode="learned_mhc" '
                    'for differentiable native attention training.'
                )
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
            if not is_sliding and float(cfg.yarn_scale_factor) > 1.0:
                raw_cache_tmp = getattr(past_key_values, "cache", past_key_values)
                v_idx_tmp = int(getattr(past_key_values, "virtual_idx", phys_idx)) if past_key_values is not None else phys_idx
                if cache_position is not None and cache_position.numel() == q_len:
                    pos_for_yarn = cache_position.to(device=hidden_states.device, dtype=torch.long)
                elif isinstance(raw_cache_tmp, Loop72HybridPagedCache):
                    s_p_y = raw_cache_tmp.get_visit_slot(v_idx_tmp).logical_seq_length
                    pos_for_yarn = torch.arange(s_p_y, s_p_y + q_len, device=hidden_states.device, dtype=torch.long)
                else:
                    pos_for_yarn = torch.arange(0, q_len, device=hidden_states.device, dtype=torch.long)
                cos_y, sin_y = compute_yarn_partial_rope_cos_sin(
                    pos_for_yarn,
                    head_dim=h_dim,
                    rope_theta=5_000_000.0,
                    partial_rotary_factor=float(rot_dim) / float(h_dim),
                    scale_factor=float(cfg.yarn_scale_factor),
                    orig_max_position=int(cfg.yarn_orig_max_position),
                    beta_fast=float(cfg.yarn_beta_fast),
                    beta_slow=float(cfg.yarn_beta_slow),
                    device=hidden_states.device,
                )
                if cos.ndim == 3 and cos_y.ndim == 2:
                    cos = cos_y.unsqueeze(0).to(dtype=cos.dtype)
                    sin = sin_y.unsqueeze(0).to(dtype=sin.dtype)
                else:
                    cos = cos_y.to(dtype=cos.dtype)
                    sin = sin_y.to(dtype=sin.dtype)

            if apply_rope_fn is not None:
                q_rot = apply_rope_fn(query_states[..., :rot_dim], cos, sin).to(query_states.dtype)
                k_rot = apply_rope_fn(key_states[..., :rot_dim], cos, sin).to(key_states.dtype)
                while q_rot.ndim > query_states.ndim and q_rot.shape[0] == 1:
                    q_rot = q_rot.squeeze(0)
                while k_rot.ndim > key_states.ndim and k_rot.shape[0] == 1:
                    k_rot = k_rot.squeeze(0)
                if rot_dim < h_dim:
                    query_states = torch.cat([q_rot, query_states[..., rot_dim:]], dim=-1)
                    key_states = torch.cat([k_rot, key_states[..., rot_dim:]], dim=-1)
                else:
                    query_states, key_states = q_rot, k_rot

            if getattr(self_attn, "_capture_sleep_qkv", False):
                self_attn._sleep_last_qkv = (
                    query_states.detach(),
                    key_states.detach(),
                    value_states.detach(),
                )
                visit_qkv_log = getattr(self_attn, "_sleep_visit_qkv_list", None)
                if isinstance(visit_qkv_log, list):
                    visit_qkv_log.append(self_attn._sleep_last_qkv)

            # Check if `past_key_values` is a `_VisitCache` wrapping `Loop72HybridPagedCache`
            raw_cache = getattr(past_key_values, "cache", past_key_values)
            visit_context = getattr(self_attn, '_sparkx_active_visit', None)
            if visit_context is not None and int(visit_context[0]) != phys_idx:
                raise RuntimeError('Attention visit context belongs to a different physical layer')
            if past_key_values is not None:
                v_idx = int(getattr(past_key_values, 'virtual_idx', phys_idx))
            elif visit_context is not None:
                v_idx = int(visit_context[1])
            elif len(getattr(loop_model, 'execution_plan', layers)) > len(layers):
                raise RuntimeError('Uncached looped D-LSA requires explicit visit identity; call the loop model forward')
            else:
                v_idx = phys_idx
            read_only = bool(getattr(past_key_values, "read_only", False))

            if isinstance(raw_cache, Loop72HybridPagedCache):
                slot = raw_cache.get_visit_slot(v_idx)
                if cache_position is not None and cache_position.numel() == q_len:
                    q_pos = cache_position.to(device=hidden_states.device, dtype=torch.long)
                elif q_len == 1:
                    q_pos = None
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
                    if q_pos is None:
                        q_pos = k_pos[-1:] if k_pos is not None else torch.tensor([slot.logical_seq_length - 1], device=hidden_states.device, dtype=torch.long)
                    sw = int(getattr(self_attn, "_sparkx_sliding_window", None) or getattr(self_attn, "sliding_window", None) or default_sw)
                    attn_out = local_sliding_attention_forward(
                        query_states,
                        k_win,
                        v_win,
                        q_positions=q_pos,
                        k_positions=k_pos,
                        sliding_window=sw,
                        scaling=scaling,
                        window_already_bounded=(q_len == 1 and not read_only),
                        enable_native_gqa=cfg.enable_native_gqa,
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
                        sparse_cfg=cfg,
                        known_upto=slot.logical_seq_length if q_len == 1 else None,
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
                        query_states,
                        key_states,
                        value_states,
                        q_positions=q_pos,
                        k_positions=k_pos,
                        sliding_window=sw,
                        scaling=scaling,
                        enable_native_gqa=cfg.enable_native_gqa,
                    )
                else:
                    # Uncached global visit: route through a transient `GlobalPagedVisitSlot` so sparse causality & per-visit D-LSA still run!
                    v_ord = int(visit_context[2]) if visit_context is not None else 0
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
                        sparse_cfg=cfg,
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
