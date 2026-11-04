"""sparkx_dynamic_memory.py — Faithful D-LSA Paging (`LSA-F4`), Dynamic Latent Memory Graph (`LSA-F6`),
and Sleep-Time Vertical Memory Consolidation (`LSA-F7`).

All components operate strictly inside the layer's native `S^{d_idx - 1}` latent index space built by
`sparkx_dlsa.PagedLatentIndex` and `FaithfulDLSAIndexer`:
  - Zero `Layer 0` (`embed_tokens`) text re-embedding or prompt context-stuffing.
  - Equal-budget key selection semantics compatible with `sparkx_dlsa.evaluate_selectors`.
  - Grounded in primary-source formulations from:
      * Latent Sparse Attention (`arXiv:2610.42147`, Ward clustering in S^{63})
      * HippoRAG 2 (`arXiv:2502.14802`, Personalized PageRank spreading activation with damping)
      * A-MEM (`arXiv:2502.12110`, Zettelkasten cross-block semantic link evolution)
      * Zep / Graphiti (`arXiv:2501.13956`, bi-temporal `[t_valid, t_invalid)` edge & node supersession)
      * AlphaEdit (`arXiv:2410.02355`, null-space constrained weight updates) & WISE (`arXiv:2405.14768`, activation scope gate)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparkx_dlsa import (
    FaithfulDLSAIndexer,
    PagedLatentIndex,
    ward_partition,
)


# ======================================================================================
# LSA-F4: Hot/Cold Paged KV Store with LRU GPU Bucket Cache & Optional FP8 Cold Storage
# ======================================================================================
@dataclass
class ColdBucketRecord:
    """Compact CPU-resident KV slice for a single Ward bucket `(block_id, cluster_id)`."""
    block_id: int
    cluster_id: int
    token_positions: torch.Tensor        # [n_tok] int64 on CPU
    k_data: torch.Tensor                 # [H_kv, n_tok, D] (float16/bfloat16/float32 or float8_e4m3fn) on CPU
    v_data: torch.Tensor                 # [H_kv, n_tok, D] on CPU
    k_scale: Optional[torch.Tensor] = None  # [H_kv, 1, 1] float32 if FP8 quantized
    v_scale: Optional[torch.Tensor] = None  # [H_kv, 1, 1] float32 if FP8 quantized
    evicted: bool = False

    @property
    def num_tokens(self) -> int:
        return int(self.token_positions.numel())

    @property
    def nbytes(self) -> int:
        b = self.token_positions.numel() * self.token_positions.element_size()
        b += self.k_data.numel() * self.k_data.element_size()
        b += self.v_data.numel() * self.v_data.element_size()
        if self.k_scale is not None:
            b += self.k_scale.numel() * self.k_scale.element_size()
        if self.v_scale is not None:
            b += self.v_scale.numel() * self.v_scale.element_size()
        return int(b)


class HotColdPagedKVStore:
    """Two-tier KV store keyed by `(block_id, cluster_id)` Ward buckets from `PagedLatentIndex` (`LSA-F4`).

    - **GPU Hot Always-Resident**: `sink` (`[0:n_sink)`) and `tail` (`[clustered_end:T)`).
    - **GPU Hot Bucket Cache**: LRU cache holding up to `hot_bucket_capacity` active buckets on `hot_device`.
    - **Host Cold Store**: CPU memory (optionally quantized to `float8_e4m3fn` with per-head scales) holding all
      non-empty middle-region buckets.
    """

    def __init__(
        self,
        hot_bucket_capacity: int = 64,
        quantize_fp8: bool = False,
        hot_device: Optional[torch.device] = None,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        self.hot_bucket_capacity = max(1, int(hot_bucket_capacity))
        self.quantize_fp8 = bool(quantize_fp8) and hasattr(torch, "float8_e4m3fn")
        self.hot_device = hot_device or torch.device("cpu")
        self.compute_dtype = compute_dtype

        # Hot always-resident slices
        self.k_sink: Optional[torch.Tensor] = None
        self.v_sink: Optional[torch.Tensor] = None
        self.sink_positions: Optional[torch.Tensor] = None
        self.k_tail: Optional[torch.Tensor] = None
        self.v_tail: Optional[torch.Tensor] = None
        self.tail_positions: Optional[torch.Tensor] = None

        # Cold bucket store: (block_id, cluster_id) -> ColdBucketRecord
        self.cold_buckets: Dict[Tuple[int, int], ColdBucketRecord] = {}
        # Hot LRU cache: (block_id, cluster_id) -> (k_gpu, v_gpu, pos_gpu, last_step)
        self.hot_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        self._ingested_blocks: int = 0
        self._step_clock: int = 0
        self._total_seq_len: int = 0
        self._index_nbytes: int = 0

        # Telemetry counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.lru_evictions: int = 0
        self.bytes_transferred: int = 0
        self.evicted_buckets_count: int = 0
        self.evicted_tokens_count: int = 0

    @staticmethod
    def _to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        amax = xf.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        scale = amax / 240.0
        q = (xf / scale).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
        return q.cpu(), scale.cpu()

    @staticmethod
    def _from_fp8(q: torch.Tensor, scale: Optional[torch.Tensor], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if scale is None:
            return q.to(device=device, dtype=dtype)
        return (q.to(device=device, dtype=torch.float32) * scale.to(device=device, dtype=torch.float32)).to(dtype)

    @torch.no_grad()
    def ingest(self, index: PagedLatentIndex, k: torch.Tensor, v: torch.Tensor) -> int:
        """Incrementally ingest newly clustered blocks from `index` and update hot `sink` + `tail`.
        `k, v`: `[H_kv, T, D]`. Returns the number of newly ingested Ward blocks."""
        assert k.shape == v.shape and k.ndim == 3
        self.hot_device = k.device if self.hot_device is None else self.hot_device
        self.compute_dtype = k.dtype if self.compute_dtype is None else self.compute_dtype
        t = int(k.shape[1])
        self._total_seq_len = t
        self._index_nbytes = int(
            index.labels.numel() * index.labels.element_size()
            + index.centroids.numel() * index.centroids.element_size()
            + index.bucket_tokens.numel() * index.bucket_tokens.element_size()
        )

        s_end = min(index.n_sink, t)
        self.k_sink = k[:, :s_end].to(self.hot_device, dtype=self.compute_dtype).contiguous()
        self.v_sink = v[:, :s_end].to(self.hot_device, dtype=self.compute_dtype).contiguous()
        self.sink_positions = torch.arange(0, s_end, device=self.hot_device, dtype=torch.long)

        c_end = index.clustered_end if index.n_blocks > 0 else s_end
        self.k_tail = k[:, c_end:t].to(self.hot_device, dtype=self.compute_dtype).contiguous()
        self.v_tail = v[:, c_end:t].to(self.hot_device, dtype=self.compute_dtype).contiguous()
        self.tail_positions = torch.arange(c_end, t, device=self.hot_device, dtype=torch.long)

        new_blocks = max(0, index.n_blocks - self._ingested_blocks)
        if new_blocks == 0:
            return 0

        for b in range(self._ingested_blocks, index.n_blocks):
            labels_b = index.labels[b]  # [P]
            btoks_b = index.bucket_tokens[b]  # [C]
            nz_clusters = torch.nonzero(btoks_b > 0, as_tuple=False).reshape(-1).tolist()
            for c in nz_clusters:
                pool_ids = torch.nonzero(labels_b == c, as_tuple=False).reshape(-1)
                if pool_ids.numel() == 0:
                    continue
                base = index.mid_start + b * index.n_cluster + pool_ids * index.n_pool
                tok_pos = (base.unsqueeze(1) + torch.arange(index.n_pool, device=base.device).unsqueeze(0)).reshape(-1)
                tok_pos_cpu = tok_pos.to(dtype=torch.long, device="cpu")
                k_slice = k[:, tok_pos.to(k.device)]
                v_slice = v[:, tok_pos.to(v.device)]
                if self.quantize_fp8:
                    k_cpu, k_sc = self._to_fp8(k_slice)
                    v_cpu, v_sc = self._to_fp8(v_slice)
                else:
                    k_cpu, k_sc = k_slice.detach().cpu().contiguous(), None
                    v_cpu, v_sc = v_slice.detach().cpu().contiguous(), None
                self.cold_buckets[(b, int(c))] = ColdBucketRecord(
                    block_id=b,
                    cluster_id=int(c),
                    token_positions=tok_pos_cpu,
                    k_data=k_cpu,
                    v_data=v_cpu,
                    k_scale=k_sc,
                    v_scale=v_sc,
                )
        self._ingested_blocks = index.n_blocks
        return int(new_blocks)

    @torch.no_grad()
    def _fetch_bucket(self, key: Tuple[int, int]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        self._step_clock += 1
        if key in self.hot_cache:
            k_g, v_g, p_g, _ = self.hot_cache[key]
            self.hot_cache[key] = (k_g, v_g, p_g, self._step_clock)
            self.cache_hits += 1
            return k_g, v_g, p_g

        rec = self.cold_buckets.get(key)
        if rec is None or rec.evicted:
            return None
        self.cache_misses += 1
        dtype = self.compute_dtype or torch.float32
        dev = self.hot_device
        if self.quantize_fp8:
            k_g = self._from_fp8(rec.k_data, rec.k_scale, dtype=dtype, device=dev)
            v_g = self._from_fp8(rec.v_data, rec.v_scale, dtype=dtype, device=dev)
        else:
            k_g = rec.k_data.to(device=dev, dtype=dtype)
            v_g = rec.v_data.to(device=dev, dtype=dtype)
        p_g = rec.token_positions.to(device=dev, dtype=torch.long)
        self.bytes_transferred += rec.nbytes

        if len(self.hot_cache) >= self.hot_bucket_capacity:
            lru_key = min(self.hot_cache.keys(), key=lambda k_: self.hot_cache[k_][3])
            del self.hot_cache[lru_key]
            self.lru_evictions += 1
        self.hot_cache[key] = (k_g, v_g, p_g, self._step_clock)
        return k_g, v_g, p_g

    @torch.no_grad()
    def prefetch_buckets(self, bucket_ids: Sequence[Tuple[int, int]]) -> int:
        """Prefetch a list of `(block_id, cluster_id)` buckets into the GPU hot LRU cache."""
        fetched = 0
        for bid in bucket_ids:
            if self._fetch_bucket((int(bid[0]), int(bid[1]))) is not None:
                fetched += 1
        return fetched

    @torch.no_grad()
    def assemble_selected_kv(
        self,
        selected_buckets: Sequence[Tuple[int, int]],
        token_budget: Optional[int] = None,
        q_idx: Optional[torch.Tensor] = None,
        k_idx_n: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble `(k_selected, v_selected, token_positions)` sorted ascending by position from:
        hot `sink` + fetched `selected_buckets` (optionally token-reranked to `token_budget`) + hot `tail`."""
        assert self.k_sink is not None and self.k_tail is not None
        dev = self.hot_device
        k_mid_list: List[torch.Tensor] = []
        v_mid_list: List[torch.Tensor] = []
        p_mid_list: List[torch.Tensor] = []

        for b_id, c_id in selected_buckets:
            fetched = self._fetch_bucket((int(b_id), int(c_id)))
            if fetched is not None:
                k_g, v_g, p_g = fetched
                k_mid_list.append(k_g)
                v_mid_list.append(v_g)
                p_mid_list.append(p_g)

        if p_mid_list:
            k_mid = torch.cat(k_mid_list, dim=1)
            v_mid = torch.cat(v_mid_list, dim=1)
            p_mid = torch.cat(p_mid_list, dim=0)
            if token_budget is not None and p_mid.numel() > int(token_budget) and q_idx is not None and k_idx_n is not None:
                q_vec = F.normalize(q_idx.float().reshape(-1, q_idx.shape[-1]).mean(dim=0), dim=-1, eps=1e-6).to(dev)
                scores = k_idx_n.to(dev)[p_mid] @ q_vec
                top_i = scores.topk(int(token_budget)).indices
                k_mid = k_mid[:, top_i]
                v_mid = v_mid[:, top_i]
                p_mid = p_mid[top_i]
            order = torch.argsort(p_mid)
            k_mid = k_mid[:, order]
            v_mid = v_mid[:, order]
            p_mid = p_mid[order]
            k_out = torch.cat([self.k_sink, k_mid, self.k_tail], dim=1)
            v_out = torch.cat([self.v_sink, v_mid, self.v_tail], dim=1)
            p_out = torch.cat([self.sink_positions, p_mid, self.tail_positions], dim=0)
        else:
            k_out = torch.cat([self.k_sink, self.k_tail], dim=1)
            v_out = torch.cat([self.v_sink, self.v_tail], dim=1)
            p_out = torch.cat([self.sink_positions, self.tail_positions], dim=0)
        return k_out, v_out, p_out

    def evict_bucket_permanently(self, block_id: int, cluster_id: int) -> int:
        """Permanently evict a bucket after Sleep Consolidation (`LSA-F7`) into vertical memory."""
        key = (int(block_id), int(cluster_id))
        if key in self.hot_cache:
            del self.hot_cache[key]
        rec = self.cold_buckets.get(key)
        if rec is None or rec.evicted:
            return 0
        n_tok = rec.num_tokens
        rec.evicted = True
        rec.k_data = torch.empty(0)
        rec.v_data = torch.empty(0)
        self.evicted_buckets_count += 1
        self.evicted_tokens_count += n_tok
        return n_tok

    def memory_stats(self) -> Dict[str, Any]:
        """Compute exact byte footprints and cache hit statistics."""
        hot_fixed = 0
        for t in (self.k_sink, self.v_sink, self.k_tail, self.v_tail):
            if t is not None:
                hot_fixed += t.numel() * t.element_size()
        hot_lru = sum(
            k_g.numel() * k_g.element_size() + v_g.numel() * v_g.element_size()
            for k_g, v_g, _, _ in self.hot_cache.values()
        )
        cold_bytes = sum(rec.nbytes for rec in self.cold_buckets.values() if not rec.evicted)
        hot_total = hot_fixed + hot_lru + self._index_nbytes
        # Dense baseline: full sequence K and V resident on GPU in compute_dtype
        elem_sz = 2
        if self.k_sink is not None and self.k_sink.numel() > 0:
            elem_sz = self.k_sink.element_size()
            h_kv, _, d = self.k_sink.shape
            dense_bytes = 2 * h_kv * self._total_seq_len * d * elem_sz
        else:
            dense_bytes = max(1, hot_total)
        total_reqs = self.cache_hits + self.cache_misses
        return {
            "hot_fixed_bytes": int(hot_fixed),
            "hot_lru_bytes": int(hot_lru),
            "index_bytes": int(self._index_nbytes),
            "hot_total_gpu_bytes": int(hot_total),
            "cold_cpu_bytes": int(cold_bytes),
            "dense_gpu_bytes": int(dense_bytes),
            "gpu_memory_saving_ratio": float(1.0 - hot_total / max(1, dense_bytes)),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_hit_rate": float(self.cache_hits / max(1, total_reqs)),
            "lru_evictions": int(self.lru_evictions),
            "bytes_transferred": int(self.bytes_transferred),
            "evicted_buckets": int(self.evicted_buckets_count),
            "evicted_tokens": int(self.evicted_tokens_count),
            "quantize_fp8": bool(self.quantize_fp8),
        }


# ======================================================================================
# LSA-F6: Dynamic Latent Memory Graph (DLMG) in S^{63}
# ======================================================================================
@dataclass
class BucketNodeState:
    """State of a Ward bucket `(block_id, cluster_id)` in `DynamicLatentMemoryGraph`."""
    global_id: int
    block_id: int
    cluster_id: int
    num_tokens: int
    access_mass: float = 0.0
    access_count: int = 0
    surprise_sum: float = 0.0
    surprise_max: float = 0.0
    valence: float = 1.0
    t_created: int = 0
    t_last_access: int = 0
    # Zep/Graphiti bi-temporal validity (`arXiv:2501.13956`)
    t_valid: int = 0
    t_invalid: Optional[int] = None
    superseded_by: Optional[int] = None
    compiled_and_evicted: bool = False
    super_topic_id: int = -1
    # Optional consolidated activation snapshots for LSA-F7 sleep compilation
    h_bar: Optional[torch.Tensor] = None
    y_target: Optional[torch.Tensor] = None

    @property
    def is_active(self) -> bool:
        return (self.num_tokens > 0) and (self.t_invalid is None) and (not self.compiled_and_evicted)

    @property
    def mean_surprise(self) -> float:
        return self.surprise_sum / max(1, self.access_count) if self.access_count > 0 else max(1.0, self.surprise_max)


class DynamicLatentMemoryGraph:
    """Dynamic Latent Memory Graph (`DLMG`, `LSA-F6`) built directly over `PagedLatentIndex` Ward buckets in `S^{d_idx-1}`.

    Combines:
      1. **Temporal / Episodic Adjacency (`w_temporal`)**: links consecutive non-empty buckets along token time.
      2. **A-MEM Cross-Block Semantic Links (`arXiv:2502.12110`)**: links each new bucket to its top-`sem_top_m` nearest
         earlier-block buckets with cosine >= `sem_link_threshold`.
      3. **Hebbian Co-Access Edges (Titans / Nested Learning)**: strengthens edges between buckets co-attended by the same query:
         `W_hebb[u, v] <- (1 - edge_decay) * W_hebb[u, v] + hebbian_lr * sqrt(alpha_u * alpha_v)`.
      4. **Zep/Graphiti Bi-Temporal Validity (`arXiv:2501.13956`)**: `[t_valid, t_invalid)` intervals and `superseded_by`
         redirection so queries hitting an outdated bucket automatically route spreading-activation mass to the newer bucket.
      5. **HippoRAG-2 Personalized PageRank (`arXiv:2502.14802`)**: damped spreading activation over the sparse bucket graph
         under strict equal-budget key selection.
    """

    def __init__(
        self,
        index: PagedLatentIndex,
        sem_top_m: int = 4,
        sem_link_threshold: float = 0.55,
        w_temporal: float = 0.20,
        w_semantic: float = 0.50,
        hebbian_lr: float = 0.35,
        edge_decay: float = 0.01,
        ppr_alpha: float = 0.35,
        ppr_steps: int = 2,
        ppr_weight: float = 0.45,
        superseded_discount: float = 0.05,
        n_super_topics: int = 8,
    ):
        self.index = index
        self.sem_top_m = max(1, int(sem_top_m))
        self.sem_link_threshold = float(sem_link_threshold)
        self.w_temporal = float(w_temporal)
        self.w_semantic = float(w_semantic)
        self.hebbian_lr = float(hebbian_lr)
        self.edge_decay = float(edge_decay)
        self.ppr_alpha = float(ppr_alpha)
        self.ppr_steps = max(1, int(ppr_steps))
        self.ppr_weight = float(ppr_weight)
        self.superseded_discount = float(superseded_discount)
        self.n_super_topics = max(2, int(n_super_topics))

        self.nodes: Dict[int, BucketNodeState] = {}
        # Adjacency: adj[u][v] = edge weight (u -> v)
        self.sem_edges: Dict[int, Dict[int, float]] = {}
        self.temp_edges: Dict[int, Dict[int, float]] = {}
        self.hebb_edges: Dict[int, Dict[int, float]] = {}
        self.super_centroids: Optional[torch.Tensor] = None
        self._synced_blocks: int = 0
        self.step_counter: int = 0

        self.sync_with_index()

    @property
    def pools_per_block(self) -> int:
        return int(self.index.pools_per_block)

    def to_global_id(self, block_id: int, cluster_id: int) -> int:
        return int(block_id) * self.pools_per_block + int(cluster_id)

    def from_global_id(self, global_id: int) -> Tuple[int, int]:
        c = self.pools_per_block
        return int(global_id // c), int(global_id % c)

    def get_centroid(self, global_id: int) -> torch.Tensor:
        b, c = self.from_global_id(global_id)
        return self.index.centroids[b, c]

    @torch.no_grad()
    def sync_with_index(self) -> int:
        """Incrementally register nodes and A-MEM cross-block semantic + temporal edges for any newly clustered blocks."""
        if self.index.n_blocks <= self._synced_blocks:
            return 0
        new_nodes = 0
        c_per_blk = self.pools_per_block
        prev_gid: Optional[int] = None
        if self._synced_blocks > 0:
            # Find last non-empty bucket in previously synced blocks
            for g in sorted(self.nodes.keys(), reverse=True):
                if self.nodes[g].num_tokens > 0:
                    prev_gid = g
                    break

        for b in range(self._synced_blocks, self.index.n_blocks):
            btoks_b = self.index.bucket_tokens[b]
            cents_b = self.index.centroids[b]  # [C, D]
            nz_c = torch.nonzero(btoks_b > 0, as_tuple=False).reshape(-1).tolist()

            # Prepare earlier block centroids for A-MEM cross-block semantic linking
            earlier_gids: List[int] = [
                g for g, st in self.nodes.items() if st.block_id < b and st.is_active
            ]
            earlier_cents: Optional[torch.Tensor] = None
            if earlier_gids:
                earlier_cents = torch.stack([self.get_centroid(g) for g in earlier_gids], dim=0)

            for c in nz_c:
                gid = self.to_global_id(b, int(c))
                n_tok = int(btoks_b[c].item())
                self.nodes[gid] = BucketNodeState(
                    global_id=gid,
                    block_id=b,
                    cluster_id=int(c),
                    num_tokens=n_tok,
                    t_created=self.step_counter,
                    t_valid=self.step_counter,
                )
                self.sem_edges.setdefault(gid, {})
                self.temp_edges.setdefault(gid, {})
                self.hebb_edges.setdefault(gid, {})
                new_nodes += 1

                # 1. Temporal edge to immediately preceding bucket
                if prev_gid is not None and prev_gid != gid:
                    self.temp_edges[prev_gid][gid] = self.w_temporal
                    self.temp_edges[gid][prev_gid] = self.w_temporal * 0.5
                prev_gid = gid

                # 2. A-MEM cross-block semantic links (`arXiv:2502.12110`)
                if earlier_cents is not None and earlier_cents.shape[0] > 0:
                    cent = cents_b[c].to(earlier_cents.device)
                    sims = earlier_cents @ cent
                    k_take = min(self.sem_top_m, int(sims.numel()))
                    top_vals, top_idx = sims.topk(k_take)
                    for val, idx_e in zip(top_vals.tolist(), top_idx.tolist()):
                        if val >= self.sem_link_threshold:
                            egid = earlier_gids[int(idx_e)]
                            w = float(val) * self.w_semantic
                            self.sem_edges[gid][egid] = max(self.sem_edges[gid].get(egid, 0.0), w)
                            self.sem_edges[egid][gid] = max(self.sem_edges[egid].get(gid, 0.0), w)

        self._synced_blocks = self.index.n_blocks
        self._rebuild_super_topics()
        return new_nodes

    @torch.no_grad()
    def _rebuild_super_topics(self) -> None:
        """2nd-level Ward clustering over active bucket centroids to form topic super-buckets."""
        active_gids = [g for g, st in self.nodes.items() if st.is_active]
        n_act = len(active_gids)
        if n_act < 4:
            return
        cents = torch.stack([self.get_centroid(g) for g in active_gids], dim=0).unsqueeze(0)  # [1, N, D]
        target_b = max(2, min(self.n_super_topics, n_act // 2))
        max_sz = max(2, (n_act * 2) // target_b)
        labels = ward_partition(cents, target_buckets=target_b, max_pools_per_bucket=max_sz)[0]  # [N]
        unique_lbls = labels.unique(sorted=True).tolist()
        remap = {old: i for i, old in enumerate(unique_lbls)}
        super_cents = []
        for old in unique_lbls:
            mask = labels == old
            c_mean = F.normalize(cents[0, mask].mean(dim=0), dim=-1, eps=1e-6)
            super_cents.append(c_mean)
        self.super_centroids = torch.stack(super_cents, dim=0)
        for gid, lbl in zip(active_gids, labels.tolist()):
            self.nodes[gid].super_topic_id = remap[int(lbl)]

    def record_access_and_hebbian_coactivation(
        self,
        bucket_masses: Dict[int, float],
        surprise: float = 1.0,
        valence: float = 1.0,
        top_coactive: int = 6,
    ) -> int:
        """Update per-bucket access mass/surprise and strengthen Hebbian co-access edges between co-attended buckets."""
        self.step_counter += 1
        if not bucket_masses:
            return 0
        # Decay existing Hebbian edges slightly
        if self.edge_decay > 0.0:
            keep = 1.0 - self.edge_decay
            for u in list(self.hebb_edges.keys()):
                nbrs = self.hebb_edges[u]
                for v in list(nbrs.keys()):
                    nbrs[v] *= keep
                    if nbrs[v] < 1e-4:
                        del nbrs[v]

        ranked = sorted(
            [(int(g), float(m)) for g, m in bucket_masses.items() if g in self.nodes and m > 0],
            key=lambda kv: kv[1],
            reverse=True,
        )
        for gid, mass in ranked:
            st = self.nodes[gid]
            st.access_mass += mass
            st.access_count += 1
            st.surprise_sum += float(surprise) * mass
            st.surprise_max = max(st.surprise_max, float(surprise))
            st.valence = 0.9 * st.valence + 0.1 * float(valence)
            st.t_last_access = self.step_counter

        co_list = ranked[: max(2, int(top_coactive))]
        updates = 0
        for i in range(len(co_list)):
            u, mu = co_list[i]
            for j in range(i + 1, len(co_list)):
                v, mv = co_list[j]
                if u == v:
                    continue
                delta = self.hebbian_lr * math.sqrt(max(0.0, mu * mv))
                self.hebb_edges.setdefault(u, {})[v] = self.hebb_edges.get(u, {}).get(v, 0.0) + delta
                self.hebb_edges.setdefault(v, {})[u] = self.hebb_edges.get(v, {}).get(u, 0.0) + delta
                updates += 2
        return updates

    def invalidate_superseded_bucket(self, old_gid: int, new_gid: int, t_invalid: Optional[int] = None) -> None:
        """Mark `old_gid` as superseded by `new_gid` (`Zep/Graphiti` bi-temporal invalidation, `arXiv:2501.13956`)."""
        self.step_counter += 1
        t_inv = self.step_counter if t_invalid is None else int(t_invalid)
        if old_gid in self.nodes:
            self.nodes[old_gid].t_invalid = t_inv
            self.nodes[old_gid].superseded_by = int(new_gid)
            # Add strong forward link from old -> new so any residual walk transitions to new_gid
            self.sem_edges.setdefault(int(old_gid), {})[int(new_gid)] = 1.0

    @torch.no_grad()
    def detect_and_invalidate_contradictions(
        self,
        new_block_id: int,
        anchor_dims: int = 32,
        anchor_sim_threshold: float = 0.82,
        payload_max_sim: float = 0.55,
    ) -> List[Tuple[int, int]]:
        """Automatically detect when a bucket in `new_block_id` shares a high entity/topic anchor (`[:anchor_dims]`)
        with an older bucket (`block_id < new_block_id`) but has a diverged value/payload slice (`[anchor_dims:]`),
        and bi-temporally invalidate the older bucket in favor of the newer one."""
        self.sync_with_index()
        new_gids = [
            g for g, st in self.nodes.items() if st.block_id == int(new_block_id) and st.is_active
        ]
        old_gids = [
            g for g, st in self.nodes.items() if st.block_id < int(new_block_id) and st.is_active
        ]
        if not new_gids or not old_gids:
            return []
        d = self.index.centroids.shape[-1]
        a_dim = min(max(4, int(anchor_dims)), d - 4)
        old_cents = torch.stack([self.get_centroid(g) for g in old_gids], dim=0)
        old_anc = F.normalize(old_cents[:, :a_dim], dim=-1, eps=1e-6)
        old_pay = F.normalize(old_cents[:, a_dim:], dim=-1, eps=1e-6)

        superseded_pairs: List[Tuple[int, int]] = []
        for ngid in new_gids:
            nc = self.get_centroid(ngid).to(old_cents.device)
            n_anc = F.normalize(nc[:a_dim], dim=-1, eps=1e-6)
            n_pay = F.normalize(nc[a_dim:], dim=-1, eps=1e-6)
            anc_sims = old_anc @ n_anc
            pay_sims = old_pay @ n_pay
            matches = torch.nonzero((anc_sims >= anchor_sim_threshold) & (pay_sims <= payload_max_sim), as_tuple=False).reshape(-1)
            for idx_m in matches.tolist():
                ogid = old_gids[int(idx_m)]
                if self.nodes[ogid].t_invalid is None:
                    self.invalidate_superseded_bucket(ogid, ngid)
                    superseded_pairs.append((ogid, ngid))
        return superseded_pairs

    def _combined_neighbors(self, gid: int, max_block: int) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for ed in (self.hebb_edges.get(gid, {}), self.sem_edges.get(gid, {}), self.temp_edges.get(gid, {})):
            for nbr, w in ed.items():
                st = self.nodes.get(nbr)
                if st is None or st.compiled_and_evicted or st.num_tokens <= 0 or st.block_id >= max_block:
                    continue
                out[nbr] = out.get(nbr, 0.0) + float(w)
        return out

    @torch.no_grad()
    def select_with_spreading_activation(
        self,
        q_idx: torch.Tensor,
        budget: int,
        upto: Optional[int] = None,
        seed_top_k: int = 8,
        tau_seed: float = 12.0,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], Dict[int, float]]:
        """Select exact key positions under equal-budget `budget` using HippoRAG-2 Personalized PageRank (`PPR`)
        spreading activation + Zep/Graphiti bi-temporal supersession redirection.

        Returns `(token_positions, selected_buckets, bucket_scores)`."""
        self.sync_with_index()
        dev = q_idx.device
        upto_t = self.index.t if upto is None else min(int(upto), self.index.t)
        limit = (
            0
            if self.index.n_blocks == 0
            else max(0, min(self.index.n_blocks, (upto_t - self.index.window - self.index.mid_start) // self.index.n_cluster))
        )
        clustered_end = self.index.mid_start + limit * self.index.n_cluster
        sink = torch.arange(0, min(self.index.n_sink, upto_t), device=dev)
        tail = torch.arange(clustered_end, upto_t, device=dev)
        if limit == 0:
            return torch.cat([sink, tail]).unique(), [], {}

        q = F.normalize(q_idx.float().reshape(-1, q_idx.shape[-1]).mean(dim=0), dim=-1, eps=1e-6)
        cents = self.index.centroids[:limit].to(dev)
        toks = self.index.bucket_tokens[:limit].to(dev)
        cos_scores = torch.einsum("bcd,d->bc", cents, q).masked_fill(toks == 0, float("-inf"))

        # Mask out buckets that have been compiled & evicted
        for gid, st in self.nodes.items():
            if st.block_id < limit and st.compiled_and_evicted:
                cos_scores[st.block_id, st.cluster_id] = float("-inf")

        flat_cos = cos_scores.reshape(-1)
        finite_idx = torch.nonzero(torch.isfinite(flat_cos), as_tuple=False).reshape(-1)
        if finite_idx.numel() == 0:
            return torch.cat([sink, tail]).unique(), [], {}

        # Step 1: Seed distribution over top-K cosine buckets
        k_seed = min(int(seed_top_k), int(finite_idx.numel()))
        seed_vals, seed_pos = flat_cos[finite_idx].topk(k_seed)
        seed_gids = finite_idx[seed_pos].tolist()
        seed_probs = torch.softmax(seed_vals * float(tau_seed), dim=0).tolist()
        s0: Dict[int, float] = {int(g): float(p) for g, p in zip(seed_gids, seed_probs)}

        # Step 2: Damped Personalized PageRank (`arXiv:2502.14802`)
        pi: Dict[int, float] = dict(s0)
        alpha = self.ppr_alpha
        for _ in range(self.ppr_steps):
            nxt: Dict[int, float] = {g: (1.0 - alpha) * p for g, p in s0.items()}
            for u, mass_u in pi.items():
                if mass_u <= 1e-6:
                    continue
                nbrs = self._combined_neighbors(u, max_block=limit)
                if not nbrs:
                    nxt[u] = nxt.get(u, 0.0) + alpha * mass_u
                    continue
                z = sum(nbrs.values())
                for v, w_uv in nbrs.items():
                    nxt[v] = nxt.get(v, 0.0) + alpha * mass_u * (w_uv / z)
            pi = nxt

        # Step 3: Bi-temporal supersession redirection (`arXiv:2501.13956`)
        redirected_pi: Dict[int, float] = {}
        for g, mass in pi.items():
            st = self.nodes.get(g)
            if st is not None and st.t_invalid is not None and st.superseded_by is not None:
                target_g = int(st.superseded_by)
                target_st = self.nodes.get(target_g)
                if target_st is not None and target_st.block_id < limit and not target_st.compiled_and_evicted:
                    redirected_pi[g] = redirected_pi.get(g, 0.0) + mass * self.superseded_discount
                    redirected_pi[target_g] = redirected_pi.get(target_g, 0.0) + mass * (1.0 - self.superseded_discount)
                    continue
            redirected_pi[g] = redirected_pi.get(g, 0.0) + mass

        # Also penalize raw cosine of superseded buckets and boost associative flow targets
        adj_scores = flat_cos.clone()
        flow_pi: Dict[int, float] = {
            g: max(0.0, mass - (1.0 - alpha) * s0.get(g, 0.0)) for g, mass in redirected_pi.items()
        }
        max_flow = max(0.05, max(flow_pi.values()) if flow_pi else 0.05)
        for gid, st in self.nodes.items():
            if st.block_id >= limit or st.compiled_and_evicted or st.num_tokens <= 0:
                continue
            if st.t_invalid is not None:
                adj_scores[gid] = adj_scores[gid] - 1.0
        for g, flow_mass in flow_pi.items():
            if 0 <= g < adj_scores.numel() and torch.isfinite(adj_scores[g]):
                adj_scores[g] = adj_scores[g] + self.ppr_weight * (flow_mass / max_flow)

        # Step 4: Equal-budget bucket selection + optional Stage-2 token rerank
        flat_s, flat_i = adj_scores.sort(descending=True)
        flat_t = toks.reshape(-1)[flat_i]
        csum = torch.cumsum(flat_t, dim=0)
        take = (csum - flat_t) < int(budget) * self.index.rerank
        take &= torch.isfinite(flat_s)
        sel_flat = flat_i[take]

        selected_buckets: List[Tuple[int, int]] = [self.from_global_id(int(g)) for g in sel_flat.tolist()]
        bucket_score_map: Dict[int, float] = {
            int(g): float(s) for g, s in zip(sel_flat.tolist(), flat_s[take].tolist())
        }

        sel_mask = torch.zeros(limit * self.pools_per_block, dtype=torch.bool, device=dev)
        sel_mask[sel_flat] = True
        sel_mask = sel_mask.view(limit, self.pools_per_block)
        pool_sel = sel_mask.gather(1, self.index.labels[:limit].to(dev))
        pool_ids = torch.nonzero(pool_sel)
        base = self.index.mid_start + pool_ids[:, 0] * self.index.n_cluster + pool_ids[:, 1] * self.index.n_pool
        mid = (base.unsqueeze(1) + torch.arange(self.index.n_pool, device=dev).unsqueeze(0)).reshape(-1)
        if self.index.rerank > 1 and mid.numel() > int(budget) and self.index.k_idx_n is not None:
            tok_scores = self.index.k_idx_n.to(dev)[mid] @ q
            mid = mid[tok_scores.topk(int(budget)).indices]
        return torch.cat([sink, mid, tail]).unique(), selected_buckets, bucket_score_map

    def stationary_pagerank(self, steps: int = 15, damping: float = 0.85) -> Dict[int, float]:
        """Compute global stationary PageRank over active buckets for `LSA-F7` sleep salience ranking."""
        active = [g for g, st in self.nodes.items() if st.is_active]
        n = len(active)
        if n == 0:
            return {}
        pr = {g: 1.0 / n for g in active}
        base = (1.0 - damping) / n
        max_blk = self.index.n_blocks
        for _ in range(steps):
            nxt = {g: base for g in active}
            dangling = 0.0
            for u in active:
                nbrs = self._combined_neighbors(u, max_block=max_blk)
                if not nbrs:
                    dangling += pr[u]
                    continue
                z = sum(nbrs.values())
                for v, w in nbrs.items():
                    if v in nxt:
                        nxt[v] += damping * pr[u] * (w / z)
            if dangling > 0:
                share = damping * dangling / n
                for g in active:
                    nxt[g] += share
            pr = nxt
        return pr


# ======================================================================================
# LSA-F7: Sleep-Time Vertical Memory Consolidation Engine (Null-Space Ridge + WISE Scope)
# ======================================================================================
@dataclass
class ConsolidationLedgerEntry:
    """Audit record for a single Sleep Consolidation batch (`LSA-F7`), supporting exact one-step unlearning."""
    entry_id: int
    timestamp: float
    compiled_bucket_gids: List[int]
    evicted_tokens: int
    delta_weight: torch.Tensor          # [D_in, D_out] weight increment applied to vertical memory
    prototypes: torch.Tensor            # [M, D_in] unit-norm key prototypes registered in WISE scope bank
    basis_count_before: int
    basis_count_after: int
    rel_reconstruction_error: float
    unlearned: bool = False


class NullSpaceVerticalAdapter(nn.Module):
    """Lightweight vertical parametric memory bank with:
      1. **AlphaEdit Null-Space Constrained Closed-Form Ridge Solver (`arXiv:2410.02355`)** so sequential sleep
         compilations have mathematically zero interference on previously consolidated memories.
      2. **WISE Activation Scope Gate (`arXiv:2405.14768`)** so out-of-scope general queries whose cosine similarity
         with all consolidated prototypes is below `scope_threshold` receive **exact `0.0`** perturbation.
    """

    def __init__(self, d_model: int = 64, d_hidden: int = 128, scope_threshold: float = 0.65, ridge_alpha: float = 1e-4):
        super().__init__()
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.scope_threshold = float(scope_threshold)
        self.ridge_alpha = float(ridge_alpha)

        self.w_gate = nn.Linear(self.d_model, self.d_hidden, bias=False)
        self.w_up = nn.Linear(self.d_model, self.d_hidden, bias=False)
        nn.init.orthogonal_(self.w_gate.weight)
        nn.init.orthogonal_(self.w_up.weight)
        for p in (self.w_gate.weight, self.w_up.weight):
            p.requires_grad_(False)

        # w_down: [d_hidden, d_model] initialized to exact zero
        self.register_buffer("w_down", torch.zeros(self.d_hidden, self.d_model, dtype=torch.float32))
        # Orthonormal basis V_active: [K, d_hidden] of preserved post-SwiGLU activations
        self.register_buffer("active_basis", torch.zeros(0, self.d_hidden, dtype=torch.float32))
        # WISE scope prototypes: [P, d_model] unit-norm query directions
        self.register_buffer("scope_prototypes", torch.zeros(0, self.d_model, dtype=torch.float32))

    def swiglu_features(self, h: torch.Tensor) -> torch.Tensor:
        hf = F.normalize(h.float(), dim=-1, eps=1e-6)
        a = F.silu(F.linear(hf, self.w_gate.weight.float())) * F.linear(hf, self.w_up.weight.float())
        return F.normalize(a, dim=-1, eps=1e-6)

    def scope_gate(self, h: torch.Tensor) -> torch.Tensor:
        """WISE Activation Scope Indicator (`arXiv:2405.14768` Eq. 5): returns `0.0` when `max_p cos(h, p) < scope_threshold`."""
        if self.scope_prototypes.shape[0] == 0:
            return torch.zeros(h.shape[:-1] + (1,), dtype=h.dtype, device=h.device)
        hf = F.normalize(h.float(), dim=-1, eps=1e-6)
        sims = hf @ self.scope_prototypes.to(hf.device).t()  # [..., P]
        max_sim = sims.max(dim=-1, keepdim=True).values
        active = (max_sim >= self.scope_threshold).to(h.dtype)
        return active

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        gate = self.scope_gate(h)
        if float(gate.abs().max().item()) == 0.0:
            return torch.zeros_like(h)
        a = self.swiglu_features(h)
        out = (a @ self.w_down.to(a.device)).to(h.dtype)
        return gate * out

    @torch.no_grad()
    def compile_nullspace_ridge(self, h_Triggers: torch.Tensor, y_targets: torch.Tensor) -> Tuple[torch.Tensor, int, int, float]:
        """Compile `(h_Triggers [M, D], y_targets [M, D])` into `w_down` inside the null space of `active_basis`."""
        dev = self.w_down.device
        h_f = F.normalize(h_Triggers.float().to(dev), dim=-1, eps=1e-6)
        y_f = y_targets.float().to(dev)
        a_1 = self.swiglu_features(h_f)  # [M, d_hidden]

        # Residual needed on top of current prediction
        current_pred = a_1 @ self.w_down
        delta_z = y_f - current_pred  # [M, d_model]

        # Double Classical Gram-Schmidt projection into null space of active_basis
        v_act = self.active_basis
        a_perp = a_1.clone()
        if v_act.shape[0] > 0:
            for _ in range(2):
                a_perp = a_perp - (a_perp @ v_act.t()) @ v_act

        m = a_perp.shape[0]
        gram = a_perp @ a_perp.t() + self.ridge_alpha * torch.eye(m, device=dev, dtype=torch.float32)
        dual_sol = torch.linalg.solve(gram, delta_z)  # [M, d_model]
        delta_w = a_perp.t() @ dual_sol               # [d_hidden, d_model]
        self.w_down.add_(delta_w)

        # Expand orthonormal active_basis with newly compiled directions
        b_before = int(v_act.shape[0])
        v_list = [v_act[i] for i in range(b_before)]
        for i in range(m):
            u = a_perp[i].clone()
            for _ in range(2):
                for b_vec in v_list:
                    u = u - torch.dot(u, b_vec) * b_vec
            nrm = float(u.norm().item())
            if nrm > 1e-5 and len(v_list) < self.d_hidden:
                v_list.append(u / nrm)
        self.active_basis = torch.stack(v_list, dim=0) if v_list else torch.zeros(0, self.d_hidden, device=dev)
        b_after = int(self.active_basis.shape[0])

        # Register prototypes in WISE scope bank
        self.scope_prototypes = (
            torch.cat([self.scope_prototypes, h_f], dim=0) if self.scope_prototypes.shape[0] > 0 else h_f.clone()
        )
        recon = self.forward(h_f)
        rel_err = float((recon - y_f).norm().item() / max(1e-6, float(y_f.norm().item())))
        return delta_w, b_before, b_after, rel_err


class SleepConsolidationEngine:
    """`LSA-F7` Sleep Consolidation Engine:
      1. Ranks active `DLMG` buckets by `Salience(g) * (1 + beta_pr * PageRank(g))`, strictly excluding superseded
         buckets (`t_invalid is not None`).
      2. Compiles top-`max_buckets` salient buckets into `NullSpaceVerticalAdapter` (or `ZipfianTieredMoEEngram`).
      3. Permanently evicts compiled buckets from `DLMG` and `HotColdPagedKVStore`, freeing horizontal KV tokens.
      4. Logs every consolidation batch to `ledger` with one-step exact unlearning (`unlearn_entry`).
    """

    def __init__(
        self,
        dlmg: DynamicLatentMemoryGraph,
        kv_store: Optional[HotColdPagedKVStore],
        vertical_memory: NullSpaceVerticalAdapter,
        beta_pr: float = 2.0,
    ):
        self.dlmg = dlmg
        self.kv_store = kv_store
        self.vertical_memory = vertical_memory
        self.beta_pr = float(beta_pr)
        self.ledger: List[ConsolidationLedgerEntry] = []

    def rank_consolidatable_buckets(self) -> List[Tuple[int, float]]:
        """Rank active, non-superseded buckets that have `(h_bar, y_target)` snapshots ready for vertical compilation."""
        pr = self.dlmg.stationary_pagerank()
        scored: List[Tuple[int, float]] = []
        for gid, st in self.dlmg.nodes.items():
            if not st.is_active or st.h_bar is None or st.y_target is None:
                continue
            salience = max(1e-4, st.access_mass) * max(1.0, st.mean_surprise) * abs(st.valence)
            score = salience * (1.0 + self.beta_pr * pr.get(gid, 0.0))
            scored.append((gid, float(score)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    @torch.no_grad()
    def consolidate_top_buckets(self, max_buckets: int = 8) -> Dict[str, Any]:
        """Compile top-`max_buckets` into `vertical_memory` and evict their horizontal KV cache slots."""
        ranked = self.rank_consolidatable_buckets()[: max(1, int(max_buckets))]
        if not ranked:
            return {"compiled_buckets": 0, "evicted_tokens": 0}

        gids = [gid for gid, _ in ranked]
        h_stack = torch.stack([self.dlmg.nodes[g].h_bar.reshape(-1) for g in gids], dim=0)
        y_stack = torch.stack([self.dlmg.nodes[g].y_target.reshape(-1) for g in gids], dim=0)
        delta_w, b_before, b_after, rel_err = self.vertical_memory.compile_nullspace_ridge(h_stack, y_stack)

        evicted_toks = 0
        for gid in gids:
            st = self.dlmg.nodes[gid]
            st.compiled_and_evicted = True
            if self.kv_store is not None:
                evicted_toks += self.kv_store.evict_bucket_permanently(st.block_id, st.cluster_id)
            else:
                evicted_toks += st.num_tokens

        entry = ConsolidationLedgerEntry(
            entry_id=len(self.ledger),
            timestamp=time.time(),
            compiled_bucket_gids=gids,
            evicted_tokens=int(evicted_toks),
            delta_weight=delta_w.detach().cpu(),
            prototypes=F.normalize(h_stack.float(), dim=-1, eps=1e-6).detach().cpu(),
            basis_count_before=b_before,
            basis_count_after=b_after,
            rel_reconstruction_error=rel_err,
        )
        self.ledger.append(entry)
        return {
            "entry_id": entry.entry_id,
            "compiled_buckets": len(gids),
            "compiled_gids": gids,
            "evicted_tokens": int(evicted_toks),
            "rel_reconstruction_error": float(rel_err),
            "basis_count_after": int(b_after),
        }

    @torch.no_grad()
    def unlearn_entry(self, entry_id: int) -> bool:
        """Exact one-step surgical unlearning of a consolidated batch via the ledger."""
        if entry_id < 0 or entry_id >= len(self.ledger):
            return False
        entry = self.ledger[entry_id]
        if entry.unlearned:
            return False
        dev = self.vertical_memory.w_down.device
        self.vertical_memory.w_down.sub_(entry.delta_weight.to(dev))
        # Remove prototypes matching this entry
        if self.vertical_memory.scope_prototypes.shape[0] > 0:
            protos = self.vertical_memory.scope_prototypes
            rem = entry.prototypes.to(dev)
            sims = protos @ rem.t()
            keep_mask = sims.max(dim=-1).values < 0.999
            self.vertical_memory.scope_prototypes = protos[keep_mask]
        entry.unlearned = True
        return True


# ======================================================================================
# Controlled Evaluation Suites for LSA-F4, LSA-F6 (DLMG), and LSA-F7 (Sleep Consolidation)
# ======================================================================================
@torch.no_grad()
def evaluate_dlmg_recall_suite(seed: int = 42) -> Dict[str, Any]:
    """Evaluate `DynamicLatentMemoryGraph` (`LSA-F6`) vs flat `PagedLatentIndex` under strict equal token budgets across:
      1. `single_hop`: direct cosine needle retrieval (must have zero regression).
      2. `multi_hop_bridge`: Query matches Bridge Bucket A (`block 0`), which was co-accessed with orthogonal Target
         Evidence Bucket B (`block 2`). Measures recall of Target Bucket B tokens under equal token budget.
      3. `temporal_update`: Stale Fact Bucket V1 (`block 0`) is superseded by Updated Fact Bucket V2 (`block 2`) sharing
         the same entity anchor slice `[:32]` with a new value slice `[32:]`. Measures V2-vs-V1 selection preference.
    """
    g = torch.Generator().manual_seed(seed)
    n_sink, n_pool, n_cluster, window = 4, 4, 64, 32
    t, d = 320, 64  # 4 blocks of 64 tokens + sink(4) + tail(60)

    # Plant clean cluster directions per block
    k_idx = F.normalize(torch.randn(t, d, generator=g) * 0.15, dim=-1)
    # Block 0: [4..68), Block 1: [68..132), Block 2: [132..196), Block 3: [196..260)
    u_bridge = F.normalize(torch.randn(d, generator=g), dim=-1)
    # Orthogonal target vector for 2-hop bridge
    u_target_raw = torch.randn(d, generator=g)
    u_target = F.normalize(u_target_raw - torch.dot(u_target_raw, u_bridge) * u_bridge, dim=-1)
    u_single = F.normalize(torch.randn(d, generator=g), dim=-1)

    # Entity anchor + two conflicting payload vectors for temporal update test
    anchor = F.normalize(torch.randn(32, generator=g), dim=-1)
    pay_v1 = F.normalize(torch.randn(32, generator=g), dim=-1)
    pay_v2 = F.normalize(-pay_v1 + 0.2 * torch.randn(32, generator=g), dim=-1)
    u_fact_v1 = F.normalize(torch.cat([anchor, pay_v1], dim=0), dim=-1)
    u_fact_v2 = F.normalize(torch.cat([anchor, pay_v2], dim=0), dim=-1)
    u_entity_query = F.normalize(torch.cat([anchor, pay_v1 * 0.15], dim=0), dim=-1)
    # Ensure u_bridge is orthogonal to u_target, u_fact_v1, u_fact_v2, u_single
    for vec in (u_target, u_fact_v1, u_fact_v2, u_single):
        u_bridge = F.normalize(u_bridge - torch.dot(u_bridge, vec) * vec, dim=-1)

    # Inject planted tokens
    bridge_span = slice(4, 20)      # Block 0, 16 tokens
    v1_span = slice(20, 36)         # Block 0, 16 tokens (stale fact V1)
    single_span = slice(68, 84)     # Block 1, 16 tokens (single-hop needle)
    target_span = slice(132, 148)   # Block 2, 16 tokens (2-hop target B)
    v2_span = slice(148, 164)       # Block 2, 16 tokens (updated fact V2)

    k_idx[bridge_span] = F.normalize(u_bridge.unsqueeze(0) + 0.02 * torch.randn(16, d, generator=g), dim=-1)
    k_idx[v1_span] = F.normalize(u_fact_v1.unsqueeze(0) + 0.02 * torch.randn(16, d, generator=g), dim=-1)
    k_idx[single_span] = F.normalize(u_single.unsqueeze(0) + 0.02 * torch.randn(16, d, generator=g), dim=-1)
    k_idx[target_span] = F.normalize(u_target.unsqueeze(0) + 0.02 * torch.randn(16, d, generator=g), dim=-1)
    k_idx[v2_span] = F.normalize(u_fact_v2.unsqueeze(0) + 0.02 * torch.randn(16, d, generator=g), dim=-1)

    idx = PagedLatentIndex(k_idx, n_sink=n_sink, n_pool=n_pool, n_cluster=n_cluster, avg_bucket_size=16, max_bucket_size=16, window=window)
    dlmg = DynamicLatentMemoryGraph(idx, ppr_alpha=0.45, ppr_steps=2, ppr_weight=0.80)

    # Identify bucket global IDs containing bridge_span and target_span
    bridge_c = int(idx.labels[0, 0].item())
    target_c = int(idx.labels[2, 0].item())
    bridge_gid = dlmg.to_global_id(0, bridge_c)
    target_gid = dlmg.to_global_id(2, target_c)

    # Record prior episodic co-access between Bridge Bucket A and Target Bucket B
    for _ in range(3):
        dlmg.record_access_and_hebbian_coactivation({bridge_gid: 0.5, target_gid: 0.5}, surprise=2.5)

    # Run automatic contradiction detection on Block 2 (detects V2 superseding V1)
    superseded = dlmg.detect_and_invalidate_contradictions(new_block_id=2, anchor_dims=32, anchor_sim_threshold=0.75, payload_max_sim=0.50)

    budget = 32  # Enough for 2 buckets of 16 tokens

    # 1. Single-hop evaluation
    sel_flat_single = idx.select(u_single, budget=budget)
    sel_dlmg_single, _, _ = dlmg.select_with_spreading_activation(u_single, budget=budget)
    single_pos = set(range(68, 84))
    rec_flat_single = len(single_pos.intersection(sel_flat_single.tolist())) / len(single_pos)
    rec_dlmg_single = len(single_pos.intersection(sel_dlmg_single.tolist())) / len(single_pos)

    # 2. Multi-hop associative bridge evaluation (query is u_bridge; target evidence is target_span in Block 2)
    sel_flat_hop = idx.select(u_bridge, budget=budget)
    sel_dlmg_hop, _, _ = dlmg.select_with_spreading_activation(u_bridge, budget=budget, seed_top_k=2)
    target_pos = set(range(132, 148))
    rec_flat_hop = len(target_pos.intersection(sel_flat_hop.tolist())) / len(target_pos)
    rec_dlmg_hop = len(target_pos.intersection(sel_dlmg_hop.tolist())) / len(target_pos)

    # 3. Bi-temporal contradiction update evaluation (budget=16 so only 1 bucket of 16 tokens can be chosen first)
    sel_flat_temp = idx.select(u_entity_query, budget=16)
    sel_dlmg_temp, _, _ = dlmg.select_with_spreading_activation(u_entity_query, budget=16, seed_top_k=4)
    v1_pos, v2_pos = set(range(20, 36)), set(range(148, 164))
    flat_v2_hits = len(v2_pos.intersection(sel_flat_temp.tolist()))
    flat_v1_hits = len(v1_pos.intersection(sel_flat_temp.tolist()))
    dlmg_v2_hits = len(v2_pos.intersection(sel_dlmg_temp.tolist()))
    dlmg_v1_hits = len(v1_pos.intersection(sel_dlmg_temp.tolist()))

    return {
        "single_hop": {
            "flat_recall": float(rec_flat_single),
            "dlmg_recall": float(rec_dlmg_single),
            "flat_long_keys": int(sel_flat_single.numel() - n_sink - (t - idx.clustered_end)),
            "dlmg_long_keys": int(sel_dlmg_single.numel() - n_sink - (t - idx.clustered_end)),
        },
        "multi_hop_bridge": {
            "flat_target_recall": float(rec_flat_hop),
            "dlmg_target_recall": float(rec_dlmg_hop),
            "gain": float(rec_dlmg_hop - rec_flat_hop),
            "flat_long_keys": int(sel_flat_hop.numel() - n_sink - (t - idx.clustered_end)),
            "dlmg_long_keys": int(sel_dlmg_hop.numel() - n_sink - (t - idx.clustered_end)),
        },
        "temporal_update": {
            "superseded_pairs_detected": len(superseded),
            "flat_v2_share": float(flat_v2_hits / max(1, flat_v1_hits + flat_v2_hits)),
            "dlmg_v2_share": float(dlmg_v2_hits / max(1, dlmg_v1_hits + dlmg_v2_hits)),
            "dlmg_v2_recall": float(dlmg_v2_hits / len(v2_pos)),
            "dlmg_v1_stale_recall": float(dlmg_v1_hits / len(v1_pos)),
        },
    }


@torch.no_grad()
def evaluate_sleep_consolidation_suite(seed: int = 101) -> Dict[str, Any]:
    """Evaluate `LSA-F4` (`HotColdPagedKVStore`) + `LSA-F7` (`SleepConsolidationEngine`) end-to-end:
      - Two sequential sleep consolidation rounds into `NullSpaceVerticalAdapter`.
      - Measures Round-1 reconstruction error, Round-1 drift after Round-2 compilation (null-space zero forgetting),
        WISE scope gate leakage on unrelated general queries, horizontal KV token eviction, and surgical unlearning.
    """
    g = torch.Generator().manual_seed(seed)
    t, h_kv, d = 320, 4, 64
    k = torch.randn(h_kv, t, d, generator=g)
    v = torch.randn(h_kv, t, d, generator=g)
    k_idx = F.normalize(k[0], dim=-1)

    idx = PagedLatentIndex(k_idx, n_sink=4, n_pool=4, n_cluster=64, avg_bucket_size=16, max_bucket_size=16, window=32)
    store = HotColdPagedKVStore(hot_bucket_capacity=8, quantize_fp8=True)
    store.ingest(idx, k, v)
    dlmg = DynamicLatentMemoryGraph(idx)

    # Populate synthetic episodic residual targets on 8 active buckets across blocks 0..3
    active_gids = [gid for gid, st in sorted(dlmg.nodes.items()) if st.is_active][:8]
    for rank_i, gid in enumerate(active_gids):
        st = dlmg.nodes[gid]
        st.access_mass = float(10 - rank_i)
        st.access_count = 2
        st.surprise_sum = 4.0
        st.h_bar = F.normalize(torch.randn(d, generator=g), dim=-1)
        st.y_target = torch.randn(d, generator=g) * 0.5

    adapter = NullSpaceVerticalAdapter(d_model=d, d_hidden=128, scope_threshold=0.65)
    engine = SleepConsolidationEngine(dlmg, store, adapter)

    # Round 1: Consolidate top 4 buckets
    r1 = engine.consolidate_top_buckets(max_buckets=4)
    r1_gids = r1["compiled_gids"]
    h_r1 = torch.stack([dlmg.nodes[gid].h_bar for gid in r1_gids], dim=0)
    y_r1 = torch.stack([dlmg.nodes[gid].y_target for gid in r1_gids], dim=0)
    pred_r1_before_r2 = adapter(h_r1).clone()

    # Round 2: Consolidate next 4 buckets in the null space of Round 1
    r2 = engine.consolidate_top_buckets(max_buckets=4)
    pred_r1_after_r2 = adapter(h_r1)
    r1_nullspace_drift = float((pred_r1_after_r2 - pred_r1_before_r2).abs().max().item())

    # Out-of-scope general queries (random unit vectors in R^64 have max cosine << 0.65)
    h_general = F.normalize(torch.randn(32, d, generator=g), dim=-1)
    general_out = adapter(h_general)
    general_leakage = float(general_out.abs().max().item())

    # Surgical unlearning of Round 2 while keeping Round 1 intact
    engine.unlearn_entry(r2["entry_id"])
    pred_r1_after_unlearn = adapter(h_r1)
    r1_err_after_unlearn = float((pred_r1_after_unlearn - y_r1).norm().item() / max(1e-6, float(y_r1.norm().item())))

    mem_stats = store.memory_stats()
    return {
        "round1_rel_error": float(r1["rel_reconstruction_error"]),
        "round2_rel_error": float(r2["rel_reconstruction_error"]),
        "round1_nullspace_drift_after_round2": float(r1_nullspace_drift),
        "general_query_leakage_max": float(general_leakage),
        "r1_rel_error_after_r2_unlearn": float(r1_err_after_unlearn),
        "evicted_tokens_total": int(mem_stats["evicted_tokens"]),
        "gpu_memory_saving_ratio": float(mem_stats["gpu_memory_saving_ratio"]),
    }
