"""sparkx_dlsa.py — faithful Distilled Latent Sparse Attention (D-LSA) for the 9 global layers of Spark-X2.6-4B.

Why this module exists (verified against `Prefrontal-AI/latent-sparse-attention` docs and this repo's code, 2026-11-04):

* The in-model `LatentSparseAttentionIndexer.select_key_indices` groups keys into **contiguous** spans of 8 tokens ("Ward"
  is only adjacent pairing), sums *all 16* query heads into one 256-d vector (no per-KV-group alignment), and the fit
  (`fit_dlsa_index`) distils on a single short sample. The G2 gate used a ~25-token sentence with `k_budget=32`, i.e. the
  selector returned every key and recall was trivially 1.0.
* The official method: keys are pooled (n_pool=4), clustered per 512-key block by **bottom-up mutual-nearest-neighbour Ward**
  in a learned space; a query scores bucket centroids and reads whole buckets (+ sink + window). The learned space is trained
  by **per-key KL to the frozen teacher's attention** ("a loss aimed at the bucket score ... is a dead end; the loss shapes
  the space and Ward shapes the partition"). The index input is the **position-free part of the logit**: each KV head's
  unrotated dims side by side, with each KV group's query heads summed onto the slice of the key head they read.

This file implements exactly that, as pure torch (CPU-testable), without touching the backbone:

    free_slice_inputs      -> group-aligned position-free index inputs  [T, n_kv * free]
    ward_partition         -> batched vectorised MNN Ward over blocks of pooled keys
    PagedLatentIndex       -> centroids + bucket membership; per-query exact key-index selection (exact K/V are never averaged)
    FaithfulDLSAIndexer    -> learned W_q/W_k (per-layer), per-key KL distillation, drop-in `select_key_indices`
    recall metrics         -> attention-mass recall vs. oracle / random / contiguous baselines
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# Index inputs (position-free slice, group-aligned)
# ======================================================================================
def free_slice_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_dim: int = 64,
    rope_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`q`: [H_q, T_q, D] (post-RoPE), `k`: [H_kv, T_k, D] (post-RoPE) -> `q_in` [T_q, H_kv*F], `k_in` [T_k, H_kv*F].

    `F = D - rope_dim`. Query heads of one KV group are *summed* and placed on the slice of the KV head they read, so that
    `<q_in, k_in> = sum_g sum_{h in g} <q_h^free, k_g^free>` equals the summed position-free logit over heads.
    """
    hq, tq, d = q.shape
    hkv, tk, _ = k.shape
    assert hq % hkv == 0, "H_q must be a multiple of H_kv"
    g = hq // hkv
    sl = slice(rope_dim, d) if rope_first else slice(0, d - rope_dim)
    qf = q[..., sl].float().reshape(hkv, g, tq, -1).sum(dim=1)  # [H_kv, T_q, F]
    kf = k[..., sl].float()  # [H_kv, T_k, F]
    q_in = qf.permute(1, 0, 2).reshape(tq, -1)
    k_in = kf.permute(1, 0, 2).reshape(tk, -1)
    return q_in, k_in


def free_logit_fidelity(q: torch.Tensor, k: torch.Tensor, q_positions: Sequence[int], rope_dim: int = 64,
                        rope_first: bool = True) -> Dict[str, float]:
    """Diagnostic: how well does the position-free logit predict the full (rotated) logit? Mean Pearson r over sampled
    queries and heads, on causal keys. Low values mean the free slice is a poor index space for that layer."""
    hq, _, d = q.shape
    hkv = k.shape[0]
    g = hq // hkv
    kk = k.repeat_interleave(g, dim=0).float()
    qq = q.float()
    sl = slice(rope_dim, d) if rope_first else slice(0, d - rope_dim)
    rs: List[float] = []
    for p in q_positions:
        full = torch.einsum("hd,htd->ht", qq[:, p], kk[:, : p + 1])
        free = torch.einsum("hd,htd->ht", qq[:, p, sl], kk[:, : p + 1, sl])
        a = full - full.mean(dim=-1, keepdim=True)
        b = free - free.mean(dim=-1, keepdim=True)
        r = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)
        rs.append(float(r.mean()))
    return {"mean_pearson_r": sum(rs) / max(1, len(rs)), "n_queries": len(rs)}


# ======================================================================================
# Batched mutual-nearest-neighbour Ward clustering
# ======================================================================================
@torch.no_grad()
def ward_partition(
    pools: torch.Tensor,
    target_buckets: int,
    max_pools_per_bucket: int = 4,
    max_rounds: int = 16,
) -> torch.Tensor:
    """Bottom-up MNN Ward clustering, batched over blocks.

    `pools`: [B, P, d] (unit-norm pooled key vectors). Returns `labels` [B, P] (long) with cluster ids in `[0, P)` (ids are
    slot indices of the surviving clusters, not contiguous). Ward distance: `s_a s_b/(s_a+s_b) * ||c_a - c_b||^2`, merges
    forbidden above `max_pools_per_bucket`. Each round merges all *mutual* nearest-neighbour pairs (never dropping a block
    below `target_buckets`); if a block has none left but is still above target, its globally closest valid pair is merged
    so every block makes progress.
    """
    b, p, d = pools.shape
    dev = pools.device
    x = pools.float()
    sums = x.clone()  # [B, P, d]
    sizes = torch.ones(b, p, device=dev)
    active = torch.ones(b, p, dtype=torch.bool, device=dev)
    labels = torch.arange(p, device=dev).unsqueeze(0).repeat(b, 1)
    big = torch.tensor(float("inf"), device=dev)
    eye = torch.eye(p, dtype=torch.bool, device=dev).unsqueeze(0)
    idx = torch.arange(p, device=dev).unsqueeze(0).expand(b, p)

    for _ in range(max_rounds):
        n_act = active.sum(dim=1)  # [B]
        need = n_act > target_buckets
        if not bool(need.any()):
            break
        cent = sums / sizes.clamp(min=1).unsqueeze(-1)
        d2 = torch.cdist(cent, cent).pow(2)  # [B, P, P]
        ssum = sizes.unsqueeze(2) + sizes.unsqueeze(1)
        ward = d2 * (sizes.unsqueeze(2) * sizes.unsqueeze(1) / ssum.clamp(min=1))
        invalid = eye | ~active.unsqueeze(2) | ~active.unsqueeze(1) | (ssum > max_pools_per_bucket) | ~need.view(b, 1, 1)
        ward = torch.where(invalid, big, ward)
        nn_d, nn_i = ward.min(dim=2)  # [B, P]
        mutual = (nn_i.gather(1, nn_i) == idx) & torch.isfinite(nn_d) & (nn_i > idx)  # pair (i, nn_i) with i < nn_i
        budget = (n_act - target_buckets).clamp(min=0)  # merges allowed per block
        rank = torch.cumsum(mutual.long(), dim=1)
        mutual = mutual & (rank <= budget.unsqueeze(1))
        if not bool(mutual.any()):
            flat = ward.view(b, -1)
            best_d, best_f = flat.min(dim=1)
            ok = torch.isfinite(best_d) & need
            if not bool(ok.any()):
                break
            bi = torch.nonzero(ok).squeeze(1)
            i_idx, j_idx = best_f[bi] // p, best_f[bi] % p
            lo, hi = torch.minimum(i_idx, j_idx), torch.maximum(i_idx, j_idx)
            mutual = torch.zeros(b, p, dtype=torch.bool, device=dev)
            mutual[bi, lo] = True
            nn_i = nn_i.clone()
            nn_i[bi, lo] = hi
        src = torch.nonzero(mutual)  # [M, 2] -> (block, i)
        if src.numel() == 0:
            break
        bb, ii = src[:, 0], src[:, 1]
        jj = nn_i[bb, ii]  # merge j into i
        sums[bb, ii] = sums[bb, ii] + sums[bb, jj]
        sizes[bb, ii] = sizes[bb, ii] + sizes[bb, jj]
        active[bb, jj] = False
        sizes[bb, jj] = 0
        sums[bb, jj] = 0
        remap = torch.arange(p, device=dev).unsqueeze(0).repeat(b, 1)
        remap[bb, jj] = ii  # merged pairs are disjoint, so a single hop suffices
        labels = remap.gather(1, labels)
    return labels


# ======================================================================================
# Paged latent index with exact-key selection
# ======================================================================================
class PagedLatentIndex:
    """Hierarchical (pool -> Ward bucket -> optional geometric tree) index over the *middle* region `[n_sink, T - tail)` of a key sequence.

    Only complete blocks of `n_cluster` keys are clustered; the remainder (< n_cluster keys) is returned as part of the
    always-read tail. Selection returns **exact key positions** (sink + selected buckets + tail), never averaged values.

    Loop72 LSA scaling upgrades (§3.3, §4.2, §5.3, P2, P6):
      - Append-only block pages (`_cent_pages`, `_label_pages`, `_btok_pages`) eliminate quadratic full-prefix `torch.cat` traffic.
      - Source-side candidate gather (`index_select` on `k_idx_n`'s device before `.to(dev)`) prevents full-history host->GPU copy.
      - Optional `persistent_token_latents=False` avoids keeping full-history FP32 64D per-token arrays in RAM/VRAM; stage-2
        reranking can evaluate an on-the-fly `candidate_key_fn(mid_indices)` callback.
      - Supports both `search_mode="flat"` (exact bucket scan) and `search_mode="tree"` (bounded-beam geometric-series
        hierarchical traversal in $O(r f w \\log_f P + K r)$ time, strictly respecting causal block horizon `limit`).
    """

    def __init__(self, k_idx: torch.Tensor, n_sink: int = 4, n_pool: int = 4, n_cluster: int = 512,
                 avg_bucket_size: int = 8, max_bucket_size: int = 16, window: int = 512, rerank: int = 1,
                 search_mode: str = "flat", branching_factor: int = 16, beam_width: int = 32,
                 persistent_token_latents: Optional[bool] = None):
        assert n_cluster % n_pool == 0
        self.rerank = max(1, int(rerank))  # two-stage: buckets up to rerank*budget tokens, then token-level top-`budget` on k_idx
        self.search_mode = str(search_mode).lower().strip()
        self.branching_factor = max(2, int(branching_factor))
        self.beam_width = max(4, int(beam_width))
        if persistent_token_latents is None:
            persistent_token_latents = bool(self.rerank > 1)
        self.persistent_token_latents = bool(persistent_token_latents)
        self.t = int(k_idx.shape[0])
        self.n_sink, self.n_pool, self.n_cluster = int(n_sink), int(n_pool), int(n_cluster)
        self.window = int(window)
        self.d = int(k_idx.shape[-1])
        d = self.d
        mid_end = max(self.n_sink, self.t - self.window)
        n_blocks = max(0, (mid_end - self.n_sink) // self.n_cluster)
        self.n_blocks = n_blocks
        self.mid_start = self.n_sink
        self.clustered_end = self.n_sink + n_blocks * self.n_cluster  # keys in [clustered_end, T) are always read
        self.pools_per_block = self.n_cluster // self.n_pool
        self.avg_bucket_size = int(avg_bucket_size)
        self.max_bucket_size = int(max_bucket_size)

        self._cent_pages: List[torch.Tensor] = []
        self._label_pages: List[torch.Tensor] = []
        self._btok_pages: List[torch.Tensor] = []
        self._k_idx_pages: List[torch.Tensor] = []
        self._materialized_blocks: int = -1
        self._mat_centroids: torch.Tensor = torch.zeros(0, 0, d)
        self._mat_labels: torch.Tensor = torch.zeros(0, 0, dtype=torch.long)
        self._mat_btoks: torch.Tensor = torch.zeros(0, 0)
        self._mat_k_idx_n: Optional[torch.Tensor] = None
        self._k_idx_mat_t: int = -1

        if self.persistent_token_latents and self.t > 0:
            kk = F.normalize(k_idx.float(), dim=-1, eps=1e-6)
            self._k_idx_pages.append(kk)
        else:
            kk = F.normalize(k_idx[self.mid_start: self.clustered_end].float(), dim=-1, eps=1e-6) if n_blocks > 0 else None

        if n_blocks == 0:
            return

        blk_src = kk[self.mid_start: self.clustered_end] if self.persistent_token_latents else kk
        blk = blk_src.view(n_blocks, self.pools_per_block, self.n_pool, d)
        pools = F.normalize(blk.mean(dim=2), dim=-1, eps=1e-6)  # [B, P, d]
        target = max(1, self.n_cluster // int(avg_bucket_size))
        labels = ward_partition(pools, target, max_pools_per_bucket=max(1, int(max_bucket_size) // self.n_pool))
        p = self.pools_per_block
        onehot = F.one_hot(labels, num_classes=p).float()  # [B, P, C]
        sums = torch.einsum("bpc,bpd->bcd", onehot, pools)
        cents = F.normalize(sums, dim=-1, eps=1e-6)  # [B, C, d]
        btoks = onehot.sum(dim=1) * self.n_pool  # [B, C] token count per bucket (0 = empty slot)
        self._cent_pages.append(cents)
        self._label_pages.append(labels)
        self._btok_pages.append(btoks)

    def _ensure_materialized(self) -> None:
        if self._materialized_blocks == self.n_blocks:
            return
        if not self._cent_pages or self.n_blocks == 0:
            self._mat_centroids = torch.zeros(0, 0, self.d)
            self._mat_labels = torch.zeros(0, 0, dtype=torch.long)
            self._mat_btoks = torch.zeros(0, 0)
        elif len(self._cent_pages) == 1:
            self._mat_centroids = self._cent_pages[0]
            self._mat_labels = self._label_pages[0]
            self._mat_btoks = self._btok_pages[0]
        else:
            dev = self._cent_pages[0].device
            ldev = self._label_pages[0].device
            self._mat_centroids = torch.cat([c.to(dev) for c in self._cent_pages], dim=0)
            self._mat_labels = torch.cat([l.to(ldev) for l in self._label_pages], dim=0)
            self._mat_btoks = torch.cat([b.to(dev) for b in self._btok_pages], dim=0)
            self._cent_pages = [self._mat_centroids]
            self._label_pages = [self._mat_labels]
            self._btok_pages = [self._mat_btoks]
        self._materialized_blocks = self.n_blocks

    @property
    def centroids(self) -> torch.Tensor:
        self._ensure_materialized()
        return self._mat_centroids

    @centroids.setter
    def centroids(self, val: torch.Tensor) -> None:
        self._cent_pages = [val] if val.numel() > 0 else []
        self._mat_centroids = val
        self._materialized_blocks = self.n_blocks

    @property
    def labels(self) -> torch.Tensor:
        self._ensure_materialized()
        return self._mat_labels

    @labels.setter
    def labels(self, val: torch.Tensor) -> None:
        self._label_pages = [val] if val.numel() > 0 else []
        self._mat_labels = val
        self._materialized_blocks = self.n_blocks

    @property
    def bucket_tokens(self) -> torch.Tensor:
        self._ensure_materialized()
        return self._mat_btoks

    @bucket_tokens.setter
    def bucket_tokens(self, val: torch.Tensor) -> None:
        self._btok_pages = [val] if val.numel() > 0 else []
        self._mat_btoks = val
        self._materialized_blocks = self.n_blocks

    @property
    def k_idx_n(self) -> Optional[torch.Tensor]:
        if not self.persistent_token_latents or not self._k_idx_pages:
            return None
        if self._k_idx_mat_t != self.t or self._mat_k_idx_n is None:
            if len(self._k_idx_pages) == 1:
                self._mat_k_idx_n = self._k_idx_pages[0]
            else:
                dev = self._k_idx_pages[0].device
                self._mat_k_idx_n = torch.cat([p.to(dev) for p in self._k_idx_pages], dim=0)
                self._k_idx_pages = [self._mat_k_idx_n]
            self._k_idx_mat_t = self.t
        return self._mat_k_idx_n

    @k_idx_n.setter
    def k_idx_n(self, val: Optional[torch.Tensor]) -> None:
        if val is None:
            self._k_idx_pages = []
            self._mat_k_idx_n = None
            self._k_idx_mat_t = -1
        else:
            self._k_idx_pages = [val]
            self._mat_k_idx_n = val
            self._k_idx_mat_t = int(val.shape[0])

    @torch.no_grad()
    def append_new_keys(self, k_idx_new: torch.Tensor) -> int:
        """Append only newly arrived key latents `k_idx_new` (`[Delta_T, d]`) without passing the full prefix."""
        delta_t = int(k_idx_new.shape[0])
        if delta_t <= 0:
            return 0
        # Keep an unclustered tail buffer of at most `n_cluster + window` key latents so block closing never needs full-history k_idx
        tail_buf = getattr(self, "_unclustered_k_buf", None)
        kk_new = F.normalize(k_idx_new.float(), dim=-1, eps=1e-6)
        if self.persistent_token_latents:
            self._k_idx_pages.append(kk_new)
        if tail_buf is None:
            if self.persistent_token_latents and self.k_idx_n is not None:
                # Extract from the already-stored pages starting at clustered_end
                full_k = self.k_idx_n
                old_t = self.t
                self.t = old_t + delta_t
                return self._cluster_new_blocks_from_slice(full_k[self.clustered_end: self.t])
            tail_buf = kk_new
        else:
            tail_buf = torch.cat([tail_buf.to(kk_new.device), kk_new], dim=0)
        self.t += delta_t
        added = self._cluster_new_blocks_from_slice(tail_buf)
        consumed = added * self.n_cluster
        self._unclustered_k_buf = tail_buf[consumed:]
        return added

    def _cluster_new_blocks_from_slice(self, unclustered_slice: torch.Tensor) -> int:
        mid_end = max(self.n_sink, self.t - self.window)
        new_blocks = max(0, (mid_end - self.n_sink) // self.n_cluster)
        delta_blocks = new_blocks - self.n_blocks
        if delta_blocks <= 0:
            return 0
        need_toks = delta_blocks * self.n_cluster
        if int(unclustered_slice.shape[0]) < need_toks:
            return 0
        kk_new = unclustered_slice[:need_toks]
        blk = kk_new.view(delta_blocks, self.pools_per_block, self.n_pool, self.d)
        pools = F.normalize(blk.mean(dim=2), dim=-1, eps=1e-6)
        avg_bs = getattr(self, "avg_bucket_size", 8)
        max_bs = getattr(self, "max_bucket_size", 16)
        target = max(1, self.n_cluster // int(avg_bs))
        new_labels = ward_partition(pools, target, max_pools_per_bucket=max(1, int(max_bs) // self.n_pool))
        p = self.pools_per_block
        onehot = F.one_hot(new_labels, num_classes=p).float()
        sums = torch.einsum("bpc,bpd->bcd", onehot, pools)
        new_centroids = F.normalize(sums, dim=-1, eps=1e-6)
        new_btoks = onehot.sum(dim=1) * self.n_pool
        self._cent_pages.append(new_centroids)
        self._label_pages.append(new_labels)
        self._btok_pages.append(new_btoks)
        self.n_blocks = new_blocks
        self.clustered_end = self.n_sink + new_blocks * self.n_cluster
        return int(delta_blocks)

    @torch.no_grad()
    def extend_to(self, k_idx: torch.Tensor) -> int:
        """Incrementally extend the index to a longer key sequence `k_idx` (`[T_new, d]`, `T_new >= self.t`) without
        re-clustering already-completed blocks `[mid_start, clustered_end)`. Returns the number of newly clustered blocks."""
        new_t = int(k_idx.shape[0])
        if new_t < self.t:
            raise ValueError(f"extend_to requires new_t ({new_t}) >= current t ({self.t})")
        if new_t == self.t:
            return 0
        old_t = self.t
        self.t = new_t
        if self.persistent_token_latents:
            kk_tail = F.normalize(k_idx[old_t:new_t].float(), dim=-1, eps=1e-6)
            self._k_idx_pages.append(kk_tail)
        mid_end = max(self.n_sink, new_t - self.window)
        new_blocks = max(0, (mid_end - self.n_sink) // self.n_cluster)
        delta_blocks = new_blocks - self.n_blocks
        if delta_blocks <= 0:
            return 0
        start = self.clustered_end
        end = self.n_sink + new_blocks * self.n_cluster
        kk_new = F.normalize(k_idx[start:end].float(), dim=-1, eps=1e-6)
        return self._cluster_new_blocks_from_slice(kk_new)

    def bucket_stats(self) -> Dict[str, float]:
        if self.n_blocks == 0:
            return {"blocks": 0}
        nz = self.bucket_tokens[self.bucket_tokens > 0]
        return {"blocks": self.n_blocks, "buckets": int(nz.numel()), "mean_tokens": float(nz.mean()),
                "max_tokens": float(nz.max())}

    def index_storage_bytes(self) -> int:
        """Compute resident index storage bytes (centroids + labels + bucket token counts + optional token latents)."""
        total = 0
        for page_list in (self._cent_pages, self._label_pages, self._btok_pages, self._k_idx_pages):
            for t_obj in page_list:
                if isinstance(t_obj, torch.Tensor):
                    total += int(t_obj.numel() * t_obj.element_size())
        return total

    @torch.no_grad()
    def _select_tree_buckets(
        self,
        cents: torch.Tensor,
        toks: torch.Tensor,
        q: torch.Tensor,
        stage1_token_budget: int,
        upto: int,
        dist_alpha_over_tau: float = 0.0,
    ) -> torch.Tensor:
        """Bounded-beam geometric-series hierarchical tree search over causal blocks `[0, limit)`.
        Complexity: $O(r f w \\log_f P + K r)$ where $P$ is the number of non-empty leaf buckets, $f$ is `branching_factor`,
        and $w$ is `beam_width`."""
        limit, c_slots, d = cents.shape
        dev = cents.device
        flat_toks = toks.reshape(-1)
        valid_mask = flat_toks > 0
        valid_flat_ids = torch.nonzero(valid_mask).squeeze(1)
        n_valid = int(valid_flat_ids.numel())
        if n_valid == 0:
            return valid_flat_ids

        f = self.branching_factor
        target_leaf_buckets = max(self.beam_width, int(math.ceil(stage1_token_budget / max(1, self.avg_bucket_size))) * 2)
        if n_valid <= max(f * 2, target_leaf_buckets):
            # Small enough to score valid leaf buckets directly
            cand_flat = valid_flat_ids
        else:
            valid_cents = cents.reshape(-1, d)[valid_flat_ids]
            # Build geometric hierarchy levels of branching factor `f` over valid causal buckets
            levels_cents: List[torch.Tensor] = [valid_cents]
            cur = valid_cents
            while int(cur.shape[0]) > f * self.beam_width:
                n_cur = int(cur.shape[0])
                n_parent = (n_cur + f - 1) // f
                pad = n_parent * f - n_cur
                if pad > 0:
                    cur_padded = torch.cat([cur, torch.zeros(pad, d, device=dev, dtype=cur.dtype)], dim=0)
                else:
                    cur_padded = cur
                cur = F.normalize(cur_padded.view(n_parent, f, d).sum(dim=1), dim=-1, eps=1e-6)
                levels_cents.append(cur)

            # Top-down bounded-beam traversal from coarsest level to leaf valid indices
            top_cents = levels_cents[-1]
            top_scores = top_cents @ q
            w_beam = min(int(top_cents.shape[0]), max(self.beam_width, target_leaf_buckets // f))
            active_idx = top_scores.topk(w_beam, largest=True, sorted=False).indices

            for lvl in range(len(levels_cents) - 2, -1, -1):
                lvl_cents = levels_cents[lvl]
                n_lvl = int(lvl_cents.shape[0])
                children = (
                    active_idx.unsqueeze(1) * f + torch.arange(f, device=dev, dtype=torch.long).unsqueeze(0)
                ).reshape(-1)
                children = children[children < n_lvl]
                ch_scores = lvl_cents.index_select(0, children) @ q
                keep_k = min(int(children.numel()), target_leaf_buckets if lvl == 0 else max(self.beam_width * 2, target_leaf_buckets // f))
                active_idx = children[ch_scores.topk(keep_k, largest=True, sorted=False).indices]

            cand_flat = valid_flat_ids.index_select(0, active_idx)

        cand_cents = cents.reshape(-1, d).index_select(0, cand_flat)
        cand_scores = cand_cents @ q
        if dist_alpha_over_tau > 0.0:
            q_pos = float(upto - 1)
            blk_ids = cand_flat // c_slots
            blk_centers = self.mid_start + (blk_ids.float() + 0.5) * float(self.n_cluster)
            cand_scores = cand_scores - float(dist_alpha_over_tau) * torch.log1p(
                (q_pos - blk_centers).clamp(min=0.0) / max(1.0, float(self.window))
            )
        order = cand_scores.argsort(descending=True)
        sorted_flat = cand_flat.index_select(0, order)
        sorted_toks = flat_toks.index_select(0, sorted_flat)
        csum = torch.cumsum(sorted_toks, dim=0)
        take = (csum - sorted_toks) < stage1_token_budget
        return sorted_flat[take]

    @torch.no_grad()
    def select(self, q_idx: torch.Tensor, budget: int, upto: Optional[int] = None,
               dist_alpha_over_tau: float = 0.0, search_mode: Optional[str] = None,
               candidate_key_fn: Optional[Any] = None) -> torch.Tensor:
        """`q_idx`: [d] or [Tq, d] unit-norm (mean-pooled if 2-D). Returns sorted exact key positions (sink+selected+tail).

        `upto`: causal horizon for a query at position `upto - 1` (keys `>= upto` do not exist for it). Because Ward blocks are
        independent, restricting to the blocks that end before `upto - window` is identical to re-indexing `k[:upto]`; this lets
        one index serve every decode step (and grow incrementally as the cache grows).
        `dist_alpha_over_tau`: optional query-time log-distance decay weight `alpha / tau >= 0` (leaves `k_idx` and Ward centroids
        100% position-free and static while accounting for RoPE recency decay at query time).
        """
        dev = q_idx.device
        upto = self.t if upto is None else min(int(upto), self.t)
        limit = 0 if self.n_blocks == 0 else max(0, min(self.n_blocks, (upto - self.window - self.mid_start) // self.n_cluster))
        clustered_end = self.mid_start + limit * self.n_cluster
        sink = torch.arange(0, min(self.n_sink, upto), device=dev)
        tail = torch.arange(clustered_end, upto, device=dev)
        if limit == 0:
            return torch.cat([sink, tail]).unique()
        q = F.normalize(q_idx.float().reshape(-1, q_idx.shape[-1]).mean(dim=0), dim=-1, eps=1e-6)
        cents = self.centroids[:limit].to(dev)
        toks = self.bucket_tokens[:limit].to(dev)
        stage1_budget = int(budget) * self.rerank
        eff_mode = str(search_mode or self.search_mode).lower().strip()

        if eff_mode == "tree":
            sel_flat = self._select_tree_buckets(
                cents, toks, q, stage1_token_budget=stage1_budget, upto=upto, dist_alpha_over_tau=dist_alpha_over_tau
            )
        else:
            scores = torch.einsum("bcd,d->bc", cents, q)
            if dist_alpha_over_tau > 0.0:
                q_pos = float(upto - 1)
                blk_centers = self.mid_start + (torch.arange(limit, device=dev, dtype=torch.float32) + 0.5) * float(self.n_cluster)
                blk_pen = float(dist_alpha_over_tau) * torch.log1p((q_pos - blk_centers).clamp(min=0.0) / max(1.0, float(self.window)))
                scores = scores - blk_pen.unsqueeze(1)
            scores = scores.masked_fill(toks == 0, float("-inf"))
            flat_s, flat_i = scores.reshape(-1).sort(descending=True)
            flat_t = toks.reshape(-1)[flat_i]
            csum = torch.cumsum(flat_t, dim=0)
            take = (csum - flat_t) < stage1_budget  # include a bucket while the (stage-1) budget is not yet exhausted
            take &= torch.isfinite(flat_s)
            sel_flat = flat_i[take]

        sel_mask = torch.zeros(limit * self.pools_per_block, dtype=torch.bool, device=dev)
        sel_mask[sel_flat] = True
        sel_mask = sel_mask.view(limit, self.pools_per_block)  # [B, C]
        pool_sel = sel_mask.gather(1, self.labels[:limit].to(dev))  # [B, P]
        pool_ids = torch.nonzero(pool_sel)  # [M, 2]
        base = self.mid_start + pool_ids[:, 0] * self.n_cluster + pool_ids[:, 1] * self.n_pool
        mid = (base.unsqueeze(1) + torch.arange(self.n_pool, device=dev).unsqueeze(0)).reshape(-1)
        if self.rerank > 1 and mid.numel() > int(budget):
            cand_k_lat: Optional[torch.Tensor] = None
            if candidate_key_fn is not None:
                cand_k_lat = candidate_key_fn(mid).to(device=dev, dtype=q.dtype)
            elif self.k_idx_n is not None:
                # Fix §3.3: gather candidates on the source device first instead of copying full-history k_idx_n to `dev`
                k_src = self.k_idx_n
                cand_k_lat = k_src.index_select(0, mid.to(k_src.device)).to(device=dev, dtype=q.dtype)
            if cand_k_lat is not None:
                tok_scores = cand_k_lat @ q  # stage 2: token-level scores of the candidates only
                if dist_alpha_over_tau > 0.0:
                    q_pos = float(upto - 1)
                    tok_pen = float(dist_alpha_over_tau) * torch.log1p((q_pos - mid.float()).clamp(min=0.0) / max(1.0, float(self.window)))
                    tok_scores = tok_scores - tok_pen
                mid = mid[tok_scores.topk(int(budget)).indices]
        return torch.cat([sink, mid, tail]).unique()  # `unique` sorts ascending


# ======================================================================================
# Learned index space + per-key KL distillation
# ======================================================================================
class FaithfulDLSAIndexer(nn.Module):
    """Per-layer learned maps `W_q, W_k: (n_kv*F) -> d_idx` (+ learned temperature and optional query-time log-distance prior),
    L2-normalised.

    Initialised from PCA of the position-free key inputs (training-free D-LSA) via `fit_pca`, then refined by `distill`.
    """

    def __init__(self, n_kv: int = 4, head_dim: int = 256, rope_dim: int = 64, d_idx: int = 64, rope_first: bool = True,
                 n_sink: int = 4, n_pool: int = 4, n_cluster: int = 512, avg_bucket_size: int = 8, max_bucket_size: int = 16,
                 window: int = 512, k_long_range: int = 2048, rerank: int = 1,
                 selection_reuse_steps: int = 1, reuse_cos_threshold: float = 0.92,
                 search_mode: str = "flat", branching_factor: int = 16, beam_width: int = 32,
                 persistent_token_latents: Optional[bool] = None):
        super().__init__()
        self.rerank = max(1, int(rerank))
        self.selection_reuse_steps = max(1, int(selection_reuse_steps))
        self.reuse_cos_threshold = float(reuse_cos_threshold)
        self.search_mode = str(search_mode).lower().strip()
        self.branching_factor = max(2, int(branching_factor))
        self.beam_width = max(4, int(beam_width))
        self.persistent_token_latents = persistent_token_latents
        self.n_kv, self.head_dim, self.rope_dim, self.d_idx, self.rope_first = n_kv, head_dim, rope_dim, d_idx, rope_first
        self.n_sink, self.n_pool, self.n_cluster = n_sink, n_pool, n_cluster
        self.avg_bucket_size, self.max_bucket_size, self.window, self.k_long_range = avg_bucket_size, max_bucket_size, window, k_long_range
        din = n_kv * (head_dim - rope_dim)
        self.w_q = nn.Linear(din, d_idx, bias=False)
        self.w_k = nn.Linear(din, d_idx, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(12.0)))
        # Query-time log-distance prior parameter (zero at init = -6.0; non-negative when distilled)
        self.raw_dist_alpha = nn.Parameter(torch.tensor(-6.0))
        nn.init.orthogonal_(self.w_k.weight)
        self.w_q.weight.data.copy_(self.w_k.weight.data)
        self._visit_state_slots: Dict[Any, Dict[str, Any]] = {}

    def dist_alpha(self) -> torch.Tensor:
        """Non-negative query-time log-distance decay weight `alpha >= 0` (exactly `0.0` at init `-6.0`)."""
        raw = getattr(self, "raw_dist_alpha", None)
        if raw is None:
            return torch.tensor(0.0, device=self.log_tau.device)
        base = F.softplus(torch.tensor(-6.0, device=raw.device, dtype=torch.float32))
        return (F.softplus(raw.float()) - base).clamp(min=0.0, max=8.0)

    def dist_alpha_over_tau(self) -> float:
        tau = float(self.log_tau.detach().float().exp().clamp(1.0, 64.0).item())
        return float(self.dist_alpha().detach().item()) / tau

    # ---- projections
    def project_q(self, q_in: torch.Tensor) -> torch.Tensor:
        return F.normalize(F.linear(q_in.float(), self.w_q.weight.float()), dim=-1, eps=1e-6)

    def project_k(self, k_in: torch.Tensor) -> torch.Tensor:
        return F.normalize(F.linear(k_in.float(), self.w_k.weight.float()), dim=-1, eps=1e-6)

    def inputs(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return free_slice_inputs(q, k, self.rope_dim, self.rope_first)

    @torch.no_grad()
    def fit_pca(self, k_in: torch.Tensor) -> None:
        """Training-free init: top-`d_idx` principal directions of centred key inputs (shared by W_q and W_k so the
        initial score is the cosine in the dominant key subspace)."""
        x = k_in.float()
        x = x - x.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        top = vh[: self.d_idx]
        w = torch.zeros_like(self.w_k.weight)
        w[: top.shape[0]] = top
        self.w_k.weight.copy_(w)
        self.w_q.weight.copy_(w)

    # ---- selection (drop-in for the in-model API)
    @torch.no_grad()
    def build_index(self, k: torch.Tensor) -> PagedLatentIndex:
        """`k`: [H_kv, T, D] -> index over this layer's keys."""
        _, k_in = self.inputs(_dummy_q(k), k)
        return PagedLatentIndex(
            self.project_k(k_in),
            self.n_sink,
            self.n_pool,
            self.n_cluster,
            self.avg_bucket_size,
            self.max_bucket_size,
            self.window,
            self.rerank,
            search_mode=self.search_mode,
            branching_factor=self.branching_factor,
            beam_width=self.beam_width,
            persistent_token_latents=self.persistent_token_latents,
        )

    def reset_decode_cache(self, visit_key: Any = None) -> None:
        """Clear cached decode `PagedLatentIndex` and step-to-step selection reuse state (for `visit_key` or all visits)."""
        if visit_key is not None and hasattr(self, "_visit_state_slots"):
            self._visit_state_slots.pop(visit_key, None)
            return
        self._visit_state_slots = {}
        self._cached_index: Optional[PagedLatentIndex] = None
        self._cached_k_idx: Optional[torch.Tensor] = None
        self._cached_sig: Optional[Tuple[Any, ...]] = None
        self._last_q_vec: Optional[torch.Tensor] = None
        self._last_mid_sel: Optional[torch.Tensor] = None
        self._last_clustered_end: int = -1
        self._reuse_age: int = 0
        self.decode_cache_hits: int = 0
        self.decode_cache_misses: int = 0
        self.selection_reuse_hits: int = 0

    def _get_slot(self, visit_key: Any = None, state_slot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state_slot is not None:
            return state_slot
        if not hasattr(self, "_visit_state_slots") or self._visit_state_slots is None:
            self._visit_state_slots = {}
        key = "__default__" if visit_key is None else visit_key
        if key not in self._visit_state_slots:
            self._visit_state_slots[key] = {}
        return self._visit_state_slots[key]

    @staticmethod
    def _prefix_signature(k0: torch.Tensor, check_len: int) -> Tuple[Any, ...]:
        if check_len <= 0:
            return (int(k0.shape[0]), int(k0.shape[2]), 0)
        # Deterministic strided digest over 8 anchor positions along `check_len`
        pts = sorted({0, check_len // 4, check_len // 2, (3 * check_len) // 4, check_len - 1})
        vals = tuple(round(float(k0[0, p, p % int(k0.shape[2])].item()), 5) for p in pts)
        return (int(k0.shape[0]), int(k0.shape[2]), int(check_len), vals)

    @torch.no_grad()
    def get_or_update_index(
        self,
        k0: torch.Tensor,
        visit_key: Any = None,
        state_slot: Optional[Dict[str, Any]] = None,
    ) -> PagedLatentIndex:
        """Return a `PagedLatentIndex` for `k0` `[H_kv, T, D]`, incrementally projecting & Ward-clustering only new keys when
        `k0` extends the previously indexed sequence for this specific `(visit_key, state_slot)`."""
        t = int(k0.shape[1])
        slot = self._get_slot(visit_key=visit_key, state_slot=state_slot)
        idx: Optional[PagedLatentIndex] = slot.get("index")
        if idx is None and visit_key is None and state_slot is None:
            idx = getattr(self, "_cached_index", None)
        sig_prev = slot.get("sig", getattr(self, "_cached_sig", None) if (visit_key is None and state_slot is None) else None)

        if idx is not None and idx.t <= t and sig_prev is not None:
            if self._prefix_signature(k0, idx.t) == sig_prev:
                if t > idx.t:
                    # Only project the unclustered tail + new keys `[idx.clustered_end : t)` — no full-prefix k_idx copy!
                    tail_start = idx.clustered_end
                    _, k_in_tail = self.inputs(_dummy_q(k0[:, tail_start:t]), k0[:, tail_start:t])
                    k_idx_tail = self.project_k(k_in_tail)
                    if idx.persistent_token_latents and idx.k_idx_n is not None:
                        _, k_in_new = self.inputs(_dummy_q(k0[:, idx.t:t]), k0[:, idx.t:t])
                        k_idx_new = self.project_k(k_in_new)
                        idx._k_idx_pages.append(F.normalize(k_idx_new.float(), dim=-1, eps=1e-6))
                    idx.t = t
                    idx._cluster_new_blocks_from_slice(F.normalize(k_idx_tail.float(), dim=-1, eps=1e-6))
                    sig_prev = self._prefix_signature(k0, t)
                    slot["sig"] = sig_prev
                    if visit_key is None and state_slot is None:
                        self._cached_sig = sig_prev
                self.decode_cache_hits = int(getattr(self, "decode_cache_hits", 0)) + 1
                slot["hits"] = int(slot.get("hits", 0)) + 1
                return idx

        _, k_in = self.inputs(_dummy_q(k0), k0)
        k_idx_full = self.project_k(k_in)
        idx = PagedLatentIndex(
            k_idx_full,
            self.n_sink,
            self.n_pool,
            self.n_cluster,
            self.avg_bucket_size,
            self.max_bucket_size,
            self.window,
            self.rerank,
            search_mode=self.search_mode,
            branching_factor=self.branching_factor,
            beam_width=self.beam_width,
            persistent_token_latents=self.persistent_token_latents,
        )
        sig_now = self._prefix_signature(k0, t)
        slot["index"] = idx
        slot["sig"] = sig_now
        slot["misses"] = int(slot.get("misses", 0)) + 1
        if visit_key is None and state_slot is None:
            self._cached_index = idx
            self._cached_k_idx = k_idx_full if idx.persistent_token_latents else None
            self._cached_sig = sig_now
        self.decode_cache_misses = int(getattr(self, "decode_cache_misses", 0)) + 1
        return idx

    @torch.no_grad()
    def select_key_indices(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        k_budget: Optional[int] = None,
        use_cache: bool = True,
        visit_key: Any = None,
        state_slot: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """API-compatible with `LatentSparseAttentionIndexer.select_key_indices`: `query` [1, H_q, T_q, D], `key`
        [1, H_kv, T_k, D] -> exact key positions shared by all heads.

        State is isolated by `(visit_key, state_slot)` so recurrent visits to the same physical layer never collide.
        """
        assert query.shape[0] == 1 and key.shape[0] == 1
        budget = int(k_budget or self.k_long_range)
        t = key.shape[2]
        if t <= self.n_sink + self.window + budget:
            return torch.arange(t, device=key.device)
        slot = self._get_slot(visit_key=visit_key, state_slot=state_slot)
        idx = (
            self.get_or_update_index(key[0], visit_key=visit_key, state_slot=slot)
            if use_cache
            else self.build_index(key[0])
        )
        q_in, _ = self.inputs(query[0], key[0][:, :1])
        q_proj = self.project_q(q_in)
        q_vec = F.normalize(q_proj.float().reshape(-1, q_proj.shape[-1]).mean(dim=0), dim=-1, eps=1e-6)
        reuse_steps = int(getattr(self, "selection_reuse_steps", 1) or 1)
        reuse_thr = float(getattr(self, "reuse_cos_threshold", 0.92) or 0.92)
        last_q = slot.get("last_q_vec", getattr(self, "_last_q_vec", None) if (visit_key is None and state_slot is None) else None)
        last_mid = slot.get("last_mid_sel", getattr(self, "_last_mid_sel", None) if (visit_key is None and state_slot is None) else None)
        last_ce = int(slot.get("last_clustered_end", getattr(self, "_last_clustered_end", -1) if (visit_key is None and state_slot is None) else -1))
        age = int(slot.get("reuse_age", getattr(self, "_reuse_age", 0) if (visit_key is None and state_slot is None) else 0))
        if (
            use_cache
            and reuse_steps > 1
            and query.shape[2] == 1
            and last_q is not None
            and last_mid is not None
            and last_ce == idx.clustered_end
            and age + 1 < reuse_steps
            and float(torch.dot(q_vec, last_q.to(q_vec.device)).item()) >= reuse_thr
        ):
            slot["reuse_age"] = age + 1
            if visit_key is None and state_slot is None:
                self._reuse_age = age + 1
            self.selection_reuse_hits = int(getattr(self, "selection_reuse_hits", 0)) + 1
            sink = torch.arange(0, min(self.n_sink, t), device=key.device)
            tail = torch.arange(idx.clustered_end, t, device=key.device)
            return torch.cat([sink, last_mid.to(key.device), tail]).unique()

        def _cand_key_fn(mid_indices: torch.Tensor) -> torch.Tensor:
            k_cand = key[0].index_select(1, mid_indices.to(key.device))
            _, k_in_c = self.inputs(_dummy_q(k_cand), k_cand)
            return self.project_k(k_in_c)

        sel = idx.select(
            q_proj,
            budget,
            upto=t,
            dist_alpha_over_tau=self.dist_alpha_over_tau(),
            search_mode=self.search_mode,
            candidate_key_fn=_cand_key_fn if not idx.persistent_token_latents else None,
        ).to(key.device)
        if use_cache and reuse_steps > 1 and query.shape[2] == 1:
            mid_mask = (sel >= self.n_sink) & (sel < idx.clustered_end)
            slot["last_q_vec"] = q_vec.detach()
            slot["last_mid_sel"] = sel[mid_mask].detach()
            slot["last_clustered_end"] = idx.clustered_end
            slot["reuse_age"] = 0
            if visit_key is None and state_slot is None:
                self._last_q_vec = q_vec.detach()
                self._last_mid_sel = sel[mid_mask].detach()
                self._last_clustered_end = idx.clustered_end
                self._reuse_age = 0
        return sel

    # ---- distillation
    def distill_loss(self, q_in: torch.Tensor, k_in: torch.Tensor, teacher_p: torch.Tensor,
                     key_mask: Optional[torch.Tensor] = None,
                     q_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-key KL(teacher || student) averaged over queries. `teacher_p`: [Tq, T] (rows sum to 1 over the allowed keys),
        `key_mask`: [Tq, T] bool of keys the student must rank (e.g. middle region, causal).
        When `q_positions` (`[Tq]`) is provided, includes the query-time log-distance prior `-alpha * log(1 + (p - j)/window)`."""
        tau = self.log_tau.float().exp().clamp(1.0, 64.0)
        logits = tau * self.project_q(q_in) @ self.project_k(k_in).t()
        if q_positions is not None:
            alpha = self.dist_alpha()
            q_pos = q_positions.to(device=logits.device, dtype=torch.float32).unsqueeze(1)
            k_pos = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32).unsqueeze(0)
            dist_pen = alpha * torch.log1p((q_pos - k_pos).clamp(min=0.0) / max(1.0, float(self.window)))
            logits = logits - dist_pen
        if key_mask is not None:
            logits = logits.masked_fill(~key_mask, float("-inf"))
        logp = torch.log_softmax(logits, dim=-1)
        p = teacher_p.clamp(min=0)
        term = torch.where(p > 0, p * (torch.log(p.clamp(min=1e-12)) - logp), torch.zeros_like(p))
        return term.sum(dim=-1).mean()


def _dummy_q(k: torch.Tensor) -> torch.Tensor:
    """A zero query tensor shaped [H_kv, 1, D] so `free_slice_inputs` can be reused to build key inputs only."""
    return torch.zeros(k.shape[0], 1, k.shape[2], dtype=k.dtype, device=k.device)


def teacher_attention(q: torch.Tensor, k: torch.Tensor, q_positions: Sequence[int], scale: Optional[float] = None) -> torch.Tensor:
    """Exact causal attention probabilities of the frozen model for sampled query positions, **mean over heads**.
    `q`: [H_q, T, D], `k`: [H_kv, T, D] -> [len(q_positions), T]. Loops over heads to bound memory."""
    hq, t, d = q.shape
    g = hq // k.shape[0]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    pos = torch.as_tensor(list(q_positions), device=q.device)
    out = torch.zeros(len(q_positions), t, device=q.device)
    kpos = torch.arange(t, device=q.device).unsqueeze(0)
    for h in range(hq):
        logits = (q[h, pos].float() @ k[h // g].float().t()) * scale
        logits = logits.masked_fill(kpos > pos.unsqueeze(1), float("-inf"))
        out += torch.softmax(logits, dim=-1)
    return out / hq


def teacher_attention_heads(q: torch.Tensor, k: torch.Tensor, q_positions: Sequence[int], scale: Optional[float] = None) -> torch.Tensor:
    """Per-head causal attention probabilities `[H_q, len(q_positions), T]` (diagnostics: shared vs per-group vs per-head selection)."""
    hq, t, d = q.shape
    g = hq // k.shape[0]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    pos = torch.as_tensor(list(q_positions), device=q.device)
    out = torch.zeros(hq, len(q_positions), t, device=q.device)
    kpos = torch.arange(t, device=q.device).unsqueeze(0)
    for h in range(hq):
        logits = (q[h, pos].float() @ k[h // g].float().t()) * scale
        out[h] = torch.softmax(logits.masked_fill(kpos > pos.unsqueeze(1), float("-inf")), dim=-1)
    return out


def mid_region_mask(q_positions: Sequence[int], t: int, n_sink: int, window: int, device: Any = "cpu") -> torch.Tensor:
    """Keys the indexer must rank for each query: causal, excluding sink and the always-read local window."""
    pos = torch.as_tensor(list(q_positions), device=device).unsqueeze(1)
    kpos = torch.arange(t, device=device).unsqueeze(0)
    return (kpos >= n_sink) & (kpos <= pos - window)


@torch.no_grad()
def eval_kl(indexer: FaithfulDLSAIndexer, samples: Sequence[Dict[str, torch.Tensor]], queries_per_sample: int = 64, seed: int = 123,
            scale: Optional[float] = None) -> Optional[float]:
    """Per-key KL (middle region) of the indexer on samples it was NOT trained on (same definition as the training loss)."""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for s in samples:
        q, k = s["q"], s["k"]
        t = q.shape[1]
        lo = indexer.n_sink + indexer.window + 64
        if t <= lo + 1:
            continue
        pos = torch.randint(lo, t, (min(queries_per_sample, t - lo),), generator=g).sort().values.tolist()
        pos_t = torch.as_tensor(pos, device=q.device, dtype=torch.long)
        q_in_all, k_in = indexer.inputs(q, k)
        mask = mid_region_mask(pos, t, indexer.n_sink, indexer.window, q.device)
        p = teacher_attention(q, k, pos, scale) * mask
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        vals.append(float(indexer.distill_loss(q_in_all[pos], k_in, p, mask, q_positions=pos_t)))
    return sum(vals) / len(vals) if vals else None


def _kl_cache(indexer: FaithfulDLSAIndexer, samples: Sequence[Dict[str, torch.Tensor]], queries_per_sample: int, g: torch.Generator,
              scale: Optional[float], focal_top_k: int = 0, focal_boost: float = 0.0) -> List[Tuple[torch.Tensor, ...]]:
    cache = []
    for s in samples:
        q, k = s["q"], s["k"]
        t = q.shape[1]
        lo = indexer.n_sink + indexer.window + 64
        if t <= lo + 1:
            continue
        pos = torch.randint(lo, t, (min(queries_per_sample, t - lo),), generator=g).sort().values.tolist()
        pos_t = torch.as_tensor(pos, device=q.device, dtype=torch.long)
        q_in_all, k_in = indexer.inputs(q, k)
        mask = mid_region_mask(pos, t, indexer.n_sink, indexer.window, q.device)
        p = teacher_attention(q, k, pos, scale)
        p = p * mask  # renormalise on the middle region: this is what the indexer is responsible for ranking
        p_raw = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        p_train = p_raw
        if focal_top_k > 0 and focal_boost > 0.0:
            k_foc = min(int(focal_top_k), p_raw.shape[-1])
            top_idx = p_raw.topk(k_foc, dim=-1).indices
            boost_mask = torch.zeros_like(p_raw).scatter_(1, top_idx, float(focal_boost))
            p_train = p_raw * (1.0 + boost_mask)
            p_train = p_train / p_train.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        cache.append((q_in_all[pos], k_in, p_train, p_raw, mask, pos_t))
    return cache


def distill_indexer(indexer: FaithfulDLSAIndexer, samples: Sequence[Dict[str, torch.Tensor]], steps: int = 200,
                    lr: float = 3e-3, queries_per_sample: int = 64, seed: int = 0, log_every: int = 50,
                    log=print, scale: Optional[float] = None, val_samples: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
                    eval_every: int = 25, weight_decay: float = 0.0,
                    focal_top_k: int = 0, focal_boost: float = 0.0) -> Dict[str, Any]:
    """Distil `W_q, W_k, tau, raw_dist_alpha` against the frozen teacher with per-key KL on the middle region.

    `samples`: dicts with `q` [H_q, T, D] and `k` [H_kv, T, D] (post-RoPE tensors captured from the frozen model on long
    documents). Queries are sampled among positions with a non-empty middle region. Returns KL before/after on all samples.

    `val_samples` (recommended): held-out samples used for **early stopping** — every `eval_every` steps the held-out KL is
    measured and the parameters with the lowest held-out KL (including the initial/PCA state) are restored at the end, so the
    result is never worse than the init on held-out KL.
    """
    g = torch.Generator().manual_seed(seed)
    cache = _kl_cache(indexer, samples, queries_per_sample, g, scale, focal_top_k=focal_top_k, focal_boost=focal_boost)
    if not cache:
        return {"kl_before": None, "kl_after": None, "n_samples": 0}
    vcache = _kl_cache(indexer, val_samples, queries_per_sample, torch.Generator().manual_seed(seed + 1), scale) if val_samples else []

    @torch.no_grad()
    def mean_kl(c) -> float:
        return float(sum(indexer.distill_loss(a, b, p_raw, m, q_positions=pos_t) for a, b, _, p_raw, m, pos_t in c) / len(c))

    kl0 = mean_kl(cache)
    best_v = mean_kl(vcache) if vcache else None
    v0 = best_v
    best_state = {k_: v.detach().clone() for k_, v in indexer.state_dict().items()} if vcache else None
    best_step = 0
    opt = torch.optim.AdamW(indexer.parameters(), lr=lr, weight_decay=weight_decay)
    for step in range(1, int(steps) + 1):
        a, b, p_train, _, m, pos_t = cache[(step - 1) % len(cache)]
        opt.zero_grad(set_to_none=True)
        loss = indexer.distill_loss(a, b, p_train, m, q_positions=pos_t)
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            log(f"[D-LSA distill] step {step}/{steps} KL={float(loss):.4f}")
        if vcache and step % int(eval_every) == 0:
            v = mean_kl(vcache)
            if v < best_v:
                best_v, best_step = v, step
                best_state = {k_: t.detach().clone() for k_, t in indexer.state_dict().items()}
    out = {"kl_before": kl0, "n_samples": len(cache)}
    if vcache:
        indexer.load_state_dict(best_state)
        out.update(val_kl_before=v0, val_kl_after=best_v, best_step=best_step, n_val_samples=len(vcache))
    out["kl_after"] = mean_kl(cache)
    out["dist_alpha"] = float(indexer.dist_alpha().detach().item())
    return out


# ======================================================================================
# NSA / MagicPIG Residual Block-Summary Compensation
# ======================================================================================
class BlockSummaryCache:
    """GPU-resident per-block sufficient statistics `(sum_K, sum_K^2, sum_V)` for `n_cluster`-token blocks (`< 0.6%` overhead).
    Synthesises DeepSeek NSA's compressed block branch (arXiv:2502.11089) with MagicPIG's partition-function bias correction
    (arXiv:2410.16179): given any fine-grained selected key set `sel`, subtracts `sel` in O(|sel|) time to obtain the **exact**
    unselected block mean key, anisotropic key variance, and mean value without reading a single unselected token from cold RAM."""

    def __init__(self, kf: torch.Tensor, vf: torch.Tensor, n_sink: int = 4, n_cluster: int = 512, window: int = 512):
        self.n_sink = int(n_sink)
        self.n_cluster = int(n_cluster)
        self.window = int(window)
        self.h_kv, self.t, self.d = kf.shape
        self.n_blocks = max(0, (self.t - self.n_sink) // self.n_cluster)
        end = self.n_sink + self.n_blocks * self.n_cluster
        if self.n_blocks > 0:
            kb = kf[:, self.n_sink:end, :].view(self.h_kv, self.n_blocks, self.n_cluster, self.d)
            vb = vf[:, self.n_sink:end, :].view(self.h_kv, self.n_blocks, self.n_cluster, self.d)
            self.k_blk_sum = kb.sum(dim=2)
            self.k_blk_sq_sum = (kb * kb).sum(dim=2)
            self.v_blk_sum = vb.sum(dim=2)
        else:
            self.k_blk_sum = kf.new_zeros((self.h_kv, 0, self.d))
            self.k_blk_sq_sum = kf.new_zeros((self.h_kv, 0, self.d))
            self.v_blk_sum = vf.new_zeros((self.h_kv, 0, self.d))


@torch.no_grad()
def sparse_attention_with_block_compensation(
    q: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    p: int,
    sel: torch.Tensor,
    scale: float,
    blk_cache: Optional[BlockSummaryCache] = None,
    cumulant_scale: float = 0.25,
) -> torch.Tensor:
    """Compute multi-head attention output `[H_q, D]` at query position `p` over selected keys `sel`, augmented with
    unselected block summary tokens `(k_unsel_mean, v_unsel_mean)` and anisotropic right-truncated cumulant log-multiplicity."""
    hq = q.shape[0]
    h_kv = kf.shape[0]
    g = hq // h_kv
    upto = int(p) + 1
    if blk_cache is None:
        blk_cache = BlockSummaryCache(kf[:, :upto], vf[:, :upto])
    limit = 0 if blk_cache.n_blocks == 0 else max(0, min(blk_cache.n_blocks, (upto - blk_cache.window - blk_cache.n_sink) // blk_cache.n_cluster))
    clustered_end = blk_cache.n_sink + limit * blk_cache.n_cluster
    if limit == 0 or sel.numel() >= upto:
        rows = []
        for h in range(hq):
            kk, vv = kf[h // g, sel], vf[h // g, sel]
            logits = (kk @ q[h, p].float()) * scale
            rows.append(torch.softmax(logits, dim=-1) @ vv)
        return torch.stack(rows, dim=0)

    in_blk = (sel >= blk_cache.n_sink) & (sel < clustered_end)
    sel_mid = sel[in_blk]
    blk_ids = (sel_mid - blk_cache.n_sink) // blk_cache.n_cluster
    sel_counts = torch.zeros(limit, device=kf.device, dtype=torch.float32)
    if blk_ids.numel() > 0:
        sel_counts.index_add_(0, blk_ids, torch.ones_like(blk_ids, dtype=torch.float32))
    unsel_counts = (float(blk_cache.n_cluster) - sel_counts).clamp(min=0.0)

    k_sum = blk_cache.k_blk_sum[:, :limit, :].clone()
    k_sq_sum = blk_cache.k_blk_sq_sum[:, :limit, :].clone()
    v_sum = blk_cache.v_blk_sum[:, :limit, :].clone()
    if sel_mid.numel() > 0:
        ks = kf[:, sel_mid, :]
        vs = vf[:, sel_mid, :]
        k_sum.index_add_(1, blk_ids, -ks)
        k_sq_sum.index_add_(1, blk_ids, -(ks * ks))
        v_sum.index_add_(1, blk_ids, -vs)

    denom = unsel_counts.clamp(min=1.0).view(1, limit, 1)
    k_unsel_mean = k_sum / denom
    v_unsel_mean = v_sum / denom
    k_unsel_var = (k_sq_sum / denom - k_unsel_mean * k_unsel_mean).clamp(min=0.0)
    frac_sq = (unsel_counts / float(blk_cache.n_cluster)) ** 2
    log_counts = torch.where(unsel_counts > 0.0, torch.log(unsel_counts.clamp(min=1e-9)), torch.full_like(unsel_counts, float("-inf")))

    rows = []
    for h in range(hq):
        gi = h // g
        q_hp = q[h, p].float()
        logits_sel = (kf[gi, sel] @ q_hp) * scale
        l_mean = (k_unsel_mean[gi] @ q_hp) * scale
        var_l = (scale * scale) * (k_unsel_var[gi] @ (q_hp * q_hp))
        cumulant = (float(cumulant_scale) * frac_sq * var_l).clamp(min=0.0, max=1.5)
        logits_blk = l_mean + cumulant + log_counts
        logits_all = torch.cat([logits_sel, logits_blk], dim=0)
        v_all = torch.cat([vf[gi, sel], v_unsel_mean[gi]], dim=0)
        rows.append(torch.softmax(logits_all, dim=-1) @ v_all)
    return torch.stack(rows, dim=0)


# ======================================================================================
# Recall metrics and baselines
# ======================================================================================
def recall_of_selection(teacher_p_row: torch.Tensor, selected: torch.Tensor) -> float:
    """Fraction of the teacher's total attention mass for one query that falls on `selected` key positions."""
    return float(teacher_p_row[selected].sum() / teacher_p_row.sum().clamp(min=1e-12))


def _out_rel_err(q: torch.Tensor, kf: torch.Tensor, vf: torch.Tensor, p: int, sels: Dict[str, torch.Tensor],
                 scale: float, blk_cache: Optional[BlockSummaryCache] = None) -> Dict[str, float]:
    """Per-head attention-OUTPUT relative error of sparse (softmax renormalised over the selected keys) vs exact causal attention
    for the query at position `p`, mean over heads. `kf`/`vf`: float [H_kv, T, D]. When `blk_cache` is provided, also computes
    `faithful_comp` using NSA/MagicPIG Residual Block-Summary Compensation."""
    hq = q.shape[0]
    g = hq // kf.shape[0]
    errs = {a: 0.0 for a in sels}
    o_full_heads = []
    den_heads = []
    for h in range(hq):
        kk, vv = kf[h // g, : p + 1], vf[h // g, : p + 1]
        logits = (kk @ q[h, p].float()) * scale
        o_full = torch.softmax(logits, dim=-1) @ vv
        den = o_full.norm().clamp(min=1e-8)
        o_full_heads.append(o_full)
        den_heads.append(den)
        for a, sel in sels.items():
            o = torch.softmax(logits[sel], dim=-1) @ vv[sel]
            errs[a] += float((o - o_full).norm() / den)
    res = {a: e / hq for a, e in errs.items()}
    if blk_cache is not None and "faithful" in sels:
        o_comp = sparse_attention_with_block_compensation(q, kf, vf, p, sels["faithful"], scale, blk_cache=blk_cache)
        err_c = sum(float((o_comp[h] - o_full_heads[h]).norm() / den_heads[h]) for h in range(hq)) / hq
        res["faithful_comp"] = err_c
    return res


@torch.no_grad()
def evaluate_selectors(
    indexer: FaithfulDLSAIndexer,
    q: torch.Tensor,
    k: torch.Tensor,
    q_positions: Sequence[int],
    budget: int = 2048,
    seed: int = 0,
    scale: Optional[float] = None,
    v: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Attention-mass recall (and, if `v` is given, attention-output relative error `err_<arm>` plus `err_faithful_comp`) for
    sampled decode-like queries.

    **Equal total budget.** The faithful selection reads sink + whole buckets + the unclustered tail, which can exceed `budget`
    long-range keys; every other arm is given exactly as many *non-sink/window* keys as the faithful arm actually read for that
    query (`mean_long_keys`), so recall differences are not a key-count artefact. Arms: `faithful`, `flat_oracle_index`,
    `contiguous`, `random`, `oracle`.
    """
    g = torch.Generator().manual_seed(seed)
    dev = q.device
    d = q.shape[-1]
    sc = scale if scale is not None else 1.0 / math.sqrt(d)
    q_in_all, k_in = indexer.inputs(q, k)
    k_idx_all = indexer.project_k(k_in)
    index = PagedLatentIndex(k_idx_all, indexer.n_sink, indexer.n_pool, indexer.n_cluster, indexer.avg_bucket_size,
                             indexer.max_bucket_size, indexer.window, indexer.rerank)  # built once; `upto` gives per-query causality
    arms = ("faithful", "flat_oracle_index", "contiguous", "random", "oracle")
    res: Dict[str, List[float]] = {a: [] for a in arms}
    errs: Dict[str, List[float]] = {a: [] for a in list(arms) + ["faithful_comp"]}
    n_long_list: List[int] = []
    kf = k.float() if v is not None else None
    vf = v.float() if v is not None else None
    blk_cache = BlockSummaryCache(kf, vf, indexer.n_sink, indexer.n_cluster, indexer.window) if v is not None else None
    da_tau = indexer.dist_alpha_over_tau()
    for p in q_positions:
        tp = teacher_attention(q, k, [p], scale)[0]  # [T]
        n = p + 1
        always_mask = torch.zeros(k.shape[1], dtype=torch.bool, device=dev)
        always_mask[: min(indexer.n_sink, n)] = True
        always_mask[max(0, n - indexer.window): n] = True
        always = torch.nonzero(always_mask).flatten()
        mid_lo, mid_hi = indexer.n_sink, max(indexer.n_sink, n - indexer.window)
        if mid_hi - mid_lo <= budget:
            continue
        qv = indexer.project_q(q_in_all[p: p + 1])
        sel = index.select(qv, budget, upto=n, dist_alpha_over_tau=da_tau)
        n_long = int((~always_mask[sel]).sum())
        n_long_list.append(n_long)
        scores = (k_idx_all[mid_lo:mid_hi] @ qv[0])
        if da_tau > 0.0:
            mid_pos = torch.arange(mid_lo, mid_hi, device=dev, dtype=torch.float32)
            scores = scores - float(da_tau) * torch.log1p((float(p) - mid_pos).clamp(min=0.0) / max(1.0, float(indexer.window)))
        top = scores.topk(min(n_long, scores.numel())).indices + mid_lo
        span = 8
        n_spans = (mid_hi - mid_lo) // span
        span_scores = (F.normalize(k_idx_all[mid_lo: mid_lo + n_spans * span].view(n_spans, span, -1).sum(1), dim=-1) @ qv[0])
        top_sp = span_scores.topk(min(max(1, math.ceil(n_long / span)), n_spans)).indices
        cont = (mid_lo + top_sp.unsqueeze(1) * span + torch.arange(span, device=dev).unsqueeze(0)).reshape(-1)[:n_long]
        rnd = (torch.randperm(mid_hi - mid_lo, generator=g)[:n_long] + mid_lo).to(dev)
        orc = (tp[mid_lo:mid_hi].topk(min(n_long, mid_hi - mid_lo)).indices + mid_lo)
        sels = {"faithful": sel,
                "flat_oracle_index": torch.cat([always, top]).unique(),
                "contiguous": torch.cat([always, cont]).unique(),
                "random": torch.cat([always, rnd]).unique(),
                "oracle": torch.cat([always, orc]).unique()}
        for a, sl in sels.items():
            res[a].append(recall_of_selection(tp, sl))
        if v is not None:
            for a, e in _out_rel_err(q, kf, vf, p, sels, sc, blk_cache=blk_cache).items():
                errs[a].append(e)
    out: Dict[str, Any] = {a: (sum(x) / len(x) if x else None) for a, x in res.items()}
    if v is not None:
        for a, x in errs.items():
            out["err_" + a] = sum(x) / len(x) if x else None
    out["n_queries"] = len(n_long_list)
    out["mean_long_keys"] = (sum(n_long_list) / len(n_long_list)) if n_long_list else None
    out["budget"] = budget
    return out


# ======================================================================================
# Loop72 Visit Schedule, Explicit State Key & 18-Visit D-LSA Index Bank (§5.1–§5.2, P2)
# ======================================================================================
GLOBAL_PHYSICAL_LAYERS: Tuple[int, ...] = (3, 7, 11, 15, 19, 23, 27, 31, 35)


def build_loop72_visit_schedule(
    start: int = 4,
    end: int = 28,
    passes: int = 2,
    enable_p3: bool = True,
    p3_start: int = 12,
    p3_end: int = 24,
    n_layers: int = 36,
    global_layers: Sequence[int] = GLOBAL_PHYSICAL_LAYERS,
) -> List[Dict[str, Any]]:
    """Build the 72-visit execution schedule table (`virtual_visit`, `physical_layer`, `pass_idx`, `is_global`,
    `layer_visit_ordinal`, `global_visit_id`)."""
    global_set = {int(x) for x in global_layers}
    segments: List[Tuple[int, int, int, str]] = [(0, start, 0, "prelude")]
    for p in range(passes):
        segments.append((start, end, p, f"core_p{p + 1}"))
    if enable_p3 and passes >= 2:
        segments.append((p3_start, p3_end, 2, "inner_p3"))
    segments.append((end, n_layers, 0, "coda"))

    schedule: List[Dict[str, Any]] = []
    visit_counts: Dict[int, int] = {}
    v_idx = 0
    g_id = 0
    for s_lo, s_hi, pass_idx, seg_name in segments:
        for l_idx in range(s_lo, s_hi):
            ord_i = visit_counts.get(l_idx, 0)
            visit_counts[l_idx] = ord_i + 1
            is_g = l_idx in global_set
            cur_g_id = g_id if is_g else None
            if is_g:
                g_id += 1
            schedule.append(
                {
                    "virtual_visit": v_idx,
                    "physical_layer": l_idx,
                    "pass_idx": pass_idx,
                    "segment": seg_name,
                    "is_global": is_g,
                    "layer_visit_ordinal": ord_i,
                    "global_visit_id": cur_g_id,
                }
            )
            v_idx += 1
    return schedule


LOOP72_VISIT_SCHEDULE: List[Dict[str, Any]] = build_loop72_visit_schedule()
LOOP72_GLOBAL_VISIT_SCHEDULE: List[Dict[str, Any]] = [
    dict(row) for row in LOOP72_VISIT_SCHEDULE if row["is_global"]
]


class VisitStateKey:
    """Immutable 7-tuple state key for per-visit cache and index ownership (§5.2):
    `(model_revision, behavior_fingerprint, projection_revision, session_id, memory_epoch, physical_layer, virtual_visit)`."""

    __slots__ = (
        "model_revision",
        "behavior_fingerprint",
        "projection_revision",
        "session_id",
        "memory_epoch",
        "physical_layer",
        "virtual_visit",
    )

    def __init__(
        self,
        model_revision: str = "XHToken/Spark-X2.6-4B",
        behavior_fingerprint: str = "native_gelu_loop72",
        projection_revision: str = "v1",
        session_id: str = "default",
        memory_epoch: int = 0,
        physical_layer: int = 3,
        virtual_visit: int = 3,
    ) -> None:
        self.model_revision = str(model_revision)
        self.behavior_fingerprint = str(behavior_fingerprint)
        self.projection_revision = str(projection_revision)
        self.session_id = str(session_id)
        self.memory_epoch = int(memory_epoch)
        self.physical_layer = int(physical_layer)
        self.virtual_visit = int(virtual_visit)

    def as_tuple(self) -> Tuple[str, str, str, str, int, int, int]:
        return (
            self.model_revision,
            self.behavior_fingerprint,
            self.projection_revision,
            self.session_id,
            self.memory_epoch,
            self.physical_layer,
            self.virtual_visit,
        )

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, VisitStateKey):
            return self.as_tuple() == other.as_tuple()
        if isinstance(other, tuple) and len(other) == 7:
            return self.as_tuple() == other
        return False

    def __repr__(self) -> str:
        return f"VisitStateKey{self.as_tuple()}"


class VisitAwareDLSAIndexBank(nn.Module):
    """Per-global-visit D-LSA selector bank for Spark X2.6 4B Loop72 (`18` global visits, `1,769,472` map parameters).

    When `maps_per_global_visit=True` (default per §5.1):
      - Allocates 18 independent `FaithfulDLSAIndexer` modules (`18 * 2 * 768 * 64 = 1,769,472` projection weights),
        indexed by `global_visit_id in [0..17]` (and mapped from `virtual_visit` or `(physical_layer, layer_visit_ordinal)`).
    When `maps_per_global_visit=False` (ablation):
      - Allocates 9 per-physical-layer `FaithfulDLSAIndexer` modules while still keeping mutable index state strictly
        isolated per `virtual_visit`.
    """

    def __init__(
        self,
        maps_per_global_visit: bool = True,
        n_kv: int = 4,
        head_dim: int = 256,
        rope_dim: int = 64,
        d_idx: int = 64,
        rope_first: bool = True,
        n_sink: int = 4,
        n_pool: int = 4,
        n_cluster: int = 512,
        avg_bucket_size: int = 8,
        max_bucket_size: int = 16,
        window: int = 512,
        k_long_range: int = 4096,
        rerank: int = 1,
        search_mode: str = "flat",
        branching_factor: int = 16,
        beam_width: int = 32,
        persistent_token_latents: bool = False,
        per_visit_budget_overrides: Optional[Dict[Any, int]] = None,
        visit_schedule: Optional[Sequence[Dict[str, Any]]] = None,
        projection_revision: str = "loop72_dlsa_v1",
    ) -> None:
        super().__init__()
        self.maps_per_global_visit = bool(maps_per_global_visit)
        self.n_kv = int(n_kv)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.d_idx = int(d_idx)
        self.k_long_range = int(k_long_range)
        self.search_mode = str(search_mode).lower().strip()
        self.persistent_token_latents = bool(persistent_token_latents)
        self.projection_revision = str(projection_revision)
        self.schedule = list(visit_schedule) if visit_schedule is not None else list(LOOP72_VISIT_SCHEDULE)
        self.global_schedule = [row for row in self.schedule if row["is_global"]]

        self._vvisit_to_gvisit: Dict[int, int] = {}
        self._layer_ord_to_gvisit: Dict[Tuple[int, int], int] = {}
        self._gvisit_meta: Dict[int, Dict[str, Any]] = {}
        for row in self.global_schedule:
            gid = int(row["global_visit_id"])
            vv = int(row["virtual_visit"])
            pl = int(row["physical_layer"])
            lo = int(row["layer_visit_ordinal"])
            self._vvisit_to_gvisit[vv] = gid
            self._layer_ord_to_gvisit[(pl, lo)] = gid
            self._gvisit_meta[gid] = dict(row)

        def _make_ix() -> FaithfulDLSAIndexer:
            return FaithfulDLSAIndexer(
                n_kv=n_kv,
                head_dim=head_dim,
                rope_dim=rope_dim,
                d_idx=d_idx,
                rope_first=rope_first,
                n_sink=n_sink,
                n_pool=n_pool,
                n_cluster=n_cluster,
                avg_bucket_size=avg_bucket_size,
                max_bucket_size=max_bucket_size,
                window=window,
                k_long_range=k_long_range,
                rerank=rerank,
                search_mode=search_mode,
                branching_factor=branching_factor,
                beam_width=beam_width,
                persistent_token_latents=persistent_token_latents,
            )

        if self.maps_per_global_visit:
            self.indexers = nn.ModuleDict({str(gid): _make_ix() for gid in range(len(self.global_schedule))})
        else:
            unique_layers = sorted({int(r["physical_layer"]) for r in self.global_schedule})
            self.indexers = nn.ModuleDict({str(l_idx): _make_ix() for l_idx in unique_layers})

        self.per_visit_budget_overrides: Dict[int, int] = {}
        if per_visit_budget_overrides:
            for k, v in per_visit_budget_overrides.items():
                self.set_visit_budget_override(k, int(v))

    def resolve_global_visit_id(
        self,
        physical_layer: int,
        virtual_visit: Optional[int] = None,
        layer_visit_ordinal: int = 0,
        global_visit_id: Optional[int] = None,
    ) -> int:
        if global_visit_id is not None:
            return int(global_visit_id)
        if virtual_visit is not None and int(virtual_visit) in self._vvisit_to_gvisit:
            return self._vvisit_to_gvisit[int(virtual_visit)]
        key = (int(physical_layer), int(layer_visit_ordinal))
        if key in self._layer_ord_to_gvisit:
            return self._layer_ord_to_gvisit[key]
        # Fallback to first visit of `physical_layer`
        return self._layer_ord_to_gvisit.get((int(physical_layer), 0), 0)

    def get_indexer(
        self,
        physical_layer: int,
        virtual_visit: Optional[int] = None,
        layer_visit_ordinal: int = 0,
        global_visit_id: Optional[int] = None,
    ) -> FaithfulDLSAIndexer:
        if self.maps_per_global_visit:
            gid = self.resolve_global_visit_id(
                physical_layer,
                virtual_visit=virtual_visit,
                layer_visit_ordinal=layer_visit_ordinal,
                global_visit_id=global_visit_id,
            )
            return self.indexers[str(gid)]
        return self.indexers[str(int(physical_layer))]

    def set_visit_budget_override(self, visit_or_layer: Any, budget: int) -> None:
        b = max(64, int(budget))
        if isinstance(visit_or_layer, str) and visit_or_layer.startswith("g"):
            gid = int(visit_or_layer[1:])
            self.per_visit_budget_overrides[gid] = b
            if self.maps_per_global_visit and str(gid) in self.indexers:
                self.indexers[str(gid)].k_long_range = b
            return
        idx_val = int(visit_or_layer)
        if idx_val in self._vvisit_to_gvisit:
            gid = self._vvisit_to_gvisit[idx_val]
            self.per_visit_budget_overrides[gid] = b
            if self.maps_per_global_visit and str(gid) in self.indexers:
                self.indexers[str(gid)].k_long_range = b
        elif idx_val in self._gvisit_meta:
            self.per_visit_budget_overrides[idx_val] = b
            if self.maps_per_global_visit and str(idx_val) in self.indexers:
                self.indexers[str(idx_val)].k_long_range = b
        else:
            # Physical layer override -> apply to all visits of that physical layer
            for gid, meta in self._gvisit_meta.items():
                if int(meta["physical_layer"]) == idx_val:
                    self.per_visit_budget_overrides[gid] = b
                    if self.maps_per_global_visit and str(gid) in self.indexers:
                        self.indexers[str(gid)].k_long_range = b
            if not self.maps_per_global_visit and str(idx_val) in self.indexers:
                self.indexers[str(idx_val)].k_long_range = b

    def get_visit_budget(
        self,
        physical_layer: int,
        virtual_visit: Optional[int] = None,
        layer_visit_ordinal: int = 0,
        global_visit_id: Optional[int] = None,
    ) -> int:
        gid = self.resolve_global_visit_id(
            physical_layer,
            virtual_visit=virtual_visit,
            layer_visit_ordinal=layer_visit_ordinal,
            global_visit_id=global_visit_id,
        )
        if gid in self.per_visit_budget_overrides:
            return int(self.per_visit_budget_overrides[gid])
        ix = self.get_indexer(
            physical_layer,
            virtual_visit=virtual_visit,
            layer_visit_ordinal=layer_visit_ordinal,
            global_visit_id=gid,
        )
        return int(ix.k_long_range)

    def reset_all_decode_caches(self) -> None:
        for ix in self.indexers.values():
            ix.reset_decode_cache()

    def parameter_manifest(self) -> Dict[str, Any]:
        map_params = sum(
            ix.w_q.weight.numel() + ix.w_k.weight.numel()
            for ix in self.indexers.values()
        )
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "maps_per_global_visit": self.maps_per_global_visit,
            "num_indexer_modules": len(self.indexers),
            "global_visits": len(self.global_schedule),
            "map_parameters": int(map_params),
            "total_parameters": int(total_params),
            "projection_revision": self.projection_revision,
            "search_mode": self.search_mode,
            "default_k_long_range": self.k_long_range,
            "per_visit_budget_overrides": dict(self.per_visit_budget_overrides),
        }

    def export_manifest_state(self) -> Dict[str, Any]:
        return {
            "manifest": self.parameter_manifest(),
            "global_schedule": self.global_schedule,
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }

    def load_from_legacy_or_visit_checkpoint(self, ckpt_data: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """Load either an 18-visit `VisitAwareDLSAIndexBank` checkpoint or broadcast a 9-layer `LSA-F2` checkpoint
        across each physical layer's visits."""
        if "state_dict" in ckpt_data and isinstance(ckpt_data["state_dict"], dict):
            sd = ckpt_data["state_dict"]
            if any(k.startswith("indexers.") for k in sd.keys()):
                self.load_state_dict(sd, strict=strict)
                return {"mode": "visit_bank_state_dict", "loaded_modules": len(self.indexers)}
        raw_layers = ckpt_data.get("indexers", ckpt_data)
        loaded = 0
        if self.maps_per_global_visit:
            for gid, meta in self._gvisit_meta.items():
                pl = int(meta["physical_layer"])
                src_sd = (
                    raw_layers.get(f"g{gid}")
                    or raw_layers.get(str(gid))
                    or raw_layers.get(pl)
                    or raw_layers.get(str(pl))
                )
                if isinstance(src_sd, dict):
                    self.indexers[str(gid)].load_state_dict(src_sd, strict=strict)
                    loaded += 1
        else:
            for l_key, ix in self.indexers.items():
                src_sd = raw_layers.get(int(l_key)) or raw_layers.get(str(l_key))
                if isinstance(src_sd, dict):
                    ix.load_state_dict(src_sd, strict=strict)
                    loaded += 1
        return {"mode": "broadcast_layer_indexers", "loaded_modules": loaded}

