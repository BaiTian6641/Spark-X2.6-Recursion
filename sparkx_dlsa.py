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
    """Hierarchical (pool -> Ward bucket) index over the *middle* region `[n_sink, T - tail)` of a key sequence.

    Only complete blocks of `n_cluster` keys are clustered; the remainder (< n_cluster keys) is returned as part of the
    always-read tail. Selection returns **exact key positions** (sink + selected buckets + tail), never averaged values.
    """

    def __init__(self, k_idx: torch.Tensor, n_sink: int = 4, n_pool: int = 4, n_cluster: int = 512,
                 avg_bucket_size: int = 8, max_bucket_size: int = 16, window: int = 512, rerank: int = 1):
        assert n_cluster % n_pool == 0
        self.rerank = max(1, int(rerank))  # two-stage: buckets up to rerank*budget tokens, then token-level top-`budget` on k_idx
        self.t = int(k_idx.shape[0])
        self.n_sink, self.n_pool, self.n_cluster = int(n_sink), int(n_pool), int(n_cluster)
        self.window = int(window)
        d = k_idx.shape[-1]
        mid_end = max(self.n_sink, self.t - self.window)
        n_blocks = max(0, (mid_end - self.n_sink) // self.n_cluster)
        self.n_blocks = n_blocks
        self.mid_start = self.n_sink
        self.clustered_end = self.n_sink + n_blocks * self.n_cluster  # keys in [clustered_end, T) are always read
        self.pools_per_block = self.n_cluster // self.n_pool
        if n_blocks == 0:
            self.centroids = torch.zeros(0, 0, d)
            self.labels = torch.zeros(0, 0, dtype=torch.long)
            self.bucket_tokens = torch.zeros(0, 0)
            return
        kk = F.normalize(k_idx.float(), dim=-1, eps=1e-6)
        self.k_idx_n = kk if self.rerank > 1 else None
        blk = kk[self.mid_start: self.clustered_end].view(n_blocks, self.pools_per_block, self.n_pool, d)
        pools = F.normalize(blk.mean(dim=2), dim=-1, eps=1e-6)  # [B, P, d]
        target = max(1, self.n_cluster // int(avg_bucket_size))
        self.labels = ward_partition(pools, target, max_pools_per_bucket=max(1, int(max_bucket_size) // self.n_pool))
        p = self.pools_per_block
        onehot = F.one_hot(self.labels, num_classes=p).float()  # [B, P, C]
        sums = torch.einsum("bpc,bpd->bcd", onehot, pools)
        self.centroids = F.normalize(sums, dim=-1, eps=1e-6)  # [B, C, d]
        self.bucket_tokens = onehot.sum(dim=1) * self.n_pool  # [B, C] token count per bucket (0 = empty slot)

    def bucket_stats(self) -> Dict[str, float]:
        if self.n_blocks == 0:
            return {"blocks": 0}
        nz = self.bucket_tokens[self.bucket_tokens > 0]
        return {"blocks": self.n_blocks, "buckets": int(nz.numel()), "mean_tokens": float(nz.mean()),
                "max_tokens": float(nz.max())}

    @torch.no_grad()
    def select(self, q_idx: torch.Tensor, budget: int, upto: Optional[int] = None) -> torch.Tensor:
        """`q_idx`: [d] or [Tq, d] unit-norm (mean-pooled if 2-D). Returns sorted exact key positions (sink+selected+tail).

        `upto`: causal horizon for a query at position `upto - 1` (keys `>= upto` do not exist for it). Because Ward blocks are
        independent, restricting to the blocks that end before `upto - window` is identical to re-indexing `k[:upto]`; this lets
        one index serve every decode step (and grow incrementally as the cache grows).
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
        scores = torch.einsum("bcd,d->bc", cents, q).masked_fill(toks == 0, float("-inf"))
        flat_s, flat_i = scores.reshape(-1).sort(descending=True)
        flat_t = toks.reshape(-1)[flat_i]
        csum = torch.cumsum(flat_t, dim=0)
        take = (csum - flat_t) < int(budget) * self.rerank  # include a bucket while the (stage-1) budget is not yet exhausted
        take &= torch.isfinite(flat_s)
        sel_flat = flat_i[take]
        sel_mask = torch.zeros(limit * self.pools_per_block, dtype=torch.bool, device=dev)
        sel_mask[sel_flat] = True
        sel_mask = sel_mask.view(limit, self.pools_per_block)  # [B, C]
        pool_sel = sel_mask.gather(1, self.labels[:limit].to(dev))  # [B, P]
        pool_ids = torch.nonzero(pool_sel)  # [M, 2]
        base = self.mid_start + pool_ids[:, 0] * self.n_cluster + pool_ids[:, 1] * self.n_pool
        mid = (base.unsqueeze(1) + torch.arange(self.n_pool, device=dev).unsqueeze(0)).reshape(-1)
        if self.rerank > 1 and mid.numel() > int(budget) and self.k_idx_n is not None:
            tok_scores = self.k_idx_n.to(dev)[mid] @ q  # stage 2: token-level scores of the candidates only
            mid = mid[tok_scores.topk(int(budget)).indices]
        return torch.cat([sink, mid, tail]).unique()  # `unique` sorts ascending


# ======================================================================================
# Learned index space + per-key KL distillation
# ======================================================================================
class FaithfulDLSAIndexer(nn.Module):
    """Per-layer learned maps `W_q, W_k: (n_kv*F) -> d_idx` (+ learned temperature), L2-normalised.

    Initialised from PCA of the position-free key inputs (training-free D-LSA) via `fit_pca`, then refined by `distill`.
    """

    def __init__(self, n_kv: int = 4, head_dim: int = 256, rope_dim: int = 64, d_idx: int = 64, rope_first: bool = True,
                 n_sink: int = 4, n_pool: int = 4, n_cluster: int = 512, avg_bucket_size: int = 8, max_bucket_size: int = 16,
                 window: int = 512, k_long_range: int = 2048, rerank: int = 1):
        super().__init__()
        self.rerank = max(1, int(rerank))
        self.n_kv, self.head_dim, self.rope_dim, self.d_idx, self.rope_first = n_kv, head_dim, rope_dim, d_idx, rope_first
        self.n_sink, self.n_pool, self.n_cluster = n_sink, n_pool, n_cluster
        self.avg_bucket_size, self.max_bucket_size, self.window, self.k_long_range = avg_bucket_size, max_bucket_size, window, k_long_range
        din = n_kv * (head_dim - rope_dim)
        self.w_q = nn.Linear(din, d_idx, bias=False)
        self.w_k = nn.Linear(din, d_idx, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(12.0)))
        nn.init.orthogonal_(self.w_k.weight)
        self.w_q.weight.data.copy_(self.w_k.weight.data)

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
        return PagedLatentIndex(self.project_k(k_in), self.n_sink, self.n_pool, self.n_cluster, self.avg_bucket_size,
                                self.max_bucket_size, self.window, self.rerank)

    @torch.no_grad()
    def select_key_indices(self, query: torch.Tensor, key: torch.Tensor, k_budget: Optional[int] = None) -> torch.Tensor:
        """API-compatible with `LatentSparseAttentionIndexer.select_key_indices`: `query` [1, H_q, T_q, D], `key`
        [1, H_kv, T_k, D] -> exact key positions shared by all heads (decode / <=64 queries; mean of the query index vectors)."""
        assert query.shape[0] == 1 and key.shape[0] == 1
        budget = int(k_budget or self.k_long_range)
        t = key.shape[2]
        if t <= self.n_sink + self.window + budget:
            return torch.arange(t, device=key.device)
        idx = self.build_index(key[0])
        q_in, _ = self.inputs(query[0], key[0][:, :1])
        return idx.select(self.project_q(q_in), budget).to(key.device)

    # ---- distillation
    def distill_loss(self, q_in: torch.Tensor, k_in: torch.Tensor, teacher_p: torch.Tensor,
                     key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-key KL(teacher || student) averaged over queries. `teacher_p`: [Tq, T] (rows sum to 1 over the allowed keys),
        `key_mask`: [Tq, T] bool of keys the student must rank (e.g. middle region, causal)."""
        tau = self.log_tau.float().exp().clamp(1.0, 64.0)
        logits = tau * self.project_q(q_in) @ self.project_k(k_in).t()
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
        q_in_all, k_in = indexer.inputs(q, k)
        mask = mid_region_mask(pos, t, indexer.n_sink, indexer.window, q.device)
        p = teacher_attention(q, k, pos, scale) * mask
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        vals.append(float(indexer.distill_loss(q_in_all[pos], k_in, p, mask)))
    return sum(vals) / len(vals) if vals else None


def _kl_cache(indexer: FaithfulDLSAIndexer, samples: Sequence[Dict[str, torch.Tensor]], queries_per_sample: int, g: torch.Generator,
              scale: Optional[float]) -> List[Tuple[torch.Tensor, ...]]:
    cache = []
    for s in samples:
        q, k = s["q"], s["k"]
        t = q.shape[1]
        lo = indexer.n_sink + indexer.window + 64
        if t <= lo + 1:
            continue
        pos = torch.randint(lo, t, (min(queries_per_sample, t - lo),), generator=g).sort().values.tolist()
        q_in_all, k_in = indexer.inputs(q, k)
        mask = mid_region_mask(pos, t, indexer.n_sink, indexer.window, q.device)
        p = teacher_attention(q, k, pos, scale)
        p = p * mask  # renormalise on the middle region: this is what the indexer is responsible for ranking
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        cache.append((q_in_all[pos], k_in, p, mask))
    return cache


def distill_indexer(indexer: FaithfulDLSAIndexer, samples: Sequence[Dict[str, torch.Tensor]], steps: int = 200,
                    lr: float = 3e-3, queries_per_sample: int = 64, seed: int = 0, log_every: int = 50,
                    log=print, scale: Optional[float] = None, val_samples: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
                    eval_every: int = 25, weight_decay: float = 0.0) -> Dict[str, Any]:
    """Distil `W_q, W_k, tau` against the frozen teacher with per-key KL on the middle region.

    `samples`: dicts with `q` [H_q, T, D] and `k` [H_kv, T, D] (post-RoPE tensors captured from the frozen model on long
    documents). Queries are sampled among positions with a non-empty middle region. Returns KL before/after on all samples.

    `val_samples` (recommended): held-out samples used for **early stopping** — every `eval_every` steps the held-out KL is
    measured and the parameters with the lowest held-out KL (including the initial/PCA state) are restored at the end, so the
    result is never worse than the init on held-out KL. The first F2b run showed held-out KL rising under training (L3 full-head).
    """
    g = torch.Generator().manual_seed(seed)
    cache = _kl_cache(indexer, samples, queries_per_sample, g, scale)
    if not cache:
        return {"kl_before": None, "kl_after": None, "n_samples": 0}
    vcache = _kl_cache(indexer, val_samples, queries_per_sample, torch.Generator().manual_seed(seed + 1), scale) if val_samples else []

    @torch.no_grad()
    def mean_kl(c) -> float:
        return float(sum(indexer.distill_loss(a, b, cc, m) for a, b, cc, m in c) / len(c))

    kl0 = mean_kl(cache)
    best_v = mean_kl(vcache) if vcache else None
    v0 = best_v
    best_state = {k_: v.detach().clone() for k_, v in indexer.state_dict().items()} if vcache else None
    best_step = 0
    opt = torch.optim.AdamW(indexer.parameters(), lr=lr, weight_decay=weight_decay)
    for step in range(1, int(steps) + 1):
        a, b, c, m = cache[(step - 1) % len(cache)]
        opt.zero_grad(set_to_none=True)
        loss = indexer.distill_loss(a, b, c, m)
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
    return out


# ======================================================================================
# Recall metrics and baselines
# ======================================================================================
def recall_of_selection(teacher_p_row: torch.Tensor, selected: torch.Tensor) -> float:
    """Fraction of the teacher's total attention mass for one query that falls on `selected` key positions."""
    return float(teacher_p_row[selected].sum() / teacher_p_row.sum().clamp(min=1e-12))


def _out_rel_err(q: torch.Tensor, kf: torch.Tensor, vf: torch.Tensor, p: int, sels: Dict[str, torch.Tensor],
                 scale: float) -> Dict[str, float]:
    """Per-head attention-OUTPUT relative error of sparse (softmax renormalised over the selected keys) vs exact causal attention
    for the query at position `p`, mean over heads. `kf`/`vf`: float [H_kv, T, D]. The renormalisation is why output error is
    usually much smaller than the missed attention mass suggests."""
    hq = q.shape[0]
    g = hq // kf.shape[0]
    errs = {a: 0.0 for a in sels}
    for h in range(hq):
        kk, vv = kf[h // g, : p + 1], vf[h // g, : p + 1]
        logits = (kk @ q[h, p].float()) * scale
        o_full = torch.softmax(logits, dim=-1) @ vv
        den = o_full.norm().clamp(min=1e-8)
        for a, sel in sels.items():
            o = torch.softmax(logits[sel], dim=-1) @ vv[sel]
            errs[a] += float((o - o_full).norm() / den)
    return {a: e / hq for a, e in errs.items()}


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
    """Attention-mass recall (and, if `v` is given, attention-output relative error `err_<arm>`) for sampled decode-like queries.

    **Equal total budget.** The faithful selection reads sink + whole buckets + the unclustered tail, which can exceed `budget`
    long-range keys; every other arm is given exactly as many *non-sink/window* keys as the faithful arm actually read for that
    query (`mean_long_keys`), so recall differences are not a key-count artefact (the first F2/F2b runs compared the faithful arm at
    up to ~25% more keys). Arms: `faithful`, `flat_oracle_index` (same learned scores, token-level top-n: isolates clustering loss),
    `contiguous` (8-token spans scored in the learned space = the repo's previous selector), `random`, `oracle` (true top-n).
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
    errs: Dict[str, List[float]] = {a: [] for a in arms}
    n_long_list: List[int] = []
    kf = k.float() if v is not None else None
    vf = v.float() if v is not None else None
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
        sel = index.select(qv, budget, upto=n)
        n_long = int((~always_mask[sel]).sum())
        n_long_list.append(n_long)
        scores = (k_idx_all[mid_lo:mid_hi] @ qv[0])
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
            for a, e in _out_rel_err(q, kf, vf, p, sels, sc).items():
                errs[a].append(e)
    out: Dict[str, Any] = {a: (sum(x) / len(x) if x else None) for a, x in res.items()}
    if v is not None:
        for a, x in errs.items():
            out["err_" + a] = sum(x) / len(x) if x else None
    out["n_queries"] = len(n_long_list)
    out["mean_long_keys"] = (sum(n_long_list) / len(n_long_list)) if n_long_list else None
    out["budget"] = budget
    return out
