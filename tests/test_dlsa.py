"""Tests for sparkx_dlsa (faithful D-LSA). Require torch; skipped when unavailable. Synthetic tensors only (plumbing and
mechanism tests — they do NOT validate recall on real Spark attention; that is claim gate LSA-F2)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn.functional as F
    from sparkx_dlsa import (FaithfulDLSAIndexer, PagedLatentIndex, distill_indexer, evaluate_selectors,
                             free_logit_fidelity, free_slice_inputs, mid_region_mask, teacher_attention, ward_partition)
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

H_Q, H_KV, D, ROPE = 4, 2, 32, 8


def make_scattered(t=1024, n_topics=16, seed=0, strength=3.0, noise=0.3):
    """Keys belong to random (non-contiguous) topics; queries attend to one topic. Free slice carries topic structure."""
    g = torch.Generator().manual_seed(seed)
    free = D - ROPE
    topics = F.normalize(torch.randn(n_topics, H_KV, free, generator=g), dim=-1)
    tok_topic = torch.randint(0, n_topics, (t,), generator=g)
    k = torch.randn(H_KV, t, D, generator=g) * 0.1
    k[..., ROPE:] = strength * topics[tok_topic].permute(1, 0, 2) + noise * torch.randn(H_KV, t, free, generator=g)
    q = torch.randn(H_Q, t, D, generator=g) * 0.1
    q_topic = torch.randint(0, n_topics, (t,), generator=g)
    for h in range(H_Q):
        q[h, :, ROPE:] = strength * topics[q_topic][:, h // (H_Q // H_KV)] + noise * torch.randn(t, free, generator=g)
    return q, k, tok_topic, q_topic


def small_indexer(**kw):
    cfg = dict(n_kv=H_KV, head_dim=D, rope_dim=ROPE, d_idx=16, n_sink=4, n_pool=4, n_cluster=64, avg_bucket_size=8,
               max_bucket_size=16, window=32, k_long_range=128)
    cfg.update(kw)
    return FaithfulDLSAIndexer(**cfg)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestDLSA(unittest.TestCase):
    def test_free_slice_inner_product_equals_summed_free_logits(self):
        q, k, _, _ = make_scattered(t=64)
        q_in, k_in = free_slice_inputs(q, k, ROPE)
        g = H_Q // H_KV
        for p in (5, 40):
            for t in (3, 17):
                got = float(q_in[p] @ k_in[t])
                want = sum(float(q[h, p, ROPE:] @ k[h // g, t, ROPE:]) for h in range(H_Q))
                self.assertAlmostEqual(got, want, places=3)

    def test_free_logit_fidelity_high_when_signal_in_free_slice(self):
        q, k, _, _ = make_scattered(t=256)
        r = free_logit_fidelity(q, k, [100, 200], rope_dim=ROPE)["mean_pearson_r"]
        self.assertGreater(r, 0.8)

    def test_ward_partition_planted_clusters_pure_and_target_respected(self):
        g = torch.Generator().manual_seed(1)
        centers = F.normalize(torch.randn(8, 16, generator=g), dim=-1)
        ids = torch.arange(128) % 8  # interleaved (non-contiguous) membership
        x = F.normalize(centers[ids] + 0.02 * torch.randn(128, 16, generator=g), dim=-1).unsqueeze(0)
        labels = ward_partition(x, target_buckets=40, max_pools_per_bucket=4)[0]  # slack: greedy size-capped merging
        n_clusters = int(labels.unique().numel())
        self.assertGreaterEqual(n_clusters, 40)
        self.assertLessEqual(n_clusters, 44)
        for c in labels.unique():
            members = ids[labels == c]
            self.assertEqual(int(members.unique().numel()), 1, "bucket mixes planted clusters")
            self.assertLessEqual(int((labels == c).sum()), 4)

    def test_ward_partition_never_below_target_and_batched(self):
        x = F.normalize(torch.randn(3, 64, 8, generator=torch.Generator().manual_seed(2)), dim=-1)
        labels = ward_partition(x, target_buckets=16, max_pools_per_bucket=4)
        for b in range(3):
            n = int(labels[b].unique().numel())
            self.assertGreaterEqual(n, 16)
            self.assertLessEqual(n, 32)  # makes progress toward target
            self.assertLessEqual(int(torch.bincount(labels[b]).max()), 4)

    def test_selection_budget_sorted_unique_causal(self):
        q, k, _, _ = make_scattered(t=1024)
        ix = small_indexer()
        ix.fit_pca(ix.inputs(q, k)[1])
        sel = ix.select_key_indices(q[None, :, 900:901], k[None, :, :901], k_budget=128)
        self.assertTrue(bool((sel[1:] > sel[:-1]).all()))  # sorted unique
        self.assertLess(int(sel.max()), 901)
        self.assertEqual(int(sel.min()), 0)
        self.assertTrue(bool((sel[:4] == torch.arange(4)).all()))  # sink
        self.assertTrue(bool(torch.isin(torch.arange(901 - 32, 901), sel).all()))  # window / tail kept
        idx = ix.build_index(k[None, :, :901][0])
        n_mid = int(((sel >= 4) & (sel < idx.clustered_end)).sum())  # remainder keys up to clustered_end..T are always read
        self.assertLessEqual(n_mid, 128 + 16)

    def test_short_context_returns_all_keys(self):
        q, k, _, _ = make_scattered(t=64)
        ix = small_indexer()
        sel = ix.select_key_indices(q[None, :, -1:], k[None], k_budget=128)
        self.assertEqual(sel.numel(), 64)

    def test_faithful_beats_contiguous_and_random_on_scattered_structure(self):
        q, k, _, _ = make_scattered(t=1024, seed=3)
        ix = small_indexer()
        ix.fit_pca(ix.inputs(q, k)[1])
        res = evaluate_selectors(ix, q, k, [700, 800, 900, 1000], budget=128)
        self.assertEqual(res["n_queries"], 4)
        self.assertGreater(res["faithful"], res["contiguous"] + 0.05)
        self.assertGreater(res["faithful"], res["random"] + 0.05)
        self.assertGreaterEqual(res["oracle"] + 1e-6, res["faithful"])
        self.assertGreaterEqual(res["oracle"] + 1e-6, res["flat_oracle_index"])

    def test_distillation_reduces_kl(self):
        q, k, _, _ = make_scattered(t=512, seed=4)
        ix = small_indexer(window=16)
        out = distill_indexer(ix, [{"q": q, "k": k}], steps=60, lr=5e-3, queries_per_sample=16, log_every=0)
        self.assertEqual(out["n_samples"], 1)
        self.assertLess(out["kl_after"], out["kl_before"])

    def test_teacher_attention_rows_sum_to_one_and_causal(self):
        q, k, _, _ = make_scattered(t=64)
        p = teacher_attention(q, k, [10, 63])
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(2), atol=1e-4))
        self.assertEqual(float(p[0, 11:].sum()), 0.0)

    def test_mid_region_mask(self):
        m = mid_region_mask([100], 128, n_sink=4, window=16)
        self.assertFalse(bool(m[0, :4].any()))
        self.assertFalse(bool(m[0, 85:].any()))
        self.assertTrue(bool(m[0, 4:85].all()))

    def test_paged_index_short_has_no_blocks(self):
        idx = PagedLatentIndex(torch.randn(90, 8), n_sink=4, n_cluster=64, window=32)
        self.assertEqual(idx.n_blocks, 0)
        self.assertEqual(idx.select(torch.randn(8), 16).numel(), 90)


if __name__ == "__main__":
    unittest.main()
