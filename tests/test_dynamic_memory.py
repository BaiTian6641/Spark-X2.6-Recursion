import unittest

import torch
import torch.nn.functional as F

from sparkx_dlsa import PagedLatentIndex
from sparkx_dynamic_memory import (
    DynamicLatentMemoryGraph,
    HotColdPagedKVStore,
    NullSpaceVerticalAdapter,
    SleepConsolidationEngine,
    evaluate_dlmg_recall_suite,
    evaluate_sleep_consolidation_suite,
)


class TestDynamicMemory(unittest.TestCase):

    def test_hot_cold_paged_kv_store_exact_and_lru(self):
        g = torch.Generator().manual_seed(11)
        h_kv, t, d = 4, 260, 32
        k = torch.randn(h_kv, t, d, generator=g)
        v = torch.randn(h_kv, t, d, generator=g)
        k_idx = F.normalize(k[0], dim=-1)
        idx = PagedLatentIndex(k_idx, n_sink=4, n_pool=4, n_cluster=64, avg_bucket_size=16, max_bucket_size=16, window=32)

        store = HotColdPagedKVStore(hot_bucket_capacity=2, quantize_fp8=False)
        n_ing = store.ingest(idx, k, v)
        self.assertEqual(n_ing, idx.n_blocks)

        all_buckets = list(store.cold_buckets.keys())
        self.assertGreaterEqual(len(all_buckets), 4)
        pick = all_buckets[:3]
        k_sel, v_sel, pos_sel = store.assemble_selected_kv(pick)
        self.assertTrue(torch.equal(pos_sel, pos_sel.sort().values))
        self.assertTrue(torch.allclose(k_sel, k[:, pos_sel], atol=1e-6))
        self.assertTrue(torch.allclose(v_sel, v[:, pos_sel], atol=1e-6))
        self.assertGreaterEqual(store.lru_evictions, 1)

        # Second access to the last 2 buckets in `pick` should hit LRU cache
        hits_before = store.cache_hits
        store.prefetch_buckets(pick[1:3])
        self.assertEqual(store.cache_hits - hits_before, 2)

    def test_hot_cold_paged_kv_store_fp8_and_incremental_ingest(self):
        g = torch.Generator().manual_seed(22)
        h_kv, t, d = 4, 320, 32
        k = torch.randn(h_kv, t, d, generator=g)
        v = torch.randn(h_kv, t, d, generator=g)
        k_idx = F.normalize(k[0], dim=-1)

        idx = PagedLatentIndex(k_idx[:180], n_sink=4, n_pool=4, n_cluster=64, avg_bucket_size=16, max_bucket_size=16, window=32)
        store_fp8 = HotColdPagedKVStore(hot_bucket_capacity=4, quantize_fp8=True)
        b1 = store_fp8.ingest(idx, k[:, :180], v[:, :180])
        self.assertEqual(b1, 2)

        idx.extend_to(k_idx)
        b2 = store_fp8.ingest(idx, k, v)
        self.assertEqual(b2, 2)

        pick = list(store_fp8.cold_buckets.keys())[:2]
        k_sel, v_sel, pos_sel = store_fp8.assemble_selected_kv(pick)
        cos_k = F.cosine_similarity(k_sel.reshape(-1), k[:, pos_sel].reshape(-1), dim=0)
        self.assertGreater(float(cos_k.item()), 0.995)
        stats = store_fp8.memory_stats()
        self.assertGreater(stats["gpu_memory_saving_ratio"], 0.30)

    def test_dlmg_recall_suite_single_multihop_and_temporal(self):
        res = evaluate_dlmg_recall_suite(seed=42)
        # 1. Single-hop: no regression vs flat D-LSA while using <= keys
        self.assertGreaterEqual(res["single_hop"]["dlmg_recall"], res["single_hop"]["flat_recall"])
        self.assertLessEqual(res["single_hop"]["dlmg_long_keys"], res["single_hop"]["flat_long_keys"])

        # 2. Multi-hop associative bridge: DLMG PPR recovers orthogonal target bucket via Hebbian co-access edge at equal key count
        self.assertGreater(res["multi_hop_bridge"]["dlmg_target_recall"], res["multi_hop_bridge"]["flat_target_recall"] + 0.5)
        self.assertEqual(res["multi_hop_bridge"]["dlmg_long_keys"], res["multi_hop_bridge"]["flat_long_keys"])

        # 3. Bi-temporal contradiction invalidation: flat D-LSA picks stale V1 (0.0 V2 share), DLMG routes to V2 (1.0 V2 share)
        self.assertGreaterEqual(res["temporal_update"]["superseded_pairs_detected"], 1)
        self.assertEqual(res["temporal_update"]["flat_v2_share"], 0.0)
        self.assertEqual(res["temporal_update"]["dlmg_v2_share"], 1.0)
        self.assertEqual(res["temporal_update"]["dlmg_v2_recall"], 1.0)
        self.assertEqual(res["temporal_update"]["dlmg_v1_stale_recall"], 0.0)

    def test_sleep_consolidation_nullspace_wise_scope_and_unlearning(self):
        res = evaluate_sleep_consolidation_suite(seed=101)
        self.assertLess(res["round1_rel_error"], 0.01)
        self.assertLess(res["round2_rel_error"], 0.01)
        # AlphaEdit null-space projection guarantees zero drift on Round-1 consolidated memories during Round-2
        self.assertLess(res["round1_nullspace_drift_after_round2"], 1e-5)
        # WISE activation scope gate guarantees exact 0.0 leakage on out-of-scope general queries
        self.assertEqual(res["general_query_leakage_max"], 0.0)
        # Surgical unlearning of Round-2 leaves Round-1 intact
        self.assertLess(res["r1_rel_error_after_r2_unlearn"], 0.01)
        self.assertGreater(res["evicted_tokens_total"], 0)

    def test_dlmg_incremental_sync_and_super_topics(self):
        g = torch.Generator().manual_seed(55)
        k_idx = F.normalize(torch.randn(360, 32, generator=g), dim=-1)
        idx = PagedLatentIndex(k_idx[:200], n_sink=4, n_pool=4, n_cluster=64, avg_bucket_size=8, max_bucket_size=16, window=32)
        dlmg = DynamicLatentMemoryGraph(idx, n_super_topics=4)
        n1 = len(dlmg.nodes)
        self.assertGreater(n1, 0)
        idx.extend_to(k_idx)
        added = dlmg.sync_with_index()
        self.assertGreater(added, 0)
        self.assertEqual(len(dlmg.nodes), n1 + added)
        self.assertIsNotNone(dlmg.super_centroids)


if __name__ == "__main__":
    unittest.main()
