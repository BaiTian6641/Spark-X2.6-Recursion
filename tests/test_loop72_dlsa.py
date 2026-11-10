import os
import unittest

from sparkx_kv_codec import (
    KVCodecConfig,
    compute_loop72_capacity_table,
    decode_kv_page,
    encode_kv_page,
    measure_codec_error,
)
from sparkx_memory_utils import EpisodicSideMemory

try:
    import torch
    import torch.nn.functional as F
    from sparkx_dlsa import (
        DLSAIndexConfig,
        LOOP72_GLOBAL_VISIT_SCHEDULE,
        LOOP72_VISIT_SCHEDULE,
        PagedLatentIndex,
        VisitAwareDLSAIndexBank,
        VisitStateKey,
    )
    from sparkx_paged_cache import Loop72HybridPagedCache, PagedCacheConfig
    from sparkx_sparse_attention import (
        SparseAttentionConfig,
        compute_yarn_partial_rope_cos_sin,
        local_sliding_attention_forward,
        paged_dlsa_global_attention_forward,
    )
    from train_spark_dlsa import VisitDistillConfig, distill_single_visit_indexer
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


class TestLoop72CapacityAndCodecs(unittest.TestCase):

    def test_loop72_capacity_arithmetic_matches_plan_appendix(self):
        cap_5m = compute_loop72_capacity_table(n_tokens=5_000_000)
        self.assertEqual(cap_5m["total_visits"], 72)
        self.assertEqual(cap_5m["global_visits"], 18)
        self.assertEqual(cap_5m["local_visits"], 54)
        self.assertEqual(cap_5m["selector_map_parameters"], 1_769_472)
        self.assertEqual(cap_5m["global_visit_counts_by_layer"], {3: 1, 7: 2, 11: 2, 15: 3, 19: 3, 23: 3, 27: 2, 31: 1, 35: 1})
        self.assertAlmostEqual(cap_5m["local_rings_mib"], 108.0, places=2)
        self.assertAlmostEqual(cap_5m["unbounded_all72_bf16_gib"], 1373.29, places=1)
        self.assertAlmostEqual(cap_5m["global_payload_gib"]["16bit_gib"], 343.32, places=1)
        self.assertAlmostEqual(cap_5m["global_payload_gib"]["8bit_gib"], 171.66, places=1)
        self.assertAlmostEqual(cap_5m["global_payload_gib"]["4bit_gib"], 85.83, places=1)
        self.assertAlmostEqual(cap_5m["mixed_rotary_bf16_gib"], 118.02, places=1)

        cap_32k = compute_loop72_capacity_table(n_tokens=32768)
        self.assertAlmostEqual(cap_32k["unbounded_all72_bf16_gib"], 9.0, places=2)
        self.assertAlmostEqual(cap_32k["global_payload_gib"]["16bit_gib"], 2.25, places=2)

        cap_128k = compute_loop72_capacity_table(n_tokens=131072)
        self.assertAlmostEqual(cap_128k["unbounded_all72_bf16_gib"], 36.0, places=2)
        self.assertAlmostEqual(cap_128k["global_payload_gib"]["16bit_gib"], 9.0, places=2)
        self.assertAlmostEqual(cap_128k["global_payload_gib"]["8bit_gib"], 4.5, places=2)
        self.assertAlmostEqual(cap_128k["global_payload_gib"]["4bit_gib"], 2.25, places=2)

    def test_kv_page_codecs_roundtrip_and_byte_accounting(self):
        # 2 KV heads, 8 tokens, 16 head_dim (4 rotated + 12 free)
        k_page = [
            [
                [(h + 1) * 0.1 + t * 0.03 + ((d % 5) - 2) * 0.25 for d in range(16)]
                for t in range(8)
            ]
            for h in range(2)
        ]
        v_page = [
            [
                [(h + 1) * -0.08 + t * 0.02 + ((d % 7) - 3) * 0.15 for d in range(16)]
                for t in range(8)
            ]
            for h in range(2)
        ]

        bf16_cfg = KVCodecConfig(codec="bf16", group_size=8, rope_dim=4, rope_first=True)
        int8_cfg = KVCodecConfig(codec="int8", group_size=8, rope_dim=4, rope_first=True)
        int4_cfg = KVCodecConfig(codec="int4", group_size=8, rope_dim=4, rope_first=True)
        mixed_cfg = KVCodecConfig(codec="mixed_rotary_bf16", group_size=8, rope_dim=4, rope_first=True)

        p_bf16 = encode_kv_page(k_page, v_page, cfg=bf16_cfg, positions=range(8))
        p_int8 = encode_kv_page(k_page, v_page, cfg=int8_cfg, positions=range(8))
        p_int4 = encode_kv_page(k_page, v_page, cfg=int4_cfg, positions=range(8))
        p_mix = encode_kv_page(k_page, v_page, cfg=mixed_cfg, positions=range(8))

        self.assertLess(p_int8.storage_bytes(), p_bf16.storage_bytes())
        self.assertLess(p_int4.storage_bytes(), p_int8.storage_bytes())
        self.assertLess(p_mix.storage_bytes(), p_int8.storage_bytes())
        self.assertGreater(p_mix.storage_bytes(), p_int4.storage_bytes())

        err_int8 = measure_codec_error(k_page, v_page, int8_cfg)
        err_mix = measure_codec_error(k_page, v_page, mixed_cfg)
        self.assertLess(err_int8["key_rel_err"], 0.03)
        self.assertLess(err_mix["key_rel_err"], 0.15)
        self.assertEqual(err_mix["key_rotary_rel_err"], 0.0)

        # In mixed_rotary_bf16, the first 4 rotated dims of K must be exact
        k_dec_mix, _ = decode_kv_page(p_mix)
        if HAVE_TORCH and isinstance(k_dec_mix, torch.Tensor):
            k_dec_mix = k_dec_mix.tolist()
        for h in range(2):
            for t in range(8):
                for d in range(4):
                    self.assertAlmostEqual(k_dec_mix[h][t][d], k_page[h][t][d], places=5)

    def test_yaml_config_schema(self):
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "spark4b_loop72_lsa.yaml",
        )
        self.assertTrue(os.path.exists(cfg_path))
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = f.read()
        self.assertIn("preserve_native_activation: true", raw)
        self.assertIn("enable_p3: true", raw)
        self.assertIn("expected_visits: 72", raw)
        self.assertIn("global_visits: 18", raw)
        self.assertIn("local_visits: 54", raw)
        self.assertIn("maps_per_global_visit: true", raw)
        self.assertIn("construct: false", raw)
        self.assertIn("generation_logits: last_position", raw)

    def test_esm_serialization_and_checksum_verification(self):
        esm1 = EpisodicSideMemory(width=16, r=12, route_dim=8, use_torch=False)
        a_in = [[0.2 * ((i + j) % 5 - 2) for j in range(12)] for i in range(2)]
        delta_in = [[0.1 * ((i + 2 * j) % 7 - 3) for j in range(16)] for i in range(2)]
        h_query = [[0.15 * ((j % 5) - 2) for j in range(16)]]
        c_in = esm1.route_embed(h_query)
        rep = esm1.write(shard=0, A=a_in, Delta=delta_in, centroids=c_in)
        self.assertEqual(esm1.memory_epoch, 1)
        self.assertGreater(esm1.shard_storage_bytes(), 0)
        out_before = esm1.forward(h_query, use_gate=True)

        saved = esm1.esm_state_dict()
        esm2 = EpisodicSideMemory(width=16, r=12, route_dim=8, use_torch=False)
        load_info = esm2.load_esm_state_dict(saved, verify_checksum=True)
        self.assertEqual(load_info["shards_restored"], 1)
        self.assertEqual(esm2.memory_epoch, 1)
        out_after = esm2.forward(h_query, use_gate=True)
        for j in range(16):
            self.assertAlmostEqual(out_before[0][j], out_after[0][j], places=6)

        # Corrupting the checksum must raise ValueError when verify_checksum=True
        corrupted = dict(saved)
        corrupted["shards"] = {0: dict(saved["shards"][0], sha256="0" * 64)}
        with self.assertRaises(ValueError):
            esm2.load_esm_state_dict(corrupted, verify_checksum=True)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestLoop72TorchComponents(unittest.TestCase):

    def test_visit_schedule_and_visit_aware_dlsa_bank(self):
        self.assertEqual(len(LOOP72_VISIT_SCHEDULE), 72)
        self.assertEqual(len(LOOP72_GLOBAL_VISIT_SCHEDULE), 18)
        counts = {}
        for v in LOOP72_GLOBAL_VISIT_SCHEDULE:
            counts[v.physical_layer] = counts.get(v.physical_layer, 0) + 1
        self.assertEqual(counts, {3: 1, 7: 2, 11: 2, 15: 3, 19: 3, 23: 3, 27: 2, 31: 1, 35: 1})

        bank = VisitAwareDLSAIndexBank(
            base_cfg=DLSAIndexConfig(d_idx=64),
            maps_per_global_visit=True,
            per_visit_max_selected_keys={3: 3072, 31: 1536, 35: 1536},
        )
        self.assertEqual(bank.total_map_parameters(), 1_769_472)
        self.assertEqual(bank.get_visit_indexer(0).cfg.max_selected_keys, 3072)   # L3
        self.assertEqual(bank.get_visit_indexer(16).cfg.max_selected_keys, 1536)  # L31
        self.assertEqual(bank.get_visit_indexer(17).cfg.max_selected_keys, 1536)  # L35

    def test_paged_latent_index_append_and_tree_search(self):
        g = torch.Generator().manual_seed(7)
        k1 = F.normalize(torch.randn(180, 32, generator=g), dim=-1)
        k2 = F.normalize(torch.randn(140, 32, generator=g), dim=-1)
        idx = PagedLatentIndex(
            k1,
            n_sink=4,
            n_pool=4,
            n_cluster=64,
            avg_bucket_size=16,
            max_bucket_size=16,
            window=32,
            persistent_token_latents=False,
            search_mode="tree",
            branching_factor=4,
            beam_width=8,
        )
        self.assertIsNone(idx.k_idx_n)
        b1 = idx.n_blocks
        idx.append_new_keys(k2)
        self.assertGreater(idx.n_blocks, b1)
        q = F.normalize(torch.randn(32, generator=g), dim=-1)
        sel_tree, _ = idx.select_with_buckets(q, budget=32, search_mode="tree")
        sel_flat, _ = idx.select_with_buckets(q, budget=32, search_mode="flat")
        self.assertGreater(sel_tree.numel(), 0)
        self.assertEqual(sel_tree.numel(), sel_flat.numel())

    def test_loop72_hybrid_paged_cache_local_ring_and_global_gather(self):
        cfg = PagedCacheConfig(
            local_window=16,
            sinks=4,
            recent_tokens=16,
            global_hot_tokens_per_visit=32,
            kv_page_tokens=8,
            cold_codec="bf16",
        )
        cache = Loop72HybridPagedCache(config=cfg)
        self.assertEqual(len(cache.slots), 72)

        # Visit 0 is local (layer 0): feed 40 tokens in two chunks; resident ring must stay capped at 16
        k_loc1 = torch.randn(1, 4, 24, 16)
        v_loc1 = torch.randn(1, 4, 24, 16)
        k_out1, _, pos1 = cache.update_local_visit(0, k_loc1, v_loc1)
        self.assertEqual(k_out1.shape[2], 16)
        self.assertEqual(pos1.tolist(), list(range(8, 24)))
        k_loc2 = torch.randn(1, 4, 16, 16)
        v_loc2 = torch.randn(1, 4, 16, 16)
        k_out2, _, pos2 = cache.update_local_visit(0, k_loc2, v_loc2)
        self.assertEqual(k_out2.shape[2], 16)
        self.assertEqual(cache.get_logical_seq_length(0), 40)
        self.assertEqual(pos2.tolist(), list(range(24, 40)))

        # Visit 3 is global (layer 3): feed 48 tokens, verify gather_selected matches exact tokens
        # The BF16 archive is lossless for BF16 inputs, not arbitrary FP32 values.
        # Match the model KV dtype and require exact recovery across cold pages.
        k_glob = torch.randn(1, 4, 48, 16).to(torch.bfloat16)
        v_glob = torch.randn(1, 4, 48, 16).to(torch.bfloat16)
        slot_g = cache.append_global_visit(3, k_glob[:, :, :24], v_glob[:, :, :24])
        cache.append_global_visit(3, k_glob[:, :, 24:], v_glob[:, :, 24:])
        self.assertEqual(slot_g.logical_seq_length, 48)
        self.assertGreaterEqual(len(slot_g.cold_pages), 3)

        sel_pos = torch.tensor([0, 1, 10, 11, 25, 47], dtype=torch.long)
        k_sel, v_sel, g_pos = slot_g.gather_selected(sel_pos)
        self.assertEqual(g_pos.tolist(), sel_pos.tolist())
        self.assertTrue(torch.equal(k_sel[0], k_glob[0, :, sel_pos, :]))
        self.assertTrue(torch.equal(v_sel[0], v_glob[0, :, sel_pos, :]))

    def test_paged_dlsa_global_attention_strict_causality(self):
        g = torch.Generator().manual_seed(99)
        h_q, h_kv, t, d = 8, 2, 96, 32
        q_a = torch.randn(1, h_q, t, d, generator=g)
        k_a = torch.randn(1, h_kv, t, d, generator=g)
        v_a = torch.randn(1, h_kv, t, d, generator=g)

        # Mutate tokens >= 64 and verify prefix [0..64) outputs are 100% unchanged under paged sparse prefill
        q_b = q_a.clone()
        k_b = k_a.clone()
        v_b = v_a.clone()
        q_b[:, :, 64:, :] = torch.randn(1, h_q, t - 64, d, generator=g) * 5.0
        k_b[:, :, 64:, :] = torch.randn(1, h_kv, t - 64, d, generator=g) * 5.0
        v_b[:, :, 64:, :] = torch.randn(1, h_kv, t - 64, d, generator=g) * 5.0

        idx_cfg = DLSAIndexConfig(
            d_idx=16,
            n_kv_heads=h_kv,
            n_q_heads=h_q,
            head_dim=d,
            rope_dim=8,
            n_pool=4,
            n_cluster=16,
            k_clusters=4,
            sinks=4,
            recent_tokens=16,
            max_selected_keys=24,
            exact_threshold=32,
        )
        from sparkx_dlsa import FaithfulDLSAIndexer
        indexer = FaithfulDLSAIndexer(idx_cfg)
        p_cfg = PagedCacheConfig(
            local_window=16,
            sinks=4,
            recent_tokens=16,
            global_hot_tokens_per_visit=32,
            kv_page_tokens=8,
        )
        s_cfg = SparseAttentionConfig(exact_threshold=32, sparse_query_tile_tokens=8)

        from sparkx_sparse_attention import SparseRuntimeTelemetry
        telem_a = SparseRuntimeTelemetry()
        cache_a = Loop72HybridPagedCache(config=p_cfg)
        slot_a = cache_a.append_global_visit(3, k_a, v_a, indexer=indexer)
        out_a = paged_dlsa_global_attention_forward(
            module=torch.nn.Module(),
            query=q_a,
            slot=slot_a,
            indexer=indexer,
            q_positions=torch.arange(t),
            scaling=1.0 / (d ** 0.5),
            sparse_cfg=s_cfg,
            telemetry=telem_a,
        )
        self.assertGreaterEqual(telem_a.sparse_prefill_calls, 1)
        self.assertEqual(telem_a.dense_fallback_calls, 0)

        telem_b = SparseRuntimeTelemetry()
        cache_b = Loop72HybridPagedCache(config=p_cfg)
        slot_b = cache_b.append_global_visit(3, k_b, v_b, indexer=indexer)
        out_b = paged_dlsa_global_attention_forward(
            module=torch.nn.Module(),
            query=q_b,
            slot=slot_b,
            indexer=indexer,
            q_positions=torch.arange(t),
            scaling=1.0 / (d ** 0.5),
            sparse_cfg=s_cfg,
            telemetry=telem_b,
        )
        self.assertGreaterEqual(telem_b.sparse_prefill_calls, 1)
        self.assertEqual(telem_b.dense_fallback_calls, 0)
        max_prefix_diff = float((out_a[:, :, :64, :] - out_b[:, :, :64, :]).abs().max().item())
        self.assertLess(max_prefix_diff, 1e-6)


class TestLoop72BenchmarkSuiteAndNotebook(unittest.TestCase):

    def test_official_benchmark_table_and_modern_stream(self):
        import json
        from sparkx_loop72_dlsa_benchmarks import (
            OFFICIAL_BENCHMARK_TABLE,
            OFFICIAL_SPARK_X26_SAMPLING,
            SIGNIFICANT_BENCHMARK_SUITE,
            build_modern_long_eval_stream,
            compute_omp_hashline_tag,
            evaluate_cold_kv_codecs_summary,
            evaluate_oh_my_pi_hashline_task,
            format_official_comparison_markdown,
            generate_graphwalks_bfs_instance,
            generate_mrcr_v2_instance,
            generate_omp_models_yaml,
            run_full_length_1m_context_benchmark,
            score_aa_omniscience_v11,
            score_graphwalks_bfs_f1,
            score_mrcr_v2_response,
            verify_bfcl_tool_call,
            verify_code_stdio,
            verify_gdpval_deliverable,
            verify_json_schema,
            verify_math_answer,
            verify_scicode_in_sandbox,
            verify_swe_patch_in_sandbox,
            verify_terminal_bench_task,
        )

        self.assertTrue(OFFICIAL_SPARK_X26_SAMPLING["thinking_mode"])
        self.assertEqual(OFFICIAL_SPARK_X26_SAMPLING["thinking_level"], "high")
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["Average"]["Spark_X2.6_4B"], 63.3)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["Gaokao 2026"]["Spark_X2.6_4B"], 134.9)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["AIME 2026"]["Spark_X2.6_4B"], 91.2)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["HMMT Feb 26"]["Spark_X2.6_4B"], 81.3)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["Graphwalks bfs 256k f1"]["Spark_X2.6_4B"], 92.2)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["Openai_MRCR v2 256k-512k"]["Spark_X2.6_4B"], 65.6)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["AA-LCRv1.1"]["Spark_X2.6_4B"], 67.3)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["LongBench v2"]["Spark_X2.6_4B"], 53.3)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["IFEval"]["Spark_X2.6_4B"], 93.6)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["IFBench"]["Spark_X2.6_4B"], 75.8)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["GPQA Diamond"]["Spark_X2.6_4B"], 77.1)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["BFCL-V4"]["Spark_X2.6_4B"], 71.8)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["SWE Pro"]["Spark_X2.6_4B"], 54.3)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["SWE Verified"]["Spark_X2.6_4B"], 62.8)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["SWE Multilingual"]["Spark_X2.6_4B"], 64.7)
        self.assertEqual(OFFICIAL_BENCHMARK_TABLE["SciCode"]["Spark_X2.6_4B"], 36.8)
        self.assertIn("AA-Omniscience v1.1", OFFICIAL_BENCHMARK_TABLE)
        self.assertIn("Terminal-Bench 4.0", OFFICIAL_BENCHMARK_TABLE)
        self.assertIn("openai/gdpval", OFFICIAL_BENCHMARK_TABLE)

        # Verify Gaokao 2026 recurrence gold is 2944 ((3*8 - 1) * 2^7 = 23 * 128 = 2944)
        gk_item = next(it for it in SIGNIFICANT_BENCHMARK_SUITE if it["id"] == "gaokao_2026_seq_recurrence")
        self.assertEqual(gk_item["gold"], "2944")

        # Verify unclosed `<think>` fallback and physics unit stripping on GPQA Diamond
        self.assertTrue(verify_math_answer("<think>Reasoning step 1... \\boxed{2944}", "2944").ok)
        self.assertTrue(verify_math_answer("<think>Threshold energy is 7 m_p c^2</think>\\boxed{7 m_p c^2}", "7").ok)
        self.assertTrue(verify_math_answer("<think>Net bill is $103,000 USD. Therefore final answer = $103,000", "103000").ok)

        stream = build_modern_long_eval_stream(target_tokens=1024)
        self.assertGreater(len(stream.split()), 700)

        gw = generate_graphwalks_bfs_instance(num_nodes=48, branching=2, target_depth=2, seed=42)
        self.assertEqual(len(gw["gold_nodes"]), 4)
        gw_perfect = score_graphwalks_bfs_f1("Final Answer: [" + ", ".join(gw["gold_nodes"]) + "]", gw["gold_nodes"])
        self.assertEqual(gw_perfect["f1"], 1.0)

        mrcr = generate_mrcr_v2_instance(num_rounds=16, target_occurrence=3, seed=101)
        mrcr_ok = score_mrcr_v2_response(f"Answer: \\boxed{{{mrcr['gold_answer']}}}", mrcr["gold_answer"])
        self.assertTrue(mrcr_ok["exact_match"])

        for item in SIGNIFICANT_BENCHMARK_SUITE:
            if item["verifier"] == "math":
                res = verify_math_answer(f"\\boxed{{{item['gold']}}}", item["gold"])
                self.assertTrue(res.ok, f"Failed gold verification for {item['id']}")
            elif item["verifier"] == "bfcl_tool":
                t_spec = item["tool_spec"]
                # Verify both `{"name": ..., "arguments": ...}` and `<tool_call>` formats execute in sandbox
                payload_name = {"name": t_spec["expected_tool"], "arguments": t_spec["expected_args"]}
                res_name = verify_bfcl_tool_call(f"<tool_call>{json.dumps(payload_name)}</tool_call>", t_spec)
                self.assertTrue(res_name.ok, f"Failed tool verification for {item['id']}: {res_name.detail}")
                self.assertTrue(res_name.detail.get("sandbox_executed"))
            elif item["verifier"] == "swe_sandbox":
                if item["id"] == "swe_pro_lru_page_eviction_bug":
                    fix_code = (
                        "```python\n"
                        "class HotPageLRU:\n"
                        "    def __init__(self, capacity: int):\n"
                        "        self.capacity = int(capacity)\n"
                        "        self.order = OrderedDict()\n"
                        "    def access(self, page_id: int) -> list:\n"
                        "        if page_id in self.order:\n"
                        "            self.order.move_to_end(page_id)\n"
                        "        else:\n"
                        "            if len(self.order) >= self.capacity:\n"
                        "                self.order.popitem(last=False)\n"
                        "            self.order[page_id] = True\n"
                        "        return list(self.order.keys())\n"
                        "```"
                    )
                elif item["id"] == "swe_verified_fenwick_range_query_bug":
                    fix_code = "```python\ndef range_token_mass(weights: list, left: int, right: int) -> int:\n    return sum(weights[left : right + 1])\n```"
                else:
                    fix_code = "```python\ndef truncate_utf8_bytes(text: str, max_bytes: int) -> str:\n    return text.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')\n```"
                swe_res = verify_swe_patch_in_sandbox(fix_code, item["swe_spec"])
                self.assertTrue(swe_res.ok, f"Failed SWE sandbox for {item['id']}: {swe_res.detail}")
            elif item["verifier"] == "scicode_sandbox":
                sci_res = verify_scicode_in_sandbox(
                    "```python\nimport sys\nsa, sb = map(int, sys.stdin.readline().split())\nx1, x2, y1, y2 = map(float, sys.stdin.readline().split())\nprint(f'{(sa*sb)/(sa+sb)*((x1-y1)**2+(x2-y2)**2):.4f}')\n```",
                    item,
                )
                self.assertTrue(sci_res.ok)
            elif item["verifier"] == "omniscience":
                omni_ok = score_aa_omniscience_v11(f"\\boxed{{{item['gold']}}}", item["gold"])
                self.assertTrue(omni_ok["passed"])
                self.assertEqual(omni_ok["omniscience_index"], 100.0)
                omni_abs = score_aa_omniscience_v11("\\boxed{ABSTAIN}", item["gold"])
                self.assertEqual(omni_abs["omniscience_index"], 0.0)
                omni_bad = score_aa_omniscience_v11("\\boxed{999}", item["gold"])
                self.assertEqual(omni_bad["omniscience_index"], -100.0)
            elif item["verifier"] == "terminal_bench":
                tb_ok = verify_terminal_bench_task(
                    "```bash\ngit worktree add /srv/loop72-bench loop72-dlsa\ncd /srv/loop72-bench && python3 -m unittest discover -s tests -p \"test_*.py\"\n```",
                    item["terminal_spec"],
                )
                self.assertTrue(tb_ok.ok)
                self.assertTrue(tb_ok.detail.get("sandbox_ok"))
            elif item["verifier"] == "gdpval":
                gdp_ok = verify_gdpval_deliverable(
                    '```json\n{"global_bf16_gib": 72.0, "global_int4_gib": 18.0, "gib_saved_per_session": 54.0, "gpu_hot_kv_mib": 684, "recommended_tiering": "M1_GPU_Hot_M2_Host_Cold"}\n```',
                    item["gdpval_spec"],
                )
                self.assertTrue(gdp_ok.ok)

        # Local synthetic answers cannot fill an official comparison table.
        all_bench_names = [b for b in OFFICIAL_BENCHMARK_TABLE.keys() if b != "Average"]
        synthetic_live = {"tasks": [{"benchmark": b, "score": 100.0} for b in all_bench_names]}
        md_full = format_official_comparison_markdown(synthetic_live)
        self.assertIn("| — |", md_full)
        self.assertNotIn("**150.0**", md_full)
        self.assertIn("excluded from this comparison", md_full)

        # Test 1M context window benchmark suite (capacity + long-context specs)
        ctx_1m = run_full_length_1m_context_benchmark(model=None, tokenizer=None)
        self.assertIsNone(ctx_1m["context_limit_tokens"])
        self.assertEqual(ctx_1m["target_context_tokens"], 1_048_576)
        self.assertEqual(ctx_1m["status"], "not_run")
        self.assertIsNone(ctx_1m["validated_context_tokens"])
        self.assertFalse(ctx_1m["quality_validated"])
        self.assertAlmostEqual(ctx_1m["capacity_1m_summary"]["global_18_bf16_gib"], 72.0, places=2)
        self.assertAlmostEqual(ctx_1m["capacity_1m_summary"]["global_18_int4_gib"], 18.0, places=2)
        self.assertAlmostEqual(ctx_1m["capacity_1m_summary"]["local_54_rings_mib"], 108.0, places=2)

        # Test Oh My Pi (omp) hashline edit verification & models.yml generation
        src_lines = ["def f():", "    x = 1", "    return x"]
        tag_2 = compute_omp_hashline_tag(2, src_lines[1]).split("|")[0]
        omp_v = evaluate_oh_my_pi_hashline_task(
            f'```json\n{{"anchor": "{tag_2}", "replacement": "    x = 72"}}\n```',
            src_lines,
            "x = 72",
        )
        self.assertTrue(omp_v.ok)
        omp_yaml = generate_omp_models_yaml(output_path=None, port=8000)
        self.assertEqual(omp_yaml["context_window"], 131_072)
        self.assertIn('"contextWindow": 131072', omp_yaml["yaml"])
        self.assertIn('"supportsUsageInStreaming": true', omp_yaml["yaml"])

        json_ok = verify_json_schema(
            '```json\n{"model": "Spark", "total_visits": 72, "thinking_mode": true}\n```',
            {"required_keys": ["model", "total_visits", "thinking_mode"], "key_types": {"model": "str", "total_visits": "int", "thinking_mode": "bool"}},
        )
        self.assertTrue(json_ok.ok)

        code_ok = verify_code_stdio(
            "```python\nimport sys\nsa, sb = map(int, sys.stdin.readline().split())\nx1, x2, y1, y2 = map(float, sys.stdin.readline().split())\nprint(f'{(sa*sb)/(sa+sb)*((x1-y1)**2+(x2-y2)**2):.4f}')\n```",
            [{"stdin": "4 4\n1.0 0.0 0.0 1.0\n", "expected_stdout": "4.0000"}],
        )
        self.assertTrue(code_ok.ok)

        codecs = evaluate_cold_kv_codecs_summary()
        self.assertEqual(codecs["mixed_rotary_bf16"]["key_rotary_rel_err"], 0.0)
        md = format_official_comparison_markdown()
        self.assertIn("Graphwalks bfs 256k f1", md)
        self.assertIn("Gaokao 2026", md)
        self.assertIn("AA-Omniscience v1.1", md)
        self.assertIn("Terminal-Bench 4.0", md)
        self.assertIn("openai/gdpval", md)

    def test_loader_and_module_interface_contracts_static(self):
        import ast

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = [
            "sparkx_dlsa.py",
            "sparkx_paged_cache.py",
            "sparkx_sparse_attention.py",
            "sparkx_loop_v03_core.py",
            "load_inference_v03.py",
            "train_spark_dlsa.py",
            "sparkx_loop72_dlsa_benchmarks.py",
        ]
        trees = {}
        for fn in files:
            p = os.path.join(root, fn)
            with open(p, "r", encoding="utf-8") as f:
                trees[fn] = ast.parse(f.read(), filename=fn)

        dlsa_defs = {n.name for n in ast.walk(trees["sparkx_dlsa.py"]) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        self.assertIn("DLSAIndexConfig", dlsa_defs)
        self.assertIn("VisitAwareDLSAIndexBank", dlsa_defs)
        self.assertIn("get_visit_indexer", dlsa_defs)
        self.assertIn("total_map_parameters", dlsa_defs)
        self.assertIn("select_with_buckets", dlsa_defs)

        # Verify every `base_cfg.<attr>` or `base_idx_cfg.<attr>` accessed in `sparkx_dlsa.py` and `load_inference_v03.py`
        # exists on `DLSAIndexConfig`
        cfg_fields = set()
        for node in ast.walk(trees["sparkx_dlsa.py"]):
            if isinstance(node, ast.ClassDef) and node.name == "DLSAIndexConfig":
                for sub in node.body:
                    if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                        cfg_fields.add(sub.target.id)
        for fn in ("sparkx_dlsa.py", "load_inference_v03.py"):
            for node in ast.walk(trees[fn]):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("base_cfg", "base_idx_cfg"):
                    self.assertIn(
                        node.attr,
                        cfg_fields,
                        f"{fn} accesses `{node.value.id}.{node.attr}`, which is missing from DLSAIndexConfig!",
                    )

        cache_defs = {n.name for n in ast.walk(trees["sparkx_paged_cache.py"]) if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        self.assertIn("Loop72HybridPagedCache", cache_defs)
        self.assertIn("is_fully_gpu_resident", cache_defs)
        self.assertIn("get_resident_prefix", cache_defs)
        self.assertIn("update_local_visit", cache_defs)
        self.assertIn("append_global_visit", cache_defs)

    def test_colab_benchmark_notebook_valid_json_and_syntax(self):
        import ast
        import json

        nb_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "SparkX_Loop72_DLSA_Benchmark_Colab.ipynb",
        )
        self.assertTrue(os.path.exists(nb_path))
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        self.assertEqual(nb.get("nbformat"), 4)
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 5)
        for idx, cell in enumerate(code_cells):
            src = "".join(cell.get("source", []))
            ast.parse(src, filename=f"SparkX_Loop72_DLSA_Benchmark_Colab.ipynb#cell_{idx}")

    def test_hf_row_conversion_and_real_verifiers(self):
        from sparkx_loop72_dlsa_benchmarks import (
            SIGNIFICANT_BENCHMARK_SUITE,
            convert_hf_row_to_eval_task,
            score_aa_omniscience_v11,
            verify_bfcl_tool_call,
            verify_ifeval_constraints,
            verify_math_answer,
            verify_mcq_choice,
            verify_swe_patch_in_sandbox,
        )

        # 1. AIME 2026 (`MathArena/aime_2026` schema)
        aime_task = convert_hf_row_to_eval_task(
            "AIME 2026",
            {"problem_idx": 1, "problem": "Find the remainder when 2^10 is divided by 1000.", "answer": "24"},
            row_idx=0,
        )
        self.assertIsNotNone(aime_task)
        self.assertEqual(aime_task["source"], "hf_real")
        self.assertEqual(aime_task["verifier"], "math")
        self.assertTrue(verify_math_answer("\\boxed{24}", aime_task["gold"]).ok)

        # 2. IMO-AnswerBench (`Hwilner/imo-answerbench` schema)
        imo_task = convert_hf_row_to_eval_task(
            "IMO-AnswerBench",
            {"Problem ID": "alg-01", "Problem": "Solve for x: 3x = 27.", "Short Answer": "$9$"},
            row_idx=0,
        )
        self.assertIsNotNone(imo_task)
        self.assertEqual(imo_task["gold"], "9")
        self.assertTrue(verify_math_answer("\\boxed{9}", imo_task["gold"]).ok)

        # 3. LongBench v2 (`zai-org/LongBench-v2` schema)
        lb_task = convert_hf_row_to_eval_task(
            "LongBench v2",
            {
                "context": "Spark-X2.6-4B uses 18 global attention visits across 72 virtual layers.",
                "question": "How many global attention visits does Spark-X2.6-4B use in Loop72?",
                "choice_A": "9",
                "choice_B": "18",
                "choice_C": "36",
                "choice_D": "72",
                "answer": "B",
            },
            row_idx=0,
        )
        self.assertIsNotNone(lb_task)
        self.assertEqual(lb_task["verifier"], "mcq")
        self.assertTrue(verify_mcq_choice("<think>It uses 18 visits.</think>\\boxed{B}", lb_task["mcq_spec"]).ok)

        # 4. GPQA Diamond (`Idavidrein/gpqa` schema)
        gpqa_task = convert_hf_row_to_eval_task(
            "GPQA Diamond",
            {
                "Question": "What is the proton threshold energy factor?",
                "Correct Answer": "7 m_p c^2",
                "Incorrect Answer 1": "3 m_p c^2",
                "Incorrect Answer 2": "5 m_p c^2",
                "Incorrect Answer 3": "9 m_p c^2",
            },
            row_idx=1,
        )
        self.assertIsNotNone(gpqa_task)
        gold_ltr = gpqa_task["mcq_spec"]["gold_letter"]
        self.assertTrue(verify_mcq_choice(f"Final Answer: \\boxed{{{gold_ltr}}}", gpqa_task["mcq_spec"]).ok)

        # 5. IFEval (`google/IFEval` schema)
        ifeval_task = convert_hf_row_to_eval_task(
            "IFEval",
            {
                "key": 1001,
                "prompt": "Write a short sentence about sparse attention without using any commas and include the word hypersphere.",
                "instruction_id_list": ["punctuation:no_comma", "keywords:existence", "length_constraints:number_words"],
                "kwargs": [{}, {"keywords": ["hypersphere"]}, {"relation": "at least", "num_words": 5}],
            },
            row_idx=0,
        )
        self.assertIsNotNone(ifeval_task)
        if_res = verify_ifeval_constraints(
            "Latent sparse attention projects queries onto a unit hypersphere cleanly.",
            ifeval_task["ifeval_spec"],
        )
        self.assertTrue(if_res.ok, f"IFEval failed: {if_res.detail}")

        # 6. BFCL-V4 (`gorilla-llm/Berkeley-Function-Calling-Leaderboard` schema + positional AST call)
        bfcl_task = convert_hf_row_to_eval_task(
            "BFCL-V4",
            {
                "id": "simple_0",
                "question": [[{"role": "user", "content": "Calculate triangle area with base 10 and height 5."}]],
                "function": [
                    {
                        "name": "calculate_triangle_area",
                        "parameters": {
                            "type": "dict",
                            "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
                            "required": ["base", "height"],
                        },
                    }
                ],
                "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
            },
            row_idx=0,
        )
        self.assertIsNotNone(bfcl_task)
        # Test positional Python AST call matching
        pos_res = verify_bfcl_tool_call("```python\ncalculate_triangle_area(10, 5)\n```", bfcl_task["tool_spec"])
        self.assertTrue(pos_res.ok, f"Positional AST tool call failed: {pos_res.detail}")

        # 7. SWE Pro method-only patch (when model emits only `def access(self, page_id: int)` without `class HotPageLRU:`)
        swe_pro_item = next(it for it in SIGNIFICANT_BENCHMARK_SUITE if it["id"] == "swe_pro_lru_page_eviction_bug")
        method_only_patch = (
            "```python\n"
            "def access(self, page_id: int) -> list:\n"
            "    if page_id in self.order:\n"
            "        self.order.move_to_end(page_id)\n"
            "    else:\n"
            "        if len(self.order) >= self.capacity:\n"
            "            self.order.popitem(last=False)\n"
            "        self.order[page_id] = True\n"
            "    return list(self.order.keys())\n"
            "```"
        )
        swe_method_res = verify_swe_patch_in_sandbox(method_only_patch, swe_pro_item["swe_spec"])
        self.assertTrue(swe_method_res.ok, f"Method-only SWE Pro patch failed: {swe_method_res.detail}")

        # 8. AA-Omniscience v1.1 when prose mentions 'do not ABSTAIN' alongside the right boxed answer
        omni_res = score_aa_omniscience_v11(
            "Since the hidden size is known (do not ABSTAIN here), Final Answer: \\boxed{2560}",
            "2560",
            should_abstain=False,
        )
        self.assertTrue(omni_res["passed"])
        self.assertEqual(omni_res["normalized_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
