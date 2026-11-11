import json
import tempfile
import unittest
from pathlib import Path

import sparkx_stat_benchmark as stat
from sparkx_full_benchmark import ResumeIntegrityError


class _DummyLoopModel:
    def __init__(self):
        self.loop_passes = 2
        self.use_mhc = True
        self.enable_p3 = True

        class _SparseCfg:
            exact_threshold = 4612

        self.sparse_attn_cfg = _SparseCfg()

    def loop_config(self, *, passes=None, use_mhc=None, use_engram=None, enable_p3=None, use_esm=None, **_kw):
        model = self
        prev = (model.loop_passes, model.use_mhc, model.enable_p3)

        class _Ctx:
            def __enter__(self_inner):
                if passes is not None:
                    model.loop_passes = passes
                if use_mhc is not None:
                    model.use_mhc = use_mhc
                if enable_p3 is not None:
                    model.enable_p3 = enable_p3
                return model

            def __exit__(self_inner, exc_type, exc, tb):
                model.loop_passes, model.use_mhc, model.enable_p3 = prev
                return False

        return _Ctx()

    def execution_manifest(self):
        return {
            "total_virtual_layers": 72 if self.loop_passes == 2 else 36,
            "effective_recurrence_mode": "learned_mhc" if self.use_mhc else "off",
        }


class StatisticalBenchmarkTests(unittest.TestCase):
    def test_fpc_and_confidence_intervals(self):
        # 1. Complete finite census (n == N == 30) -> zero sampling error
        scores_30 = [100.0] * 24 + [0.0] * 6
        census = stat.compute_statistical_summary(scores_30, population_size=30, population_exact=True)
        self.assertEqual(census["fpc_factor"], 0.0)
        self.assertEqual(census["se_mean"], 0.0)
        self.assertEqual(census["margin_of_error"], 0.0)
        self.assertEqual(census["ci_low"], 80.0)
        self.assertEqual(census["ci_high"], 80.0)
        self.assertEqual(census["ci_method"], "exact_finite_census")

        # 2. Binary sample n=15 out of N=30 has strictly tighter CI than infinite population
        sub_15 = [100.0] * 12 + [0.0] * 3
        fin = stat.compute_statistical_summary(sub_15, population_size=30, population_exact=True)
        inf = stat.compute_statistical_summary(sub_15, population_size=None, population_exact=False)
        self.assertLess(fin["margin_of_error"], inf["margin_of_error"])
        self.assertGreaterEqual(fin["ci_low"], 0.0)
        self.assertLessEqual(fin["ci_high"], 100.0)

        # 3. Perfect 15/15 sample with N=200 never falsely claims +-0.0%
        all_100 = [100.0] * 15
        w_100 = stat.compute_statistical_summary(all_100, population_size=200, population_exact=True)
        self.assertLess(w_100["ci_low"], 100.0)
        self.assertEqual(w_100["ci_high"], 100.0)
        self.assertGreater(w_100["margin_of_error"], 2.0)

        # 4. Continuous scores use Rao-Wu FPC bootstrap deterministically
        cont = [12.5, 45.0, 62.5, 75.0, 88.0, 91.5, 55.0, 68.0, 79.5, 83.0]
        c1 = stat.compute_statistical_summary(cont, population_size=40, population_exact=True, seed=42)
        c2 = stat.compute_statistical_summary(cont, population_size=40, population_exact=True, seed=42)
        c_unbounded = stat.compute_statistical_summary(cont, population_size=40, population_exact=False, seed=42)
        self.assertEqual(c1, c2)
        self.assertEqual(c1["ci_method"], "rao_wu_bootstrap_fpc")
        self.assertLess(c1["margin_of_error"], c_unbounded["margin_of_error"])

        # 5. Cochran sample size shrinks under finite N
        n_inf = stat.recommend_sample_size(7.5, prior_p=0.7, population_size=None)
        n_200 = stat.recommend_sample_size(7.5, prior_p=0.7, population_size=200)
        n_30 = stat.recommend_sample_size(7.5, prior_p=0.7, population_size=30)
        self.assertLess(n_200, n_inf)
        self.assertLessEqual(n_30, 30)

    def test_stratified_sampling_eliminates_prefix_bias_and_filters_pre_sample(self):
        # Construct 30 AIME-like problems ordered easy -> medium -> hard, plus 2 invalid rows
        rows = []
        for i in range(30):
            tier = "easy" if i < 10 else ("med" if i < 20 else "hard")
            rows.append({"id": f"p_{i}", "problem": f"[{tier}] Compute {i} + 1.", "answer": str(i + 1)})
        rows.append({"id": "bad_empty", "problem": "", "answer": ""})

        spec = stat.STAT_SUITE_SPECS["aime_2026"]
        sampled, meta = stat.build_candidate_frame_and_sample(
            spec,
            sample_size=12,
            seed=20261110,
            rows=rows,
        )
        self.assertEqual(meta["rows_seen"], 31)
        self.assertEqual(meta["excluded_unsupported"], 1)
        self.assertEqual(meta["eligible_population_in_frame"], 30)
        self.assertTrue(meta["population_exact_for_frame"])
        self.assertEqual(len(sampled), 12)
        # Proportional allocation across 3 equal tertiles of 10 -> exactly 4 from each tertile!
        self.assertEqual(meta["stratum_sample_counts"], {"pos_bin_0": 4, "pos_bin_1": 4, "pos_bin_2": 4})
        # Definitely NOT the first 12 rows!
        self.assertNotEqual([c["row_index"] for c in sampled], list(range(12)))

        # Check scan_cap disables population_exact_for_frame when stream continues past cap
        _, capped_meta = stat.build_candidate_frame_and_sample(
            spec,
            sample_size=6,
            seed=20261110,
            max_scan_rows=15,
            rows=rows,
        )
        self.assertFalse(capped_meta["scan_exhausted"])
        self.assertFalse(capped_meta["population_exact_for_frame"])

    def test_ifeval_unsupported_rules_excluded_before_sampling(self):
        rows = [
            {
                "key": 1,
                "prompt": "Write two sentences without commas.",
                "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}],
            },
            {
                "key": 2,
                "prompt": "Translate using a fictional unsupported rule.",
                "instruction_id_list": ["custom_unimplemented:foobar_rule"],
                "kwargs": [{}],
            },
            {
                "key": 3,
                "prompt": "Reply in ALL CAPS.",
                "instruction_id_list": ["change_case:english_capital"],
                "kwargs": [{}],
            },
        ]
        spec = stat.STAT_SUITE_SPECS["ifeval"]
        sampled, meta = stat.build_candidate_frame_and_sample(spec, sample_size=2, seed=99, rows=rows)
        self.assertEqual(meta["excluded_unsupported"], 1)
        self.assertEqual(meta["eligible_population_in_frame"], 2)
        self.assertEqual(len(sampled), 2)
        self.assertEqual([c["row"]["key"] for c in sampled], [1, 3])

    def test_resumable_suite_and_honesty_invariants(self):
        rows = [
            {"_id": f"g_{i}", "problem_type": "bfs" if i % 2 == 0 else "parents", "prompt": f"BFS graph {i}", "answer_nodes": ["a"]}
            for i in range(20)
        ]
        spec = stat.STAT_SUITE_SPECS["graphwalks_bfs_128k"]
        seen_ids = []

        def fake_eval_crash_after_2(_model, _tok, task, **_kw):
            seen_ids.append(task["row_id"])
            if len(seen_ids) == 3:
                raise RuntimeError("simulated_disconnect")
            return {
                "status": "completed",
                "score": 100.0 if int(str(task["row_id"]).split("_")[1]) % 4 == 0 else 50.0,
                "generated_tokens": 320,
                "timing": {"decode_tok_per_s": 27.5},
            }

        with tempfile.TemporaryDirectory() as tmp:
            # First run: processes 2 items cleanly, 3rd hits error; simulate half-written 4th line too
            def _eval_stop(_model, _tok, task, **_kw):
                if len(seen_ids) >= 2:
                    raise KeyboardInterrupt("disconnect")
                seen_ids.append(task["row_id"])
                return {
                    "status": "completed",
                    "score": 75.0,
                    "generated_tokens": 300,
                    "timing": {"decode_tok_per_s": 27.5},
                }

            with self.assertRaises(KeyboardInterrupt):
                stat.run_statistical_suite(
                    _DummyLoopModel(),
                    None,
                    spec,
                    tmp,
                    sample_size=5,
                    seed=1234,
                    rows=rows,
                    evaluator=_eval_stop,
                    verbose=False,
                    resume=True,
                )
            jsonl_file = Path(tmp) / f"{spec.name}.stat.jsonl"
            with jsonl_file.open("a", encoding="utf-8") as fh:
                fh.write('{"row_index": 999, "truncated": ')

            # Resume remaining 3 items
            completed_calls = []

            def _eval_finish(_model, _tok, task, **_kw):
                completed_calls.append(task["row_id"])
                return {
                    "status": "completed",
                    "score": 100.0,
                    "generated_tokens": 250,
                    "timing": {"decode_tok_per_s": 28.0},
                }

            summary = stat.run_statistical_suite(
                _DummyLoopModel(),
                None,
                spec,
                tmp,
                sample_size=5,
                seed=1234,
                rows=rows,
                evaluator=_eval_finish,
                verbose=False,
                resume=True,
            )
            self.assertEqual(len(seen_ids), 2)
            self.assertEqual(len(completed_calls), 3)
            self.assertEqual(summary["counts"]["scored"], 5)
            self.assertEqual(summary["counts"]["resumed"], 2)
            self.assertFalse(summary["complete"])
            self.assertFalse(summary["official_comparable"])
            self.assertFalse(summary["published_score_comparable"])
            self.assertEqual(summary["sampling_mode"], "statistical_subset")
            self.assertAlmostEqual(summary["statistics"]["mean"], (2 * 75.0 + 3 * 100.0) / 5.0, places=3)

            # Tampering with a row must fail closed with ResumeIntegrityError
            bad_rows = [dict(r) for r in rows]
            for r in bad_rows:
                r["prompt"] = r["prompt"] + " TAMPERED"
            with self.assertRaises(ResumeIntegrityError):
                stat.run_statistical_suite(
                    _DummyLoopModel(),
                    None,
                    spec,
                    tmp,
                    sample_size=5,
                    seed=1234,
                    rows=bad_rows,
                    evaluator=_eval_finish,
                    verbose=False,
                    resume=True,
                )

    def test_paired_common_random_numbers_ablation(self):
        rows = [{"id": f"q_{i}", "problem": f"Solve x = {i}", "answer": str(i)} for i in range(16)]
        model = _DummyLoopModel()
        observed_modes = []

        def fake_eval(m, _tok, task, **_kw):
            is_72l = m.loop_passes == 2 and m.use_mhc
            observed_modes.append((is_72l, m.sparse_attn_cfg.exact_threshold, task["row_index"]))
            idx = int(task["row_index"])
            if is_72l:
                score = 100.0 if idx % 4 != 3 else 0.0  # 75%
            else:
                score = 100.0 if idx % 2 == 0 else 0.0  # 50% (strict subset of 72L wins!)
            return {"status": "completed", "score": score, "generated_tokens": 200, "timing": {"decode_tok_per_s": 27.0}}

        with tempfile.TemporaryDirectory() as tmp:
            paired = stat.run_paired_ablation_suite(
                model,
                None,
                "aime_2026",
                tmp,
                sample_size=12,
                seed=2026,
                rows=rows,
                evaluator=fake_eval,
                verbose=False,
            )
            # Model state restored cleanly after 36L context
            self.assertEqual(model.loop_passes, 2)
            self.assertTrue(model.use_mhc)
            self.assertEqual(model.sparse_attn_cfg.exact_threshold, 4612)

            comp = paired["paired_comparison"]
            self.assertEqual(comp["n_paired"], 12)
            self.assertGreater(comp["mean_diff"], 0.0)
            self.assertEqual(comp["wins_b"], 0)
            self.assertFalse(paired["complete"])
            self.assertFalse(paired["official_comparable"])

            # Verify identical row indices evaluated in 72L and 36L passes
            idx_72 = [idx for is_72, _th, idx in observed_modes if is_72]
            idx_36 = [idx for is_72, _th, idx in observed_modes if not is_72]
            self.assertEqual(idx_72, idx_36)


if __name__ == "__main__":
    unittest.main()
