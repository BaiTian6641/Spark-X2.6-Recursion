import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sparkx_full_benchmark import NATIVE_PROTOCOLS, preflight, run_dataset, run_full_benchmarks


class Model:
    loaded_meta = {"fixture": True}
    def execution_manifest(self):
        return {"total_virtual_layers": 72, "effective_recurrence_mode": "learned_mhc"}


def convert(name, row, row_idx=0, task_mode="official"):
    return {"id": str(row_idx), "prompt": row["prompt"], "gold": row.get("answer")}


def evaluate(model, tokenizer, item, **kwargs):
    return {"status": "completed", "score": 100.0, "passed": True, "input_tokens": len(item["prompt"]),
            "benchmark_id": item["benchmark_id"], "reference_adapter_key": item["reference_adapter_key"]}


class TestFullBenchmark(unittest.TestCase):
    def test_all_rows_and_declared_filter_are_accounted(self):
        rows = [{"problem_type": "bfs", "prompt": "a", "answer": ["b"]},
                {"problem_type": "parents", "prompt": "c", "answer": ["d"]},
                {"problem_type": "bfs", "prompt": "e", "answer": ["f"]}]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dataset(Model(), None, NATIVE_PROTOCOLS["graphwalks_bfs"], tmp,
                                 rows=rows, evaluator=evaluate, converter=convert)
            self.assertEqual(result["attempted"], 2)
            self.assertEqual(result["excluded_by_declared_filter"], 1)
            self.assertTrue(result["complete"])
            self.assertFalse(result["model_quality_run"])
            self.assertFalse(result["published_score_comparable"])
            records = [json.loads(x) for x in Path(result["predictions"]).read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(len(x["row_sha256"]) == 64 for x in records))
            self.assertTrue(all(x["benchmark_id"] == "graphwalks_bfs" for x in records))
            self.assertTrue(all(x["reference_adapter_key"] == NATIVE_PROTOCOLS["graphwalks_bfs"].benchmark_name for x in records))

    def test_numeric_score_from_unsupported_result_is_rejected(self):
        def invalid_score(*args, **kwargs):
            return {"status": "unsupported", "score": 100.0}
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dataset(Model(), None, NATIVE_PROTOCOLS["longbench_v2"], tmp,
                                 rows=[{"prompt": "a"}], evaluator=invalid_score, converter=convert)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["score"])
            self.assertEqual(result["errors"], 1)

    def test_limit_never_claims_complete_even_for_short_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dataset(Model(), None, NATIVE_PROTOCOLS["longbench_v2"], tmp,
                                 rows=[{"prompt": "a"}], evaluator=evaluate, converter=convert, limit=1)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["score"])

    def test_blocked_row_and_stream_error_prevent_full_score(self):
        def fail(model, tokenizer, item, **kwargs):
            return {"status": "unsupported", "score": None, "reason": "context too long"}
        def broken_stream():
            yield {"prompt": "a"}
            raise RuntimeError("dataset download interrupted")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_dataset(Model(), None, NATIVE_PROTOCOLS["longbench_v2"], tmp,
                                 rows=broken_stream(), evaluator=fail, converter=convert)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["score"])
            self.assertEqual(result["blocked"], 1)
            self.assertIn("interrupted", result["suite_error"])

    def test_external_harnesses_cannot_turn_into_fake_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_full_benchmarks(Model(), None, tmp, suites=["swe_verified", "terminal_bench_4"])
            self.assertFalse(result["complete"])
            self.assertIsNone(result["macro_score"])
            self.assertTrue(all(x["score"] is None for x in result["suites"]))
            self.assertIn("4.0.0", result["suites"][1]["reason"])

    def test_existing_predictions_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "longbench_v2.jsonl"
            output.write_text("keep me")
            with self.assertRaises(FileExistsError):
                run_dataset(Model(), None, NATIVE_PROTOCOLS["longbench_v2"], tmp,
                            rows=[], evaluator=evaluate, converter=convert)
            self.assertEqual(output.read_text(), "keep me")

    def test_preflight_is_not_a_model_benchmark(self):
        result = preflight(["swe_verified"])
        self.assertFalse(result["model_benchmark_executed"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["suites"][0]["score"])

    def test_all_blocked_default_inputs_do_not_claim_model_quality_execution(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("sparkx_full_benchmark.load_pinned_rows", return_value=iter([{"prompt": "a"}])), \
             patch("sparkx_loop72_dlsa_benchmarks.convert_hf_row_to_eval_task", return_value={"benchmark_id": "LongBench v2"}), \
             patch("sparkx_loop72_dlsa_benchmarks.evaluate_benchmark_task", return_value={"status": "blocked", "score": None}):
            result = run_dataset(Model(), None, NATIVE_PROTOCOLS["longbench_v2"], tmp)
            self.assertTrue(result["intended_model_run"])
            self.assertFalse(result["model_quality_run"])
            self.assertFalse(result["complete"])

    def test_resume_skips_verified_rows_recovers_truncated_tail_and_rejects_tampering(self):
        rows = [
            {"problem_type": "bfs", "prompt": "r0", "answer": ["a0"]},
            {"problem_type": "parents", "prompt": "skip", "answer": ["x"]},
            {"problem_type": "bfs", "prompt": "r2", "answer": ["a2"]},
            {"problem_type": "bfs", "prompt": "r3", "answer": ["a3"]},
        ]
        seen_prompts: list[str] = []

        def tracked_eval(model, tokenizer, item, **kwargs):
            seen_prompts.append(item["prompt"])
            return evaluate(model, tokenizer, item, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            # Step 1: Run partial subset of 1 eligible row, then append a simulated truncated crash line
            part = run_dataset(Model(), None, NATIVE_PROTOCOLS["graphwalks_bfs"], tmp,
                               rows=rows, evaluator=tracked_eval, converter=convert, limit=1)
            self.assertEqual(part["attempted"], 1)
            self.assertEqual(seen_prompts, ["r0"])
            jsonl_path = Path(part["predictions"])
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write('{"row_index": 2, "truncated": ')

            # Step 2: Resume full dataset; r0 is verified & skipped, truncated tail is cleaned, only r2 & r3 run
            seen_prompts.clear()
            full_res = run_dataset(Model(), None, NATIVE_PROTOCOLS["graphwalks_bfs"], tmp,
                                   rows=rows, evaluator=tracked_eval, converter=convert, resume=True)
            self.assertEqual(seen_prompts, ["r2", "r3"])
            self.assertEqual(full_res["attempted"], 3)
            self.assertEqual(full_res["scored"], 3)
            self.assertEqual(full_res["resumed_rows"], 1)
            self.assertTrue(full_res["complete"])
            self.assertEqual(full_res["score"], 100.0)
            saved_lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["row_index"] for r in saved_lines], [0, 2, 3])

            # Step 3: Calling resume=True again on an already-complete suite returns immediately with 0 eval calls
            seen_prompts.clear()
            cached_res = run_dataset(Model(), None, NATIVE_PROTOCOLS["graphwalks_bfs"], tmp,
                                     rows=rows, evaluator=tracked_eval, converter=convert, resume=True)
            self.assertEqual(seen_prompts, [])
            self.assertTrue(cached_res["complete"])

            # Step 4: Tampering with input rows when summary is incomplete raises ValueError on sha256 mismatch
            (Path(tmp) / "graphwalks_bfs.summary.json").unlink()
            tampered_rows = [{"problem_type": "bfs", "prompt": "TAMPERED", "answer": ["a0"]}, *rows[1:]]
            with self.assertRaisesRegex(ValueError, "Resume integrity error"):
                run_dataset(Model(), None, NATIVE_PROTOCOLS["graphwalks_bfs"], tmp,
                            rows=tampered_rows, evaluator=tracked_eval, converter=convert, resume=True)


if __name__ == "__main__":
    unittest.main()
