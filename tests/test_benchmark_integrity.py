"""Protocol integrity tests; no model weights, network, torch or CUDA required."""
import contextlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sparkx_loop72_dlsa_benchmarks as bench


def longbench_row(context="Evidence remains in the middle."):
    return {"_id": "row1", "context": context, "question": "Which answer?",
            "choice_A": "one", "choice_B": "two", "choice_C": "three", "choice_D": "four", "answer": "B"}


def mrcr_row():
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}:" + "x" * 5000} for i in range(12)]
    messages.append({"role": "user", "content": "Repeat the requested response with PREFIX."})
    return {"prompt": json.dumps(messages), "answer": "PREFIX:the complete original response",
            "random_string_to_prepend": "PREFIX:", "desired_msg_index": 5, "n_needles": 8}


class TestBenchmarkProtocolIntegrity(unittest.TestCase):
    def test_mrcr_preserves_every_turn_and_never_uses_gold_pointer(self):
        row = mrcr_row()
        task = bench.convert_hf_row_to_eval_task("Openai_MRCR v2 256k-512k", row, task_mode="official")
        self.assertEqual(task["status"], "ready")
        self.assertEqual(task["messages"], json.loads(row["prompt"]))
        self.assertEqual(task["gold_answer"], row["answer"])
        row["desired_msg_index"] = 1
        changed = bench.convert_hf_row_to_eval_task("Openai_MRCR v2 256k-512k", row, task_mode="official")
        self.assertEqual(task["messages"], changed["messages"])
        self.assertNotIn("prompt", task)

    def test_mrcr_prefix_only_is_not_a_correct_answer(self):
        row = mrcr_row()
        task = bench.convert_hf_row_to_eval_task("Openai_MRCR v2 256k-512k", row, task_mode="official")
        self.assertEqual(bench.score_benchmark_task(task, "PREFIX:")["score"], 0.0)
        self.assertEqual(bench.score_benchmark_task(task, row["answer"])["score"], 100.0)
        self.assertEqual(bench.score_benchmark_task(task, " " + row["answer"])["score"], 0.0)
        ratio = bench.score_benchmark_task(task, "PREFIX:the complete original respons")
        self.assertGreater(ratio["score"], 90.0)
        self.assertLess(ratio["score"], 100.0)

    def test_longbench_keeps_full_context_and_uses_upstream_parser(self):
        row = longbench_row("a" * 25000 + "UNIQUE_MIDDLE_EVIDENCE" + "b" * 25000)
        task = bench.convert_hf_row_to_eval_task("LongBench v2", row, task_mode="official")
        self.assertIn(row["context"], task["prompt"])
        self.assertFalse(task["prompt_truncated"])
        self.assertEqual(bench.score_benchmark_task(task, "The correct answer is (B)")["score"], 100.0)
        self.assertEqual(bench.score_benchmark_task(task, "\\boxed{B}")["score"], 0.0)
        self.assertEqual(bench.score_benchmark_task(task, "The correct answer is (A)")["score"], 0.0)

    def test_document_template_tokens_are_not_rewritten_recursively(self):
        task = bench.convert_hf_row_to_eval_task("LongBench v2", longbench_row("The document says $Q$ and $C_A$."))
        self.assertIn("The document says $Q$ and $C_A$.", task["prompt"])

    def test_graphwalks_ten_character_nodes_and_empty_answers(self):
        task = bench.convert_hf_row_to_eval_task("Graphwalks bfs 256k f1", {
            "prompt": "x" * 60000, "answer_nodes": ["1679091c5a"], "problem_type": "bfs"}, task_mode="official")
        self.assertEqual(len(task["prompt"]), 60000)
        score = bench.score_benchmark_task(task, "Final Answer: [1679091c5a]")
        self.assertEqual(score["score"], 100.0)
        self.assertEqual(bench.score_benchmark_task(task, "Final Answer: [1679091c5a,ffffffff00]")["raw_score"], 2 / 3)
        empty = bench.convert_hf_row_to_eval_task("Graphwalks bfs 256k f1", {
            "prompt": "No reachable nodes.", "answer_nodes": [], "problem_type": "bfs"}, task_mode="official")
        self.assertEqual(empty["status"], "ready")
        self.assertEqual(bench.score_benchmark_task(empty, "Final Answer: []")["score"], 100.0)

    def test_graphwalks_uses_only_the_literal_final_line_and_requires_valid_empty_syntax(self):
        scored = bench.score_graphwalks_bfs_f1(
            "I visited deadbeef00 while reasoning.\nFinal Answer: [1679091c5a]", ["1679091c5a"])
        self.assertEqual(scored["f1"], 1.0)
        self.assertEqual(scored["predicted"], ["1679091c5a"])
        self.assertFalse(scored["failed_to_parse"])
        valid_empty = bench.score_graphwalks_bfs_f1("Visited deadbeef00.\nFinal Answer: []", [])
        self.assertEqual(valid_empty["f1"], 1.0)
        self.assertFalse(valid_empty["failed_to_parse"])
        for malformed in ("[]", "\\boxed{}", "Final Answer:", "Final Answer: []\n"):
            with self.subTest(response=malformed):
                failure = bench.score_graphwalks_bfs_f1(malformed, [])
                self.assertTrue(failure["failed_to_parse"])
                self.assertEqual(failure["f1"], 0.0)
        wrong_case = bench.score_graphwalks_bfs_f1("Final Answer: [1679091C5A]", ["1679091c5a"])
        self.assertEqual(wrong_case["f1"], 0.0)

    def test_graphwalks_parents_cannot_be_labelled_bfs(self):
        task = bench.convert_hf_row_to_eval_task("Graphwalks bfs 256k f1", {
            "prompt": "Find parents", "answer_nodes": [], "problem_type": "parents"}, task_mode="official")
        self.assertEqual(task["status"], "filtered")
        self.assertIsNone(bench.score_benchmark_task(task, "anything")["score"])

    def test_no_gold_answer_or_claim_in_model_prompt(self):
        for name, row in (
            ("AA-LCRv1.1", {"question": "What happened?", "answer": "SECRET_GOLD_ANSWER"}),
            ("MCP-Atlas", {"PROMPT": "Find an event", "GTFA_CLAIMS": ["SECRET_GOLD_CLAIM"]}),
        ):
            task = bench.convert_hf_row_to_eval_task(name, row, task_mode="diagnostic")
            self.assertEqual(task["status"], "unsupported")
            self.assertNotIn("SECRET_GOLD", task.get("prompt", ""))
            self.assertIsNone(task["score"])

    def test_external_suites_are_unsupported_not_fake_successes(self):
        for name in ("AIME 2026", "IFEval", "BFCL-V4", "SciCode", "SWE Verified", "MCP-Atlas", "openai/gdpval"):
            task = bench.convert_hf_row_to_eval_task(name, {"problem": "1+1", "answer": "2"}, task_mode="official")
            self.assertEqual(task["status"], "unsupported", name)
            self.assertIsNone(bench.score_benchmark_task(task, "2")["score"])
        for verifier in ("mcp_claims", "swe_hf_patch", "terminal_hf", "gdpval_hf"):
            scored = bench.score_benchmark_task({"verifier": verifier}, "import os; # gold patch text " * 20)
            self.assertEqual(scored["status"], "unsupported")
            self.assertIsNone(scored["score"])

    def test_missing_scicode_tests_cannot_pass(self):
        verdict = bench.verify_scicode_in_sandbox("No solution", {"scicode_spec": {"test_suite_code": "assert callable(foo)"}})
        self.assertFalse(verdict.ok)
        self.assertIsNone(verdict.detail["score"])
        self.assertEqual(verdict.detail["status"], "unsupported")

    def test_wrong_or_extended_bfcl_function_name_cannot_pass(self):
        spec = {"function_name": "area", "expected_args": {"x": 7}}
        for function in ("wrong_function", "area_extra", "prefix.area"):
            verdict = bench.verify_bfcl_tool_call(json.dumps({"name": function, "arguments": {"x": 7}}), spec)
            self.assertFalse(verdict.ok)
        self.assertTrue(bench.verify_bfcl_tool_call('{"name":"area","arguments":{"x":7}}', spec).ok)
        self.assertFalse(bench.verify_bfcl_tool_call('{"name":"area","arguments":{"x":7,"y":8}}', spec).ok)

    def test_unimplemented_ifeval_rule_has_no_score(self):
        verdict = bench.verify_ifeval_constraints("hello", ["detectable_format:number_highlighted_sections"], [{"num_highlights": 3}])
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.detail["status"], "unsupported")
        self.assertIsNone(verdict.detail["score"])

    def test_ambiguous_mcq_and_answer_mentions_do_not_pass(self):
        self.assertFalse(bench.verify_mcq_choice("\\boxed{A/B}", "A").ok)
        self.assertFalse(bench.verify_mcq_choice("The gold word is mentioned but Answer: A", "B", "gold word").ok)

    def test_math_zero_survives_conversion(self):
        task = bench.convert_hf_row_to_eval_task("AIME 2026", {"problem": "Subtract 1 from 1", "answer": 0})
        self.assertEqual(task["gold"], "0")
        self.assertEqual(task["benchmark"], "Diagnostic / AIME 2026")

    def test_smoke_names_and_scores_cannot_fill_published_comparison(self):
        self.assertTrue(all(t["benchmark"].startswith("Smoke / ") for t in bench.SIGNIFICANT_BENCHMARK_SUITE))
        self.assertTrue(all(t["official_spark_x26_score"] is None for t in bench.SIGNIFICANT_BENCHMARK_SUITE))
        table = bench.format_official_comparison_markdown({"tasks": [{"benchmark": "AIME 2026", "score": 100.0}]})
        row = next(line for line in table.splitlines() if "**AIME 2026**" in line)
        self.assertTrue(row.endswith("| — |"))
        average = next(line for line in table.splitlines() if "**Average**" in line)
        self.assertTrue(average.endswith("| — |"))

    def test_no_model_is_not_one_million_context_validation(self):
        result = bench.run_full_length_1m_context_benchmark(None, None, max_live_tokens=16384)
        self.assertEqual(result["status"], "not_run")
        self.assertIsNone(result["validated_context_tokens"])
        self.assertIsNone(result["observed_max_live_tokens"])
        self.assertFalse(result["quality_validated"])
        self.assertEqual(len(result["live_runs"]), 0)
        self.assertTrue(all(row["status"] == "not_run" for row in result["horizon_status"]))

    def test_calibration_requires_separate_input_and_is_off_by_default(self):
        import inspect
        self.assertFalse(inspect.signature(bench.evaluate_modern_text_perplexity_and_dlsa).parameters["calibrate_pca"].default)
        with self.assertRaisesRegex(ValueError, "separate calibration"):
            bench.evaluate_modern_text_perplexity_and_dlsa(None, None, calibrate_pca=True)

    def test_sampler_zero_does_not_fetch_and_large_sample_refuses_hidden_cap(self):
        with patch.object(bench, "_fetch_hf_rows_via_rest_api", side_effect=AssertionError("network must not run")):
            self.assertEqual(bench.try_load_hf_benchmark_samples(0), {})
        with self.assertRaises(ValueError):
            bench.try_load_hf_benchmark_samples(21)


class TestBenchmarkScope(unittest.TestCase):
    @staticmethod
    def completed(_model, _tokenizer, item, *_args):
        result = bench._task_result_base(item)
        result.update(status="completed", score=100.0, raw_score=1.0, passed=True, metric="accuracy")
        return result

    def test_none_consumes_all_supplied_rows_without_smoke(self):
        rows = (longbench_row(str(i)) for i in range(3))
        with patch.object(bench, "evaluate_benchmark_task", side_effect=self.completed) as evaluate:
            result = bench.evaluate_selected_benchmark_tasks(None, None, hf_samples={"LongBench v2": rows}, max_hf_per_benchmark=None)
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(result["total_tasks"], 3)
        self.assertTrue(result["all_supplied_rows_completed"])
        self.assertFalse(result["full_suite_complete"])
        self.assertIsNone(result["mean_score"])
        self.assertIsNone(result["pass_rate_pct"])

    def test_capped_or_empty_rows_never_report_complete(self):
        with patch.object(bench, "evaluate_benchmark_task", side_effect=self.completed):
            capped = bench.evaluate_selected_benchmark_tasks(None, None, hf_samples={"LongBench v2": [longbench_row()] * 3}, max_hf_per_benchmark=1)
            empty = bench.evaluate_selected_benchmark_tasks(None, None, hf_samples={"LongBench v2": []})
            zero = bench.evaluate_selected_benchmark_tasks(None, None, hf_samples={"LongBench v2": [longbench_row()]}, max_hf_per_benchmark=0)
        self.assertEqual(capped["total_tasks"], 1)
        self.assertFalse(capped["all_supplied_rows_completed"])
        for result in (empty, zero):
            self.assertFalse(result["all_supplied_rows_completed"])
            self.assertFalse(result["full_suite_complete"])
            self.assertEqual(result["status"], "blocked")

    def test_official_mode_never_falls_back_to_unpinned_sampler(self):
        with patch.object(bench, "try_load_hf_benchmark_samples", side_effect=AssertionError("must not sample")):
            result = bench.evaluate_selected_benchmark_tasks(None, None)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["tasks"], [])


class FakeTensor:
    def __init__(self, values):
        self.values = [list(row) for row in values]
        self.shape = (len(self.values), len(self.values[0]) if self.values else 0)

    def __getitem__(self, index):
        rows, cols = index
        if isinstance(rows, int):
            return SimpleNamespace(tolist=lambda: self.values[rows][cols])
        return FakeTensor([row[cols] for row in self.values[rows]])

    def contiguous(self):
        return self

    def view(self, *_args):
        return self

    def numel(self):
        return sum(map(len, self.values))


class FakeTokenizer:
    all_special_tokens = ["<think>", "</think>", "<|im_end|>"]

    def __init__(self, open_think=False):
        self.messages = None
        self.open_think = open_think

    def encode(self, text, **_kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        text = "".join(chr(i) for i in ids)
        if skip_special_tokens:
            for token in self.all_special_tokens:
                text = text.replace(token, "")
        return text

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return self.encode(json.dumps(messages) + ("<think>" if self.open_think else ""))


class FakeModel:
    def __init__(self, output="The correct answer is (B)", timing=True):
        self.output = output
        self.write_timing = timing
        self.calls = []
        self.base = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
        self.paged_cache_cfg = SimpleNamespace(context_limit_tokens=1_000_000)
        self.total_virtual_layers = 72
        self.last_generation_timing = {"stale": True}

    def make_inference_cache(self, **_kwargs):
        return SimpleNamespace(memory_ledger=lambda: {"m1_gpu_total_mib": 1.0, "m2_host_cold_pages_mib": 2.0})

    def generate_cached(self, inp, max_new_tokens, force_think_close=True, **kwargs):
        self.calls.append({**kwargs, "force_think_close": force_think_close, "max_new_tokens": max_new_tokens})
        output = [ord(c) for c in self.output][:max_new_tokens]
        if self.write_timing:
            self.last_generation_timing = {"generated_tokens": len(output), "finish_reason": "length" if len(output) == max_new_tokens else "stop",
                "ttft_ms": 3.0, "prefill_tok_per_s": 100.0, "decode_tok_per_s": 34.0,
                "e2e_tok_per_s": 20.0, "itl_p50_ms": 29.0, "itl_p95_ms": 32.0, "itl_p99_ms": 34.0}
        return FakeTensor([inp.values[0] + output])


class TestGenerationProtocolWithMocks(unittest.TestCase):
    def setUp(self):
        fake_torch = SimpleNamespace(tensor=lambda values, **kwargs: FakeTensor(values), long="long", manual_seed=lambda seed: None)
        self.patch_torch = patch.object(bench, "torch", fake_torch)
        self.patch_has_torch = patch.object(bench, "HAS_TORCH", True)
        self.patch_torch.start()
        self.patch_has_torch.start()
        self.addCleanup(self.patch_torch.stop)
        self.addCleanup(self.patch_has_torch.stop)

    def test_raw_mrcr_messages_and_single_budgeted_generation(self):
        row = mrcr_row()
        task = bench.convert_hf_row_to_eval_task("Openai_MRCR v2 256k-512k", row, task_mode="official")
        model, tokenizer = FakeModel(row["answer"]), FakeTokenizer()
        result = bench.evaluate_benchmark_task(model, tokenizer, task, max_new_tokens=100)
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(tokenizer.messages, json.loads(row["prompt"]))
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(model.calls[0]["force_think_close"])
        self.assertEqual(model.calls[0]["repetition_penalty"], 1.0)
        self.assertEqual(model.calls[0]["max_new_tokens"], 100)
        self.assertEqual(result["generated_tokens"], len(row["answer"]))

    def test_full_runner_uses_actual_converter_and_evaluator_interfaces(self):
        from sparkx_full_benchmark import NATIVE_PROTOCOLS, run_dataset
        model, tokenizer = FakeModel(), FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            result = run_dataset(model, tokenizer, NATIVE_PROTOCOLS["longbench_v2"], directory,
                                 rows=[longbench_row()], max_new_tokens=64)
            with open(result["predictions"], encoding="utf-8") as handle:
                prediction = json.loads(handle.readline())
        self.assertTrue(result["complete"])
        self.assertFalse(result["model_quality_run"])
        self.assertFalse(result["published_score_comparable"])
        self.assertEqual(prediction["status"], "completed")
        self.assertEqual(prediction["score"], 100.0)
        self.assertEqual(prediction["protocol"], "longbench_v2_0shot")
        self.assertEqual(len(model.calls), 1)

    def test_template_failure_is_explicit_and_does_not_fall_back(self):
        model, tokenizer = FakeModel(), FakeTokenizer()
        tokenizer.apply_chat_template = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad template"))
        task = bench.convert_hf_row_to_eval_task("LongBench v2", longbench_row(), task_mode="official")
        result = bench.evaluate_benchmark_task(model, tokenizer, task)
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("chat_template_error", result["reason"])
        self.assertIsNone(result["score"])
        self.assertEqual(model.calls, [])

    def test_unclosed_thinking_is_not_scored_as_a_final_answer(self):
        task = bench.convert_hf_row_to_eval_task("LongBench v2", longbench_row(), task_mode="official")
        model, tokenizer = FakeModel("The correct answer is (B)"), FakeTokenizer(open_think=True)
        result = bench.evaluate_benchmark_task(model, tokenizer, task, max_new_tokens=32)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["response"], "")
        self.assertTrue(result["output_incomplete"])
        self.assertEqual(len(model.calls), 1)

    def test_final_answer_is_separated_from_reasoning_and_eos(self):
        row = mrcr_row()
        task = bench.convert_hf_row_to_eval_task("Openai_MRCR v2 256k-512k", row, task_mode="official")
        model = FakeModel("Reasoning</think>" + row["answer"] + "<|im_end|>")
        result = bench.evaluate_benchmark_task(model, FakeTokenizer(open_think=True), task, max_new_tokens=100)
        self.assertEqual(result["response"], row["answer"])
        self.assertEqual(result["reasoning"], "Reasoning")
        self.assertEqual(result["score"], 100.0)

    def test_context_overflow_never_truncates_input(self):
        task = bench.convert_hf_row_to_eval_task("LongBench v2", longbench_row(), task_mode="official")
        model = FakeModel()
        model.paged_cache_cfg.context_limit_tokens = 10
        result = bench.evaluate_benchmark_task(model, FakeTokenizer(), task, max_new_tokens=100)
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["score"])
        self.assertFalse(result["prompt_truncated"])
        self.assertEqual(model.calls, [])

    def test_official_forcing_override_is_rejected(self):
        task = bench.convert_hf_row_to_eval_task("LongBench v2", longbench_row(), task_mode="official")
        result = bench.evaluate_benchmark_task(FakeModel(), FakeTokenizer(), task, generation_options={"force_think_close": True})
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["score"])

    def test_generation_errors_are_not_zero_scores(self):
        model = FakeModel()

        def broken(inp, max_new_tokens, force_think_close=True, **kwargs):
            raise RuntimeError("out of memory")

        model.generate_cached = broken
        task = bench.convert_hf_row_to_eval_task("LongBench v2", longbench_row(), task_mode="official")
        result = bench.evaluate_benchmark_task(model, FakeTokenizer(), task)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["score"])

    def test_speed_warmups_repeats_and_actual_output_counts(self):
        model = FakeModel("x" * 1000)
        report = bench.benchmark_loop72_decode_speed(model, FakeTokenizer(), prompt_lengths=(32, 64), gen_tokens=512, warmups=1, repeats=3)
        self.assertEqual(len(model.calls), 8)
        self.assertEqual(report["status"], "completed")
        for row in report["horizons"]:
            self.assertEqual(len(row["trials"]), 3)
            self.assertEqual(row["warmups_completed"], 1)
            self.assertEqual(row["generated_tokens"], 512)
            self.assertTrue(row["sustained_512_token_decode_measured"])
            self.assertEqual(row["decode_tok_per_s"], 34.0)

    def test_speed_early_eos_is_not_a_sustained_512_token_run(self):
        report = bench.benchmark_loop72_decode_speed(FakeModel("stop"), FakeTokenizer(), (32,), gen_tokens=512, warmups=0, repeats=1)
        row = report["horizons"][0]
        self.assertEqual(row["generated_tokens"], 4)
        self.assertFalse(row["sustained_512_token_decode_measured"])
        self.assertFalse(row["requested_output_length_reached_in_all_trials"])

    def test_missing_speed_timing_is_error_instead_of_zero_measurements(self):
        report = bench.benchmark_loop72_decode_speed(FakeModel(timing=False), FakeTokenizer(), (32,), gen_tokens=32, warmups=0, repeats=1)
        row = report["horizons"][0]
        self.assertEqual(row["status"], "error")
        self.assertNotIn("decode_tok_per_s", row)

    def test_perplexity_restores_modes_and_threshold_after_sparse_failure(self):
        model = FakeModel()
        config = SimpleNamespace(search_mode="original_mode")
        model.dlsa_index_bank = SimpleNamespace(visit_indexers={0: SimpleNamespace(cfg=config)})
        model.sparse_attention_cfg = SimpleNamespace(exact_threshold=42)
        model.settings = SimpleNamespace(prefill_chunk_tokens=2)
        model.training = True
        model.eval = lambda: setattr(model, "training", False)
        model.train = lambda state: setattr(model, "training", state)
        model.loop_config = lambda **_kwargs: contextlib.nullcontext()

        class FakeLogits:
            shape = (1, 4, 3)
            def __getitem__(self, _index): return self
            def float(self): return self
            def reshape(self, *_args): return self

        def forward(*_args, **_kwargs):
            if config.search_mode == "flat":
                raise RuntimeError("sparse failure")
            return FakeLogits()

        model.forward_logits = forward
        fake_f = SimpleNamespace(cross_entropy=lambda *_args, **_kwargs: SimpleNamespace(item=lambda: 1.0))
        with patch.object(bench, "F", fake_f), self.assertRaisesRegex(RuntimeError, "sparse failure"):
            bench.evaluate_modern_text_perplexity_and_dlsa(model, FakeTokenizer(), seq_len=4, eval_text="abcd")
        self.assertEqual(config.search_mode, "original_mode")
        self.assertEqual(model.sparse_attention_cfg.exact_threshold, 42)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
