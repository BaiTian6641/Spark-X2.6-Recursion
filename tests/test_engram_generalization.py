"""Torch-free tests for `sparkx_engram_generalization` (leak-free diagnostic tables + summarizer)."""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

import sparkx_engram_generalization as eg

_BENCH = Path(__file__).resolve().parents[1] / "sparkx_loop_v03_benchmarks.py"


def _bench_source() -> str:
    return _BENCH.read_text(encoding="utf-8")


def _leaked_anchor_texts() -> list:
    tree = ast.parse(_bench_source())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_HELD_OUT_EVAL_TEXTS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError("_HELD_OUT_EVAL_TEXTS not found")


class TestGeneralizationTables(unittest.TestCase):
    def test_paraphrase_ids_exist_in_curriculum(self):
        ids = set(re.findall(r'"id":\s*"([a-z0-9_]+)"', _bench_source()))
        for tid in eg.HELDOUT_PARAPHRASES:
            self.assertIn(tid, ids, tid)

    def test_paraphrase_counts_and_uniqueness(self):
        for tid, paras in eg.HELDOUT_PARAPHRASES.items():
            self.assertGreaterEqual(len(paras), 2, tid)
            self.assertEqual(len(set(paras)), len(paras), tid)

    def test_paraphrases_are_not_the_training_prompt(self):
        src = _bench_source()
        for tid, paras in eg.HELDOUT_PARAPHRASES.items():
            for p in paras:
                self.assertNotIn(p[:60], src.replace("\\\"", '"'), f"{tid} paraphrase copied from source")

    def test_true_heldout_disjoint_from_training_anchors(self):
        leaked = _leaked_anchor_texts()
        for t in eg.TRUE_HELDOUT_TEXTS:
            self.assertNotIn(t, leaked)
            # user turn must differ too
            u = t.split("<|User|>")[1].split("<")[0][:50]
            for l in leaked:
                self.assertNotIn(u, l)

    def test_conditions_cover_four_arms(self):
        self.assertEqual(set(eg.CONDITIONS), {"OFF", "FULL", "SEM_ONLY", "LEX_ONLY"})
        self.assertFalse(eg.CONDITIONS["OFF"]["use_engram"])
        self.assertEqual(eg.CONDITIONS["SEM_ONLY"]["lex"], 0.0)
        self.assertEqual(eg.CONDITIONS["LEX_ONLY"]["sem"], 0.0)


class TestSummarizer(unittest.TestCase):
    def test_overlap_metric(self):
        self.assertAlmostEqual(eg.word_trigram_overlap("a b c d", "a b c d"), 1.0)
        self.assertEqual(eg.word_trigram_overlap("a b c d", "w x y z"), 0.0)
        self.assertEqual(eg.word_trigram_overlap("", "a b c"), 0.0)

    def _table(self, orig_full, para_full):
        return {
            "t": {
                "orig": {"OFF": 2.0, "FULL": 2.0 - orig_full, "SEM_ONLY": 2.0 - orig_full / 2, "LEX_ONLY": 2.0 - orig_full / 2},
                "para": [{"OFF": 2.5, "FULL": 2.5 - para_full, "SEM_ONLY": 2.5 - para_full, "LEX_ONLY": 2.5, "overlap": 0.1}],
            }
        }

    def test_transfer_ratio(self):
        s = eg.summarize_transfer(self._table(orig_full=1.0, para_full=0.25))
        self.assertAlmostEqual(s["mean_transfer"]["FULL"], 0.25)
        self.assertAlmostEqual(s["mean_transfer"]["LEX_ONLY"], 0.0)

    def test_no_gain_is_inconclusive(self):
        s = eg.summarize_transfer(self._table(orig_full=0.0, para_full=0.0))
        self.assertIsNone(s["mean_transfer"]["FULL"])
        self.assertTrue(eg.interpret(s["mean_transfer"]).startswith("INCONCLUSIVE"))

    def test_interpret_buckets(self):
        self.assertIn("LEXICAL", eg.interpret({"FULL": 0.1}))
        self.assertIn("PARTIAL", eg.interpret({"FULL": 0.5}))
        self.assertIn("GOOD", eg.interpret({"FULL": 0.9}))


if __name__ == "__main__":
    unittest.main()
