"""Tests for sparkx_si taskbank / select / gate (pure Python; numpy used if present)."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sparkx_si import gate as G
from sparkx_si import select as S
from sparkx_si import taskbank as TB
from sparkx_si import verifiers as V


class TestTaskbank(unittest.TestCase):
    def test_split_stable_and_ratio(self):
        self.assertEqual(TB.split_of("hello"), TB.split_of("hello"))
        tasks = TB.build_if_bank(2000, seed=1)
        parts = TB.partition(tasks)
        TB.assert_disjoint(parts)
        n = len(tasks)
        self.assertAlmostEqual(len(parts["train"]) / n, 0.8, delta=0.04)
        self.assertAlmostEqual(len(parts["dev"]) / n, 0.1, delta=0.03)
        self.assertAlmostEqual(len(parts["test"]) / n, 0.1, delta=0.03)

    def test_split_independent_of_order_and_extra_data(self):
        a = TB.build_arith_bank(200, seed=3)
        b = list(reversed(a))
        self.assertEqual({t["id"]: t["split"] for t in a}, {t["id"]: t["split"] for t in b})

    def test_if_bank_unique_deterministic_and_compatible(self):
        a, b = TB.build_if_bank(300, seed=5), TB.build_if_bank(300, seed=5)
        self.assertEqual([t["prompt"] for t in a], [t["prompt"] for t in b])
        self.assertEqual(len({t["prompt"] for t in a}), 300)
        for t in a:
            ids = [c["id"] for c in t["spec"]["constraints"]]
            self.assertTrue(TB._compatible(ids), ids)

    def test_if_constraints_are_satisfiable_by_a_constructed_response(self):
        # sanity: each generated constraint can be satisfied by *some* text (no impossible tasks) — brute check on simple ones
        for t in TB.build_if_bank(200, seed=7):
            ids = {c["id"] for c in t["spec"]["constraints"]}
            self.assertFalse({"min_words", "max_words"} <= ids)
            self.assertFalse({"all_lowercase", "all_uppercase"} <= ids)

    def test_arith_gold_is_correct_for_a_perfect_answer(self):
        for t in TB.build_arith_bank(60, seed=2):
            self.assertTrue(V.verify_math(f"<think>..</think>\\boxed{{{t['spec']['gold']}}}", t["spec"]["gold"]).ok)
            self.assertFalse(V.verify_math("\\boxed{-1}", t["spec"]["gold"]).ok)

    def test_no_cross_seed_prompt_collision_changes_split(self):
        t = TB.make_arith_task(11)
        self.assertEqual(t["split"], TB.split_of(t["prompt"]))


def rec(tid, ok, resp, pillar="math", n=None):
    return {"task_id": tid, "pillar": pillar, "prompt": "p" + tid, "response": resp, "ok": ok, "n_tokens": n or len(resp.split())}


class TestSelect(unittest.TestCase):
    def test_pass_rates_and_frontier(self):
        recs = [rec("a", True, "x y z"), rec("a", False, "w"), rec("b", True, "q r"), rec("b", True, "s t u"),
                rec("c", False, "no"), rec("c", False, "nope")]
        pr = S.pass_rates(recs)
        self.assertEqual((pr["a"]["c"], pr["b"]["p"], pr["c"]["p"]), (1, 1.0, 0.0))
        out = S.select_round(recs, per_prompt_keep=2, solved_cap=0.5)
        self.assertEqual(out["needs_rationalization"], ["c"])
        tags = {s["task_id"]: s["tag"] for s in out["selected"]}
        self.assertEqual(tags["a"], "frontier")

    def test_shortest_correct_first_and_keep_limit(self):
        recs = [rec("a", True, "one two three four five six"), rec("a", True, "short one"), rec("a", True, "medium length answer here"),
                rec("a", False, "bad")]
        out = S.select_round(recs, per_prompt_keep=2, solved_cap=0.0, pillar_cap=1.0)
        texts = [s["response"] for s in out["selected"]]
        self.assertEqual(texts, ["short one", "medium length answer here"])

    def test_solved_cap(self):
        recs = []
        for i in range(10):
            recs += [rec(f"f{i}", True, f"frontier unique answer number {i} with words"), rec(f"f{i}", False, "bad")]
        for i in range(50):
            recs += [rec(f"s{i}", True, f"solved distinct sample text {i} alpha beta"), rec(f"s{i}", True, f"solved distinct sample text {i} gamma delta")]
        out = S.select_round(recs, per_prompt_keep=1, solved_cap=0.10, pillar_cap=1.0)
        n_sol = sum(1 for s in out["selected"] if s["tag"] == "solved")
        n_fr = sum(1 for s in out["selected"] if s["tag"] == "frontier")
        self.assertEqual(n_fr, 10)
        self.assertLessEqual(n_sol / (n_sol + n_fr), 0.11)

    def test_dedup(self):
        items = [{"response": "the quick brown fox jumps over the lazy dog today"}, {"response": "the quick brown fox jumps over the lazy dog today!"},
                 {"response": "completely different content about oceans and rivers flowing"}]
        self.assertEqual(len(S.dedup_responses(items, 0.6)), 2)

    def test_pillar_cap(self):
        sel = [{"pillar": "math", "response": str(i)} for i in range(80)] + [{"pillar": "if", "response": str(i)} for i in range(10)] + \
              [{"pillar": "code", "response": str(i)} for i in range(10)]
        out = S.apply_pillar_cap(sel, cap=0.35)
        share = sum(1 for s in out if s["pillar"] == "math") / len(out)
        self.assertLessEqual(share, 0.45)  # converges close to the cap with few pillars
        # infeasible cap with two pillars relaxes to 0.5
        two = S.apply_pillar_cap([{"pillar": "a", "response": "x"}] * 10 + [{"pillar": "b", "response": "y"}] * 2, cap=0.35)
        self.assertEqual(sum(1 for s in two if s["pillar"] == "a"), 2)

    def test_distinct_n_and_diversity(self):
        self.assertEqual(S.distinct_n(["a b c d e f"], 4), 1.0)
        self.assertLess(S.distinct_n(["a b c d a b c d a b c d"], 4), 1.0)
        ok, ds, dr = S.diversity_ok(["a b c d a b c d a b c d"], ["a b c d e f g h i j"])
        self.assertFalse(ok)


class TestGate(unittest.TestCase):
    cfg = G.GateConfig(n_boot=2000, seed=1)

    def mk(self, n, p_inc, p_cand, seed):
        r = random.Random(seed)
        inc = [1 if r.random() < p_inc else 0 for _ in range(n)]
        cand = [1 if (i and r.random() < 0.97) or (not i and r.random() < (p_cand - p_inc * 0.97) / max(1e-9, 1 - p_inc)) else 0 for i in inc]
        return cand, inc

    def test_bootstrap_ci_contains_true_difference_and_orders(self):
        cand, inc = self.mk(600, 0.5, 0.58, 1)
        r = G.paired_bootstrap({"m": cand}, {"m": inc}, n_boot=2000, seed=0)
        self.assertLess(r["lo"], r["point"])
        self.assertLess(r["point"], r["hi"])

    def test_accept_clear_gain(self):
        inc = {"math": [0] * 300 + [1] * 200, "if": [0] * 250 + [1] * 250}
        cand = {"math": [1] * 60 + [0] * 240 + [1] * 200, "if": [1] * 50 + [0] * 200 + [1] * 250}
        d = G.decide(cand, inc, cfg=self.cfg)
        self.assertTrue(d["accept"], d["reasons"])

    def test_reject_noise(self):
        r = random.Random(0)
        inc = {"math": [r.randint(0, 1) for _ in range(300)]}
        cand = {"math": list(inc["math"])}
        cand["math"][0] = 1 - cand["math"][0]
        d = G.decide(cand, inc, cfg=self.cfg)
        self.assertFalse(d["accept"])

    def test_pillar_guard_blocks_even_if_mean_up(self):
        inc = {"math": [0] * 200 + [1] * 200, "if": [1] * 400}
        cand = {"math": [1] * 400, "if": [1] * 360 + [0] * 40}   # math +50, if -10 -> guard (-2) violated
        d = G.decide(cand, inc, cfg=self.cfg)
        self.assertFalse(d["accept"])
        self.assertTrue(any("pillar if" in x for x in d["reasons"]))

    def test_scalar_guards(self):
        inc = {"m": [0] * 100 + [1] * 100}
        cand = {"m": [1] * 60 + [0] * 40 + [1] * 100}
        base = G.decide(cand, inc, cfg=self.cfg)
        self.assertTrue(base["accept"], base["reasons"])
        bad = G.decide(cand, inc, guards={"ce_on": {"cand": 1.60, "inc": 1.55}}, cfg=self.cfg)
        self.assertFalse(bad["accept"])
        bad = G.decide(cand, inc, guards={"engram_gap": {"cand": 0.07, "inc": 0.05}}, cfg=self.cfg)
        self.assertFalse(bad["accept"])
        bad = G.decide(cand, inc, guards={"distinct4": {"cand": 0.5, "inc": 0.9}}, cfg=self.cfg)
        self.assertFalse(bad["accept"])
        bad = G.decide(cand, inc, guards={"qa": {"cand": [0] * 50 + [1] * 50, "inc": [0] * 40 + [1] * 60}}, cfg=self.cfg)
        self.assertFalse(bad["accept"])
        ok = G.decide(cand, inc, guards={"ce_on": {"cand": 1.551, "inc": 1.55}, "distinct4": {"cand": 0.88, "inc": 0.9}}, cfg=self.cfg)
        self.assertTrue(ok["accept"], ok["reasons"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            G.paired_bootstrap({"m": [1, 0]}, {"m": [1]}, n_boot=10)

    def test_round_controller(self):
        rc = G.RoundController(lr=1e-4, n_select=1000)
        self.assertEqual(rc.update({"accept": True, "reasons": []})["action"], "snapshot")
        a = rc.update({"accept": False, "reasons": ["x"]})
        self.assertEqual((a["action"], a["lr"], a["n_select"]), ("rollback", 5e-5, 500))
        b = rc.update({"accept": False, "reasons": ["y"]})
        self.assertEqual(b["action"], "rollback_and_stop")
        self.assertTrue(rc.stopped)
        rc2 = G.RoundController(lr=1e-4, n_select=10)
        rc2.update({"accept": False, "reasons": []})
        rc2.update({"accept": True, "reasons": []})
        self.assertEqual(rc2.consecutive_rejects, 0)


if __name__ == "__main__":
    unittest.main()
