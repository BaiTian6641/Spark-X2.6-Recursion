"""Pure-Python tests for sparkx_long_run (no torch / no network)."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import sparkx_long_run as lr


def fake_rows(prefix, n):
    return [{"u": f"{prefix} question number {i}", "a": f"answer {i}"} for i in range(n)]


class FakeOpen:
    """open_fn double: each (name, shard) is a finite list of rows."""

    def __init__(self, data):
        self.data = data
        self.opened = []

    def __call__(self, name, subset, split, shard_path=None):
        self.opened.append((name, shard_path))
        key = (name, shard_path)
        if key not in self.data:
            raise RuntimeError("no such shard")
        return iter(self.data[key]), "fake"


def extract(row, pillar):
    return {"user": row["u"], "thought": "t", "answer": row["a"]}


class TestHelpers(unittest.TestCase):
    def test_val_split_deterministic_and_disjoint(self):
        recs = [{"user": f"prompt {i}"} for i in range(5000)]
        keys = [lr.doc_key(r) for r in recs]
        assert len(set(keys)) == 5000
        val = [k for k in keys if lr.is_val_doc(k, 25)]
        assert 100 < len(val) < 320  # ~4%
        assert val == [k for k in keys if lr.is_val_doc(k, 25)]
        # whitespace/case-insensitive key
        assert lr.doc_key({"user": "Hello   World"}) == lr.doc_key({"user": "hello world"})

    def test_plan_run_matches_cell4_scale(self):
        p = lr.plan_run(20.0)
        assert 700 <= p["steps"] <= 800
        assert p["tokens_per_step"] == 131072
        assert 70e6 < p["real_tokens"] < 100e6
        assert 35_000 < p["docs_needed"] < 55_000
        assert lr.plan_run(10.0)["steps"] * 2 - lr.plan_run(20.0)["steps"] in (0, 1, -1, 2, -2)

    def test_redistribute_quotas(self):
        ratios = {"a": 0.5, "b": 0.3, "c": 0.2}
        q = lr.redistribute_quotas(1000, ratios, {"a": 1000, "b": 1000, "c": 1000})
        assert q == {"a": 500, "b": 300, "c": 200}
        q = lr.redistribute_quotas(1000, ratios, {"a": 100, "b": 1000, "c": 1000})
        assert q["a"] == 100 and sum(q.values()) == 1000 and q["b"] > 300 and q["c"] > 200
        q = lr.redistribute_quotas(1000, ratios, {"a": 0, "b": 0, "c": 50})
        assert q == {"a": 0, "b": 0, "c": 50}
        assert sum(lr.redistribute_quotas(7, ratios, {"a": 7, "b": 7, "c": 7}).values()) == 7

    def test_next_shard_path(self):
        p = "split_0/train-00000-of-00030.parquet"
        assert lr.next_shard_path(p, 0) == p
        assert lr.next_shard_path(p, 1) == "split_0/train-00001-of-00030.parquet"
        assert lr.next_shard_path(p, 29) == "split_0/train-00029-of-00030.parquet"
        assert lr.next_shard_path(p, 30) is None
        assert lr.next_shard_path("data.jsonl", 1) is None
        assert lr.next_shard_path(None, 1) is None

    def test_lr_schedule_continuity_across_resume(self):
        T = 700
        full = [lr.lr_multiplier(s, T, 20) for s in range(T)]
        assert full[0] < full[19] <= 1.0 + 1e-9
        assert abs(full[20] - 1.0) < 1e-2 and full[-1] >= 0.15 - 1e-9
        # resume at 300: global schedule is the same, only a short re-warm factor is applied
        r = [lr.lr_multiplier(300 + k, T, 20, steps_since_resume=k, rewarm_steps=10) for k in range(30)]
        assert r[0] < 0.2 * full[300]
        assert abs(r[15] - full[315]) < 1e-9  # after the re-warm window it matches the uninterrupted schedule
        assert all(r[i] <= r[i + 1] + 0.02 for i in range(9))

    def test_state_roundtrip_and_unknown_keys(self):
        s = lr.LongRunState(total_steps=700, global_step=55, best_val_ce_on=1.23, baseline={"ce_on": 1.5})
        s2 = lr.LongRunState.from_json(s.to_json())
        assert s2.global_step == 55 and s2.best_val_ce_on == 1.23 and s2.baseline == {"ce_on": 1.5}
        d = json.loads(s.to_json())
        d["future_field"] = 1
        assert lr.LongRunState.from_json(json.dumps(d)).total_steps == 700

    def test_jsonl_gz_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [{"user": "héllo ✓", "thought": "x", "answer": "y", "history_turns": [{"role": "user", "content": "q"}]}]
            p = Path(td) / "sub" / "c.jsonl.gz"
            lr.write_jsonl_gz(p, rows)
            assert lr.read_jsonl_gz(p) == rows

    def test_plateau_policy(self):
        a, best, ni, red = lr.decide_after_eval(1.00, None, 0, 0)
        assert a == "improved" and best == 1.00
        a, best, ni, red = lr.decide_after_eval(1.001, best, ni, red, min_delta=0.002, patience=3, max_reductions=1)
        assert a == "continue" and ni == 1
        a, best, ni, red = lr.decide_after_eval(1.01, best, ni, red, patience=3, max_reductions=1)
        a, best, ni, red = lr.decide_after_eval(1.01, best, ni, red, patience=3, max_reductions=1)
        assert a == "rollback" and red == 1 and ni == 0
        for _ in range(2):
            a, best, ni, red = lr.decide_after_eval(1.02, best, ni, red, patience=3, max_reductions=1)
        a, best, ni, red = lr.decide_after_eval(1.02, best, ni, red, patience=3, max_reductions=1)
        assert a == "stop"


class TestCuration(unittest.TestCase):
    def setUp(self):
        self.ratios = {"p1": 0.5, "p2": 0.5}
        self.sources = {
            "p1": [("dsA", None, "train", "d/train-00000-of-00003.parquet")],
            "p2": [("dsB", None, "train"), ("dsC", None, "train", "x/train-00000-of-00001.parquet")],
        }
        self.data = {
            ("dsA", "d/train-00000-of-00003.parquet"): fake_rows("A0", 300),
            ("dsA", "d/train-00001-of-00003.parquet"): fake_rows("A1", 300),
            ("dsA", "d/train-00002-of-00003.parquet"): fake_rows("A2", 300),
            ("dsB", None): fake_rows("B", 60),
            ("dsC", "x/train-00000-of-00001.parquet"): fake_rows("C", 1000),
        }

    def run_chunk(self, manifest, seen, n, idx):
        return lr.curate_chunk(n, self.ratios, self.sources, manifest, seen, FakeOpen(self.data), extract, lambda r: True,
                               val_mod=25, seed=1, chunk_idx=idx, log=lambda *_: None)

    def test_chunks_are_disjoint_and_exclude_val(self):
        m, seen = lr.new_manifest(), set()
        t0, v0, s0 = self.run_chunk(m, seen, 100, 0)
        t1, v1, s1 = self.run_chunk(m, seen, 100, 1)
        k0 = {lr.doc_key(r) for r in t0}
        k1 = {lr.doc_key(r) for r in t1}
        assert len(t0) == 100 and len(t1) == 100
        assert not (k0 & k1)
        assert not any(lr.is_val_doc(k, 25) for k in k0 | k1)
        assert all(lr.is_val_doc(lr.doc_key(r), 25) for r in v0 + v1)
        assert not ({lr.doc_key(r) for r in v0 + v1} & (k0 | k1))
        assert all(r["pillar"] in ("p1", "p2") and "source" in r for r in t0)

    def test_shard_advance_exhaustion_and_redistribution(self):
        m, seen = lr.new_manifest(), set()
        total = 0
        stats = []
        for i in range(20):
            t, v, s = self.run_chunk(m, seen, 200, i)
            total += len(t)
            stats.append(s)
            if m["exhausted"]:
                break
        # dsA has 3 shards x ~288 non-val docs; dsB 60; dsC 1000 -> finite pool, so it must eventually exhaust
        assert m["exhausted"]
        assert m["sources"][lr._src_key(self.sources["p1"][0])]["done"]
        assert m["sources"][lr._src_key(self.sources["p2"][0])]["done"]
        # shortfall of small dsB was covered by dsC (pillar p2 still delivered its quota in chunk 0)
        assert stats[0]["by_pillar"]["p2"] >= 90
        # every doc exactly once
        assert total == len(seen) - m["val_docs"]

    def test_dead_source_skipped(self):
        data = dict(self.data)
        data[("dsB", None)] = fake_rows("B", 60)
        m, seen = lr.new_manifest(), set()
        o = FakeOpen(data)
        lr.curate_chunk(50, self.ratios, self.sources, m, seen, o, extract, lambda r: False, val_mod=25, chunk_idx=0,
                        log=lambda *_: None)
        # nothing passes the filter -> after two empty chunks the sources are marked dead (no infinite loop)
        lr.curate_chunk(50, self.ratios, self.sources, m, seen, o, extract, lambda r: False, val_mod=25, chunk_idx=1,
                        log=lambda *_: None)
        assert m["exhausted"]


class TestProvider(unittest.TestCase):
    def test_provider_serves_chunks_and_freezes_val(self):
        ratios = {"p1": 1.0}
        sources = {"p1": [("dsA", None, "train", "d/train-00000-of-00004.parquet")]}
        data = {("dsA", f"d/train-{i:05d}-of-00004.parquet"): fake_rows(f"S{i}", 400) for i in range(4)}
        m_ratios = ratios

        def curate(n, manifest, seen, idx):
            return lr.curate_chunk(n, m_ratios, sources, manifest, seen, FakeOpen(data), extract, lambda r: True,
                                   val_mod=10, seed=3, chunk_idx=idx, log=lambda *_: None)

        with tempfile.TemporaryDirectory() as td:
            prov = lr.ChunkProvider(Path(td) / "data", curate, n_chunks_target=3, chunk_docs=100, hub=None, lookahead=3)
            prov.start()
            try:
                v1 = prov.frozen_val(10)
                assert 0 < len(v1) <= 10
                recs0, eff0, w0 = prov.get_train(0, timeout=30)
                recs1, eff1, w1 = prov.get_train(1, timeout=30)
                assert len(recs0) == 100 and not w0 and eff1 == 1
                assert not ({lr.doc_key(r) for r in recs0} & {lr.doc_key(r) for r in recs1})
                assert not ({lr.doc_key(r) for r in v1} & {lr.doc_key(r) for r in recs0 + recs1})
                # frozen val is stable on re-request
                assert [lr.doc_key(r) for r in prov.frozen_val(10)] == [lr.doc_key(r) for r in v1]
            finally:
                prov.close()
                prov.thread.join(timeout=0.5)
            # a fresh provider (new session) continues from the persisted manifest, never re-issuing docs
            prov2 = lr.ChunkProvider(Path(td) / "data", curate, n_chunks_target=3, chunk_docs=100, hub=None)
            assert prov2.n_available() >= 2
            assert len(prov2.seen) >= 200


if __name__ == "__main__":
    unittest.main()
