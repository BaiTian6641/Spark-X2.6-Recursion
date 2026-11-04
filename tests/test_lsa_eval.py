"""CPU tests for sparkx_lsa_eval with a fake model/tokenizer and synthetic scattered-topic q/k (mechanism only)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import sparkx_lsa_eval as le
    from sparkx_dlsa import FaithfulDLSAIndexer
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

HQ, HKV, D, ROPE = 16, 4, 256, 64


class FakeTok:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (ord(c) % 250) for c in text]}


if HAVE_TORCH:
    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Linear(1, 1)
            self.num_key_value_heads, self.head_dim, self.scaling = HKV, D, 1.0 / 16.0

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self, n=6):
            super().__init__()
            self.ls = nn.ModuleList([FakeLayer() for _ in range(n)])
            g = torch.Generator().manual_seed(0)
            self.topics = F.normalize(torch.randn(16, HKV, D - ROPE, generator=g), dim=-1)

        def _layers(self):
            return self.ls

        def capture_attention_io(self, ids, layers, visit=0):
            t = ids.shape[1]
            tt = (ids[0] * 7 + ids[0] // 13) % 16  # scattered topics derived from the tokens
            out = {}
            for l in layers:
                g = torch.Generator().manual_seed(int(ids.sum()) % 1000 + l)
                k = torch.randn(HKV, t, D, generator=g) * 0.1
                k[..., ROPE:] = 3.0 * self.topics[tt].permute(1, 0, 2) + 0.3 * torch.randn(HKV, t, D - ROPE, generator=g)
                q = torch.randn(HQ, t, D, generator=g) * 0.1
                qt = torch.randint(0, 16, (t,), generator=g)
                for h in range(HQ):
                    q[h, :, ROPE:] = 3.0 * self.topics[qt][:, h // (HQ // HKV)] + 0.3 * torch.randn(t, D - ROPE, generator=g)
                out[l] = {"query": q[None], "key": k[None]}
            return out


def fake_texts(n=80, ln=900, seed=0):
    import random
    r = random.Random(seed)
    return ["".join(chr(97 + r.randint(0, 25)) for _ in range(ln)) for _ in range(n)]


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestLsaEval(unittest.TestCase):
    def test_pack_to_length_exact_and_distinct(self):
        ids = le.pack_to_length(FakeTok(), fake_texts(), 5000, seed=1)
        self.assertEqual(tuple(ids.shape), (1, 5000))
        self.assertIsNone(le.pack_to_length(FakeTok(), fake_texts(2), 5000, seed=1))

    def test_sample_positions_range(self):
        pos = le.sample_positions(10000, 10, 3000, 0)
        self.assertTrue(all(5000 <= p <= 9999 for p in pos))
        self.assertEqual(le.sample_positions(100, 5, 3000, 0), [])

    def test_slice_fidelity_is_diagnostic_only(self):
        m = FakeModel()
        cap = m.capture_attention_io(torch.randint(0, 100, (1, 600)), [1])[1]
        r = le.slice_fidelity(cap["query"][0], cap["key"][0], ROPE, [300, 500])
        self.assertGreater(r["r_free_is_tail"], r["r_free_is_head"])

    def test_resolve_rope_first_falls_back_and_says_so(self):
        r = le.resolve_rope_first(FakeModel(), 1, {"head_dim": D, "rope_dim": ROPE})
        self.assertTrue(r["rope_first"])
        self.assertEqual(r["source"], "config_default_unverified")

    def test_probe_rotated_dims_with_fake_rotary_module(self):
        import types
        mod = types.ModuleType("fake_rot_mod")
        def compute_rope_cos_sin(pos, head_dim, theta, prf, device="cpu"):
            rd = int(head_dim * prf)
            f = torch.outer(pos, 1.0 / theta ** (torch.arange(0, rd, 2).float() / rd))
            f = torch.cat([f, f], -1)
            return f.cos(), f.sin()
        def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
            rd = cos.shape[-1]
            cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
            xr, xp = x[..., :rd], x[..., rd:]
            h = xr.shape[-1] // 2
            rot = torch.cat([-xr[..., h:], xr[..., :h]], -1)
            return torch.cat([xr * cos + rot * sin, xp], -1)
        mod.compute_rope_cos_sin, mod.apply_rotary_pos_emb = compute_rope_cos_sin, apply_rotary_pos_emb
        sys.modules["fake_rot_mod"] = mod

        class A(nn.Module):
            __module__ = "fake_rot_mod"
            def __init__(self):
                super().__init__()
                self.d = nn.Linear(1, 1)
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = A()
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.ls = nn.ModuleList([L(), L()])
            def _layers(self):
                return self.ls
        moved = le.probe_rotated_dims(M(), 0, D, ROPE, theta=1e4)
        self.assertEqual(moved, list(range(ROPE)))
        r = le.resolve_rope_first(M(), 0, {"head_dim": D, "rope_dim": ROPE})
        self.assertEqual((r["rope_first"], r["source"]), (True, "probe"))

    def test_oracle_diagnostics_ordering_and_monotone_budget(self):
        m = FakeModel()
        cap = m.capture_attention_io(torch.randint(0, 100, (1, 4096)), [1])[1]
        q, k = cap["query"][0], cap["key"][0]
        d = le.oracle_diagnostics(q, k, [3000, 3500, 4000], budgets=(128, 512), scale=1 / 16)
        for b in (128, 512):
            self.assertEqual(d[b]["n_queries"], 3)
            self.assertGreaterEqual(d[b]["per_head"] + 1e-6, d[b]["per_group"])
            self.assertGreaterEqual(d[b]["per_group"] + 1e-6, d[b]["shared"])
        self.assertGreaterEqual(d[512]["shared"], d[128]["shared"])

    def test_run_f2b_end_to_end_on_fake_model(self):
        m = FakeModel()
        with tempfile.TemporaryDirectory() as td:
            rep = le.run_lsa_f2b(m, FakeTok(), fake_texts(80, 900, 0), fake_texts(80, 900, 1), layers=[1],
                                 variants={"base": {}, "token_buckets": {"n_pool": 1, "avg_bucket_size": 2, "max_bucket_size": 4}},
                                 train_lengths=(4096,), val_length=8192, train_docs_per_length=1, val_docs=1, k_budget=256,
                                 distill_steps=10, n_eval_queries=4, oracle_budgets=(256,), out_dir=td, log=lambda *_: None)
            self.assertEqual(rep["errors"], [], rep["errors"])
            v = rep["layers"]["1"]["variants"]
            self.assertEqual(sorted(v), ["base", "token_buckets"])
            self.assertIsNotNone(v["base"]["distilled"]["heldout_kl"])
            self.assertIn("ORACLE", le.format_f2b(rep))

    def test_capture_qk_shapes_bf16(self):
        m = FakeModel()
        out = le.capture_qk(m, torch.randint(0, 100, (1, 300)), [1, 2])
        self.assertEqual(tuple(out[1]["q"].shape), (HQ, 300, D))
        self.assertEqual(tuple(out[1]["k"].shape), (HKV, 300, D))
        self.assertEqual(out[1]["q"].dtype, torch.bfloat16)

    def test_layer_geometry_defaults(self):
        g = le.layer_geometry(FakeModel(), 1)
        self.assertEqual((g["n_kv"], g["head_dim"], g["rope_dim"]), (HKV, D, ROPE))
        self.assertAlmostEqual(g["scale"], 1 / 16)

    def test_run_f2_end_to_end_on_fake_model_and_upgrade(self):
        m = FakeModel()
        with tempfile.TemporaryDirectory() as td:
            rep = le.run_lsa_f2(m, FakeTok(), fake_texts(80, 900, 0), fake_texts(80, 900, 1), layers=[1, 2],
                                train_lengths=(4096,), val_lengths=(8192,), train_docs_per_length=1, layers_per_pass=2,
                                k_budget=256, distill_steps=15, queries_per_sample=16, n_eval_queries=4, out_dir=td,
                                log=lambda *_: None)
            self.assertEqual(rep["errors"], [], rep["errors"])
            self.assertEqual(sorted(rep["layers"]), ["1", "2"])
            r = rep["layers"]["1"]["val"]["8192"]["distilled"]["recall"]
            self.assertGreater(r["faithful"], r["random"])
            self.assertGreater(r["faithful"], r["contiguous"])
            self.assertTrue(os.path.exists(os.path.join(td, "dlsa_indexers.pt")))
            json.dumps(rep)
            tbl = le.format_table(rep)
            self.assertIn("distilled", tbl)
            g = le.gate_f2(rep)
            self.assertTrue(g["evaluated"])
            self.assertFalse(g["claim_grade"])  # 8K < 64K => never claim-grade
            ixs = le.load_indexers(os.path.join(td, "dlsa_indexers.pt"))
            self.assertEqual(le.upgrade_lsa_indexers(m, ixs, k_budget=256), [1, 2])
            self.assertIsInstance(m._layers()[1].self_attn.lsa_indexer, FaithfulDLSAIndexer)

    def test_gate_logic(self):
        def lay(f, c, fl, r=0.1, o=0.99):
            return {"val": {"65536": {"distilled": {"recall": {"faithful": f, "contiguous": c, "flat_oracle_index": fl,
                                                              "random": r, "oracle": o}}}}}
        rep = {"layers": {"3": lay(0.95, 0.70, 0.96), "7": lay(0.85, 0.5, 0.86), "11": lay(0.95, 0.90, 0.96),
                          "15": lay(0.95, 0.5, 0.99)}}
        g = le.gate_f2(rep)
        self.assertTrue(g["claim_grade"])
        self.assertEqual(g["layers_passed"], [3])
        self.assertEqual(g["layers_failed_use_exact_path"], [7, 11, 15])
        self.assertFalse(g["all_pass"])
        self.assertFalse(le.gate_f2({"layers": {}})["evaluated"])


if __name__ == "__main__":
    unittest.main()
