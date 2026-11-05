"""CPU tests for sparkx_bie: first-order attribution must match finite differences of the held-out NLL."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import sparkx_bie as bie
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

V, H = 50, 16

if HAVE_TORCH:
    class FakeEngram(nn.Module):
        def __init__(self, scale=1.0):
            super().__init__()
            self.w = nn.Linear(H, H, bias=False)
            self.scale = scale

        def forward(self, input_ids, dtype, prev_ids=None, hidden_states=None, pass_idx=0, ngram_indices=None):
            return (self.w(hidden_states) * self.scale).to(dtype)

    class FakeModel(nn.Module):
        """Embedding -> [block, anchorA] -> [block, anchorB, anchorA again as 'pass 1'] -> head. Engram writes are additive."""
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = nn.Embedding(V, H)
            self.b1, self.b2 = nn.Linear(H, H), nn.Linear(H, H)
            self.head = nn.Linear(H, V)
            self.a1, self.a2 = FakeEngram(), FakeEngram()
            self.engram_scale = 1.0

        def iter_engram_anchors(self):
            return [(3, self.a1), (7, self.a2)]

        def forward_logits(self, ids):
            h = self.emb(ids)
            h = torch.tanh(self.b1(h))
            s = self.engram_scale
            h = h + s * self.a1(ids, h.dtype, hidden_states=h, pass_idx=0)
            h = torch.tanh(self.b2(h))
            h = h + s * self.a2(ids, h.dtype, hidden_states=h, pass_idx=0)
            h = torch.tanh(self.b1(h))
            h = h + s * self.a1(ids, h.dtype, hidden_states=h, pass_idx=1)
            return self.head(h)


def nll(model, ids):
    with torch.no_grad():
        lg = model.forward_logits(ids)
    return float(F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum"))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestBIE(unittest.TestCase):
    def test_attribution_matches_finite_difference(self):
        m = FakeModel()
        ids = torch.randint(0, V, (1, 24), generator=torch.Generator().manual_seed(1))
        r = bie.engram_attribution_one(m, ids)
        self.assertEqual(r["n_tokens"], 23)
        self.assertEqual(sorted((c["layer"], c["pass"]) for c in r["calls"]), [(3, 0), (3, 1), (7, 0)])
        total_attr = sum(c["attr_sum"] for c in r["calls"])
        # d NLL / d s at s=1 with all writes scaled together (first-order) = total attribution
        eps = 1e-3
        m.engram_scale = 1 + eps
        up = nll(m, ids)
        m.engram_scale = 1 - eps
        dn = nll(m, ids)
        fd = (up - dn) / (2 * eps)
        self.assertAlmostEqual(total_attr, fd, delta=max(1e-3, 1e-3 * abs(fd)))

    def test_per_call_attribution_matches_single_scaling(self):
        m = FakeModel()
        ids = torch.randint(0, V, (1, 20), generator=torch.Generator().manual_seed(2))
        r = bie.engram_attribution_one(m, ids)
        a2 = [c for c in r["calls"] if c["layer"] == 7][0]["attr_sum"]
        # scale only anchor a2's output
        orig = m.a2.scale
        eps = 1e-3
        m.a2.scale = 1 + eps
        up = nll(m, ids)
        m.a2.scale = 1 - eps
        dn = nll(m, ids)
        m.a2.scale = orig
        self.assertAlmostEqual(a2, (up - dn) / (2 * eps), delta=max(1e-3, 1e-3 * abs(a2)))

    def test_aggregate_and_format(self):
        m = FakeModel()
        g = torch.Generator().manual_seed(3)
        rs = [bie.engram_attribution_one(m, torch.randint(0, V, (1, 18), generator=g)) for _ in range(3)]
        agg = bie.aggregate_attribution(rs)
        self.assertEqual(agg["n_tokens"], 3 * 17)
        self.assertEqual(len(agg["rows"]), 3)
        self.assertAlmostEqual(agg["total_first_order_dce"], sum(agg["by_layer"].values()))
        txt = bie.format_attribution(agg)
        self.assertIn("TOTAL", txt)
        for row in agg["rows"]:
            self.assertGreater(row["rel_write_size"], 0)
            self.assertTrue(0 <= row["frac_tokens_hurt"] <= 1)
            self.assertTrue(-1.0001 <= row["cos_with_descent"] <= 1.0001)

    def test_hooks_removed_after_call(self):
        m = FakeModel()
        bie.engram_attribution_one(m, torch.randint(0, V, (1, 10)))
        self.assertEqual(len(m.a1._forward_hooks), 0)
        self.assertEqual(len(m.a2._forward_hooks), 0)

    def test_interpret(self):
        rep = {"engram_delta_ce": 0.05, "pass_filter_ce": {"all_passes(trained)": 1.6, "pass0_only": 1.55, "pass0+1": 1.58},
               "attribution": {"by_layer": {3: -0.01, 7: 0.02}, "rows": [
                   {"layer": 3, "pass": 0, "attr_nats_per_token": -0.01, "cos_with_descent": 0.1},
                   {"layer": 7, "pass": 1, "attr_nats_per_token": 0.02, "cos_with_descent": 0.0}]}}
        v = bie.interpret_bie(rep)
        self.assertEqual(v["engram_net"], "hurts")
        self.assertEqual(v["first_order_harmful_layers"], [7])
        self.assertEqual(v["first_order_helpful_layers"], [3])
        self.assertEqual(v["best_pass_filter"], "pass0_only")
        self.assertAlmostEqual(v["pass_filter_gain_vs_trained"], 0.05)
        self.assertEqual(v["misaligned_calls(cos<0.02)"], [(7, 1)])

    def test_zipfian_moe_engram_selective_gating_scope_and_qk(self):
        from sparkx_loop_v03_core import ZipfianTieredMoEEngram

        torch.manual_seed(7)
        eng = ZipfianTieredMoEEngram(
            hot_buckets=64,
            cold_buckets=64,
            eng_dim=32,
            width=64,
            shared_rank=16,
            num_experts=4,
            initial_active_experts=4,
            top_k=2,
            initial_top_k=2,
            expert_rank=32,
            param_dtype=torch.float32,
        )
        nn.init.orthogonal_(eng.project.weight)
        eng.project.weight.data.mul_(0.25)
        nn.init.normal_(eng.shared_down.weight, std=0.1)
        eng.eval()

        ids = torch.randint(0, 50, (1, 8))
        h = torch.randn(1, 8, 64)
        out_default = eng(ids, torch.float32, hidden_states=h, pass_idx=0)
        eng.configure_selective_gating(enabled=False)
        out_off_flag = eng(ids, torch.float32, hidden_states=h, pass_idx=0)
        self.assertTrue(torch.equal(out_default, out_off_flag))

        # Compile 1 cluster at trigger_pos=0 and enable WISE scope gate
        h_trig = torch.randn(64)
        y_target = torch.randn(64) * 0.2
        rep = eng.compile_clusters_alphaedit(
            clusters=[{"query_hidden": h_trig, "target_residual": y_target, "trigger_pos": 0}],
            input_ids=torch.tensor([[10]]),
            alpha_ridge=1e-6,
            target_scale=1.0,
        )
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["scope_prototypes_count"], 1)

        eng.configure_selective_gating(
            enabled=True, scope_threshold=0.75, qk_gate_enabled=False, max_rel_delta_norm=None
        )
        # Out-of-scope general tokens should get exact 0.0 output
        h_gen = torch.randn(1, 16, 64)
        out_gen = eng(torch.randint(0, 50, (1, 16)), torch.float32, hidden_states=h_gen, pass_idx=0)
        self.assertEqual(float(out_gen.abs().max().item()), 0.0)

        # In-scope trigger token should activate and match compiled direction
        out_trig = eng(torch.tensor([[10]]), torch.float32, hidden_states=h_trig.view(1, 1, 64), pass_idx=0)
        self.assertGreater(float(out_trig.norm().item()), 0.01)

        # Trust-region cap test
        eng.configure_selective_gating(
            enabled=True, scope_threshold=0.0, qk_gate_enabled=True, max_rel_delta_norm=0.02
        )
        out_capped = eng(ids, torch.float32, hidden_states=h, pass_idx=0)
        rel_norms = out_capped.norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-6)
        self.assertLessEqual(float(rel_norms.max().item()), 0.02 + 1e-5)

    def test_knowledge_locus_scan_and_gap_write_test(self):
        from sparkx_loop_v03_core import ZipfianTieredMoEEngram

        V_gap = 128
        vocab_map = {}

        class DummyTok:
            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                toks = []
                for w in text.split():
                    if w not in vocab_map:
                        vocab_map[w] = (len(vocab_map) % (V_gap - 2)) + 1
                    toks.append(vocab_map[w])
                toks = toks or [1, 2]
                if return_tensors == "pt":
                    return type("Obj", (), {"input_ids": torch.tensor([toks], dtype=torch.long)})()
                return type("Obj", (), {"input_ids": toks})()

            def decode(self, ids):
                return "tok"

        class ToyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(64, 64)

            def forward(self, h):
                return h + 0.2 * torch.tanh(self.lin(h))

        class ToyLoopWithRealEngram(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(19)
                self.emb = nn.Embedding(V_gap, 64)
                self.layers = nn.ModuleList([ToyBlock(), ToyBlock(), ToyBlock()])
                self.norm = nn.LayerNorm(64)
                self.lm_head = nn.Linear(64, V_gap, bias=False)
                self.eng = ZipfianTieredMoEEngram(
                    hot_buckets=64, cold_buckets=64, eng_dim=32, width=64,
                    shared_rank=16, num_experts=4, initial_active_experts=4,
                    top_k=2, initial_top_k=2, expert_rank=32, param_dtype=torch.float32,
                )
                nn.init.orthogonal_(self.eng.project.weight)
                self.eng.project.weight.data.mul_(0.25)
                nn.init.normal_(self.eng.shared_down.weight, std=0.2)
                self.use_engram = True
                self.engram_pass_filter = None

            def iter_engram_anchors(self):
                return [(1, self.eng)]

            def forward_logits(self, ids):
                h = self.emb(ids)
                h = self.layers[0](h)
                h = self.layers[1](h)
                if self.use_engram and (self.engram_pass_filter is None or 0 in self.engram_pass_filter):
                    h = h + self.eng(ids, h.dtype, hidden_states=h, pass_idx=0)
                h = self.layers[2](h)
                return self.lm_head(self.norm(h))

        m = ToyLoopWithRealEngram()
        tok = DummyTok()
        texts = ["alpha beta gamma delta epsilon zeta", "theta iota kappa lambda mu nu"]

        locus = bie.knowledge_locus_scan(m, tok, texts)
        self.assertEqual(locus["status"], "ok")
        self.assertEqual(len(locus["steps"]), 3)
        self.assertAlmostEqual(locus["steps"][-1]["kl_to_final"], 0.0, places=5)

        gd = bie.gate_design_sweep(m, tok, texts)
        self.assertAlmostEqual(gd["wise_scope_gated"], gd["off"], places=5)

        gap = bie.run_gap_write_test(
            m,
            tok,
            facts=[
                {"prompt": "service ZephyrVault listens on port", "paraphrase": "default port for ZephyrVault is", "target_token_id": 7},
                {"prompt": "Project QuartzHelix assigns salt code", "paraphrase": "QuartzHelix salt code is", "target_token_id": 13},
            ],
            general_texts=texts,
            anchor_layer=1,
            scope_threshold=0.70,
            target_scale=1.5,
        )
        self.assertEqual(gap["status"], "ok")
        self.assertGreater(gap["mean_orig_nll_gain"], 0.0)
        self.assertAlmostEqual(gap["general_locality_delta_ce"], 0.0, places=5)
        self.assertLess(gap["max_null_space_leakage"], 1e-4)

        # Test under small `project.weight` (~0.02), `multi_horizon_recall=True`, and `flight_top_k > active_top_k`
        m.eng.project.weight.data.mul_(0.08)
        m.eng.multi_horizon_recall = True
        m.eng.flight_top_k = 4
        m.eng.active_top_k = 2
        gap_dk = bie.run_gap_write_test(
            m,
            tok,
            facts=[
                {
                    "prompt": "service ZephyrVault listens on port",
                    "probe_alt": "configured ZephyrVault endpoint uses port",
                    "paraphrase": "default ZephyrVault connection uses port",
                    "target_token_id": 7,
                },
                {
                    "prompt": "Project QuartzHelix assigns salt code",
                    "probe_alt": "within Project QuartzHelix verify code",
                    "paraphrase": "QuartzHelix supervisor checks code",
                    "target_token_id": 13,
                },
            ],
            general_texts=texts,
            anchor_layer=1,
            scope_threshold=0.55,
            target_scale=1.0,
        )
        self.assertEqual(gap_dk["status"], "ok")
        self.assertGreater(gap_dk["mean_orig_nll_gain"], 1.0)
        self.assertGreater(gap_dk["mean_para_nll_gain"], 1.0)
        self.assertEqual(gap_dk["top1_hit_rate_after"], 1.0)
        self.assertEqual(gap_dk["para_top1_hit_rate_after"], 1.0)
        self.assertAlmostEqual(gap_dk["general_locality_delta_ce"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

