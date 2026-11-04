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


if __name__ == "__main__":
    unittest.main()
