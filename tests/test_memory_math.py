"""Unit test suite for Phase 0 (`docs/roadmap/02_IMPLEMENTATION_PLAN.md` Task 0.8).

Tests both the pure-Python reference path (runs in any lightweight VM) and the `torch.Tensor` path
(runs automatically whenever PyTorch is installed, e.g. in Colab):
  1. Damped pseudoinverse pullback on non-normal matrices (verifies M5 fix vs lstsq / exact identity).
  2. Multi-round null-space orthogonality against general-corpus basis + historical writes.
  3. Exact memory removal (`W + dW1 + dW2 - dW1`) preserving other memories and restoring pre-write outputs.
  4. `CapacityExhaustedError` raised when active basis rank exceeds `max_rank` (verifies M4 fix).
  5. `BackboneFingerprint` and `loop_config` context manager safety.
"""
from __future__ import annotations

import math
import random
import unittest

from sparkx_memory_utils import (
    HAS_TORCH,
    BackboneFingerprint,
    CapacityExhaustedError,
    EpisodicSideMemory,
    MemoryLedger,
    SleepScheduler,
    _py_matmul,
    _py_solve_spd,
    _py_transpose,
    calibrate_esm_gate,
    damped_pseudoinverse_pullback,
    exact_remove_delta,
    generate_self_study_probes,
    loop_config,
    project_into_null_space,
    reconstruct_spark_attention_output,
    solve_null_space_ridge,
    update_orthonormal_basis,
)

if HAS_TORCH:
    import torch


def _rand_matrix(rows: int, cols: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _max_abs(mat: list[list[float]]) -> float:
    return max((abs(v) for row in mat for v in row), default=0.0)


class TestMemoryMathPurePython(unittest.TestCase):
    """Pure-Python verification of all Phase 0 mathematical identities."""

    def test_m5_pseudoinverse_pullback_non_normal(self) -> None:
        # Construct a strongly non-normal matrix W (upper triangular + asymmetric off-diagonals)
        n = 10
        w = _rand_matrix(n, n, seed=42)
        for i in range(n):
            w[i][i] += 3.0
            for j in range(i + 1, n):
                w[i][j] *= 2.5
                w[j][i] *= 0.2
        r_vec = [float(i + 1) * 0.5 for i in range(n)]
        lam = 1e-8

        delta_c = damped_pseudoinverse_pullback(w, r_vec, lam=lam)
        recon = [row[0] for row in _py_matmul(w, [[x] for x in delta_c])]
        recon_err = max(abs(a - b) for a, b in zip(recon, r_vec))
        self.assertLess(recon_err, 1e-5, f"Correct pullback should reconstruct r_vec; got err={recon_err}")

        # Verify equivalence with dual ridge formula (W^T W + lam I)^{-1} W^T r
        w_t = _py_transpose(w)
        wtw = _py_matmul(w_t, w)
        for i in range(n):
            wtw[i][i] += lam
        wtr = _py_matmul(w_t, [[x] for x in r_vec])
        alt_c = [row[0] for row in _py_solve_spd(wtw, wtr)]
        formula_diff = max(abs(a - b) for a, b in zip(delta_c, alt_c))
        self.assertLess(formula_diff, 1e-7)

        # Demonstrate that the old Stage 2 buggy formula W^T (W^T W + lam I)^{-1} r fails on non-normal W
        buggy_inner = _py_solve_spd(wtw, [[x] for x in r_vec])
        buggy_c = [row[0] for row in _py_matmul(w_t, buggy_inner)]
        buggy_recon = [row[0] for row in _py_matmul(w, [[x] for x in buggy_c])]
        buggy_err = max(abs(a - b) for a, b in zip(buggy_recon, r_vec))
        self.assertGreater(buggy_err, 0.1, "Buggy formula should have large error on non-normal W")

    def test_multi_round_null_space_orthogonality(self) -> None:
        r_dim = 24
        w_dim = 16
        # Start with a general-corpus protected basis U_gen of rank 6
        u_gen_raw = _rand_matrix(6, r_dim, seed=100)
        v_basis = update_orthonormal_basis(None, u_gen_raw, max_rank=20)
        self.assertEqual(len(v_basis), r_dim)
        self.assertEqual(len(v_basis[0]), 6)

        # Check V^T V = I_6
        vtv = _py_matmul(_py_transpose(v_basis), v_basis)
        for i in range(6):
            for j in range(6):
                target = 1.0 if i == j else 0.0
                self.assertAlmostEqual(vtv[i][j], target, places=6)

        hist_acts: list[list[list[float]]] = []
        for round_idx in range(3):
            a_batch = _rand_matrix(3, r_dim, seed=200 + round_idx)
            dz_batch = _rand_matrix(3, w_dim, seed=300 + round_idx)
            dw, a_perp, _, rel_err = solve_null_space_ridge(
                a_batch, dz_batch, v_basis=v_basis, alpha_ridge=1e-4
            )
            self.assertLess(rel_err, 1e-2)

            # General-corpus basis must have zero leakage: ||U_gen^T @ dW||_inf < 1e-6
            gen_leak = _max_abs(_py_matmul(u_gen_raw, dw))
            self.assertLess(gen_leak, 1e-6, f"Round {round_idx} leaked into general basis: {gen_leak}")

            # All prior rounds' activations must have zero leakage: ||A_prev @ dW||_inf < 1e-6
            for prev_idx, a_prev in enumerate(hist_acts):
                leak = _max_abs(_py_matmul(a_prev, dw))
                self.assertLess(leak, 1e-6, f"Round {round_idx} overwrote round {prev_idx}: leak={leak}")

            v_basis = update_orthonormal_basis(v_basis, a_perp, max_rank=20)
            hist_acts.append(a_batch)

    def test_exact_memory_removal(self) -> None:
        r_dim = 20
        w_dim = 8
        w0 = _rand_matrix(r_dim, w_dim, seed=11)
        a1 = _rand_matrix(2, r_dim, seed=21)
        dz1 = _rand_matrix(2, w_dim, seed=31)
        a2 = _rand_matrix(2, r_dim, seed=41)
        dz2 = _rand_matrix(2, w_dim, seed=51)

        # Write memory 1
        dw1, a1_perp, _, _ = solve_null_space_ridge(a1, dz1, v_basis=None, alpha_ridge=1e-5)
        v_basis = update_orthonormal_basis(None, a1_perp, max_rank=18)
        w1 = [[w0[i][j] + dw1[i][j] for j in range(w_dim)] for i in range(r_dim)]

        # Write memory 2 in the null space of memory 1
        dw2, a2_perp, _, _ = solve_null_space_ridge(a2, dz2, v_basis=v_basis, alpha_ridge=1e-5)
        w2 = [[w1[i][j] + dw2[i][j] for j in range(w_dim)] for i in range(r_dim)]

        # Output on A2 after both writes
        out_a2_before_remove = _py_matmul(a2, w2)
        # Output on A1 before any write
        out_a1_initial = _py_matmul(a1, w0)

        # Remove memory 1 while keeping memory 2
        w_removed_1 = exact_remove_delta(w2, dw1)
        out_a1_after_remove = _py_matmul(a1, w_removed_1)
        # Because A1 @ dW2 == 0, removing dW1 restores A1's output to its exact initial value A1 @ W0!
        diff_a1 = max(
            abs(out_a1_after_remove[i][j] - out_a1_initial[i][j])
            for i in range(2)
            for j in range(w_dim)
        )
        self.assertLess(diff_a1, 1e-6)

        # And on A2_perp, memory 2's response is preserved
        out_a2p_before = _py_matmul(a2_perp, w2)
        out_a2p_after = _py_matmul(a2_perp, w_removed_1)
        diff_a2p = max(
            abs(out_a2p_after[i][j] - out_a2p_before[i][j])
            for i in range(2)
            for j in range(w_dim)
        )
        self.assertLess(diff_a2p, 1e-6)
        self.assertGreater(_max_abs(out_a2_before_remove), 0.1)

    def test_m4_capacity_exhausted_raises(self) -> None:
        r_dim = 12
        a_batch = _rand_matrix(5, r_dim, seed=77)
        v_basis = update_orthonormal_basis(None, a_batch, max_rank=8)
        self.assertEqual(len(v_basis[0]), 5)
        a_more = _rand_matrix(5, r_dim, seed=88)
        a_more_perp = project_into_null_space(a_more, v_basis)
        with self.assertRaises(CapacityExhaustedError):
            update_orthonormal_basis(v_basis, a_more_perp, max_rank=8)

    def test_backbone_fingerprint_and_loop_config(self) -> None:
        class _DummyModel:
            def __init__(self) -> None:
                self.base = {
                    "layers.0.self_attn.out_proj.weight": [0.1, 0.2, 0.3, -0.4],
                    "layers.3.self_attn.lsa_indexer.w_q_idx.weight": [1.0, 2.0],
                }
                self.controllers = {
                    "4.extra_gain": [0.092],
                }
                self.loop_start = 4
                self.loop_end = 28
                self.loop_passes = 2
                self.mhc_enabled = True
                self.engram_enabled = True
                self.enable_p3 = True
                self.p3_start = 12
                self.p3_end = 24
                self.esm_enabled = False

        dummy = _DummyModel()
        fp = BackboneFingerprint(dummy, include_controllers=True, store_full_clone=True)
        res_ok = fp.verify(dummy)
        self.assertTrue(res_ok["passed"])
        self.assertTrue(res_ok["sha256_match"])
        self.assertEqual(res_ok["max_abs_diff"], 0.0)

        # Mutate lsa_indexer (ignored) -> still passes
        dummy.base["layers.3.self_attn.lsa_indexer.w_q_idx.weight"][0] = 9.9
        self.assertTrue(fp.verify(dummy)["passed"])

        # Perturb base out_proj by 1e-5 -> must fail and identify exact tensor
        dummy.base["layers.0.self_attn.out_proj.weight"][2] += 1e-5
        res_bad = fp.verify(dummy, tol=0.0)
        self.assertFalse(res_bad["passed"])
        self.assertFalse(res_bad["sha256_match"])
        self.assertIn("base.layers.0.self_attn.out_proj.weight", res_bad["changed_tensors"])
        self.assertAlmostEqual(res_bad["max_abs_diff"], 1e-5, places=9)

        # Restore base, perturb controller -> must fail when include_controllers=True
        dummy.base["layers.0.self_attn.out_proj.weight"][2] -= 1e-5
        self.assertTrue(fp.verify(dummy)["passed"])
        dummy.controllers["4.extra_gain"][0] += 5e-4
        res_ctrl = fp.verify(dummy, tol=0.0)
        self.assertFalse(res_ctrl["passed"])
        self.assertIn("controllers.4.extra_gain", res_ctrl["changed_tensors"])

        # Test loop_config context manager restores state even on exception
        try:
            with loop_config(dummy, passes=1, use_mhc=False, use_engram=False, enable_p3=False, use_esm=True):
                self.assertEqual(dummy.loop_passes, 1)
                self.assertFalse(dummy.mhc_enabled)
                self.assertFalse(dummy.engram_enabled)
                self.assertFalse(dummy.enable_p3)
                self.assertTrue(dummy.esm_enabled)
                raise ValueError("simulated error inside trace")
        except ValueError:
            pass
        self.assertEqual(dummy.loop_passes, 2)
        self.assertTrue(dummy.mhc_enabled)
        self.assertTrue(dummy.engram_enabled)
        self.assertTrue(dummy.enable_p3)
        self.assertFalse(dummy.esm_enabled)

    def test_g1_spark_attention_output_reconstruction(self) -> None:
        # Phase 1 Gate G1: verify exact reconstruction out_proj(sigmoid(g_proj(LN h)) * SDPA(q, k, v))
        seq_t, d_attn, w_dim = 6, 16, 12
        sdpa_out = _rand_matrix(seq_t, d_attn, seed=501)
        normed_h = _rand_matrix(seq_t, w_dim, seed=502)
        out_proj_w = _rand_matrix(w_dim, d_attn, seed=503)
        g_proj_w = _rand_matrix(d_attn, w_dim, seed=504)

        recon = reconstruct_spark_attention_output(
            sdpa_out=sdpa_out,
            normed_input=normed_h,
            out_proj_weight=out_proj_w,
            g_proj_weight=g_proj_w,
        )
        # Independently compute step-by-step reference
        g_logits = _py_matmul(normed_h, _py_transpose(g_proj_w))
        gated = [
            [
                sdpa_out[t][d] / (1.0 + math.exp(-g_logits[t][d]))
                for d in range(d_attn)
            ]
            for t in range(seq_t)
        ]
        ref_out = _py_matmul(gated, _py_transpose(out_proj_w))
        max_diff = max(
            abs(recon[t][j] - ref_out[t][j])
            for t in range(seq_t)
            for j in range(w_dim)
        )
        self.assertLess(max_diff, 1e-12)

    def test_g3_episodic_side_memory_write_gate_and_remove(self) -> None:
        # Phase 3 Gate G3: empty bit-identity, protected general+past basis, canary gate calibration, exact removal
        esm = EpisodicSideMemory(width=16, r=24, route_dim=8, max_shards=4, gate_T=0.02)
        h_probe = _rand_matrix(3, 16, seed=601)
        # 1. Empty ESM -> exact 0.0
        zero_out = esm(h_probe)
        self.assertEqual(_max_abs(zero_out), 0.0)

        # 2. Install general-corpus basis U_gen of rank 4
        h_gen = _rand_matrix(4, 16, seed=602)
        phi_gen = esm.features(h_gen, shard=0)
        u_gen = update_orthonormal_basis(None, phi_gen, max_rank=20)
        esm.set_general_basis(u_gen)

        # 3. Write episodic memory 1 into shard 0
        phi_1 = esm.features(h_probe, shard=0)
        delta_1 = _rand_matrix(3, 16, seed=603)
        c_1 = esm.route_embed([h_probe[0]])
        rep1 = esm.write(shard=0, A=phi_1, Delta=delta_1, alpha=1e-5, centroids=c_1)
        self.assertEqual(rep1.shard_id, 0)
        self.assertLess(rep1.rel_fit_error, 1e-2)
        self.assertLess(rep1.max_prot_leakage, 1e-6)

        # Verify general-corpus states experience zero perturbation even if gate is bypassed (use_gate=False)
        gen_pert = esm(h_gen, use_gate=False)
        self.assertLess(_max_abs(gen_pert), 1e-6)

        # 4. Calibrate gate on 50 random canary states -> open_rate <= 0.05
        canary_h = _rand_matrix(50, 16, seed=604)
        cal = calibrate_esm_gate(esm, canary_h, shard=0, max_open_rate=0.02)
        self.assertLessEqual(cal["open_rate"], 0.02)

        # 5. Exact removal restores exact 0.0 output
        esm.remove(rep1.delta_ref)
        self.assertEqual(_max_abs(esm(h_probe, use_gate=False)), 0.0)

    def test_g4_self_study_probes(self) -> None:
        span = list(range(100, 148))
        pset = generate_self_study_probes(span, n_questions=6, window_tokens=8, stride_tokens=4)
        self.assertGreaterEqual(len(pset.probe_ids_list), 3)
        self.assertGreaterEqual(len(pset.holdout_ids_list), 1)

    def test_g5_sleep_scheduler_ledger_and_forget(self) -> None:
        # Phase 5 Gate G5: streaming tokens trigger sleep when exceeding kv_budget_tokens, and forget() restores exact state
        esm_dict = {3: EpisodicSideMemory(width=12, r=20, route_dim=8), 7: EpisodicSideMemory(width=12, r=20, route_dim=8)}
        ledger = MemoryLedger()
        sched = SleepScheduler(esm_modules=esm_dict, ledger=ledger, kv_budget_tokens=32, span_tokens=16)

        # Feed 48 tokens (3 spans of 16 tokens > budget of 32)
        sched.on_tokens(list(range(10, 58)), surprise=1.5, attention_mass=1.2)
        self.assertEqual(sched.buffer.active_kv_tokens(), 48)

        rep = sched.maybe_sleep()
        self.assertIsNotNone(rep)
        assert rep is not None
        self.assertLessEqual(rep.active_kv_tokens_after, 32)
        self.assertGreaterEqual(rep.spans_compiled, 1)
        self.assertEqual(len(rep.memory_ids), rep.spans_compiled)

        # Forget all compiled memories -> ESM returns to exact 0.0
        h_test = _rand_matrix(2, 12, seed=701)
        self.assertGreater(_max_abs(esm_dict[3](h_test, use_gate=False)), 1e-3)
        for mid in rep.memory_ids:
            sched.forget(mid)
        self.assertLess(_max_abs(esm_dict[3](h_test, use_gate=False)), 1e-7)
        self.assertLess(_max_abs(esm_dict[7](h_test, use_gate=False)), 1e-7)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed in this environment")
class TestMemoryMathTorch(unittest.TestCase):
    """PyTorch tensor verification of all Phase 0 mathematical identities (runs on Colab CPU/GPU)."""

    def test_torch_pseudoinverse_vs_lstsq(self) -> None:
        torch.manual_seed(42)
        w = torch.randn(32, 32, dtype=torch.float32) + 2.5 * torch.triu(torch.ones(32, 32), diagonal=1)
        r = torch.randn(32, dtype=torch.float32)
        delta_c = damped_pseudoinverse_pullback(w, r, lam=1e-6)
        lstsq_c = torch.linalg.lstsq(w, r.unsqueeze(-1)).solution.squeeze(-1)
        self.assertLess(float((delta_c - lstsq_c).norm().item() / lstsq_c.norm().item()), 1e-4)

    def test_torch_null_space_and_exact_removal(self) -> None:
        torch.manual_seed(7)
        r_dim, w_dim = 64, 32
        w0 = torch.randn(r_dim, w_dim, dtype=torch.float32) * 0.02
        u_gen = torch.randn(8, r_dim, dtype=torch.float32)
        v_basis = update_orthonormal_basis(None, u_gen, max_rank=48)

        a1 = torch.randn(4, r_dim, dtype=torch.float32)
        dz1 = torch.randn(4, w_dim, dtype=torch.float32)
        dw1, a1_perp, _, err1 = solve_null_space_ridge(a1, dz1, v_basis=v_basis, alpha_ridge=1e-4)
        self.assertLess(err1, 1e-2)
        self.assertLess(float((u_gen @ dw1).abs().max().item()), 1e-5)
        v_basis = update_orthonormal_basis(v_basis, a1_perp, max_rank=48)

        a2 = torch.randn(4, r_dim, dtype=torch.float32)
        dz2 = torch.randn(4, w_dim, dtype=torch.float32)
        dw2, a2_perp, _, err2 = solve_null_space_ridge(a2, dz2, v_basis=v_basis, alpha_ridge=1e-4)
        self.assertLess(err2, 1e-2)
        self.assertLess(float((a1 @ dw2).abs().max().item()), 1e-5)

        w_after_2 = w0 + dw1 + dw2
        w_removed_1 = exact_remove_delta(w_after_2, dw1)
        self.assertLess(float(((a1 @ w_removed_1) - (a1 @ w0)).abs().max().item()), 1e-5)

        with self.assertRaises(CapacityExhaustedError):
            update_orthonormal_basis(v_basis, a2_perp, max_rank=12)


if __name__ == "__main__":
    unittest.main()
