"""CPU runtime contracts. These tests do not measure Spark quality or GPU speed."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

if torch is not None:
    from sparkx_loop_v03_core import (
        LoopV03Settings,
        SparkXLoopV03,
        load_loop_v03_state_dict,
        loop_v03_state_dict,
    )

    class TinyLayer(torch.nn.Module):
        def __init__(self, index, calls):
            super().__init__()
            self.index = index
            self.calls = calls
            self.contexts = []
            self.layer_type = 'full_attention' if index % 4 == 3 else 'sliding_attention'
            self.mlp = torch.nn.Identity()
            self.self_attn = torch.nn.Identity()

        def forward(self, hidden_states, **kwargs):
            self.calls.append(self.index)
            self.contexts.append(getattr(self.self_attn, '_sparkx_active_visit', None))
            if getattr(self, 'use_attention', False):
                return hidden_states + self.self_attn(hidden_states=hidden_states, **kwargs)[0]
            return hidden_states + (self.index + 1) * 0.001

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.config = types.SimpleNamespace(hidden_size=8, tie_word_embeddings=False)
            self.model = torch.nn.Module()
            self.model.embedding = torch.nn.Embedding(16, 8)
            self.model.layers = torch.nn.ModuleList([TinyLayer(i, self.calls) for i in range(36)])
            self.model.norm = torch.nn.Identity()
            self.lm_head = torch.nn.Linear(8, 16, bias=False)
            self.requires_grad_(False)

    def tiny_loop(recurrence_mode='auto'):
        settings = LoopV03Settings.for_loop72_lsa(
            checkpoint_layers=False, recurrence_mode=recurrence_mode,
        )
        model = SparkXLoopV03(TinyBackbone(), settings, strict=False, patch_sdpa=False)

        def context(self, embeds, attention_mask, past_key_values=None, cache_position=None):
            kinds = ('full_attention', 'sliding_attention')
            return ({k: None for k in kinds}, {k: None for k in kinds}), torch.arange(embeds.shape[1])

        model._forward_context = types.MethodType(context, model)
        return model

    class GenerationStub(torch.nn.Module):
        generate_cached = SparkXLoopV03.generate_cached

        def __init__(self, token_ids):
            super().__init__()
            self.settings = types.SimpleNamespace(use_paged_hybrid_cache=False, prefill_chunk_tokens=0)
            self.token_ids = token_ids
            self.calls = []

        def forward_logits(self, input_ids, **kwargs):
            token = self.token_ids[min(len(self.calls), len(self.token_ids) - 1)]
            self.calls.append(input_ids.clone())
            logits = torch.full((1, 1, 16), -100.0)
            logits[0, 0, token] = 100.0
            return logits


@unittest.skipIf(torch is None, 'torch not installed')
class RuntimeContractTests(unittest.TestCase):
    def test_auto_mode_exposes_existing_train_eval_difference_and_72_visits(self):
        model = tiny_loop()
        controller_calls = []
        handles = [c.register_forward_hook(lambda *args: controller_calls.append(1)) for c in model.controllers.values()]
        try:
            model.eval()
            manifest = model.execution_manifest()
            self.assertEqual(manifest['effective_recurrence_mode'], 'native_gelu_depth_matched_highway')
            self.assertTrue(manifest['recurrence_train_eval_mismatch'])
            self.assertFalse(manifest['mhc_executed'])
            model.forward_hidden(torch.tensor([[6, 7]]))
            self.assertEqual(model.base.calls, model.execution_plan)
            self.assertEqual(len(model.base.calls), 72)
            self.assertEqual(controller_calls, [])
            model.base.calls.clear()
            model.train()
            model.forward_hidden(torch.tensor([[6, 7]]))
            self.assertEqual(model.base.calls, model.execution_plan)
            self.assertEqual(len(controller_calls), 60)
        finally:
            for handle in handles:
                handle.remove()

    def test_explicit_learned_mhc_has_matching_dispatch_and_gain_gradients(self):
        model = tiny_loop('learned_mhc')
        calls = []
        handles = [c.register_forward_hook(lambda *args: calls.append(1)) for c in model.controllers.values()]
        try:
            for training in (True, False):
                model.train(training)
                manifest = model.execution_manifest()
                self.assertEqual(manifest['effective_recurrence_mode'], 'learned_mhc')
                self.assertFalse(manifest['recurrence_train_eval_mismatch'])
                self.assertTrue(manifest['mhc_executed'])
                calls.clear()
                model.base.calls.clear()
                output = model.forward_hidden(torch.tensor([[6, 7]]))
                self.assertEqual(len(calls), 60)
                self.assertEqual(model.base.calls, model.execution_plan)
                if training:
                    output.square().mean().backward()
                    self.assertIsNotNone(model.controllers['4'].extra_gain.grad)
                    self.assertGreater(float(model.controllers['4'].extra_gain.grad.abs()), 0.0)
                    self.assertIsNotNone(model.controllers['12'].p3_extra_gain.grad)
                    self.assertGreater(float(model.controllers['12'].p3_extra_gain.grad.abs()), 0.0)
                    self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        finally:
            for handle in handles:
                handle.remove()

    def test_explicit_highway_has_matching_dispatch(self):
        model = tiny_loop('depth_matched_highway')
        self.assertEqual(model.effective_recurrence_mode(True), model.effective_recurrence_mode(False))
        self.assertFalse(model.execution_manifest()['recurrence_train_eval_mismatch'])

    def test_learned_checkpoint_roundtrip_preserves_mode_and_controller(self):
        original = tiny_loop('learned_mhc')
        with torch.no_grad():
            original.controllers['4'].extra_gain.fill_(0.123)
        payload = loop_v03_state_dict(original)
        restored = tiny_loop()
        load_loop_v03_state_dict(restored, payload, strict=True)
        restored.eval()
        self.assertEqual(restored.settings.recurrence_mode, 'learned_mhc')
        self.assertEqual(restored.effective_recurrence_mode(), 'learned_mhc')
        self.assertTrue(torch.equal(original.controllers['4'].extra_gain, restored.controllers['4'].extra_gain))

    def test_negative_legacy_gain_is_not_convex_residual_mixing(self):
        model = tiny_loop('learned_mhc')
        controller = model.controllers['4']
        with torch.no_grad():
            controller.extra_gain.fill_(-0.5)
        telemetry = controller.controller_telemetry()
        self.assertIn('p2_attn', telemetry['gains_outside_unit_interval'])
        self.assertLess(telemetry['static_effective_mix_min_entry']['p2_attn'], 0.0)

    def test_attached_paged_attention_rejects_silent_detached_training(self):
        from sparkx_sparse_attention import attach_loop72_dlsa_attention
        model = tiny_loop('learned_mhc')
        for layer in model.base.model.layers:
            layer.self_attn = torch.nn.Identity()
        bank = types.SimpleNamespace(parameter_manifest=lambda: {})
        attach_loop72_dlsa_attention(model, index_bank=bank)
        self.assertFalse(model.attention_training_supported)
        attn = model.base.model.layers[3].self_attn
        for training in (True, False):
            with self.subTest(training=training):
                attn.train(training)
                hidden = torch.randn(1, 2, 8, requires_grad=not training)
                with self.assertRaisesRegex(RuntimeError, 'detached inference KV'):
                    attn(hidden, position_embeddings=(None, None))

    def test_uncached_global_selectors_follow_all_18_virtual_visits(self):
        from sparkx_dlsa import DLSAIndexConfig, VisitAwareDLSAIndexBank
        from sparkx_sparse_attention import attach_loop72_dlsa_attention, paged_dlsa_global_attention_forward

        class TinyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head_dim, self.num_heads, self.num_key_value_heads = 4, 2, 1
                self.q_k_v_proj = torch.nn.Linear(8, 16, bias=False)
                self.out_proj = torch.nn.Linear(8, 8, bias=False)
                self.g_proj = torch.nn.Linear(8, 2, bias=False)

            def forward(self, *args, **kwargs):
                raise AssertionError('attachment did not install attention dispatch')

        for recurrence in ('learned_mhc', 'depth_matched_highway'):
            with self.subTest(recurrence=recurrence):
                model = tiny_loop(recurrence)
                for layer in model.base.model.layers:
                    layer.self_attn = TinyAttention()
                    layer.use_attention = True

                def context(self, embeds, attention_mask, past_key_values=None, cache_position=None):
                    kinds = ('full_attention', 'sliding_attention')
                    rope = (torch.empty(embeds.shape[1], 0), torch.empty(embeds.shape[1], 0))
                    return ({k: None for k in kinds}, {k: rope for k in kinds}), torch.arange(embeds.shape[1])

                model._forward_context = types.MethodType(context, model)
                bank = VisitAwareDLSAIndexBank(base_cfg=DLSAIndexConfig(n_kv=1, head_dim=4, rope_dim=0, d_idx=2))
                attach_loop72_dlsa_attention(model, index_bank=bank)
                model.eval()
                with mock.patch('sparkx_sparse_attention.paged_dlsa_global_attention_forward',
                                wraps=paged_dlsa_global_attention_forward) as attended:
                    with torch.no_grad():
                        model.forward_hidden(torch.tensor([[6, 7]]))
                expected = [(row['physical_layer'], row['virtual_visit'], row['layer_visit_ordinal'])
                            for row in bank.global_schedule]
                actual = [(call.kwargs['slot'].physical_layer, call.kwargs['slot'].virtual_visit,
                           call.kwargs['slot'].layer_visit_ordinal) for call in attended.call_args_list]
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), 18)
                for global_id, call in enumerate(attended.call_args_list):
                    self.assertIs(call.kwargs['indexer'], bank.get_visit_indexer(global_id))
                self.assertTrue(all(not hasattr(layer.self_attn, '_sparkx_active_visit') for layer in model.base.model.layers))

    def test_distance_prior_cache_tracks_parameter_updates_and_training(self):
        from sparkx_dlsa import FaithfulDLSAIndexer
        indexer = FaithfulDLSAIndexer(n_kv=1, head_dim=4, rope_dim=0, d_idx=2).eval()
        with torch.no_grad():
            indexer.raw_dist_alpha.fill_(0.5)
        with mock.patch.object(indexer, 'dist_alpha', wraps=indexer.dist_alpha) as compute:
            first = indexer.dist_alpha_over_tau()
            self.assertEqual(indexer.dist_alpha_over_tau(), first)
            self.assertEqual(compute.call_count, 1)
            with torch.no_grad():
                indexer.log_tau.add_(0.5)
            changed = indexer.dist_alpha_over_tau()
            self.assertNotEqual(changed, first)
            self.assertEqual(compute.call_count, 2)
            state = {key: value.clone() for key, value in indexer.state_dict().items()}
            state['raw_dist_alpha'].fill_(1.5)
            indexer.load_state_dict(state)
            self.assertNotEqual(indexer.dist_alpha_over_tau(), changed)
            self.assertEqual(compute.call_count, 3)
            indexer.to(dtype=torch.float64)
            indexer.dist_alpha_over_tau()
            self.assertEqual(compute.call_count, 4)
            indexer.train()
            indexer.dist_alpha_over_tau()
            indexer.dist_alpha_over_tau()
            self.assertEqual(compute.call_count, 6)

    def test_checkpoint_replay_preserves_visit_context_and_controller_gradients(self):
        direct = tiny_loop('learned_mhc')
        replayed = tiny_loop('learned_mhc')
        replayed.load_state_dict(direct.state_dict())
        replayed.enable_checkpoint = True
        ids = torch.tensor([[6, 7]])
        direct.forward_hidden(ids).square().mean().backward()
        replayed.forward_hidden(ids).square().mean().backward()
        for (name_a, parameter_a), (name_b, parameter_b) in zip(
            direct.controllers.named_parameters(), replayed.controllers.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            if parameter_a.grad is not None:
                torch.testing.assert_close(parameter_a.grad, parameter_b.grad)
        counts = {}
        expected = {}
        for virtual, physical in enumerate(replayed.execution_plan):
            ordinal = counts.get(physical, 0)
            counts[physical] = ordinal + 1
            expected.setdefault(physical, set()).add((physical, virtual, ordinal))
        for layer in replayed.base.model.layers:
            self.assertTrue(set(layer.contexts).issubset(expected[layer.index]))
            self.assertFalse(hasattr(layer.self_attn, '_sparkx_active_visit'))

    def test_chunked_logits_keep_requested_tail_across_boundary(self):
        model = types.SimpleNamespace(
            base=types.SimpleNamespace(config=types.SimpleNamespace(tie_word_embeddings=False), lm_head=torch.nn.Identity()),
            forward_hidden=lambda ids, **kwargs: ids.float().unsqueeze(-1),
        )
        ids = torch.arange(10).reshape(1, 10)
        for keep in (None, 0, -1, 1, 3, 7, 10):
            with self.subTest(keep=keep):
                got = SparkXLoopV03.forward_logits(model, ids, past_key_values=object(), chunk_size=4, logits_to_keep=keep)
                expected = ids if keep is None or keep <= 0 else ids[:, -keep:]
                self.assertTrue(torch.equal(got[..., 0], expected.float()))

    def test_mhc_eval_cache_parity_and_invalidation(self):
        from sparkx_loop_v03_core import TwoStreamMHC

        torch.manual_seed(7)
        dim = 16
        ctrl = TwoStreamMHC(width=dim, max_passes=3, pass_adapter_rank=4, gain_mode='sigmoid')
        with torch.no_grad():
            ctrl.dynamic_proj.weight.normal_(0.0, 0.05)
            ctrl.ffn_dynamic_proj.weight.normal_(0.0, 0.05)
            ctrl.pass_routing_bias.normal_(0.0, 0.1)
            for p_idx in range(3):
                ctrl.pass_up[p_idx].weight.normal_(0.0, 0.05)
                ctrl.pass_down[p_idx].weight.normal_(0.0, 0.05)
            ctrl.extra_gain.fill_(0.8)
            ctrl.ffn_extra_gain.fill_(0.3)
            ctrl.p3_extra_gain.fill_(-0.4)
            ctrl.p3_ffn_extra_gain.fill_(-0.2)

        class DummySublayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_type = 'sliding_attention'
                self.input_layernorm = torch.nn.RMSNorm(dim, eps=1e-6)
                self.post_attention_layernorm = torch.nn.RMSNorm(dim, eps=1e-6)
                self.lin1 = torch.nn.Linear(dim, dim, bias=False)
                self.mlp = torch.nn.Linear(dim, dim, bias=False)
                self.mlp.gate_proj = self.mlp
                self.self_attn = lambda **kw: (self.lin1(kw['hidden_states']), None)

        layer = DummySublayer()
        s0 = torch.randn(1, 5, dim)
        s1 = torch.randn(1, 5, dim)

        for p_idx in (0, 1, 2):
            ctrl.train(True)
            train_out = ctrl(torch.stack([s0.clone(), s1.clone()], dim=0), layer, None, None, pass_idx=p_idx)
            ctrl.train(False)
            with torch.no_grad():
                eval_s0, eval_s1 = ctrl((s0.clone(), s1.clone()), layer, None, None, pass_idx=p_idx)
            torch.testing.assert_close(train_out[0], eval_s0, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(train_out[1], eval_s1, atol=1e-5, rtol=1e-5)

        key_before = getattr(ctrl, '_eval_cache_last_key', None)
        self.assertIsNotNone(key_before)
        with torch.no_grad():
            ctrl.extra_gain.add_(0.75)
            mutated_s0, _ = ctrl((s0.clone(), s1.clone()), layer, None, None, pass_idx=1)
        self.assertNotEqual(key_before, getattr(ctrl, '_eval_cache_last_key', None))
        ctrl.train(True)
        train_mut = ctrl(torch.stack([s0.clone(), s1.clone()], dim=0), layer, None, None, pass_idx=1)
        torch.testing.assert_close(mutated_s0, train_mut[0], atol=1e-5, rtol=1e-5)

    def test_paged_visit_slot_gpu_resident_views_and_coalesced_cold_gather(self):
        from sparkx_paged_cache import GlobalPagedVisitSlot, PagedCacheConfig

        torch.manual_seed(11)
        cfg = PagedCacheConfig(
            sinks=4,
            recent_tokens=16,
            kv_page_tokens=8,
            global_hot_tokens_per_visit=32,
            cold_codec='bf16',
        )
        slot = GlobalPagedVisitSlot(
            virtual_visit=0,
            physical_layer=3,
            pass_idx=0,
            global_visit_id=0,
            layer_visit_ordinal=0,
            cfg=cfg,
        )
        k_all = torch.randn(1, 2, 48, 8, dtype=torch.bfloat16)
        v_all = torch.randn(1, 2, 48, 8, dtype=torch.bfloat16)

        slot.append(k_all[:, :, :24, :], v_all[:, :, :24, :])
        self.assertTrue(slot.is_fully_gpu_resident())
        for step in range(24, 32):
            slot.append(k_all[:, :, step:step + 1, :], v_all[:, :, step:step + 1, :])
        self.assertTrue(slot.is_fully_gpu_resident())
        k_res, v_res = slot.gather_selected(torch.tensor([0, 3, 7, 19, 31]), return_positions=False)
        torch.testing.assert_close(k_res, k_all[:, :, [0, 3, 7, 19, 31], :])
        torch.testing.assert_close(v_res, v_all[:, :, [0, 3, 7, 19, 31], :])

        # Append beyond gpu_resident_cap=32 to trigger cold page archival and coalesced gather
        slot.append(k_all[:, :, 32:48, :], v_all[:, :, 32:48, :])
        self.assertFalse(slot.is_fully_gpu_resident())
        sel = torch.tensor([0, 2, 5, 9, 14, 18, 25, 33, 44, 47])
        k_cold, v_cold = slot.gather_selected(sel, return_positions=False)
        torch.testing.assert_close(k_cold, k_all[:, :, sel, :])
        torch.testing.assert_close(v_cold, v_all[:, :, sel, :])

    def test_vectorized_dlsa_sparse_prefill_causality_and_decode_consistency(self):
        from sparkx_dlsa import DLSAIndexConfig, FaithfulDLSAIndexer
        from sparkx_paged_cache import GlobalPagedVisitSlot, PagedCacheConfig
        from sparkx_sparse_attention import SparseAttentionConfig, paged_dlsa_global_attention_forward

        torch.manual_seed(19)
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 8
        rope_dim = 2
        seq_len = 48

        dlsa_cfg = DLSAIndexConfig(
            n_kv=num_kv_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            d_idx=8,
            n_sink=4,
            n_pool=2,
            n_cluster=16,
            avg_bucket_size=4,
            max_bucket_size=8,
            window=16,
            k_long_range=8,
        )
        indexer = FaithfulDLSAIndexer(dlsa_cfg).eval()
        sparse_cfg = SparseAttentionConfig(
            sinks=4,
            recent_tokens=16,
            long_range_tokens=8,
            exact_threshold=24,
            sparse_query_tile_tokens=8,
        )
        cache_cfg = PagedCacheConfig(
            sinks=4,
            recent_tokens=16,
            kv_page_tokens=8,
            global_hot_tokens_per_visit=64,
        )

        q = torch.randn(1, num_q_heads, seq_len, head_dim)
        k = torch.randn(1, num_kv_heads, seq_len, head_dim)
        v = torch.randn(1, num_kv_heads, seq_len, head_dim)

        q_pert = q.clone()
        k_pert = k.clone()
        v_pert = v.clone()
        q_pert[:, :, 40:, :] = torch.randn(1, num_q_heads, 8, head_dim)
        k_pert[:, :, 40:, :] = torch.randn(1, num_kv_heads, 8, head_dim)
        v_pert[:, :, 40:, :] = torch.randn(1, num_kv_heads, 8, head_dim)

        slot_a = GlobalPagedVisitSlot(0, 3, 0, 0, 0, cache_cfg)
        slot_a.append(k, v, indexer=indexer)
        out_a = paged_dlsa_global_attention_forward(
            query=q, slot=slot_a, indexer=indexer, sparse_cfg=sparse_cfg
        )

        slot_b = GlobalPagedVisitSlot(0, 3, 0, 0, 0, cache_cfg)
        slot_b.append(k_pert, v_pert, indexer=indexer)
        out_b = paged_dlsa_global_attention_forward(
            query=q_pert, slot=slot_b, indexer=indexer, sparse_cfg=sparse_cfg
        )
        # Strict causality: perturbing tokens >= 40 must have zero effect on queries < 40
        torch.testing.assert_close(out_a[:, :, :40, :], out_b[:, :, :40, :], atol=1e-6, rtol=1e-6)

        # Last query of vectorized multi-query prefill must match single-query decode at position 47
        slot_split = GlobalPagedVisitSlot(0, 3, 0, 0, 0, cache_cfg)
        slot_split.append(k[:, :, :47, :], v[:, :, :47, :], indexer=indexer)
        slot_split.append(k[:, :, 47:48, :], v[:, :, 47:48, :], indexer=indexer)
        out_decode_last = paged_dlsa_global_attention_forward(
            query=q[:, :, 47:48, :], slot=slot_split, indexer=indexer, sparse_cfg=sparse_cfg, known_upto=48
        )
        torch.testing.assert_close(out_a[:, :, 47:48, :], out_decode_last, atol=1e-5, rtol=1e-5)


@unittest.skipIf(torch is None, 'torch not installed')
class GenerationHookTests(unittest.TestCase):
    def setUp(self):
        # Generation uses an injected cache; the adapter import is irrelevant to
        # this test and avoids requiring Transformers for CPU callback contracts.
        cache_module = types.ModuleType('transformers.cache_utils')
        cache_module.DynamicCache = object
        self.cache_patch = mock.patch.dict(sys.modules, {'transformers.cache_utils': cache_module})
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def generate(self, model, **kwargs):
        return model.generate_cached(
            torch.tensor([[6]]), cache=object(), do_sample=False,
            repetition_penalty=1.0, stream_print=False, **kwargs,
        )

    def test_callback_observes_first_normal_and_eos_once(self):
        model = GenerationStub([7, 8, 2])
        seen = []
        output = self.generate(model, max_new_tokens=8, eos_token_id=2, token_callback=seen.append)
        self.assertEqual(seen, output[0, 1:].tolist())
        self.assertEqual(seen, [7, 8, 2])
        self.assertEqual(model.last_generation_timing['finish_reason'], 'stop')
        self.assertTrue(model.training)

    def test_callback_observes_budget_tokens_and_can_disable_injection(self):
        for force in (True, False):
            with self.subTest(force=force):
                model = GenerationStub([7])
                seen = []
                output = self.generate(model, max_new_tokens=4, think_budget_tokens=1,
                                       force_think_close=force, token_callback=seen.append)
                self.assertEqual(seen, output[0, 1:].tolist())
                self.assertEqual(4 in seen, force)
                self.assertEqual(model.last_generation_timing['finish_reason'], 'length')

    def test_cancellation_before_prefill_and_between_tokens_restores_mode(self):
        for cancel_at in (0, 2):
            with self.subTest(cancel_at=cancel_at):
                model = GenerationStub([7])
                seen = []
                with self.assertRaises(InterruptedError):
                    self.generate(model, max_new_tokens=8, token_callback=seen.append,
                                  should_stop=lambda: len(seen) >= cancel_at)
                self.assertEqual(len(seen), cancel_at)
                self.assertEqual(len(model.calls), cancel_at)
                self.assertEqual(model.last_generation_timing['finish_reason'], 'cancelled')
                self.assertTrue(model.training)

    def test_cancellation_between_injected_tokens(self):
        model = GenerationStub([7])
        seen = []
        with self.assertRaises(InterruptedError):
            self.generate(model, max_new_tokens=8, think_budget_tokens=1, token_callback=seen.append,
                          should_stop=lambda: len(seen) >= 2)
        self.assertEqual(seen, [7, 4])
        self.assertEqual(len(model.calls), 2)

    def test_callback_failure_restores_mode(self):
        model = GenerationStub([7])
        with self.assertRaisesRegex(RuntimeError, 'consumer failed'):
            self.generate(model, max_new_tokens=2, token_callback=lambda _: (_ for _ in ()).throw(RuntimeError('consumer failed')))
        self.assertEqual(model.last_generation_timing['finish_reason'], 'error')
        self.assertTrue(model.training)

    def test_zero_budget_does_not_prefill_or_emit(self):
        model = GenerationStub([7])
        seen = []
        output = self.generate(model, max_new_tokens=0, token_callback=seen.append)
        self.assertEqual(output.tolist(), [[6]])
        self.assertEqual(seen, [])
        self.assertEqual(model.calls, [])


@unittest.skipIf(torch is None, 'torch not installed')
class LoaderContractTests(unittest.TestCase):
    def setUp(self):
        # Execute the checked-in loader functions with download boundaries stubbed.
        # Actual selector modules, controller state and checkpoint serialization run
        # on CPU; this is not a Hugging Face download or full-backbone integration test.
        source = ast.parse((ROOT / 'load_inference_v03.py').read_text())
        names = {'_file_sha256', 'configure_model_for_loop72_lsa', 'load_sparkx_loop72_lsa'}
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes.extend(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names)
        self.pretrained_calls = []

        def backbone(source, **kwargs):
            self.pretrained_calls.append(('model', source, kwargs))
            result = TinyBackbone()
            result.config._commit_hash = 'resolved-model-sha'
            return result

        def tokenizer(source, **kwargs):
            self.pretrained_calls.append(('tokenizer', source, kwargs))
            return types.SimpleNamespace(pad_token_id=0, name_or_path=source,
                                         init_kwargs={'_commit_hash': 'resolved-tokenizer-sha'})

        self.cfg = {'model': {'preserve_native_activation': True}, 'loop': {}, 'memory': {'m3_enabled': False}}
        self.loader = {
            'torch': torch, 'os': os, 'hashlib': hashlib, 'json': json,
            'LoopV03Settings': LoopV03Settings,
            'SparkXLoopV03': lambda base, settings, **kw: SparkXLoopV03(base, settings, strict=False, patch_sdpa=False),
            'load_loop_v03_state_dict': load_loop_v03_state_dict,
            'patch_spark_for_unsloth_triton': lambda *args, **kwargs: {},
            '_patch_transformers_v5_tied_weights': lambda: None,
            'load_loop72_lsa_yaml_config': lambda *args: self.cfg,
            'AutoModelForCausalLM': types.SimpleNamespace(from_pretrained=backbone),
            'AutoTokenizer': types.SimpleNamespace(from_pretrained=tokenizer),
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'loader_contract', 'exec'), self.loader)

        def attach(model, index_bank, sparse_cfg):
            model.dlsa_index_bank = index_bank
            model.sparse_attention_cfg = sparse_cfg
            model.attention_runtime = 'paged_dlsa_inference'
            model.attention_training_supported = False
            return {'test_attachment': True}

        self.attach_patch = mock.patch('sparkx_sparse_attention.attach_loop72_dlsa_attention', side_effect=attach)
        self.attach_patch.start()
        self.addCleanup(self.attach_patch.stop)

    def configure(self, model, **kwargs):
        return self.loader['configure_model_for_loop72_lsa'](model, config=self.cfg, **kwargs)

    def test_reconfigure_preserves_controller_unless_reset_explicit(self):
        model = tiny_loop('learned_mhc')
        with torch.no_grad():
            model.controllers['4'].extra_gain.fill_(0.123)
        before = model.controllers['4'].extra_gain.detach().clone()
        self.configure(model)
        self.assertTrue(torch.equal(before, model.controllers['4'].extra_gain))
        self.assertEqual(model.settings.recurrence_mode, 'learned_mhc')
        self.assertEqual(model.dlsa_checkpoint_status['initialization'], 'random_orthogonal')
        self.cfg['loop']['reset_mhc_controllers'] = True
        self.configure(model)
        self.assertEqual(float(model.controllers['4'].extra_gain), 0.0)

    def test_missing_checkpoint_fails_and_real_checkpoint_is_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / 'selector.pt'
            with self.assertRaises(FileNotFoundError):
                self.configure(tiny_loop(), dlsa_checkpoint=str(checkpoint))
            source = tiny_loop()
            self.configure(source)
            with torch.no_grad():
                source.dlsa_index_bank.indexers['0'].w_q.weight.fill_(0.375)
            torch.save(source.dlsa_index_bank.export_manifest(), checkpoint)
            restored = tiny_loop()
            self.configure(restored, dlsa_checkpoint=str(checkpoint))
            self.assertTrue(torch.equal(source.dlsa_index_bank.indexers['0'].w_q.weight,
                                        restored.dlsa_index_bank.indexers['0'].w_q.weight))
            status = restored.dlsa_checkpoint_status
            self.assertEqual(status['initialization'], 'checkpoint_loaded_unverified')
            self.assertEqual(status['checkpoint_sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(status['quality_evidence'], 'not_evaluated_in_loader')

    def test_local_loop_state_restores_selector_mode_and_revision_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = tiny_loop('learned_mhc')
            self.configure(source)
            with torch.no_grad():
                source.controllers['4'].extra_gain.fill_(0.123)
                source.dlsa_index_bank.indexers['0'].w_q.weight.fill_(0.375)
            checkpoint = pathlib.Path(directory) / 'loop_v03_state.pt'
            torch.save(loop_v03_state_dict(source), checkpoint)
            restored, _, meta = self.loader['load_sparkx_loop72_lsa'](
                repo_or_dir=directory, device='cpu', dtype=torch.float32,
                revision='requested-revision', return_meta=True,
            )
            self.assertEqual(restored.effective_recurrence_mode(), 'learned_mhc')
            self.assertTrue(torch.equal(source.controllers['4'].extra_gain, restored.controllers['4'].extra_gain))
            self.assertTrue(torch.equal(source.dlsa_index_bank.indexers['0'].w_q.weight,
                                        restored.dlsa_index_bank.indexers['0'].w_q.weight))
            self.assertEqual(restored.dlsa_checkpoint_status['initialization'], 'loop_state_checkpoint_loaded_unverified')
            self.assertEqual(meta['base_runtime']['resolved_revision'], 'resolved-model-sha')
            self.assertEqual(meta['base_runtime']['tokenizer_resolved_revision'], 'resolved-tokenizer-sha')
            self.assertTrue(all(call[2]['revision'] == 'requested-revision' for call in self.pretrained_calls))
            self.assertEqual(meta['loop72_lsa']['manifest'], restored.execution_manifest())

    def test_dense_training_loader_never_attaches_inference_archive(self):
        model, _ = self.loader['load_sparkx_loop72_lsa'](
            device='cpu', dtype=torch.float32, recurrence_mode='learned_mhc',
            attach_sparse_attention=False,
        )
        self.assertIsNone(getattr(model, 'dlsa_index_bank', None))
        self.assertFalse(model.settings.use_paged_hybrid_cache)
        self.assertEqual(model.execution_manifest()['attention_runtime'], 'native_differentiable_sdpa')
        self.assertTrue(model.execution_manifest()['attention_training_supported'])
        self.assertEqual(model.effective_recurrence_mode(True), 'learned_mhc')
        self.assertEqual(model.effective_recurrence_mode(False), 'learned_mhc')


if __name__ == '__main__':
    unittest.main()
