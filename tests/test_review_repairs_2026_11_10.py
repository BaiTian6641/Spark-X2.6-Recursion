"""Targeted regression suite for the 2026-11-10 Loop72 + Per-Visit Paged D-LSA audit & repair (`F01`–`F18`, `§17.2`)."""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from load_inference_v03 import configure_model_for_loop72_lsa, load_sparkx_loop72_lsa
from sparkx_dlsa import (
    DLSAIndexConfig,
    FaithfulDLSAIndexer,
    PagedLatentIndex,
    VisitAwareDLSAIndexBank,
    canonical_state_dict_sha256,
)
from sparkx_kv_codec import KVCodecConfig
from sparkx_loop_v03_core import (
    LoopV03Settings,
    SparkXLoopV03,
    _notify_sparkx_qkv_capture,
)
from sparkx_omp_server import DEFAULT_MODEL_ID, _TokenTextStreamer
from sparkx_paged_cache import (
    GlobalPagedVisitSlot,
    LocalSlidingRingSlot,
    Loop72HybridPagedCache,
    PagedCacheConfig,
)
from sparkx_sparse_attention import (
    SparseAttentionConfig,
    paged_dlsa_global_attention_forward,
)
from sparkx_training_data import (
    OMP_V18_18_1_TOOLS,
    _synthesize_verified_code_and_math_records,
    build_omp_tool_trajectory_examples,
    build_real_five_domain_corpus,
    render_conversation_exact_labels,
)
from train_spark_dlsa import (
    VisitDistillConfig,
    VisitDistillSample,
    capture_loop72_global_visit_qkv,
    distill_loop72_visit_bank,
    distill_single_visit_indexer,
)
from train_spark_mhc import (
    MHCTrainingConfig,
    _parameter_sha256,
    apply_hardware_profile_to_config,
    export_and_upload_recursive72_to_hub,
    finalize_release_from_hub_stages,
    render_example_to_Ids_and_labels,
    run_mhc_training_pipeline,
    run_stage_a_verification,
    verify_clean_export_directory,
)


class _NativeCallbackSelfAttn(nn.Module):
    """Mock native Spark attention module with `q_norm`/`k_norm` that emits Q/K via `_notify_sparkx_qkv_capture`."""

    def __init__(
        self,
        hidden_size: int = 16,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 4,
        emit_callback: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.emit_callback = emit_callback
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.emit_callback:
            _notify_sparkx_qkv_capture(self, q, k, v)
        k_rep = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        v_rep = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        merged = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(merged), None


class _TinyMLP(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyDecoderLayer(nn.Module):
    def __init__(self, index: int, hidden_size: int = 16, emit_callback: bool = True) -> None:
        super().__init__()
        self.index = index
        self.layer_type = "full_attention" if index % 4 == 3 else "sliding_attention"
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.self_attn = _NativeCallbackSelfAttn(hidden_size=hidden_size, emit_callback=emit_callback)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.mlp = _TinyMLP(hidden_size=hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        h = hidden_states + self.self_attn(self.input_layernorm(hidden_states), **kwargs)[0]
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class _TinySparkConfig:
    def __init__(self, hidden_size: int = 16, vocab_size: int = 64) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = False

    def get_partial_rotary_factor(self, layer_type: str) -> float:
        return 0.25 if layer_type == "full_attention" else 1.0


class _TinyBackboneInner(nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 64, emit_callback: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embed_tokens = self.embedding
        self.layers = nn.ModuleList(
            [_TinyDecoderLayer(i, hidden_size=hidden_size, emit_callback=emit_callback) for i in range(36)]
        )
        self.norm = nn.RMSNorm(hidden_size)


class _TinyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 64, emit_callback: bool = True) -> None:
        super().__init__()
        self.config = _TinySparkConfig(hidden_size=hidden_size, vocab_size=vocab_size)
        self.model = _TinyBackboneInner(hidden_size=hidden_size, vocab_size=vocab_size, emit_callback=emit_callback)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _build_tiny_loop_model(emit_callback: bool = True) -> SparkXLoopV03:
    torch.manual_seed(7)
    base = _TinyCausalLM(hidden_size=16, vocab_size=64, emit_callback=emit_callback)
    base.requires_grad_(False)
    settings = LoopV03Settings.for_loop72_lsa(
        checkpoint_layers=False,
        recurrence_mode="learned_mhc",
        mhc_gain_mode="sigmoid",
    )
    model = SparkXLoopV03(base, settings, strict=False, patch_sdpa=False)

    def _ctx(self, embeds, attention_mask, past_key_values=None, cache_position=None):
        kinds = ("full_attention", "sliding_attention")
        return ({k: None for k in kinds}, {k: None for k in kinds}), torch.arange(embeds.shape[1], device=embeds.device)

    model._forward_context = types.MethodType(_ctx, model)
    model.base_runtime_info = {
        "source": "XHToken/Spark-X2.6-4B",
        "requested_revision": "64ca8f2a295b8ceb2757b9d4749a6558c4759e5b",
        "resolved_revision": "64ca8f2a295b8ceb2757b9d4749a6558c4759e5b",
    }
    return model


class _OffsetCharTokenizer:
    """Deterministic character-level test tokenizer supporting `return_offsets_mapping=True`."""

    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids = [max(3, (ord(ch) % 60) + 3) for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def encode(self, text: str, add_special_tokens: bool = False):
        return [max(3, (ord(ch) % 60) + 3) for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs):
        return "".join(chr(65 + (int(t) % 26)) for t in token_ids)


class TestReviewRepairs20261110(unittest.TestCase):
    """All 14 targeted regression tests required by §17.2 of the 2026-11-10 Loop72 + D-LSA Plan."""

    def test_01_native_18_of_18_global_visit_qk_capture_and_fail_closed(self) -> None:
        model = _build_tiny_loop_model(emit_callback=True)
        inp = torch.randint(1, 60, (1, 12), dtype=torch.long)
        captured = capture_loop72_global_visit_qkv(model, inp, include_values=False, strict=True)
        self.assertEqual(len(captured), 18)
        for gv in range(18):
            self.assertIn(gv, captured)
            q_t, k_t = captured[gv]
            self.assertEqual(tuple(q_t.shape), (4, 12, 4))
            self.assertEqual(tuple(k_t.shape), (2, 12, 4))

        broken_model = _build_tiny_loop_model(emit_callback=False)
        with self.assertRaises(RuntimeError):
            capture_loop72_global_visit_qkv(broken_model, inp, include_values=False, strict=True)

    def test_02_stage_e_fail_closed_on_missing_or_skipped_visit(self) -> None:
        bank = VisitAwareDLSAIndexBank(
            base_cfg=DLSAIndexConfig(n_kv=2, head_dim=4, rope_dim=1, d_idx=4, sinks=1, recent_tokens=2, n_cluster=2, n_pool=1),
            maps_per_global_visit=True,
        )
        q = torch.randn(4, 10, 4)
        k = torch.randn(2, 10, 4)
        incomplete_samples = {gv: [(q, k)] for gv in range(17)}  # Missing visit 17!
        with self.assertRaises(RuntimeError):
            distill_loop72_visit_bank(
                bank,
                incomplete_samples,
                cfg=VisitDistillConfig(d_idx=4, steps=1, fail_on_skipped_visit=True, require_global_visits=18),
            )

    def test_03_exact_assistant_only_masking_and_zero_target_truncation_rejection(self) -> None:
        tok = _OffsetCharTokenizer()
        example = {
            "id": "mask_test_1",
            "domain": "tool_trajectories",
            "messages": [
                {"role": "system", "content": "SYSTEM_HEADER_PROMPT_DO_NOT_SUPERVISE"},
                {"role": "user", "content": "USER_LONG_PROBLEM_STATEMENT_NEVER_SUPERVISE_THIS_PREFIX"},
                {"role": "assistant", "reasoning_content": "plan step", "content": "final_answer_42"},
            ],
        }
        full_rw = render_conversation_exact_labels(example, tok, max_seq_len=512, allow_tail_rewindow=False)
        self.assertTrue(full_rw.valid)
        self.assertGreater(full_rw.supervised_tokens, 0)
        self.assertGreater(full_rw.masked_tokens, full_rw.supervised_tokens)

        # Truncating inside the system/user prefix (`max_seq_len=24`) MUST reject rather than supervise user tokens!
        trunc_rw = render_conversation_exact_labels(example, tok, max_seq_len=24, allow_tail_rewindow=False)
        self.assertFalse(trunc_rw.valid)
        self.assertEqual(trunc_rw.supervised_tokens, 0)
        self.assertTrue(all(lbl == -100 for lbl in trunc_rw.labels))
        with self.assertRaises(ValueError):
            render_example_to_Ids_and_labels(example, tok, max_seq_len=24, allow_tail_rewindow=False)

    def test_04_stage_a_bit_exact_state_neutrality(self) -> None:
        model = _build_tiny_loop_model(emit_callback=True)
        with torch.no_grad():
            for idx, ctrl in enumerate(model.controllers.values()):
                ctrl.pass_up[1].weight.normal_(mean=0.02 * (idx + 1), std=0.1)
                ctrl.extra_gain.fill_(-1.2 + 0.05 * idx)
                ctrl.p3_extra_gain.fill_(-2.1 + 0.03 * idx)
        # Set custom requires_grad pattern to verify exact restoration
        for idx, ctrl in enumerate(model.controllers.values()):
            ctrl.requires_grad_(idx % 2 == 0)

        sha_before = canonical_state_dict_sha256(model.controllers.state_dict())
        req_grad_before = {n: p.requires_grad for n, p in model.controllers.named_parameters()}
        torch.manual_seed(12345)
        rng_before = torch.get_rng_state().clone()

        cfg = MHCTrainingConfig(hardware_profile="cpu_test", seq_len=16)
        rep = run_stage_a_verification(model, cfg=cfg, vocab_size=64, seq_len=12)
        self.assertTrue(rep["passed"])
        self.assertTrue(rep["state_neutral"])
        self.assertEqual(rep["mutated_controller_tensors"], 0)
        self.assertEqual(rep["captured_global_visits"], 18)

        sha_after = canonical_state_dict_sha256(model.controllers.state_dict())
        req_grad_after = {n: p.requires_grad for n, p in model.controllers.named_parameters()}
        rng_after = torch.get_rng_state()
        self.assertEqual(sha_before, sha_after)
        self.assertEqual(req_grad_before, req_grad_after)
        self.assertTrue(torch.equal(rng_before, rng_after))

    def test_05_full_byte_sha256_detects_interior_tensor_mutation(self) -> None:
        param = nn.Parameter(torch.zeros(16384, dtype=torch.float32))
        sha_p0 = _parameter_sha256([("big_w", param)])
        sha_sd0 = canonical_state_dict_sha256({"big_w": param.data})

        # Mutate a single interior element at index 8192 (outside any 4096 head/tail slice!)
        with torch.no_grad():
            param[8192] = 1e-5
        sha_p1 = _parameter_sha256([("big_w", param)])
        sha_sd1 = canonical_state_dict_sha256({"big_w": param.data})
        self.assertNotEqual(sha_p0, sha_p1)
        self.assertNotEqual(sha_sd0, sha_sd1)

    def test_06_single_pass_hardware_profile_and_explicit_override_precedence(self) -> None:
        mocked_l4 = {
            "profile": "l4_24gb",
            "gpu_name": "NVIDIA L4",
            "detected_vram_gb": 23.8,
            "seq_len": 4096,
            "stage_e_seq_len": 8192,
            "max_recommended_seq_len": 4096,
            "micro_batch_size": 1,
            "grad_accum_steps": 16,
            "sampled_kl_positions": 32,
            "ce_chunk_tokens": 1024,
        }
        with mock.patch("train_spark_mhc.detect_gpu_training_profile", return_value=mocked_l4):
            cfg_auto = MHCTrainingConfig(hardware_profile="auto")
            apply_hardware_profile_to_config(cfg_auto)
            self.assertEqual(cfg_auto.seq_len, 4096)
            self.assertEqual(cfg_auto.grad_accum_steps, 16)
            self.assertFalse(cfg_auto.allow_smoke_fixtures)

        cfg_explicit = MHCTrainingConfig(hardware_profile="colab_g4_96gb", seq_len=2048, grad_accum_steps=3)
        cfg_copy = copy.deepcopy(cfg_explicit)
        apply_hardware_profile_to_config(cfg_copy)
        self.assertEqual(cfg_copy.seq_len, 2048)
        self.assertEqual(cfg_copy.grad_accum_steps, 3)
        self.assertFalse(cfg_copy.allow_smoke_fixtures)

    def test_07_stage_e_scalar_gradients_and_matched_support_kl(self) -> None:
        torch.manual_seed(99)
        idx_cfg = DLSAIndexConfig(
            n_kv=2, head_dim=4, rope_dim=1, d_idx=4, sinks=2, recent_tokens=4, n_cluster=4, n_pool=2, k_clusters=2, max_selected_keys=6
        )
        indexer = FaithfulDLSAIndexer(idx_cfg)
        q_sub = torch.randn(4, 8, 4)
        k_full = torch.randn(2, 24, 4)
        q_pos = torch.tensor([12, 14, 16, 18, 19, 21, 22, 23], dtype=torch.long)
        sample = VisitDistillSample(q_sub=q_sub, k_full=k_full, q_positions=q_pos)

        rep = distill_single_visit_indexer(
            indexer,
            [sample],
            cfg=VisitDistillConfig(
                d_idx=4, steps=12, lr=5e-3, scalar_lr=1e-2, sampled_queries=8, require_long_range_support=True, eval_every_steps=4
            ),
            visit_id=0,
        )
        self.assertEqual(rep["status"], "ok")
        self.assertTrue(rep["proj_grads_nonzero"])
        self.assertTrue(rep["scalar_grads_nonzero"])
        self.assertLessEqual(rep["final_loss"], rep["initial_loss"] + 1e-6)
        self.assertGreater(float(indexer.dist_alpha().item()), 0.0)

    def test_08_strict_selector_provenance_and_unpinned_revision_prohibition(self) -> None:
        model = _build_tiny_loop_model(emit_callback=True)
        idx_cfg = DLSAIndexConfig(n_kv=2, head_dim=4, rope_dim=1, d_idx=4, sinks=1, recent_tokens=2, n_cluster=2, n_pool=1)
        bank = VisitAwareDLSAIndexBank(base_cfg=idx_cfg, maps_per_global_visit=True)
        samples = {gv: [(torch.randn(4, 10, 4), torch.randn(2, 10, 4))] for gv in range(18)}
        distill_loop72_visit_bank(
            bank,
            samples,
            cfg=VisitDistillConfig(d_idx=4, steps=2, fail_on_skipped_visit=True, require_global_visits=18),
            model=model,
        )
        tiny_lsa_cfg = {
            "model": {"preserve_native_activation": True},
            "loop": {"recurrence_mode": "learned_mhc", "mhc_gain_mode": "sigmoid"},
            "memory": {"m3_enabled": False},
            "attention": {
                "n_kv": 2,
                "head_dim": 4,
                "rope_dim": 1,
                "d_idx": 4,
                "sinks": 1,
                "recent_tokens": 2,
                "n_cluster": 2,
                "n_pool": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            sel_path = pathlib.Path(tmp) / "visit_selectors_stage_e.pt"
            torch.save(bank.export_manifest(), sel_path)
            configure_model_for_loop72_lsa(
                model,
                config=tiny_lsa_cfg,
                selector_checkpoint=str(sel_path),
                recurrence_mode="learned_mhc",
                strict_selector_provenance=True,
                require_verified_selectors=True,
            )
            self.assertEqual(
                model.execution_manifest()["selector_checkpoint_status"]["initialization"],
                "checkpoint_loaded_verified",
            )

            # Mutating controller weights after Stage E MUST fail strict selector provenance!
            mutated_model = _build_tiny_loop_model(emit_callback=True)
            with torch.no_grad():
                next(iter(mutated_model.controllers.values())).extra_gain.add_(0.5)
            with self.assertRaises(ValueError):
                configure_model_for_loop72_lsa(
                    mutated_model,
                    config=tiny_lsa_cfg,
                    selector_checkpoint=str(sel_path),
                    recurrence_mode="learned_mhc",
                    strict_selector_provenance=True,
                    require_verified_selectors=True,
                )

        # Unpinned revision fallback must be prohibited by default (`F05`)
        call_revisions = []

        def _failing_from_pretrained(*args, **kwargs):
            call_revisions.append(kwargs.get("revision"))
            raise OSError("Pinned revision unreachable")

        with mock.patch("load_inference_v03.AutoTokenizer.from_pretrained", side_effect=lambda *a, **kw: _OffsetCharTokenizer()), \
             mock.patch("load_inference_v03.AutoModelForCausalLM.from_pretrained", side_effect=_failing_from_pretrained):
            with self.assertRaises(OSError):
                load_sparkx_loop72_lsa(
                    repo_or_dir="XHToken/Spark-X2.6-4B",
                    revision="deadbeef",
                    allow_unpinned_fallback=False,
                )
        self.assertEqual(call_revisions, ["deadbeef"])

    def test_09_isolated_clean_export_subprocess_without_repo_on_pythonpath(self) -> None:
        model = _build_tiny_loop_model(emit_callback=True)
        cfg = MHCTrainingConfig(hardware_profile="cpu_test", push_to_hub=False, export_standalone_backbone=False)
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = pathlib.Path(tmp) / "clean_export"
            export_and_upload_recursive72_to_hub(model, _OffsetCharTokenizer(), export_dir=exp_dir, cfg=cfg, report={"stages": {}})
            check = verify_clean_export_directory(exp_dir)
            self.assertTrue(check["passed"], f"Clean export subprocess failed: {check}")
            self.assertEqual(check["bad_imports"], [])

    def test_10_physical_2x_ring_accounting_and_owned_hot_slab_non_aliasing(self) -> None:
        ring = LocalSlidingRingSlot(virtual_visit=0, physical_layer=0, pass_idx=0, local_window=8)
        for _ in range(3):
            ring.update(torch.randn(1, 2, 1, 4, dtype=torch.float32), torch.randn(1, 2, 1, 4, dtype=torch.float32))
        # Physical backing ring allocates `2 * local_window = 16` tokens for K, V, and int64 abs_positions (`> logical_gpu_bytes()`)
        expected_phys = 2 * (1 * 2 * 16 * 4 * 4) + (16 * 8)
        self.assertEqual(ring.allocated_gpu_bytes(), expected_phys)
        self.assertGreater(ring.allocated_gpu_bytes(), ring.logical_gpu_bytes())

        pcfg = PagedCacheConfig(sinks=2, recent_tokens=16, kv_page_tokens=8, global_hot_tokens_per_visit=26, cold_codec="bf16")
        slot = GlobalPagedVisitSlot(
            virtual_visit=3, physical_layer=3, pass_idx=0, global_visit_id=0, layer_visit_ordinal=0, cfg=pcfg
        )
        pages_ref = []
        for step_i in range(6):
            k_chunk = torch.full((1, 2, 8, 4), float(step_i + 1), dtype=torch.bfloat16)
            v_chunk = torch.full((1, 2, 8, 4), float((step_i + 1) * 10), dtype=torch.bfloat16)
            pages_ref.append((k_chunk.clone(), v_chunk.clone()))
            slot.append(k_chunk, v_chunk)

        # Gather tokens spanning multiple archived/hot pages simultaneously and verify zero slab aliasing
        sel = torch.tensor([2, 3, 10, 11, 18, 19, 26, 27], dtype=torch.long)
        g_k, g_v = slot.gather_selected(sel, device=torch.device("cpu"), dtype=torch.bfloat16)
        full_k = torch.cat([p[0] for p in pages_ref], dim=2)
        full_v = torch.cat([p[1] for p in pages_ref], dim=2)
        self.assertTrue(torch.equal(g_k, full_k[:, :, sel, :]))
        self.assertTrue(torch.equal(g_v, full_v[:, :, sel, :]))

    def test_11_nvme_cold_spill_reload_and_gpu_index_budget_offload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_nvme:
            idx_cfg = DLSAIndexConfig(
                n_kv=2, head_dim=4, rope_dim=1, d_idx=4, sinks=2, recent_tokens=16, n_cluster=8, n_pool=2
            )
            bank = VisitAwareDLSAIndexBank(base_cfg=idx_cfg, maps_per_global_visit=True)
            cache_cfg = PagedCacheConfig(
                local_window=8,
                sinks=2,
                recent_tokens=16,
                kv_page_tokens=8,
                global_hot_tokens_per_visit=26,
                archive_backend="cpu_then_nvme",
                nvme_enabled=True,
                nvme_spill_dir=tmp_nvme,
                host_cold_budget_bytes=64,  # Force immediate spill of cold pages from RAM to NVMe
                index_gpu_budget_bytes=1,
            )
            cache = Loop72HybridPagedCache(cfg=cache_cfg)
            g_slot = cache.get_visit_slot(3)  # visit 3 is global visit #0 (`layer 3`)
            indexer = bank.get_visit_indexer(0)

            k_all = torch.randn(1, 2, 48, 4, dtype=torch.bfloat16)
            v_all = torch.randn(1, 2, 48, 4, dtype=torch.bfloat16)
            g_slot.append(k_all, v_all, indexer=indexer)

            ledger = cache.memory_ledger()
            self.assertGreater(ledger["nvme_cold_archive_bytes"], 0)
            self.assertGreater(ledger["nvme_spills"], 0)

            sel = torch.tensor([2, 5, 11, 19, 25], dtype=torch.long)
            gk, gv = g_slot.gather_selected(sel, device=torch.device("cpu"), dtype=torch.bfloat16)
            self.assertTrue(torch.equal(gk, k_all[:, :, sel, :]))
            self.assertTrue(torch.equal(gv, v_all[:, :, sel, :]))

            idx_obj = g_slot.dlsa_state_slot.get("index")
            self.assertIsNotNone(idx_obj)
            idx_obj.offload_index_to_cpu()
            self.assertEqual(idx_obj.index_allocated_bytes("cuda"), 0)

    def test_12_sink_boundary_1_to_3_token_prefill_equivalence_and_dense_fallback_guard(self) -> None:
        torch.manual_seed(21)
        idx_cfg = DLSAIndexConfig(n_kv=2, head_dim=4, rope_dim=1, d_idx=4, sinks=4, recent_tokens=4, n_cluster=4, n_pool=2)
        indexer = FaithfulDLSAIndexer(idx_cfg)
        k_seq = torch.randn(2, 20, 4)

        k_lat_all = indexer.encode_keys(k_seq)
        oneshot = PagedLatentIndex(k_lat_all, n_sink=4, n_pool=2, n_cluster=4, window=4)
        # Incremental path starting with ONLY 2 tokens (`< sinks=4`) followed by chunks!
        inc = PagedLatentIndex(k_lat_all[:2], n_sink=4, n_pool=2, n_cluster=4, window=4)
        for start_p, end_p in ((2, 5), (5, 11), (11, 16), (16, 20)):
            inc.append_new_keys(k_lat_all[start_p:end_p])

        self.assertEqual(oneshot.t, inc.t)
        self.assertEqual(oneshot.n_blocks, inc.n_blocks)
        self.assertEqual(oneshot._compact_len, inc._compact_len)
        self.assertTrue(
            torch.allclose(
                oneshot._buf_compact_cents[: oneshot._compact_len],
                inc._buf_compact_cents[: inc._compact_len],
                atol=1e-5,
            )
        )

        # Fail-closed guard when `seq_len > exact_threshold`, `slot` has no `PagedLatentIndex`, and `unbounded_dense_fallback=False` (`F12`)
        pcfg = PagedCacheConfig(sinks=4, recent_tokens=16, kv_page_tokens=8, global_hot_tokens_per_visit=32)
        slot = GlobalPagedVisitSlot(
            virtual_visit=3, physical_layer=3, pass_idx=0, global_visit_id=0, layer_visit_ordinal=0, cfg=pcfg
        )
        q = torch.randn(1, 4, 24, 4, dtype=torch.bfloat16)
        k = torch.randn(1, 2, 24, 4, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 24, 4, dtype=torch.bfloat16)
        slot.append(k, v, indexer=None)  # Intentionally omit index creation
        sparse_cfg = SparseAttentionConfig(exact_threshold=8, unbounded_dense_fallback=False)
        with self.assertRaises(RuntimeError):
            paged_dlsa_global_attention_forward(q, slot=slot, indexer=indexer, sparse_cfg=sparse_cfg)

    def test_13_bounded_sparse_attention_workspace_tiling_parity(self) -> None:
        torch.manual_seed(33)
        idx_cfg = DLSAIndexConfig(
            n_kv=2,
            head_dim=4,
            rope_dim=1,
            d_idx=4,
            sinks=2,
            recent_tokens=16,
            n_cluster=8,
            n_pool=2,
            k_clusters=2,
            max_selected_keys=10,
            exact_threshold=20,
        )
        indexer = FaithfulDLSAIndexer(idx_cfg)
        q = torch.randn(1, 4, 36, 4, dtype=torch.bfloat16)
        k = torch.randn(1, 2, 36, 4, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 36, 4, dtype=torch.bfloat16)

        pcfg = PagedCacheConfig(sinks=2, recent_tokens=16, kv_page_tokens=8, global_hot_tokens_per_visit=64)
        slot_a = GlobalPagedVisitSlot(
            virtual_visit=3, physical_layer=3, pass_idx=0, global_visit_id=0, layer_visit_ordinal=0, cfg=pcfg
        )
        slot_a.append(k, v, indexer=indexer)
        cfg_a = SparseAttentionConfig(
            exact_threshold=20, sparse_query_tile_tokens=1, max_workspace_bytes=1024 * 1024
        )
        out_a = paged_dlsa_global_attention_forward(q, slot=slot_a, indexer=indexer, budget=8, sparse_cfg=cfg_a)

        slot_b = GlobalPagedVisitSlot(
            virtual_visit=3, physical_layer=3, pass_idx=0, global_visit_id=0, layer_visit_ordinal=0, cfg=pcfg
        )
        slot_b.append(k, v, indexer=indexer)
        cfg_b = SparseAttentionConfig(
            exact_threshold=20, sparse_query_tile_tokens=16, max_workspace_bytes=256  # Forces _workspace_safe_tile_len -> 1
        )
        out_b = paged_dlsa_global_attention_forward(q, slot=slot_b, indexer=indexer, budget=8, sparse_cfg=cfg_b)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-2, rtol=1e-2))

    def test_14_omp_v18_schema_bounded_streamer_and_intermediate_plus_final_hub_commits(self) -> None:
        self.assertEqual(DEFAULT_MODEL_ID, "Spark-X2.6-4B-Recursive72")
        edit_schema = next(t["function"] for t in OMP_V18_18_1_TOOLS if t["function"]["name"] == "edit")
        self.assertEqual(edit_schema["parameters"]["required"], ["input"])
        omp_examples = build_omp_tool_trajectory_examples(count=4)
        found_edit = False
        for ex in omp_examples:
            for m in ex["messages"]:
                for tc in m.get("tool_calls", []) or []:
                    if tc["function"]["name"] == "edit":
                        tc_args = tc["function"]["arguments"]
                        args = json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                        self.assertIn("input", args)
                        self.assertIn("*** Begin Patch", args["input"])
                        found_edit = True
        self.assertTrue(found_edit)

        emitted = []
        streamer = _TokenTextStreamer(_OffsetCharTokenizer(), emit=emitted.append, max_window_tokens=48)
        for t_id in range(1, 301):
            streamer.put(t_id)
            self.assertLessEqual(len(streamer.cache), 49)
        streamer.finish()
        self.assertEqual(len("".join(emitted)), 300)

        # Verify intermediate + final Hub commits to `TassP0lar/Spark-X2.6-4B-Recursive72` (`4 + 1 = 5` commits!)
        model = _build_tiny_loop_model(emit_callback=True)
        base_sha_0 = _parameter_sha256(model.base.named_parameters())
        hub_commits = []

        class _MockHubApi:
            def whoami(self):
                return {"name": "TassP0lar"}

            def create_repo(self, **kwargs):
                return None

            def upload_folder(self, **kwargs):
                hub_commits.append(kwargs)
                return f"commit_{len(hub_commits)}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = MHCTrainingConfig(
                output_dir=tmp_dir,
                hardware_profile="cpu_test",
                allow_smoke_fixtures=True,
                seq_len=16,
                stage_e_seq_len=16,
                stage_b_steps=1,
                stage_c_steps=1,
                stage_d_steps=1,
                stage_e_calibration_seqs=1,
                stage_e_val_seqs=1,
                stage_e_distill_steps=2,
                hf_repo_id="TassP0lar/Spark-X2.6-4B-Recursive72",
                push_to_hub=True,
                push_intermediate_stages=True,
                hf_token="hf_mock_token",
                export_standalone_backbone=False,
            )
            rep = run_mhc_training_pipeline(model, _OffsetCharTokenizer(), cfg=cfg, api_client=_MockHubApi())
            self.assertTrue(rep["base_backbone_immutable"])
            self.assertEqual(_parameter_sha256(model.base.named_parameters()), base_sha_0)
            self.assertEqual(len(rep["intermediate_hub_commits"]), 4)
            self.assertTrue(all(c["uploaded"] for c in rep["intermediate_hub_commits"]))
            self.assertTrue(rep["huggingface_export"]["uploaded"])
            # 4 intermediate commits (Stage B, C, D, E) + 1 final release commit = 5 commits to TassP0lar/Spark-X2.6-4B-Recursive72
            self.assertEqual(len(hub_commits), 5)
            for c_kw in hub_commits:
                self.assertEqual(c_kw["repo_id"], "TassP0lar/Spark-X2.6-4B-Recursive72")

        # Verify all programmatic code/math synthesis branches & corpus builder execute cleanly
        aug_records = _synthesize_verified_code_and_math_records(count_code=9, count_math=6, seed=42)
        self.assertEqual(len(aug_records), 15)
        splits_aug, sum_aug = build_real_five_domain_corpus(
            target_total_records=40,
            seed=42,
            use_hf_streaming=False,
            allow_programmatic_aug=True,
            allow_smoke_fixtures=True,
        )
        self.assertGreaterEqual(sum_aug["total_records"], 40)
        self.assertGreater(len(splits_aug["train"]), 0)
        self.assertGreater(len(splits_aug["dev"]), 0)

    def test_15_finalize_release_from_hub_stages_and_embedded_selector_verified_status(self) -> None:
        import shutil
        import yaml

        model = _build_tiny_loop_model(emit_callback=True)
        fresh_base_snapshot = copy.deepcopy(model.base)
        tiny_lsa_cfg = {
            "model": {"preserve_native_activation": True},
            "loop": {"recurrence_mode": "learned_mhc", "mhc_gain_mode": "sigmoid"},
            "memory": {"m3_enabled": False},
            "attention": {
                "n_kv": 2,
                "head_dim": 4,
                "rope_dim": 1,
                "d_idx": 6,
                "sinks": 1,
                "recent_tokens": 2,
                "n_cluster": 2,
                "n_pool": 1,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_root:
            hub_mirror = pathlib.Path(tmp_root) / "hub_mirror"
            hub_mirror.mkdir(parents=True, exist_ok=True)
            train_out = pathlib.Path(tmp_root) / "train_out"
            finalize_out = pathlib.Path(tmp_root) / "finalize_out"
            final_commits = []

            class _MirrorHubApi:
                def whoami(self):
                    return {"name": "TassP0lar"}

                def create_repo(self, **kwargs):
                    return None

                def upload_folder(self, **kwargs):
                    src = pathlib.Path(kwargs["folder_path"])
                    for item in src.rglob("*"):
                        if item.is_file():
                            dst = hub_mirror / item.relative_to(src)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dst)
                    final_commits.append(kwargs)
                    return f"mirror_commit_{len(final_commits)}"

            cfg_train = MHCTrainingConfig(
                output_dir=str(train_out),
                hardware_profile="cpu_test",
                allow_smoke_fixtures=True,
                seq_len=16,
                stage_e_seq_len=16,
                stage_b_steps=1,
                stage_c_steps=1,
                stage_d_steps=1,
                stage_e_calibration_seqs=1,
                stage_e_val_seqs=1,
                stage_e_distill_steps=2,
                hf_repo_id="TassP0lar/Spark-X2.6-4B-Recursive72",
                push_to_hub=True,
                push_intermediate_stages=True,
                hf_token="hf_mock_token",
                export_standalone_backbone=False,
            )
            run_mhc_training_pipeline(
                model,
                _OffsetCharTokenizer(),
                cfg=cfg_train,
                run_final_export=False,
                api_client=_MirrorHubApi(),
            )
            # Simulate non-smoke corpus provenance in saved Hub stage manifests so release gate #3 is also exercised
            for st_key in ("stage_b_retention", "stage_c_dense_mhc", "stage_d_omp_mhc"):
                m_path = hub_mirror / "stages" / st_key / "training_manifest.json"
                m_json = json.loads(m_path.read_text(encoding="utf-8"))
                m_json["training_config"]["allow_smoke_fixtures"] = False
                m_path.write_text(json.dumps(m_json, indent=2) + "\n", encoding="utf-8")

            fresh_model = _build_tiny_loop_model(emit_callback=True)
            cfg_fin = MHCTrainingConfig(
                output_dir=str(finalize_out),
                hardware_profile="cpu_test",
                seq_len=16,
                hf_repo_id="TassP0lar/Spark-X2.6-4B-Recursive72",
                push_to_hub=True,
                hf_token="hf_mock_token",
                export_standalone_backbone=False,
            )
            fin_commits_before = len(final_commits)
            fin_rep = finalize_release_from_hub_stages(
                cfg=cfg_fin,
                model=fresh_model,
                tokenizer=_OffsetCharTokenizer(),
                run_stage_a=True,
                run_stage_f=False,
                verify_local_reload_parity=False,
                download_fn=lambda rel: str(hub_mirror / rel),
                api_client=_MirrorHubApi(),
                lsa_config_override=tiny_lsa_cfg,
            )
            self.assertTrue(fin_rep["completion_gates"]["production_release_ready"])
            self.assertTrue(fin_rep["huggingface_export"]["uploaded"])
            self.assertEqual(len(final_commits), fin_commits_before + 1)

            # Verify `load_sparkx_loop72_lsa` against `final_loop72_mhc_dlsa_export` with embedded `dlsa_index_bank`
            exp_dir = pathlib.Path(fin_rep["huggingface_export"]["export_dir"])
            tiny_yaml = exp_dir / "configs" / "tiny_loop72_lsa.yaml"
            tiny_yaml.write_text(yaml.safe_dump(tiny_lsa_cfg), encoding="utf-8")
            with mock.patch("load_inference_v03.AutoTokenizer.from_pretrained", side_effect=lambda *a, **kw: _OffsetCharTokenizer()), \
                 mock.patch("load_inference_v03.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **kw: copy.deepcopy(fresh_base_snapshot)), \
                 mock.patch("load_inference_v03.SparkXLoopV03", side_effect=lambda b, s, **kw: SparkXLoopV03(b, s, strict=False, patch_sdpa=False)):
                reloaded, _ = load_sparkx_loop72_lsa(
                    repo_or_dir=str(exp_dir),
                    config_path=str(tiny_yaml),
                    recurrence_mode="learned_mhc",
                    strict_selector_provenance=True,
                    require_verified_selectors=True,
                )
            sel_status = reloaded.execution_manifest()["selector_checkpoint_status"]
            self.assertEqual(sel_status["initialization"], "loop_state_checkpoint_loaded_verified")
            self.assertTrue(sel_status["verified"])


if __name__ == "__main__":
    unittest.main()
