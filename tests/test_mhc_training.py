"""Unit tests for Milestone 2 Differentiable TwoStreamMHC Training & Stage E Provenance Lock."""
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
    import torch.nn as nn
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    nn = None

from train_spark_mhc import (
    DEFAULT_HF_MODEL_NAME,
    DEFAULT_HF_REPO_ID,
    DOMAIN_MIXTURE_WEIGHTS,
    EXPECTED_DEFAULT_CONTROLLER_PARAMS,
    MHCTrainingConfig,
    apply_hardware_profile_to_config,
    build_five_domain_seed_bank,
    build_omp_tool_trajectory_examples,
    build_recursive72_model_card,
    detect_gpu_training_profile,
    export_and_upload_recursive72_to_hub,
    normalize_hf_repo_id,
    render_example_to_Ids_and_labels,
)

if torch is not None:
    from sparkx_dlsa import DLSAIndexConfig, VisitAwareDLSAIndexBank
    from sparkx_loop_v03_core import (
        LoopV03Settings,
        SparkXLoopV03,
        TwoStreamMHC,
        load_loop_v03_state_dict,
        loop_v03_state_dict,
    )
    from train_spark_mhc import (
        build_mhc_optimizer,
        compute_teacher_retention_step_loss,
        configure_controller_unfreeze_phase,
        prepare_model_for_mhc_training,
        run_mhc_training_pipeline,
        run_stage_a_verification,
        sample_supervised_positions,
        update_cosine_lr,
    )

    class _TinySelfAttn(nn.Module):
        def __init__(self, hidden_size: int = 16, num_heads: int = 4, num_kv_heads: int = 2, head_dim: int = 4):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.num_key_value_heads = num_kv_heads
            self.head_dim = head_dim
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        def forward(self, hidden_states: torch.Tensor, **kwargs):
            bsz, seq_len, _ = hidden_states.shape
            q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            repeat = self.num_heads // self.num_key_value_heads
            k_rep = k.repeat_interleave(repeat, dim=1)
            v_rep = v.repeat_interleave(repeat, dim=1)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
            merged = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
            return self.out_proj(merged), None

    class _TinyMLP(nn.Module):
        def __init__(self, hidden_size: int = 16):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class _TinyDecoderLayer(nn.Module):
        def __init__(self, index: int, hidden_size: int = 16):
            super().__init__()
            self.index = index
            self.layer_type = "full_attention" if index % 4 == 3 else "sliding_attention"
            self.input_layernorm = nn.RMSNorm(hidden_size)
            self.self_attn = _TinySelfAttn(hidden_size=hidden_size)
            self.post_attention_layernorm = nn.RMSNorm(hidden_size)
            self.mlp = _TinyMLP(hidden_size=hidden_size)

        def forward(self, hidden_states: torch.Tensor, **kwargs):
            h = hidden_states + self.self_attn(self.input_layernorm(hidden_states), **kwargs)[0]
            h = h + self.mlp(self.post_attention_layernorm(h))
            return h

    class _TinySparkConfig:
        def __init__(self, hidden_size: int = 16, vocab_size: int = 64):
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size
            self.tie_word_embeddings = False

        def get_partial_rotary_factor(self, layer_type: str) -> float:
            return 0.25 if layer_type == "full_attention" else 1.0

    class _TinySparkBackbone(nn.Module):
        def __init__(self, hidden_size: int = 16, vocab_size: int = 64):
            super().__init__()
            self.config = _TinySparkConfig(hidden_size=hidden_size, vocab_size=vocab_size)
            self.model = nn.Module()
            self.model.embedding = nn.Embedding(vocab_size, hidden_size)
            self.model.layers = nn.ModuleList([_TinyDecoderLayer(i, hidden_size=hidden_size) for i in range(36)])
            self.model.norm = nn.RMSNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.requires_grad_(False)

    def _make_tiny_training_loop(gain_mode: str = "sigmoid") -> SparkXLoopV03:
        torch.manual_seed(7)
        settings = LoopV03Settings.for_loop72_lsa(
            checkpoint_layers=False,
            recurrence_mode="learned_mhc",
            mhc_gain_mode=gain_mode,
        )
        base = _TinySparkBackbone(hidden_size=16, vocab_size=64)
        model = SparkXLoopV03(base, settings, strict=False, patch_sdpa=False)

        def _ctx(self, embeds, attention_mask, past_key_values=None, cache_position=None):
            kinds = ("full_attention", "sliding_attention")
            return ({k: None for k in kinds}, {k: None for k in kinds}), torch.arange(embeds.shape[1], device=embeds.device)

        model._forward_context = types.MethodType(_ctx, model)
        return model


class TestMHCTrainingPurePythonContracts(unittest.TestCase):
    def test_colab_g4_96gb_and_other_hardware_profiles(self):
        g4 = detect_gpu_training_profile(96.0)
        self.assertEqual(g4["profile"], "colab_g4_96gb")
        self.assertEqual(g4["seq_len"], 8192)
        self.assertEqual(g4["max_recommended_seq_len"], 16384)
        self.assertEqual(g4["grad_accum_steps"], 8)
        self.assertEqual(g4["sampled_kl_positions"], 64)

        a100_80 = detect_gpu_training_profile(80.0)
        self.assertEqual(a100_80["profile"], "a100_80gb")
        a100_40 = detect_gpu_training_profile(40.0)
        self.assertEqual(a100_40["profile"], "a100_40gb")
        l4 = detect_gpu_training_profile(24.0)
        self.assertEqual(l4["profile"], "l4_24gb")
        self.assertEqual(l4["seq_len"], 4096)
        cpu = detect_gpu_training_profile(0.0)
        self.assertEqual(cpu["profile"], "cpu_test")

        cfg = MHCTrainingConfig(hardware_profile="colab_g4_96gb")
        applied = apply_hardware_profile_to_config(cfg)
        self.assertEqual(applied["profile"], "colab_g4_96gb")
        self.assertEqual(cfg.seq_len, 8192)
        self.assertEqual(cfg.grad_accum_steps, 8)

    def test_five_domain_mixture_and_assistant_only_label_masking(self):
        self.assertAlmostEqual(sum(DOMAIN_MIXTURE_WEIGHTS.values()), 1.0, places=6)
        bank = build_five_domain_seed_bank()
        domains = {ex["domain"] for ex in bank}
        self.assertEqual(
            domains,
            {"code_debug", "general_instruct", "math_symbolic", "tool_trajectories", "long_doc_repo"},
        )
        omp_examples = build_omp_tool_trajectory_examples()
        self.assertGreaterEqual(len(omp_examples), 2)

        class _CharTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False):
                return [max(1, ord(ch) % 60) for ch in text]

        tok = _CharTokenizer()
        for ex in omp_examples:
            ids, labels = render_example_to_Ids_and_labels(ex, tok, max_seq_len=4096)
            self.assertEqual(len(ids), len(labels))
            masked_count = sum(1 for y in labels if y == -100)
            supervised_count = sum(1 for y in labels if y != -100)
            self.assertGreater(masked_count, 0)
            self.assertGreater(supervised_count, 0)

    def test_huggingface_recursive72_repo_naming_and_model_card(self):
        self.assertEqual(DEFAULT_HF_MODEL_NAME, "Spark X2.6 4B Recursive72")
        self.assertEqual(DEFAULT_HF_REPO_ID, "TassP0lar/Spark-X2.6-4B-Recursive72")
        self.assertEqual(
            normalize_hf_repo_id("Spark X2.6 4B Recursive72"),
            "TassP0lar/Spark-X2.6-4B-Recursive72",
        )
        fake_api = types.SimpleNamespace(whoami=lambda: {"name": "BaiTian6641"})
        self.assertEqual(
            normalize_hf_repo_id("Spark X2.6 4B Recursive72", api_client=fake_api),
            "BaiTian6641/Spark-X2.6-4B-Recursive72",
        )
        self.assertEqual(
            normalize_hf_repo_id("BaiTian6641/Spark X2.6 4B Recursive72"),
            "BaiTian6641/Spark-X2.6-4B-Recursive72",
        )
        cfg = MHCTrainingConfig()
        card = build_recursive72_model_card(cfg, {"stages": {}}, "BaiTian6641/Spark-X2.6-4B-Recursive72")
        self.assertIn("# Spark X2.6 4B Recursive72", card)
        self.assertIn("BaiTian6641/Spark-X2.6-4B-Recursive72", card)


@unittest.skipIf(torch is None, "torch not installed")
class TestMHCTrainingTorchContracts(unittest.TestCase):
    def test_full_size_controller_parameter_count_is_10078176(self):
        ctrl = TwoStreamMHC(
            width=2560,
            pass_adapter_rank=16,
            max_passes=4,
            gain_mode="sigmoid",
            init_p2_gain=0.01,
            init_p3_gain=0.005,
        )
        per_ctrl = sum(p.numel() for p in ctrl.parameters())
        self.assertEqual(per_ctrl, 419_924)
        self.assertEqual(per_ctrl * 24, EXPECTED_DEFAULT_CONTROLLER_PARAMS)

    def test_bounded_sigmoid_gain_preserves_birkhoff_polytope_under_extreme_logits(self):
        ctrl = TwoStreamMHC(
            width=16,
            pass_adapter_rank=4,
            max_passes=4,
            gain_mode="sigmoid",
            init_p2_gain=0.01,
            init_p3_gain=0.005,
        )
        telem_init = ctrl.controller_telemetry()
        self.assertEqual(telem_init["gain_mode"], "sigmoid")
        self.assertEqual(telem_init["schema_version"], 2)
        self.assertAlmostEqual(telem_init["gain_p2_attn"], 0.01, places=5)
        self.assertAlmostEqual(telem_init["gain_p3_attn"], 0.005, places=5)
        self.assertEqual(telem_init["gains_outside_unit_interval"], [])
        self.assertGreaterEqual(min(telem_init["static_effective_mix_min_entry"].values()), 0.0)

        with torch.no_grad():
            ctrl.extra_gain.fill_(-12.0)
            ctrl.ffn_extra_gain.fill_(12.0)
            ctrl.p3_extra_gain.fill_(-8.0)
            ctrl.p3_ffn_extra_gain.fill_(8.0)
            ctrl.mix_logits.copy_(torch.tensor([[0.0, 6.0], [6.0, 0.0]]))

        telem_extreme = ctrl.controller_telemetry()
        self.assertEqual(telem_extreme["gains_outside_unit_interval"], [])
        self.assertGreaterEqual(min(telem_extreme["static_effective_mix_min_entry"].values()), 0.0)

        # Gate multiplier = 0.0 forces exact zero gain on Pass 2/3
        ctrl.set_gain_mode("sigmoid", extra_pass_gate_multiplier=0.0, reset_gains=False)
        telem_zero = ctrl.controller_telemetry()
        self.assertEqual(telem_zero["gain_p2_attn"], 0.0)
        self.assertEqual(telem_zero["gain_p3_attn"], 0.0)

    def test_staged_unfreezing_and_optimizer_groups(self):
        model = _make_tiny_training_loop(gain_mode="sigmoid")
        cfg = MHCTrainingConfig(
            hardware_profile="cpu_test",
            adapter_lr=2e-4,
            routing_lr=3e-5,
            weight_decay=0.01,
        )
        prepare_model_for_mhc_training(model, cfg=cfg, reset_gains=True)

        p1 = configure_controller_unfreeze_phase(model, "warmup_gains_and_up")
        p2 = configure_controller_unfreeze_phase(model, "adapters_and_static_routing")
        p3 = configure_controller_unfreeze_phase(model, "full_controllers")
        self.assertGreater(p1["active_trainable_parameters"], 0)
        self.assertGreater(p2["active_trainable_parameters"], p1["active_trainable_parameters"])
        self.assertEqual(p3["active_trainable_parameters"], p3["total_controller_parameters"])

        opt, summary = build_mhc_optimizer(model, cfg=cfg)
        self.assertEqual(summary["total_optimizer_params"], p3["total_controller_parameters"])
        self.assertEqual(len(opt.param_groups), 3)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 2e-4)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 3e-5)
        self.assertAlmostEqual(opt.param_groups[2]["lr"], 3e-5)
        self.assertEqual(opt.param_groups[2]["weight_decay"], 0.0)

        lrs_start = update_cosine_lr(opt, step=0, total_steps=100, warmup_ratio=0.10, min_lr_ratio=0.10)
        lrs_end = update_cosine_lr(opt, step=100, total_steps=100, warmup_ratio=0.10, min_lr_ratio=0.10)
        self.assertLess(lrs_end["adapter_group"], 2e-4)
        self.assertGreaterEqual(lrs_end["adapter_group"], 2e-5 - 1e-7)
        self.assertGreater(lrs_start["adapter_group"], 0.0)

    def test_sampled_position_teacher_retention_kl_and_stage_a_verification(self):
        model = _make_tiny_training_loop(gain_mode="sigmoid")
        cfg = MHCTrainingConfig(
            hardware_profile="cpu_test",
            seq_len=24,
            sampled_kl_positions=8,
            kl_beta=0.5,
            kl_temperature=1.5,
        )
        stage_a = run_stage_a_verification(model, cfg=cfg, vocab_size=64, seq_len=24)
        self.assertTrue(stage_a["passed"], msg=json.dumps(stage_a, indent=2, default=str))
        self.assertEqual(stage_a["execution_manifest"]["mhc_gain_mode"], "sigmoid")
        self.assertEqual(stage_a["execution_manifest"]["mhc_schema_version"], 2)
        self.assertEqual(stage_a["save_load_max_logit_diff"], 0.0)

        labels = torch.full((1, 24), -100, dtype=torch.long)
        labels[0, 10:20] = torch.arange(10, 20, dtype=torch.long)
        pos_idx, pos_mask = sample_supervised_positions(labels, max_positions=6)
        self.assertEqual(tuple(pos_idx.shape), (1, 6))
        self.assertTrue(bool(pos_mask.all().item()))
        for idx_val in pos_idx[0].tolist():
            self.assertGreaterEqual(idx_val, 9)
            self.assertLessEqual(idx_val, 18)

    def test_end_to_end_stages_a_to_e_and_loader_provenance_verification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = _make_tiny_training_loop(gain_mode="sigmoid")
            cfg = MHCTrainingConfig(
                output_dir=tmp_dir,
                hardware_profile="cpu_test",
                seq_len=16,
                grad_accum_steps=1,
                sampled_kl_positions=4,
                stage_b_steps=1,
                stage_c_steps=2,
                stage_d_steps=1,
                run_stage_d_omp=True,
                run_stage_e_dlsa=True,
                stage_e_calibration_seqs=1,
                stage_e_seq_len=16,
                stage_e_distill_steps=2,
                seed=11,
            )

            class _TinyTok:
                def encode(self, txt: str, add_special_tokens: bool = False):
                    return [max(1, ord(c) % 60) for c in txt]

            rep = run_mhc_training_pipeline(model, _TinyTok(), cfg=cfg)
            self.assertIn("stage_a", rep["stages"])
            self.assertIn("stage_b", rep["stages"])
            self.assertIn("stage_c", rep["stages"])
            self.assertIn("stage_d", rep["stages"])
            self.assertIn("stage_e", rep["stages"])

            stage_c_ckpt = pathlib.Path(tmp_dir) / "stage_c_dense_mhc" / "loop_v03_state.pt"
            stage_d_ckpt = pathlib.Path(tmp_dir) / "stage_d_omp_mhc" / "loop_v03_state.pt"
            stage_e_sel = pathlib.Path(tmp_dir) / "visit_selectors_stage_e.pt"
            final_ckpt = pathlib.Path(tmp_dir) / "final_loop72_mhc_dlsa_export" / "loop_v03_state.pt"
            self.assertTrue(stage_c_ckpt.is_file())
            self.assertTrue(stage_d_ckpt.is_file())
            self.assertTrue(stage_e_sel.is_file())
            self.assertTrue(final_ckpt.is_file())

            prov = rep["stages"]["stage_e"]["provenance"]
            expected_d_sha = hashlib.sha256(stage_d_ckpt.read_bytes()).hexdigest()
            self.assertEqual(prov["mhc_checkpoint_sha256"], expected_d_sha)
            self.assertEqual(prov["recurrence_mode"], "learned_mhc")
            self.assertEqual(prov["mhc_gain_mode"], "sigmoid")
            self.assertEqual(prov["mhc_schema_version"], 2)

            # Verify load_inference_v03 restores both the sigmoid mHC schema and selector provenance
            restored = _make_tiny_training_loop(gain_mode="legacy_tanh")
            loaded_state = torch.load(final_ckpt, map_location="cpu", weights_only=False)
            load_info = load_loop_v03_state_dict(restored, loaded_state, strict=True)
            self.assertEqual(load_info["mhc_gain_mode"], "sigmoid")
            self.assertEqual(load_info["mhc_schema_version"], 2)
            self.assertEqual(restored.settings.mhc_gain_mode, "sigmoid")
            self.assertEqual(restored.settings.mhc_schema_version, 2)

            # Also verify VisitAwareDLSAIndexBank restores provenance from visit_selectors_stage_e.pt
            sel_payload = torch.load(stage_e_sel, map_location="cpu", weights_only=False)
            bank = VisitAwareDLSAIndexBank(
                base_cfg=DLSAIndexConfig(n_kv=2, head_dim=4, rope_dim=1, d_idx=6),
                maps_per_global_visit=True,
            )
            bank_info = bank.load_from_legacy_or_visit_checkpoint(sel_payload)
            self.assertEqual(bank_info["provenance"]["mhc_checkpoint_sha256"], expected_d_sha)
            self.assertEqual(bank_info["provenance"]["recurrence_mode"], "learned_mhc")

            # Verify Hugging Face Recursive72 export artifacts & mock Hub upload
            export_dir = pathlib.Path(tmp_dir) / "final_loop72_mhc_dlsa_export"
            readme_file = export_dir / "README.md"
            cfg_file = export_dir / "loop_config_v03.json"
            self.assertTrue(readme_file.is_file())
            self.assertIn("# Spark X2.6 4B Recursive72", readme_file.read_text(encoding="utf-8"))
            cfg_json = json.loads(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(cfg_json["model_name"], "Spark X2.6 4B Recursive72")

            created_repos = []
            uploaded_folders = []

            class _FakeHfApi:
                def whoami(self):
                    return {"name": "BaiTian6641"}

                def create_repo(self, **kwargs):
                    created_repos.append(kwargs)

                def upload_folder(self, **kwargs):
                    uploaded_folders.append(kwargs)

            cfg.push_to_hub = True
            cfg.hf_repo_id = "Spark X2.6 4B Recursive72"
            hub_res = export_and_upload_recursive72_to_hub(
                model,
                _TinyTok(),
                export_dir=export_dir,
                cfg=cfg,
                report=rep,
                api_client=_FakeHfApi(),
            )
            self.assertTrue(hub_res["uploaded"])
            self.assertEqual(hub_res["model_name"], "Spark X2.6 4B Recursive72")
            self.assertEqual(hub_res["hf_repo_id"], "BaiTian6641/Spark-X2.6-4B-Recursive72")
            self.assertEqual(created_repos[0]["repo_id"], "BaiTian6641/Spark-X2.6-4B-Recursive72")
            self.assertEqual(uploaded_folders[0]["repo_id"], "BaiTian6641/Spark-X2.6-4B-Recursive72")


if __name__ == "__main__":
    unittest.main()
