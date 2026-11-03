"""One-call loader for Spark-X2.6-4B-En4B-A1B (SparkX-Loop v0.3: 8.34B Total / 5.33B Active, 72 Effective Layers)."""
from __future__ import annotations
import dataclasses
import json
import os
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from sparkx_loop_v03_core import (
    LoopV03Settings,
    SparkXLoopV03,
    count_en4b_a1b_params,
    load_loop_v03_state_dict,
    patch_spark_attention_for_loglinear_kv,
    patch_spark_for_unsloth_triton,
)


def _patch_transformers_v5_tied_weights() -> None:
    """Ensure `modeling_spark.py` (`_tied_weights_keys = ['lm_head.weight']`) works seamlessly on both
    Transformers 4.x (list format) and Transformers 5.x (dict mapping format).
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if hasattr(PreTrainedModel, 'get_expanded_tied_weights_keys') and not getattr(PreTrainedModel, '_sparkx_tied_patched', False):
        orig_fn = PreTrainedModel.get_expanded_tied_weights_keys

        def _safe_get_expanded_tied_weights_keys(self, *args, **kwargs):
            try:
                return orig_fn(self, *args, **kwargs)
            except AttributeError:
                tw = getattr(self, '_tied_weights_keys', None)
                if isinstance(tw, (list, tuple, set)):
                    mapped = {k: ('model.embedding.weight' if k == 'lm_head.weight' else k) for k in tw}
                    try:
                        setattr(type(self), '_tied_weights_keys', mapped)
                    except Exception:
                        pass
                    self._tied_weights_keys = mapped
                    return orig_fn(self, *args, **kwargs)
                raise

        PreTrainedModel.get_expanded_tied_weights_keys = _safe_get_expanded_tied_weights_keys
        PreTrainedModel._sparkx_tied_patched = True

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        _orig_act = getattr(PreTrainedTokenizerBase, 'apply_chat_template', None)
        if callable(_orig_act) and not getattr(_orig_act, '_sparkx_v5_patched', False):
            def _patched_apply_chat_template(self, *args, **kwargs):
                want_dict = bool(kwargs.get('return_dict', False))
                res = _orig_act(self, *args, **kwargs)
                if not want_dict:
                    if hasattr(res, 'input_ids'):
                        return res.input_ids
                    if isinstance(res, dict) and 'input_ids' in res:
                        return res['input_ids']
                return res
            _patched_apply_chat_template._sparkx_v5_patched = True
            PreTrainedTokenizerBase.apply_chat_template = _patched_apply_chat_template
    except Exception:
        pass


def load_sparkx_loop_v03(
    repo_or_dir: str = 'TassP0lar/Spark-X2.6-4B-En4B-A1B',
    repo_id: Optional[str] = None,
    base_repo: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.bfloat16,
    enable_loglinear_kv: bool = True,
    loglinear_exact_threshold: int = 32768,
    loglinear_recent_window: int = 4096,
    loglinear_sink_tokens: int = 32,
    offload_tier1b_cpu: bool = True,
    use_pristine_base: bool = True,
    token: Optional[str] = None,
    return_meta: bool = False,
):
    """Load the standalone Spark-X2.6-4B-En4B-A1B (v0.3) checkpoint (72 effective layers, 8.23B total / 5.23B active)."""
    _patch_transformers_v5_tied_weights()
    if repo_id is not None:
        repo_or_dir = repo_id
    if token is None:
        token = os.environ.get('HF_TOKEN') or None
        if token is None:
            try:
                from google.colab import userdata
                token = userdata.get('HF_TOKEN')
            except Exception:
                pass

    if os.path.isdir(repo_or_dir):
        cfg_path = os.path.join(repo_or_dir, 'loop_config_v03.json')
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(repo_or_dir, 'loop_config.json')
        state_path = os.path.join(repo_or_dir, 'loop_v03_state.pt')
        if not os.path.exists(state_path):
            state_path = os.path.join(repo_or_dir, 'loop_state.pt')
    else:
        try:
            cfg_path = hf_hub_download(repo_or_dir, 'loop_config_v03.json', token=token)
        except Exception:
            cfg_path = hf_hub_download(repo_or_dir, 'loop_config.json', token=token)
        try:
            state_path = hf_hub_download(repo_or_dir, 'loop_v03_state.pt', token=token)
        except Exception:
            state_path = hf_hub_download(repo_or_dir, 'loop_state.pt', token=token)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    state = torch.load(state_path, map_location='cpu')
    valid_fields = {f.name for f in dataclasses.fields(LoopV03Settings)}
    raw_settings = dict(state.get('settings', meta.get('settings', {})))
    settings_dict = {k: v for k, v in raw_settings.items() if k in valid_fields}
    settings_dict['checkpoint_layers'] = False
    settings_dict['offload_checkpoint_to_cpu'] = False
    settings_dict['eng_offload_tier1b_cpu'] = offload_tier1b_cpu
    settings = LoopV03Settings(**settings_dict)

    has_merged_backbone = bool(meta.get('merged_standalone', True)) and not bool(use_pristine_base)
    backbone_source = repo_or_dir if has_merged_backbone else (base_repo or meta.get('base_model', 'XHToken/Spark-X2.6-4B'))

    print(f"[Loader] Loading tokenizer & standalone 36L backbone from `{backbone_source}` (+ mHC/Engram state from `{repo_or_dir}`)...")
    try:
        tok = AutoTokenizer.from_pretrained(backbone_source, trust_remote_code=True, token=token)
    except Exception:
        tok = AutoTokenizer.from_pretrained(repo_or_dir, trust_remote_code=True, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        backbone_source,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        token=token,
    ).to(device)
    patch_spark_for_unsloth_triton(base)
    base._sparkx_is_pristine_base = not has_merged_backbone

    model = SparkXLoopV03(base, settings, patch_sdpa=not enable_loglinear_kv).to(device)
    model.controllers.to(device=device)
    model.engram.to(device=device)
    model.backbone_engrams.to(device=device)

    load_loop_v03_state_dict(model, state, strict=False)
    active_meta = meta.get('active_meta', state.get('active_meta', {}))
    act_e = int(active_meta.get('engram_active_experts', 32))
    act_k = int(active_meta.get('engram_active_top_k', 7))
    for mod in model.all_engram_modules():
        mod.active_experts = min(act_e, mod.num_experts)
        mod.active_top_k = min(act_k, mod.active_experts)

    stage = meta.get('stage', {})
    model.set_loop(
        start=int(stage.get('start', settings.start)),
        end=int(stage.get('end', settings.end)),
        passes=int(stage.get('passes', settings.passes)),
        use_mhc=bool(stage.get('mhc', settings.use_mhc)),
        use_engram=bool(stage.get('engram', settings.use_engram)),
        enable_p3=bool(stage.get('enable_p3', settings.enable_p3)),
        p3_start=int(stage.get('p3_start', settings.p3_start)),
        p3_end=int(stage.get('p3_end', settings.p3_end)),
    )

    if enable_loglinear_kv:
        patch_spark_attention_for_loglinear_kv(
            model.base,
            exact_threshold=loglinear_exact_threshold,
            recent_window=loglinear_recent_window,
            sink_tokens=loglinear_sink_tokens,
        )
        model.base._sparkx_exact_kv_32k = bool(loglinear_exact_threshold >= 32768)

    model.loaded_meta = meta
    model.eval()
    p_stats = count_en4b_a1b_params(model)
    m_info = meta.get('metrics', {})
    print(
        f"[Loader Ready] Spark-X2.6-4B-En4B-A1B ({meta.get('curriculum_version', 'v0.3')}, step={meta.get('global_step')})\n"
        f"  Effective Depth : {model.total_virtual_layers}L (passes={model.loop_passes}, enable_p3={model.enable_p3})\n"
        f"  Total / Active  : {p_stats['model_total_params']/1e9:.2f}B Total / {p_stats['model_active_params']/1e9:.2f}B Active "
        f"(Engram: {model.engram.active_experts} experts, Top-{model.engram.active_top_k})\n"
        f"  Final Train Loss: ce_loss={m_info.get('ce_loss')}, z_loss={m_info.get('z_loss')}, "
        f"mHC_gain={m_info.get('mhc_gain')} (p2={m_info.get('mhc_p2_gain')}, p3={m_info.get('mhc_p3_gain')})"
    )
    if return_meta:
        return model, tok, meta
    return model, tok

