"""One-call loader for Spark-X2.6-4B-En4B-A1B (SparkX-Loop v0.3: 8.2B Total / 5.2B Active, 72 Effective Layers)."""
from __future__ import annotations
import json
import os
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from sparkx_loop_v03_core import (
    LoopV03Settings,
    SparkXLoopV03,
    load_loop_v03_state_dict,
    patch_spark_attention_for_loglinear_kv,
)


def load_sparkx_loop_v03(
    repo_or_dir: str = 'TassP0lar/Spark-X2.6-4B-En4B-A1B',
    base_repo: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.bfloat16,
    enable_loglinear_kv: bool = False,
    loglinear_recent_window: int = 1024,
    loglinear_sink_tokens: int = 16,
    offload_tier1b_cpu: bool = True,
    token: Optional[str] = None,
):
    """Load a Spark-X2.6-4B-En4B-A1B (v0.3) checkpoint (72 effective layers, 8.2B total / 5.2B active)."""
    if os.path.isdir(repo_or_dir):
        cfg_path = os.path.join(repo_or_dir, 'loop_config.json')
        state_path = os.path.join(repo_or_dir, 'loop_v03_state.pt')
        if not os.path.exists(state_path):
            state_path = os.path.join(repo_or_dir, 'loop_state.pt')
    else:
        cfg_path = hf_hub_download(repo_or_dir, 'loop_config.json', token=token)
        try:
            state_path = hf_hub_download(repo_or_dir, 'loop_v03_state.pt', token=token)
        except Exception:
            state_path = hf_hub_download(repo_or_dir, 'loop_state.pt', token=token)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    state = torch.load(state_path, map_location='cpu')
    settings_dict = dict(state.get('settings', meta.get('settings', {})))
    settings_dict['checkpoint_layers'] = False
    settings_dict['offload_checkpoint_to_cpu'] = False
    settings_dict['eng_offload_tier1b_cpu'] = offload_tier1b_cpu
    settings = LoopV03Settings(**settings_dict)

    has_merged_backbone = bool(meta.get('merged_standalone', True))
    backbone_source = repo_or_dir if has_merged_backbone else (base_repo or meta.get('base_model', 'XHToken/Spark-X2.6-4B'))

    tok = AutoTokenizer.from_pretrained(repo_or_dir, trust_remote_code=True, token=token)
    base = AutoModelForCausalLM.from_pretrained(
        backbone_source,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        token=token,
    ).to(device)

    model = SparkXLoopV03(base, settings, patch_sdpa=not enable_loglinear_kv).to(device)
    load_loop_v03_state_dict(model, state, strict=False)

    stage = meta.get('stage', {})
    if stage:
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
            recent_window=loglinear_recent_window,
            sink_tokens=loglinear_sink_tokens,
        )

    model.eval()
    return model, tok, meta
