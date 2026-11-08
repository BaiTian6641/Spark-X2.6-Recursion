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
    patch_spark_attention_for_lsa,
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
    use_pristine_base: bool = False,
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
    base._sparkx_backbone_source = backbone_source

    model = SparkXLoopV03(base, settings, patch_sdpa=not enable_loglinear_kv).to(device)
    model.controllers.to(device=device)
    if model.engram is not None:
        model.engram.to(device=device)
    model.backbone_engrams.to(device=device)
    if hasattr(model, 'esm'):
        model.esm.to(device=device)

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
        patch_spark_attention_for_lsa(
            model.base,
            exact_threshold=loglinear_exact_threshold,
        )
        model.base._sparkx_exact_kv_32k = bool(loglinear_exact_threshold >= 32768)

    model.loaded_meta = meta
    model.eval()
    p_stats = count_en4b_a1b_params(model)
    m_info = meta.get('metrics', {})
    eng_desc = (
        f"Engram: {model.engram.active_experts} experts, Top-{model.engram.active_top_k}"
        if model.engram is not None
        else "Engram: disabled (0 params)"
    )
    print(
        f"[Loader Ready] Spark-X2.6-4B-En4B-A1B ({meta.get('curriculum_version', 'v0.3')}, step={meta.get('global_step')})\n"
        f"  Effective Depth : {model.total_virtual_layers}L (passes={model.loop_passes}, enable_p3={model.enable_p3})\n"
        f"  Total / Active  : {p_stats['model_total_params']/1e9:.2f}B Total / {p_stats['model_active_params']/1e9:.2f}B Active "
        f"({eng_desc})\n"
        f"  Final Train Loss: ce_loss={m_info.get('ce_loss')}, z_loss={m_info.get('z_loss')}, "
        f"mHC_gain={m_info.get('mhc_gain')} (p2={m_info.get('mhc_p2_gain')}, p3={m_info.get('mhc_p3_gain')})"
    )
    if return_meta:
        return model, tok, meta
    return model, tok


def load_loop72_lsa_yaml_config(config_path: Optional[str] = None) -> dict:
    """Load `configs/spark4b_loop72_lsa.yaml` (using PyYAML if available, or a built-in minimal indentation parser)."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'spark4b_loop72_lsa.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        import yaml  # type: ignore
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    def _parse_scalar(val: str):
        v = val.strip()
        if v.startswith('[') and v.endswith(']'):
            inner = v[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(part) for part in inner.split(',')]
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        low = v.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        if low in ('null', 'none'):
            return None
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line in raw.splitlines():
        clean = line.split('#', 1)[0].rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(' '))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        cur_dict = stack[-1][1]
        if ':' not in clean:
            continue
        k_part, v_part = clean.strip().split(':', 1)
        key = _parse_scalar(k_part)
        if not v_part.strip():
            sub: dict = {}
            cur_dict[key] = sub
            stack.append((indent, sub))
        else:
            cur_dict[key] = _parse_scalar(v_part)
    return root


def configure_model_for_loop72_lsa(
    model: SparkXLoopV03,
    config: Optional[dict] = None,
    dlsa_checkpoint: Optional[str] = None,
) -> dict:
    """Configure an existing `SparkXLoopV03` instance for the Loop72 + Per-Visit Paged D-LSA runtime."""
    from sparkx_dlsa import DLSAIndexConfig, VisitAwareDLSAIndexBank
    from sparkx_paged_cache import PagedCacheConfig
    from sparkx_sparse_attention import SparseAttentionConfig, attach_loop72_dlsa_attention

    cfg = config if isinstance(config, dict) else load_loop72_lsa_yaml_config()
    model_cfg = cfg.get('model', {})
    loop_cfg = cfg.get('loop', {})
    engram_cfg = cfg.get('engram', {})
    attn_cfg = cfg.get('attention', cfg.get('dlsa', {}))
    cache_cfg = cfg.get('cache', cfg.get('runtime', {}))
    prefill_cfg = cfg.get('prefill', cfg.get('runtime', {}))
    mem_cfg = cfg.get('memory', {})

    preserve_native = bool(model_cfg.get('preserve_native_activation', True))
    runtime_identity = str(model_cfg.get('runtime_identity', 'native_gelu_loop72'))
    model.settings.preserve_native_activation = preserve_native
    model.settings.runtime_identity = runtime_identity
    model.settings.use_paged_hybrid_cache = True
    model.settings.prefill_chunk_tokens = int(prefill_cfg.get('hidden_chunk_tokens', 2048))
    model.settings.ce_chunk_tokens = int(prefill_cfg.get('ce_chunk_tokens', 2048))

    patch_spark_for_unsloth_triton(model.base, preserve_native_activation=preserve_native)

    model.set_loop(
        start=int(loop_cfg.get('start', 4)),
        end=int(loop_cfg.get('end_exclusive', 28)),
        passes=int(loop_cfg.get('passes', loop_cfg.get('core_passes', 2))),
        use_mhc=bool(loop_cfg.get('mhc_enabled', True)),
        use_engram=bool(engram_cfg.get('load', mem_cfg.get('engram_enabled', False))),
        enable_p3=bool(loop_cfg.get('enable_p3', True)),
        p3_start=int(loop_cfg.get('p3_start', 12)),
        p3_end=int(loop_cfg.get('p3_end_exclusive', 24)),
    )
    model.esm_enabled = bool(mem_cfg.get('m3_enabled', mem_cfg.get('esm_enabled', True)))
    if hasattr(model, 'esm'):
        esm_gpu_b = int(mem_cfg.get('gpu_resident_budget_bytes', int(float(mem_cfg.get('esm_gpu_budget_mb', 256)) * 1024 * 1024)))
        esm_disk_b = int(mem_cfg.get('persistent_shard_budget_bytes', int(float(mem_cfg.get('esm_persistent_budget_gb', 1.0)) * (1024**3))))
        for esm_mod in model.esm.values():
            if hasattr(esm_mod, 'gpu_resident_budget_bytes'):
                esm_mod.gpu_resident_budget_bytes = esm_gpu_b
            if hasattr(esm_mod, 'persistent_shard_budget_bytes'):
                esm_mod.persistent_shard_budget_bytes = esm_disk_b

    pool_sz = int(attn_cfg.get('pool_size', attn_cfg.get('n_pool', 4)))
    cluster_blk = int(attn_cfg.get('cluster_block_tokens', attn_cfg.get('n_cluster', 512)))
    avg_b_tok = int(attn_cfg.get('average_bucket_tokens', 16))
    k_clust = int(attn_cfg.get('k_clusters', max(4, cluster_blk // max(1, avg_b_tok))))
    base_idx_cfg = DLSAIndexConfig(
        d_idx=int(attn_cfg.get('index_dim', attn_cfg.get('d_idx', 64))),
        n_pool=pool_sz,
        n_cluster=cluster_blk,
        k_clusters=k_clust,
        sinks=int(attn_cfg.get('sinks', 4)),
        recent_tokens=int(attn_cfg.get('recent_tokens', 512)),
        max_selected_keys=int(attn_cfg.get('long_range_tokens', attn_cfg.get('max_selected_keys_default', 2048))),
        exact_threshold=int(attn_cfg.get('exact_threshold', 2048)),
        search_mode=str(attn_cfg.get('search', 'flat')),
        branching_factor=int(attn_cfg.get('tree_branching_factor', 16)),
        beam_width=int(attn_cfg.get('tree_beam_width', 32)),
        residual_block_compensation=bool(attn_cfg.get('residual_compensation', attn_cfg.get('residual_block_compensation', False))),
        persistent_token_latents=bool(cache_cfg.get('persistent_token_latents', attn_cfg.get('persistent_token_latents', False))),
    )
    per_layer_budgets = attn_cfg.get('per_visit_budget_overrides') or attn_cfg.get('per_layer_max_selected_keys', {3: 3072, 31: 1536, 35: 1536})
    per_visit_budgets: dict = {}
    if isinstance(per_layer_budgets, dict):
        for k_lyr, v_bud in per_layer_budgets.items():
            try:
                per_visit_budgets[int(k_lyr)] = int(v_bud)
            except Exception:
                pass

    device = next(model.base.parameters()).device
    bank = VisitAwareDLSAIndexBank(
        base_cfg=base_idx_cfg,
        maps_per_global_visit=bool(attn_cfg.get('maps_per_global_visit', True)),
        per_visit_max_selected_keys=per_visit_budgets,
        enable_p3=bool(model.enable_p3),
        loop_passes=int(model.loop_passes),
        loop_start=int(model.loop_start),
        loop_end=int(model.loop_end),
        p3_start=int(model.p3_start),
        p3_end=int(model.p3_end),
        n_layers=int(model.n_layers),
    ).to(device)

    if dlsa_checkpoint is not None and os.path.exists(dlsa_checkpoint):
        ckpt_data = torch.load(dlsa_checkpoint, map_location='cpu')
        bank.load_from_checkpoint_dict(ckpt_data)

    sparse_cfg = SparseAttentionConfig(
        exact_threshold=int(attn_cfg.get('exact_threshold', 2048)),
        sparse_query_tile_tokens=int(prefill_cfg.get('sparse_query_tile_tokens', 16)),
        residual_block_compensation=bool(attn_cfg.get('residual_compensation', False)),
        local_window=int(cache_cfg.get('local_window', cache_cfg.get('local_ring_tokens', 512))),
    )
    model.paged_cache_cfg = PagedCacheConfig(
        local_window=int(cache_cfg.get('local_window', cache_cfg.get('local_ring_tokens', 512))),
        sinks=int(attn_cfg.get('sinks', 4)),
        recent_tokens=int(attn_cfg.get('recent_tokens', 512)),
        global_hot_tokens_per_visit=int(cache_cfg.get('global_hot_tokens_per_visit', 8192)),
        kv_page_tokens=int(cache_cfg.get('kv_page_tokens', 64)),
        cold_codec=str(cache_cfg.get('cold_codec', 'bf16')),
        index_gpu_budget_bytes=int(cache_cfg.get('index_gpu_budget_bytes', 2 * (1024**3))),
        persistent_token_latents=bool(cache_cfg.get('persistent_token_latents', False)),
        context_limit_tokens=int(cache_cfg.get('context_limit_tokens', model_cfg.get('context_limit_tokens', 131072))),
        model_revision=str(model_cfg.get('base', model_cfg.get('backbone_id', 'XHToken/Spark-X2.6-4B'))),
        behavior_fingerprint=runtime_identity,
    )
    attach_info = attach_loop72_dlsa_attention(model, index_bank=bank, sparse_cfg=sparse_cfg)
    return {
        'manifest': model.execution_manifest(),
        'attach_info': attach_info,
        'dlsa_map_parameters': bank.total_map_parameters(),
    }


def load_sparkx_loop72_lsa(
    repo_or_dir: str = 'TassP0lar/Spark-X2.6-4B-En4B-A1B',
    base_repo: str = 'XHToken/Spark-X2.6-4B',
    config_path: Optional[str] = None,
    dlsa_checkpoint: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: torch.dtype = torch.bfloat16,
    use_pristine_base: bool = False,
    token: Optional[str] = None,
    return_meta: bool = False,
):
    """Load the Spark-X2.6-4B-Loop72-LSA architecture (72 visits = 18 global + 54 local, native GELU, 0 Engram params)."""
    _patch_transformers_v5_tied_weights()
    cfg = load_loop72_lsa_yaml_config(config_path)
    if token is None:
        token = os.environ.get('HF_TOKEN') or None

    meta: dict = {}
    state: Optional[dict] = None
    if os.path.isdir(repo_or_dir):
        for cname in ('loop_config_v03.json', 'loop_config.json'):
            p = os.path.join(repo_or_dir, cname)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                break
        for sname in ('loop_v03_state.pt', 'loop_state.pt'):
            p = os.path.join(repo_or_dir, sname)
            if os.path.exists(p):
                state = torch.load(p, map_location='cpu')
                break
    else:
        try:
            cfg_p = hf_hub_download(repo_or_dir, 'loop_config_v03.json', token=token)
            with open(cfg_p, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            pass
        try:
            st_p = hf_hub_download(repo_or_dir, 'loop_v03_state.pt', token=token)
            state = torch.load(st_p, map_location='cpu')
        except Exception:
            pass

    has_merged_backbone = bool(meta.get('merged_standalone', True)) and not bool(use_pristine_base)
    backbone_source = repo_or_dir if has_merged_backbone else base_repo

    try:
        tok = AutoTokenizer.from_pretrained(backbone_source, trust_remote_code=True, token=token)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_repo, trust_remote_code=True, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        backbone_source,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        token=token,
    ).to(device)

    settings = LoopV03Settings.for_loop72_lsa(checkpoint_layers=False, offload_checkpoint_to_cpu=False)
    model = SparkXLoopV03(base, settings, patch_sdpa=True).to(device)
    model.controllers.to(device=device)
    if hasattr(model, 'esm'):
        model.esm.to(device=device)

    if state is not None:
        load_loop_v03_state_dict(model, state, strict=False)

    loop72_info = configure_model_for_loop72_lsa(model, config=cfg, dlsa_checkpoint=dlsa_checkpoint)
    model.loaded_meta = {**meta, 'loop72_lsa': loop72_info}
    model.eval()
    if return_meta:
        return model, tok, model.loaded_meta
    return model, tok

