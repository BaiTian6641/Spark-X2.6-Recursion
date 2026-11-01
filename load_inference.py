"""SparkX-Loop v0.2 inference from either a Formal Merged Hugging Face Checkpoint
(`TassP0lar/Spark-X2.6-4B-Loop-mHC`) or a stage export (`TassP0lar/Spark-X2.6-4B-Loop-mHC-R24`).

Supports:
- Per-visit KV-cached looped decoding (`generate_cached` via `_VisitCache`)
- Runtime-only O(n log n) Log-Linear NoPE KV Cache (`LOGLINEAR_KV=1`)
- Spark X2.6 native `<think>...</think>` reasoning mode (`ENABLE_THINKING=1`)

Usage:
    pip install torch transformers==4.57.1 accelerate safetensors huggingface_hub
    HF_TOKEN=hf_xxx LOOP_REPO=TassP0lar/Spark-X2.6-4B-Loop-mHC python load_inference.py
"""
import json
import os
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparkx_loop_core import (
    LoopSettings, SparkXLoop, attach_lora, patch_spark_attention_for_loglinear_kv
)

repo = os.environ.get('LOOP_REPO', 'TassP0lar/Spark-X2.6-4B-Loop-mHC')
stage = os.environ.get('LOOP_STAGE', 'R24')
token = os.environ.get('HF_TOKEN') or os.environ.get('HF_TOKEM')
enable_thinking = os.environ.get('ENABLE_THINKING', '1').lower() not in ('0', 'false', 'no')
use_visit_cache = os.environ.get('USE_VISIT_CACHE', '1').lower() not in ('0', 'false', 'no')
enable_loglinear_kv = os.environ.get('LOGLINEAR_KV', '0').lower() in ('1', 'true', 'yes')
max_new_tokens = int(os.environ.get('MAX_NEW_TOKENS', '512' if enable_thinking else '128'))
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU required for default inference script')

root = Path(snapshot_download(repo_id=repo, token=token))
folder = (root / 'stages' / stage) if (root / 'stages' / stage / 'loop_config.json').exists() else root
meta = json.loads((folder / 'loop_config.json').read_text())
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

is_formal = bool(meta.get('is_formal_merged_checkpoint', False)) or (folder / 'loop_modules.safetensors').exists()
base_source = str(folder) if is_formal else meta['base_id']
base_rev = None if is_formal else meta.get('base_revision')

base = AutoModelForCausalLM.from_pretrained(
    base_source, revision=base_rev, trust_remote_code=True,
    torch_dtype=dtype, device_map={'': 'cuda:0'}, token=token, attn_implementation='eager'
)
tok = AutoTokenizer.from_pretrained(
    base_source, revision=base_rev, trust_remote_code=True, token=token
)
base.config.use_cache = False
if not is_formal and meta.get('lora_rank', 0) > 0:
    attach_lora(base, start=4, end=36, rank=meta['lora_rank'], alpha=meta['lora_alpha'])

settings = LoopSettings(
    start=4, end=28, passes=meta['loop_passes'],
    eng_buckets=meta['engram_buckets'], eng_dim=meta['engram_dim'],
    eng_experts=meta.get('engram_experts', 16),
    eng_top_k=meta.get('engram_top_k', 2),
    eng_expert_rank=meta.get('eng_expert_rank', meta.get('engram_expert_rank', 16)),
    eng_shared_rank=meta.get('eng_shared_rank', meta.get('engram_shared_rank', 16)),
    backbone_engram_stride=meta.get('backbone_engram_stride', 4),
    checkpoint_layers=False, dynamic_mhc=meta.get('dynamic_mhc', True),
    sublayer_mhc=meta.get('sublayer_mhc', True),
    pass_adapter_rank=meta.get('pass_adapter_rank', 16),
)
net = SparkXLoop(base, settings, patch_sdpa=True).to('cuda:0')
if enable_loglinear_kv:
    patch_spark_attention_for_loglinear_kv(base)

weights_file = (folder / 'loop_modules.safetensors') if (folder / 'loop_modules.safetensors').exists() else (folder / 'trainable.safetensors')
state = load_file(str(weights_file), device='cpu')
missing, unexpected = net.load_state_dict(state, strict=False)
assert not unexpected, unexpected
net.set_loop(meta['loop_start'], meta['loop_end'], meta['loop_passes'],
             use_mhc=meta['mhc_enabled'], use_engram=meta['engram_enabled'])
net.eval()

messages = [{'role': 'user', 'content': 'Explain why looped depth recurrence with sublayer manifold-constrained hyper-connections improves parameter efficiency and multi-step reasoning.'}]
ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                              enable_thinking=enable_thinking, reasoning_effort='high', return_tensors='pt')
if hasattr(ids, 'input_ids'):
    ids = ids.input_ids
elif isinstance(ids, dict):
    ids = ids['input_ids']
ids = ids.to('cuda:0')
if use_visit_cache:
    out = net.generate_cached(ids, max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id)
else:
    out = net.generate_slow(ids, max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=False))

