"""SparkX-Loop v0.2 inference from a Hugging Face stage export.

Supports both per-visit KV-cached looped decoding (`generate_cached` via `_VisitCache`)
and exact cache-free full-recompute decoding (`generate_slow`), plus Spark X2.6's native
`<think>...</think>` reasoning mode (`ENABLE_THINKING=1`) and 1M-context fused SDPA.

Usage (outside Colab after training):
    pip install torch transformers==4.57.1 accelerate safetensors huggingface_hub
    HF_TOKEN=hf_xxx LOOP_REPO=TassP0lar/Spark-X2.6-4B-Loop-mHC LOOP_STAGE=R24 python load_inference.py
"""
import json
import os
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparkx_loop_core import LoopSettings, SparkXLoop, attach_lora

repo=os.environ.get('LOOP_REPO','TassP0lar/Spark-X2.6-4B-Loop-mHC')
stage=os.environ.get('LOOP_STAGE','R24')
token=os.environ.get('HF_TOKEN') or os.environ.get('HF_TOKEM')
enable_thinking=os.environ.get('ENABLE_THINKING','1').lower() not in ('0','false','no')
use_visit_cache=os.environ.get('USE_VISIT_CACHE','1').lower() not in ('0','false','no')
max_new_tokens=int(os.environ.get('MAX_NEW_TOKENS','256' if enable_thinking else '64'))
if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for default inference script')
root=Path(snapshot_download(repo_id=repo,token=token,allow_patterns=[f'stages/{stage}/*']))
folder=root/'stages'/stage
meta=json.loads((folder/'loop_config.json').read_text())
dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base=AutoModelForCausalLM.from_pretrained(meta['base_id'],revision=meta['base_revision'],
    trust_remote_code=True,torch_dtype=dtype,device_map={'':'cuda:0'},token=token,attn_implementation='eager')
tok=AutoTokenizer.from_pretrained(meta['base_id'],revision=meta['base_revision'],trust_remote_code=True,token=token)
base.config.use_cache=False
if meta.get('lora_rank',0)>0:
    attach_lora(base,start=4,end=36,rank=meta['lora_rank'],alpha=meta['lora_alpha'])
settings=LoopSettings(
    start=4,end=28,passes=meta['loop_passes'],
    eng_buckets=meta['engram_buckets'],eng_dim=meta['engram_dim'],
    eng_experts=meta.get('engram_experts',8),
    eng_top_k=meta.get('engram_top_k',2),
    eng_expert_rank=meta.get('engram_expert_rank',16),
    eng_shared_rank=meta.get('engram_shared_rank',16),
    backbone_engram_stride=meta.get('backbone_engram_stride',4),
    checkpoint_layers=False,dynamic_mhc=meta.get('dynamic_mhc',True),
    sublayer_mhc=meta.get('sublayer_mhc',True),
    pass_adapter_rank=meta.get('pass_adapter_rank',16),
)
net=SparkXLoop(base,settings,patch_sdpa=True).to('cuda:0')
state=load_file(str(folder/'trainable.safetensors'),device='cpu')
missing,unexpected=net.load_state_dict(state,strict=False)
assert not unexpected,unexpected
net.set_loop(meta['loop_start'],meta['loop_end'],meta['loop_passes'],
             use_mhc=meta['mhc_enabled'],use_engram=meta['engram_enabled'])
net.eval()
messages=[{'role':'user','content':'Explain why looped depth recurrence with sublayer manifold-constrained hyper-connections improves parameter efficiency and multi-step reasoning.'}]
ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
    enable_thinking=enable_thinking,reasoning_effort='high',return_tensors='pt')
if hasattr(ids,'input_ids'): ids=ids.input_ids
elif isinstance(ids,dict): ids=ids['input_ids']
ids=ids.to('cuda:0')
if use_visit_cache:
    out=net.generate_cached(ids,max_new_tokens=max_new_tokens,eos_token_id=tok.eos_token_id)
else:
    out=net.generate_slow(ids,max_new_tokens=max_new_tokens,eos_token_id=tok.eos_token_id)
print(tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=False))
