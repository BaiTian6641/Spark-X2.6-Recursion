"""Offline CPU structural smoke tests for SparkX-Loop v0.2 (Sublayer Dual-mHC, Pass-Indexed
Adapters, Multi-Scale Causal Engram, Chunked Masked Cross-Entropy, and _VisitCache).
"""
import torch
import torch.nn as nn
from sparkx_loop_core import (
    LoopSettings, SparkXLoop, _VisitCache, attach_lora, set_trainables,
    optimizer_groups, count_params, fused_masked_cross_entropy
)


class FakeAttn(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_k_v_proj = nn.Linear(width, width * 3)
        self.g_proj = nn.Linear(width, 4)
        self.out_proj = nn.Linear(width, width)

    def forward(self, hidden_states, **kw):
        att = self.q_k_v_proj(hidden_states)[..., :hidden_states.shape[-1]]
        gate = torch.sigmoid(self.g_proj(hidden_states).mean(dim=-1, keepdim=True))
        return self.out_proj(torch.tanh(att) * gate), None


class FakeMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(width, width * 2)
        self.up_proj = nn.Linear(width, width * 2)
        self.down_proj = nn.Linear(width * 2, width)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class FakeLayer(nn.Module):
    layer_type = 'full_attention'

    def __init__(self, width):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.self_attn = FakeAttn(width)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.mlp = FakeMLP(width)

    def forward(self, h, **kw):
        residual = h
        z, _ = self.self_attn(self.input_layernorm(h), **kw)
        h = residual + z
        residual = h
        h = residual + self.mlp(self.post_attention_layernorm(h))
        return h


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 32)
        self.layers = nn.ModuleList([FakeLayer(32) for _ in range(8)])
        self.norm = nn.LayerNorm(32)

    def make_loop_context(self, embed, mask):
        return {'full_attention': None}, {'full_attention': None}


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeModel()
        self.lm_head = nn.Linear(32, 64, bias=False)
        self.lm_head.weight = self.model.embedding.weight
        self.config = type('C', (), {'hidden_size': 32, 'tie_word_embeddings': True, 'layer_types': ['full_attention'] * 8})()

    def original(self, ids):
        h = self.model.embedding(ids).float()
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(self.model.norm(h))


torch.manual_seed(42)
base = FakeBase()
changed = attach_lora(base, 4, 8, rank=4)
assert len(changed) == 24  # 4 layers * (q_k_v_proj, out_proj, g_proj, gate_proj, up_proj, down_proj)
loop = SparkXLoop(
    base,
    LoopSettings(start=4, end=8, passes=2, checkpoint_layers=True, dynamic_mhc=True, sublayer_mhc=True, ce_chunk_tokens=4),
    strict=False,
    patch_sdpa=False,
)
assert loop.total_virtual_layers == 12
ids = torch.randint(1, 64, (1, 9))
loop.set_loop(4, 8, 1, False, False)
raw = base.original(ids)
wrapped = loop.forward_logits(ids)
assert torch.allclose(raw, wrapped, rtol=1e-6, atol=1e-6), (raw - wrapped).abs().max()

# Verify that extra_gain=0 + ffn_extra_gain=0 + zero-initialized dynamic_proj makes pass 2 an exact identity map over pass 1.
loop.set_loop(4, 8, 1, True, True)
pass1_logits = loop.forward_logits(ids)
loop.set_loop(4, 8, 2, True, True)
pass2_logits = loop.forward_logits(ids)
assert torch.allclose(pass1_logits, pass2_logits, rtol=1e-6, atol=1e-6), (pass1_logits - pass2_logits).abs().max()

# Verify _VisitCache redirects physical layer_idx to virtual_idx
class DummyCache:
    def __init__(self):
        self.seen = []
        self.layers = [f'slot_{i}' for i in range(12)]
    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.seen.append(layer_idx)
        return k, v

dc = DummyCache()
vc = _VisitCache(dc, physical_idx=4, virtual_idx=8)
vc.update('k', 'v', 4)
assert dc.seen == [8] and vc.layers[4] == 'slot_8'

params = set_trainables(loop, full_layer_ids=[6])
groups = optimizer_groups(loop)
assert len(groups) == 3 and groups[1]['weight_decay'] == 0.0 and len(groups[1]['params']) > 0
assert not base.model.layers[0].mlp.gate_proj.weight.requires_grad
assert base.model.layers[6].mlp.gate_proj.weight.requires_grad
assert not base.model.layers[7].mlp.gate_proj.weight.requires_grad
assert len(params) > 0
p = loop.controllers['4'].mixing_matrix()
assert torch.allclose(p.sum(0), torch.ones(2), atol=1e-5)
assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-5)
dyn_p = loop.controllers['4'].mixing_matrix(torch.randn(2, 5, 2, 2))
assert torch.allclose(dyn_p.sum(-1), torch.ones(2, 5, 2), atol=1e-5)
assert torch.allclose(dyn_p.sum(-2), torch.ones(2, 5, 2), atol=1e-5)
label = ids.clone()
label[:, :2] = -100
result = loop(ids, labels=label)
assert torch.isfinite(result['loss'])
result['loss'].backward()
assert loop.controllers['4'].extra_gain.grad is not None
assert loop.controllers['4'].ffn_extra_gain.grad is not None
assert loop.controllers['4'].dynamic_proj.weight.grad is not None
assert loop.controllers['4'].ffn_dynamic_proj.weight.grad is not None
assert loop.controllers['4'].pass_up[1].weight.grad is not None
assert base.model.layers[6].mlp.gate_proj.base.weight.grad is not None
assert base.model.layers[7].mlp.gate_proj.lora_B.weight.grad is not None
assert loop.engram.project.weight.grad is not None

# Verify full_finetune=True unfreezes all layers
all_params = set_trainables(loop, full_finetune=True)
assert base.model.layers[0].mlp.gate_proj.weight.requires_grad
assert base.model.layers[7].mlp.gate_proj.base.weight.requires_grad
print('PASS cpu: Dual-mHC, pass adapters, multi-scale Engram, chunked CE, _VisitCache, full/hybrid train', count_params(loop))
