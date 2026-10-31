"""SparkX-Loop v0.2: Looped Recursive Transformer with Sublayer Dual-mHC, Pass-Conditioned
Deliberation Adapters, Multi-Scale Causal Engram, 1M-Context Fused SDPA, and Unsloth/Chunked CE.

Intended upstream: XHToken/Spark-X2.6-4B, transformers==4.57.1.
- Preserves Spark-X2.6-4B's 1M context window architecture (3 x sliding_attention w=512 RoPE +
  1 x full_attention NoPE per 4-layer macro-block, plus headwise sigmoid output gating g_proj).
- Upgrades recurrent layers [4, 28) with SenseNova-Loop-inspired sublayer Dual-mHC (mhc_attn + mhc_ffn),
  token-dependent FP32 RMSNorm Sinkhorn routing, pass-indexed conditioning, and zero-gated identity init.
- Supports both full-parameter finetuning (all 36 physical layers + mHC + Engram on 96 GB G4) and
  hybrid LoRA + selective layer finetuning.
- Supports fast per-visit KV-cached looped generation (`generate_cached` via `_VisitCache`) and
  exact full-recompute generation (`generate_slow`).
"""
from __future__ import annotations
from contextlib import nullcontext
import copy
import dataclasses
import importlib
import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_SDPA_SUPPORTS_GQA = (
    hasattr(F, 'scaled_dot_product_attention')
    and 'enable_gqa' in inspect.signature(F.scaled_dot_product_attention).parameters
)


@dataclasses.dataclass
class LoopSettings:
    start: int = 4
    end: int = 28           # exclusive: 6 x 4-layer Spark macro-blocks (18 sliding + 6 full NoPE)
    passes: int = 2
    max_passes: int = 4     # supports test-time scaling up to 4 recurrent passes
    mhc_streams: int = 2
    eng_buckets: int = 16384
    eng_dim: int = 64
    checkpoint_layers: bool = True
    offload_checkpoint_to_cpu: bool = False  # uses 130 GB System RAM pinned memory when True
    use_mhc: bool = True
    use_engram: bool = True
    dynamic_mhc: bool = True
    sublayer_mhc: bool = True   # Dual-mHC: independent mhc_attn + mhc_ffn per recurrent layer
    pass_adapter_rank: int = 16 # Pass-indexed low-rank deliberation adapter inside mHC
    ce_chunk_tokens: int = 512  # Token-chunked masked CE to avoid [B, T, 167168] FP32 logits spike


class _CacheLayersView:
    """Expose the virtual execution cache slot when a physical layer indexes cache.layers."""
    def __init__(self, cache, physical: int, virtual: int):
        self.cache = cache
        self.physical = physical
        self.virtual = virtual

    def __getitem__(self, index: int):
        if index != self.physical:
            raise IndexError('A physical layer may only access its own virtual cache slot')
        return self.cache.layers[self.virtual]


class _VisitCache:
    """Redirects physical layer_idx KV cache updates to unique per-visit virtual_idx slots.

    Mirrors the SenseNova-Loop-9B-Preview _VisitCache design so each recurrent pass
    over physical layers [loop_start, loop_end) maintains an independent KV cache slot.
    """
    def __init__(self, cache, physical_idx: int, virtual_idx: int):
        self.cache = cache
        self.physical_idx = physical_idx
        self.virtual_idx = virtual_idx
        if hasattr(cache, 'layers'):
            self.layers = _CacheLayersView(cache, physical_idx, virtual_idx)

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        target = self.virtual_idx if layer_idx == self.physical_idx else layer_idx
        return self.cache.update(key_states, value_states, target, cache_kwargs)

    def __getattr__(self, name):
        return getattr(self.cache, name)


def patch_spark_attention_for_long_context(base_model: nn.Module) -> int:
    """Replace Spark2_5Attention's O(T^2) eager_attention_forward with fused PyTorch SDPA / GQA.

    Preserves Spark-X2.6-4B's 1M context window architecture (sliding_window=512 RoPE on
    local layers + NoPE on full_attention layers + headwise sigmoid g_proj gate) while
    avoiding materializing the [B, 32, T, T] attention matrix and 4x repeat_kv expansion.
    """
    if not hasattr(base_model, 'model') or not hasattr(base_model.model, 'layers'):
        return 0
    layers = base_model.model.layers
    if not layers or not hasattr(layers[0], 'self_attn'):
        return 0
    attn_mod = importlib.import_module(type(layers[0].self_attn).__module__)
    if not hasattr(attn_mod, 'eager_attention_forward'):
        return 0
    if getattr(attn_mod, '_sparkx_sdpa_patched', False):
        return len(layers)

    orig_repeat_kv = getattr(attn_mod, 'repeat_kv', None)

    def fused_sdpa_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        if scaling is None:
            scaling = 1.0 / math.sqrt(query.shape[-1])
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, : key.shape[-2]]
        use_gqa = _SDPA_SUPPORTS_GQA and (module.num_key_value_groups > 1)
        if not use_gqa and orig_repeat_kv is not None and module.num_key_value_groups > 1:
            key = orig_repeat_kv(key, module.num_key_value_groups)
            value = orig_repeat_kv(value, module.num_key_value_groups)
        sdpa_kwargs = {'scale': scaling}
        if use_gqa:
            sdpa_kwargs['enable_gqa'] = True
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=False,
            **sdpa_kwargs,
        )
        return out, None

    attn_mod.eager_attention_forward = fused_sdpa_attention_forward
    attn_mod._sparkx_sdpa_patched = True
    return len(layers)


class LoRALinear(nn.Module):
    """Simple native torch LoRA, intentionally keeps .weight accessible for Spark MLP code."""
    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError(f"LoRA expected nn.Linear, got {type(original)}")
        self.base = original
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Linear(original.in_features, rank, bias=False, dtype=torch.float32, device=original.weight.device)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False, dtype=torch.float32, device=original.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scale
        return out + delta.to(out.dtype)


def attach_lora(base_model: nn.Module, start: int = 4, end: int = 36, rank: int = 8, alpha: float = 16.0,
                attention: bool = True, mlp: bool = True):
    """Attach LoRA once to original physical layers. The same LoRA weights run in every recurrence."""
    changed = []
    layers = base_model.model.layers
    for idx in range(start, min(end, len(layers))):
        layer = layers[idx]
        paths = []
        if attention:
            paths += [(layer.self_attn, 'q_k_v_proj'), (layer.self_attn, 'out_proj')]
            if getattr(layer.self_attn, 'g_proj', None) is not None:
                paths.append((layer.self_attn, 'g_proj'))
        if mlp:
            paths += [(layer.mlp, 'gate_proj'), (layer.mlp, 'up_proj'), (layer.mlp, 'down_proj')]
        for parent, attr in paths:
            module = getattr(parent, attr)
            if isinstance(module, LoRALinear):
                continue
            setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))
            changed.append(f'{idx}:{attr}')
    return changed


class HashedBigramEngram(nn.Module):
    """Multi-scale causal N-gram memory (2-gram + 3-gram + skip-2-gram) with context-aware gating.

    Position-independent and strictly causal (depends only on tokens t-2, t-1, t), so it
    generalizes seamlessly across Spark-X2.6-4B's 1M token context window.
    Zero-initialized `project.weight` guarantees exact 0.0 output at step 0.
    """
    def __init__(self, hidden: int, buckets: int = 16384, dim: int = 64, rmsnorm_eps: float = 1e-6):
        super().__init__()
        self.buckets = buckets
        self.dim = dim
        self.rmsnorm_eps = rmsnorm_eps
        self.emb = nn.Embedding(buckets, dim, dtype=torch.float32)
        self.emb_tri = nn.Embedding(buckets, dim, dtype=torch.float32)
        self.emb_skip = nn.Embedding(buckets, dim, dtype=torch.float32)
        self.ctx_gate = nn.Linear(hidden, dim, bias=True, dtype=torch.float32)
        self.project = nn.Linear(dim, hidden, bias=False, dtype=torch.float32)
        self.gain = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.emb_tri.weight, std=0.02)
        nn.init.normal_(self.emb_skip.weight, std=0.02)
        nn.init.zeros_(self.ctx_gate.weight)
        nn.init.zeros_(self.ctx_gate.bias)
        nn.init.zeros_(self.project.weight)

    def forward(self, input_ids: torch.LongTensor, dtype: torch.dtype,
                prev_ids: Optional[torch.LongTensor] = None,
                hidden_states: Optional[torch.Tensor] = None):
        # Deterministic causal hashing over (t-2, t-1, t); zero access to future tokens.
        bsz, seq_len = input_ids.shape
        if prev_ids is None:
            z1 = torch.zeros_like(input_ids[:, :1])
            prev1 = torch.cat((z1, input_ids[:, :-1]), dim=1)
            z2 = torch.zeros_like(input_ids[:, :2]) if seq_len >= 2 else torch.zeros_like(input_ids)
            prev2 = torch.cat((z2, input_ids[:, :-2]), dim=1) if seq_len >= 2 else torch.zeros_like(input_ids)
        else:
            last1 = prev_ids[:, -1:]
            prev1 = torch.cat((last1, input_ids[:, :-1]), dim=1) if seq_len > 1 else last1
            if prev_ids.shape[1] >= 2:
                last2 = prev_ids[:, -2:-1]
            else:
                last2 = torch.zeros_like(last1)
            if seq_len == 1:
                prev2 = last2
            else:
                prev2 = torch.cat((last2, last1, input_ids[:, :-2]), dim=1)
        idx_bi = ((prev1 * 1000003 + input_ids * 9176 + 29) % self.buckets).long()
        idx_tri = ((prev2 * 2000003 + prev1 * 1000033 + input_ids * 19391 + 73) % self.buckets).long()
        idx_skip = ((prev2 * 1500007 + input_ids * 31337 + 53) % self.buckets).long()
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            mem = self.emb(idx_bi) + 0.5 * self.emb_tri(idx_tri) + 0.5 * self.emb_skip(idx_skip)
            mem = mem * torch.rsqrt(mem.pow(2).mean(dim=-1, keepdim=True) + self.rmsnorm_eps)
            if hidden_states is not None:
                gate = 2.0 * torch.sigmoid(self.ctx_gate(hidden_states.float()))
                mem = mem * gate
            z = self.project(mem) * torch.sigmoid(self.gain)
        return z.to(dtype)


class TwoStreamMHC(nn.Module):
    """Two-stream Manifold-Constrained Hyper-Connections (mHC) with:
    1. Sublayer Dual-mHC (`mhc_attn` + `mhc_ffn`) matching SenseNova-Loop-9B-Preview.
    2. Token-dependent FP32 RMSNorm + zero-initialized linear Sinkhorn routing (`dynamic_proj`).
    3. Pass-indexed routing biases & zero-initialized low-rank deliberation adapters so Pass 0
       and Pass 1+ can specialize despite sharing physical backbone weights.
    4. Exact zero-gated identity initialization (`extra_gain=0`, `ffn_extra_gain=0`) so Pass 2
       is an exact mathematical identity over Pass 1 at step 0.
    """
    def __init__(self, n_streams: int = 2, sinkhorn_steps: int = 20,
                 hidden_size: int = 0, dynamic_routing: bool = True,
                 sublayer_mhc: bool = True, max_passes: int = 4,
                 pass_adapter_rank: int = 16, rmsnorm_eps: float = 1e-6):
        super().__init__()
        if n_streams != 2:
            raise ValueError('Prototype intentionally supports exactly two streams')
        self.n = n_streams
        self.sinkhorn_steps = sinkhorn_steps
        self.hidden_size = hidden_size
        self.dynamic_routing = bool(dynamic_routing and hidden_size > 0)
        self.sublayer_mhc = bool(sublayer_mhc)
        self.max_passes = max(1, int(max_passes))
        self.rmsnorm_eps = rmsnorm_eps

        # Primary / Attention mHC parameters
        self.mix_logits = nn.Parameter(torch.tensor([[4., -4.], [-4., 4.]], dtype=torch.float32))
        self.read_logits = nn.Parameter(torch.tensor([5., -5.], dtype=torch.float32))
        self.write_logits = nn.Parameter(torch.tensor([5., -5.], dtype=torch.float32))
        self.extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Sublayer FFN mHC parameters (used when layer exposes input_layernorm & post_attention_layernorm)
        self.ffn_mix_logits = nn.Parameter(torch.tensor([[4., -4.], [-4., 4.]], dtype=torch.float32))
        self.ffn_read_logits = nn.Parameter(torch.tensor([5., -5.], dtype=torch.float32))
        self.ffn_write_logits = nn.Parameter(torch.tensor([5., -5.], dtype=torch.float32))
        self.ffn_extra_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Pass-specific routing bias: [max_passes, 2 (attn/ffn), 8 (2 read + 2 write + 4 mix)]
        self.pass_routing_bias = nn.Parameter(torch.zeros(self.max_passes, 2, 2 * n_streams + n_streams * n_streams, dtype=torch.float32))

        if self.dynamic_routing:
            out_features = 2 * n_streams + n_streams * n_streams
            self.dynamic_proj = nn.Linear(n_streams * hidden_size, out_features, bias=False, dtype=torch.float32)
            self.ffn_dynamic_proj = nn.Linear(n_streams * hidden_size, out_features, bias=False, dtype=torch.float32)
            nn.init.zeros_(self.dynamic_proj.weight)
            nn.init.zeros_(self.ffn_dynamic_proj.weight)
        else:
            self.register_parameter('dynamic_proj', None)
            self.register_parameter('ffn_dynamic_proj', None)

        # Pass-conditioned low-rank deliberation adapter (zero-initialized up-projection)
        if hidden_size > 0 and pass_adapter_rank > 0:
            self.pass_down = nn.ModuleList([
                nn.Linear(hidden_size, pass_adapter_rank, bias=False, dtype=torch.float32)
                for _ in range(self.max_passes)
            ])
            self.pass_up = nn.ModuleList([
                nn.Linear(pass_adapter_rank, hidden_size, bias=False, dtype=torch.float32)
                for _ in range(self.max_passes)
            ])
            for d, u in zip(self.pass_down, self.pass_up):
                nn.init.kaiming_uniform_(d.weight, a=math.sqrt(5))
                nn.init.zeros_(u.weight)
        else:
            self.pass_down = None
            self.pass_up = None

    def mixing_matrix(self, dyn_mix: Optional[torch.Tensor] = None, base_logits: Optional[torch.Tensor] = None):
        # Sinkhorn in log space for stability, supporting both [2, 2] and [B, T, 2, 2].
        x = (self.mix_logits if base_logits is None else base_logits).float()
        if dyn_mix is not None:
            x = x + dyn_mix.float()
        for _ in range(self.sinkhorn_steps):
            x = x - torch.logsumexp(x, dim=-1, keepdim=True)
            x = x - torch.logsumexp(x, dim=-2, keepdim=True)
        return torch.exp(x)

    def _compute_routing(self, streams_f32: torch.Tensor, pass_idx: int = 0, is_ffn: bool = False):
        # streams_f32: [2, B, T, D]
        p_idx = min(max(0, int(pass_idx)), self.max_passes - 1)
        sub_idx = 1 if is_ffn else 0
        base_r = self.ffn_read_logits if is_ffn else self.read_logits
        base_w = self.ffn_write_logits if is_ffn else self.write_logits
        base_m = self.ffn_mix_logits if is_ffn else self.mix_logits
        proj = self.ffn_dynamic_proj if is_ffn else self.dynamic_proj
        p_bias = self.pass_routing_bias[p_idx, sub_idx].float()

        if not self.dynamic_routing or proj is None:
            r_log = base_r.float() + p_bias[:self.n]
            w_log = base_w.float() + p_bias[self.n:2 * self.n]
            m_log = base_m.float() + p_bias[2 * self.n:].reshape(self.n, self.n)
            rw = r_log.softmax(dim=0)[:, None, None, None]
            ww = w_log.softmax(dim=0)[:, None, None, None]
            mat = self.mixing_matrix(base_logits=m_log)[:, :, None, None, None]
            return rw, ww, mat

        with torch.autocast(device_type=streams_f32.device.type, enabled=False):
            _, bsz, seq_len, dim = streams_f32.shape
            flat = streams_f32.permute(1, 2, 0, 3).reshape(bsz, seq_len, self.n * dim)
            normed = flat * torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + self.rmsnorm_eps)
            dyn = proj(normed) + p_bias.view(1, 1, -1)  # [B, T, 8]
            dyn_read = dyn[..., :self.n]
            dyn_write = dyn[..., self.n:2 * self.n]
            dyn_mix = dyn[..., 2 * self.n:].reshape(bsz, seq_len, self.n, self.n)
            rw = (base_r.float() + dyn_read).softmax(dim=-1).permute(2, 0, 1).unsqueeze(-1)
            ww = (base_w.float() + dyn_write).softmax(dim=-1).permute(2, 0, 1).unsqueeze(-1)
            mat = self.mixing_matrix(dyn_mix, base_logits=base_m).permute(2, 3, 0, 1).unsqueeze(-1)
            return rw, ww, mat

    def _apply_pass_adapter(self, x_f32: torch.Tensor, pass_idx: int) -> torch.Tensor:
        if self.pass_down is None or self.pass_up is None:
            return torch.zeros_like(x_f32)
        p_idx = min(max(0, int(pass_idx)), self.max_passes - 1)
        with torch.autocast(device_type=x_f32.device.type, enabled=False):
            normed = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.rmsnorm_eps)
            return self.pass_up[p_idx](F.silu(self.pass_down[p_idx](normed)))

    def forward(self, streams: torch.Tensor, layer: nn.Module, rope, mask,
                extra: bool = False, pass_idx: int = 0,
                past_key_values=None, cache_position=None):
        # streams: [2, batch, seq, hidden]. Layer is shared across passes.
        streams_f32 = streams.float()
        identity = torch.eye(self.n, dtype=torch.float32, device=streams.device)[:, :, None, None, None]

        # Sublayer Dual-mHC path when wrapping a real Spark2_5DecoderLayer
        if self.sublayer_mhc and hasattr(layer, 'input_layernorm') and hasattr(layer, 'post_attention_layernorm'):
            # 1. Attention sublayer mHC
            rw_a, ww_a, mat_a = self._compute_routing(streams_f32, pass_idx=pass_idx, is_ffn=False)
            mixed_a = (rw_a * streams_f32).sum(dim=0)
            target_dtype = layer.mlp.gate_proj.weight.dtype
            normed_a = layer.input_layernorm(mixed_a.to(streams.dtype)).to(target_dtype)
            attn_out, _ = layer.self_attn(
                hidden_states=normed_a,
                position_embeddings=rope,
                attention_mask=mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            gain_a = torch.tanh(self.extra_gain.float()) if extra else 1.0
            eff_mat_a = identity + gain_a * (mat_a - identity)
            streams_mid = (eff_mat_a * streams_f32.unsqueeze(0)).sum(dim=1) + gain_a * ww_a * attn_out.float().unsqueeze(0)

            # 2. FFN sublayer mHC + Pass-conditioned deliberation adapter
            rw_f, ww_f, mat_f = self._compute_routing(streams_mid, pass_idx=pass_idx, is_ffn=True)
            mixed_f = (rw_f * streams_mid).sum(dim=0)
            normed_f = layer.post_attention_layernorm(mixed_f.to(streams.dtype)).to(target_dtype)
            mlp_out = layer.mlp(normed_f).float() + self._apply_pass_adapter(mixed_f, pass_idx)
            gain_f = torch.tanh(self.ffn_extra_gain.float()) if extra else 1.0
            eff_mat_f = identity + gain_f * (mat_f - identity)
            out = (eff_mat_f * streams_mid.unsqueeze(0)).sum(dim=1) + gain_f * ww_f * mlp_out.unsqueeze(0)
            return out.to(streams.dtype)

        # Fallback single-wrapper mHC path (e.g. simplified test doubles without separate layernorms)
        rw, ww, mat = self._compute_routing(streams_f32, pass_idx=pass_idx, is_ffn=False)
        mixed_input = (rw * streams_f32).sum(dim=0).to(streams.dtype)
        updated = layer(
            mixed_input, position_embeddings=rope, attention_mask=mask,
            past_key_values=past_key_values, cache_position=cache_position,
        )
        delta = updated.float() - mixed_input.float() + self._apply_pass_adapter(mixed_input.float(), pass_idx)
        gain = torch.tanh(self.extra_gain.float()) if extra else 1.0
        eff_mat = identity + gain * (mat - identity)
        routed = (eff_mat * streams_f32.unsqueeze(0)).sum(dim=1)
        out = routed + gain * ww * delta.unsqueeze(0)
        return out.to(streams.dtype)


def fused_masked_cross_entropy(hidden_states: torch.Tensor, head_weight: torch.Tensor,
                               labels: torch.Tensor, chunk_tokens: int = 512) -> torch.Tensor:
    """Unsloth-style memory-efficient masked cross-entropy over vocabulary size 167,168.

    Gathers only supervised positions (`shift_labels != -100`) and evaluates the FP32
    vocabulary projection in small token chunks under gradient checkpointing so the
    massive [B, T, 167168] FP32 logits tensor is never materialized in VRAM.
    """
    shift_h = hidden_states[:, :-1].contiguous().reshape(-1, hidden_states.shape[-1])
    shift_y = labels[:, 1:].contiguous().reshape(-1).to(shift_h.device)
    valid_mask = shift_y != -100
    if not torch.any(valid_mask):
        return hidden_states.sum() * 0.0
    valid_h = shift_h[valid_mask]
    valid_y = shift_y[valid_mask]
    n_valid = valid_y.numel()
    if n_valid <= chunk_tokens:
        logits = F.linear(valid_h.to(head_weight.dtype), head_weight).float()
        return F.cross_entropy(logits, valid_y, reduction='mean')

    def _chunk_ce(h_chunk: torch.Tensor, y_chunk: torch.Tensor) -> torch.Tensor:
        lg = F.linear(h_chunk.to(head_weight.dtype), head_weight).float()
        return F.cross_entropy(lg, y_chunk, reduction='sum')

    total_loss = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    use_ckpt = valid_h.requires_grad or head_weight.requires_grad
    for start_idx in range(0, n_valid, chunk_tokens):
        h_c = valid_h[start_idx:start_idx + chunk_tokens]
        y_c = valid_y[start_idx:start_idx + chunk_tokens]
        if use_ckpt:
            total_loss = total_loss + checkpoint(_chunk_ce, h_c, y_c, use_reentrant=False)
        else:
            total_loss = total_loss + _chunk_ce(h_c, y_c)
    return total_loss / float(n_valid)


class SparkXLoop(nn.Module):
    """Wrap original Spark2_5ForCausalLM and execute real layers with shared depth recurrence.

    Reproduces the upstream model's masks, type-dependent RoPE and norm/head path.
    Supports both cache-free training/verification (generate_slow) and per-visit
    KV-cached looped decoding (generate_cached via _VisitCache).
    """
    def __init__(self, base: nn.Module, settings: Optional[LoopSettings] = None,
                 strict: bool = True, patch_sdpa: bool = True):
        super().__init__()
        self.base = base
        self.settings = settings or LoopSettings()
        self.n_layers = len(self.base.model.layers)
        self.width = int(self.base.config.hidden_size)
        if strict and (self.n_layers != 36 or self.width != 2560):
            raise RuntimeError(f'Expected Spark 36x2560, got {self.n_layers}x{self.width}')
        if self.settings.start < 0 or self.settings.end > self.n_layers:
            raise ValueError('Bad recurrent range')
        if patch_sdpa:
            patch_spark_attention_for_long_context(self.base)
        self.controllers = nn.ModuleDict({
            str(i): TwoStreamMHC(
                self.settings.mhc_streams,
                hidden_size=self.width,
                dynamic_routing=self.settings.dynamic_mhc,
                sublayer_mhc=self.settings.sublayer_mhc,
                max_passes=self.settings.max_passes,
                pass_adapter_rank=self.settings.pass_adapter_rank,
            )
            for i in range(self.settings.start, self.settings.end)
        })
        self.engram = HashedBigramEngram(self.width, self.settings.eng_buckets, self.settings.eng_dim)
        self.enable_checkpoint = self.settings.checkpoint_layers
        self.offload_checkpoint_to_cpu = self.settings.offload_checkpoint_to_cpu
        self.loop_start, self.loop_end, self.loop_passes = self.settings.start, self.settings.end, self.settings.passes
        self.mhc_enabled = self.settings.use_mhc
        self.engram_enabled = self.settings.use_engram
        self._validate_loop()

    def _validate_loop(self):
        if not (0 <= self.loop_start < self.loop_end <= self.n_layers):
            raise ValueError('Invalid loop boundaries')
        if (self.loop_start % 4 or self.loop_end % 4):
            raise ValueError('Loop boundaries must align to Spark groups of four')
        if self.loop_passes < 1:
            raise ValueError('loop_passes must be >= 1')

    @property
    def execution_plan(self) -> list[int]:
        return (
            list(range(0, self.loop_start))
            + list(range(self.loop_start, self.loop_end)) * self.loop_passes
            + list(range(self.loop_end, self.n_layers))
        )

    @property
    def total_virtual_layers(self) -> int:
        return len(self.execution_plan)

    def _make_loop_cache_config(self):
        cfg = copy.deepcopy(self.base.config)
        plan = self.execution_plan
        cfg.num_hidden_layers = len(plan)
        if hasattr(cfg, 'layer_types') and cfg.layer_types:
            cfg.layer_types = [cfg.layer_types[i] for i in plan]
        return cfg

    def _checkpoint_context(self):
        if self.training and self.enable_checkpoint and self.offload_checkpoint_to_cpu:
            return torch.autograd.graph.save_on_cpu(pin_memory=True)
        return nullcontext()

    def set_loop(self, start: int, end: int, passes: int, use_mhc: bool = True, use_engram: bool = True):
        self.loop_start, self.loop_end, self.loop_passes = start, end, passes
        self.mhc_enabled, self.engram_enabled = use_mhc, use_engram
        self._validate_loop()
        for idx in range(start, end):
            if str(idx) not in self.controllers:
                raise RuntimeError(f'Controller for layer {idx} was not created: use initial superset')

    def _forward_context(self, embeds, attention_mask, past_key_values=None, cache_position=None):
        # Keep the same mask and rotary logic as Spark2_5Model.forward.
        model = self.base.model
        cfg = self.base.config
        device = embeds.device
        if cache_position is None:
            past_seen = past_key_values.get_seq_length(0) if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + embeds.shape[1], device=device)
        position_ids = cache_position.unsqueeze(0)
        # Test doubles can supply a minimal context without importing Transformers.
        if hasattr(model, 'make_loop_context'):
            return model.make_loop_context(embeds, attention_mask), cache_position
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        mask_cfg = self._make_loop_cache_config() if past_key_values is not None else cfg
        kwargs = dict(config=mask_cfg, input_embeds=embeds, attention_mask=attention_mask,
                      cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids)
        masks = {'full_attention': create_causal_mask(**kwargs)}
        if getattr(model, 'has_sliding_layers', False):
            masks['sliding_attention'] = create_sliding_window_causal_mask(**kwargs)
        implementation = importlib.import_module(type(model.layers[0].self_attn).__module__)
        rope = {}
        for layer_type in set(cfg.layer_types):
            rope[layer_type] = implementation.compute_rope_cos_sin(
                cache_position, cfg.head_dim, cfg.get_rope_theta(layer_type),
                partial_rotary_factor=cfg.get_partial_rotary_factor(layer_type), device=device,
            )
        return (masks, rope), cache_position

    def _apply_layer(self, idx, h, masks, ropes, past_key_values=None, cache_position=None, virtual_idx: Optional[int] = None):
        layer = self.base.model.layers[idx]
        kind = layer.layer_type
        rp, mk = ropes[kind], masks[kind]
        v_cache = _VisitCache(past_key_values, idx, virtual_idx if virtual_idx is not None else idx) if past_key_values is not None else None
        def step(x, _layer=layer, _rp=rp, _mk=mk, _vc=v_cache, _cp=cache_position):
            return _layer(x, position_embeddings=_rp, attention_mask=_mk,
                          past_key_values=_vc, cache_position=_cp)
        if v_cache is None and self.training and self.enable_checkpoint and (
            h.requires_grad or any(p.requires_grad for p in layer.parameters())
        ):
            with self._checkpoint_context():
                return checkpoint(step, h, use_reentrant=False)
        return step(h)

    def forward_hidden(self, input_ids: torch.LongTensor, attention_mask=None,
                       past_key_values=None, cache_position=None,
                       prev_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError('Expected input_ids shape [batch, sequence]')
        embeds = self.base.model.embedding(input_ids)
        (masks, ropes), cache_position = self._forward_context(
            embeds, attention_mask, past_key_values=past_key_values, cache_position=cache_position
        )
        h = embeds.float()  # matches upstream Spark 2.5 model forward
        virtual_idx = 0

        # Original prelude layers, unchanged.
        for idx in range(self.loop_start):
            h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                  cache_position=cache_position, virtual_idx=virtual_idx)
            virtual_idx += 1
        # Multi-scale causal Engram inserted after prelude; zero-initialized project keeps step-0 parity exact.
        if self.engram_enabled:
            h = h + self.engram(input_ids, h.dtype, prev_ids=prev_ids, hidden_states=h)

        if not self.mhc_enabled:
            # Direct recurrence negative control.
            for _ in range(self.loop_passes):
                for idx in range(self.loop_start, self.loop_end):
                    h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                          cache_position=cache_position, virtual_idx=virtual_idx)
                    virtual_idx += 1
        else:
            streams = torch.stack((h, h), dim=0)
            for p in range(self.loop_passes):
                for idx in range(self.loop_start, self.loop_end):
                    kind = self.base.model.layers[idx].layer_type
                    layer = self.base.model.layers[idx]
                    ctrl = self.controllers[str(idx)]
                    rp, mk = ropes[kind], masks[kind]
                    v_cache = _VisitCache(past_key_values, idx, virtual_idx) if past_key_values is not None else None
                    virtual_idx += 1
                    # Capture immutable locals, not a changing Python loop variable.
                    def step(s, _ctrl=ctrl, _layer=layer, _rp=rp, _mk=mk, _extra=(p > 0), _p=p, _vc=v_cache, _cp=cache_position):
                        return _ctrl(s, _layer, _rp, _mk, extra=_extra, pass_idx=_p,
                                     past_key_values=_vc, cache_position=_cp)
                    if v_cache is None and self.training and self.enable_checkpoint:
                        with self._checkpoint_context():
                            streams = checkpoint(step, streams, use_reentrant=False)
                    else:
                        streams = step(streams)
            h = streams[0]

        for idx in range(self.loop_end, self.n_layers):
            h = self._apply_layer(idx, h, masks, ropes, past_key_values=past_key_values,
                                  cache_position=cache_position, virtual_idx=virtual_idx)
            virtual_idx += 1
        return self.base.model.norm(h).to(self.base.model.embedding.weight.dtype)

    def _head_weight(self) -> torch.Tensor:
        if getattr(self.base.config, 'tie_word_embeddings', False):
            return self.base.model.embedding.weight
        return self.base.lm_head.weight

    def forward_logits(self, input_ids: torch.LongTensor, attention_mask=None,
                       past_key_values=None, cache_position=None,
                       prev_ids: Optional[torch.LongTensor] = None):
        h = self.forward_hidden(
            input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, cache_position=cache_position,
            prev_ids=prev_ids,
        )
        if getattr(self.base.config, 'tie_word_embeddings', False):
            return F.linear(h, self.base.model.embedding.weight)
        return self.base.lm_head(h)

    def forward(self, input_ids, labels=None, attention_mask=None, return_logits: bool = False):
        if labels is None:
            logits = self.forward_logits(input_ids, attention_mask)
            return {'logits': logits}
        if input_ids.shape[1] < 2:
            raise ValueError('Need at least 2 tokens for next-token objective')
        h = self.forward_hidden(input_ids, attention_mask=attention_mask)
        loss = fused_masked_cross_entropy(
            h, self._head_weight(), labels, chunk_tokens=self.settings.ce_chunk_tokens
        )
        if return_logits:
            logits = F.linear(h, self._head_weight()) if getattr(self.base.config, 'tie_word_embeddings', False) else self.base.lm_head(h)
            return {'logits': logits, 'loss': loss}
        return {'logits': None, 'loss': loss}

    @torch.no_grad()
    def generate_slow(self, input_ids, max_new_tokens=24, eos_token_id=None):
        """Correct full-recompute greedy looped generation. Slow by design; no KV cache."""
        prev_train = self.training
        self.eval()
        out = input_ids
        try:
            for _ in range(max_new_tokens):
                next_id = self.forward_logits(out)[:, -1].argmax(-1, keepdim=True)
                out = torch.cat((out, next_id), dim=-1)
                if eos_token_id is not None and torch.all(next_id == eos_token_id):
                    break
        finally:
            self.train(prev_train)
        return out

    @torch.no_grad()
    def generate_cached(self, input_ids, max_new_tokens=128, eos_token_id=None):
        """Fast per-visit KV-cached greedy looped generation using _VisitCache."""
        from transformers.cache_utils import DynamicCache
        prev_train = self.training
        self.eval()
        try:
            cache = DynamicCache(config=self._make_loop_cache_config())
        except TypeError:
            cache = DynamicCache()
        out = input_ids
        try:
            # Prefill full prompt
            logits = self.forward_logits(out, past_key_values=cache)
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            prev_tok = out[:, -2:]
            out = torch.cat((out, next_id), dim=-1)
            if eos_token_id is not None and torch.all(next_id == eos_token_id):
                return out
            for _ in range(max_new_tokens - 1):
                logits = self.forward_logits(next_id, past_key_values=cache, prev_ids=prev_tok)
                prev_tok = out[:, -2:]
                next_id = logits[:, -1].argmax(-1, keepdim=True)
                out = torch.cat((out, next_id), dim=-1)
                if eos_token_id is not None and torch.all(next_id == eos_token_id):
                    break
        finally:
            self.train(prev_train)
        return out


def set_trainables(model: SparkXLoop, full_layer_ids=(), full_finetune: bool = False,
                   train_embeddings: bool = False):
    """Configure trainable parameters for either Full Finetuning (all 36 layers + mHC + Engram)
    or Hybrid LoRA + selective backbone layers.
    """
    if full_finetune:
        full_layer_ids = set(range(model.n_layers))
    else:
        full_layer_ids = set(full_layer_ids)
    for p in model.base.parameters():
        p.requires_grad_(False)
    if train_embeddings:
        for p in model.base.model.embedding.parameters():
            p.requires_grad_(True)
        for p in model.base.model.norm.parameters():
            p.requires_grad_(True)
        if not getattr(model.base.config, 'tie_word_embeddings', False) and hasattr(model.base, 'lm_head'):
            for p in model.base.lm_head.parameters():
                p.requires_grad_(True)
    elif full_finetune:
        for p in model.base.model.norm.parameters():
            p.requires_grad_(True)
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            for p in list(mod.lora_A.parameters()) + list(mod.lora_B.parameters()):
                p.requires_grad_(True)
    for p in model.controllers.parameters():
        p.requires_grad_(True)
    for p in model.engram.parameters():
        p.requires_grad_(True)
    for idx in full_layer_ids:
        if not (0 <= idx < model.n_layers):
            raise ValueError(f'Invalid full layer {idx}')
        for name, p in model.base.model.layers[idx].named_parameters():
            if not (name.startswith('self_attn.q_k_v_proj.lora_') or
                    name.startswith('self_attn.out_proj.lora_') or '.lora_' in name):
                p.requires_grad_(True)
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def optimizer_groups(model: SparkXLoop, lr_adapter=1e-4, lr_full=1e-5, weight_decay=0.01, lr_mhc=5e-3):
    adapters, routing, backbone = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (name.startswith('controllers.') and '.pass_' not in name) or name == 'engram.gain':
            routing.append(p)
        elif name.startswith('base.model.') and '.lora_' not in name:
            backbone.append(p)
        else:
            adapters.append(p)
    return [{'params': adapters, 'lr': lr_adapter, 'weight_decay': weight_decay},
            {'params': routing, 'lr': lr_mhc, 'weight_decay': 0.0},
            {'params': backbone, 'lr': lr_full, 'weight_decay': weight_decay}]


def build_optimizer(model: SparkXLoop, lr_adapter=1e-4, lr_full=1e-5, weight_decay=0.01,
                    lr_mhc=5e-3, use_8bit: bool = True):
    """Build Unsloth / bitsandbytes PagedAdamW8bit (paging to 130 GB System RAM when needed)
    or fused PyTorch AdamW.
    """
    groups = optimizer_groups(
        model, lr_adapter=lr_adapter, lr_full=lr_full, weight_decay=weight_decay, lr_mhc=lr_mhc
    )
    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(groups, betas=(0.9, 0.95), eps=1e-8)
        except Exception:
            pass
    use_fused = torch.cuda.is_available() and 'fused' in inspect.signature(torch.optim.AdamW.__init__).parameters
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)


def trainable_state(model: SparkXLoop):
    return {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}


def count_params(model: SparkXLoop):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
