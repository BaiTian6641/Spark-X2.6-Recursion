"""sparkx_kv_codec.py — Quantized Cold KV Page Codecs & Capacity Arithmetic for Spark X2.6 4B Loop72 LSA (Work Package P6).

Supports:
  - `bf16`: Lossless 16-bit baseline (`2` bytes/scalar).
  - `int8`: Per-group symmetric 8-bit quantization (`1` byte/scalar + FP16 scale per group).
  - `int4`: Packed 2-values-per-byte (`uint8`) 4-bit group quantization (`0.5` bytes/scalar + FP16 scale/zero per group).
  - `mixed_rotary_bf16`: Preserves the 64 rotated K dimensions in BF16 (`2` bytes/scalar) while quantizing the 192
    RoPE-free K dimensions and all 256 V dimensions to packed 4-bit (`118.02 GiB` ideal payload at 5M tokens across 18 global visits).

Works with both PyTorch tensors and pure-Python nested lists/floats so unit tests execute in CPU-only VMs without PyTorch.
"""
from __future__ import annotations

from collections import Counter
import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


VALID_KV_CODECS = ("bf16", "int8", "int4", "mixed_rotary_bf16")


@dataclasses.dataclass
class KVCodecConfig:
    codec: str = "bf16"
    group_size: int = 64
    rope_dim: int = 64
    rope_first: bool = True

    def __post_init__(self) -> None:
        c = str(self.codec).lower().strip()
        if c not in VALID_KV_CODECS:
            raise ValueError(f"Unsupported KV codec '{self.codec}'. Expected one of {VALID_KV_CODECS}")
        self.codec = c
        self.group_size = max(8, int(self.group_size))
        self.rope_dim = max(0, int(self.rope_dim))
        self.rope_first = bool(self.rope_first)


@dataclasses.dataclass
class QuantizedKVPage:
    codec: str
    shape: Tuple[int, ...]  # (H_kv, L, D) or (B, H_kv, L, D)
    group_size: int
    rope_dim: int
    rope_first: bool
    payload: Dict[str, Any]
    token_positions: Optional[List[int]] = None

    @property
    def num_tokens(self) -> int:
        return int(self.shape[-2]) if len(self.shape) >= 2 else 0

    @property
    def head_dim(self) -> int:
        return int(self.shape[-1]) if len(self.shape) >= 1 else 0

    @property
    def num_kv_heads(self) -> int:
        if len(self.shape) == 4:
            return int(self.shape[1])
        if len(self.shape) == 3:
            return int(self.shape[0])
        return 1

    def ideal_payload_bytes(self) -> int:
        """Ideal payload bytes (excluding scale/zero metadata) matching §6.1 of the Loop72 LSA plan."""
        b = int(self.shape[0]) if len(self.shape) == 4 else 1
        h = self.num_kv_heads
        n = self.num_tokens
        d = self.head_dim
        if self.codec == "bf16":
            return b * n * h * d * 2 * 2
        if self.codec == "int8":
            return b * n * h * d * 2 * 1
        if self.codec == "int4":
            return (b * n * h * d * 2 + 1) // 2
        if self.codec == "mixed_rotary_bf16":
            r = min(self.rope_dim, d)
            free_d = d - r
            # K: r dims at 2B + free_d dims at 0.5B; V: d dims at 0.5B
            return int(round(b * n * h * (r * 2.0 + free_d * 0.5 + d * 0.5)))
        return 0

    def storage_bytes(self) -> int:
        """Actual allocated bytes including scale/zero metadata."""
        if HAS_TORCH:
            total = 0
            for val in self.payload.values():
                if isinstance(val, torch.Tensor):
                    total += int(val.numel() * val.element_size())
            if total > 0:
                return total
        # Pure-Python fallback accounting
        meta_bytes = int(self.payload.get("_meta_bytes", 0))
        return self.ideal_payload_bytes() + meta_bytes


# ==============================================================================
# Torch Quantization / Dequantization Primitives
# ==============================================================================
def _torch_quantize_int8(x: "torch.Tensor", group_size: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
    x_f = x.detach().float()
    orig_shape = x_f.shape
    d = orig_shape[-1]
    g = math.gcd(d, group_size) or d
    grouped = x_f.reshape(-1, d // g, g)
    abs_max = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = (abs_max / 127.0).to(torch.float16)
    q = torch.clamp(torch.round(grouped / scale.float()), -127, 127).to(torch.int8)
    return q.reshape(orig_shape), scale.squeeze(-1)


def _torch_dequantize_int8(
    q: "torch.Tensor", scale: "torch.Tensor", group_size: int, device: Any, dtype: Any
) -> "torch.Tensor":
    orig_shape = q.shape
    d = orig_shape[-1]
    g = math.gcd(d, group_size) or d
    q_g = q.to(device=device, dtype=torch.float32).reshape(-1, d // g, g)
    s_g = scale.to(device=device, dtype=torch.float32).reshape(-1, d // g, 1)
    return (q_g * s_g).reshape(orig_shape).to(dtype=dtype)


def _torch_quantize_int4(x: "torch.Tensor", group_size: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    x_f = x.detach().float()
    orig_shape = x_f.shape
    d = orig_shape[-1]
    if d % 2 != 0:
        raise ValueError(f"int4 packing requires even last dimension, got {d}")
    g = math.gcd(d, group_size) or d
    grouped = x_f.reshape(-1, d // g, g)
    mn = grouped.amin(dim=-1, keepdim=True)
    mx = grouped.amax(dim=-1, keepdim=True)
    scale = ((mx - mn).clamp(min=1e-6) / 15.0).to(torch.float16)
    zero = mn.to(torch.float16)
    q_u4 = torch.clamp(torch.round((grouped - zero.float()) / scale.float()), 0, 15).to(torch.uint8)
    q_flat = q_u4.reshape(*orig_shape[:-1], d)
    packed = (q_flat[..., 0::2] << 4) | (q_flat[..., 1::2] & 0x0F)
    return packed, scale.squeeze(-1), zero.squeeze(-1)


def _torch_dequantize_int4(
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    zero: "torch.Tensor",
    group_size: int,
    device: Any,
    dtype: Any,
) -> "torch.Tensor":
    p_dev = packed.to(device=device)
    hi = (p_dev >> 4) & 0x0F
    lo = p_dev & 0x0F
    unpacked = torch.stack([hi, lo], dim=-1).reshape(*p_dev.shape[:-1], p_dev.shape[-1] * 2).to(torch.float32)
    orig_shape = unpacked.shape
    d = orig_shape[-1]
    g = math.gcd(d, group_size) or d
    u_g = unpacked.reshape(-1, d // g, g)
    s_g = scale.to(device=device, dtype=torch.float32).reshape(-1, d // g, 1)
    z_g = zero.to(device=device, dtype=torch.float32).reshape(-1, d // g, 1)
    return (u_g * s_g + z_g).reshape(orig_shape).to(dtype=dtype)


# ==============================================================================
# Pure-Python Fallback Quantization (for CPU-only test environments without Torch)
# ==============================================================================
def _py_flatten_shape(arr: Any) -> Tuple[List[float], Tuple[int, ...]]:
    shape: List[int] = []
    cur = arr
    while isinstance(cur, (list, tuple)):
        shape.append(len(cur))
        cur = cur[0] if len(cur) > 0 else []
    flat: List[float] = []

    def _walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)
        else:
            flat.append(float(node))

    _walk(arr)
    return flat, tuple(shape)


def _py_unflatten(flat: List[float], shape: Tuple[int, ...]) -> Any:
    if not shape:
        return flat[0] if flat else 0.0
    if len(shape) == 1:
        return list(flat[: shape[0]])
    step = 1
    for s in shape[1:]:
        step *= s
    return [_py_unflatten(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _py_quant_dequant_1d(vals: List[float], bits: int, group_size: int) -> Tuple[List[float], int]:
    n = len(vals)
    g = max(1, min(group_size, n))
    out: List[float] = []
    n_groups = 0
    qmax = (1 << bits) - 1
    for i in range(0, n, g):
        chunk = vals[i : i + g]
        n_groups += 1
        mn = min(chunk)
        mx = max(chunk)
        sc = max(1e-6, (mx - mn) / float(qmax))
        for v in chunk:
            q = max(0, min(qmax, int(round((v - mn) / sc))))
            out.append(mn + q * sc)
    meta_bytes = n_groups * (2 if bits == 8 else 4)
    return out, meta_bytes


def encode_kv_page(
    key: Any,
    value: Any,
    cfg: Optional[KVCodecConfig] = None,
    positions: Optional[Sequence[int]] = None,
    store_device: str = "cpu",
) -> QuantizedKVPage:
    """Encode `key` and `value` (`[H_kv, L, D]` or `[B, H_kv, L, D]`) into a `QuantizedKVPage`."""
    cfg = cfg or KVCodecConfig()
    pos_list = [int(p) for p in positions] if positions is not None else None

    if HAS_TORCH and isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
        k_src = key.detach().to(store_device)
        v_src = value.detach().to(store_device)
        shape = tuple(int(s) for s in k_src.shape)
        if cfg.codec == "bf16":
            k_bf = k_src.to(torch.bfloat16)
            v_bf = v_src.to(torch.bfloat16)
            return QuantizedKVPage(
                codec="bf16",
                shape=shape,
                group_size=cfg.group_size,
                rope_dim=cfg.rope_dim,
                rope_first=cfg.rope_first,
                payload={"k": k_bf, "v": v_bf},
                token_positions=pos_list,
            )
        if cfg.codec == "int8":
            kq, ks = _torch_quantize_int8(k_src, cfg.group_size)
            vq, vs = _torch_quantize_int8(v_src, cfg.group_size)
            return QuantizedKVPage(
                codec="int8",
                shape=shape,
                group_size=cfg.group_size,
                rope_dim=cfg.rope_dim,
                rope_first=cfg.rope_first,
                payload={"kq": kq, "ks": ks, "vq": vq, "vs": vs},
                token_positions=pos_list,
            )
        if cfg.codec == "int4":
            kp, ks, kz = _torch_quantize_int4(k_src, cfg.group_size)
            vp, vs, vz = _torch_quantize_int4(v_src, cfg.group_size)
            return QuantizedKVPage(
                codec="int4",
                shape=shape,
                group_size=cfg.group_size,
                rope_dim=cfg.rope_dim,
                rope_first=cfg.rope_first,
                payload={"kp": kp, "ks": ks, "kz": kz, "vp": vp, "vs": vs, "vz": vz},
                token_positions=pos_list,
            )
        if cfg.codec == "mixed_rotary_bf16":
            d = shape[-1]
            r = min(cfg.rope_dim, d)
            if cfg.rope_first:
                k_rot = k_src[..., :r].to(torch.bfloat16)
                k_free = k_src[..., r:]
            else:
                k_free = k_src[..., : d - r]
                k_rot = k_src[..., d - r :].to(torch.bfloat16)
            kp, ks, kz = _torch_quantize_int4(k_free, cfg.group_size)
            vp, vs, vz = _torch_quantize_int4(v_src, cfg.group_size)
            return QuantizedKVPage(
                codec="mixed_rotary_bf16",
                shape=shape,
                group_size=cfg.group_size,
                rope_dim=cfg.rope_dim,
                rope_first=cfg.rope_first,
                payload={"k_rot": k_rot, "kp": kp, "ks": ks, "kz": kz, "vp": vp, "vs": vs, "vz": vz},
                token_positions=pos_list,
            )

    # Pure-Python fallback
    k_flat, shape = _py_flatten_shape(key)
    v_flat, _ = _py_flatten_shape(value)
    if cfg.codec == "bf16":
        return QuantizedKVPage(
            codec="bf16",
            shape=shape,
            group_size=cfg.group_size,
            rope_dim=cfg.rope_dim,
            rope_first=cfg.rope_first,
            payload={"k_flat": k_flat, "v_flat": v_flat, "_meta_bytes": 0},
            token_positions=pos_list,
        )
    if cfg.codec == "int8":
        kd, mk = _py_quant_dequant_1d(k_flat, 8, cfg.group_size)
        vd, mv = _py_quant_dequant_1d(v_flat, 8, cfg.group_size)
        return QuantizedKVPage(
            codec="int8",
            shape=shape,
            group_size=cfg.group_size,
            rope_dim=cfg.rope_dim,
            rope_first=cfg.rope_first,
            payload={"k_flat": kd, "v_flat": vd, "_meta_bytes": mk + mv},
            token_positions=pos_list,
        )
    if cfg.codec == "int4":
        kd, mk = _py_quant_dequant_1d(k_flat, 4, cfg.group_size)
        vd, mv = _py_quant_dequant_1d(v_flat, 4, cfg.group_size)
        return QuantizedKVPage(
            codec="int4",
            shape=shape,
            group_size=cfg.group_size,
            rope_dim=cfg.rope_dim,
            rope_first=cfg.rope_first,
            payload={"k_flat": kd, "v_flat": vd, "_meta_bytes": mk + mv},
            token_positions=pos_list,
        )
    # mixed_rotary_bf16 in pure Python
    d = shape[-1]
    r = min(cfg.rope_dim, d)
    kd: List[float] = []
    mk_total = 0
    for row_start in range(0, len(k_flat), d):
        row = k_flat[row_start : row_start + d]
        if cfg.rope_first:
            rot_part = row[:r]
            free_part, m_r = _py_quant_dequant_1d(row[r:], 4, cfg.group_size)
            kd.extend(rot_part + free_part)
        else:
            free_part, m_r = _py_quant_dequant_1d(row[: d - r], 4, cfg.group_size)
            rot_part = row[d - r :]
            kd.extend(free_part + rot_part)
        mk_total += m_r
    vd, mv = _py_quant_dequant_1d(v_flat, 4, cfg.group_size)
    return QuantizedKVPage(
        codec="mixed_rotary_bf16",
        shape=shape,
        group_size=cfg.group_size,
        rope_dim=cfg.rope_dim,
        rope_first=cfg.rope_first,
        payload={"k_flat": kd, "v_flat": vd, "_meta_bytes": mk_total + mv},
        token_positions=pos_list,
    )


def decode_kv_page(
    page: QuantizedKVPage,
    device: Any = None,
    dtype: Any = None,
) -> Tuple[Any, Any]:
    """Decode a `QuantizedKVPage` back to `(key, value)` tensors on `device` with `dtype`."""
    if HAS_TORCH and ("k" in page.payload or "kq" in page.payload or "kp" in page.payload):
        target_dtype = dtype if dtype is not None else torch.bfloat16
        target_dev = device if device is not None else "cpu"
        if page.codec == "bf16":
            return (
                page.payload["k"].to(device=target_dev, dtype=target_dtype),
                page.payload["v"].to(device=target_dev, dtype=target_dtype),
            )
        if page.codec == "int8":
            k = _torch_dequantize_int8(
                page.payload["kq"], page.payload["ks"], page.group_size, target_dev, target_dtype
            )
            v = _torch_dequantize_int8(
                page.payload["vq"], page.payload["vs"], page.group_size, target_dev, target_dtype
            )
            return k, v
        if page.codec == "int4":
            k = _torch_dequantize_int4(
                page.payload["kp"], page.payload["ks"], page.payload["kz"], page.group_size, target_dev, target_dtype
            )
            v = _torch_dequantize_int4(
                page.payload["vp"], page.payload["vs"], page.payload["vz"], page.group_size, target_dev, target_dtype
            )
            return k, v
        if page.codec == "mixed_rotary_bf16":
            k_rot = page.payload["k_rot"].to(device=target_dev, dtype=target_dtype)
            k_free = _torch_dequantize_int4(
                page.payload["kp"], page.payload["ks"], page.payload["kz"], page.group_size, target_dev, target_dtype
            )
            k = torch.cat([k_rot, k_free], dim=-1) if page.rope_first else torch.cat([k_free, k_rot], dim=-1)
            v = _torch_dequantize_int4(
                page.payload["vp"], page.payload["vs"], page.payload["vz"], page.group_size, target_dev, target_dtype
            )
            return k, v

    return (
        _py_unflatten(page.payload["k_flat"], page.shape),
        _py_unflatten(page.payload["v_flat"], page.shape),
    )


def measure_codec_error(key: Any, value: Any, cfg: KVCodecConfig) -> Dict[str, float]:
    """Measure separate key (rotary vs free slice) and value relative L2 reconstruction error and byte savings."""
    page = encode_kv_page(key, value, cfg)
    k_hat, v_hat = decode_kv_page(page, dtype=getattr(key, "dtype", None))

    if HAS_TORCH and isinstance(key, torch.Tensor):
        kf, vf = key.detach().float(), value.detach().float()
        khf, vhf = k_hat.detach().float().to(kf.device), v_hat.detach().float().to(vf.device)
        k_err = float((kf - khf).norm() / kf.norm().clamp(min=1e-8))
        v_err = float((vf - vhf).norm() / vf.norm().clamp(min=1e-8))
        r = min(cfg.rope_dim, kf.shape[-1])
        if cfg.rope_first:
            k_rot_err = float((kf[..., :r] - khf[..., :r]).norm() / kf[..., :r].norm().clamp(min=1e-8))
            k_free_err = float((kf[..., r:] - khf[..., r:]).norm() / kf[..., r:].norm().clamp(min=1e-8))
        else:
            d = kf.shape[-1]
            k_free_err = float((kf[..., : d - r] - khf[..., : d - r]).norm() / kf[..., : d - r].norm().clamp(min=1e-8))
            k_rot_err = float((kf[..., d - r :] - khf[..., d - r :]).norm() / kf[..., d - r :].norm().clamp(min=1e-8))
    else:
        kf, shape = _py_flatten_shape(key)
        vf, _ = _py_flatten_shape(value)
        khf, _ = _py_flatten_shape(k_hat)
        vhf, _ = _py_flatten_shape(v_hat)
        kn = math.sqrt(sum(x * x for x in kf)) or 1e-8
        vn = math.sqrt(sum(x * x for x in vf)) or 1e-8
        k_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(kf, khf))) / kn
        v_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(vf, vhf))) / vn
        d = shape[-1]
        r = min(cfg.rope_dim, d)
        rot_diff_sq, rot_norm_sq, free_diff_sq, free_norm_sq = 0.0, 0.0, 0.0, 0.0
        for s_i in range(0, len(kf), d):
            for j in range(d):
                is_rot = (j < r) if cfg.rope_first else (j >= d - r)
                diff_sq = (kf[s_i + j] - khf[s_i + j]) ** 2
                val_sq = kf[s_i + j] ** 2
                if is_rot:
                    rot_diff_sq += diff_sq
                    rot_norm_sq += val_sq
                else:
                    free_diff_sq += diff_sq
                    free_norm_sq += val_sq
        k_rot_err = math.sqrt(rot_diff_sq) / max(1e-8, math.sqrt(rot_norm_sq))
        k_free_err = math.sqrt(free_diff_sq) / max(1e-8, math.sqrt(free_norm_sq))

    bf16_bytes = QuantizedKVPage(
        codec="bf16",
        shape=page.shape,
        group_size=cfg.group_size,
        rope_dim=cfg.rope_dim,
        rope_first=cfg.rope_first,
        payload={},
    ).ideal_payload_bytes()
    actual_bytes = page.storage_bytes()
    ideal_bytes = page.ideal_payload_bytes()
    return {
        "codec": cfg.codec,
        "key_rel_err": round(k_err, 6),
        "key_rotary_rel_err": round(k_rot_err, 6),
        "key_free_rel_err": round(k_free_err, 6),
        "val_rel_err": round(v_err, 6),
        "bf16_bytes": bf16_bytes,
        "ideal_payload_bytes": ideal_bytes,
        "actual_storage_bytes": actual_bytes,
        "ideal_compression_ratio": round(bf16_bytes / max(1, ideal_bytes), 4),
        "actual_compression_ratio": round(bf16_bytes / max(1, actual_bytes), 4),
    }


def compute_loop72_capacity_table(
    n_tokens: int = 5_000_000,
    start: int = 4,
    end: int = 28,
    passes: int = 2,
    enable_p3: bool = True,
    p3_start: int = 12,
    p3_end: int = 24,
    n_layers: int = 36,
    h_kv: int = 4,
    head_dim: int = 256,
    rope_dim: int = 64,
    window: int = 512,
    d_idx: int = 64,
) -> Dict[str, Any]:
    """Reproduce the exact Loop72 visit counts, KV capacity budgets, and PCIe transfer bounds from the Review Plan."""
    plan = list(range(0, start)) + list(range(start, end)) * passes
    if enable_p3 and passes >= 2:
        plan = plan + list(range(p3_start, p3_end))
    plan = plan + list(range(end, n_layers))

    global_physical = set(range(3, n_layers, 4))
    global_visits = sum(layer in global_physical for layer in plan)
    local_visits = len(plan) - global_visits
    global_counts = dict(sorted(Counter(layer for layer in plan if layer in global_physical).items()))

    gib = float(2**30)
    mib = float(2**20)
    unbounded_all_72_bf16_gib = (n_tokens * len(plan) * 2 * h_kv * head_dim * 2) / gib
    payload_by_bits_gib = {}
    for bits in (16, 8, 4, 2):
        g_bytes = n_tokens * global_visits * 2 * h_kv * head_dim * bits / 8.0
        payload_by_bits_gib[f"{bits}bit_gib"] = round(g_bytes / gib, 2)

    local_rings_mib = (local_visits * min(n_tokens, window) * 2 * h_kv * head_dim * 2) / mib
    index_bytes_per_token_visit = (64 * 2 + 16 * 4 + 4) / 8.0 + 4.0 / 8192.0
    flat_index_gib = (n_tokens * global_visits * index_bytes_per_token_visit) / gib
    map_params = global_visits * 2 * (h_kv * (head_dim - rope_dim)) * d_idx
    mixed_bytes_per_token_visit = h_kv * (rope_dim * 2.0 + (head_dim - rope_dim) * 0.5 + head_dim * 0.5)
    mixed_codec_gib = (n_tokens * global_visits * mixed_bytes_per_token_visit) / gib

    transfer_table = {}
    for k_read in (2048, 4096):
        for bits in (16, 4):
            moved = global_visits * k_read * 2 * h_kv * head_dim * bits / 8.0
            transfer_table[f"k{k_read}_{bits}bit"] = {
                "mib_per_token": round(moved / mib, 2),
                "ms_at_25gbps": round((moved / 25e9) * 1000.0, 2),
            }

    return {
        "total_visits": len(plan),
        "global_visits": global_visits,
        "local_visits": local_visits,
        "global_visit_counts_by_layer": global_counts,
        "n_tokens": n_tokens,
        "unbounded_all72_bf16_gib": round(unbounded_all_72_bf16_gib, 2),
        "global_payload_gib": payload_by_bits_gib,
        "mixed_rotary_bf16_gib": round(mixed_codec_gib, 2),
        "local_rings_mib": round(local_rings_mib, 2),
        "flat_index_gib": round(flat_index_gib, 3),
        "selector_map_parameters": int(map_params),
        "transfer_bounds": transfer_table,
    }
