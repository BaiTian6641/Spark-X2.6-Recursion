"""train_spark_dlsa.py — Per-Visit Streamed PCA Initialization, Sampled-Query KL Distillation & 4-Way Positional Evaluation for Spark X2.6 4B Loop72 LSA (Work Packages P5 & P8).

Implements §8.1–§8.4 and §9.2 of the Spark X2.6 4B Loop72 LSA Review & Development Plan:
  1. Per-global-visit (18 visits: `{3:1, 7:2, 11:2, 15:3, 19:3, 23:3, 27:2, 31:1, 35:1}`) streamed PCA initialization
     from the 192-D RoPE-free key slice (`[64:256)` across `H_kv = 4` heads -> `768` features -> `64` latent dims).
  2. Sampled-query teacher-student KL + focal-KL distillation (`sampled_queries <= 32` per visit per sequence) without
     materializing full `[T, T]` attention matrices, keeping backbone and mHC weights 100% frozen (`1,769,472` selector
     parameters total across all 18 global visits).
  3. 4-way positional & sparse ablation matrix (`Dense_NativeRoPE`, `Sparse_NativeRoPE`, `Dense_YaRNRoPE`, `Sparse_YaRNRoPE`)
     separating YaRN partial-RoPE extrapolation from D-LSA sparse selection error.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

if HAS_TORCH:
    from sparkx_dlsa import LOOP72_GLOBAL_VISIT_SCHEDULE, VisitAwareDLSAIndexBank


@dataclasses.dataclass
class VisitDistillConfig:
    """Configuration for 18-visit Loop72 D-LSA selector training (Work Package P5)."""
    d_idx: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    steps: int = 200
    sampled_queries: int = 32
    tau_teacher: float = 1.0
    tau_student: float = 0.08
    focal_gamma: float = 1.5
    focal_weight: float = 0.35
    log_distance_prior_alpha: float = 0.02
    seed: int = 42


def capture_loop72_global_visit_qkv(
    model: Any,
    input_ids: "torch.LongTensor",
    visit_indices: Optional[Sequence[int]] = None,
) -> Dict[int, Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
    """Run a single no-grad Loop72 forward pass and capture `(Q, K, V)` (`[H, T, D]`) at requested global visits `0..17`."""
    if not HAS_TORCH:
        raise RuntimeError("capture_loop72_global_visit_qkv requires PyTorch.")
    target_set = (
        set(range(len(LOOP72_GLOBAL_VISIT_SCHEDULE)))
        if visit_indices is None
        else {int(v) for v in visit_indices}
    )
    layers = model.base.model.layers
    global_layers = sorted({int(v.physical_layer) for v in LOOP72_GLOBAL_VISIT_SCHEDULE})
    visit_counters: Dict[int, int] = {l_idx: 0 for l_idx in global_layers}
    captured: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    hooks = []

    from sparkx_dlsa import GLOBAL_VISIT_BY_LAYER_PASS
    layer_occ_to_gv: Dict[Tuple[int, int], int] = {
        (int(v.physical_layer), int(v.visit_occurrence)): int(v.global_visit_idx)
        for v in LOOP72_GLOBAL_VISIT_SCHEDULE
    }

    def _make_hook(l_idx: int):
        def _hook(mod, _args, _kwargs, _out):
            occ = visit_counters[l_idx]
            visit_counters[l_idx] = occ + 1
            gv_idx = layer_occ_to_gv.get((l_idx, occ))
            if gv_idx is None or gv_idx not in target_set:
                return
            qkv = getattr(mod, "_sleep_last_qkv", None)
            if qkv is not None:
                q_t, k_t, v_t = qkv
                captured[gv_idx] = (
                    q_t[0].detach(),
                    k_t[0].detach(),
                    v_t[0].detach(),
                )
        return _hook

    for l_idx in global_layers:
        attn = layers[l_idx].self_attn
        attn._capture_sleep_qkv = True
        attn._sleep_last_qkv = None
        hooks.append(attn.register_forward_hook(_make_hook(l_idx), with_kwargs=True))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _ = model.forward_hidden(input_ids)
    finally:
        for h in hooks:
            h.remove()
        for l_idx in global_layers:
            layers[l_idx].self_attn._capture_sleep_qkv = False
            layers[l_idx].self_attn._sleep_last_qkv = None
        if was_training:
            model.train()
    return captured


@torch.no_grad()
def fit_loop72_visit_bank_pca(
    bank: "VisitAwareDLSAIndexBank",
    visit_qkv_samples: Dict[int, Sequence[Tuple["torch.Tensor", "torch.Tensor"]]],
) -> Dict[int, Dict[str, Any]]:
    """Initialize each of the 18 global-visit `FaithfulDLSAIndexer` maps via streamed SVD/PCA on RoPE-free key slices
    followed by closed-form Procrustes query alignment (`W_q` aligned to `W_k` on top-attended keys).
    """
    if not HAS_TORCH:
        raise RuntimeError("fit_loop72_visit_bank_pca requires PyTorch.")
    report: Dict[int, Dict[str, Any]] = {}
    for gv_idx in range(len(LOOP72_GLOBAL_VISIT_SCHEDULE)):
        indexer = bank.get_visit_indexer(gv_idx)
        samples = visit_qkv_samples.get(gv_idx, [])
        if not samples:
            report[gv_idx] = {"status": "skipped_no_samples"}
            continue
        k_cat = torch.cat([k_s for _, k_s in samples], dim=1)  # [H_kv, T_tot, D]
        indexer.fit_pca_init(k_cat)

        # Align W_q to W_k via ridge regression on sampled query -> attended key latent expectation
        q_feats_list: List[torch.Tensor] = []
        k_lat_targets: List[torch.Tensor] = []
        for q_s, k_s in samples:
            h_q, t_seq, d_h = q_s.shape
            h_kv = k_s.shape[0]
            g_sz = h_q // h_kv
            k_lat = indexer.encode_keys(k_s)  # [T, d_idx]
            step = max(1, t_seq // 32)
            q_positions = list(range(min(t_seq - 1, 16), t_seq, step))[:32]
            if not q_positions:
                q_positions = [t_seq - 1]
            q_sub = q_s[:, q_positions, :]  # [H_q, S, D]
            qf = indexer._group_free_query_features(q_sub)  # [S, H_kv * d_free]
            scale = 1.0 / (d_h ** 0.5)
            q_g = q_sub.view(h_kv, g_sz, len(q_positions), d_h).float()
            k_f = k_s.float()
            for s_i, p_tok in enumerate(q_positions):
                upto = p_tok + 1
                logits = torch.einsum("gdh,td->gt", q_g[:, :, s_i, :], k_f[:, :upto, :]) * scale
                probs = torch.softmax(logits, dim=-1).mean(dim=0)  # [upto]
                target_lat = F.normalize(probs @ k_lat[:upto].float(), dim=-1, eps=1e-6)
                q_feats_list.append(qf[s_i].float())
                k_lat_targets.append(target_lat)

        if q_feats_list:
            x_mat = torch.stack(q_feats_list, dim=0)  # [M, 768]
            y_mat = torch.stack(k_lat_targets, dim=0)  # [M, 64]
            gram = x_mat.t() @ x_mat + 1e-2 * torch.eye(x_mat.shape[1], device=x_mat.device, dtype=torch.float32)
            w_q_T = torch.linalg.solve(gram, x_mat.t() @ y_mat)  # [768, 64]
            indexer.W_q.weight.data.copy_(w_q_T.t().to(indexer.W_q.weight.dtype))

        report[gv_idx] = {
            "status": "ok",
            "total_tokens": int(k_cat.shape[1]),
            "aligned_queries": len(q_feats_list),
        }
    return report


def distill_single_visit_indexer(
    indexer: Any,
    qkv_samples: Sequence[Tuple["torch.Tensor", "torch.Tensor"]],
    cfg: Optional[VisitDistillConfig] = None,
) -> Dict[str, Any]:
    """Train a single visit's `(W_k, W_q)` (`98,304` params) with sampled-query KL + focal-KL distillation."""
    if not HAS_TORCH:
        raise RuntimeError("distill_single_visit_indexer requires PyTorch.")
    cfg = cfg or VisitDistillConfig(d_idx=int(indexer.cfg.d_idx))
    for p in indexer.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(indexer.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    loss_first: Optional[float] = None
    loss_last: float = 0.0
    for step_i in range(max(1, int(cfg.steps))):
        q_s, k_s = qkv_samples[step_i % len(qkv_samples)]
        h_q, t_seq, d_h = q_s.shape
        h_kv = k_s.shape[0]
        g_sz = h_q // h_kv
        if t_seq < 4:
            continue

        n_samp = min(int(cfg.sampled_queries), t_seq - 1)
        perm = torch.randperm(t_seq - 1, generator=g)[:n_samp] + 1
        q_pos = perm.sort().values.to(q_s.device)

        k_lat = indexer.encode_keys(k_s)  # [T, d_idx]
        q_lat = indexer.encode_queries(q_s[:, q_pos, :])  # [S, d_idx]

        scale = 1.0 / (d_h ** 0.5)
        q_g = q_s[:, q_pos, :].view(h_kv, g_sz, n_samp, d_h).float()
        k_f = k_s.float()

        step_loss = torch.tensor(0.0, device=q_s.device, dtype=torch.float32)
        for s_i in range(n_samp):
            p_tok = int(q_pos[s_i].item())
            upto = p_tok + 1
            with torch.no_grad():
                # Teacher per-key probability mass across all query heads
                t_logits = torch.einsum("gdh,gid->gti", q_g[:, :, s_i, :].unsqueeze(-1), k_f[:, :upto, :].unsqueeze(1)).squeeze(-1)
                # t_logits: [H_kv, g_sz, upto]
                t_probs = torch.softmax((t_logits * scale) / max(1e-4, cfg.tau_teacher), dim=-1).mean(dim=(0, 1))
                t_probs = t_probs.clamp(min=1e-8)
                t_probs = t_probs / t_probs.sum()

            s_logits = (k_lat[:upto] @ q_lat[s_i]) / max(1e-4, cfg.tau_student)
            if cfg.log_distance_prior_alpha > 0.0:
                dist = (p_tok - torch.arange(upto, device=q_s.device, dtype=torch.float32)).clamp(min=0.0)
                s_logits = s_logits - float(cfg.log_distance_prior_alpha) * torch.log1p(dist)
            log_s = F.log_softmax(s_logits, dim=-1)
            kl = F.kl_div(log_s, t_probs, reduction="sum")
            if cfg.focal_weight > 0.0:
                s_probs = log_s.exp()
                focal_mod = (1.0 - s_probs).clamp(min=0.0) ** float(cfg.focal_gamma)
                focal_kl = (focal_mod * t_probs * (t_probs.log() - log_s)).sum()
                kl = (1.0 - float(cfg.focal_weight)) * kl + float(cfg.focal_weight) * focal_kl
            step_loss = step_loss + kl / float(n_samp)

        opt.zero_grad()
        step_loss.backward()
        opt.step()
        val = float(step_loss.detach().item())
        if loss_first is None:
            loss_first = val
        loss_last = val

    for p in indexer.parameters():
        p.requires_grad_(False)
    return {
        "steps": int(cfg.steps),
        "loss_first": float(loss_first or 0.0),
        "loss_last": float(loss_last),
    }


def distill_loop72_visit_bank(
    bank: "VisitAwareDLSAIndexBank",
    visit_qkv_samples: Dict[int, Sequence[Tuple["torch.Tensor", "torch.Tensor"]]],
    cfg: Optional[VisitDistillConfig] = None,
) -> Dict[str, Any]:
    """Run PCA initialization + sampled-query KL distillation across all 18 global visits of `VisitAwareDLSAIndexBank`."""
    if not HAS_TORCH:
        raise RuntimeError("distill_loop72_visit_bank requires PyTorch.")
    t0 = time.perf_counter()
    pca_rep = fit_loop72_visit_bank_pca(bank, visit_qkv_samples)
    distill_rep: Dict[int, Dict[str, Any]] = {}
    for gv_idx in range(len(LOOP72_GLOBAL_VISIT_SCHEDULE)):
        samples = visit_qkv_samples.get(gv_idx, [])
        if not samples:
            distill_rep[gv_idx] = {"status": "skipped_no_samples"}
            continue
        indexer = bank.get_visit_indexer(gv_idx)
        distill_rep[gv_idx] = distill_single_visit_indexer(indexer, samples, cfg=cfg)
    return {
        "status": "ok",
        "num_global_visits": len(LOOP72_GLOBAL_VISIT_SCHEDULE),
        "total_map_parameters": bank.total_map_parameters(),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "pca_init": pca_rep,
        "distill": distill_rep,
    }


@torch.no_grad()
def run_loop72_four_way_eval(
    model: Any,
    input_ids: "torch.LongTensor",
    yarn_scale_factor: float = 4.0,
) -> Dict[str, Any]:
    """Run the 4-way positional/sparse ablation matrix from §8.4 of the Loop72 LSA Plan:
      1. `Dense_NativeRoPE`: Exact global attention + native partial RoPE (`yarn_scale_factor = 1.0`).
      2. `Sparse_NativeRoPE`: Paged D-LSA sparse global attention + native partial RoPE (`yarn_scale_factor = 1.0`).
      3. `Dense_YaRNRoPE`: Exact global attention + YaRN-scaled partial RoPE (`yarn_scale_factor > 1.0`).
      4. `Sparse_YaRNRoPE`: Paged D-LSA sparse global attention + YaRN-scaled partial RoPE (`yarn_scale_factor > 1.0`).
    """
    if not HAS_TORCH:
        raise RuntimeError("run_loop72_four_way_eval requires PyTorch.")
    sparse_cfg = getattr(model, "sparse_attention_cfg", None)
    if sparse_cfg is None:
        raise RuntimeError("Model must be configured with attach_loop72_dlsa_attention before running 4-way eval.")

    saved_thresh = int(sparse_cfg.exact_threshold)
    saved_yarn = float(sparse_cfg.yarn_scale_factor)
    seq_len = int(input_ids.shape[1])
    results: Dict[str, Dict[str, float]] = {}

    modes = [
        ("Dense_NativeRoPE", seq_len + 1024, 1.0),
        ("Sparse_NativeRoPE", saved_thresh, 1.0),
        ("Dense_YaRNRoPE", seq_len + 1024, float(yarn_scale_factor)),
        ("Sparse_YaRNRoPE", saved_thresh, float(yarn_scale_factor)),
    ]
    ref_logits: Optional[torch.Tensor] = None
    try:
        for name, thresh, y_scale in modes:
            sparse_cfg.exact_threshold = int(thresh)
            sparse_cfg.yarn_scale_factor = float(y_scale)
            cache = model.make_inference_cache(session_id=f"eval_{name}")
            t0 = time.perf_counter()
            logits = model.forward_logits(
                input_ids,
                past_key_values=cache,
                chunk_size=getattr(model.settings, "prefill_chunk_tokens", 2048),
            ).float()
            dt = time.perf_counter() - t0
            if input_ids.shape[1] >= 2:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                nll = float(
                    F.cross_entropy(
                        shift_logits.view(-1, shift_logits.shape[-1]),
                        shift_labels.view(-1),
                    ).item()
                )
            else:
                nll = 0.0
            if ref_logits is None:
                ref_logits = logits.detach().clone()
                logit_diff = 0.0
            else:
                logit_diff = float((logits - ref_logits).abs().mean().item())
            results[name] = {
                "nll": round(nll, 6),
                "mean_abs_logit_diff_vs_dense_native": round(logit_diff, 6),
                "latency_sec": round(dt, 4),
            }
    finally:
        sparse_cfg.exact_threshold = saved_thresh
        sparse_cfg.yarn_scale_factor = saved_yarn

    return {
        "seq_len": seq_len,
        "yarn_scale_factor": float(yarn_scale_factor),
        "modes": results,
    }
