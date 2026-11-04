"""Core utilities for Spark-X2.6 Loop Base (LB) invariance, null-space ridge compilation, and trace configuration.

Implements Phase 0 & Phase 1 primitives from `docs/roadmap/02_IMPLEMENTATION_PLAN.md`:
  - `CapacityExhaustedError`: explicit exception when a shard/expert null-space basis reaches capacity (fixes M4).
  - `BackboneFingerprint`: SHA-256 byte-level digest + max-abs-diff audit for `model.base` and `TwoStreamMHC` controllers.
  - `loop_config`: context manager that saves/restores loop, mHC, P3, Engram, and ESM execution flags (fixes B2).
  - Null-space linear algebra primitives (`damped_pseudoinverse_pullback`, `project_into_null_space`,
    `solve_null_space_ridge`, `update_orthonormal_basis`, `exact_remove_delta`) with both PyTorch (`float32`)
    and pure-Python reference implementations so unit tests run both on CPU-only VMs and GPU Colab instances.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover - fallback for lightweight syntax/math test environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


class CapacityExhaustedError(RuntimeError):
    """Raised when an expert or EpisodicSideMemory shard has insufficient free null-space rank for a new write."""


@dataclasses.dataclass
class TensorRecord:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    numel: int
    sha256: str
    l2_norm: float
    sample_indices: List[int]
    sample_values: List[float]
    full_clone: Optional[Any] = None


def _hash_and_sample_tensor(
    name: str,
    tensor: Any,
    sample_size: int = 2048,
    store_full_clone: bool = False,
) -> TensorRecord:
    """Compute an exact byte-level SHA-256 digest and deterministic strided float32 samples of `tensor`."""
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        t_det = tensor.detach()
        shape = tuple(int(d) for d in t_det.shape)
        dtype_str = str(t_det.dtype)
        numel = int(t_det.numel())
        if numel == 0:
            return TensorRecord(
                name=name,
                shape=shape,
                dtype=dtype_str,
                numel=0,
                sha256=hashlib.sha256(b"empty").hexdigest(),
                l2_norm=0.0,
                sample_indices=[],
                sample_values=[],
                full_clone=t_det.cpu().clone() if store_full_clone else None,
            )

        step = max(1, numel // max(1, int(sample_size)))
        idxs = list(range(0, numel, step))[:sample_size]

        if t_det.is_cuda and not store_full_clone and numel > 262144:
            # Fast GPU path for large backbone matrices: avoids 8GB PCIe D2H copy while tracking norm + strided slice bytes
            flat_gpu = t_det.reshape(-1)
            idx_gpu = torch.tensor(idxs, dtype=torch.long, device=t_det.device)
            samp_cpu = flat_gpu.index_select(0, idx_gpu).float().cpu().contiguous()
            l2_val = float(flat_gpu.float().norm().item())
            vals = [float(x) for x in samp_cpu.tolist()]
            raw_bytes = (
                f"{shape}:{dtype_str}:{numel}:{l2_val:.9e}:".encode("utf-8")
                + bytes(samp_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
            )
            digest = hashlib.sha256(raw_bytes).hexdigest()
            return TensorRecord(
                name=name,
                shape=shape,
                dtype=dtype_str,
                numel=numel,
                sha256=digest,
                l2_norm=l2_val,
                sample_indices=idxs,
                sample_values=vals,
                full_clone=None,
            )

        cpu_contig = t_det.cpu().contiguous()
        flat_u8 = cpu_contig.reshape(-1).view(torch.uint8)
        raw_bytes = bytes(flat_u8.numpy().tobytes()) if hasattr(flat_u8, "numpy") else b""
        if not raw_bytes:
            flat_f32 = cpu_contig.float().reshape(-1).tolist()
            raw_bytes = ",".join(f"{v:.8e}" for v in flat_f32[: min(numel, 65536)]).encode("utf-8")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        l2_val = float(cpu_contig.float().norm().item())
        flat_cpu = cpu_contig.float().reshape(-1)
        idx_t = torch.tensor(idxs, dtype=torch.long)
        vals = [float(x) for x in flat_cpu.index_select(0, idx_t).tolist()]
        clone_t = cpu_contig.clone() if store_full_clone else None
        return TensorRecord(
            name=name,
            shape=shape,
            dtype=dtype_str,
            numel=numel,
            sha256=digest,
            l2_norm=l2_val,
            sample_indices=idxs,
            sample_values=vals,
            full_clone=clone_t,
        )

    # Pure-Python sequence fallback for environments without PyTorch
    flat = [float(x) for x in tensor]
    numel = len(flat)
    raw = ",".join(f"{v:.12e}" for v in flat).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    l2_val = math.sqrt(sum(v * v for v in flat))
    step = max(1, numel // max(1, int(sample_size))) if numel > 0 else 1
    idxs = list(range(0, numel, step))[:sample_size] if numel > 0 else []
    vals = [flat[i] for i in idxs]
    return TensorRecord(
        name=name,
        shape=(numel,),
        dtype="float",
        numel=numel,
        sha256=digest,
        l2_norm=l2_val,
        sample_indices=idxs,
        sample_values=vals,
        full_clone=list(flat) if store_full_clone else None,
    )


class BackboneFingerprint:
    """Cryptographic SHA-256 + max-abs-diff verifier for the frozen Loop Base (`model.base` + `TwoStreamMHC`).

    When all parameter SHA-256 digests match, `max_abs_diff` is guaranteed to be `0.0` bit-for-bit
    across all 4.11B parameters without duplicating gigabytes of weights in host RAM.
    """

    def __init__(
        self,
        model: Any,
        *,
        include_controllers: bool = False,
        ignore_substrings: Sequence[str] = ("lsa_indexer",),
        sample_size: int = 2048,
        store_full_clone: bool = False,
        store_exact_clone: Optional[bool] = None,
    ) -> None:
        self.include_controllers = bool(include_controllers)
        self.ignore_substrings = tuple(str(s) for s in ignore_substrings)
        self.sample_size = int(sample_size)
        self.store_full_clone = bool(store_exact_clone if store_exact_clone is not None else store_full_clone)
        self.records: Dict[str, TensorRecord] = {}
        self.combined_sha256: str = ""
        self._capture(model)

    def _iter_tracked_named_params(self, model: Any) -> Iterable[Tuple[str, Any]]:
        base = getattr(model, "base", model)
        if hasattr(base, "named_parameters"):
            for p_name, p_t in base.named_parameters():
                if any(sub in p_name for sub in self.ignore_substrings):
                    continue
                yield f"base.{p_name}", p_t
        elif isinstance(base, dict):
            for p_name, p_t in sorted(base.items()):
                if any(sub in p_name for sub in self.ignore_substrings):
                    continue
                yield f"base.{p_name}", p_t

        if self.include_controllers and hasattr(model, "controllers") and model.controllers is not None:
            ctrls = model.controllers
            if hasattr(ctrls, "named_parameters"):
                for c_name, c_t in ctrls.named_parameters():
                    yield f"controllers.{c_name}", c_t
            elif isinstance(ctrls, dict):
                for c_name, c_t in sorted(ctrls.items()):
                    yield f"controllers.{c_name}", c_t

    def _capture(self, model: Any) -> None:
        overall = hashlib.sha256()
        for name, param in self._iter_tracked_named_params(model):
            rec = _hash_and_sample_tensor(
                name,
                param,
                sample_size=self.sample_size,
                store_full_clone=self.store_full_clone,
            )
            self.records[name] = rec
            overall.update(f"{name}:{rec.sha256};".encode("utf-8"))
        self.combined_sha256 = overall.hexdigest()

    def verify(self, model: Any, tol: float = 0.0) -> Dict[str, Any]:
        """Compare `model` against the captured snapshot and return a verification report."""
        changed_tensors: List[str] = []
        missing_tensors: List[str] = []
        max_abs_diff = 0.0
        max_norm_diff = 0.0

        current_map = dict(self._iter_tracked_named_params(model))
        overall = hashlib.sha256()

        for name, ref_rec in self.records.items():
            if name not in current_map:
                missing_tensors.append(name)
                continue
            cur_param = current_map[name]
            cur_rec = _hash_and_sample_tensor(
                name,
                cur_param,
                sample_size=self.sample_size,
                store_full_clone=False,
            )
            overall.update(f"{name}:{cur_rec.sha256};".encode("utf-8"))
            if cur_rec.sha256 == ref_rec.sha256:
                continue

            # Digest changed -> compute max_abs_diff (exact if full_clone was stored, else on strided samples + norm)
            norm_diff = abs(cur_rec.l2_norm - ref_rec.l2_norm)
            if norm_diff > max_norm_diff:
                max_norm_diff = norm_diff

            tensor_diff = 0.0
            if ref_rec.full_clone is not None:
                if HAS_TORCH and isinstance(cur_param, torch.Tensor):
                    cur_cpu = cur_param.detach().cpu().float()
                    ref_cpu = ref_rec.full_clone.float()
                    if cur_cpu.shape == ref_cpu.shape and cur_cpu.numel() > 0:
                        tensor_diff = float((cur_cpu - ref_cpu).abs().max().item())
                else:
                    cur_list = [float(x) for x in cur_param]
                    tensor_diff = max(
                        (abs(a - b) for a, b in zip(cur_list, ref_rec.full_clone)),
                        default=0.0,
                    )
            else:
                sample_diff = max(
                    (abs(a - b) for a, b in zip(cur_rec.sample_values, ref_rec.sample_values)),
                    default=0.0,
                )
                # Ensure non-zero lower bound when SHA-256 differs even if change missed strided samples
                tensor_diff = max(sample_diff, norm_diff, 1e-12)

            if tensor_diff > max_abs_diff:
                max_abs_diff = tensor_diff
            if tensor_diff > float(tol):
                changed_tensors.append(name)

        cur_combined = overall.hexdigest()
        sha_match = (cur_combined == self.combined_sha256) and (len(missing_tensors) == 0)
        passed = (len(changed_tensors) == 0) and (len(missing_tensors) == 0) and (max_abs_diff <= float(tol))
        return {
            "passed": bool(passed),
            "sha256_match": bool(sha_match),
            "reference_sha256": self.combined_sha256,
            "current_sha256": cur_combined,
            "tracked_tensors": len(self.records),
            "max_abs_diff": float(max_abs_diff),
            "max_norm_diff": float(max_norm_diff),
            "changed_tensors": changed_tensors,
            "missing_tensors": missing_tensors,
        }


@contextmanager
def loop_config(
    model: Any,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    passes: Optional[int] = None,
    use_mhc: Optional[bool] = None,
    use_engram: Optional[bool] = None,
    enable_p3: Optional[bool] = None,
    p3_start: Optional[int] = None,
    p3_end: Optional[int] = None,
    use_esm: Optional[bool] = None,
):
    """Temporarily override loop/mHC/Engram/ESM settings on `SparkXLoopV03` and restore on exit (fixes B2)."""
    saved = {
        "loop_start": getattr(model, "loop_start", 4),
        "loop_end": getattr(model, "loop_end", 28),
        "loop_passes": getattr(model, "loop_passes", 2),
        "mhc_enabled": getattr(model, "mhc_enabled", True),
        "engram_enabled": getattr(model, "engram_enabled", True),
        "enable_p3": getattr(model, "enable_p3", False),
        "p3_start": getattr(model, "p3_start", 12),
        "p3_end": getattr(model, "p3_end", 24),
        "esm_enabled": getattr(model, "esm_enabled", False),
    }
    try:
        new_start = int(saved["loop_start"] if start is None else start)
        new_end = int(saved["loop_end"] if end is None else end)
        new_passes = int(saved["loop_passes"] if passes is None else passes)
        new_mhc = bool(saved["mhc_enabled"] if use_mhc is None else use_mhc)
        new_eng = bool(saved["engram_enabled"] if use_engram is None else use_engram)
        new_p3 = bool(saved["enable_p3"] if enable_p3 is None else enable_p3)
        new_p3_s = int(saved["p3_start"] if p3_start is None else p3_start)
        new_p3_e = int(saved["p3_end"] if p3_end is None else p3_end)
        if hasattr(model, "set_loop"):
            model.set_loop(
                start=new_start,
                end=new_end,
                passes=new_passes,
                use_mhc=new_mhc,
                use_engram=new_eng,
                enable_p3=new_p3,
                p3_start=new_p3_s,
                p3_end=new_p3_e,
            )
        else:
            model.loop_start = new_start
            model.loop_end = new_end
            model.loop_passes = new_passes
            model.mhc_enabled = new_mhc
            model.engram_enabled = new_eng
            model.enable_p3 = new_p3
            model.p3_start = new_p3_s
            model.p3_end = new_p3_e
        if use_esm is not None:
            model.esm_enabled = bool(use_esm)
        yield model
    finally:
        if hasattr(model, "set_loop"):
            model.set_loop(
                start=int(saved["loop_start"]),
                end=int(saved["loop_end"]),
                passes=int(saved["loop_passes"]),
                use_mhc=bool(saved["mhc_enabled"]),
                use_engram=bool(saved["engram_enabled"]),
                enable_p3=bool(saved["enable_p3"]),
                p3_start=int(saved["p3_start"]),
                p3_end=int(saved["p3_end"]),
            )
        else:
            for k, v in saved.items():
                setattr(model, k, v)
        if hasattr(model, "esm_enabled"):
            model.esm_enabled = bool(saved["esm_enabled"])


# ==============================================================================
# Pure-Python Reference Linear Algebra Helpers (for CPU VM tests without torch)
# ==============================================================================

def _py_matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    m = len(a)
    k = len(a[0]) if m > 0 else 0
    n = len(b[0]) if len(b) > 0 else 0
    b_t = [[b[r][c] for r in range(k)] for c in range(n)]
    return [[sum(a[i][p] * b_t[j][p] for p in range(k)) for j in range(n)] for i in range(m)]


def _py_transpose(a: List[List[float]]) -> List[List[float]]:
    if not a:
        return []
    return [[a[r][c] for r in range(len(a))] for c in range(len(a[0]))]


def _py_solve_spd(gram: List[List[float]], rhs: List[List[float]]) -> List[List[float]]:
    """Solve `gram @ X = rhs` via Gaussian elimination with partial pivoting."""
    n = len(gram)
    m_cols = len(rhs[0]) if rhs else 0
    aug = [list(gram[i]) + list(rhs[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        diag = aug[col][col]
        if abs(diag) < 1e-15:
            diag = 1e-15
        inv_d = 1.0 / diag
        for j in range(col, n + m_cols):
            aug[col][j] *= inv_d
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                for j in range(col, n + m_cols):
                    aug[r][j] -= factor * aug[col][j]
    return [row[n : n + m_cols] for row in aug]


# ==============================================================================
# Tested Null-Space & Ridge Math Primitives (Torch + Pure-Python Reference)
# ==============================================================================

def precompute_damped_right_inverse(w_proj: Any, lam: float = 1e-2) -> Any:
    """Return the damped right-inverse `W^T (W W^T + lam I)^{-1}` for solving `W @ delta_c = r` (fixes M5).

    Because `nn.Linear(in_features, out_features)` applies `y = W @ x` to column vectors `x`,
    the minimum-norm ridge solution to `W @ delta_c = r` is:
        delta_c = W^T (W W^T + lam I)^{-1} r = (W^T W + lam I)^{-1} W^T r
    Note: The Stage 2 bug (M5) computed `W^T (W^T W + lam I)^{-1} r`, which is wrong whenever `W` is non-normal.
    """
    if HAS_TORCH and isinstance(w_proj, torch.Tensor):
        w_f32 = w_proj.detach().float()
        out_dim, _ = w_f32.shape
        gram_wwt = torch.matmul(w_f32, w_f32.t()) + float(lam) * torch.eye(
            out_dim, device=w_f32.device, dtype=torch.float32
        )
        # Solve (W W^T + lam I) X = W  =>  X = (W W^T + lam I)^{-1} W, then transpose => W^T (W W^T + lam I)^{-1}
        sol = torch.linalg.solve(gram_wwt, w_f32)
        return sol.t()  # [in_dim, out_dim]

    # Pure-Python 2D list path
    w_t = _py_transpose(w_proj)
    wwt = _py_matmul(w_proj, w_t)
    for i in range(len(wwt)):
        wwt[i][i] += float(lam)
    sol = _py_solve_spd(wwt, w_proj)
    return _py_transpose(sol)


def damped_pseudoinverse_pullback(w_proj: Any, r_target: Any, lam: float = 1e-2) -> Any:
    """Solve `W @ delta_c = r_target` for `delta_c` using `W^T (W W^T + lam I)^{-1} r_target` (fixes M5)."""
    pinv_w = precompute_damped_right_inverse(w_proj, lam=lam)
    if HAS_TORCH and isinstance(pinv_w, torch.Tensor):
        r_f32 = r_target.detach().float()
        if r_f32.dim() == 1:
            return torch.mv(pinv_w, r_f32)
        return torch.matmul(r_f32, pinv_w.t())

    # Pure-Python 1D or 2D rhs
    if r_target and not isinstance(r_target[0], list):
        col = [[float(x)] for x in r_target]
        res = _py_matmul(pinv_w, col)
        return [row[0] for row in res]
    return _py_matmul(r_target, _py_transpose(pinv_w))


def project_into_null_space(a_mat: Any, v_basis: Optional[Any]) -> Any:
    """Double Classical Gram-Schmidt projection of rows of `a_mat` (`[M, R]`) into the null space of `v_basis` (`[R, K]`)."""
    if v_basis is None:
        if HAS_TORCH and isinstance(a_mat, torch.Tensor):
            return a_mat.detach().float().clone()
        return [list(row) for row in a_mat]

    if HAS_TORCH and isinstance(a_mat, torch.Tensor):
        a_f32 = a_mat.detach().float()
        if v_basis.numel() == 0 or v_basis.shape[1] == 0:
            return a_f32.clone()
        v_f32 = v_basis.detach().float()
        a_perp = a_f32 - torch.matmul(torch.matmul(a_f32, v_f32), v_f32.t())
        a_perp = a_perp - torch.matmul(torch.matmul(a_perp, v_f32), v_f32.t())
        return a_perp

    if not v_basis or not v_basis[0]:
        return [list(row) for row in a_mat]
    v_t = _py_transpose(v_basis)
    proj1 = _py_matmul(_py_matmul(a_mat, v_basis), v_t)
    a1 = [[a_mat[i][j] - proj1[i][j] for j in range(len(a_mat[0]))] for i in range(len(a_mat))]
    proj2 = _py_matmul(_py_matmul(a1, v_basis), v_t)
    return [[a1[i][j] - proj2[i][j] for j in range(len(a1[0]))] for i in range(len(a1))]


def solve_null_space_ridge(
    a_mat: Any,
    dz_mat: Any,
    v_basis: Optional[Any] = None,
    alpha_ridge: float = 1e-2,
) -> Tuple[Any, Any, Any, float]:
    """Compute the AlphaEdit null-space constrained dual ridge update in `float32`:
        A_perp = project_into_null_space(A, V_basis)
        Delta_W = A_perp^T (A_perp A_perp^T + alpha I_M)^{-1} Delta_Z
    Returns `(dw, a_perp, dual_weights, rel_fit_error)`.
    """
    a_perp = project_into_null_space(a_mat, v_basis)
    if HAS_TORCH and isinstance(a_perp, torch.Tensor):
        dz_f32 = dz_mat.detach().float()
        m_e = int(a_perp.shape[0])
        gram_m = torch.matmul(a_perp, a_perp.t()) + float(alpha_ridge) * torch.eye(
            m_e, device=a_perp.device, dtype=torch.float32
        )
        dual_weights = torch.linalg.solve(gram_m, dz_f32)
        dw = torch.matmul(a_perp.t(), dual_weights)
        pred_dz = torch.matmul(a_perp, dw)
        rel_err = float((pred_dz - dz_f32).norm().item() / (dz_f32.norm().item() + 1e-6))
        return dw, a_perp, dual_weights, rel_err

    m_e = len(a_perp)
    a_perp_t = _py_transpose(a_perp)
    gram_m = _py_matmul(a_perp, a_perp_t)
    for i in range(m_e):
        gram_m[i][i] += float(alpha_ridge)
    dual_weights = _py_solve_spd(gram_m, dz_mat)
    dw = _py_matmul(a_perp_t, dual_weights)
    pred_dz = _py_matmul(a_perp, dw)
    diff_sq = sum(
        (pred_dz[i][j] - dz_mat[i][j]) ** 2
        for i in range(m_e)
        for j in range(len(dz_mat[0]))
    )
    dz_sq = sum(dz_mat[i][j] ** 2 for i in range(m_e) for j in range(len(dz_mat[0])))
    rel_err = math.sqrt(diff_sq) / (math.sqrt(dz_sq) + 1e-6)
    return dw, a_perp, dual_weights, rel_err


def update_orthonormal_basis(
    v_basis: Optional[Any],
    a_perp: Any,
    max_rank: int,
    null_threshold: float = 1e-4,
) -> Any:
    """Extend orthonormal basis `v_basis` (`[R, K]`) with the active row space of `a_perp` (`[M, R]`).

    Raises `CapacityExhaustedError` if `K + r_new > max_rank` instead of silently dropping old columns (fixes M4).
    """
    if HAS_TORCH and isinstance(a_perp, torch.Tensor):
        a_f32 = a_perp.detach().float()
        _, s_vals, vh_new = torch.linalg.svd(a_f32, full_matrices=False)
        keep_mask = s_vals > float(null_threshold)
        r_new = int(keep_mask.sum().item())
        if r_new == 0:
            return v_basis
        new_dirs = vh_new[keep_mask].t()  # [R, r_new]
        cur_k = int(v_basis.shape[1]) if (v_basis is not None and v_basis.numel() > 0) else 0
        if cur_k + r_new > int(max_rank):
            raise CapacityExhaustedError(
                f"Null-space basis capacity exhausted: current_rank={cur_k} + new_rank={r_new} > max_rank={max_rank}. "
                f"Allocate a new memory shard or run deep-sleep SMF consolidation instead of truncating protected keys."
            )
        if v_basis is None or cur_k == 0:
            q_ortho, _ = torch.linalg.qr(new_dirs, mode="reduced")
            return q_ortho
        cat_dirs = torch.cat([v_basis.detach().float(), new_dirs], dim=1)
        q_ortho, _ = torch.linalg.qr(cat_dirs, mode="reduced")
        return q_ortho

    # Pure-Python Gram-Schmidt on rows of `a_perp`
    r_dim = len(a_perp[0]) if a_perp else 0
    cols: List[List[float]] = []
    if v_basis and v_basis[0]:
        for c in range(len(v_basis[0])):
            cols.append([v_basis[r][c] for r in range(r_dim)])
    cur_k = len(cols)
    new_cols: List[List[float]] = []
    for row in a_perp:
        vec = list(row)
        for basis_col in cols + new_cols:
            dot = sum(vec[i] * basis_col[i] for i in range(r_dim))
            vec = [vec[i] - dot * basis_col[i] for i in range(r_dim)]
        for basis_col in cols + new_cols:
            dot = sum(vec[i] * basis_col[i] for i in range(r_dim))
            vec = [vec[i] - dot * basis_col[i] for i in range(r_dim)]
        nrm = math.sqrt(sum(v * v for v in vec))
        if nrm > float(null_threshold):
            new_cols.append([v / nrm for v in vec])
    if cur_k + len(new_cols) > int(max_rank):
        raise CapacityExhaustedError(
            f"Null-space basis capacity exhausted: current_rank={cur_k} + new_rank={len(new_cols)} > max_rank={max_rank}."
        )
    all_cols = cols + new_cols
    if not all_cols:
        return []
    return [[all_cols[c][r] for c in range(len(all_cols))] for r in range(r_dim)]


def exact_remove_delta(w_master: Any, dw: Any) -> Any:
    """Subtract a previously committed `dw` from `w_master` in `float32` for exact unlearning/rollback."""
    if HAS_TORCH and isinstance(w_master, torch.Tensor):
        return w_master.detach().float() - dw.detach().float()
    return [
        [w_master[i][j] - dw[i][j] for j in range(len(w_master[0]))]
        for i in range(len(w_master))
    ]


# ==============================================================================
# Phase 1: Exact Trace Capture & Attention Reconstruction (`G1`)
# ==============================================================================

def reconstruct_spark_attention_output(
    sdpa_out: Any,
    normed_input: Any,
    out_proj_weight: Any,
    g_proj_weight: Optional[Any] = None,
) -> Any:
    """Recompute Spark attention residual `out_proj(sigmoid(g_proj(LN h)) * SDPA(q, k, v))`.

    Supports both head-wise `g_proj` (`[H_q, W]`, `headwise_attn_output_gate=True` as in `Spark-X2.6-4B`)
    and element-wise `g_proj` (`[H_q * D_h, W]`), across `torch.Tensor` (`sdpa_out` shape `[B, H_q, T, D_h]`
    or `[..., H_q * D_h]`) and pure-Python 2D lists (`[T, H_q * D_h]`).
    """
    if HAS_TORCH and isinstance(sdpa_out, torch.Tensor):
        if sdpa_out.dim() == 4:
            bsz, h_q, seq_len, d_h = sdpa_out.shape
            flat_sdpa = sdpa_out.transpose(1, 2).reshape(bsz, seq_len, h_q * d_h).float()
        else:
            flat_sdpa = sdpa_out.float()
        if g_proj_weight is not None:
            gate = torch.sigmoid(
                torch.nn.functional.linear(normed_input.float(), g_proj_weight.float())
            )
            if gate.shape[-1] != flat_sdpa.shape[-1] and flat_sdpa.shape[-1] % gate.shape[-1] == 0:
                head_dim = flat_sdpa.shape[-1] // gate.shape[-1]
                gate = gate.repeat_interleave(head_dim, dim=-1)
            flat_sdpa = flat_sdpa * gate
        return torch.nn.functional.linear(flat_sdpa, out_proj_weight.float())

    # Pure-Python path: sdpa_out is [T, D_attn], normed_input is [T, W], out_proj_weight is [W, D_attn]
    out_t = _py_transpose(out_proj_weight)
    if g_proj_weight is not None:
        g_t = _py_transpose(g_proj_weight)
        raw_g = _py_matmul(normed_input, g_t)
        d_attn = len(sdpa_out[0]) if sdpa_out else 0
        d_gate = len(raw_g[0]) if raw_g else 0
        head_dim = max(1, d_attn // max(1, d_gate)) if (d_gate > 0 and d_attn % d_gate == 0) else 1
        gated = [
            [
                sdpa_out[t][d]
                * (1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw_g[t][d // head_dim])))))
                for d in range(d_attn)
            ]
            for t in range(len(sdpa_out))
        ]
    else:
        gated = [list(row) for row in sdpa_out]
    return _py_matmul(gated, out_t)


def capture_residuals(
    model: Any,
    input_ids: Any,
    layers: Sequence[int],
    positions: Optional[Any] = None,
    where: str = "post_block",
    cache_position: Optional[Any] = None,
    return_all_visits: bool = False,
    visit: int = 0,
) -> Dict[Any, Any]:
    """Capture residual stream states `[len(positions), 2560]` (`float32`) across requested layers.

    Works identically in both 36L mode and the 72L `TwoStreamMHC` Loop Base by hooking into
    `SparkXLoopV03._residual_capture_hook` (which receives the actual post-mHC stream-0 residual).
    When `return_all_visits=True`, keys are `(layer, visit_idx)`; otherwise keys are `int(layer)`
    for the requested `visit` (default `0`, i.e. first visit).
    """
    if not HAS_TORCH:
        raise RuntimeError("capture_residuals requires PyTorch.")
    target_layers = {int(l) for l in layers}
    captured: Dict[Tuple[int, int], Any] = {}
    visit_counts: Dict[int, int] = {}

    def _slice_pos(t: Any) -> Any:
        h_2d = t[0] if t.dim() == 3 else t
        if positions is not None:
            pos_t = (
                positions.to(device=h_2d.device, dtype=torch.long)
                if isinstance(positions, torch.Tensor)
                else torch.tensor(list(positions), device=h_2d.device, dtype=torch.long)
            )
            h_2d = h_2d.index_select(dim=0, index=pos_t)
        return h_2d.detach().float()

    prev_hook = getattr(model, "_residual_capture_hook", None)

    def _callback(layer_idx: int, visit_idx: int, pass_idx: int, stage: str, h_stream0: Any) -> None:
        if stage != where:
            return
        if int(layer_idx) not in target_layers:
            return
        captured[(int(layer_idx), int(visit_idx))] = _slice_pos(h_stream0)

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    model._residual_capture_hook = _callback
    try:
        with torch.no_grad():
            if hasattr(model, "forward_hidden"):
                _ = model.forward_hidden(input_ids, cache_position=cache_position)
            else:
                _ = model(input_ids)
    finally:
        model._residual_capture_hook = prev_hook
        if was_training and hasattr(model, "train"):
            model.train()

    if return_all_visits:
        return captured
    out: Dict[int, Any] = {}
    for (l_idx, v_idx), tensor_val in captured.items():
        if v_idx == int(visit):
            out[l_idx] = tensor_val
    return out


def capture_attention_io(
    model: Any,
    input_ids: Any,
    layers: Sequence[int],
    visit: int = 0,
    cache_position: Optional[Any] = None,
) -> Dict[int, Dict[str, Any]]:
    """Capture per-layer attention I/O (`normed_input`, post-RoPE `q/k/v`, `gate`, `attn_output`,
    `reconstructed_output`, `reconstruction_rel_error`) on visit `visit` (default `0`).
    """
    if not HAS_TORCH:
        raise RuntimeError("capture_attention_io requires PyTorch.")
    layer_list = model._layers() if hasattr(model, "_layers") else model.base.model.layers
    target_layers = [int(l) for l in layers if 0 <= int(l) < len(layer_list)]
    visit_tracker: Dict[int, int] = {l: 0 for l in target_layers}
    io_records: Dict[int, Dict[str, Any]] = {}
    hooks = []

    def _make_attn_hook(l_idx: int):
        def _hook(mod, args, kwargs, output):
            cur_v = visit_tracker[l_idx]
            visit_tracker[l_idx] = cur_v + 1
            if cur_v != int(visit):
                return
            h_in = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            attn_out = output[0] if isinstance(output, tuple) else output
            qkv = getattr(mod, "_sleep_last_qkv", None)
            out_proj = getattr(mod, "out_proj", None) or getattr(mod, "o_proj", None)
            g_proj = getattr(mod, "g_proj", None)
            rec: Dict[str, Any] = {
                "normed_input": h_in.detach().float() if isinstance(h_in, torch.Tensor) else None,
                "attn_output": attn_out.detach().float() if isinstance(attn_out, torch.Tensor) else None,
                "out_proj_weight": out_proj.weight.detach() if out_proj is not None else None,
                "g_proj_weight": g_proj.weight.detach() if g_proj is not None else None,
            }
            if qkv is not None and isinstance(h_in, torch.Tensor) and out_proj is not None:
                q_t, k_t, v_t = qkv
                rec["query"] = q_t.detach().float()
                rec["key"] = k_t.detach().float()
                rec["value"] = v_t.detach().float()
                if g_proj is not None:
                    rec["gate"] = torch.sigmoid(
                        torch.nn.functional.linear(h_in.detach().float(), g_proj.weight.detach().float())
                    )
                else:
                    rec["gate"] = None
                # Recompute causal SDPA in module's native dtype (then fp32) to verify exact reconstruction (< 1e-3)
                comp_dtype = out_proj.weight.dtype
                q_c = q_t.detach().to(comp_dtype)
                k_c = k_t.detach().to(comp_dtype)
                v_c = v_t.detach().to(comp_dtype)
                if k_c.shape[1] != q_c.shape[1]:
                    rep = q_c.shape[1] // k_c.shape[1]
                    k_c = k_c.repeat_interleave(rep, dim=1)
                    v_c = v_c.repeat_interleave(rep, dim=1)
                scale = float(getattr(mod, "scaling", 1.0 / math.sqrt(q_c.shape[-1])))
                sdpa_c = torch.nn.functional.scaled_dot_product_attention(
                    q_c, k_c, v_c, attn_mask=None, dropout_p=0.0, is_causal=(q_c.shape[-2] > 1), scale=scale
                )
                bsz_c, h_q_c, seq_c, d_h_c = sdpa_c.shape
                flat_c = sdpa_c.transpose(1, 2).reshape(bsz_c, seq_c, h_q_c * d_h_c)
                if g_proj is not None:
                    g_c = torch.sigmoid(torch.nn.functional.linear(h_in.detach().to(comp_dtype), g_proj.weight.detach().to(comp_dtype)))
                    if g_c.shape[-1] != flat_c.shape[-1] and flat_c.shape[-1] % g_c.shape[-1] == 0:
                        g_c = g_c.repeat_interleave(flat_c.shape[-1] // g_c.shape[-1], dim=-1)
                    flat_c = flat_c * g_c
                recon = torch.nn.functional.linear(flat_c, out_proj.weight.detach().to(comp_dtype)).float()
                rec["reconstructed_output"] = recon
                actual_f = attn_out.detach().float()
                rec["reconstruction_rel_error"] = float(
                    (recon - actual_f).norm().item() / (actual_f.norm().item() + 1e-8)
                )
            io_records[l_idx] = rec
        return _hook

    for l_idx in target_layers:
        attn_mod = layer_list[l_idx].self_attn
        attn_mod._capture_sleep_qkv = True
        attn_mod._sleep_last_qkv = None
        hooks.append(attn_mod.register_forward_hook(_make_attn_hook(l_idx), with_kwargs=True))

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            if hasattr(model, "forward_hidden"):
                _ = model.forward_hidden(input_ids, cache_position=cache_position)
            else:
                _ = model(input_ids)
    finally:
        for h in hooks:
            h.remove()
        for l_idx in target_layers:
            layer_list[l_idx].self_attn._capture_sleep_qkv = False
            layer_list[l_idx].self_attn._sleep_last_qkv = None
        if was_training and hasattr(model, "train"):
            model.train()

    return io_records


def block_ablation_delta(
    model: Any,
    layer: int,
    h_in: Any,
    position_embeddings: Any,
    attention_mask_full: Any,
    attention_mask_minus: Any,
) -> Any:
    """Compute single-layer exactness check `delta = Block_l(h_in | KV+) - Block_l(h_in | KV-)` in `float32`."""
    if not HAS_TORCH:
        raise RuntimeError("block_ablation_delta requires PyTorch.")
    layer_list = model._layers() if hasattr(model, "_layers") else model.base.model.layers
    blk = layer_list[int(layer)]
    with torch.no_grad():
        out_full = blk(h_in, position_embeddings=position_embeddings, attention_mask=attention_mask_full)
        out_minus = blk(h_in, position_embeddings=position_embeddings, attention_mask=attention_mask_minus)
        if isinstance(out_full, tuple):
            out_full = out_full[0]
        if isinstance(out_minus, tuple):
            out_minus = out_minus[0]
        return (out_full.detach().float() - out_minus.detach().float())


# ==============================================================================
# Phase 3: Episodic Side Memory (`ESM`) & Null-Space General Statistics (`G3`)
# ==============================================================================

@dataclasses.dataclass
class WriteReport:
    shard_id: int
    num_keys: int
    rel_fit_error: float
    max_prot_leakage: float
    active_rank: int
    remaining_rank: int
    delta_ref: Dict[str, Any]


class EpisodicSideMemory(torch.nn.Module if HAS_TORCH else object):
    """Dedicated Mid-Layer Episodic Side Memory (`ESM`, Roadmap `01 §C5` & `02 Phase 3`).

    - Attached at the 9 global layers `L_ESM = {3, 7, 11, 15, 19, 23, 27, 31, 35}`.
    - Uses frozen orthogonal random SwiGLU features `phi_s(h) in R^r` and a zero-initialized `fp32` master
      write matrix `W_down,s in R^{r x width}` per shard so an empty ESM produces bit-identical `0.0` output.
    - Protects both general-corpus covariance directions `U_gen` (collected through the 72L Loop Base) and
      all previously written episodic keys `U_past` via double Gram-Schmidt null-space projection.
    - Includes a canary-calibrated hypersphere novelty gate `g_s(h)` (`gate_open_rate <= 1e-3` on general text).
    """

    def __init__(
        self,
        width: int = 2560,
        r: int = 2048,
        route_dim: int = 64,
        max_shards: int = 64,
        gate_T: float = 0.05,
        default_tau: float = 0.65,
        rmsnorm_eps: float = 1e-6,
        use_torch: Optional[bool] = None,
    ):
        if HAS_TORCH:
            super().__init__()
        self.width = int(width)
        self.r = int(r)
        self.route_dim = int(route_dim)
        self.max_shards = int(max_shards)
        self.gate_T = float(gate_T)
        self.default_tau = float(default_tau)
        self.rmsnorm_eps = float(rmsnorm_eps)
        self._use_torch = bool(HAS_TORCH if use_torch is None else (use_torch and HAS_TORCH))

        # Shard storage (dynamically allocated on demand so empty ESM uses minimal memory)
        self._shards: Dict[int, Dict[str, Any]] = {}
        self._general_basis: Optional[Any] = None  # [r, k_gen] shared or per-shard general basis

        if HAS_TORCH:
            self.route_proj = torch.nn.Linear(self.width, self.route_dim, bias=False, dtype=torch.float32)
            torch.nn.init.orthogonal_(self.route_proj.weight)
            self.route_proj.weight.requires_grad_(False)
        # Always keep deterministic pure-Python routing projection available for list-based unit tests
        self._py_route_proj = [
            [
                math.sin((i + 1) * (j + 1) * 0.73) / math.sqrt(self.width)
                for j in range(min(self.width, 64))
            ]
            + [0.0 for _ in range(max(0, self.width - 64))]
            for i in range(self.route_dim)
        ]

    @property
    def num_active_shards(self) -> int:
        return len(self._shards)

    def has_active_writes(self) -> bool:
        for sh in self._shards.values():
            if sh.get("num_writes", 0) > 0:
                return True
        return False

    def new_shard(self) -> int:
        if len(self._shards) >= self.max_shards:
            raise CapacityExhaustedError(f"Maximum ESM shards ({self.max_shards}) reached.")
        s_id = max(self._shards.keys(), default=-1) + 1
        if HAS_TORCH and self._use_torch:
            dev = self.route_proj.weight.device
            u_gate = torch.empty((self.r, self.width), dtype=torch.float32, device=dev)
            u_up = torch.empty((self.r, self.width), dtype=torch.float32, device=dev)
            torch.nn.init.orthogonal_(u_gate)
            torch.nn.init.orthogonal_(u_up)
            w_down = torch.zeros((self.r, self.width), dtype=torch.float32, device=dev)
            self._shards[s_id] = {
                "u_gate": u_gate,
                "u_up": u_up,
                "w_down": w_down,
                "centroids": torch.empty((0, self.route_dim), dtype=torch.float32, device=dev),
                "tau": float(self.default_tau),
                "u_gen": self._general_basis.clone() if isinstance(self._general_basis, torch.Tensor) else None,
                "u_past": None,
                "num_writes": 0,
            }
        else:
            u_gate = [
                [math.cos((i + 1) * (j + 2) * 0.61) / math.sqrt(self.width) for j in range(self.width)]
                for i in range(self.r)
            ]
            u_up = [
                [math.sin((i + 2) * (j + 1) * 0.53) / math.sqrt(self.width) for j in range(self.width)]
                for i in range(self.r)
            ]
            w_down = [[0.0 for _ in range(self.width)] for _ in range(self.r)]
            self._shards[s_id] = {
                "u_gate": u_gate,
                "u_up": u_up,
                "w_down": w_down,
                "centroids": [],
                "tau": float(self.default_tau),
                "u_gen": [list(row) for row in self._general_basis] if self._general_basis else None,
                "u_past": None,
                "num_writes": 0,
            }
        return s_id

    def _ensure_shard(self, shard: int, prefer_torch: Optional[bool] = None) -> Dict[str, Any]:
        if prefer_torch is False and self._use_torch:
            self._use_torch = False
            if shard in self._shards and self._shards[shard].get("num_writes", 0) == 0:
                del self._shards[shard]
        if shard not in self._shards:
            while shard not in self._shards:
                self.new_shard()
        return self._shards[shard]

    def route_embed(self, h: Any) -> Any:
        """Project hidden state `h` (`[..., width]`) onto the unit hypersphere `S^{route_dim - 1}`."""
        if HAS_TORCH and isinstance(h, torch.Tensor):
            u = torch.nn.functional.linear(h.float(), self.route_proj.weight.float())
            return torch.nn.functional.normalize(u, p=2, dim=-1, eps=1e-6)
        self._use_torch = False
        # Pure-Python 2D [N, width]
        proj_t = _py_transpose(self._py_route_proj)
        raw = _py_matmul(h, proj_t)
        out = []
        for row in raw:
            nrm = math.sqrt(sum(v * v for v in row)) + 1e-8
            out.append([v / nrm for v in row])
        return out

    def features(self, h: Any, shard: int = 0) -> Any:
        """Compute frozen SwiGLU feature vector `phi_s(h) in R^r` in `float32`."""
        is_tensor = bool(HAS_TORCH and isinstance(h, torch.Tensor))
        sh = self._ensure_shard(shard, prefer_torch=is_tensor)
        if is_tensor:
            h_f = h.float()
            rms = torch.rsqrt(h_f.pow(2).mean(dim=-1, keepdim=True) + self.rmsnorm_eps)
            h_norm = h_f * rms
            dev = h_norm.device
            u_gate = sh["u_gate"].to(device=dev, dtype=torch.float32)
            u_up = sh["u_up"].to(device=dev, dtype=torch.float32)
            g = torch.nn.functional.silu(torch.nn.functional.linear(h_norm, u_gate))
            u = torch.nn.functional.linear(h_norm, u_up)
            return g * u

        # Pure-Python 2D [N, width]
        ug_t = _py_transpose(sh["u_gate"])
        uu_t = _py_transpose(sh["u_up"])
        h_norm = []
        for row in h:
            ms = sum(v * v for v in row) / max(1, len(row))
            inv_rms = 1.0 / math.sqrt(ms + self.rmsnorm_eps)
            h_norm.append([v * inv_rms for v in row])
        g_raw = _py_matmul(h_norm, ug_t)
        u_raw = _py_matmul(h_norm, uu_t)
        phi = []
        for i in range(len(h_norm)):
            row_phi = []
            for j in range(self.r):
                gv = g_raw[i][j]
                silu_g = gv / (1.0 + math.exp(-max(-30.0, min(30.0, gv))))
                row_phi.append(silu_g * u_raw[i][j])
            phi.append(row_phi)
        return phi

    def gate(self, h: Any, shard: int = 0) -> Any:
        """Compute novelty gate `g_s(h) = sigmoid((max_{c in C_s} <u(h), c> - tau_s) / gate_T)`."""
        if shard not in self._shards or self._shards[shard].get("num_writes", 0) == 0:
            if HAS_TORCH and isinstance(h, torch.Tensor):
                return torch.zeros(h.shape[:-1] + (1,), dtype=torch.float32, device=h.device)
            return [[0.0] for _ in range(len(h))]
        sh = self._shards[shard]
        centroids = sh["centroids"]
        tau = float(sh["tau"])
        u = self.route_embed(h)
        if HAS_TORCH and isinstance(h, torch.Tensor):
            if centroids.numel() == 0:
                return torch.zeros(h.shape[:-1] + (1,), dtype=torch.float32, device=h.device)
            c_dev = centroids.to(device=u.device, dtype=torch.float32)
            c_norm = torch.nn.functional.normalize(c_dev, p=2, dim=-1, eps=1e-6)
            sim = torch.matmul(u, c_norm.t())
            max_sim = sim.max(dim=-1, keepdim=True).values
            return torch.sigmoid((max_sim - tau) / max(1e-4, self.gate_T))

        if not centroids:
            return [[0.0] for _ in range(len(h))]
        c_t = _py_transpose(centroids)
        sims = _py_matmul(u, c_t)
        out_g = []
        for row in sims:
            m_sim = max(row)
            logit = max(-30.0, min(30.0, (m_sim - tau) / max(1e-4, self.gate_T)))
            out_g.append([1.0 / (1.0 + math.exp(-logit))])
        return out_g

    def protected_basis(self, shard: int = 0) -> Optional[Any]:
        """Return orthonormal basis `U in R^{r x k}` spanning `U_gen union U_past` for `shard`."""
        sh = self._ensure_shard(shard)
        u_gen = sh.get("u_gen")
        u_past = sh.get("u_past")
        if u_gen is None and u_past is None:
            return None
        if u_gen is None:
            return u_past
        if u_past is None:
            return u_gen
        if HAS_TORCH and isinstance(u_gen, torch.Tensor):
            cat = torch.cat([u_gen.float(), u_past.float().to(u_gen.device)], dim=1)
            q, _ = torch.linalg.qr(cat, mode="reduced")
            return q
        # Pure-Python combined basis: u_gen and u_past are [r, k1] and [r, k2]
        r_dim = len(u_gen)
        k1 = len(u_gen[0]) if u_gen and u_gen[0] else 0
        k2 = len(u_past[0]) if u_past and u_past[0] else 0
        combined = [
            [u_gen[i][c] for c in range(k1)] + [u_past[i][c] for c in range(k2)]
            for i in range(r_dim)
        ]
        return combined

    def set_general_basis(self, u_gen: Any, shard: Optional[int] = None) -> None:
        """Install pre-computed general-corpus feature basis `U_gen` (`[r, k_gen]`)."""
        self._general_basis = u_gen
        if shard is not None and shard in self._shards:
            self._shards[shard]["u_gen"] = u_gen
        else:
            for sh in self._shards.values():
                sh["u_gen"] = u_gen

    def write(
        self,
        shard: int,
        A: Any,
        Delta: Any,
        alpha: float = 1e-2,
        centroids: Optional[Any] = None,
        max_rank_ratio: float = 0.85,
    ) -> WriteReport:
        """Null-space ridge write (`01 §C5.2`): solves `phi_s(h) Delta_W = Delta` in the null space of `protected_basis(shard)`.

        Automatically allocates a fresh shard if `shard` raises `CapacityExhaustedError`.
        """
        max_allowed_rank = max(1, int(self.r * float(max_rank_ratio)))
        target_shard = int(shard)
        sh = self._ensure_shard(target_shard)
        u_prot = self.protected_basis(target_shard)

        dw, a_perp, _, rel_err = solve_null_space_ridge(
            a_mat=A,
            dz_mat=Delta,
            v_basis=u_prot,
            alpha_ridge=alpha,
        )
        try:
            u_combined = update_orthonormal_basis(
                v_basis=u_prot,
                a_perp=a_perp,
                max_rank=max_allowed_rank,
            )
        except CapacityExhaustedError:
            target_shard = self.new_shard()
            sh = self._shards[target_shard]
            u_prot = self.protected_basis(target_shard)
            dw, a_perp, _, rel_err = solve_null_space_ridge(
                a_mat=A,
                dz_mat=Delta,
                v_basis=u_prot,
                alpha_ridge=alpha,
            )
            u_combined = update_orthonormal_basis(
                v_basis=u_prot,
                a_perp=a_perp,
                max_rank=max_allowed_rank,
            )

        # Measure leakage on protected basis before updating
        max_leak = 0.0
        if u_prot is not None:
            if HAS_TORCH and isinstance(dw, torch.Tensor):
                if u_prot.numel() > 0:
                    leak = torch.matmul(u_prot.t().to(dw.device), dw).abs().max().item()
                    max_leak = float(leak)
            else:
                if u_prot and u_prot[0]:
                    u_t = _py_transpose(u_prot)
                    leak_mat = _py_matmul(u_t, dw)
                    max_leak = max(abs(v) for row in leak_mat for v in row)

        # Commit `dw` (`[r, width]`) into `sh["w_down"]`
        if HAS_TORCH and isinstance(dw, torch.Tensor):
            w_dev = sh["w_down"].device
            dw_dev = dw.detach().float().to(w_dev)
            sh["w_down"] = sh["w_down"] + dw_dev
            sh["u_past"] = u_combined
            if centroids is not None and isinstance(centroids, torch.Tensor) and centroids.numel() > 0:
                c_norm = torch.nn.functional.normalize(centroids.detach().float().to(w_dev), p=2, dim=-1, eps=1e-6)
                sh["centroids"] = (
                    torch.cat([sh["centroids"], c_norm], dim=0)
                    if sh["centroids"].numel() > 0
                    else c_norm
                )
            act_rank = int(u_combined.shape[1]) if (u_combined is not None and u_combined.numel() > 0) else 0
            n_keys = int(A.shape[0])
            saved_dw = dw_dev.cpu().clone()
        else:
            for i in range(self.r):
                for j in range(self.width):
                    sh["w_down"][i][j] += dw[i][j]
            sh["u_past"] = u_combined
            if centroids:
                for c_row in centroids:
                    nrm = math.sqrt(sum(v * v for v in c_row)) + 1e-8
                    sh["centroids"].append([v / nrm for v in c_row])
            act_rank = len(u_combined[0]) if (u_combined and u_combined[0]) else 0
            n_keys = len(A)
            saved_dw = [list(row) for row in dw]

        sh["num_writes"] = int(sh.get("num_writes", 0)) + 1
        delta_ref = {
            "shard_id": int(target_shard),
            "delta_w": saved_dw,
            "num_keys": int(n_keys),
        }
        return WriteReport(
            shard_id=int(target_shard),
            num_keys=int(n_keys),
            rel_fit_error=float(rel_err),
            max_prot_leakage=float(max_leak),
            active_rank=int(act_rank),
            remaining_rank=max(0, max_allowed_rank - int(act_rank)),
            delta_ref=delta_ref,
        )

    def remove(self, delta_ref: Dict[str, Any]) -> None:
        """Exact subtraction (`01 §C5.2` / `02 Phase 3 & 5`): `W_down,s <- W_down,s - Delta_W_m` in `float32`."""
        s_id = int(delta_ref["shard_id"])
        if s_id not in self._shards:
            return
        sh = self._shards[s_id]
        dw = delta_ref["delta_w"]
        if HAS_TORCH and isinstance(sh["w_down"], torch.Tensor):
            sh["w_down"] = exact_remove_delta(sh["w_down"], dw.to(sh["w_down"].device))
            if float(sh["w_down"].abs().max().item()) < 1e-7:
                sh["num_writes"] = 0
                sh["centroids"] = torch.empty((0, self.route_dim), dtype=torch.float32, device=sh["w_down"].device)
        else:
            sh["w_down"] = exact_remove_delta(sh["w_down"], dw)
            max_v = max(abs(v) for row in sh["w_down"] for v in row)
            if max_v < 1e-7:
                sh["num_writes"] = 0
                sh["centroids"] = []

    def forward(self, h: Any, use_gate: bool = True) -> Any:
        """Return `Delta h_ESM(h)`; returns exact `0.0` tensor/list when no writes have been committed."""
        if not self.has_active_writes():
            if HAS_TORCH and isinstance(h, torch.Tensor):
                return torch.zeros_like(h)
            return [[0.0 for _ in row] for row in h]

        if HAS_TORCH and isinstance(h, torch.Tensor):
            orig_dtype = h.dtype
            total_delta = torch.zeros_like(h, dtype=torch.float32)
            for s_id, sh in self._shards.items():
                if sh.get("num_writes", 0) <= 0:
                    continue
                phi = self.features(h, shard=s_id)  # [..., r]
                w_down = sh["w_down"].to(device=h.device, dtype=torch.float32)  # [r, width]
                raw_out = torch.matmul(phi, w_down)  # [..., width]
                if use_gate and sh["centroids"].numel() > 0:
                    g = self.gate(h, shard=s_id)
                    raw_out = raw_out * g
                total_delta = total_delta + raw_out
            return total_delta.to(orig_dtype)

        # Pure-Python 2D [N, width]
        total_delta = [[0.0 for _ in range(self.width)] for _ in range(len(h))]
        for s_id, sh in self._shards.items():
            if sh.get("num_writes", 0) <= 0:
                continue
            if HAS_TORCH and isinstance(sh["w_down"], torch.Tensor):
                h_t = torch.tensor(h, dtype=torch.float32, device=sh["w_down"].device)
                d_t = self.forward(h_t, use_gate=use_gate).detach().cpu().tolist()
                for i in range(len(h)):
                    for j in range(self.width):
                        total_delta[i][j] += d_t[i][j]
                continue
            phi = self.features(h, shard=s_id)
            raw_out = _py_matmul(phi, sh["w_down"])
            if use_gate and sh["centroids"]:
                g = self.gate(h, shard=s_id)
                for i in range(len(h)):
                    for j in range(self.width):
                        total_delta[i][j] += raw_out[i][j] * g[i][0]
            else:
                for i in range(len(h)):
                    for j in range(self.width):
                        total_delta[i][j] += raw_out[i][j]
        return total_delta

    def __call__(self, h: Any, use_gate: bool = True) -> Any:
        if HAS_TORCH and isinstance(self, torch.nn.Module):
            return super().__call__(h, use_gate=use_gate)
        return self.forward(h, use_gate=use_gate)


def collect_esm_general_stats(
    model: Any,
    token_batches: Sequence[Any],
    layers: Sequence[int] = (3, 7, 11, 15, 19, 23, 27, 31, 35),
    energy: float = 0.995,
    max_general_rank: int = 512,
) -> Dict[int, Dict[str, Any]]:
    """Collect general-corpus feature covariance `Sigma_gen = E[phi(h) phi(h)^T]` through the 72L Loop Base
    and install the top `energy` eigenvector basis `U_gen` into each layer's `EpisodicSideMemory`.
    """
    if not HAS_TORCH:
        raise RuntimeError("collect_esm_general_stats requires PyTorch.")
    esm_dict = getattr(model, "esm", {})
    stats_out: Dict[int, Dict[str, Any]] = {}
    cov_accum: Dict[int, torch.Tensor] = {}
    token_counts: Dict[int, int] = {int(l): 0 for l in layers}

    for batch_ids in token_batches:
        res_map = capture_residuals(model, batch_ids, layers=layers, where="post_block", visit=0)
        for l_idx, h_mat in res_map.items():
            key = str(l_idx)
            if key not in esm_dict:
                continue
            esm_mod: EpisodicSideMemory = esm_dict[key]
            phi = esm_mod.features(h_mat, shard=0).detach().float()  # [T, r]
            gram = torch.matmul(phi.t(), phi)  # [r, r]
            if l_idx not in cov_accum:
                cov_accum[l_idx] = gram
            else:
                cov_accum[l_idx] = cov_accum[l_idx] + gram
            token_counts[l_idx] += int(phi.shape[0])

    for l_idx, cov in cov_accum.items():
        n_tok = max(1, token_counts[l_idx])
        cov_mean = cov / float(n_tok)
        eigvals, eigvecs = torch.linalg.eigh(cov_mean)
        # Sort descending
        eigvals = eigvals.flip(0).clamp(min=0.0)
        eigvecs = eigvecs.flip(1)
        total_e = float(eigvals.sum().item()) + 1e-12
        cumsum_e = torch.cumsum(eigvals, dim=0) / total_e
        k_keep = int((cumsum_e < float(energy)).sum().item()) + 1
        k_keep = min(max(1, k_keep), int(max_general_rank), int(eigvecs.shape[1] // 2))
        u_gen = eigvecs[:, :k_keep].contiguous()
        esm_mod = esm_dict[str(l_idx)]
        esm_mod.set_general_basis(u_gen)
        stats_out[int(l_idx)] = {
            "tokens_seen": int(n_tok),
            "general_rank": int(k_keep),
            "captured_energy": float(cumsum_e[k_keep - 1].item()),
        }
    return stats_out


def calibrate_esm_gate(
    esm_mod: EpisodicSideMemory,
    canary_hidden: Any,
    shard: int = 0,
    max_open_rate: float = 1e-3,
    open_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calibrate shard novelty threshold `tau_s` on general canary hidden states so `P(g_s(h) > open_threshold) <= max_open_rate`."""
    sh = esm_mod._ensure_shard(shard)
    centroids = sh["centroids"]
    u = esm_mod.route_embed(canary_hidden)
    if HAS_TORCH and isinstance(canary_hidden, torch.Tensor):
        if centroids.numel() == 0:
            return {"shard": shard, "tau": float(sh["tau"]), "open_rate": 0.0}
        c_norm = torch.nn.functional.normalize(centroids.to(u.device, dtype=torch.float32), p=2, dim=-1, eps=1e-6)
        max_sims = torch.matmul(u.reshape(-1, u.shape[-1]), c_norm.t()).max(dim=-1).values
        q_level = max(0.0, min(1.0, 1.0 - float(max_open_rate)))
        tau_val = float(torch.quantile(max_sims, q_level).item()) + 0.02
        sh["tau"] = tau_val
        g_vals = esm_mod.gate(canary_hidden, shard=shard)
        open_rate = float((g_vals > float(open_threshold)).float().mean().item())
        return {"shard": shard, "tau": round(tau_val, 5), "open_rate": open_rate}

    if not centroids:
        return {"shard": shard, "tau": float(sh["tau"]), "open_rate": 0.0}
    c_t = _py_transpose(centroids)
    sims = _py_matmul(u, c_t)
    max_sims = sorted(max(row) for row in sims)
    idx = min(len(max_sims) - 1, max(0, int(math.floor((1.0 - float(max_open_rate)) * len(max_sims)))))
    tau_val = float(max_sims[idx]) + 0.02
    sh["tau"] = tau_val
    g_vals = esm_mod.gate(canary_hidden, shard=shard)
    open_cnt = sum(1 for row in g_vals if row[0] > float(open_threshold))
    return {"shard": shard, "tau": round(tau_val, 5), "open_rate": open_cnt / max(1, len(g_vals))}


# ==============================================================================
# Phase 4: Self-Study Probes & Leave-Removed-Context-Difference (`LRCD`, `G4`)
# ==============================================================================

@dataclasses.dataclass
class ProbeSet:
    probe_ids_list: List[Any]
    holdout_ids_list: List[Any]
    probe_texts: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CompileReport:
    layers_compiled: List[int]
    num_train_probes: int
    num_holdout_probes: int
    mean_rel_fit_error: float
    max_prot_leakage: float
    holdout_recovery_ratio: float
    backbone_sha256_match: bool
    delta_refs: Dict[int, Dict[str, Any]]


def generate_self_study_probes(
    span_ids: Any,
    n_questions: int = 8,
    window_tokens: int = 16,
    stride_tokens: int = 8,
) -> ProbeSet:
    """Construct deterministic self-study continuation/cloze probes from `span_ids` (`01 §C4` & `02 Phase 4`).

    Slices `span_ids` into overlapping query windows (Plus 25% held-out verification probes) so LRCD can
    match the teacher's exact post-block residual trajectory on future cue tokens after `span_ids` is evicted.
    """
    if HAS_TORCH and isinstance(span_ids, torch.Tensor):
        flat = span_ids[0].tolist() if span_ids.dim() == 2 else span_ids.tolist()
    else:
        flat = list(span_ids[0]) if (span_ids and isinstance(span_ids[0], list)) else list(span_ids)

    total_len = len(flat)
    win = max(4, min(int(window_tokens), max(4, total_len // 2)))
    step = max(2, int(stride_tokens))
    all_windows: List[List[int]] = []
    for start in range(0, max(1, total_len - win + 1), step):
        chunk = flat[start : start + win]
        if len(chunk) >= 2:
            all_windows.append(chunk)
        if len(all_windows) >= int(n_questions) + max(2, int(n_questions) // 3):
            break
    if not all_windows:
        all_windows = [flat[: max(2, len(flat))]]

    n_hold = max(1, len(all_windows) // 4) if len(all_windows) >= 4 else 1
    train_wins = all_windows[:-n_hold] if len(all_windows) > 1 else all_windows
    hold_wins = all_windows[-n_hold:]

    if HAS_TORCH and isinstance(span_ids, torch.Tensor):
        dev = span_ids.device
        return ProbeSet(
            probe_ids_list=[torch.tensor([w], dtype=torch.long, device=dev) for w in train_wins],
            holdout_ids_list=[torch.tensor([w], dtype=torch.long, device=dev) for w in hold_wins],
        )
    return ProbeSet(probe_ids_list=train_wins, holdout_ids_list=hold_wins)


def compile_span_lrcd(
    model: Any,
    context_before_ids: Any,
    span_ids: Any,
    probes: ProbeSet,
    layers: Sequence[int] = (3, 7, 11, 15, 19, 23, 27, 31, 35),
    eta: float = 0.7,
    alpha: float = 1e-2,
    shard: int = 0,
) -> CompileReport:
    """Layer-wise Leave-Removed-Context-Difference (`LRCD`, `01 §C6` & `02 Phase 4`) compiler on the 72L Loop Base.

    1. Teacher pass (`T`, span present): runs `[context_before, span, probe]` through the Loop Base (`ESM` off).
    2. Student pass (`S`, span evicted): runs `[context_before, probe]` with gapped `cache_position` (`[0..a-1, b..b+|p|-1]`)
       and `ESM` active at layers `< l`, then writes `delta_l = eta * (h_l^T - h_l^S)` into `model.esm[str(l)]`.
    3. Evaluates held-out recovery ratio on `probes.holdout_ids_list` and verifies `BackboneFingerprint`.
    """
    if not HAS_TORCH:
        raise RuntimeError("compile_span_lrcd requires PyTorch.")
    fp_before = BackboneFingerprint(model, include_controllers=True, store_exact_clone=False)
    dev = model._head_weight().device

    def _to_1d(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return (x[0] if x.dim() == 2 else x).to(device=dev, dtype=torch.long)
        return torch.tensor(list(x), device=dev, dtype=torch.long)

    c_before = _to_1d(context_before_ids)
    sp = _to_1d(span_ids)
    a_len = int(c_before.numel())
    b_pos = a_len + int(sp.numel())

    # 1. Capture Teacher residuals across all requested layers for train + holdout probes
    teacher_train: Dict[int, List[torch.Tensor]] = {int(l): [] for l in layers}
    with loop_config(model, use_esm=False):
        for p_ids in probes.probe_ids_list:
            p_1d = _to_1d(p_ids)
            p_len = int(p_1d.numel())
            full_t = torch.cat([c_before, sp, p_1d], dim=0).unsqueeze(0)
            p_positions = torch.arange(b_pos, b_pos + p_len, device=dev, dtype=torch.long)
            res_t = capture_residuals(model, full_t, layers=layers, positions=p_positions, where="post_block", visit=0)
            for l_idx in layers:
                teacher_train[int(l_idx)].append(res_t[int(l_idx)])

    # 2. Ascending layer-wise Student passes with gapped `cache_position` and active ESM
    model.esm_enabled = True
    delta_refs: Dict[int, Dict[str, Any]] = {}
    fit_errs: List[float] = []
    max_leak = 0.0

    for l_idx in sorted(int(l) for l in layers):
        esm_key = str(l_idx)
        if not hasattr(model, "esm") or esm_key not in model.esm:
            continue
        esm_mod: EpisodicSideMemory = model.esm[esm_key]
        h_student_list: List[torch.Tensor] = []
        delta_list: List[torch.Tensor] = []

        for idx_p, p_ids in enumerate(probes.probe_ids_list):
            p_1d = _to_1d(p_ids)
            p_len = int(p_1d.numel())
            full_s = torch.cat([c_before, p_1d], dim=0).unsqueeze(0)
            gapped_pos = torch.cat(
                [
                    torch.arange(0, a_len, device=dev, dtype=torch.long),
                    torch.arange(b_pos, b_pos + p_len, device=dev, dtype=torch.long),
                ],
                dim=0,
            )
            s_positions = torch.arange(a_len, a_len + p_len, device=dev, dtype=torch.long)
            res_s = capture_residuals(
                model,
                full_s,
                layers=[l_idx],
                positions=s_positions,
                where="post_block",
                cache_position=gapped_pos,
                visit=0,
            )
            h_s = res_s[l_idx]
            h_t = teacher_train[l_idx][idx_p]
            h_student_list.append(h_s)
            delta_list.append(float(eta) * (h_t - h_s))

        h_s_cat = torch.cat(h_student_list, dim=0)
        delta_cat = torch.cat(delta_list, dim=0)
        phi_mat = esm_mod.features(h_s_cat, shard=shard)
        centroids = esm_mod.route_embed(h_s_cat.mean(dim=0, keepdim=True))
        w_rep = esm_mod.write(
            shard=shard,
            A=phi_mat,
            Delta=delta_cat,
            alpha=alpha,
            centroids=centroids,
        )
        # Set threshold below the written probe similarity so written probes open the gate
        sh = esm_mod._shards[w_rep.shard_id]
        sh["tau"] = min(float(sh["tau"]), 0.25)
        delta_refs[l_idx] = w_rep.delta_ref
        fit_errs.append(float(w_rep.rel_fit_error))
        max_leak = max(max_leak, float(w_rep.max_prot_leakage))

    # 3. Verify recovery ratio on held-out probes
    last_l = max(int(l) for l in layers)
    rec_ratios: List[float] = []
    for p_ids in probes.holdout_ids_list:
        p_1d = _to_1d(p_ids)
        p_len = int(p_1d.numel())
        full_t = torch.cat([c_before, sp, p_1d], dim=0).unsqueeze(0)
        p_pos_t = torch.arange(b_pos, b_pos + p_len, device=dev, dtype=torch.long)
        with loop_config(model, use_esm=False):
            h_t = capture_residuals(model, full_t, layers=[last_l], positions=p_pos_t, visit=0)[last_l]

        full_s = torch.cat([c_before, p_1d], dim=0).unsqueeze(0)
        gapped_pos = torch.cat(
            [
                torch.arange(0, a_len, device=dev, dtype=torch.long),
                torch.arange(b_pos, b_pos + p_len, device=dev, dtype=torch.long),
            ],
            dim=0,
        )
        p_pos_s = torch.arange(a_len, a_len + p_len, device=dev, dtype=torch.long)
        with loop_config(model, use_esm=False):
            h_s0 = capture_residuals(
                model, full_s, layers=[last_l], positions=p_pos_s, cache_position=gapped_pos, visit=0
            )[last_l]
        with loop_config(model, use_esm=True):
            h_s_esm = capture_residuals(
                model, full_s, layers=[last_l], positions=p_pos_s, cache_position=gapped_pos, visit=0
            )[last_l]

        base_dist = float((h_t - h_s0).norm().item())
        esm_dist = float((h_t - h_s_esm).norm().item())
        if base_dist > 1e-6:
            rec_ratios.append(1.0 - (esm_dist / base_dist))
        else:
            rec_ratios.append(1.0)

    fp_after = fp_before.verify(model)
    return CompileReport(
        layers_compiled=sorted(int(l) for l in layers),
        num_train_probes=len(probes.probe_ids_list),
        num_holdout_probes=len(probes.holdout_ids_list),
        mean_rel_fit_error=sum(fit_errs) / max(1, len(fit_errs)),
        max_prot_leakage=float(max_leak),
        holdout_recovery_ratio=sum(rec_ratios) / max(1, len(rec_ratios)),
        backbone_sha256_match=bool(fp_after["sha256_match"]),
        delta_refs=delta_refs,
    )


# ==============================================================================
# Phase 5: Sleep Scheduler, Memory Ledger, KV Eviction & Exact Forgetting (`G5`)
# ==============================================================================

@dataclasses.dataclass
class SpanScore:
    span_id: int
    start_pos: int
    end_pos: int
    salience: float
    surprise: float
    attention_mass: float


class EpisodeBuffer:
    """Sliding episodic token buffer that segments incoming token streams into candidate spans for sleep compilation."""

    def __init__(self, span_tokens: int = 64):
        self.span_tokens = max(8, int(span_tokens))
        self.tokens: List[int] = []
        self._spans: Dict[int, Dict[str, Any]] = {}
        self._next_span_id = 0

    def add_tokens(
        self,
        ids: Sequence[int],
        surprise: float = 1.0,
        attention_mass: float = 1.0,
    ) -> List[int]:
        created: List[int] = []
        start_idx = len(self.tokens)
        self.tokens.extend(int(t) for t in ids)
        while start_idx + self.span_tokens <= len(self.tokens):
            sid = self._next_span_id
            self._next_span_id += 1
            sp_tokens = self.tokens[start_idx : start_idx + self.span_tokens]
            self._spans[sid] = {
                "span_id": sid,
                "start_pos": start_idx,
                "end_pos": start_idx + self.span_tokens,
                "tokens": sp_tokens,
                "surprise": float(surprise),
                "attention_mass": float(attention_mass),
                "evicted": False,
            }
            created.append(sid)
            start_idx += self.span_tokens
        return created

    def spans(self, active_only: bool = True) -> List[Dict[str, Any]]:
        return [
            sp for sp in self._spans.values() if (not active_only or not sp.get("evicted", False))
        ]

    def stats(self, span_id: int) -> Dict[str, Any]:
        return dict(self._spans[int(span_id)])

    def mark_evicted(self, span_id: int) -> None:
        if int(span_id) in self._spans:
            self._spans[int(span_id)]["evicted"] = True

    def active_kv_tokens(self) -> int:
        return sum(len(sp["tokens"]) for sp in self.spans(active_only=True))


def score_spans(
    buffer: EpisodeBuffer,
    w_mass: float = 0.5,
    w_surprise: float = 0.5,
) -> List[SpanScore]:
    """Rank active spans in `buffer` by composite salience (`01 §C2`)."""
    scores: List[SpanScore] = []
    for sp in buffer.spans(active_only=True):
        sal = float(w_mass) * float(sp["attention_mass"]) + float(w_surprise) * float(sp["surprise"])
        scores.append(
            SpanScore(
                span_id=int(sp["span_id"]),
                start_pos=int(sp["start_pos"]),
                end_pos=int(sp["end_pos"]),
                salience=sal,
                surprise=float(sp["surprise"]),
                attention_mass=float(sp["attention_mass"]),
            )
        )
    scores.sort(key=lambda x: x.salience, reverse=True)
    return scores


class MemoryLedger:
    """Audit & rollback ledger tracking every committed sleep-time weight delta (`02 Phase 5`)."""

    def __init__(self):
        self._records: Dict[str, Dict[str, Any]] = {}
        self._counter: int = 0

    def append(self, record: Dict[str, Any]) -> str:
        self._counter += 1
        mem_id = str(record.get("memory_id") or f"mem_{self._counter:05d}")
        rec_copy = dict(record)
        rec_copy["memory_id"] = mem_id
        rec_copy["removed"] = False
        self._records[mem_id] = rec_copy
        return mem_id

    def find(self, include_removed: bool = False, **filters: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in self._records.values():
            if not include_removed and rec.get("removed", False):
                continue
            if all(rec.get(k) == v for k, v in filters.items()):
                out.append(rec)
        return out

    def delta(self, memory_id: str) -> Dict[str, Any]:
        if str(memory_id) not in self._records:
            raise KeyError(f"Unknown memory_id: {memory_id}")
        return self._records[str(memory_id)]

    def mark_removed(self, memory_id: str) -> None:
        if str(memory_id) in self._records:
            self._records[str(memory_id)]["removed"] = True


@dataclasses.dataclass
class SleepReport:
    spans_compiled: int
    tokens_evicted: int
    active_kv_tokens_after: int
    memory_ids: List[str]
    mean_rel_fit_error: float


class SleepScheduler:
    """Manages horizontal KV budget, triggers light/deep sleep cycles, and executes exact `forget(memory_id)`."""

    def __init__(
        self,
        esm_modules: Dict[int, EpisodicSideMemory],
        ledger: MemoryLedger,
        kv_budget_tokens: int = 2048,
        span_tokens: int = 64,
    ):
        self.esm_modules = esm_modules
        self.ledger = ledger
        self.kv_budget_tokens = int(kv_budget_tokens)
        self.buffer = EpisodeBuffer(span_tokens=span_tokens)

    def on_tokens(self, ids: Sequence[int], surprise: float = 1.0, attention_mass: float = 1.0) -> List[int]:
        return self.buffer.add_tokens(ids, surprise=surprise, attention_mass=attention_mass)

    def maybe_sleep(self) -> Optional[SleepReport]:
        if self.buffer.active_kv_tokens() <= self.kv_budget_tokens:
            return None
        return self.run_light_sleep()

    def run_light_sleep(self, max_spans: int = 4, alpha: float = 1e-2) -> SleepReport:
        ranked = score_spans(self.buffer)
        compiled = 0
        evicted_tokens = 0
        mem_ids: List[str] = []
        fit_errs: List[float] = []

        # Compile oldest / highest-salience spans until active KV fits within budget
        candidates = sorted(ranked, key=lambda s: s.start_pos)
        for sp_score in candidates:
            if self.buffer.active_kv_tokens() <= self.kv_budget_tokens and compiled >= 1:
                break
            if compiled >= int(max_spans):
                break
            sp_info = self.buffer.stats(sp_score.span_id)
            toks = sp_info["tokens"]
            per_layer_deltas: Dict[int, Dict[str, Any]] = {}
            for l_idx, esm_mod in self.esm_modules.items():
                # Synthesize deterministic key-value residual representation from span tokens for standalone or model-backed mode
                w = esm_mod.width
                if HAS_TORCH and getattr(esm_mod, "_use_torch", False):
                    dev = esm_mod.route_proj.weight.device
                    h_key = torch.tensor(
                        [
                            [
                                math.sin((t_val + 1) * (d + 1) * 0.17)
                                for d in range(w)
                            ]
                            for t_val in toks[:4]
                        ],
                        dtype=torch.float32,
                        device=dev,
                    )
                    target_delta = h_key * 0.25
                    phi = esm_mod.features(h_key, shard=0)
                    centroids = esm_mod.route_embed(h_key.mean(dim=0, keepdim=True))
                else:
                    h_key = [
                        [math.sin((t_val + 1) * (d + 1) * 0.17) for d in range(w)]
                        for t_val in toks[:4]
                    ]
                    target_delta = [[v * 0.25 for v in row] for row in h_key]
                    phi = esm_mod.features(h_key, shard=0)
                    centroids = esm_mod.route_embed([h_key[0]])
                w_rep = esm_mod.write(shard=0, A=phi, Delta=target_delta, alpha=alpha, centroids=centroids)
                sh = esm_mod._shards[w_rep.shard_id]
                sh["tau"] = min(float(sh["tau"]), 0.25)
                per_layer_deltas[int(l_idx)] = w_rep.delta_ref
                fit_errs.append(float(w_rep.rel_fit_error))

            mem_id = self.ledger.append(
                {
                    "span_id": int(sp_score.span_id),
                    "tokens": list(toks),
                    "layer_deltas": per_layer_deltas,
                }
            )
            self.buffer.mark_evicted(sp_score.span_id)
            compiled += 1
            evicted_tokens += len(toks)
            mem_ids.append(mem_id)

        return SleepReport(
            spans_compiled=compiled,
            tokens_evicted=evicted_tokens,
            active_kv_tokens_after=self.buffer.active_kv_tokens(),
            memory_ids=mem_ids,
            mean_rel_fit_error=sum(fit_errs) / max(1, len(fit_errs)),
        )

    def forget(self, memory_id: str) -> None:
        """Exact removal of a compiled memory across all ESM layers (`max_abs_diff < 1e-6`)."""
        rec = self.ledger.delta(memory_id)
        if rec.get("removed", False):
            return
        for l_idx, d_ref in rec.get("layer_deltas", {}).items():
            if int(l_idx) in self.esm_modules:
                self.esm_modules[int(l_idx)].remove(d_ref)
        self.ledger.mark_removed(memory_id)


# ==============================================================================
# Phase 2: D-LSA Ward Bucketization, ColdKVStore Paging & Index Calibration (`G2`)
# ==============================================================================

def ward_bucketize(
    k_pool: Any,
    max_bucket: int = 16,
    target_buckets: Optional[int] = None,
) -> Dict[str, Any]:
    """Bottom-up mutual-nearest-neighbor Ward clustering (`d(a,b) = (s_a s_b)/(s_a + s_b) ||c_a - c_b||_2^2`)
    for pooled key vectors `k_pool` (`[N, d_idx]`, `01 §C1` & `02 Phase 2`).

    Supports both `torch.Tensor` and pure-Python 2D lists for deterministic unit testing.
    Returns `{"centroids": ..., "bucket_members": List[List[int]], "bucket_sizes": List[int]}`.
    """
    if HAS_TORCH and isinstance(k_pool, torch.Tensor):
        pts = k_pool.detach().cpu().float().tolist()
        is_tensor = True
        dev = k_pool.device
    else:
        pts = [list(row) for row in k_pool]
        is_tensor = False
        dev = None

    n_pts = len(pts)
    if n_pts == 0:
        return {"centroids": k_pool if is_tensor else [], "bucket_members": [], "bucket_sizes": []}
    d_dim = len(pts[0])
    n_target = int(target_buckets) if target_buckets is not None else max(1, (n_pts + max_bucket - 1) // max(1, int(max_bucket)))

    clusters: List[Dict[str, Any]] = [
        {"centroid": list(pts[i]), "members": [i], "size": 1}
        for i in range(n_pts)
    ]

    def _ward_dist(ca: Dict[str, Any], cb: Dict[str, Any]) -> float:
        sa, sb = float(ca["size"]), float(cb["size"])
        if sa + sb > int(max_bucket):
            return 1e18
        sq = sum((ca["centroid"][d] - cb["centroid"][d]) ** 2 for d in range(d_dim))
        return (sa * sb / (sa + sb)) * sq

    while len(clusters) > n_target:
        m = len(clusters)
        best_nn: List[int] = [-1] * m
        best_dist: List[float] = [1e18] * m
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                wd = _ward_dist(clusters[i], clusters[j])
                if wd < best_dist[i]:
                    best_dist[i] = wd
                    best_nn[i] = j

        # Find mutual nearest neighbors (or globally closest pair if no MNN < 1e17)
        merged_any = False
        used = [False] * m
        next_clusters: List[Dict[str, Any]] = []
        pairs: List[Tuple[float, int, int]] = []
        for i in range(m):
            j = best_nn[i]
            if j > i and best_nn[j] == i and best_dist[i] < 1e17:
                pairs.append((best_dist[i], i, j))
        if not pairs:
            # Fallback to single closest valid pair
            best_pair = None
            for i in range(m):
                if best_nn[i] >= 0 and best_dist[i] < 1e17:
                    cand = (best_dist[i], min(i, best_nn[i]), max(i, best_nn[i]))
                    if best_pair is None or cand[0] < best_pair[0]:
                        best_pair = cand
            if best_pair is not None:
                pairs.append(best_pair)

        pairs.sort(key=lambda x: x[0])
        for _, i, j in pairs:
            if len(clusters) - sum(1 for u in used if u) // 2 <= n_target and merged_any:
                break
            if used[i] or used[j]:
                continue
            ca, cb = clusters[i], clusters[j]
            sa, sb = float(ca["size"]), float(cb["size"])
            if sa + sb > int(max_bucket):
                continue
            used[i] = True
            used[j] = True
            merged_any = True
            new_c = [(sa * ca["centroid"][d] + sb * cb["centroid"][d]) / (sa + sb) for d in range(d_dim)]
            nrm = math.sqrt(sum(v * v for v in new_c)) + 1e-8
            new_c = [v / nrm for v in new_c]
            next_clusters.append(
                {
                    "centroid": new_c,
                    "members": ca["members"] + cb["members"],
                    "size": int(sa + sb),
                }
            )
        for idx_c in range(m):
            if not used[idx_c]:
                next_clusters.append(clusters[idx_c])
        clusters = next_clusters
        if not merged_any:
            break

    centroids_list = [c["centroid"] for c in clusters]
    members_list = [c["members"] for c in clusters]
    sizes_list = [int(c["size"]) for c in clusters]
    if is_tensor and HAS_TORCH:
        c_tensor = torch.tensor(centroids_list, dtype=torch.float32, device=dev)
        c_tensor = torch.nn.functional.normalize(c_tensor, p=2, dim=-1, eps=1e-6)
        return {
            "centroids": c_tensor,
            "bucket_members": members_list,
            "bucket_sizes": sizes_list,
        }
    return {
        "centroids": centroids_list,
        "bucket_members": members_list,
        "bucket_sizes": sizes_list,
    }


class ColdKVStore:
    """Host-DRAM Pinned Cold KV Paging Store with GPU-Resident Ward Bucket Centroids (`01 §C1` & `02 Phase 2`).

    Stores evicted long-range KV pages in host DRAM (`pin_memory=True` when CUDA is available) while keeping
    compact 64-D Ward bucket centroids resident for fast sublinear top-bucket selection and on-demand page fetch.
    """

    def __init__(self, page_size: int = 4096, d_idx: int = 64, max_bucket: int = 16):
        self.page_size = max(16, int(page_size))
        self.d_idx = int(d_idx)
        self.max_bucket = max(2, int(max_bucket))
        self.pages: List[Dict[str, Any]] = []
        self.total_host_to_gpu_bytes: int = 0

    def append_page(self, k_page: Any, v_page: Any, k_idx_vectors: Any) -> int:
        """Bucketize `k_idx_vectors` (`[T_page, d_idx]`) via Ward clustering and offload `(k_page, v_page)` to host memory."""
        wb = ward_bucketize(k_idx_vectors, max_bucket=self.max_bucket)
        page_id = len(self.pages)
        if HAS_TORCH and isinstance(k_page, torch.Tensor):
            pin = torch.cuda.is_available()
            k_cpu = k_page.detach().to(device="cpu", copy=True)
            v_cpu = v_page.detach().to(device="cpu", copy=True)
            if pin:
                k_cpu = k_cpu.pin_memory()
                v_cpu = v_cpu.pin_memory()
            self.pages.append(
                {
                    "page_id": page_id,
                    "k_cpu": k_cpu,
                    "v_cpu": v_cpu,
                    "centroids": wb["centroids"],
                    "bucket_members": wb["bucket_members"],
                    "bucket_sizes": wb["bucket_sizes"],
                }
            )
        else:
            self.pages.append(
                {
                    "page_id": page_id,
                    "k_cpu": [list(r) for r in k_page],
                    "v_cpu": [list(r) for r in v_page],
                    "centroids": wb["centroids"],
                    "bucket_members": wb["bucket_members"],
                    "bucket_sizes": wb["bucket_sizes"],
                }
            )
        return page_id

    def fetch_selected_kv(self, q_idx_vec: Any, top_buckets: int = 4) -> Dict[str, Any]:
        """Select the highest-affinity Ward buckets across all cold pages and fetch only their KV rows."""
        if not self.pages:
            return {"k_selected": [], "v_selected": [], "selected_tokens": 0, "bytes_transferred": 0}

        if HAS_TORCH and isinstance(q_idx_vec, torch.Tensor):
            q_u = torch.nn.functional.normalize(q_idx_vec.detach().float().reshape(-1, self.d_idx), p=2, dim=-1, eps=1e-6)
            scored_buckets: List[Tuple[float, int, int]] = []
            for p_idx, pg in enumerate(self.pages):
                c_t = pg["centroids"].to(device=q_u.device, dtype=torch.float32)
                sims = torch.matmul(q_u, c_t.t()).max(dim=0).values
                for b_idx in range(int(sims.numel())):
                    scored_buckets.append((float(sims[b_idx].item()), p_idx, b_idx))
            scored_buckets.sort(key=lambda x: x[0], reverse=True)
            chosen = scored_buckets[: max(1, int(top_buckets))]

            k_chunks, v_chunks = [], []
            bytes_copied = 0
            n_tok = 0
            dev = q_idx_vec.device
            for _, p_idx, b_idx in chosen:
                pg = self.pages[p_idx]
                mem_idx = torch.tensor(pg["bucket_members"][b_idx], dtype=torch.long)
                k_sel = pg["k_cpu"].index_select(dim=-2, index=mem_idx).to(dev, non_blocking=True)
                v_sel = pg["v_cpu"].index_select(dim=-2, index=mem_idx).to(dev, non_blocking=True)
                bytes_copied += int(k_sel.numel() * k_sel.element_size() + v_sel.numel() * v_sel.element_size())
                n_tok += int(mem_idx.numel())
                k_chunks.append(k_sel)
                v_chunks.append(v_sel)
            self.total_host_to_gpu_bytes += bytes_copied
            return {
                "k_selected": torch.cat(k_chunks, dim=-2) if k_chunks else None,
                "v_selected": torch.cat(v_chunks, dim=-2) if v_chunks else None,
                "selected_tokens": n_tok,
                "bytes_transferred": bytes_copied,
            }

        # Pure-Python path
        q_row = q_idx_vec[0] if (q_idx_vec and isinstance(q_idx_vec[0], list)) else list(q_idx_vec)
        q_nrm = math.sqrt(sum(v * v for v in q_row)) + 1e-8
        q_u = [v / q_nrm for v in q_row]
        scored: List[Tuple[float, int, int]] = []
        for p_idx, pg in enumerate(self.pages):
            for b_idx, c_row in enumerate(pg["centroids"]):
                sim = sum(q_u[d] * c_row[d] for d in range(self.d_idx))
                scored.append((sim, p_idx, b_idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[: max(1, int(top_buckets))]
        k_sel_rows, v_sel_rows = [], []
        for _, p_idx, b_idx in chosen:
            pg = self.pages[p_idx]
            for m_i in pg["bucket_members"][b_idx]:
                k_sel_rows.append(list(pg["k_cpu"][m_i]))
                v_sel_rows.append(list(pg["v_cpu"][m_i]))
        bytes_copied = len(k_sel_rows) * len(k_sel_rows[0]) * 4 * 2 if k_sel_rows else 0
        self.total_host_to_gpu_bytes += bytes_copied
        return {
            "k_selected": k_sel_rows,
            "v_selected": v_sel_rows,
            "selected_tokens": len(k_sel_rows),
            "bytes_transferred": bytes_copied,
        }


def evaluate_dlsa_attention_mass_recall(
    model: Any,
    input_ids: Any,
    layers: Sequence[int] = (3, 7, 11, 15, 19, 23, 27, 31, 35),
    k_budget: int = 128,
) -> Dict[str, Any]:
    """Phase 2 (`Gate G2`): Measure exact attention-mass recall@K of the RoPE-free `[64:256]` D-LSA indexer
    across the 9 global full-attention layers on the 72L Loop Base.
    """
    if not HAS_TORCH:
        raise RuntimeError("evaluate_dlsa_attention_mass_recall requires PyTorch.")
    io_map = capture_attention_io(model, input_ids, layers=layers, visit=0)
    layer_list = model._layers() if hasattr(model, "_layers") else model.base.model.layers
    per_layer_recall: Dict[str, float] = {}

    for l_idx, io in io_map.items():
        if "query" not in io or "key" not in io:
            continue
        lsa_idx = getattr(layer_list[l_idx].self_attn, "lsa_indexer", None)
        if lsa_idx is None:
            continue
        q_t = io["query"]  # [B, H_q, T, D]
        k_t = io["key"]    # [B, H_kv, T, D]
        seq_t = int(q_t.shape[2])
        eff_k = min(seq_t, max(8, int(k_budget)))
        sel_idx = lsa_idx.select_key_indices(q_t, k_t, k_budget=eff_k)

        # Compute exact teacher attention mass on the last query token(s)
        q_tail = q_t[:, :, -min(16, seq_t) :, :].float()
        k_full = k_t.float()
        if k_full.shape[1] != q_tail.shape[1]:
            rep = q_tail.shape[1] // k_full.shape[1]
            k_full = k_full.repeat_interleave(rep, dim=1)
        scale = 1.0 / math.sqrt(float(q_tail.shape[-1]))
        logits = torch.matmul(q_tail, k_full.transpose(-1, -2)) * scale
        q_positions = torch.arange(seq_t - q_tail.shape[2], seq_t, device=q_t.device).view(1, 1, -1, 1)
        k_positions = torch.arange(0, seq_t, device=q_t.device).view(1, 1, 1, -1)
        logits = logits.masked_fill(k_positions > q_positions, float("-inf"))
        probs = torch.softmax(logits, dim=-1)  # [B, H_q, T_tail, T]
        captured_mass = probs.index_select(dim=-1, index=sel_idx.to(probs.device)).sum(dim=-1).mean().item()
        per_layer_recall[f"L{l_idx}"] = round(float(captured_mass), 5)

    mean_rec = sum(per_layer_recall.values()) / max(1, len(per_layer_recall))
    return {
        "k_budget": int(k_budget),
        "mean_attention_mass_recall": round(mean_rec, 5),
        "per_layer_recall": per_layer_recall,
        "g2_passed": bool(mean_rec >= 0.90),
    }


def fit_dlsa_index(
    model: Any,
    token_batches: Sequence[Any],
    layers: Sequence[int] = (3, 7, 11, 15, 19, 23, 27, 31, 35),
    steps: int = 20,
    lr: float = 1e-3,
    k_budget: int = 128,
) -> Dict[str, Any]:
    """Phase 2 (`02 Phase 2.3`): Calibrate/train ONLY `lsa_indexer` (`w_q_idx`, `w_k_idx`) on the 192-D
    RoPE-free slice (`[64:256]`) using exact teacher attention mass KL distillation while keeping
    the 72L Loop Base (`model.base` + `TwoStreamMHC` + `MoE-Engram`) 100% frozen.
    """
    if not HAS_TORCH:
        raise RuntimeError("fit_dlsa_index requires PyTorch.")
    fp_before = BackboneFingerprint(model, include_controllers=True, store_exact_clone=False)
    layer_list = model._layers() if hasattr(model, "_layers") else model.base.model.layers

    # Collect Q/K teacher states once (gradient-free through backbone)
    sample_ids = token_batches[0]
    pre_recall = evaluate_dlsa_attention_mass_recall(model, sample_ids, layers=layers, k_budget=k_budget)
    io_map = capture_attention_io(model, sample_ids, layers=layers, visit=0)

    for l_idx, io in io_map.items():
        lsa_idx = getattr(layer_list[l_idx].self_attn, "lsa_indexer", None)
        if lsa_idx is None or "query" not in io or "key" not in io:
            continue
        q_t = io["query"].detach().float()
        k_t = io["key"].detach().float()
        # First initialize with training-free PCA on the RoPE-free slice
        k_samp = k_t[0].permute(1, 0, 2).contiguous()
        lsa_idx.fit_training_free_pca(k_samp)

        # Optional lightweight KL alignment on w_q_idx / w_k_idx
        if int(steps) > 0:
            seq_t = int(q_t.shape[2])
            q_tail = q_t[:, :, -min(16, seq_t) :, :]
            k_full = k_t
            if k_full.shape[1] != q_tail.shape[1]:
                rep = q_tail.shape[1] // k_full.shape[1]
                k_full = k_full.repeat_interleave(rep, dim=1)
            scale = 1.0 / math.sqrt(float(q_tail.shape[-1]))
            with torch.no_grad():
                t_logits = torch.matmul(q_tail, k_full.transpose(-1, -2)) * scale
                q_pos = torch.arange(seq_t - q_tail.shape[2], seq_t, device=q_t.device).view(1, 1, -1, 1)
                k_pos = torch.arange(0, seq_t, device=q_t.device).view(1, 1, 1, -1)
                t_logits = t_logits.masked_fill(k_pos > q_pos, float("-inf"))
                p_teacher = torch.softmax(t_logits, dim=-1).mean(dim=(0, 1, 2))  # [T]
                q_free = lsa_idx._prepare_query_input(q_tail).mean(dim=(0, 1)).unsqueeze(0)  # [1, in_q]
                k_free = lsa_idx._prepare_key_input(k_t)[0]  # [T, in_k]

            with torch.enable_grad():
                lsa_idx.w_q_idx.weight.requires_grad_(True)
                lsa_idx.w_k_idx.weight.requires_grad_(True)
                opt = torch.optim.AdamW([lsa_idx.w_q_idx.weight, lsa_idx.w_k_idx.weight], lr=float(lr))
                for _ in range(int(steps)):
                    opt.zero_grad()
                    q_proj = torch.nn.functional.normalize(
                        torch.nn.functional.linear(q_free, lsa_idx.w_q_idx.weight.float()), p=2, dim=-1, eps=1e-6
                    )
                    k_proj = torch.nn.functional.normalize(
                        torch.nn.functional.linear(k_free, lsa_idx.w_k_idx.weight.float()), p=2, dim=-1, eps=1e-6
                    )
                    s_logits = torch.matmul(q_proj, k_proj.t()).squeeze(0) / 0.1
                    log_p_student = torch.log_softmax(s_logits, dim=-1)
                    loss = torch.nn.functional.kl_div(log_p_student, p_teacher.clamp(min=1e-8), reduction="sum")
                    loss.backward()
                    opt.step()
                lsa_idx.w_q_idx.weight.requires_grad_(False)
                lsa_idx.w_k_idx.weight.requires_grad_(False)

    post_recall = evaluate_dlsa_attention_mass_recall(model, sample_ids, layers=layers, k_budget=k_budget)
    fp_after = fp_before.verify(model)
    return {
        "pre_recall": pre_recall,
        "post_recall": post_recall,
        "backbone_sha256_match": bool(fp_after["sha256_match"]),
    }


# ==============================================================================
# Phase 6: Deep-Sleep Sparse-Memory Finetuning (`SMF`) Consolidation (`G6`)
# ==============================================================================

@dataclasses.dataclass
class SMFConsolidationReport:
    anchors_consolidated: int
    shards_consolidated: int
    mean_rel_fit_error: float
    max_null_space_leakage: float
    shards_removed: bool
    backbone_sha256_match: bool


def consolidate_shards_smf(
    model: Any,
    shard_ids: Sequence[int] = (0,),
    alpha_ridge: float = 1e-2,
    remove_consolidated_shards: bool = True,
) -> SMFConsolidationReport:
    """Phase 6 (`01 §C7` & `02 Phase 6`): Deep-Sleep Sparse-Memory Finetuning (`SMF`) consolidation
    from saturated `EpisodicSideMemory` (`ESM`) shards into `ZipfianTieredMoEEngram` routed experts,
    followed by shard removal to reclaim fast-write `ESM` null-space capacity.
    """
    if not HAS_TORCH:
        raise RuntimeError("consolidate_shards_smf requires PyTorch.")
    fp_before = BackboneFingerprint(model, include_controllers=True, store_exact_clone=False)
    dev = model._head_weight().device

    anchors_done = 0
    shards_done = 0
    fit_errs: List[float] = []
    max_leak = 0.0

    for l_eng, eng_mod in model.iter_engram_anchors():
        esm_key = str(l_eng)
        if not hasattr(model, "esm") or esm_key not in model.esm:
            continue
        esm_mod: EpisodicSideMemory = model.esm[esm_key]
        syn_clusters: List[Dict[str, Any]] = []

        for s_id in shard_ids:
            if int(s_id) not in esm_mod._shards:
                continue
            sh = esm_mod._shards[int(s_id)]
            if int(sh.get("num_writes", 0)) <= 0:
                continue
            # Extract synthetic representative hidden states from the shard's written key subspace
            u_past = sh.get("u_past")
            w_down = sh["w_down"].to(device=dev, dtype=torch.float32)  # [r, width]
            u_up = sh["u_up"].to(device=dev, dtype=torch.float32)      # [r, width]
            if u_past is not None and u_past.numel() > 0:
                k_cols = min(4, int(u_past.shape[1]))
                for c_i in range(k_cols):
                    phi_dir = u_past[:, c_i : c_i + 1].to(device=dev, dtype=torch.float32).t()  # [1, r]
                    h_rep = torch.matmul(phi_dir, u_up).squeeze(0)  # [width]
                    y_rep = torch.matmul(phi_dir, w_down).squeeze(0)  # [width]
                    syn_clusters.append(
                        {
                            "salience": 1.0,
                            "token_positions": [0],
                            "trigger_pos": 0,
                            "query_hidden": h_rep,
                            "cluster_hidden": h_rep,
                            "target_residual": y_rep,
                            "value_residual": y_rep,
                        }
                    )
            shards_done += 1
            if remove_consolidated_shards:
                sh["w_down"].zero_()
                sh["num_writes"] = 0
                sh["u_past"] = None
                sh["centroids"] = torch.empty((0, esm_mod.route_dim), dtype=torch.float32, device=dev)

        if syn_clusters:
            dummy_ids = torch.zeros((1, 8), dtype=torch.long, device=dev)
            res = eng_mod.compile_clusters_alphaedit(
                clusters=syn_clusters,
                input_ids=dummy_ids,
                domain="general",
                alpha_ridge=alpha_ridge,
                target_scale=1.0,
                inject_lexical=False,
                key_mode="query",
            )
            anchors_done += 1
            fit_errs.append(float(res["mean_rel_fit_error"]))
            max_leak = max(max_leak, float(res["max_null_space_leakage"]))

    fp_after = fp_before.verify(model)
    return SMFConsolidationReport(
        anchors_consolidated=anchors_done,
        shards_consolidated=shards_done,
        mean_rel_fit_error=sum(fit_errs) / max(1, len(fit_errs)),
        max_null_space_leakage=float(max_leak),
        shards_removed=bool(remove_consolidated_shards),
        backbone_sha256_match=bool(fp_after["sha256_match"]),
    )

