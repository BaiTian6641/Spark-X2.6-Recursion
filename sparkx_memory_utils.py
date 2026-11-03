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
        cpu_contig = t_det.cpu().contiguous()
        raw_bytes = bytes(cpu_contig.view(torch.uint8).numpy().tobytes()) if hasattr(cpu_contig, "numpy") else b""
        if not raw_bytes:
            # Fallback if numpy bridge is unavailable
            flat_f32 = cpu_contig.float().view(-1).tolist()
            raw_bytes = ",".join(f"{v:.8e}" for v in flat_f32[: min(numel, 65536)]).encode("utf-8")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        l2_val = float(cpu_contig.float().norm().item()) if numel > 0 else 0.0
        if numel == 0:
            idxs: List[int] = []
            vals: List[float] = []
        else:
            step = max(1, numel // max(1, int(sample_size)))
            idxs = list(range(0, numel, step))[:sample_size]
            flat_cpu = cpu_contig.float().view(-1)
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
    ) -> None:
        self.include_controllers = bool(include_controllers)
        self.ignore_substrings = tuple(str(s) for s in ignore_substrings)
        self.sample_size = int(sample_size)
        self.store_full_clone = bool(store_full_clone)
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
