"""sparkx_paged_cache.py — 72-Visit Hybrid Bounded Local Ring + Paged Global KV Cache for Spark X2.6 4B Loop72 LSA (Work Packages P1 & P3).

Resolves high-priority findings from §3.1 and implements §5.2 & §6.1–§6.4 of the Loop72 LSA Review Plan:
  1. 72 independent virtual-visit slots (`54` local sliding rings + `18` global paged archives).
  2. Local sliding visits (`M0`) are strictly bounded to `local_window = 512` resident tokens (`108 MiB` total across all 54
     visits in BF16 regardless of sequence length), while tracking monotonic `logical_seq_length` and exact `abs_positions`
     so offset-aware causal/sliding masks remain 100% exact across chunked prefill and decode.
  3. Global visits (`M1` Hot GPU + `M2` Cold CPU/Quantized Archive) are the **sole owners** of historical global K/V:
     - Protected `sinks = 4` tokens and `recent_tokens = 512` tail stay on GPU.
     - Completed middle pages (`kv_page_tokens = 64`) are archived to CPU (`bf16`, `int8`, `int4`, or `mixed_rotary_bf16`)
       with an LRU GPU hot page cache strictly bounded by `global_hot_tokens_per_visit`.
     - Coalesced page-level `gather_selected(selected_positions)` reconstructs exact selected `(K, V)` tensors without
       materializing full-history GPU K/V, tracking page hits/misses, bytes transferred, and physical page amplification.
  4. Each global visit slot owns its own `VisitStateKey` and isolated `dlsa_state_slot` so recurrent visits to the same
     physical layer (`L7, L11, L15, L19, L23, L27`) and concurrent sessions never overwrite one another's index state.
"""
from __future__ import annotations

from collections import OrderedDict
import dataclasses
import os
import resource
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

from sparkx_kv_codec import KVCodecConfig, QuantizedKVPage, decode_kv_page, encode_kv_page
if HAS_TORCH:
    from sparkx_dlsa import LOOP72_VISIT_SCHEDULE, VisitStateKey, build_loop72_visit_schedule
else:  # pragma: no cover
    LOOP72_VISIT_SCHEDULE = []
    VisitStateKey = Any  # type: ignore[misc,assignment]
    build_loop72_visit_schedule = None  # type: ignore[assignment]


def _dedup_storage_bytes(tensors: Iterable[Any], device_type: Optional[str] = None) -> int:
    """Compute unique physical `untyped_storage().nbytes()` across `tensors` deduplicated by `(device, data_ptr)` (`F10`)."""
    seen: set = set()
    total = 0
    for t in tensors:
        if not HAS_TORCH or not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        if device_type is not None and t.device.type != device_type:
            continue
        st = t.untyped_storage()
        key = (t.device.type, t.device.index, int(st.data_ptr()))
        if key not in seen:
            seen.add(key)
            total += int(st.nbytes())
    return total


def _host_rss_bytes() -> int:
    """Return current process resident set size (RSS) in bytes (`F10`)."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            parts = fh.read().split()
            if len(parts) >= 2:
                return int(parts[1]) * int(os.sysconf("SC_PAGESIZE"))
    except Exception:
        pass
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_maxrss) * 1024
    except Exception:  # pragma: no cover
        return 0


@dataclasses.dataclass
class PagedCacheConfig:
    local_window: int = 512
    sinks: int = 4
    recent_tokens: int = 512
    global_hot_tokens_per_visit: int = 8192
    kv_page_tokens: int = 64
    cold_codec: str = "bf16"
    codec_group_size: int = 64
    rope_dim: int = 64
    rope_first: bool = True
    index_gpu_budget_bytes: int = 2 * (1024**3)
    enforce_index_budget: bool = True
    persistent_token_latents: bool = False
    context_limit_tokens: int = 1_048_576
    reserved_output_tokens: int = 0
    enforce_context_limit: bool = True
    archive_backend: str = "cpu"
    nvme_enabled: bool = False
    nvme_spill_dir: Optional[str] = None
    host_cold_budget_bytes: int = 0
    num_global_visits: int = 18
    session_id: str = "default"
    model_revision: str = "XHToken/Spark-X2.6-4B"
    behavior_fingerprint: str = "native_gelu_loop72"
    projection_revision: str = "loop72_dlsa_v1"
    memory_epoch: int = 0
    keep_full_gpu_mirror_for_legacy: bool = False

    def __post_init__(self) -> None:
        backend = str(self.archive_backend or "cpu").strip().lower()
        if backend not in ("cpu", "cpu_then_nvme"):
            raise ValueError(f"Unsupported archive_backend={self.archive_backend!r}; expected 'cpu' or 'cpu_then_nvme'.")
        self.archive_backend = backend
        if backend == "cpu_then_nvme":
            if not self.nvme_enabled or not self.nvme_spill_dir:
                raise ValueError(
                    "archive_backend='cpu_then_nvme' requires nvme_enabled=True and a valid nvme_spill_dir (§5.1, F11)."
                )


class NVMeColdPageStore:
    """Disk-backed NVMe spill archive for `QuantizedKVPage` blobs when `archive_backend == 'cpu_then_nvme'` (`F11`)."""

    def __init__(self, base_dir: str, session_id: str, virtual_visit: int) -> None:
        safe_sess = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(session_id or "default"))
        self.visit_dir = os.path.join(os.path.abspath(base_dir), f"sess_{safe_sess}", f"visit_{int(virtual_visit):02d}")
        os.makedirs(self.visit_dir, exist_ok=True)
        self._page_paths: Dict[int, str] = {}
        self._page_bytes: Dict[int, int] = {}
        self.nvme_spills: int = 0
        self.nvme_reads: int = 0

    @property
    def page_paths(self) -> Dict[int, str]:
        return self._page_paths

    @property
    def total_bytes(self) -> int:
        return int(sum(self._page_bytes.values()))

    def has_page(self, page_id: int) -> bool:
        return int(page_id) in self._page_paths

    def __contains__(self, page_id: int) -> bool:
        return int(page_id) in self._page_paths

    def __len__(self) -> int:
        return len(self._page_paths)

    def spill_page(self, page_id: int, q_page: QuantizedKVPage) -> int:
        if not HAS_TORCH:
            raise RuntimeError("NVMeColdPageStore requires PyTorch.")
        pid = int(page_id)
        path = os.path.join(self.visit_dir, f"page_{pid:08d}.pt")
        tmp_path = f"{path}.tmp.{os.getpid()}"
        torch.save(q_page, tmp_path)
        os.replace(tmp_path, path)
        n_bytes = int(os.path.getsize(path))
        self._page_paths[pid] = path
        self._page_bytes[pid] = n_bytes
        self.nvme_spills += 1
        return n_bytes

    def save_page(self, page_id: int, q_page: QuantizedKVPage) -> int:
        return self.spill_page(page_id, q_page)

    def load_page(self, page_id: int) -> QuantizedKVPage:
        if not HAS_TORCH:
            raise RuntimeError("NVMeColdPageStore requires PyTorch.")
        pid = int(page_id)
        path = self._page_paths[pid]
        self.nvme_reads += 1
        return torch.load(path, map_location="cpu", weights_only=False)

    def storage_bytes(self) -> int:
        return int(sum(self._page_bytes.values()))

    def clear(self) -> None:
        self._page_paths.clear()
        self._page_bytes.clear()
        if os.path.isdir(self.visit_dir):
            shutil.rmtree(self.visit_dir, ignore_errors=True)
            os.makedirs(self.visit_dir, exist_ok=True)


class LocalSlidingRingSlot:
    """Bounded `local_window`-token KV ring for a single local sliding-window execution visit (`M0`).

    Uses a preallocated `2 * local_window` rolling GPU buffer so single-token decode (`q_len == 1`) writes in-place
    with zero `torch.cat` allocations on `local_window - 1` out of every `local_window` steps (and a single in-place
    `.copy_()` slide when the rolling buffer reaches `2 * local_window`), while always exposing contiguous chronological
    views `self.keys`, `self.values`, and `self.abs_positions` (`[<=W]`).
    """

    def __init__(
        self,
        virtual_visit: int,
        physical_layer: int,
        pass_idx: int,
        local_window: int = 512,
        context_limit_tokens: int = 1_048_576,
        reserved_output_tokens: int = 0,
        enforce_context_limit: bool = True,
    ) -> None:
        self.virtual_visit = int(virtual_visit)
        self.physical_layer = int(physical_layer)
        self.pass_idx = int(pass_idx)
        self.is_global = False
        self.local_window = max(1, int(local_window))
        self.context_limit_tokens = max(1, int(context_limit_tokens))
        self.reserved_output_tokens = max(0, int(reserved_output_tokens))
        self.enforce_context_limit = bool(enforce_context_limit)
        self.logical_seq_length: int = 0
        self.keys: Optional["torch.Tensor"] = None          # [B, H_kv, <=W, D] view into rolling buffer
        self.values: Optional["torch.Tensor"] = None        # [B, H_kv, <=W, D] view into rolling buffer
        self.abs_positions: Optional["torch.Tensor"] = None  # [<=W] view into rolling buffer
        # During a multi-token prefill chunk, holds the transient `[<=W + q_len]` positions for offset-aware masking
        self.last_returned_positions: Optional["torch.Tensor"] = None
        # Preallocated `2 * W` rolling buffers for allocation-free single-token decode
        self._buf_k: Optional["torch.Tensor"] = None
        self._buf_v: Optional["torch.Tensor"] = None
        self._buf_pos: Optional["torch.Tensor"] = None
        self._write_pos: int = 0

    @property
    def resident_tokens(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.shape[-2])

    def logical_gpu_bytes(self) -> int:
        """Visible active window payload bytes (`<= local_window` tokens)."""
        if self.keys is None or self.values is None:
            return 0
        pos_b = int(self.abs_positions.numel() * self.abs_positions.element_size()) if self.abs_positions is not None else 0
        return int(
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
            + pos_b
        )

    def allocated_gpu_bytes(self) -> int:
        """True physical backing allocation bytes including the `2 * local_window` rolling K/V/pos buffers (`F10`)."""
        return _dedup_storage_bytes(self.owned_tensors())

    def gpu_bytes(self) -> int:
        """Return physical allocated backing bytes (`F10`: never under-count the `2 * W` rolling buffer)."""
        return self.allocated_gpu_bytes()

    def owned_tensors(self) -> List["torch.Tensor"]:
        out: List["torch.Tensor"] = []
        for t in (self._buf_k, self._buf_v, self._buf_pos, self.keys, self.values, self.abs_positions):
            if isinstance(t, torch.Tensor):
                out.append(t)
        return out

    def _check_context_limit(self, new_logical_len: int, is_initial_prefill: bool = False) -> None:
        if not self.enforce_context_limit:
            return
        reserve = self.reserved_output_tokens if is_initial_prefill else 0
        if int(new_logical_len) + reserve > self.context_limit_tokens:
            raise ValueError(
                f"LocalSlidingRingSlot(visit={self.virtual_visit}): sequence length {new_logical_len} + "
                f"reserved_output_tokens {reserve} exceeds context_limit_tokens {self.context_limit_tokens}."
            )

    def _ensure_rolling_buffers(self, bsz: int, h_kv: int, head_dim: int, device: "torch.device", dtype: "torch.dtype") -> None:
        cap = 2 * self.local_window
        if (
            self._buf_k is None
            or self._buf_k.shape != (bsz, h_kv, cap, head_dim)
            or self._buf_k.device != device
            or self._buf_k.dtype != dtype
        ):
            self._buf_k = torch.empty((bsz, h_kv, cap, head_dim), device=device, dtype=dtype)
            self._buf_v = torch.empty((bsz, h_kv, cap, head_dim), device=device, dtype=dtype)
            self._buf_pos = torch.empty((cap,), device=device, dtype=torch.long)
            self._write_pos = 0
            if self.keys is not None and self.values is not None and self.abs_positions is not None:
                cur_len = min(int(self.keys.shape[-2]), self.local_window)
                if cur_len > 0:
                    self._buf_k[:, :, :cur_len, :].copy_(self.keys[:, :, -cur_len:, :].to(device=device, dtype=dtype))
                    self._buf_v[:, :, :cur_len, :].copy_(self.values[:, :, -cur_len:, :].to(device=device, dtype=dtype))
                    self._buf_pos[:cur_len].copy_(self.abs_positions[-cur_len:].to(device=device, dtype=torch.long))
                    self._write_pos = cur_len

    def update(
        self,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        cache_position: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        bsz, h_kv, q_len, head_dim = key_states.shape
        dev = key_states.device
        dtype = key_states.dtype
        start_pos = self.logical_seq_length

        # Fast path for single-token decode (`q_len == 1`): in-place write into rolling buffer with zero `torch.cat`
        if q_len == 1:
            if cache_position is not None and cache_position.numel() == 1:
                next_logical_len = (int(cache_position.reshape(-1)[0].item()) + 1) if start_pos == 0 else (start_pos + 1)
            else:
                next_logical_len = start_pos + 1
            self._check_context_limit(next_logical_len, is_initial_prefill=(start_pos == 0))
            self._ensure_rolling_buffers(bsz, h_kv, head_dim, dev, dtype)
            assert self._buf_k is not None and self._buf_v is not None and self._buf_pos is not None
            cap = 2 * self.local_window
            if self._write_pos >= cap:
                keep = self.local_window - 1
                if keep > 0:
                    # Clone slice before in-place copy to avoid overlapping memory view issues in PyTorch
                    k_tail = self._buf_k[:, :, self._write_pos - keep : self._write_pos, :].clone()
                    v_tail = self._buf_v[:, :, self._write_pos - keep : self._write_pos, :].clone()
                    p_tail = self._buf_pos[self._write_pos - keep : self._write_pos].clone()
                    self._buf_k[:, :, :keep, :].copy_(k_tail)
                    self._buf_v[:, :, :keep, :].copy_(v_tail)
                    self._buf_pos[:keep].copy_(p_tail)
                self._write_pos = keep

            wp = self._write_pos
            self._buf_k[:, :, wp : wp + 1, :].copy_(key_states.detach())
            self._buf_v[:, :, wp : wp + 1, :].copy_(value_states.detach())
            if cache_position is not None and cache_position.numel() == 1:
                self._buf_pos[wp : wp + 1].copy_(cache_position.detach().to(device=dev, dtype=torch.long).reshape(1))
            else:
                self._buf_pos[wp] = start_pos
            self.logical_seq_length = next_logical_len

            self._write_pos = wp + 1
            win_start = max(0, self._write_pos - self.local_window)
            self.keys = self._buf_k[:, :, win_start : self._write_pos, :]
            self.values = self._buf_v[:, :, win_start : self._write_pos, :]
            self.abs_positions = self._buf_pos[win_start : self._write_pos]
            self.last_returned_positions = self.abs_positions
            return self.keys, self.values

        # Multi-token prefill / chunked prefill path (`q_len > 1`)
        if cache_position is not None and cache_position.numel() == q_len:
            new_pos = cache_position.detach().to(device=dev, dtype=torch.long).reshape(-1)
            next_logical_len = max(self.logical_seq_length, int(new_pos[-1].item()) + 1)
        else:
            new_pos = torch.arange(start_pos, start_pos + q_len, device=dev, dtype=torch.long)
            next_logical_len = start_pos + q_len
        self._check_context_limit(next_logical_len, is_initial_prefill=(start_pos == 0))
        self.logical_seq_length = next_logical_len

        if self.keys is None or self.values is None or self.abs_positions is None:
            k_full = key_states
            v_full = value_states
            pos_full = new_pos
        else:
            k_full = torch.cat([self.keys.to(dev), key_states], dim=-2)
            v_full = torch.cat([self.values.to(dev), value_states], dim=-2)
            pos_full = torch.cat([self.abs_positions.to(dev), new_pos], dim=0)

        self.last_returned_positions = pos_full
        self._ensure_rolling_buffers(bsz, h_kv, head_dim, dev, dtype)
        assert self._buf_k is not None and self._buf_v is not None and self._buf_pos is not None

        keep_len = min(int(k_full.shape[-2]), self.local_window)
        self._buf_k[:, :, :keep_len, :].copy_(k_full[:, :, -keep_len:, :].detach())
        self._buf_v[:, :, :keep_len, :].copy_(v_full[:, :, -keep_len:, :].detach())
        self._buf_pos[:keep_len].copy_(pos_full[-keep_len:].detach())
        self._write_pos = keep_len
        self.keys = self._buf_k[:, :, :keep_len, :]
        self.values = self._buf_v[:, :, :keep_len, :]
        self.abs_positions = self._buf_pos[:keep_len]
        return k_full, v_full

    def reset(self) -> None:
        self.logical_seq_length = 0
        self.keys = None
        self.values = None
        self.abs_positions = None
        self.last_returned_positions = None
        self._buf_k = None
        self._buf_v = None
        self._buf_pos = None
        self._write_pos = 0


class GlobalPagedVisitSlot:
    """Sole-owner paged global KV archive (`M1` GPU hot owned slab LRU + `M2` CPU quantized pages + optional NVMe spill)
    and isolated D-LSA state slot for one global execution visit (`virtual_visit`, `global_visit_id`).

    Includes two high-speed execution regimes:
      1. **Fast GPU-Resident Contiguous Regime (`logical_seq_length <= global_hot_tokens_per_visit` and `codec == 'bf16'`)**:
         Keeps a preallocated contiguous GPU buffer `_gpu_res_k / _gpu_res_v` up to `global_hot_tokens_per_visit` (`8,192`
         tokens by default) so single-token decode writes in-place without `torch.cat` or CPU page archival, and exact/sparse
         reads slice or `index_select` directly on GPU with zero Python page loops and zero `.item()` syncs.
      2. **Paged Archive Regime (`logical_seq_length > global_hot_tokens_per_visit` or `codec != 'bf16'`)**:
         Transitions automatically into `sink_k/v` + `cold_pages` (+ optional `NVMeColdPageStore`) + owned-slab `hot_pages_lru`
         + `tail_k/v` with batched `tolist()` page-ID transfers and Hits-First Whole-Gather LRU protection.
    """

    def __init__(
        self,
        virtual_visit: int,
        physical_layer: int,
        pass_idx: int,
        global_visit_id: int,
        layer_visit_ordinal: int,
        cfg: PagedCacheConfig,
    ) -> None:
        self.virtual_visit = int(virtual_visit)
        self.physical_layer = int(physical_layer)
        self.pass_idx = int(pass_idx)
        self.global_visit_id = int(global_visit_id)
        self.layer_visit_ordinal = int(layer_visit_ordinal)
        self.is_global = True
        self.cfg = cfg
        self.codec_cfg = KVCodecConfig(
            codec=cfg.cold_codec,
            group_size=cfg.codec_group_size,
            rope_dim=cfg.rope_dim,
            rope_first=cfg.rope_first,
        )
        self.state_key = VisitStateKey(
            model_revision=cfg.model_revision,
            behavior_fingerprint=cfg.behavior_fingerprint,
            projection_revision=cfg.projection_revision,
            session_id=cfg.session_id,
            memory_epoch=cfg.memory_epoch,
            physical_layer=self.physical_layer,
            virtual_visit=self.virtual_visit,
        )

        self.sinks = max(0, int(cfg.sinks))
        self.recent_tokens = max(16, int(cfg.recent_tokens))
        self.page_tokens = max(8, int(cfg.kv_page_tokens))
        self.gpu_resident_cap = max(self.sinks + self.recent_tokens, int(cfg.global_hot_tokens_per_visit))
        hot_mid_budget = max(self.page_tokens, int(cfg.global_hot_tokens_per_visit) - self.sinks - self.recent_tokens)
        self.max_hot_pages = max(1, hot_mid_budget // self.page_tokens)

        self.logical_seq_length: int = 0
        self.sink_k: Optional["torch.Tensor"] = None   # [B, H_kv, <=sinks, D] on GPU
        self.sink_v: Optional["torch.Tensor"] = None
        self.tail_k: Optional["torch.Tensor"] = None   # [B, H_kv, tail_len, D] on GPU (covers [paged_end, logical_seq_length))
        self.tail_v: Optional["torch.Tensor"] = None
        self.paged_end: int = self.sinks               # tokens in [sinks, paged_end) live in `cold_pages` / `nvme_store`

        # Fast GPU-resident contiguous buffer for `[0..logical_seq_length)` while `<= gpu_resident_cap` and `codec == "bf16"`
        self._gpu_res_k: Optional["torch.Tensor"] = None
        self._gpu_res_v: Optional["torch.Tensor"] = None
        self._gpu_res_len: int = 0

        self.cold_pages: "OrderedDict[int, QuantizedKVPage]" = OrderedDict()
        self.nvme_store: Optional[NVMeColdPageStore] = (
            NVMeColdPageStore(cfg.nvme_spill_dir, cfg.session_id, self.virtual_visit)
            if cfg.archive_backend == "cpu_then_nvme"
            else None
        )
        self.hot_pages_lru: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._free_hot_slabs: List[Tuple["torch.Tensor", "torch.Tensor"]] = []

        # Isolated per-visit D-LSA index state slot
        self.dlsa_state_slot: Dict[str, Any] = {}

        # Operational telemetry
        self.page_hits: int = 0
        self.page_misses: int = 0
        self.bytes_transferred: int = 0
        self.tokens_requested_from_pages: int = 0
        self.tokens_fetched_in_pages: int = 0
        self.sparse_Step_calls: int = 0
        self.index_cpu_offloads: int = 0
        self.nvme_spills: int = 0
        self.nvme_loads: int = 0

    def _has_archived_pages(self) -> bool:
        return bool(self.cold_pages) or (self.nvme_store is not None and len(self.nvme_store.page_paths) > 0)

    def _acquire_hot_slab(self, k_src: "torch.Tensor", v_src: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Copy `(k_src, v_src)` into an owned page-sized slab (`F10`: never retain views into oversized `tail_k/v`)."""
        while self._free_hot_slabs:
            kp, vp = self._free_hot_slabs.pop()
            if (
                kp.shape == k_src.shape
                and vp.shape == v_src.shape
                and kp.device == k_src.device
                and kp.dtype == k_src.dtype
                and vp.dtype == v_src.dtype
            ):
                kp.copy_(k_src.detach())
                vp.copy_(v_src.detach())
                return kp, vp
        return k_src.detach().clone().contiguous(), v_src.detach().clone().contiguous()

    def _release_hot_slab(self, kp: "torch.Tensor", vp: "torch.Tensor") -> None:
        if len(self._free_hot_slabs) < self.max_hot_pages:
            self._free_hot_slabs.append((kp, vp))

    def _enforce_host_cold_budget(self) -> None:
        if self.nvme_store is None or int(self.cfg.host_cold_budget_bytes) <= 0 or not self.cold_pages:
            return
        per_visit_cap = max(1, int(self.cfg.host_cold_budget_bytes) // max(1, int(self.cfg.num_global_visits)))
        cur_bytes = self.cold_archive_bytes()
        while cur_bytes > per_visit_cap and self.cold_pages:
            ev_pid, ev_page = self.cold_pages.popitem(last=False)
            cur_bytes -= int(ev_page.storage_bytes())
            self.nvme_store.save_page(ev_pid, ev_page)
            self.nvme_spills += 1

    @property
    def keys(self) -> Optional["torch.Tensor"]:
        """Compatibility property returning resident GPU buffer or tail."""
        if self._gpu_res_k is not None and self._gpu_res_len > 0 and not self._has_archived_pages():
            return self._gpu_res_k[:, :, : self._gpu_res_len, :]
        return self.tail_k

    @property
    def values(self) -> Optional["torch.Tensor"]:
        if self._gpu_res_v is not None and self._gpu_res_len > 0 and not self._has_archived_pages():
            return self._gpu_res_v[:, :, : self._gpu_res_len, :]
        return self.tail_v

    def is_fully_gpu_resident(self, upto: Optional[int] = None) -> bool:
        """Return True if all tokens `[0..upto)` are stored contiguously in `_gpu_res_k / _gpu_res_v` on GPU."""
        target = self.logical_seq_length if upto is None else int(upto)
        return (
            self._gpu_res_k is not None
            and self._gpu_res_v is not None
            and not self._has_archived_pages()
            and self._gpu_res_len >= target
            and target > 0
        )

    def get_resident_prefix(self, upto: Optional[int] = None) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Return a zero-copy GPU slice `(k_prefix, v_prefix)` of `[0..upto)` when `is_fully_gpu_resident(upto)` is True."""
        target = self.logical_seq_length if upto is None else int(upto)
        assert self._gpu_res_k is not None and self._gpu_res_v is not None
        return self._gpu_res_k[:, :, :target, :], self._gpu_res_v[:, :, :target, :]

    def _append_to_gpu_resident(self, key_states: "torch.Tensor", value_states: "torch.Tensor") -> None:
        bsz, h_kv, q_len, head_dim = key_states.shape
        new_len = self._gpu_res_len + q_len
        if self._gpu_res_k is None or self._gpu_res_v is None:
            init_cap = min(self.gpu_resident_cap, max(1024, 1 << (max(1, new_len - 1).bit_length())))
            init_cap = max(init_cap, new_len)
            self._gpu_res_k = torch.empty((bsz, h_kv, init_cap, head_dim), device=key_states.device, dtype=key_states.dtype)
            self._gpu_res_v = torch.empty((bsz, h_kv, init_cap, head_dim), device=value_states.device, dtype=value_states.dtype)
        elif new_len > int(self._gpu_res_k.shape[-2]):
            cur_cap = int(self._gpu_res_k.shape[-2])
            next_cap = min(self.gpu_resident_cap, max(new_len, cur_cap * 2))
            next_cap = max(next_cap, new_len)
            new_k = torch.empty((bsz, h_kv, next_cap, head_dim), device=key_states.device, dtype=key_states.dtype)
            new_v = torch.empty((bsz, h_kv, next_cap, head_dim), device=value_states.device, dtype=value_states.dtype)
            if self._gpu_res_len > 0:
                new_k[:, :, : self._gpu_res_len, :].copy_(self._gpu_res_k[:, :, : self._gpu_res_len, :])
                new_v[:, :, : self._gpu_res_len, :].copy_(self._gpu_res_v[:, :, : self._gpu_res_len, :])
            self._gpu_res_k = new_k
            self._gpu_res_v = new_v

        self._gpu_res_k[:, :, self._gpu_res_len : new_len, :].copy_(key_states.detach())
        self._gpu_res_v[:, :, self._gpu_res_len : new_len, :].copy_(value_states.detach())
        self._gpu_res_len = new_len

    def _archive_completed_pages(self, force: bool = False) -> int:
        """Move completed `page_tokens` chunks that have aged out of `recent_tokens` from GPU `tail_k/v` into `cold_pages`.

        When `self.codec_cfg.codec == "bf16"` and `self.logical_seq_length <= self.gpu_resident_cap` and `not force`
        and `not self._has_archived_pages()`, we defer CPU page archival so ordinary contexts stay 100% GPU-resident.
        """
        if self.tail_k is None or self.tail_v is None:
            return 0
        if (
            not force
            and self.codec_cfg.codec == "bf16"
            and self.logical_seq_length <= self.gpu_resident_cap
            and not self._has_archived_pages()
        ):
            return 0

        tail_len = int(self.tail_k.shape[-2])
        archivable = max(0, tail_len - self.recent_tokens)
        n_new_pages = archivable // self.page_tokens
        if n_new_pages <= 0:
            return 0

        for p_i in range(n_new_pages):
            s_off = p_i * self.page_tokens
            e_off = s_off + self.page_tokens
            abs_start = self.paged_end + s_off
            page_id = (abs_start - self.sinks) // self.page_tokens
            k_slice = self.tail_k[:, :, s_off:e_off, :].detach()
            v_slice = self.tail_v[:, :, s_off:e_off, :].detach()
            pos_range = list(range(abs_start, abs_start + self.page_tokens))
            q_page = encode_kv_page(k_slice, v_slice, self.codec_cfg, positions=pos_range, store_device="cpu")
            self.cold_pages[page_id] = q_page
            if self.codec_cfg.codec == "bf16":
                kp_owned, vp_owned = self._acquire_hot_slab(k_slice, v_slice)
                self._insert_hot_page(page_id, kp_owned, vp_owned, already_owned=True)

        self._enforce_host_cold_budget()
        consumed = n_new_pages * self.page_tokens
        self.tail_k = self.tail_k[:, :, consumed:, :].detach().clone().contiguous()
        self.tail_v = self.tail_v[:, :, consumed:, :].detach().clone().contiguous()
        self.paged_end += consumed
        # Once cold pages exist, release the contiguous GPU-resident mirror so GPU memory stays bounded by `global_hot_tokens_per_visit`
        if self._has_archived_pages():
            self._gpu_res_k = None
            self._gpu_res_v = None
            self._gpu_res_len = 0
        return n_new_pages

    def _insert_hot_page(
        self,
        page_id: int,
        k_gpu: "torch.Tensor",
        v_gpu: "torch.Tensor",
        protected_pids: Optional[set] = None,
        already_owned: bool = False,
    ) -> None:
        kp_owned, vp_owned = (k_gpu, v_gpu) if already_owned else self._acquire_hot_slab(k_gpu, v_gpu)
        if page_id in self.hot_pages_lru:
            old_kp, old_vp = self.hot_pages_lru[page_id]
            if old_kp.data_ptr() != kp_owned.data_ptr():
                self._release_hot_slab(old_kp, old_vp)
            self.hot_pages_lru.move_to_end(page_id)
            self.hot_pages_lru[page_id] = (kp_owned, vp_owned)
            return
        while len(self.hot_pages_lru) >= self.max_hot_pages:
            if protected_pids:
                evict_key = next((k for k in self.hot_pages_lru.keys() if k not in protected_pids), None)
                if evict_key is not None:
                    ev_kp, ev_vp = self.hot_pages_lru.pop(evict_key)
                    self._release_hot_slab(ev_kp, ev_vp)
                    continue
            ev_pid, (ev_kp, ev_vp) = self.hot_pages_lru.popitem(last=False)
            if not protected_pids or ev_pid not in protected_pids:
                self._release_hot_slab(ev_kp, ev_vp)
        self.hot_pages_lru[page_id] = (kp_owned, vp_owned)

    def _fetch_page_to_gpu(
        self,
        page_id: int,
        device: "torch.device",
        dtype: "torch.dtype",
        protected_pids: Optional[set] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if page_id in self.hot_pages_lru:
            self.page_hits += 1
            self.hot_pages_lru.move_to_end(page_id)
            kp, vp = self.hot_pages_lru[page_id]
            return kp.to(device=device, dtype=dtype), vp.to(device=device, dtype=dtype)

        self.page_misses += 1
        if page_id in self.cold_pages:
            q_page = self.cold_pages[page_id]
            self.cold_pages.move_to_end(page_id)
        elif self.nvme_store is not None and self.nvme_store.has_page(page_id):
            q_page = self.nvme_store.load_page(page_id)
            self.nvme_loads += 1
        else:
            raise KeyError(f"GlobalPagedVisitSlot(visit={self.virtual_visit}): missing archived page_id={page_id}")
        self.bytes_transferred += q_page.storage_bytes()
        kp, vp = decode_kv_page(q_page, device=device, dtype=dtype)
        kp_owned, vp_owned = self._acquire_hot_slab(kp, vp)
        self._insert_hot_page(page_id, kp_owned, vp_owned, protected_pids=protected_pids, already_owned=True)
        return kp_owned, vp_owned

    def _check_context_limit(self, new_logical_len: int, is_initial_prefill: bool = False) -> None:
        if not self.cfg.enforce_context_limit:
            return
        reserve = int(self.cfg.reserved_output_tokens) if is_initial_prefill else 0
        if int(new_logical_len) + reserve > int(self.cfg.context_limit_tokens):
            raise ValueError(
                f"GlobalPagedVisitSlot(visit={self.virtual_visit}): sequence length {new_logical_len} + "
                f"reserved_output_tokens {reserve} exceeds context_limit_tokens {self.cfg.context_limit_tokens}."
            )

    def enforce_gpu_index_budget(self, device_type: str = "cuda") -> bool:
        """Enforce `cfg.index_gpu_budget_bytes` on this visit's `PagedLatentIndex` (`F11`). Returns True if offloaded."""
        if not self.cfg.enforce_index_budget:
            return False
        idx = self.dlsa_state_slot.get("index")
        if idx is None or not hasattr(idx, "index_allocated_bytes"):
            return False
        per_visit_cap = max(0, int(self.cfg.index_gpu_budget_bytes) // max(1, int(self.cfg.num_global_visits)))
        if int(idx.index_allocated_bytes(device_type)) > per_visit_cap:
            idx.offload_index_to_cpu()
            self.index_cpu_offloads += 1
            return True
        return False

    def append(
        self,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        cache_position: Optional["torch.Tensor"] = None,
        indexer: Optional[Any] = None,
    ) -> int:
        """Append new key/value states `[B, H_kv, q_len, D]` into this global visit's paged archive and incrementally
        update its per-visit D-LSA index (`PagedLatentIndex`) on the unrotated slice before pages are archived to CPU."""
        q_len = int(key_states.shape[-2])
        if q_len <= 0:
            return self.logical_seq_length
        old_len = self.logical_seq_length
        if cache_position is not None and cache_position.numel() == q_len:
            if q_len == 1 and old_len > 0:
                next_logical_len = old_len + 1
            else:
                next_logical_len = max(self.logical_seq_length, int(cache_position[-1].item()) + 1)
        else:
            next_logical_len = old_len + q_len
        self._check_context_limit(next_logical_len, is_initial_prefill=(old_len == 0 and q_len > 1))
        self.logical_seq_length = next_logical_len

        # 1. Update per-visit D-LSA index incrementally using only the newly arrived keys `key_states[0]`
        if indexer is not None:
            from sparkx_dlsa import PagedLatentIndex, _dummy_q
            k0_new = key_states[0]  # [H_kv, q_len, D]
            if hasattr(indexer, "_free_key_features"):
                k_in_new = indexer._free_key_features(k0_new)
            else:
                _, k_in_new = indexer.inputs(_dummy_q(k0_new), k0_new)
            k_idx_new = indexer.project_k(k_in_new)
            idx: Optional[PagedLatentIndex] = self.dlsa_state_slot.get("index")
            if idx is None:
                idx = PagedLatentIndex(
                    k_idx_new,
                    n_sink=indexer.n_sink,
                    n_pool=indexer.n_pool,
                    n_cluster=indexer.n_cluster,
                    avg_bucket_size=indexer.avg_bucket_size,
                    max_bucket_size=indexer.max_bucket_size,
                    window=indexer.window,
                    rerank=indexer.rerank,
                    search_mode=indexer.search_mode,
                    branching_factor=indexer.branching_factor,
                    beam_width=indexer.beam_width,
                    persistent_token_latents=self.cfg.persistent_token_latents,
                )
                self.dlsa_state_slot["index"] = idx
            else:
                idx.append_new_keys(k_idx_new)
            if key_states.is_cuda:
                self.enforce_gpu_index_budget("cuda")

        # 2. Maintain fast GPU-resident contiguous buffer while within `gpu_resident_cap` and `codec == "bf16"`
        if self.codec_cfg.codec == "bf16" and not self._has_archived_pages() and self.logical_seq_length <= self.gpu_resident_cap:
            self._append_to_gpu_resident(key_states, value_states)
            assert self._gpu_res_k is not None and self._gpu_res_v is not None
            sink_end = min(self._gpu_res_len, self.sinks)
            self.sink_k = self._gpu_res_k[:, :, :sink_end, :]
            self.sink_v = self._gpu_res_v[:, :, :sink_end, :]
            self.paged_end = sink_end
            if self._gpu_res_len > sink_end:
                self.tail_k = self._gpu_res_k[:, :, sink_end : self._gpu_res_len, :]
                self.tail_v = self._gpu_res_v[:, :, sink_end : self._gpu_res_len, :]
            else:
                self.tail_k = None
                self.tail_v = None
            return self.logical_seq_length
        elif self._gpu_res_k is not None:
            if self.sink_k is not None:
                self.sink_k = self.sink_k.clone().contiguous()
            if self.sink_v is not None:
                self.sink_v = self.sink_v.clone().contiguous()
            if self.tail_k is not None:
                self.tail_k = self.tail_k.clone().contiguous()
            if self.tail_v is not None:
                self.tail_v = self.tail_v.clone().contiguous()
            self._gpu_res_k = None
            self._gpu_res_v = None
            self._gpu_res_len = 0

        # 3. Route tokens into `sink_k/v` (`[0, sinks)`) and `tail_k/v` (`[paged_end, logical_seq_length)`)
        sink_cur = 0 if self.sink_k is None else int(self.sink_k.shape[-2])
        if sink_cur < self.sinks:
            need_sink = min(q_len, self.sinks - sink_cur)
            ks_part = key_states[:, :, :need_sink, :].detach().clone().contiguous()
            vs_part = value_states[:, :, :need_sink, :].detach().clone().contiguous()
            if self.sink_k is None:
                self.sink_k = ks_part
                self.sink_v = vs_part
            else:
                self.sink_k = torch.cat([self.sink_k, ks_part], dim=-2)
                self.sink_v = torch.cat([self.sink_v, vs_part], dim=-2)
            self.paged_end = int(self.sink_k.shape[-2])
            rem_k = key_states[:, :, need_sink:, :].detach()
            rem_v = value_states[:, :, need_sink:, :].detach()
        else:
            rem_k = key_states.detach()
            rem_v = value_states.detach()

        if int(rem_k.shape[-2]) > 0:
            if self.tail_k is None or self.tail_v is None:
                self.tail_k = rem_k.clone().contiguous()
                self.tail_v = rem_v.clone().contiguous()
            else:
                self.tail_k = torch.cat([self.tail_k, rem_k], dim=-2)
                self.tail_v = torch.cat([self.tail_v, rem_v], dim=-2)
            self._archive_completed_pages()

        return self.logical_seq_length

    def gather_selected(
        self,
        selected_positions: "torch.Tensor",
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        return_positions: Optional[bool] = None,
    ) -> Any:
        """Gather exact `(K, V)` tensors `[B, H_kv, |selected|, D]` (and optionally `selected_positions` when
        `return_positions=True` or called with positional `selected_positions` only) across GPU sink, paged middle archive,
        and GPU recent tail without materializing full-history K/V."""
        if return_positions is None:
            want_pos = device is None and dtype is None
        else:
            want_pos = bool(return_positions)

        ref_t = self._gpu_res_k if self._gpu_res_k is not None else (self.tail_k if self.tail_k is not None else self.sink_k)
        if ref_t is None and not self._has_archived_pages():
            raise RuntimeError(f"GlobalPagedVisitSlot(visit={self.virtual_visit}) is empty")
        if device is None:
            device = ref_t.device if ref_t is not None else torch.device("cpu")
        if dtype is None:
            dtype = ref_t.dtype if ref_t is not None else torch.bfloat16

        sel = selected_positions.to(device=device, dtype=torch.long).reshape(-1)
        if sel.numel() == 0:
            bsz = int(ref_t.shape[0]) if ref_t is not None else 1
            h_kv = int(ref_t.shape[1]) if ref_t is not None else 4
            d = int(ref_t.shape[3]) if ref_t is not None else 256
            empty = torch.zeros((bsz, h_kv, 0, d), device=device, dtype=dtype)
            return (empty, empty.clone(), sel) if want_pos else (empty, empty.clone())

        # Fast GPU-Resident Gather: single `index_select` on `_gpu_res_k / _gpu_res_v` with zero page loops and zero `.item()` syncs
        if self.is_fully_gpu_resident(self.logical_seq_length):
            assert self._gpu_res_k is not None and self._gpu_res_v is not None
            sel_dev = sel.to(self._gpu_res_k.device)
            k_out = self._gpu_res_k[:, :, : self._gpu_res_len, :].index_select(-2, sel_dev).to(device=device, dtype=dtype)
            v_out = self._gpu_res_v[:, :, : self._gpu_res_len, :].index_select(-2, sel_dev).to(device=device, dtype=dtype)
            return (k_out, v_out, sel) if want_pos else (k_out, v_out)

        sink_len = 0 if self.sink_k is None else int(self.sink_k.shape[-2])
        k_parts: List["torch.Tensor"] = []
        v_parts: List["torch.Tensor"] = []

        # 1. Sink positions `[0, sink_len)`
        sink_mask = sel < sink_len
        if self.sink_k is not None and self.sink_v is not None:
            s_idx = sel[sink_mask]
            if s_idx.numel() > 0:
                s_dev = s_idx.to(self.sink_k.device)
                k_parts.append(self.sink_k.index_select(-2, s_dev).to(device=device, dtype=dtype))
                v_parts.append(self.sink_v.index_select(-2, s_dev).to(device=device, dtype=dtype))

        # 2. Paged middle positions `[sink_len, paged_end)` coalesced by physical page
        mid_mask = (sel >= sink_len) & (sel < self.paged_end)
        mid_pos = sel[mid_mask]
        if mid_pos.numel() > 0:
            self.tokens_requested_from_pages += int(mid_pos.numel())
            page_ids = (mid_pos - self.sinks) // self.page_tokens
            page_offs = (mid_pos - self.sinks) % self.page_tokens
            unique_pages, inverse_idx = torch.unique_consecutive(page_ids, return_inverse=True)
            # Single batched D2H transfer of unique page IDs instead of per-page `.item()` inside the loop
            unique_page_list: List[int] = [int(x) for x in unique_pages.tolist()]
            self.tokens_fetched_in_pages += len(unique_page_list) * self.page_tokens

            protected_set = set(unique_page_list)
            fetched_pages: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]] = {}
            # Phase A: Service and touch all LRU hits first so no hit in the current gather can be evicted by a miss
            for pid in unique_page_list:
                if pid in self.hot_pages_lru:
                    self.page_hits += 1
                    self.hot_pages_lru.move_to_end(pid)
                    kp, vp = self.hot_pages_lru[pid]
                    fetched_pages[pid] = (kp.to(device=device, dtype=dtype), vp.to(device=device, dtype=dtype))
            # Phase B: Fetch misses with protected eviction
            for pid in unique_page_list:
                if pid not in fetched_pages:
                    fetched_pages[pid] = self._fetch_page_to_gpu(
                        pid, device=device, dtype=dtype, protected_pids=protected_set
                    )
            # Phase C: Coalesced single index_select across all fetched pages (replaces O(U) per-page kernel loop with 1 gather)
            if len(unique_page_list) == 1:
                kp0, vp0 = fetched_pages[unique_page_list[0]]
                k_parts.append(kp0.index_select(-2, page_offs))
                v_parts.append(vp0.index_select(-2, page_offs))
            else:
                k_pages = torch.cat([fetched_pages[pid][0] for pid in unique_page_list], dim=-2)
                v_pages = torch.cat([fetched_pages[pid][1] for pid in unique_page_list], dim=-2)
                flat_gather_idx = inverse_idx * self.page_tokens + page_offs
                k_parts.append(k_pages.index_select(-2, flat_gather_idx))
                v_parts.append(v_pages.index_select(-2, flat_gather_idx))

        # 3. Resident tail positions `[paged_end, logical_seq_length)`
        tail_mask = sel >= self.paged_end
        if self.tail_k is not None and self.tail_v is not None:
            t_raw = sel[tail_mask] - self.paged_end
            if t_raw.numel() > 0:
                t_idx = t_raw.to(self.tail_k.device)
                valid_t = t_idx < int(self.tail_k.shape[-2])
                t_idx = t_idx[valid_t]
                if t_idx.numel() > 0:
                    k_parts.append(self.tail_k.index_select(-2, t_idx).to(device=device, dtype=dtype))
                    v_parts.append(self.tail_v.index_select(-2, t_idx).to(device=device, dtype=dtype))

        k_cat = torch.cat(k_parts, dim=-2)
        v_cat = torch.cat(v_parts, dim=-2)
        return (k_cat, v_cat, sel) if want_pos else (k_cat, v_cat)

    def materialize_all_for_exact_control(
        self, device: Optional["torch.device"] = None, dtype: Optional["torch.dtype"] = None
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Materialize `[0..logical_seq_length)` for short sequences (`<= exact_threshold`) or exact-attention controls."""
        if self.is_fully_gpu_resident(self.logical_seq_length):
            k_res, v_res = self.get_resident_prefix(self.logical_seq_length)
            if device is not None or dtype is not None:
                return k_res.to(device=device or k_res.device, dtype=dtype or k_res.dtype), v_res.to(device=device or v_res.device, dtype=dtype or v_res.dtype)
            return k_res, v_res
        all_pos = torch.arange(0, self.logical_seq_length, device=device or "cpu", dtype=torch.long)
        return self.gather_selected(all_pos, device=device, dtype=dtype, return_positions=False)

    def logical_gpu_hot_bytes(self) -> int:
        """Logical active hot payload bytes (used elements only, excluding preallocated capacity/free slabs)."""
        if self.is_fully_gpu_resident(self.logical_seq_length):
            k_res, v_res = self.get_resident_prefix(self.logical_seq_length)
            return int(k_res.numel() * k_res.element_size() + v_res.numel() * v_res.element_size())
        total = 0
        for t in (self.sink_k, self.sink_v, self.tail_k, self.tail_v):
            if isinstance(t, torch.Tensor):
                total += int(t.numel() * t.element_size())
        for kp, vp in self.hot_pages_lru.values():
            total += int(kp.numel() * kp.element_size() + vp.numel() * vp.element_size())
        return total

    def owned_gpu_tensors(self) -> List["torch.Tensor"]:
        """Return all resident hot/slab KV tensors owned by this slot (`F10`: true backing storage)."""
        out: List["torch.Tensor"] = []
        for t in (self._gpu_res_k, self._gpu_res_v, self.sink_k, self.sink_v, self.tail_k, self.tail_v):
            if isinstance(t, torch.Tensor):
                out.append(t)
        for kp, vp in self.hot_pages_lru.values():
            out.append(kp)
            out.append(vp)
        for kp, vp in self._free_hot_slabs:
            out.append(kp)
            out.append(vp)
        return out

    def owned_cpu_tensors(self) -> List["torch.Tensor"]:
        """Return CPU cold-page tensors + CPU-offloaded index tensors owned by this slot."""
        out: List["torch.Tensor"] = []
        for p in self.cold_pages.values():
            for attr_name in ("k_data", "v_data", "k_scale", "v_scale", "k_zero", "v_zero", "k_rot", "v_rot"):
                t = getattr(p, attr_name, None)
                if isinstance(t, torch.Tensor) and t.device.type == "cpu":
                    out.append(t)
        idx = self.dlsa_state_slot.get("index")
        if idx is not None and hasattr(idx, "index_tensors"):
            for t in idx.index_tensors():
                if isinstance(t, torch.Tensor) and t.device.type == "cpu":
                    out.append(t)
        return out

    def gpu_hot_bytes(self) -> int:
        """True physical allocated backing bytes for this slot's hot KV/slab buffers (`F10`)."""
        return _dedup_storage_bytes(self.owned_gpu_tensors())

    def cold_archive_bytes(self) -> int:
        return sum(p.storage_bytes() for p in self.cold_pages.values())

    def nvme_archive_bytes(self) -> int:
        return int(self.nvme_store.total_bytes) if self.nvme_store is not None else 0

    def index_bytes(self) -> int:
        idx = self.dlsa_state_slot.get("index")
        if idx is not None and hasattr(idx, "index_storage_bytes"):
            return int(idx.index_storage_bytes())
        return 0

    def index_allocated_bytes(self, device_type: Optional[str] = None) -> int:
        idx = self.dlsa_state_slot.get("index")
        if idx is not None and hasattr(idx, "index_allocated_bytes"):
            return int(idx.index_allocated_bytes(device_type=device_type))
        return 0

    def evict_all_hot_pages(self) -> int:
        self._archive_completed_pages(force=True)
        self._gpu_res_k = None
        self._gpu_res_v = None
        self._gpu_res_len = 0
        n = len(self.hot_pages_lru)
        self.hot_pages_lru.clear()
        self._free_hot_slabs.clear()
        return n

    def reset(self) -> None:
        self.logical_seq_length = 0
        self.sink_k = None
        self.sink_v = None
        self.tail_k = None
        self.tail_v = None
        self._gpu_res_k = None
        self._gpu_res_v = None
        self._gpu_res_len = 0
        self.paged_end = self.sinks
        self.cold_pages.clear()
        if self.nvme_store is not None:
            self.nvme_store.clear()
        self.hot_pages_lru.clear()
        self._free_hot_slabs.clear()
        self.dlsa_state_slot.clear()
        self.page_hits = 0
        self.page_misses = 0
        self.bytes_transferred = 0
        self.tokens_requested_from_pages = 0
        self.tokens_fetched_in_pages = 0
        self.sparse_Step_calls = 0
        self.index_cpu_offloads = 0
        self.nvme_spills = 0
        self.nvme_loads = 0


class Loop72HybridPagedCache:
    """72-Slot Hybrid Cache Backend for Spark X2.6 4B Loop72 LSA (`54` local sliding rings + `18` global paged archives).

    Compatible with `SparkXLoopV03` (`get_seq_length`, `layers`, `slots`, `update`, `update_local_visit`,
    `append_global_visit`) while providing direct per-visit paged D-LSA ownership (`get_visit_slot(virtual_visit)`).
    """

    is_sparkx_paged_cache: bool = True

    def __init__(
        self,
        cfg: Optional[PagedCacheConfig] = None,
        config: Optional[PagedCacheConfig] = None,
        visit_schedule: Optional[Sequence[Dict[str, Any]]] = None,
        enable_p3: bool = True,
        loop_passes: int = 2,
        loop_start: int = 4,
        loop_end: int = 28,
        p3_start: int = 12,
        p3_end: int = 24,
        n_layers: int = 36,
    ) -> None:
        self.cfg = cfg or config or PagedCacheConfig()
        self.config = self.cfg
        if visit_schedule is not None:
            self.schedule = list(visit_schedule)
        elif build_loop72_visit_schedule is not None:
            self.schedule = list(
                build_loop72_visit_schedule(
                    start=loop_start,
                    end=loop_end,
                    passes=loop_passes,
                    enable_p3=enable_p3,
                    p3_start=p3_start,
                    p3_end=p3_end,
                    n_layers=n_layers,
                )
            )
        else:
            self.schedule = list(LOOP72_VISIT_SCHEDULE)
        self.cfg.num_global_visits = max(1, sum(1 for r in self.schedule if bool(r["is_global"])))
        self.layers: List[Any] = []
        self.slots = self.layers
        self.local_slots: Dict[int, LocalSlidingRingSlot] = {}
        self.global_slots: Dict[int, GlobalPagedVisitSlot] = {}
        self.global_slots_by_gid: Dict[int, GlobalPagedVisitSlot] = {}

        for row in self.schedule:
            vv = int(row["virtual_visit"])
            pl = int(row["physical_layer"])
            p_idx = int(row["pass_idx"])
            if bool(row["is_global"]):
                gid = int(row["global_visit_id"])
                lo = int(row["layer_visit_ordinal"])
                g_slot = GlobalPagedVisitSlot(
                    virtual_visit=vv,
                    physical_layer=pl,
                    pass_idx=p_idx,
                    global_visit_id=gid,
                    layer_visit_ordinal=lo,
                    cfg=self.cfg,
                )
                self.layers.append(g_slot)
                self.global_slots[vv] = g_slot
                self.global_slots_by_gid[gid] = g_slot
            else:
                l_slot = LocalSlidingRingSlot(
                    virtual_visit=vv,
                    physical_layer=pl,
                    pass_idx=p_idx,
                    local_window=self.cfg.local_window,
                    context_limit_tokens=self.cfg.context_limit_tokens,
                    reserved_output_tokens=self.cfg.reserved_output_tokens,
                    enforce_context_limit=self.cfg.enforce_context_limit,
                )
                self.layers.append(l_slot)
                self.local_slots[vv] = l_slot

    def check_capacity(self, prompt_tokens: int, max_new_tokens: int = 0) -> None:
        """Upfront fail-closed context window admission check (`prompt_tokens + max(max_new_tokens, reserved_output_tokens)`)."""
        if not self.cfg.enforce_context_limit:
            return
        needed_out = max(int(max_new_tokens), int(self.cfg.reserved_output_tokens))
        total_req = int(prompt_tokens) + needed_out
        if total_req > int(self.cfg.context_limit_tokens):
            raise ValueError(
                f"Loop72HybridPagedCache capacity exceeded: prompt_tokens={prompt_tokens} + "
                f"reserved_output={needed_out} ({total_req}) > context_limit_tokens={self.cfg.context_limit_tokens}."
            )

    def get_visit_slot(self, virtual_visit: int) -> Any:
        return self.layers[int(virtual_visit)]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return the true monotonic logical sequence length (never truncated by the 512-token sliding ring!)."""
        idx = 0 if layer_idx is None else int(layer_idx)
        if 0 <= idx < len(self.layers):
            return int(self.layers[idx].logical_seq_length)
        return 0

    def get_logical_seq_length(self, virtual_visit: int = 0) -> int:
        return self.get_seq_length(virtual_visit)

    def update_local_visit(
        self,
        virtual_visit: int,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        cache_position: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Update a local sliding ring visit and return its bounded resident `(keys, values, abs_positions)` (`[<=W]`)."""
        slot = self.layers[int(virtual_visit)]
        if not isinstance(slot, LocalSlidingRingSlot):
            raise TypeError(f"Visit {virtual_visit} is not a LocalSlidingRingSlot")
        slot.update(key_states, value_states, cache_position=cache_position)
        assert slot.keys is not None and slot.values is not None and slot.abs_positions is not None
        return slot.keys, slot.values, slot.abs_positions

    def append_global_visit(
        self,
        virtual_visit: int,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        cache_position: Optional["torch.Tensor"] = None,
        indexer: Optional[Any] = None,
    ) -> GlobalPagedVisitSlot:
        """Append new K/V tokens to a global paged visit slot and return the updated `GlobalPagedVisitSlot`."""
        slot = self.layers[int(virtual_visit)]
        if not isinstance(slot, GlobalPagedVisitSlot):
            raise TypeError(f"Visit {virtual_visit} is not a GlobalPagedVisitSlot")
        slot.append(key_states, value_states, cache_position=cache_position, indexer=indexer)
        return slot

    def update(
        self,
        key_states: "torch.Tensor",
        value_states: "torch.Tensor",
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        slot = self.layers[int(layer_idx)]
        cache_pos = cache_kwargs.get("cache_position") if isinstance(cache_kwargs, dict) else None
        if isinstance(slot, LocalSlidingRingSlot):
            return slot.update(key_states, value_states, cache_position=cache_pos)
        indexer = cache_kwargs.get("indexer") if isinstance(cache_kwargs, dict) else None
        slot.append(key_states, value_states, cache_position=cache_pos, indexer=indexer)
        if bool(cache_kwargs.get("paged_defer_materialize", False)) if isinstance(cache_kwargs, dict) else False:
            return key_states, value_states
        return slot.materialize_all_for_exact_control(device=key_states.device, dtype=key_states.dtype)

    def reset(self) -> None:
        for slot in self.layers:
            slot.reset()

    def memory_ledger(self) -> Dict[str, Any]:
        local_gpu_alloc = sum(s.allocated_gpu_bytes() for s in self.local_slots.values())
        local_gpu_logical = sum(s.logical_gpu_bytes() for s in self.local_slots.values())
        global_hot_alloc = sum(s.gpu_hot_bytes() for s in self.global_slots.values())
        global_hot_logical = sum(s.logical_gpu_hot_bytes() for s in self.global_slots.values())
        global_cold_cpu = sum(s.cold_archive_bytes() for s in self.global_slots.values())
        nvme_cold_bytes = sum(s.nvme_archive_bytes() for s in self.global_slots.values())
        index_logical = sum(s.index_bytes() for s in self.global_slots.values())
        index_alloc = sum(s.index_allocated_bytes() for s in self.global_slots.values())
        index_gpu_alloc = sum(s.index_allocated_bytes("cuda") for s in self.global_slots.values())
        index_offloads = sum(s.index_cpu_offloads for s in self.global_slots.values())
        nvme_spills = sum(s.nvme_spills for s in self.global_slots.values())
        nvme_loads = sum(s.nvme_loads for s in self.global_slots.values())

        all_gpu_like_tensors: List["torch.Tensor"] = []
        all_cpu_tensors: List["torch.Tensor"] = []
        for l_s in self.local_slots.values():
            for t in l_s.owned_tensors():
                all_gpu_like_tensors.append(t)
        for g_s in self.global_slots.values():
            for t in g_s.owned_gpu_tensors():
                all_gpu_like_tensors.append(t)
            idx_obj = g_s.dlsa_state_slot.get("index")
            if idx_obj is not None and hasattr(idx_obj, "index_tensors"):
                for t in idx_obj.index_tensors():
                    if isinstance(t, torch.Tensor) and t.device.type != "cpu":
                        all_gpu_like_tensors.append(t)
            all_cpu_tensors.extend(g_s.owned_cpu_tensors())

        unique_gpu_backing = _dedup_storage_bytes(all_gpu_like_tensors)
        unique_cpu_backing = _dedup_storage_bytes(all_cpu_tensors)

        cuda_alloc = int(torch.cuda.memory_allocated()) if (HAS_TORCH and torch.cuda.is_available()) else 0
        cuda_reserved = int(torch.cuda.memory_reserved()) if (HAS_TORCH and torch.cuda.is_available()) else 0
        cuda_peak = int(torch.cuda.max_memory_allocated()) if (HAS_TORCH and torch.cuda.is_available()) else 0
        host_rss = _host_rss_bytes()

        hits = sum(s.page_hits for s in self.global_slots.values())
        misses = sum(s.page_misses for s in self.global_slots.values())
        transferred = sum(s.bytes_transferred for s in self.global_slots.values())
        req_toks = sum(s.tokens_requested_from_pages for s in self.global_slots.values())
        fetched_toks = sum(s.tokens_fetched_in_pages for s in self.global_slots.values())
        max_local_resident = max((s.resident_tokens for s in self.local_slots.values()), default=0)
        return {
            "total_visits": len(self.layers),
            "local_visits": len(self.local_slots),
            "global_visits": len(self.global_slots),
            "logical_seq_length": self.get_seq_length(0),
            "max_local_resident_tokens": max_local_resident,
            "local_window_cap": self.cfg.local_window,
            "local_rings_gpu_bytes": local_gpu_alloc,
            "local_rings_logical_bytes": local_gpu_logical,
            "global_hot_gpu_bytes": global_hot_alloc,
            "global_hot_logical_bytes": global_hot_logical,
            "global_cold_archive_bytes": global_cold_cpu,
            "nvme_cold_archive_bytes": nvme_cold_bytes,
            "index_bytes": index_alloc,
            "index_logical_bytes": index_logical,
            "index_allocated_bytes": index_alloc,
            "index_gpu_bytes": index_gpu_alloc,
            "index_cpu_offloads": index_offloads,
            "nvme_spills": nvme_spills,
            "nvme_loads": nvme_loads,
            "total_gpu_cache_bytes": unique_gpu_backing,
            "unique_gpu_backing_bytes": unique_gpu_backing,
            "unique_cpu_backing_bytes": unique_cpu_backing,
            "cuda_allocated_bytes": cuda_alloc,
            "cuda_reserved_bytes": cuda_reserved,
            "cuda_max_allocated_bytes": cuda_peak,
            "host_rss_bytes": host_rss,
            "m0_local_rings_mib": round(local_gpu_alloc / (1024.0 ** 2), 3),
            "m1_global_hot_mib": round(global_hot_alloc / (1024.0 ** 2), 3),
            "m1_gpu_total_mib": round(unique_gpu_backing / (1024.0 ** 2), 3),
            "m2_host_cold_pages_mib": round(global_cold_cpu / (1024.0 ** 2), 3),
            "m2_nvme_cold_pages_mib": round(nvme_cold_bytes / (1024.0 ** 2), 3),
            "cold_codec": self.cfg.cold_codec,
            "archive_backend": self.cfg.archive_backend,
            "page_hits": hits,
            "page_misses": misses,
            "page_hit_rate": round(hits / max(1, hits + misses), 4),
            "bytes_transferred": transferred,
            "physical_page_amplification": round(fetched_toks / max(1, req_toks), 4) if req_toks > 0 else 1.0,
        }
