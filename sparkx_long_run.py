"""sparkx_long_run.py — resumable, multi-session (Colab) long-run MoE-Engram + mHC training.

Design goals (each addresses a verified weakness of the 4-stage Cell-4 run):

* **Unique tokens over epochs.** The Cell-4 run made ~3.9 passes over ~5M tokens. Here the number of steps is derived from a
  wall-clock / token budget and data is curated in chunks so the run is (close to) single-epoch.
* **Leak-free validation.** Documents whose `sha1(prompt) % VAL_HASH_MOD == 0` are NEVER trained on. A fixed subset is frozen
  once and re-used for every evaluation in every session, so numbers are comparable across sessions.
* **Validation-driven checkpointing.** Best-by-held-out-CE(Engram ON) weights are kept; on a plateau we roll back to best and
  halve the LR instead of silently training on.
* **Survives session death.** Run state, curated chunks, `latest` and `best` weights live in a private Hub repo
  (`LONGRUN_REPO_ID`); a new Colab session resumes exactly (same data order, same LR schedule position). Optimizer moments are
  NOT persisted (≈GBs); after a resume the LR is re-warmed for a few steps.

Pure-Python helpers (planning, quotas, shard bumping, hashing, LR schedule, state, chunk IO, curation with injected
dependencies) are unit-tested locally; the GPU training path is NOT testable without a GPU and is written crash-proof where
feasible (validation / Hub sync never raise).
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

STATE_VERSION = 1


# ======================================================================================
# Pure helpers
# ======================================================================================
def doc_key(rec: Dict[str, Any]) -> str:
    """Stable 16-hex key of a trajectory, based on the whitespace-normalised, lower-cased user prompt."""
    text = " ".join(str(rec.get("user") or "").split()).lower()
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def is_val_doc(key: str, mod: int = 25) -> bool:
    """Deterministic held-out membership: a document is validation-only iff `int(key,16) % mod == 0`."""
    return int(key, 16) % max(1, int(mod)) == 0


def plan_run(
    hours: float,
    tok_per_s: float = 1450.0,
    seq_len: int = 16384,
    grad_accum: int = 8,
    fill: float = 0.876,
    avg_doc_tokens: float = 2137.0,
    eval_overhead: float = 0.07,
    data_margin: float = 1.10,
) -> Dict[str, Any]:
    """Convert a training-hours budget into steps / tokens / documents (defaults from the measured Cell-4 run).

    `tok_per_s` is the NOMINAL rate `seq_len*grad_accum/step_time` (what the training log prints); `fill` is the measured
    packing fill so the REAL token count is lower than nominal.
    """
    tokens_per_step = int(seq_len) * int(grad_accum)
    step_s = tokens_per_step / max(1e-6, float(tok_per_s))
    steps = max(1, int(float(hours) * 3600.0 * (1.0 - float(eval_overhead)) / step_s))
    real_tokens = steps * tokens_per_step * float(fill)
    docs = int(math.ceil(real_tokens / max(1.0, float(avg_doc_tokens)) * float(data_margin)))
    return {
        "hours": float(hours), "tok_per_s": float(tok_per_s), "seq_len": int(seq_len), "grad_accum": int(grad_accum),
        "tokens_per_step": tokens_per_step, "step_seconds": step_s, "steps": steps,
        "real_tokens": int(real_tokens), "docs_needed": docs,
    }


def redistribute_quotas(total: int, ratios: Dict[str, float], capacity: Dict[str, int]) -> Dict[str, int]:
    """Split `total` across pillars proportionally to `ratios`, never exceeding `capacity[p]`; shortfall of capped pillars is
    re-spread over the remaining pillars (water-filling). Returns integer quotas with `sum <= total`."""
    quotas = {p: 0 for p in ratios}
    remaining = int(total)
    active = [p for p in ratios if capacity.get(p, 0) > 0 and ratios[p] > 0]
    guard = 0
    while remaining > 0 and active and guard < 1000:
        guard += 1
        wsum = sum(ratios[p] for p in active)
        gave = 0
        for p in active:
            room = capacity[p] - quotas[p]
            want = int(round(remaining * ratios[p] / wsum))
            take = max(0, min(room, want, remaining - gave))
            quotas[p] += take
            gave += take
        if gave == 0:  # rounding starvation -> hand out single units
            for p in active:
                if gave < remaining and capacity[p] - quotas[p] > 0:
                    quotas[p] += 1
                    gave += 1
            if gave == 0:
                break
        remaining -= gave
        active = [p for p in active if quotas[p] < capacity[p]]
    return quotas


_SHARD_RE = re.compile(r"(\d{5})-of-(\d{5})")


def next_shard_path(path: Optional[str], bump: int = 1) -> Optional[str]:
    """`.../train-00000-of-00030.parquet` + bump=k -> `.../train-0000k-of-00030.parquet`; None if past the last shard or no pattern."""
    if path is None:
        return None
    if bump == 0:
        return path
    m = _SHARD_RE.search(path)
    if not m:
        return None
    j, n = int(m.group(1)) + int(bump), int(m.group(2))
    if j >= n:
        return None
    return path[: m.start()] + f"{j:05d}-of-{n:05d}" + path[m.end():]


def lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int = 20,
    min_ratio: float = 0.15,
    steps_since_resume: Optional[int] = None,
    rewarm_steps: int = 10,
) -> float:
    """Global warmup+cosine multiplier of the *global* step (so a resume continues the same schedule). After a resume the
    multiplier is additionally ramped over `rewarm_steps` because optimizer moments restart from zero."""
    step = int(step)
    warmup_steps = max(1, int(warmup_steps))
    if step < warmup_steps:
        m = float(step + 1) / float(warmup_steps)
    else:
        prog = (step - warmup_steps) / float(max(1, int(total_steps) - warmup_steps))
        prog = min(1.0, max(0.0, prog))
        m = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    if steps_since_resume is not None and rewarm_steps > 0 and steps_since_resume < rewarm_steps:
        m *= float(steps_since_resume + 1) / float(rewarm_steps)
    return m


@dataclasses.dataclass
class LongRunState:
    """Everything needed to resume exactly. Serialised to `state.json` (inf stored as null)."""
    version: int = STATE_VERSION
    plan: Dict[str, Any] = dataclasses.field(default_factory=dict)
    plan_hours: float = 20.0
    total_steps: int = 0
    global_step: int = 0
    tokens_seen: int = 0
    chunk_idx: int = 0
    seq_in_chunk: int = 0
    data_epoch: int = 0
    best_val_ce_on: Optional[float] = None
    best_step: int = 0
    no_improve: int = 0
    reductions: int = 0
    plateau_scale: float = 1.0
    baseline: Optional[Dict[str, Any]] = None
    val_history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    segments: int = 0
    calibrated: bool = False
    finished: bool = False
    updated: str = ""

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LongRunState":
        raw = json.loads(text)
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})


def write_jsonl_gz(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ======================================================================================
# Scalable, resumable, de-duplicated curation (dependencies injected -> unit-testable)
# ======================================================================================
def _src_key(spec: Sequence[Any]) -> str:
    name, subset, split = spec[0].strip(), spec[1], spec[2]
    shard = spec[3] if len(spec) > 3 else None
    return f"{name}|{shard or subset or ''}|{split}"


def new_manifest() -> Dict[str, Any]:
    return {"n_chunks": 0, "exhausted": False, "sources": {}, "pillar_counts": {}, "val_docs": 0}


def curate_chunk(
    n_docs: int,
    ratios: Dict[str, float],
    sources_by_pillar: Dict[str, Sequence[Sequence[Any]]],
    manifest: Dict[str, Any],
    seen: Set[str],
    open_fn: Callable[..., Tuple[Any, str]],
    extract_fn: Callable[[Any, str], Optional[Dict[str, Any]]],
    filter_fn: Callable[[Dict[str, Any]], bool],
    val_mod: int = 25,
    seed: int = 42,
    chunk_idx: int = 0,
    log: Callable[[str], None] = print,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Curate ONE chunk of ~`n_docs` NEW training trajectories continuing each source from its saved row offset.

    * Documents already in `seen` are skipped; validation-hash documents are diverted to the returned `val` list.
    * A source that runs out of rows advances to the next shard (`next_shard_path`) or is marked `done`; a source that yields
      nothing for two chunks in a row is marked `dead`. Quotas of finished pillars/sources are redistributed.
    """
    pillars = {p.strip(): float(r) for p, r in ratios.items()}
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    got: Dict[str, int] = {p: 0 for p in pillars}

    def live_sources(p: str) -> List[Sequence[Any]]:
        out = []
        for spec in sources_by_pillar.get(p, []):
            st = manifest["sources"].get(_src_key(spec), {})
            if not st.get("done") and not st.get("dead"):
                out.append(spec)
        return out

    def take(spec: Sequence[Any], pillar: str, need: int) -> int:
        key = _src_key(spec)
        name, subset, split = spec[0].strip(), spec[1], spec[2]
        base_shard = spec[3] if len(spec) > 3 else None
        st = manifest["sources"].setdefault(key, {"shard": 0, "offset": 0, "done": False, "dead": False, "errors": 0,
                                                  "strikes": 0, "accepted": 0})
        n_acc = 0
        t0 = time.time()
        time_cap = max(90.0, need * 1.0)
        while n_acc < need and not st["done"] and not st["dead"]:
            shard = next_shard_path(base_shard, st["shard"]) if base_shard else None
            if base_shard and shard is None:
                st["done"] = True
                break
            if not base_shard and st["shard"] > 0:
                st["done"] = True
                break
            try:
                ds, _ = open_fn(name, subset, split, shard_path=shard)
                it = iter(ds)
                if st["offset"] > 0:
                    it = itertools.islice(it, st["offset"], None)
            except Exception as exc:
                st["errors"] += 1
                log(f"     [Skip] {name}: {str(exc).splitlines()[0][:100]}")
                if st["errors"] >= 2:
                    st["dead"] = True
                break
            scanned = 0
            acc_here = 0
            exhausted_natural = True
            try:
                for row in it:
                    st["offset"] += 1
                    scanned += 1
                    if scanned > need * 12 + 400 and acc_here == 0:
                        st["strikes"] += 1
                        if st["strikes"] >= 2:
                            st["dead"] = True
                        exhausted_natural = False
                        break
                    if time.time() - t0 > time_cap and acc_here >= 5:
                        exhausted_natural = False
                        break
                    rec = extract_fn(row, pillar)
                    if not rec or not filter_fn(rec):
                        continue
                    k = doc_key(rec)
                    if k in seen:
                        continue
                    seen.add(k)
                    rec["source"] = name
                    rec["pillar"] = pillar
                    if is_val_doc(k, val_mod):
                        val.append(rec)
                        continue
                    train.append(rec)
                    n_acc += 1
                    acc_here += 1
                    st["accepted"] += 1
                    if n_acc >= need:
                        exhausted_natural = False
                        break
            except Exception as exc:  # streaming hiccup: keep what we have, retry next chunk
                st["errors"] += 1
                log(f"     [Stream Error] {name}: {str(exc).splitlines()[0][:100]}")
                if st["errors"] >= 3:
                    st["dead"] = True
                exhausted_natural = False
            if exhausted_natural:
                st["shard"] += 1
                st["offset"] = 0
            else:
                break
        got[pillar] = got.get(pillar, 0) + n_acc
        return n_acc

    def run_round(total: int) -> int:
        capacity = {p: (total if live_sources(p) else 0) for p in pillars}
        quotas = redistribute_quotas(total, pillars, capacity)
        added = 0
        for p, q in quotas.items():
            if q <= 0:
                continue
            srcs = live_sources(p)
            if not srcs:
                continue
            per = int(math.ceil(q / len(srcs)))
            left = q
            for spec in srcs:
                if left <= 0:
                    break
                n = take(spec, p, min(per, left))
                left -= n
                added += n
            if left > 0:  # second pass over still-live sources of the same pillar
                for spec in live_sources(p):
                    if left <= 0:
                        break
                    n = take(spec, p, left)
                    left -= n
                    added += n
        return added

    remaining = int(n_docs)
    for _round in range(4):
        if remaining <= 0:
            break
        added = run_round(remaining)
        remaining = int(n_docs) - len(train)
        if added == 0:
            break

    random.Random(int(seed) + int(chunk_idx) * 7919).shuffle(train)
    stats = {
        "chunk": int(chunk_idx), "train": len(train), "val": len(val), "requested": int(n_docs),
        "by_pillar": {p: sum(1 for r in train if r.get("pillar") == p) for p in pillars},
        "live_sources": {p: len(live_sources(p)) for p in pillars},
    }
    if len(train) < 0.2 * int(n_docs):
        manifest["exhausted"] = True
    for p, c in stats["by_pillar"].items():
        manifest["pillar_counts"][p] = manifest["pillar_counts"].get(p, 0) + c
    manifest["val_docs"] = manifest.get("val_docs", 0) + len(val)
    return train, val, stats


# ======================================================================================
# Hub persistence (never raises)
# ======================================================================================
class HubSync:
    """Thin crash-proof wrapper around huggingface_hub for the private long-run repo."""

    def __init__(self, repo_id: Optional[str], token: Optional[str], private: bool = True):
        self.repo_id, self.token, self.ok, self.api = repo_id, token, False, None
        if not (repo_id and token):
            return
        try:
            from huggingface_hub import HfApi
            self.api = HfApi(token=token)
            self.api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
            self.ok = True
        except Exception as exc:
            print(f"[HubSync] disabled ({type(exc).__name__}: {str(exc)[:120]})", flush=True)

    def download(self, path_in_repo: str, local_dir: Path) -> Optional[Path]:
        if not self.ok:
            return None
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(repo_id=self.repo_id, filename=path_in_repo, token=self.token,
                                        local_dir=str(local_dir)))
        except Exception:
            return None

    def list_files(self, prefix: str) -> List[str]:
        if not self.ok:
            return []
        try:
            return [f for f in self.api.list_repo_files(self.repo_id) if f.startswith(prefix)]
        except Exception:
            return []

    def upload_file(self, local: Path, path_in_repo: str, msg: str) -> bool:
        if not self.ok:
            return False
        try:
            self.api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=self.repo_id,
                                 commit_message=msg)
            return True
        except Exception as exc:
            if "403" in str(exc) or "storage" in str(exc).lower():
                self.squash()
                try:
                    self.api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=self.repo_id,
                                         commit_message=msg)
                    return True
                except Exception:
                    pass
            print(f"[HubSync] upload_file failed {path_in_repo}: {str(exc)[:120]}", flush=True)
            return False

    def upload_folder(self, folder: Path, path_in_repo: str, msg: str) -> bool:
        if not self.ok:
            return False
        try:
            self.api.upload_folder(folder_path=str(folder), path_in_repo=path_in_repo, repo_id=self.repo_id,
                                   commit_message=msg)
            return True
        except Exception as exc:
            if "403" in str(exc) or "storage" in str(exc).lower():
                self.squash()
                try:
                    self.api.upload_folder(folder_path=str(folder), path_in_repo=path_in_repo, repo_id=self.repo_id,
                                           commit_message=msg)
                    return True
                except Exception:
                    pass
            print(f"[HubSync] upload_folder failed {path_in_repo}: {str(exc)[:120]}", flush=True)
            return False

    def squash(self) -> None:
        """Squash history so superseded multi-GB checkpoint blobs can be garbage-collected (best effort)."""
        if not self.ok:
            return
        try:
            self.api.super_squash_history(repo_id=self.repo_id)
        except Exception as exc:
            print(f"[HubSync] squash skipped: {str(exc)[:100]}", flush=True)


# ======================================================================================
# Chunk provider: background curation + persistence
# ======================================================================================
class ChunkProvider:
    """Serves train chunks in order; a worker thread curates ahead of training and persists chunks/manifest to the Hub."""

    def __init__(
        self,
        data_dir: Path,
        curate_fn: Callable[[int, Dict[str, Any], Set[str], int], Tuple[List[Dict], List[Dict], Dict]],
        n_chunks_target: int,
        chunk_docs: int = 2000,
        hub: Optional[HubSync] = None,
        lookahead: int = 2,
    ):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.curate_fn, self.chunk_docs, self.hub, self.lookahead = curate_fn, int(chunk_docs), hub, int(lookahead)
        self.n_chunks_target = int(n_chunks_target)
        self.cond = threading.Condition()
        self.consumed_idx = 0
        self.stop = False
        self.error: Optional[str] = None
        self.manifest = new_manifest()
        self.seen: Set[str] = set()
        self._load_remote_state()
        self.thread = threading.Thread(target=self._worker, daemon=True)

    # ---- persistence
    def _load_remote_state(self) -> None:
        mp = self.dir / "manifest.json"
        sp = self.dir / "seen.txt"
        if self.hub is not None and self.hub.ok:
            m = self.hub.download("data/manifest.json", self.dir.parent) or None
            s = self.hub.download("data/seen.txt", self.dir.parent) or None
            if m and Path(m) != mp and Path(m).exists():
                mp.write_bytes(Path(m).read_bytes())
            if s and Path(s) != sp and Path(s).exists():
                sp.write_bytes(Path(s).read_bytes())
        if mp.exists():
            try:
                self.manifest = json.loads(mp.read_text())
            except Exception:
                self.manifest = new_manifest()
        if sp.exists():
            self.seen = set(x for x in sp.read_text().split() if x)

    def _persist(self, files: Sequence[Path]) -> bool:
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=1))
        (self.dir / "seen.txt").write_text("\n".join(sorted(self.seen)))
        if self.hub is None or not self.hub.ok:
            return True
        ok = True
        for f in list(files) + [self.dir / "seen.txt", self.dir / "manifest.json"]:  # manifest LAST
            ok = self.hub.upload_file(f, f"data/{f.name}", f"[longrun] {f.name}") and ok
        return ok

    # ---- worker
    def _worker(self) -> None:
        retries = 0
        while not self.stop:
            with self.cond:
                idx = int(self.manifest["n_chunks"])
                done = self.manifest.get("exhausted") or idx >= self.n_chunks_target
                too_far = idx >= self.consumed_idx + self.lookahead
                if done or too_far:
                    self.cond.wait(timeout=5.0)
                    continue
            try:
                t0 = time.time()
                train, val, stats = self.curate_fn(self.chunk_docs, self.manifest, self.seen, idx)
                tp = self.dir / f"train_{idx:05d}.jsonl.gz"
                vp = self.dir / f"val_{idx:05d}.jsonl.gz"
                files = []
                if train:
                    write_jsonl_gz(tp, train)
                    files.append(tp)
                write_jsonl_gz(vp, val)
                files.append(vp)
                with self.cond:
                    if train:
                        self.manifest["n_chunks"] = idx + 1
                    ok = self._persist(files)
                    self.cond.notify_all()
                print(f"[Curation] chunk {idx}: +{stats['train']} train / +{stats['val']} val in {time.time() - t0:.0f}s "
                      f"| by_pillar={stats['by_pillar']} | live_sources={stats['live_sources']} | hub_ok={ok}", flush=True)
                retries = 0
            except Exception as exc:
                retries += 1
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[Curation Error] {self.error} (retry {retries}/3)", flush=True)
                if retries >= 3:
                    with self.cond:
                        self.manifest["exhausted"] = True
                        self.cond.notify_all()
                time.sleep(20.0)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        with self.cond:
            self.stop = True
            self.cond.notify_all()

    # ---- consumer API
    def n_available(self) -> int:
        return int(self.manifest["n_chunks"])

    def _read_chunk(self, name: str) -> Optional[List[Dict[str, Any]]]:
        p = self.dir / name
        if not p.exists() and self.hub is not None:
            got = self.hub.download(f"data/{name}", self.dir.parent)
            if got is None:
                return None
        return read_jsonl_gz(p) if p.exists() else None

    def get_train(self, idx: int, timeout: float = 3600.0) -> Tuple[List[Dict[str, Any]], int, bool]:
        """Return `(records, effective_idx, wrapped)`. Blocks until the chunk exists; wraps (2nd epoch) once data is exhausted."""
        t0 = time.time()
        with self.cond:
            self.consumed_idx = max(self.consumed_idx, int(idx))
            needed_target = int(idx) + max(1, int(self.lookahead))
            if not self.manifest.get("exhausted") and needed_target > self.n_chunks_target:
                self.n_chunks_target = needed_target
            self.cond.notify_all()
            while True:
                n = self.n_available()
                if idx < n:
                    break
                if self.manifest.get("exhausted") and n > 0:
                    break
                if not self.manifest.get("exhausted") and needed_target > self.n_chunks_target:
                    self.n_chunks_target = needed_target
                self.cond.notify_all()
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"chunk {idx} not available after {timeout:.0f}s (last curation error: {self.error})")
                self.cond.wait(timeout=15.0)
            n = self.n_available()
        eff = idx if idx < n else idx % n
        recs = self._read_chunk(f"train_{eff:05d}.jsonl.gz")
        if recs is None:
            raise RuntimeError(f"chunk file train_{eff:05d}.jsonl.gz missing locally and on the Hub")
        return recs, eff, eff != idx

    def frozen_val(self, n_docs: int, min_chunks_wait: int = 1) -> List[Dict[str, Any]]:
        """Fixed validation docs (balanced over pillars, deterministic by key). Frozen once to `val_frozen.jsonl.gz`."""
        fp = self.dir / "val_frozen.jsonl.gz"
        if not fp.exists() and self.hub is not None:
            self.hub.download("data/val_frozen.jsonl.gz", self.dir.parent)
        if fp.exists():
            return read_jsonl_gz(fp)
        self.get_train(0)  # wait for chunk 0 (val_00000 is written together with it)
        pool: List[Dict[str, Any]] = []
        for i in range(max(1, min(self.n_available(), 3))):
            rows = self._read_chunk(f"val_{i:05d}.jsonl.gz") or []
            pool.extend(rows)
            if len(pool) >= n_docs:
                break
        by_p: Dict[str, List[Dict[str, Any]]] = {}
        for r in sorted(pool, key=doc_key):
            by_p.setdefault(r.get("pillar", "?"), []).append(r)
        chosen: List[Dict[str, Any]] = []
        while len(chosen) < n_docs and any(by_p.values()):
            for p in sorted(by_p):
                if by_p[p] and len(chosen) < n_docs:
                    chosen.append(by_p[p].pop(0))
        write_jsonl_gz(fp, chosen)
        if self.hub is not None and self.hub.ok:
            self.hub.upload_file(fp, "data/val_frozen.jsonl.gz", "[longrun] frozen validation set")
        return chosen


# ======================================================================================
# Training-side helpers (torch; imported lazily so the module imports without torch)
# ======================================================================================
def _clone_cpu_state(obj: Any) -> Any:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _clone_cpu_state(v) for k, v in obj.items()}
    return obj


class SeqStream:
    """Deterministic stream of packed sequences chunk-by-chunk; resumable at `(chunk_idx, seq_in_chunk)`."""

    def __init__(self, provider: ChunkProvider, tokenizer: Any, cfg: Dict[str, Any], seq_len: int, state: LongRunState):
        self.provider, self.tok, self.cfg, self.seq_len, self.state = provider, tokenizer, cfg, int(seq_len), state
        self.seqs: List[Dict[str, Any]] = []
        self.loaded_idx: Optional[int] = None

    def _load(self, idx: int) -> None:
        import sparkx_loop_v03_train_env as te
        recs, eff, wrapped = self.provider.get_train(idx)
        if wrapped:
            self.state.data_epoch = max(self.state.data_epoch, idx // max(1, self.provider.n_available()))
            print(f"[Data Exhausted] source pool smaller than plan -> re-using chunk {eff} (data epoch {self.state.data_epoch + 1})",
                  flush=True)
        local_cfg = {"MAX_SEQ_LENGTH": self.seq_len, "STAGES": [{"seq_len": self.seq_len}]}
        pool = te.build_16k_stage_pools(recs, self.tok, local_cfg)[self.seq_len]
        random.Random(int(self.cfg.get("SEED", 42)) + idx * 104729).shuffle(pool)
        self.seqs, self.loaded_idx = pool, idx

    def next_seq(self) -> Dict[str, Any]:
        st = self.state
        if self.loaded_idx != st.chunk_idx:
            self._load(st.chunk_idx)
        while st.seq_in_chunk >= len(self.seqs):
            st.chunk_idx += 1
            st.seq_in_chunk = 0
            self._load(st.chunk_idx)
        s = self.seqs[st.seq_in_chunk]
        st.seq_in_chunk += 1
        return s


def build_val_samples(records: Sequence[Dict[str, Any]], tokenizer: Any, max_tokens: int, max_seq_len: int = 16384,
                      min_supervised: int = 128) -> List[Dict[str, Any]]:
    """Tokenise validation docs in the training format; truncate to `max_tokens`; keep docs with enough supervised tokens."""
    import torch
    import sparkx_loop_v03_train_env as te
    out = []
    for r in records:
        res = te.tokenize_and_mask_record_16k(r, tokenizer, max_seq_len)
        if res is None:
            continue
        ids, lbl = res[0][:max_tokens], res[1][:max_tokens]
        if sum(1 for y in lbl[1:] if y != -100) < min_supervised:
            continue
        out.append({"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(lbl, dtype=torch.long),
                    "pillar": r.get("pillar", "?")})
    return out


def eval_val(model: Any, samples: Sequence[Dict[str, Any]], restore_loop: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """Held-out CE with Engram ON/OFF, overall and per pillar. Crash-safe loop/train-mode restoration."""
    import torch
    import sparkx_engram_generalization as eg

    prev_ckpt = bool(getattr(model, "enable_checkpoint", False))
    model.eval()
    model.enable_checkpoint = False
    model.settings.checkpoint_layers = False
    eg._release_eval_caches(model)
    acc = {"on": {}, "off": {}}
    try:
        for tag, use_eng in (("off", False), ("on", True)):
            kw = dict(restore_loop)
            kw["use_engram"] = use_eng
            model.set_loop(**kw)
            for s in samples:
                tot, n = eg._val_ce(model, [s], max_tokens)
                a = acc[tag].setdefault(s["pillar"], [0.0, 0])
                a[0] += tot
                a[1] += n
    finally:
        model.set_loop(**restore_loop)
        eg._release_eval_caches(model)
        model.train()
        model.enable_checkpoint = prev_ckpt
        model.settings.checkpoint_layers = prev_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def agg(tag: str) -> float:
        return sum(v[0] for v in acc[tag].values()) / max(1, sum(v[1] for v in acc[tag].values()))

    by_p = {p: {"on": acc["on"][p][0] / max(1, acc["on"][p][1]), "off": acc["off"][p][0] / max(1, acc["off"][p][1]),
                "tokens": acc["on"][p][1]} for p in acc["on"]}
    rep = {"ce_on": agg("on"), "ce_off": agg("off"), "n_docs": len(samples),
           "n_tokens": sum(v[1] for v in acc["on"].values()), "by_pillar": by_p}
    rep["engram_delta_ce"] = rep["ce_on"] - rep["ce_off"]
    return rep


def decide_after_eval(
    ce_on: float, best: Optional[float], no_improve: int, reductions: int,
    min_delta: float = 0.002, patience: int = 6, max_reductions: int = 2,
) -> Tuple[str, Optional[float], int, int]:
    """Pure plateau policy -> `(action, best, no_improve, reductions)`; action in {'improved','continue','rollback','stop'}."""
    if best is None or ce_on < best - min_delta:
        return "improved", ce_on, 0, reductions
    no_improve += 1
    if no_improve >= patience:
        if reductions + 1 > max_reductions:
            return "stop", best, no_improve, reductions
        return "rollback", best, 0, reductions + 1
    return "continue", best, no_improve, reductions


def make_default_curator(cfg: Dict[str, Any]) -> Callable[..., Any]:
    import sparkx_loop_v03_train_env as te

    def curate(n_docs: int, manifest: Dict[str, Any], seen: Set[str], chunk_idx: int):
        return curate_chunk(
            n_docs, cfg["PILLAR_RATIOS"], te.DATASET_SOURCES, manifest, seen,
            te._open_streaming_dataset, te.extract_record_fields, te.passes_quality_filter,
            val_mod=int(cfg.get("VAL_HASH_MOD", 25)), seed=int(cfg.get("SEED", 42)), chunk_idx=chunk_idx,
        )
    return curate


def _save_ckpt(model: Any, state: LongRunState, out_dir: Path) -> Path:
    """Write loop/engram state + state.json to `out_dir` (atomic-ish: state.json last)."""
    import torch
    import sparkx_loop_v03_train_env as te
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(te._core_mod.loop_v03_state_dict(model, include_tier1b=True), str(out_dir / "loop_v03_state.pt"))
    (out_dir / "state.json").write_text(state.to_json())
    return out_dir


def _load_ckpt(model: Any, ckpt_dir: Path) -> bool:
    import torch
    import sparkx_loop_v03_train_env as te
    p = ckpt_dir / "loop_v03_state.pt"
    if not p.exists():
        return False
    obj = torch.load(str(p), map_location="cpu")
    te._core_mod.load_loop_v03_state_dict(model, obj, strict=False)
    del obj
    return True


def run_long_training(model: Any, tokenizer: Any, cfg: Dict[str, Any], hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Run (or resume) the long run. One call = one Colab session segment of at most `LONGRUN_SEGMENT_HOURS`.

    Returns a summary dict; `summary['finished']` is True once `total_steps` are done (best weights restored in that case).
    """
    import gc
    import torch
    import sparkx_loop_v03_train_env as te
    import sparkx_engram_generalization as eg

    core = te._core_mod
    t_session = time.time()
    work = Path(cfg.get("LONGRUN_DIR", "/content/longrun"))
    data_dir, ck_latest, ck_best = work / "data", work / "ckpt" / "latest", work / "ckpt" / "best"
    work.mkdir(parents=True, exist_ok=True)
    repo = cfg.get("LONGRUN_REPO_ID") or (f"{cfg['HF_REPO_ID']}-longrun" if cfg.get("HF_REPO_ID") else None)
    hub = HubSync(repo, hf_token, private=True)

    seq_len = int(min(int(cfg.get("MAX_SEQ_LENGTH", 16384)), 16384))
    grad_accum = int(cfg.get("GRAD_ACCUM", 8))
    chunk_docs = int(cfg.get("LONGRUN_CHUNK_DOCS", 2000))
    eval_every = int(cfg.get("LONGRUN_EVAL_EVERY", 40))
    audit_every_n_evals = int(cfg.get("LONGRUN_AUDIT_EVERY_N_EVALS", 4))
    save_every = int(cfg.get("LONGRUN_SAVE_EVERY", 40))
    push_every = int(cfg.get("LONGRUN_PUSH_EVERY", 80))
    seg_hours = float(cfg.get("LONGRUN_SEGMENT_HOURS", 22.0))
    end_margin_s = float(cfg.get("LONGRUN_END_MARGIN_MIN", 25.0)) * 60.0
    val_docs_n = int(cfg.get("VAL_EVAL_DOCS", 64))
    val_max_tokens = int(cfg.get("VAL_MAX_TOKENS", 3072))
    log_every = int(cfg.get("LOG_EVERY_STEPS", 5))

    # ---- state (resume from the newer of local / Hub)
    state = LongRunState()
    sp_local = ck_latest / "state.json"
    cand: List[Tuple[int, str]] = []
    if sp_local.exists():
        cand.append((LongRunState.from_json(sp_local.read_text()).global_step, "local"))
    hub_state = hub.download("ckpt/latest/state.json", work) if hub.ok else None
    if hub_state is not None and Path(hub_state).exists():
        cand.append((LongRunState.from_json(Path(hub_state).read_text()).global_step, "hub"))
    resumed = False
    if cand:
        step_best, where = max(cand)
        if where == "hub":
            for fn in ("ckpt/latest/loop_v03_state.pt", "ckpt/latest/state.json"):
                hub.download(fn, work)
        if _load_ckpt(model, ck_latest):
            state = LongRunState.from_json((ck_latest / "state.json").read_text())
            resumed = True
            print(f"[Resume] restored `latest` from {where} @ global_step={state.global_step} "
                  f"(chunk {state.chunk_idx}, seq {state.seq_in_chunk}, tokens_seen={state.tokens_seen:,})", flush=True)
    if state.finished:
        print("[Long Run] state says FINISHED already; nothing to do.", flush=True)
        return {"finished": True, "state": dataclasses.asdict(state)}
    if not state.total_steps:
        state.plan_hours = float(cfg.get("LONGRUN_TOTAL_HOURS", 20.0))
        state.plan = plan_run(state.plan_hours, tok_per_s=float(cfg.get("LONGRUN_ASSUMED_TOK_S", 1450.0)),
                              seq_len=seq_len, grad_accum=grad_accum)
        state.total_steps = int(cfg.get("LONGRUN_TOTAL_STEPS") or state.plan["steps"])
    state.segments += 1
    print("=" * 108)
    print(f"[Long Run Plan] total_steps={state.total_steps} (~{state.plan.get('real_tokens', 0) / 1e6:.0f}M real tokens, "
          f"~{state.plan.get('docs_needed', 0):,} docs) | seq_len={seq_len} x accum={grad_accum} | segment #{state.segments} "
          f"limit {seg_hours:.1f}h | repo={repo if hub.ok else 'LOCAL ONLY (no HF token/repo!)'}")
    print("=" * 108, flush=True)

    # ---- data
    lookahead_n = int(cfg.get("LONGRUN_LOOKAHEAD", 2))
    n_chunks_target = max(
        int(math.ceil(state.plan.get("docs_needed", chunk_docs * 20) / chunk_docs)),
        int(state.chunk_idx) + max(1, lookahead_n),
    )
    provider = ChunkProvider(data_dir, make_default_curator(cfg), n_chunks_target, chunk_docs, hub, lookahead=lookahead_n)
    provider.consumed_idx = state.chunk_idx
    provider.start()
    val_recs = provider.frozen_val(val_docs_n)
    val_samples = build_val_samples(val_recs, tokenizer, val_max_tokens)
    print(f"[Validation] frozen held-out set: {len(val_samples)} docs "
          f"({ {p: sum(1 for s in val_samples if s['pillar'] == p) for p in sorted({s['pillar'] for s in val_samples})} })", flush=True)
    stream = SeqStream(provider, tokenizer, cfg, seq_len, state)

    # ---- model prep (mirror of run_v03_four_stage_training's attribute setup; no class re-binding needed in a fresh session)
    try:
        te._reset_autograd_hooks_and_vram(model)
    except Exception:
        pass
    for eng_mod in model.all_engram_modules():
        eng_mod.load_ema_decay = float(cfg.get("ENG_LOAD_EMA_DECAY", 0.05))
        eng_mod.balance_step_size = float(cfg.get("ENG_BALANCE_STEP_SIZE", 0.0015))
        eng_mod.use_pass_gain_schedule = bool(cfg.get("ENG_USE_PASS_GAIN_SCHEDULE", True))
        eng_mod.multi_horizon_recall = bool(cfg.get("ENG_MULTI_HORIZON_RECALL", True))
        eng_mod.flight_top_k = None
        eng_mod._pending_expert_load = None
        eng_mod.last_router_z_loss = None
        eng_mod._active_z_losses = None
    loop_kw = dict(start=4, end=28, passes=2, use_mhc=True, use_engram=True, enable_p3=True, p3_start=12, p3_end=24)
    model.set_loop(**loop_kw)
    model.enable_checkpoint = True
    model.settings.checkpoint_layers = True
    device = next(model.base.parameters()).device
    vram_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0.0
    if seq_len >= 16384 and 0.0 < vram_total <= 85.0:
        model.offload_checkpoint_to_cpu = True
    core.set_trainables_v03(model, full_finetune=False, freeze_backbone=True, freeze_engram=False)
    lr_scale_base = float(cfg.get("LONGRUN_LR_SCALE", 0.5))
    mhc_mult = float(cfg.get("LONGRUN_MHC_LR_MULT", 0.35))
    warmup_steps = int(cfg.get("LONGRUN_WARMUP_STEPS", 20))
    rewarm = int(cfg.get("LONGRUN_REWARM_STEPS", 10))
    run_start_step = state.global_step

    def build_opt():
        opt = te._build_stage_optimizer(model, cfg, lr_scale=lr_scale_base * state.plateau_scale,
                                        backbone_lr_mult=0.0, mhc_lr_mult=mhc_mult)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: lr_multiplier(run_start_step + s, state.total_steps, warmup_steps,
                                                   float(cfg.get("LONGRUN_MIN_LR_RATIO", 0.15)), s, rewarm))
        return opt, sched

    # ---- baseline + initial best
    if state.baseline is None:
        t0 = time.time()
        state.baseline = eval_val(model, val_samples, loop_kw, val_max_tokens)
        state.best_val_ce_on = state.baseline["ce_on"]
        state.best_step = state.global_step
        print(f"[Baseline @ step {state.global_step}] held-out CE on={state.baseline['ce_on']:.4f} off={state.baseline['ce_off']:.4f} "
              f"delta={state.baseline['engram_delta_ce']:+.4f} ({state.baseline['n_tokens']:,} tok, {time.time() - t0:.0f}s)", flush=True)
        try:
            _save_ckpt(model, state, ck_best)
            if hub.ok:
                hub.upload_folder(ck_best, "ckpt/best", f"[longrun] baseline best step={state.global_step}")
        except Exception as exc:
            print(f"[Best Save Notice] {exc}", flush=True)
    elif not (ck_best / "loop_v03_state.pt").exists() and hub.ok:
        hub.download("ckpt/best/loop_v03_state.pt", work)
    best_available = (ck_best / "loop_v03_state.pt").exists()

    optimizer, scheduler = build_opt()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    deadline = t_session + seg_hours * 3600.0 - end_margin_s
    step_times: List[float] = []
    eval_count = len(state.val_history)
    paused = False
    stopped_by_plateau = False
    last_push_step = state.global_step

    def push_latest(tag: str) -> None:
        _save_ckpt(model, state, ck_latest)
        if hub.ok:
            hub.upload_folder(ck_latest, "ckpt/latest", f"[longrun] {tag} step={state.global_step}")
            if cfg.get("LONGRUN_SQUASH", True):
                hub.squash()

    try:
        while state.global_step < state.total_steps:
            if time.time() >= deadline:
                paused = True
                print(f"[Segment Guard] {seg_hours:.1f}h segment budget reached at step {state.global_step}; saving & pausing.", flush=True)
                break
            t_step = time.time()
            accum_ce = accum_loss = accum_z = 0.0
            step_tokens = 0
            for _ in range(grad_accum):
                s = stream.next_seq()
                ids = s["input_ids"].unsqueeze(0).to(device, non_blocking=True)
                lbl = s["labels"].unsqueeze(0).to(device, non_blocking=True)
                step_tokens += int(ids.numel())
                out = model(input_ids=ids, labels=lbl, return_logits=False)
                loss = out["loss"] / float(grad_accum)
                if not torch.isfinite(loss):
                    print(f"[Warn] non-finite loss at step {state.global_step + 1}; skipping micro-step.", flush=True)
                    continue
                loss.backward()
                accum_loss += float(out["loss"].detach().item()) / grad_accum
                accum_ce += float(out["ce_loss"].detach().item()) / grad_accum
                zv = out["z_loss"]
                accum_z += (float(zv.detach().item()) if isinstance(zv, torch.Tensor) else float(zv)) / grad_accum
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["MAX_GRAD_NORM"])).item())
            init_lr0 = float(optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]))
            lr_mult_now = float(scheduler.get_last_lr()[0]) / max(1e-12, init_lr0)
            if math.isfinite(gnorm):
                optimizer.step()
                model.step_all_tier1b_sparse_adam(lr=float(cfg["LR_TIER1B_SPARSE"]) * lr_scale_base * state.plateau_scale * lr_mult_now)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            state.global_step += 1
            state.tokens_seen += step_tokens
            dt = max(1e-3, time.time() - t_step)
            step_times.append(dt)
            if state.global_step % log_every == 0 or state.global_step <= run_start_step + 3:
                el = (time.time() - t_session) / 3600.0
                print(f"[LongRun {state.global_step:5d}/{state.total_steps}] loss={accum_loss:.4f} ce={accum_ce:.4f} z={accum_z:.5f} "
                      f"gnorm={gnorm:.3f} lr_mult={lr_mult_now:.3f} {seq_len * grad_accum / dt:6.0f} tok/s ({dt:4.1f}s/step) "
                      f"chunk={state.chunk_idx}:{state.seq_in_chunk} tokens={state.tokens_seen / 1e6:.1f}M seg_elapsed={el:.2f}h", flush=True)

            # one-time throughput calibration of the plan (first session only; the LR schedule is still in warmup/early cosine)
            if (not state.calibrated and not resumed and len(step_times) == 12
                    and bool(cfg.get("LONGRUN_AUTO_CALIBRATE", True)) and not cfg.get("LONGRUN_TOTAL_STEPS")):
                med = sorted(step_times[2:])[len(step_times[2:]) // 2]
                new_plan = plan_run(state.plan_hours, tok_per_s=seq_len * grad_accum / med, seq_len=seq_len, grad_accum=grad_accum)
                state.total_steps = max(state.global_step + 1, new_plan["steps"])
                state.plan = new_plan
                provider.n_chunks_target = int(math.ceil(new_plan["docs_needed"] / chunk_docs))
                state.calibrated = True
                print(f"[Calibration] median {med:.1f}s/step -> total_steps={state.total_steps}, docs_needed={new_plan['docs_needed']:,} "
                      f"(chunks target {provider.n_chunks_target})", flush=True)

            # validation / best / plateau policy
            due_eval = (state.global_step % eval_every == 0) or state.global_step >= state.total_steps
            if due_eval:
                eval_count += 1
                try:
                    rep = eval_val(model, val_samples, loop_kw, val_max_tokens)
                except Exception as exc:
                    print(f"[Val Notice] skipped ({type(exc).__name__}: {exc})", flush=True)
                    rep = None
                if rep is not None:
                    if audit_every_n_evals > 0 and eval_count % audit_every_n_evals == 0:
                        try:
                            aud = eg.stage_validation_report(model, val_samples, loop_kw, max_tokens=val_max_tokens, anchor_audit=True)
                            model.train()
                            model.enable_checkpoint = True
                            model.settings.checkpoint_layers = True
                            rep["removal_delta_ce"] = aud.get("removal_delta_ce")
                            rep["harmful_anchors"] = aud.get("harmful_anchors")
                        except Exception as exc:
                            print(f"[Audit Notice] skipped ({exc})", flush=True)
                    action, state.best_val_ce_on, state.no_improve, state.reductions = decide_after_eval(
                        rep["ce_on"], state.best_val_ce_on, state.no_improve, state.reductions,
                        patience=int(cfg.get("LONGRUN_PATIENCE_EVALS", 6)), max_reductions=int(cfg.get("LONGRUN_MAX_LR_REDUCTIONS", 2)))
                    rep.update({"step": state.global_step, "tokens": state.tokens_seen, "action": action,
                                "train_ce": accum_ce})
                    state.val_history.append(rep)
                    pill = " ".join(f"{p[:6]}:{v['on']:.3f}/{v['off']:.3f}" for p, v in sorted(rep["by_pillar"].items()))
                    print(f"[Val @ {state.global_step}] CE on={rep['ce_on']:.4f} off={rep['ce_off']:.4f} delta={rep['engram_delta_ce']:+.4f} "
                          f"| vs baseline on {rep['ce_on'] - state.baseline['ce_on']:+.4f} | best={state.best_val_ce_on:.4f}@{state.best_step} "
                          f"| action={action} | {pill}", flush=True)
                    if rep.get("removal_delta_ce"):
                        print("   anchor removal delta (neg = harmful): " + ", ".join(f"L{l}:{d:+.4f}" for l, d in sorted(rep["removal_delta_ce"].items())), flush=True)
                    if action == "improved":
                        state.best_step = state.global_step
                        try:
                            _save_ckpt(model, state, ck_best)
                            best_available = True
                        except Exception as exc:
                            print(f"[Best Save Notice] {exc}", flush=True)
                    elif action == "rollback":
                        if best_available:
                            print(f"[Plateau] {cfg.get('LONGRUN_PATIENCE_EVALS', 6)} evals without improvement -> rollback to best@{state.best_step}, "
                                  f"LR x0.5 (reduction {state.reductions})", flush=True)
                            _load_ckpt(model, ck_best)
                            state.plateau_scale *= 0.5
                            del optimizer, scheduler
                            gc.collect()
                            run_start_step = state.global_step
                            optimizer, scheduler = build_opt()
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            print("[Plateau] rollback requested but no best checkpoint on disk; continuing.", flush=True)
                    elif action == "stop":
                        print("[Plateau] LR reductions exhausted without improvement -> stopping early; best weights will be restored.", flush=True)
                        stopped_by_plateau = True
                        break

            if state.global_step % save_every == 0:
                try:
                    _save_ckpt(model, state, ck_latest)
                except Exception as exc:
                    print(f"[Save Notice] {exc}", flush=True)
            if push_every > 0 and state.global_step - last_push_step >= push_every:
                try:
                    push_latest("periodic")
                    if hub.ok and best_available:
                        hub.upload_folder(ck_best, "ckpt/best", f"[longrun] best step={state.best_step}")
                except Exception as exc:
                    print(f"[Push Notice] {exc}", flush=True)
                last_push_step = state.global_step
    except BaseException as exc:
        print(f"\n[Long Run] {type(exc).__name__} at step {state.global_step}; saving latest before re-raising...", flush=True)
        try:
            optimizer = scheduler = None
            te._reset_autograd_hooks_and_vram(model)
            _save_ckpt(model, state, ck_latest)
            if hub.ok and not isinstance(exc, KeyboardInterrupt):
                hub.upload_folder(ck_latest, "ckpt/latest", f"[longrun] emergency step={state.global_step}")
        except Exception as exc2:
            print(f"[Emergency Save Notice] {exc2}", flush=True)
        provider.close()
        raise

    finished = state.global_step >= state.total_steps or stopped_by_plateau
    summary: Dict[str, Any] = {"finished": finished, "paused": paused, "global_step": state.global_step,
                               "tokens_seen": state.tokens_seen, "best_val_ce_on": state.best_val_ce_on,
                               "best_step": state.best_step, "baseline": state.baseline}
    if finished:
        # final eval already ran at the last step (due_eval); restore best if it beats the final weights
        final_ce = state.val_history[-1]["ce_on"] if state.val_history else None
        if best_available and final_ce is not None and state.best_val_ce_on is not None and final_ce > state.best_val_ce_on + 1e-6:
            print(f"[Finish] restoring best weights (val CE on {state.best_val_ce_on:.4f}@{state.best_step}) over final ({final_ce:.4f}).", flush=True)
            _load_ckpt(model, ck_best)
        state.finished = True
        try:
            _save_ckpt(model, state, work / "ckpt" / "final")
            if hub.ok:
                hub.upload_folder(work / "ckpt" / "final", "ckpt/final", f"[longrun] FINAL step={state.global_step}")
                hub.upload_folder(ck_best, "ckpt/best", f"[longrun] best step={state.best_step}") if best_available else None
                (work / "state.json").write_text(state.to_json())
                hub.upload_file(work / "state.json", "ckpt/latest/state.json", "[longrun] finished flag")
        except Exception as exc:
            print(f"[Final Save Notice] {exc}", flush=True)
    else:
        try:
            push_latest("segment-end")
            if hub.ok and best_available:
                hub.upload_folder(ck_best, "ckpt/best", f"[longrun] best step={state.best_step}")
        except Exception as exc:
            print(f"[Segment-End Save Notice] {exc}", flush=True)
    provider.close()
    summary["state"] = dataclasses.asdict(state)
    print("=" * 108)
    print(f"[Long Run] {'FINISHED' if finished else 'PAUSED (re-run the notebook in a new session to resume)'} at step "
          f"{state.global_step}/{state.total_steps} | tokens={state.tokens_seen / 1e6:.1f}M | best val CE on={state.best_val_ce_on}"
          f"@{state.best_step} | baseline on={state.baseline['ce_on'] if state.baseline else None}")
    print("=" * 108, flush=True)
    return summary
