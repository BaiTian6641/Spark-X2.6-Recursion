# Spark X2.6 4B Loop72 LSA Review and Development Plan

**Review date:** November 8, 2026, UTC  
**Spark repository reviewed:** `a164e8efa3a6527bca90aec3d224c26cc433bdb1`  
**Upstream LSA reviewed:** `5ebea732d639576d2a7579b0de40e6252f9001f5`  
**Branch:** `loop72-dlsa`  
**Scope:** A Spark X2.6 4B-specific variant retaining 72 executed layer visits, adding usable D-LSA and bounded GPU cache residency, preserving tiered memory (`M0–M3`), and investigating a maximum addressable context of 5,000,000 tokens. `MoE-Engram` (`M4`) is deferred (`construct_engram=False`).

---

## 1. Recommended Decision & Branch Architecture

Build a dedicated **Spark X2.6 4B Loop72 LSA** variant around the existing 72-visit loop architecture:
- **Frozen Spark Backbone (`XHToken/Spark-X2.6-4B`) + 24 `TwoStreamMHC` Loop Controllers**:
  - `Prelude [0..3]` (4 visits) + `Core Pass 1 [4..27]` (24 visits) + `Core Pass 2 [4..27]` (24 visits) + `Inner Pass 3 [12..23]` (12 visits) + `Coda [28..35]` (8 visits) = **72 executed layer visits** (**18 global visits** `{3:1, 7:2, 11:2, 15:3, 19:3, 23:3, 27:2, 31:1, 35:1}` and **54 local sliding visits**).
  - Native GELU MLP activation preserved (`preserve_native_activation=True`), with explicit runtime identity (`native_gelu_loop72` vs `legacy_silu_loop72`) and zero `MoE-Engram` parameter allocation (`construct_engram=False`).
- **GPU Memory (`Loop72HybridPagedCache`)**:
  - 54 local sliding visits bounded strictly to `local_window = 512` tokens (`108 MiB` total in BF16 across all 54 visits regardless of sequence length).
  - 18 global visits with explicitly capped GPU hot residency (`sinks=4`, `recent_tokens=512`, and LRU hot pages up to `global_hot_tokens_per_visit=8192`) backed by sole-owner CPU/NVMe cold pages (`bf16`, `int8`, `int4`, or `mixed_rotary_bf16` via `sparkx_kv_codec.py`).
- **Per-Visit Faithful D-LSA (`VisitAwareDLSAIndexBank` & `sparkx_sparse_attention.py`)**:
  - 18 independent per-visit key/query projection maps (`18 * 2 * 768 * 64 = 1,769,472` parameters) operating on the 192-D RoPE-free slice (`[64:256)`).
  - Mutable index state isolated strictly per `(model_revision, behavior_fingerprint, projection_revision, session_id, memory_epoch, physical_layer, virtual_visit)` inside the cache backend.
  - Append-only block bucket pages (no quadratic full-prefix `torch.cat`, no full-history FP32 latent copy on GPU) with both flat control (`search="flat"`) and bounded-beam hierarchical tree traversal (`search="tree"`, $O(r f w \log_f P + K r)$).
  - Strictly causal single-query sparse decode and tiled multi-query sparse prefill (`sparse_query_tile_tokens=16`) with final-position-only generation logits (`generation_logits="last_position"`).
- **Persistent Episodic Memory (`M3`)**:
  - Versioned, byte-budgeted `EpisodicSideMemory` serialized cleanly in checkpoints (`esm_state_dict` / `load_esm_state_dict`) with reversible consolidation (`delete_raw_after_consolidation=False` by default).

---

## 2. Implementation Work Packages Status (`loop72-dlsa`)

| Package | Concrete Work | Main Files | Status |
| :--- | :--- | :--- | :--- |
| **P0: Isolate the variant** | No-Engram construction/loading/saving (`construct_engram=False`); explicit 72-visit manifest (`enable_p3=True`); native GELU vs legacy SiLU activation identity; per-instance patches | [`sparkx_loop_v03_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py), [`load_inference_v03.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/load_inference_v03.py), [`configs/spark4b_loop72_lsa.yaml`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/configs/spark4b_loop72_lsa.yaml) | 🟩 Implemented & Verified |
| **P1: Correct cache semantics** | 72-slot hybrid local/global cache (`54` bounded 512-token local rings + `18` global archives), monotonic logical positions independent of resident length, offset-aware masks, chunked prefill & last-position logits | [`sparkx_paged_cache.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_paged_cache.py), [`sparkx_loop_v03_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py) | 🟩 Implemented & Verified |
| **P2: Unify faithful D-LSA** | `VisitAwareDLSAIndexBank` (`18` global visits, `1.77M` params), per-visit state owned by cache, source-side candidate gather, per-visit budget overrides | [`sparkx_dlsa.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_dlsa.py), [`sparkx_sparse_attention.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_sparse_attention.py) | 🟩 Implemented & Verified |
| **P3: Integrate real paging** | Sole-owner cold KV pages (`kv_page_tokens=64`), bounded GPU hot LRU cache, pinned staging, byte ledger | [`sparkx_paged_cache.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_paged_cache.py) | 🟩 Implemented & Verified |
| **P4: Stream prefill** | Bounded hidden-state chunks (`hidden_chunk_tokens=2048`), per-query causal selection (`sparse_query_tile_tokens=16`), no dense prefix masks, final-row logits | [`sparkx_sparse_attention.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_sparse_attention.py), [`sparkx_loop_v03_core.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_loop_v03_core.py) | 🟩 Implemented & Verified |
| **P5: Train and calibrate** | Streamed PCA init + sampled-query KL distillation per global visit (`18` visits) | [`train_spark_dlsa.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/train_spark_dlsa.py), [`sparkx_dlsa.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_dlsa.py) | 🟩 Implemented & Verified |
| **P6: Compress and index** | `bf16` / `int8` / `int4` / `mixed_rotary_bf16` KV codecs, append-only block buckets, bounded-beam hierarchical tree (`search="tree"`), no full-history latent copies | [`sparkx_kv_codec.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_kv_codec.py), [`sparkx_dlsa.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_dlsa.py) | 🟩 Implemented & Verified |
| **P7: Persist M3** | Bounded ESM shards, complete serialization (`esm_state_dict`), memory epochs, checksums, non-destructive consolidation | [`sparkx_memory_utils.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_memory_utils.py), [`sparkx_dynamic_memory.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_dynamic_memory.py) | 🟩 Implemented & Verified |
| **P8: Length extension** | YaRN-style global rotary scaling on the 64 rotated dims while preserving local 512-window RoPE, 4-way positional/sparse ablation runner, capacity arithmetic verifier | [`sparkx_sparse_attention.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/sparkx_sparse_attention.py), [`train_spark_dlsa.py`](file:///storage/emulated/10/Documents/Spark_X2.6_Loop/train_spark_dlsa.py) | 🟩 Implemented & Verified |
