# Teacher Logit Cache Binary Format — v1 (2026-05-11, revised 2026-05-12)

This doc specifies the on-disk format used by `scripts/cache_teacher_logits.py`
and consumed by `src/myllm/data/teacher_cache.py` (both ✅ shipped).

**Why this matters:** the cache will hold ~7.2 TB (canary, 1 teacher) or
~14.4 TB (production, 2 teachers) of teacher logit data, mirrored to R2.
Getting the format right *now* — before we generate the cache — avoids
re-running the (~$5-15K) caching pipeline later. Designed for:

1. **Fast random access** during training. Each batch's positions get
   their cached top-K via a single seek + read per shard.
2. **Resumable generation**. If the cache job dies mid-run, the partial
   shards must be detectable + re-usable.
3. **Format versioning**. Field 0 of every shard is a 4-byte format
   version; the reader rejects unknown versions loudly.
4. **Content-addressed naming**. Cache shards are keyed by
   `(corpus_sha256, tokenizer_sha256, teacher_id, top_k)` so re-runs
   of the same configuration deterministically resolve to the same R2
   keys (avoiding accidental duplication of 4.8 TB).

## Per-shard layout

One shard = the top-K logits for **one contiguous range of token positions
in the training corpus**, for one teacher. We use Arrow IPC streams
(stored as `.arrow` files) — they're random-access friendly, have a stable
Python + C++ API, and support memory-mapped reads which avoids loading
4.8 TB into RAM.

Shard schema:

```
Shard {
  format_version:        uint32       (= 1 for this spec)
  corpus_sha256:         binary(32)   (sha256 of the corpus shard this teacher saw)
  tokenizer_sha256:      binary(32)   (sha256 of the tokenizer used)
  teacher_id:            utf8         ("deepseek-v4-pro-base" etc.)
  start_token_position:  uint64       (inclusive)
  end_token_position:    uint64       (exclusive)
  top_k:                 uint8        (typically 8)
  logits:                fixed_size_list[bfloat16, K]  one row per token position
  indices:               fixed_size_list[uint32,  K]   one row per token position
}
```

(Format-version = 1. Bump to 2 only if storage layout changes — adding
metadata fields can be done without a version bump.)

## Naming

Each shard's R2 key is deterministically derived:

```
distillation_cache/{teacher_id}/k{top_k}/corpus_{corpus_sha[:16]}/tokens_{start}_{end}.arrow
```

E.g.:

```
distillation_cache/deepseek-v4-pro-base/k8/corpus_a8f3...c91/tokens_0000000000_0001000000.arrow
```

A shard size target of **~10M token positions** (~38 GB per shard at K=8
bfloat16) balances:
- I/O overhead (smaller = more files, more overhead per file open)
- Random-access locality (larger = more bytes to seek past)
- Resumability (smaller = less data lost on cache-job restart)

At 1T total tokens × 0.15 decay-phase fraction = 150B tokens to cache
per teacher → **~15,000 shards** per teacher → 45K shards across all
three teachers.

## Resumability

Each shard's manifest is written **last** in the cache-generation process
(same atomic-rename pattern as our Orbax checkpoint manager). A reader
treats a shard with no manifest as "incomplete, must regenerate."

Cache generation also writes one `cache_manifest.json` per teacher with:
```json
{
  "teacher_id": "deepseek-v4-pro-base",
  "corpus_sha256": "...",
  "tokenizer_sha256": "...",
  "top_k": 8,
  "shards": [
    {"start": 0, "end": 10000000, "r2_key": "...", "sha256": "..."},
    ...
  ],
  "total_tokens_cached": 150000000000,
  "format_version": 1
}
```

This manifest lets `wind_tunnel_sweep.py`-style scripts validate that the
cache covers the expected token range before launching training.

## Reader contract

`src/myllm/data/teacher_cache.py::TeacherCacheReader` (next PR) will
expose:

```python
class TeacherCacheReader:
    def __init__(self, teacher_id: str, top_k: int, tokenizer_sha: str, corpus_sha: str): ...
    def get_topk(self, token_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each position, return (logits[K], indices[K])."""
```

Reader downloads shards lazily on first touch; subsequent reads hit the
local mmap cache. Designed for ~10K-100K random-access reads per batch
without excessive disk thrashing.

## Storage planning

Teacher plan locked 2026-05-12 (see `docs/teacher_distillation_strategy.md`):
DeepSeek-V4-Pro-Base (primary) + Olmo-3-32B (secondary, production-only after canary).

| Teacher | Tokens cached | K | Bytes/position | Total per teacher |
|---|---|---|---|---|
| DeepSeek-V4-Pro-Base | 150 B | 8 | 8×(2+4)=48 | **7.2 TB** |
| Olmo-3-1125-32B | 150 B | 8 | 48 | **7.2 TB** |

**Phase 3 canary (1 teacher): ~7.2 TB on R2.**
**Phase 3 production (2 teachers after canary): ~14.4 TB on R2.**

R2 storage cost: $15/TB-month → ~$110-220/month while training is active.
Negligible vs the ~$11-25K Phase 3 training cost.

### Teachers excluded from v1 (with reason)

| Teacher | Why excluded |
|---|---|
| Mistral-Medium-3.5-128B | "Modified MIT" license with $20M revenue-cap clause on derivatives (would be a perpetual time-bomb on our weights). Dropped 2026-05-12. |
| Qwen3.6-27B | Multimodal with vision encoder + thinking-mode-default `<think>` traces; NOT a base text-only model. Dropped 2026-05-12. |
| DeepSeek-V3-Base | "DeepSeek Model License" is custom non-OSI; redundant with V4-Pro anyway. |
| Llama 3.x / Llama 4 | 700M-MAU clause + naming-prefix requirement on derivatives. |
| Gemma 2/3/4 | Gemma TOS §3.2 explicitly covers "synthetic data Outputs by Gemma" — forbidden for distillation. |

See `docs/governance/license_register.md` for the full exclusion log.

## Cache-generation cost (one-time)

For each teacher, a single forward pass over 150B tokens. Throughput
depends on model size (MoE active params for DeepSeek; full params for
the dense ones) and inference server (vLLM batched).

| Teacher | Params | Throughput on 8× B200 (vLLM batched, est.) | Wall time | Cost @ $40/hr |
|---|---|---|---|---|
| DeepSeek-V4-Pro-Base | 1.6T MoE (49B active) | ~200-250K tok/s | ~7-9 days | **~$6,700-8,500** |
| Olmo-3-1125-32B (dense) | 32B | ~400-500K tok/s | ~3.5-5 days | **~$3,300-4,800** |
| **Phase 3 canary (DeepSeek only, 20B tokens)** | — | — | ~1 day | **~$300-500** |
| **Phase 3 production (both, 150B each)** | — | — | ~10-14 days | **~$10-13K** |

Confidence: ±40%. Unknowns are (a) actual vLLM batched throughput on B200
for each model (published numbers mostly on H100), (b) cluster-utilization
losses, (c) MoE inference overhead for DeepSeek-V4-Pro.

## Canary-first plan

Per the 2026-05-12 reviewer Q&A (`docs/archive/reviewer_qa_2026-05-12.md` §2):

1. **Phase 3 canary**: 1 teacher (DeepSeek-V4-Pro-Base) × 20B tokens.
   Run matched A/B (CE-only baseline vs CE+KL) and check 8 gates including
   style leakage + distribution drift.
2. **Phase 3 production**: only add Olmo-3-32B after canary passes.

## File touch list (status)

- [x] `scripts/cache_teacher_logits.py` — shipped (vLLM producer stubbed; real producer in Phase C)
- [x] `src/myllm/data/teacher_cache.py` — Arrow writer + memmap reader + manifest validation; bf16-as-uint16 round-trip (P0-5 fix)
- [x] Manifest format with `covered_token_range` validation (fails fast if cache misses positions)
- `tests/test_teacher_cache_format.py` — round-trip serialize/deserialize
- `tests/test_teacher_cache_reader.py` — random-access reads from a small
  synthetic cache
