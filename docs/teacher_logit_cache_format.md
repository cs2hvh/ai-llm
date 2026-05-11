# Teacher Logit Cache Binary Format — v1 (2026-05-11)

This doc specifies the on-disk format used by `scripts/cache_teacher_logits.py`
(to be written in the next R0 PR) and consumed by the future
`src/myllm/data/teacher_cache.py` reader.

**Why this matters:** the cache will hold ~14 TB of teacher logit data
across 3 teachers, mirrored to R2. Getting the format right *now* — before
we generate the cache — avoids re-running the (~$15K) caching pipeline
later. Designed for:

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

| Teacher | Tokens cached | K | Bytes/position | Total per teacher |
|---|---|---|---|---|
| DeepSeek-V4-Pro-Base | 150 B | 8 | 8×(2+4)=48 | **7.2 TB** |
| Qwen 3.6-27B | 150 B | 8 | 48 | **7.2 TB** |
| Mistral-Medium-3.5-128B | 150 B | 8 | 48 | **7.2 TB** |
| **Combined** | — | — | — | **21.6 TB on R2** |

R2 storage cost: $15/TB-month → ~$325/month while training is active.
Negligible vs the ~$60-90K Phase 3 training cost.

(My earlier estimate of "7 TB total" was wrong — I forgot to multiply by
both logit *value* and *index*. The correct figure is ~21.6 TB. Still
nowhere near a budget concern.)

## Cache-generation cost (one-time)

For each teacher, we need to do a single forward pass over 150B tokens.
Throughput depends on model size (MoE active params for DeepSeek; full
params for the dense ones) and inference server (vLLM batched).

| Teacher | Params | Throughput on 8× B200 (vLLM batched) | Wall time | Cost @ $40/hr |
|---|---|---|---|---|
| DeepSeek-V4-Pro-Base | 671B MoE (37B active) | ~250K tok/s | ~7 days | **~$6,700** |
| Qwen 3.6-27B (dense) | 27B | ~500K tok/s | ~3.5 days | **~$3,300** |
| Mistral-Medium-3.5-128B (dense) | 128B | ~175K tok/s | ~10 days | **~$9,600** |
| **Total cache generation** | — | — | **~20 days** | **~$15-25K** |

Confidence on these numbers: ±50%. The unknowns are (a) actual vLLM
batched throughput on B200 for each model — published numbers are
mostly on H100, (b) cluster-utilization losses, (c) whether DeepSeek's
MoE inference scales linearly with active params or has more overhead.
**The honest mid-range is $20K with wide error bars.**

Earlier in the project (`docs/teacher_distillation_strategy.md` v1)
I quoted $6-10K total — that was wrong, based on the dense-37B
extrapolation without accounting for the full 150B-token coverage. The
$20K figure here supersedes it. **The Phase 3 total budget should be
revised from ~$60-90K to ~$80-115K to reflect this.**

We can reduce this cost by:
1. **Dropping one teacher** (Mistral is the smallest marginal contributor;
   saves $9-10K, leaves us with DeepSeek + Qwen).
2. **Reducing decay-phase coverage** from 15% to 10% (saves ~$5K, but
   reduces distillation effect).
3. **Lower K** (K=4 saves a small amount; not worth it).

I'd suggest keeping all three teachers for v1 unless the budget delta
is a problem.

## Decision lever: drop one teacher if budget is tight

The marginal value of the third teacher (typically Mistral-Medium for
EU-data diversity) is the smallest. If we want to save ~$18K of caching
cost, drop Mistral and run with just DeepSeek + Qwen. The dossier doesn't
provide ablations at this depth so the call is a judgment one. My
recommendation: keep all three for v1; if quality is great and we want
to save for v2, then drop Mistral.

## File touch list (next PRs)

- `scripts/cache_teacher_logits.py` — main cache-generation entrypoint
- `src/myllm/data/teacher_cache.py` — runtime reader
- `src/myllm/data/teacher_cache_manifest.py` — manifest read/write/validate
- `tests/test_teacher_cache_format.py` — round-trip serialize/deserialize
- `tests/test_teacher_cache_reader.py` — random-access reads from a small
  synthetic cache
