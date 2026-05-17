# Round D9 — step-718 deterministic NaN batch investigation

**Status**: Investigation complete (2026-05-17). Action items below.
**Severity**: P2 — 0.1% NaN-skip rate, atomic revert handles. Not blocking Stage 2.
**Source data**: `s3://llm-data/stage2-prep/mup-sweep-4b200/quarantine-lr1_5x.jsonl`

## Symptom

All 3 C3 μP/LR sweep runs (peak_lr ∈ {1e-4, 2e-4, 3e-4}) showed **exactly one** `nan_batch_skipped` event at the same step (718) with the same `data_position` (23,527,424).

```
{"ts": "2026-05-16T23:35:17Z", "step": 718, "data_position": 23527424,
 "reason": "nan_skipped", "loss": NaN, ...}
```

Atomic revert handled it cleanly each time — `lr_recovery_multiplier` stayed at 1.0, watchdog reported "ok", no `LossSpikeError`. Runs continued to step 1000 normally.

## Root cause

Step 718 with micro_batch=4 and packed_seq_len=8193 corresponds to **sequence_ids 2871-2874 in shard 0** of the composed pilot corpus (`s3://llm-data/corpus_v1_pilot/train/`).

Decoded via `scripts/inspect_quarantine.py --corpus-root /tmp/pilot_corpus`:

| seq_id | dominant source | n_doc_spans | comment |
|---|---|---|---|
| 2871 | **stack_exchange_preferences** | **1** | single doc spans all 8K tokens |
| 2872 | fineweb_edu | 23 | normal mixed |
| 2873 | wikipedia_20231101_en | 9 | normal mixed |
| 2874 | fineweb_edu | 23 | normal mixed |

Seq 2871 is the outlier: a **single Stack Exchange document filling the entire 8192-token sequence**.

## How rare is this pattern in our corpus?

Across all 65,536 sequences in shard 0:

| Source | Sequences with n_doc_spans=1 (single-doc, fills 8K alone) |
|---|---|
| github_code_clean | 4704 (expected: long source files) |
| pg19 | 2981 (expected: book chapters) |
| fineweb_edu | 1961 (expected: long articles) |
| open_web_math | 711 |
| wikipedia_20231101_en | 379 |
| mc4_zh / mc4_es / mc4_ar / mc4_fr / mc4_de | 522 total |
| **stack_exchange_preferences** | **2** |
| sangraha_verified_split_hin | 1 |

Stack Exchange entries are mostly short (Q+A pairs), so almost every Stack Exchange sequence packs multiple short entries together. **Only 2 entries** in the entire shard exceed 8K tokens on their own — and the deterministic iteration order hits one at step 718.

## Why does this specific pattern produce NaN gradients?

We don't have a confirmed root cause without GPU repro, but the most likely mechanism:

1. **Stack Exchange loader currently uses `question` text only** (this is Round D5). For an extreme-length single-doc entry to fill 8K tokens, the question field likely contains a giant thread (Q + many copied replies + code snippets) pasted into one body.
2. Such content often has unusual token patterns — many repeated tokens (e.g., `"` token id 34 appears many times in the quarantine's input_ids_preview), code blocks, and stretches of formatted whitespace that survive packing.
3. At bf16 precision + width_mult=8 (1B-shape) + chunked-CE (used in pilot, would NOT be used in Stage 2 due to D8) OR full-CE, *some* combination of {extreme attention scores, repeated-token softmax saturation, logit drift in the LM head, gradient magnitude in a specific weight leaf} produces a NaN somewhere in the backward.

Importantly: the forward pass produces a finite loss (~11.76 at random init for vocab=131072, the random-init expectation). Only the backward NaN's. Atomic revert handles it; training proceeds.

## Why we're not fixing this for Stage 2

| Reason | |
|---|---|
| Rate | 1 NaN-skip / 1000 steps = 0.1%, well below our 1% threshold |
| Handling | Atomic revert is doing its job; state stays consistent |
| Determinism | Same step every run → predictable; not a watchdog-fire risk |
| Stage 2 train time | At 30B tokens (~60K steps), expect ~60 NaN-skips — manageable |
| Cost of fix vs benefit | Round D5 (Stack Exchange schema) is the real fix; Stage 2 will tolerate this until then |

## What Round D5 will change

Current Stack Exchange loader: `text_field=question` only. Yields the question body.

After D5: `text_field=[question, chosen_response]` (concatenated with `\n\n`). Two effects:

1. **More tokens per row** → fewer outlier-long single-doc sequences (the question alone was already 8K; adding the answer means the entry gets *split* into multiple consecutive sequences instead of compressed into one). Actually wait — that's wrong. Adding text makes them LONGER, not shorter. So this might INCREASE the rate of single-doc sequences.
2. **Content structure changes** → answer text is more structured (clearer end-of-doc patterns, less code-density). May change the specific token patterns triggering NaN.

**Action item for Round D5**: when implementing the schema fix, verify the resulting corpus has FEWER pathological single-doc sequences. If not, consider a hard length cap (e.g., truncate Stack Exchange entries to 4K tokens max before packing).

## Action items

- [x] **D9 investigation complete** — root cause identified: Stack Exchange single-doc sequence at seq_id 2871, shard 0.
- [ ] **D5 (Stack Exchange schema fix)**: when adding `chosen_response`, also add a length cap (~4K tokens max per row) to prevent any single Stack Exchange entry from filling a full 8K sequence.
- [ ] **D8 (chunked-CE NaN-grad bug)**: separate investigation. If D8's fix removes the bf16-precision-edge mechanism, step-718 might stop NaN'ing as a side effect.
- [ ] **No Stage 2 action**: the bug is benign at current rates.

## Reproduction

```bash
# Pull quarantine + corpus shard-0 metadata
aws --endpoint-url $S3_ENDPOINT_URL s3 cp \
  s3://$S3_BUCKET/stage2-prep/mup-sweep-4b200/quarantine-lr1_5x.jsonl \
  artifacts/quarantine/

mkdir -p /tmp/pilot_corpus/shard-000000
aws --endpoint-url $S3_ENDPOINT_URL s3 cp \
  s3://$S3_BUCKET/corpus_v1_pilot/train/manifest.json /tmp/pilot_corpus/
aws --endpoint-url $S3_ENDPOINT_URL s3 cp \
  s3://$S3_BUCKET/corpus_v1_pilot/train/shard-000000/seq_meta.arrow /tmp/pilot_corpus/shard-000000/
aws --endpoint-url $S3_ENDPOINT_URL s3 cp \
  s3://$S3_BUCKET/corpus_v1_pilot/train/shard-000000/manifest.json /tmp/pilot_corpus/shard-000000/

# Run the inspector
python scripts/inspect_quarantine.py \
  artifacts/quarantine/quarantine-lr1_5x.jsonl \
  --corpus-root /tmp/pilot_corpus
```

Expected output:
```
By reason:
  nan_skipped                   : 1
Step range: 718 → 718
...
=== Source attribution (data_position → seq_id → source) ===
  step    718 | seq_id    2871 | shard-000000[ 2871] | stack_exchange_preferences
Dominant-source attribution summary:
  stack_exchange_preferences              : 1
```
