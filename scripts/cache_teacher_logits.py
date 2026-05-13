#!/usr/bin/env python3
"""Generate the offline teacher logit cache for distillation (R0).

Runs one teacher model (loaded from HuggingFace) over a corpus shard,
computes top-K logits per token, and writes Arrow shards according to
``docs/teacher_logit_cache_format.md``.

This is the heavy lifter: each invocation can run for days and consume
$5-15K on an 8× B200 pod. Resumability is built in — interrupted runs
pick up at the next shard boundary based on the manifest. Idempotent
re-runs are cheap (skip already-written shards by content-addressed key).

CLI parameters fall into three groups:

1. **Teacher selection**  — which model to run, where to find it.
2. **Corpus selection**   — which tokens to process.
3. **Output destination** — local cache dir + optional R2 mirror.

Typical decay-phase invocation (per teacher):

    python scripts/cache_teacher_logits.py \\
        --teacher-id deepseek-v4-pro-base \\
        --teacher-hf-model deepseek-ai/DeepSeek-V4-Pro-Base \\
        --tokenized-corpus /path/to/decay_phase_corpus.bin \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --top-k 8 \\
        --shard-tokens 10000000 \\
        --output-dir artifacts/distillation_cache \\
        --r2-prefix distillation_cache/deepseek-v4-pro-base/k8/

This turn ships the orchestration skeleton + a synthetic-teacher mode
(``--synthetic-teacher``) used by tests. The real-teacher path (vLLM
loading + batched inference) is implemented in a follow-up PR; the
hook ``_load_teacher`` raises NotImplementedError for now.

vLLM rationale: HuggingFace transformers' .generate() is too slow for
cache generation (single-batch inference, no continuous batching).
vLLM's batched OfflineInference yields 3-10× higher throughput on
B200 for the model sizes we care about. The integration is straightforward
but adds a fat dependency; we'll add it when we actually run the cache.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Iterator

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.data.teacher_cache import (  # noqa: E402
    CacheManifest,
    CacheShard,
    ShardManifestEntry,
    compute_local_path,
    compute_shard_key,
    read_manifest,
    write_manifest,
    write_shard,
)
from myllm.utils import configure_logging, get_logger  # noqa: E402
from myllm.utils.io import sha256_file  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Teacher loading
# --------------------------------------------------------------------------- #
def _load_teacher(teacher_hf_model: str, synthetic: bool):
    """Return a callable ``forward(token_ids: [B,S]) -> logits: [B,S,V]``.

    For ``synthetic=True``, returns a mock that emits deterministic
    pseudo-random logits — used by tests so we don't need a 671B model
    loaded to verify the orchestration plumbing.

    For real teachers, we use HuggingFace transformers in bf16 with
    ``device_map="auto"`` (tensor-parallel across all visible GPUs). This
    is sufficient for the top-K mass audit (~65k positions). For the
    actual top-K *cache* run (multi-billion-token throughput), swap to
    vLLM's offline inference here — same callable contract.

    Why transformers (not vLLM) for the audit:
      - vLLM's ``prompt_logprobs`` API caps at the top-N logprobs and
        doesn't easily expose the full V=131k softmax denominator.
        Patching it is more work than just running transformers once.
      - The audit is one-shot, not throughput-critical; transformers'
        per-batch forward is fast enough on 2× A100 for a 32B teacher.
      - vLLM stays the right answer for the LATER cache-generation step
        where 3-10× throughput on billions of tokens matters.
    """
    if synthetic:
        return _make_synthetic_teacher(teacher_hf_model)
    return _make_transformers_teacher(teacher_hf_model)


def _make_transformers_teacher(hf_model: str):
    """Load a HuggingFace causal-LM teacher in bf16; return ``forward``.

    Lazy import — keeps the module import cheap on machines that won't
    actually run real inference (e.g. CPU dev box doing the synthetic
    audit).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as e:  # pragma: no cover — exercised on pod only
        raise ImportError(
            "real teacher loading requires `torch` + `transformers`; "
            "install on the pod with `pip install torch transformers`"
        ) from e
    import numpy as np

    # Validate torch can see CUDA before loading — bail loudly if not,
    # since "device_map='auto'" silently falls back to CPU which is
    # ~100x slower (the audit pod 2026-05-13 ran an OLMo-2 forward on
    # CPU for 15+ min before we noticed).
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. With cuda_runtime / cuDNN "
            "potentially out of sync after the cuDNN pin downgrade, "
            "fix with: pip install --upgrade --no-deps 'torch>=2.4' && "
            "pip install --upgrade 'torch>=2.4'. Then re-run the audit."
        )
    n_visible = torch.cuda.device_count()
    log.info("loading_teacher_transformers", model=hf_model, n_gpus=n_visible)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # spreads weights across all visible GPUs
        low_cpu_mem_usage=True,
        trust_remote_code=True,  # some teachers (DeepSeek-V2-Lite) need this
    )
    model.eval()
    # Sanity check that device_map="auto" actually placed weights on a
    # GPU (not CPU fallback). If not, force the first parameter device
    # to be CUDA so the forward runs on GPU.
    first_param_device = next(model.parameters()).device
    if first_param_device.type != "cuda":
        log.warning("device_map_fell_back_to_cpu", first_param_device=str(first_param_device))
        model = model.to("cuda:0")
        first_param_device = next(model.parameters()).device
    teacher_vocab = int(model.config.vocab_size)
    log.info(
        "teacher_loaded", model=hf_model, vocab_size=teacher_vocab,
        first_param_device=str(first_param_device),
    )

    @torch.inference_mode()
    def forward(token_ids):
        # token_ids: np.ndarray [B, S] (int).
        # Returns: float32 logits [B, S, V_teacher] on host (numpy).
        #
        # Clamp incoming token ids to the teacher's vocab range. The
        # audit corpus may use IDs from our 131k SentencePiece vocab,
        # but teacher vocabularies are typically 50k-130k. Without the
        # clamp, embedding(input_ids) raises
        # `IndexError: index out of range in self`.
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        ids = ids.clamp_(max=teacher_vocab - 1)
        # device_map="auto" places the embedding on the first device.
        first_device = next(model.parameters()).device
        ids = ids.to(first_device, non_blocking=True)
        out = model(ids, use_cache=False)
        # out.logits: bf16/f16 on the last shard; cast and move to CPU.
        return out.logits.float().cpu().numpy()

    return forward


def _make_synthetic_teacher(seed_name: str):
    """A deterministic mock teacher for orchestration tests."""
    import numpy as np

    rng_seed = int(hashlib.sha256(seed_name.encode()).hexdigest()[:8], 16)

    def forward(token_ids):
        # token_ids: np.ndarray [B, S]
        # Return: float32 logits [B, S, V] where V = 131072 (production vocab).
        # We don't need real probabilities — just stable, finite values that
        # round-trip through bfloat16. Hash token_id + position to derive.
        rng = np.random.default_rng(seed=rng_seed + int(token_ids.sum()))
        B, S = token_ids.shape
        V = 131072
        return rng.normal(0.0, 1.0, size=(B, S, V)).astype("float32")

    return forward


# --------------------------------------------------------------------------- #
# Corpus iteration
# --------------------------------------------------------------------------- #
def _iter_tokenized_corpus(
    corpus_path: Path,
    start_token: int,
    end_token: int,
    batch_size: int,
    sequence_length: int,
) -> Iterator[tuple[int, int, Any]]:
    """Stream the tokenized corpus, yielding ``(start_pos, end_pos, batch)``.

    The corpus file is expected to be a flat binary array of uint32 token
    IDs (the same format ``scripts/pack_shards.py`` will emit; for now,
    tests use a small synthetic corpus).

    Yields one batch at a time; the caller is responsible for batching
    these into shards.
    """
    import numpy as np

    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus file not found: {corpus_path}")

    arr = np.memmap(corpus_path, dtype="uint32", mode="r")
    total = arr.shape[0]
    if end_token > total:
        raise ValueError(
            f"requested end_token {end_token} exceeds corpus length {total}"
        )
    tokens_per_batch = batch_size * sequence_length
    pos = start_token
    while pos < end_token:
        next_pos = min(pos + tokens_per_batch, end_token)
        n = next_pos - pos
        # Pad to a full batch x seq layout for the teacher (we drop the
        # padded positions at top-K extraction time).
        n_pad = tokens_per_batch - n
        chunk = np.concatenate(
            [arr[pos:next_pos], np.zeros(n_pad, dtype="uint32")]
        )[:tokens_per_batch].reshape(batch_size, sequence_length)
        yield pos, next_pos, chunk
        pos = next_pos


# --------------------------------------------------------------------------- #
# Top-K extraction
# --------------------------------------------------------------------------- #
def _extract_topk(logits: Any, top_k: int) -> tuple[Any, Any]:
    """Return ``(topk_logits[N, K] bfloat16-as-uint16, topk_indices[N, K] uint32)``.

    Flattens batch × sequence into a single token axis.
    """
    import numpy as np

    B, S, V = logits.shape
    flat = logits.reshape(B * S, V)
    # Use argpartition for speed on huge V; sort the top-K only.
    idx_unsorted = np.argpartition(flat, -top_k, axis=-1)[:, -top_k:]
    vals_unsorted = np.take_along_axis(flat, idx_unsorted, axis=-1)
    order = np.argsort(-vals_unsorted, axis=-1)
    topk_indices = np.take_along_axis(idx_unsorted, order, axis=-1).astype("uint32")
    topk_logits_f32 = np.take_along_axis(vals_unsorted, order, axis=-1)
    # Convert float32 → bfloat16 via uint16 (truncate the lower 16 bits of
    # IEEE 754 mantissa, which is the bfloat16 rounding rule).
    topk_logits_u16 = (
        topk_logits_f32.view("uint32").astype("uint32") >> 16
    ).astype("uint16")
    return topk_logits_u16, topk_indices


# --------------------------------------------------------------------------- #
# Main: generate cache for one teacher
# --------------------------------------------------------------------------- #
def generate_teacher_cache(
    teacher_id: str,
    teacher_hf_model: str,
    tokenized_corpus: Path,
    corpus_sha256: str,
    tokenizer_sha256: str,
    output_dir: Path,
    top_k: int = 8,
    shard_tokens: int = 10_000_000,
    start_token: int = 0,
    end_token: int | None = None,
    batch_size: int = 16,
    sequence_length: int = 2048,
    synthetic_teacher: bool = False,
) -> CacheManifest:
    """Generate the full top-K cache for one teacher over the given corpus range.

    Returns the (final) CacheManifest. Resumes from the previous run by
    reading any existing manifest and skipping already-written shards.
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{teacher_id}_manifest.json"
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        if manifest.teacher_id != teacher_id:
            raise ValueError(
                f"existing manifest is for teacher {manifest.teacher_id!r}; "
                f"refusing to mix with {teacher_id!r}"
            )
        log.info(
            "cache_resuming_from_existing_manifest",
            teacher_id=teacher_id,
            existing_shards=len(manifest.shards),
            total_tokens_cached=manifest.total_tokens(),
        )
    else:
        manifest = CacheManifest(
            teacher_id=teacher_id,
            corpus_sha256=corpus_sha256,
            tokenizer_sha256=tokenizer_sha256,
            top_k=top_k,
        )

    if end_token is None:
        arr = np.memmap(tokenized_corpus, dtype="uint32", mode="r")
        end_token = arr.shape[0]

    teacher_forward = _load_teacher(teacher_hf_model, synthetic=synthetic_teacher)

    # Determine where to resume: max end_token_position across existing shards.
    covered_end = max(
        (s.end_token_position for s in manifest.shards), default=start_token
    )
    if covered_end > start_token:
        log.info(
            "cache_resume_skip_to",
            position=covered_end,
            already_covered_tokens=covered_end - start_token,
        )

    # Buffer for the current shard.
    cur_shard_start = covered_end
    cur_logits = []
    cur_indices = []
    cur_tokens_accum = 0
    t0 = time.time()

    for batch_start, batch_end, batch in _iter_tokenized_corpus(
        tokenized_corpus,
        start_token=covered_end,
        end_token=end_token,
        batch_size=batch_size,
        sequence_length=sequence_length,
    ):
        n_real_tokens = batch_end - batch_start
        # Teacher forward.
        teacher_logits = teacher_forward(batch)
        topk_logits, topk_indices = _extract_topk(teacher_logits, top_k)
        # Trim padded positions if this was the tail of the requested range.
        topk_logits = topk_logits[:n_real_tokens]
        topk_indices = topk_indices[:n_real_tokens]
        cur_logits.append(topk_logits)
        cur_indices.append(topk_indices)
        cur_tokens_accum += n_real_tokens

        # Flush shard if we've accumulated enough.
        if cur_tokens_accum >= shard_tokens:
            shard_logits = np.concatenate(cur_logits, axis=0)
            shard_indices = np.concatenate(cur_indices, axis=0)
            shard_end = cur_shard_start + cur_tokens_accum
            shard = CacheShard(
                teacher_id=teacher_id,
                corpus_sha256=corpus_sha256,
                tokenizer_sha256=tokenizer_sha256,
                start_token_position=cur_shard_start,
                end_token_position=shard_end,
                top_k=top_k,
                logits=shard_logits,
                indices=shard_indices,
            )
            key = compute_shard_key(
                teacher_id, top_k, corpus_sha256, cur_shard_start, shard_end
            )
            local_path = compute_local_path(output_dir, key)
            sha = write_shard(shard, local_path)
            manifest.shards.append(ShardManifestEntry(
                start_token_position=cur_shard_start,
                end_token_position=shard_end,
                r2_key=key,
                sha256=sha,
            ))
            write_manifest(manifest, manifest_path)
            elapsed = time.time() - t0
            log.info(
                "cache_shard_written",
                teacher_id=teacher_id,
                start=cur_shard_start,
                end=shard_end,
                tokens=cur_tokens_accum,
                shard_sha256=sha,
                local_path=str(local_path),
                tokens_per_sec=round(cur_tokens_accum / max(1e-3, elapsed), 1),
            )
            cur_shard_start = shard_end
            cur_logits = []
            cur_indices = []
            cur_tokens_accum = 0
            t0 = time.time()

    # Flush remaining partial shard if any.
    if cur_tokens_accum > 0:
        shard_logits = np.concatenate(cur_logits, axis=0)
        shard_indices = np.concatenate(cur_indices, axis=0)
        shard_end = cur_shard_start + cur_tokens_accum
        shard = CacheShard(
            teacher_id=teacher_id,
            corpus_sha256=corpus_sha256,
            tokenizer_sha256=tokenizer_sha256,
            start_token_position=cur_shard_start,
            end_token_position=shard_end,
            top_k=top_k,
            logits=shard_logits,
            indices=shard_indices,
        )
        key = compute_shard_key(
            teacher_id, top_k, corpus_sha256, cur_shard_start, shard_end
        )
        local_path = compute_local_path(output_dir, key)
        sha = write_shard(shard, local_path)
        manifest.shards.append(ShardManifestEntry(
            start_token_position=cur_shard_start,
            end_token_position=shard_end,
            r2_key=key,
            sha256=sha,
        ))
        write_manifest(manifest, manifest_path)
        log.info(
            "cache_partial_shard_flushed",
            teacher_id=teacher_id,
            start=cur_shard_start,
            end=shard_end,
            tokens=cur_tokens_accum,
        )

    log.info(
        "cache_complete",
        teacher_id=teacher_id,
        total_shards=len(manifest.shards),
        total_tokens=manifest.total_tokens(),
    )
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-id", required=True, help="Stable ID, e.g. deepseek-v4-pro-base")
    p.add_argument("--teacher-hf-model", required=True, help="HF model ID for AutoModelForCausalLM.")
    p.add_argument("--tokenized-corpus", required=True, type=Path,
                   help="Flat uint32 binary file of token IDs (output of pack_shards.py).")
    p.add_argument("--tokenizer-path", default="artifacts/tokenizer_v1.json",
                   help="Local tokenizer.json; used for tokenizer_sha256 stamp.")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--shard-tokens", type=int, default=10_000_000,
                   help="Target tokens per shard. ~38GB at K=8 bfloat16.")
    p.add_argument("--start-token", type=int, default=0)
    p.add_argument("--end-token", type=int, default=None,
                   help="If unset, runs through the end of the corpus.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/distillation_cache"))
    p.add_argument("--synthetic-teacher", action="store_true",
                   help="Use a deterministic mock teacher (for tests).")
    args = p.parse_args()

    configure_logging()

    corpus_sha = sha256_file(args.tokenized_corpus)
    tokenizer_sha = sha256_file(Path(args.tokenizer_path))

    manifest = generate_teacher_cache(
        teacher_id=args.teacher_id,
        teacher_hf_model=args.teacher_hf_model,
        tokenized_corpus=args.tokenized_corpus,
        corpus_sha256=corpus_sha,
        tokenizer_sha256=tokenizer_sha,
        output_dir=args.output_dir,
        top_k=args.top_k,
        shard_tokens=args.shard_tokens,
        start_token=args.start_token,
        end_token=args.end_token,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        synthetic_teacher=args.synthetic_teacher,
    )
    log.info(
        "cache_run_finished",
        teacher_id=args.teacher_id,
        total_shards=len(manifest.shards),
        total_tokens=manifest.total_tokens(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
