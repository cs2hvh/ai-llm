#!/usr/bin/env python3
"""L3 — Forced-kill resume bitwise-exact canary on the PACKED-CORPUS data path.

The original ``canary_l3_resume.py`` covers the **synthetic-data** path. The
senior reviewer (2026-05-12) flagged that path as different from production:
production runs go through ``PackedCorpusReader`` + ``data_position`` seek +
``iter_packed_pairs`` — a different control surface for resume. A bug in
seek arithmetic or in checkpoint→reader handoff would not be caught by the
synthetic L3 because the synthetic iterator was independently refactored
to be resume-safe (see ``2026-05-12 synthetic iter`` fix).

This script exercises the **real** production resume path:

    1. Build a tiny tokenizer (just the 6 REQUIRED special tokens + filler).
    2. Build a tiny packed corpus (3 sources, 30 sequences, seq_len=33).
    3. Run training uninterrupted for ``total_steps`` (default 4).
       Capture: final state hash, final data_position.
    4. Run training for ``total_steps // 2`` steps. The checkpoint cadence
       saves at step N/2 with the actual ``data_position`` from the run.
    5. Resume: same checkpoint_root → loop sees existing checkpoint;
       run_pretrain peeks ``data_position`` from the manifest, computes the
       start_sequence_id, seeks the reader. Continue to ``total_steps``.
       Capture: final state hash, final data_position.
    6. Assert:
         - ``hash(state)_uninterrupted == hash(state)_resumed``
         - ``data_position`` equal
         - ``step`` equal

Acceptance criteria mirror the synthetic L3, but the failing path now points
at packed-corpus seek logic / manifest-peek / sequence_id arithmetic instead
of synthetic-iter state.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Pin CPU before any keras/jax import (matches the synthetic L3 harness).
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.canary import CheckResult, hash_training_state  # noqa: E402
from myllm.utils import get_logger  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Tiny tokenizer + tiny packed corpus fixtures
# --------------------------------------------------------------------------- #
def _build_tiny_tokenizer(out_path: Path, vocab_size: int = 64) -> str:
    """Write a minimal HF tokenizers.Tokenizer JSON with REQUIRED specials.

    Trains a tiny BPE on a 1-sentence corpus so the vocab also contains a
    few content tokens. The tokenizer is only loaded by run_pretrain.py
    on the packed-corpus path for the BOS/EOS/PAD id lookups and the
    REQUIRED-tokens verification — it never tokenises anything (the data
    is already pre-tokenised on disk).

    Returns the SHA256 hex of the tokenizer file — the corpus is built
    against this SHA so the manifest matches at load.
    """
    from myllm.data.special_tokens import REQUIRED
    from tokenizers import Tokenizer, models, trainers

    tok = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(REQUIRED),
        show_progress=False,
        min_frequency=1,
    )
    tok.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog " * 8], trainer
    )
    tok.save(str(out_path))
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def _build_tiny_corpus(
    root: Path,
    *,
    sequence_length: int,
    tokenizer_sha256: str,
    n_sequences: int = 30,
    sequences_per_shard: int = 10,
) -> None:
    """Build a 3-source packed corpus with deterministic token content.

    Each sequence's tokens are a function of its ``sequence_id`` so we can
    detect any reader off-by-one or mis-seek by hashing the data the
    training loop actually consumes.

    Token values stay in [0, 64) — safely within the tiny model's
    vocab_size and small enough to keep matmul costs trivial.
    """
    import numpy as np

    from myllm.data.packed_corpus import (
        DocSpan,
        PackedCorpusWriter,
        write_corpus_manifest,
    )

    SOURCES = ("source_a", "source_b", "source_c")

    w = PackedCorpusWriter(
        root,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        tokenizer_sha256=tokenizer_sha256,
    )
    rng = np.random.default_rng(1729)
    for sid in range(n_sequences):
        # Deterministic-but-varied tokens; values in [1, 32) so they're
        # always inside the small vocab and not equal to PAD (id 0 or 1).
        toks = rng.integers(2, 32, size=sequence_length, dtype=np.uint32)
        src = SOURCES[sid % len(SOURCES)]
        # Single doc per sequence — span covers the whole sequence.
        spans = [
            DocSpan(
                doc_span_id=0,  # writer reassigns
                sequence_id=0,  # writer reassigns
                source_id=src,
                doc_id_hash=int(sid),
                dataset_revision_id="canary-l3-packed",
                token_start_in_sequence=0,
                token_end_in_sequence=sequence_length,
                text_hash=int(sid),
            )
        ]
        w.append_sequence(toks, spans)
    w.close()

    write_corpus_manifest(
        root,
        corpus_name="canary_l3_packed_test",
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        source_revisions={s: "canary-l3-packed" for s in SOURCES},
        target_source_share={s: 1.0 / len(SOURCES) for s in SOURCES},
    )


# --------------------------------------------------------------------------- #
# Tiny model + data yamls
# --------------------------------------------------------------------------- #
_TINY_MODEL_YAML = """\
name: l3_canary_packed_tiny
arch: llama_decoder
layers: 2
hidden_dim: 64
ffn_dim: 256
num_heads: 4
num_kv_heads: 2
head_dim: 16
vocab_size: 64
tie_embeddings: true
context_length: 32
position: rope
rope_base: 10000.0
norm: rmsnorm
norm_eps: 1.0e-5
activation: swiglu
init_std: 0.02
scaled_init_for_residuals: false
z_loss_coef: 1.0e-4
qk_norm: false
optimizer:
  type: adamw
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1
  eps: 1.0e-8
lr_schedule:
  type: wsd
  peak_lr: 1.0e-3
  end_lr_ratio: 0.1
  warmup_steps: 2
  decay_fraction: 0.0
batch:
  micro_batch_per_device: 2
  sequence_length: 33
  grad_accum_steps: 1
grad_clip_global_norm: 1.0
mixed_precision: bfloat16
target_tokens: 1_000_000
checkpoint_every_steps: 1
keep_last_n: 5
keep_every_n_steps: 1
"""

_TINY_DATA_YAML = """\
name: l3_canary_packed_data_stub
batch:
  micro_batch_per_device: 2
data_seed: 42
"""


# --------------------------------------------------------------------------- #
# Subprocess driver
# --------------------------------------------------------------------------- #
def _run_pretrain_subprocess(
    *,
    model_config: Path,
    data_config: Path,
    tokenizer_path: Path,
    packed_corpus_root: Path,
    checkpoint_root: Path,
    total_steps: int,
    seed: int,
    run_name: str,
) -> int:
    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "run_pretrain.py"),
        "--model-config", str(model_config),
        "--data-config", str(data_config),
        "--tokenizer-path", str(tokenizer_path),
        "--packed-corpus-root", str(packed_corpus_root),
        "--run-name", run_name,
        "--total-steps", str(total_steps),
        "--checkpoint-root", str(checkpoint_root),
        "--seed", str(seed),
        "--no-shard",   # single-device CPU
        "--no-wandb",
    ]
    env = dict(os.environ)
    env["KERAS_BACKEND"] = "jax"
    env.setdefault("JAX_PLATFORMS", "cpu")
    log.info(
        "l3_packed_subprocess_start",
        total_steps=total_steps,
        root=str(checkpoint_root),
    )
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        log.error(
            "l3_packed_subprocess_failed",
            returncode=proc.returncode,
            stderr_tail=proc.stderr[-2000:],
        )
    return proc.returncode


def _restore_final_state(checkpoint_root: Path) -> dict:
    from myllm.training.checkpoint import CheckpointConfig, CheckpointManager

    cm = CheckpointManager(CheckpointConfig(root=str(checkpoint_root)))
    step = cm.latest_complete_step()
    if step is None:
        raise RuntimeError(f"no complete checkpoint under {checkpoint_root}")
    return cm.restore(step)


# --------------------------------------------------------------------------- #
# Main protocol
# --------------------------------------------------------------------------- #
def run_l3_packed_check(*, total_steps: int = 4) -> CheckResult:
    tmpdir = Path(tempfile.mkdtemp(prefix="canary_l3_packed_"))
    try:
        # Tokenizer + corpus fixtures.
        tok_path = tmpdir / "tokenizer.json"
        tokenizer_sha = _build_tiny_tokenizer(tok_path)
        corpus_root = tmpdir / "corpus"
        # tiny model: context_length=32 → packed seq_len = 33
        _build_tiny_corpus(
            corpus_root, sequence_length=33,
            tokenizer_sha256=tokenizer_sha,
            n_sequences=30, sequences_per_shard=10,
        )

        # Tiny model + data yamls.
        model_cfg = tmpdir / "tiny_model.yaml"
        model_cfg.write_text(_TINY_MODEL_YAML)
        data_cfg = tmpdir / "tiny_data.yaml"
        data_cfg.write_text(_TINY_DATA_YAML)

        half = max(1, total_steps // 2)

        # --- Phase 1: uninterrupted reference ----------------------------
        ref_root = tmpdir / "ref"
        ref_root.mkdir()
        rc = _run_pretrain_subprocess(
            model_config=model_cfg, data_config=data_cfg,
            tokenizer_path=tok_path, packed_corpus_root=corpus_root,
            checkpoint_root=ref_root, total_steps=total_steps, seed=42,
            run_name="l3-packed-ref",
        )
        if rc != 0:
            return CheckResult(
                name="l3_packed_resume",
                passed=False,
                summary=f"reference run exited rc={rc}",
                fix_hint="Reference run failed before any resume logic "
                         "could be tested. See l3_packed_subprocess_failed "
                         "log line above for stderr_tail. Most likely a "
                         "tokenizer/corpus/model-config wiring issue — "
                         "this is upstream of the resume protocol.",
            )
        ref_state = _restore_final_state(ref_root)
        ref_hash = hash_training_state(ref_state)
        ref_data_pos = int(ref_state.get("data_position", 0))
        ref_step = int(ref_state.get("step", 0))

        # --- Phase 2: interrupted run (N/2 steps) ------------------------
        run_root = tmpdir / "run"
        run_root.mkdir()
        rc = _run_pretrain_subprocess(
            model_config=model_cfg, data_config=data_cfg,
            tokenizer_path=tok_path, packed_corpus_root=corpus_root,
            checkpoint_root=run_root, total_steps=half, seed=42,
            run_name="l3-packed-interrupt",
        )
        if rc != 0:
            return CheckResult(
                name="l3_packed_resume",
                passed=False,
                summary=f"interrupted run exited rc={rc} at {half} steps",
            )

        # --- Phase 3: resume to total_steps ------------------------------
        rc = _run_pretrain_subprocess(
            model_config=model_cfg, data_config=data_cfg,
            tokenizer_path=tok_path, packed_corpus_root=corpus_root,
            checkpoint_root=run_root, total_steps=total_steps, seed=42,
            run_name="l3-packed-resume",
        )
        if rc != 0:
            return CheckResult(
                name="l3_packed_resume",
                passed=False,
                summary=f"resumed run exited rc={rc}",
            )
        resumed_state = _restore_final_state(run_root)
        resumed_hash = hash_training_state(resumed_state)
        resumed_data_pos = int(resumed_state.get("data_position", 0))
        resumed_step = int(resumed_state.get("step", 0))

        step_match = ref_step == resumed_step
        data_pos_match = ref_data_pos == resumed_data_pos
        hash_match = ref_hash == resumed_hash

        passed = step_match and data_pos_match and hash_match
        summary = (
            "bitwise-exact resume verified on packed-corpus path"
            if passed
            else "packed-corpus resume diverged from uninterrupted run"
        )
        details = {
            "total_steps": total_steps,
            "kill_after_step": half,
            "ref_step": ref_step,
            "resumed_step": resumed_step,
            "ref_data_position": ref_data_pos,
            "resumed_data_position": resumed_data_pos,
            "ref_state_hash": ref_hash,
            "resumed_state_hash": resumed_hash,
            "step_match": step_match,
            "data_position_match": data_pos_match,
            "hash_match": hash_match,
        }
        fix_hint = None
        if not passed:
            failing = []
            if not step_match:
                failing.append(f"step ({ref_step} vs {resumed_step})")
            if not data_pos_match:
                failing.append(
                    f"data_position ({ref_data_pos} vs {resumed_data_pos})"
                )
            if not hash_match:
                failing.append("state hash")
            fix_hint = (
                "Diverging: " + ", ".join(failing) + ". Production-path "
                "specific bug — look at PackedCorpusReader.seek arithmetic, "
                "the data_position → sequence_id mapping in "
                "scripts/run_pretrain.py (peek_data_position_from_checkpoint "
                "+ sequence_id_from_data_position), and the checkpoint's "
                "manifest.extra['data_position'] field. Synthetic L3 would "
                "have passed because synthetic data path uses a different "
                "resume mechanism (start_step parameter)."
            )
        return CheckResult(
            name="l3_packed_resume",
            passed=passed,
            summary=summary,
            details=details,
            fix_hint=fix_hint,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps", type=int, default=4)
    p.add_argument("--format", choices=("text", "json"), default="text")
    args = p.parse_args()

    result = run_l3_packed_check(total_steps=args.total_steps)
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        sigil = "✓" if result.passed else "✗"
        print(f"{sigil} {result.name}: {result.summary}")
        for k, v in result.details.items():
            print(f"    {k}: {v}")
        if result.fix_hint:
            print(f"    ↳ fix: {result.fix_hint}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
