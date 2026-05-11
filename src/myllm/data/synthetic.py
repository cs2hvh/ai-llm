"""Synthetic data iterators for training-loop smoke tests.

Produces random token batches matching the schema the train loop expects:
``{"input_ids": int32[B, S], "labels": int32[B, S]}``. Bypasses the HF data
pipeline entirely so we can validate the loop wiring (forward, backward,
optimizer step, checkpointing) without an HF token, network, or GPU.

A 100-step run on synthetic data should:
    - decrease loss (model is overfitting random noise — that's fine)
    - hit the checkpoint cadence
    - exit cleanly

If any of those fail under synthetic data, they will fail catastrophically
on real data — so this is a cheap canary.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def make_synthetic_data_iter(
    micro_batch: int,
    sequence_length: int,
    vocab_size: int,
    n_steps: int | None = None,
    seed: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield random next-token-prediction batches.

    Args:
        micro_batch: per-device batch size (B).
        sequence_length: tokens per sequence including the shift (S+1 tokens
            are sampled so input/label can each be length S).
        vocab_size: drawn from [0, vocab_size).
        n_steps: stop after this many batches; ``None`` = infinite.
        seed: PRNG seed for reproducibility.
    """
    import numpy as np

    if micro_batch <= 0:
        raise ValueError("micro_batch must be > 0")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")

    rng = np.random.default_rng(seed)
    step = 0
    while n_steps is None or step < n_steps:
        # Sample S+1 tokens then split into input (first S) and label (last S).
        ids = rng.integers(
            0,
            vocab_size,
            size=(micro_batch, sequence_length + 1),
            dtype=np.int32,
        )
        yield {
            "input_ids": ids[:, :-1].copy(),
            "labels": ids[:, 1:].copy(),
        }
        step += 1
