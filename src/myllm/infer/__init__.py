"""Shared inference helpers (predict, generate).

Lifted out of scripts/generate.py + scripts/eval_checkpoint.py during
Round B4 (2026-05-16) so the release scorecard, generation CLI, and
post-hoc eval all share one checkpoint-load + decode path.
"""
from myllm.infer.predict import (
    LoadedCheckpoint,
    build_greedy_predict_fn,
    load_checkpoint_for_inference,
)

__all__ = [
    "LoadedCheckpoint",
    "build_greedy_predict_fn",
    "load_checkpoint_for_inference",
]
