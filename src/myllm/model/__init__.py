"""Keras-3 decoder-only transformer (RoPE, RMSNorm, GQA, SwiGLU).

Importing this package pulls in only ``ModelConfig`` (pure Python). The
Keras-dependent layer/model classes are exposed via ``__getattr__`` so
``import myllm.model`` works in environments that don't have Keras
installed (e.g. CPU-only orchestration tooling, smoke-tests).
"""
from __future__ import annotations

from typing import Any

from myllm.model.config import ModelConfig

__all__ = [
    "ModelConfig",
    "TransformerLM",
    "build_model",
    "causal_mask",
    "RMSNorm",
    "GroupedQueryAttention",
    "SwiGLUFFN",
    "DecoderBlock",
    "precompute_rope_cache",
    "apply_rope",
]


_LAZY = {
    "TransformerLM": ("myllm.model.transformer", "TransformerLM"),
    "build_model": ("myllm.model.transformer", "build_model"),
    "causal_mask": ("myllm.model.transformer", "causal_mask"),
    "RMSNorm": ("myllm.model.layers", "RMSNorm"),
    "GroupedQueryAttention": ("myllm.model.layers", "GroupedQueryAttention"),
    "SwiGLUFFN": ("myllm.model.layers", "SwiGLUFFN"),
    "DecoderBlock": ("myllm.model.layers", "DecoderBlock"),
    "precompute_rope_cache": ("myllm.model.layers", "precompute_rope_cache"),
    "apply_rope": ("myllm.model.layers", "apply_rope"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module 'myllm.model' has no attribute {name!r}")
