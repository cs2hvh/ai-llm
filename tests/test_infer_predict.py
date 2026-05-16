"""Tests for the shared inference path (Round B4, 2026-05-16).

These tests verify the module surface (imports, function signatures,
mock-predict wiring) without spinning up a real model. End-to-end
checkpoint load + decode requires a saved Orbax pytree and is
exercised by scripts/eval_checkpoint.py + scripts/generate.py via
manual smoke runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import pytest


class TestInferModuleSurface:
    """The shared module exports what callers expect."""

    def test_imports_present(self):
        from myllm.infer import (
            LoadedCheckpoint,
            build_greedy_predict_fn,
            load_checkpoint_for_inference,
        )
        assert LoadedCheckpoint is not None
        assert build_greedy_predict_fn is not None
        assert load_checkpoint_for_inference is not None

    def test_loaded_checkpoint_is_dataclass(self):
        from myllm.infer import LoadedCheckpoint
        # Has all the named fields the consumer (scorecard / scripts) reads.
        assert {
            "model", "trainable", "non_trainable", "model_cfg", "tokenizer",
            "forward_jit", "ctx_length", "pad_id", "eos_id", "step",
        }.issubset(LoadedCheckpoint.__dataclass_fields__.keys())

    def test_load_checkpoint_raises_on_missing_root(self, tmp_path):
        from myllm.infer import load_checkpoint_for_inference
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_checkpoint_for_inference(
                model_config_path=Path("configs/pilot_250m.yaml"),
                tokenizer_path=Path("artifacts/tokenizer_v1.json"),
                checkpoint_root=missing,
            )

    def test_load_checkpoint_raises_when_no_complete_step(self, tmp_path):
        from myllm.infer import load_checkpoint_for_inference
        empty = tmp_path / "ckpt"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="no complete checkpoint"):
            load_checkpoint_for_inference(
                model_config_path=Path("configs/pilot_250m.yaml"),
                tokenizer_path=Path("artifacts/tokenizer_v1.json"),
                checkpoint_root=empty,
            )


class TestScorecardWiring:
    """The scorecard's _build_predict_fn picks the right path."""

    def test_mock_predict_returns_constant_a(self):
        # Re-implement the mock path inline so we don't have to subprocess
        # the launcher just to verify the trivial mock.
        import argparse
        from scripts.build_release_scorecard import _build_predict_fn
        args = argparse.Namespace(
            use_mock_predict=True,
            checkpoint_root=None,
        )
        predict = _build_predict_fn(args)
        assert predict("anything") == "A"
        assert predict("Question: 2+2=?\nAnswer:") == "A"

    def test_real_predict_requires_checkpoint_root(self):
        import argparse
        from scripts.build_release_scorecard import _build_predict_fn
        args = argparse.Namespace(
            use_mock_predict=False,
            checkpoint_root=None,
        )
        with pytest.raises(ValueError, match="--checkpoint-root"):
            _build_predict_fn(args)

    def test_scorecard_no_longer_raises_notimplemented(self):
        # Round B4 removed the NotImplementedError. The scorecard now
        # either calls into infer/predict.py (real ckpt path) or returns
        # the mock. If a future refactor reintroduces a hard-block, this
        # test catches it.
        import inspect
        from scripts.build_release_scorecard import _build_predict_fn
        src = inspect.getsource(_build_predict_fn)
        assert "NotImplementedError" not in src, (
            "_build_predict_fn re-introduced NotImplementedError; the "
            "scorecard is back to scaffold-only. See Round B4."
        )
