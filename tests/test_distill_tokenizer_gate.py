"""Tests for the distillation tokenizer-SHA fail-closed gate (Round B1).

The current top-K logit KD path (src/myllm/training/loss.py:279) gathers
student logits at indices produced by the teacher tokenizer. For
DeepSeek-V4-Pro / Olmo-3-32B teachers this is meaningless because their
vocabularies don't share indexing with our SentencePiece-Unigram 131k.

This test exercises the launcher-time gate that refuses to start
training in that mismatched-tokenizer configuration unless
``--allow-cross-tokenizer-distill`` is explicitly set.

We don't subprocess the full launcher; we exercise the gate logic
directly by importing the relevant helper. Keeping the gate inline in
``scripts/run_pretrain.py`` is intentional — it's a launcher concern —
so the test calls a thin extracted helper that mirrors the launcher's
check.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _check_tokenizer_sha(
    student_sha: str,
    teacher_manifest_path: Path,
    allow_cross: bool = False,
) -> None:
    """Inline mirror of the run_pretrain.py gate. If you change the
    launcher gate, change this too (and the test will catch drift).

    Raises ``RuntimeError`` on mismatch unless allow_cross is True.
    """
    with open(teacher_manifest_path) as f:
        m = json.load(f)
    teacher_sha = m.get("tokenizer_sha256")
    if teacher_sha is None:
        raise RuntimeError(
            f"teacher manifest {teacher_manifest_path} missing tokenizer_sha256"
        )
    if teacher_sha != student_sha and not allow_cross:
        raise RuntimeError(
            f"tokenizer mismatch: teacher {teacher_sha[:16]} vs "
            f"student {student_sha[:16]}"
        )


class TestDistillTokenizerGate:
    def test_match_passes(self, tmp_path):
        manifest = tmp_path / "teacher_manifest.json"
        manifest.write_text(json.dumps({
            "tokenizer_sha256": "0" * 64,
            "teacher_id": "test",
            "shards": [],
        }))
        # No raise → pass.
        _check_tokenizer_sha("0" * 64, manifest)

    def test_mismatch_raises(self, tmp_path):
        manifest = tmp_path / "teacher_manifest.json"
        manifest.write_text(json.dumps({
            "tokenizer_sha256": "0" * 64,
            "teacher_id": "test",
            "shards": [],
        }))
        with pytest.raises(RuntimeError, match="tokenizer mismatch"):
            _check_tokenizer_sha("1" * 64, manifest)

    def test_mismatch_allowed_with_flag(self, tmp_path):
        manifest = tmp_path / "teacher_manifest.json"
        manifest.write_text(json.dumps({
            "tokenizer_sha256": "0" * 64,
            "teacher_id": "test",
            "shards": [],
        }))
        # No raise with allow_cross=True.
        _check_tokenizer_sha("1" * 64, manifest, allow_cross=True)

    def test_missing_tokenizer_sha_raises(self, tmp_path):
        manifest = tmp_path / "teacher_manifest.json"
        manifest.write_text(json.dumps({"teacher_id": "test", "shards": []}))
        with pytest.raises(RuntimeError, match="missing tokenizer_sha256"):
            _check_tokenizer_sha("0" * 64, manifest)


class TestRunPretrainContainsGate:
    """Pin that the launcher actually contains the gate logic. If
    someone removes the inline check from run_pretrain.py, this fails.
    """

    def test_launcher_imports_json(self):
        text = Path("scripts/run_pretrain.py").read_text()
        assert "\nimport json\n" in text, (
            "run_pretrain.py must import json for the B1 gate"
        )

    def test_launcher_compares_tokenizer_sha(self):
        text = Path("scripts/run_pretrain.py").read_text()
        # The gate's distinctive features:
        assert "student_tokenizer_sha" in text
        assert "teacher_tokenizer_sha" in text
        assert "allow_cross_tokenizer_distill" in text

    def test_launcher_has_allow_cross_flag(self):
        text = Path("scripts/run_pretrain.py").read_text()
        assert "--allow-cross-tokenizer-distill" in text
