"""Orchestration tests for scripts/cache_teacher_logits.py.

Validates the synthetic-teacher path end-to-end:
  - Reads a small synthetic corpus from disk.
  - Generates shards through the script's main pipeline.
  - Verifies the resulting manifest covers the requested range.
  - Verifies that re-running on the same output dir is idempotent (resumes).

Does NOT validate the real-teacher path — that requires vLLM + a real
model, which is out of scope for unit tests.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO / "scripts" / "cache_teacher_logits.py"

# Import the script as a module so we can call its public functions
# without going through subprocess.
spec = importlib.util.spec_from_file_location("cache_teacher_logits", _SCRIPT_PATH)
_cache_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_cache_script)

from myllm.data.teacher_cache import read_manifest, read_shard  # noqa: E402


def _make_synthetic_corpus(tmp_path: Path, n_tokens: int = 200) -> Path:
    """Write a small uint32 corpus file."""
    path = tmp_path / "corpus.bin"
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 131072, size=n_tokens, dtype="uint32")
    arr.tofile(path)
    return path


def test_generate_cache_writes_shards_and_manifest(tmp_path):
    corpus = _make_synthetic_corpus(tmp_path, n_tokens=200)
    output_dir = tmp_path / "cache_out"

    manifest = _cache_script.generate_teacher_cache(
        teacher_id="test-teacher",
        teacher_hf_model="test/fake-hf-id",
        tokenized_corpus=corpus,
        corpus_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        output_dir=output_dir,
        top_k=4,
        shard_tokens=64,
        batch_size=4,
        sequence_length=16,
        synthetic_teacher=True,
    )

    # Manifest covers the full corpus range.
    assert manifest.teacher_id == "test-teacher"
    assert manifest.top_k == 4
    assert manifest.total_tokens() == 200
    assert len(manifest.shards) >= 1

    # All shards exist on disk + payload round-trips.
    for entry in manifest.shards:
        path = output_dir / entry.r2_key
        assert path.exists()
        shard = read_shard(path)
        assert shard.teacher_id == "test-teacher"
        assert shard.top_k == 4
        assert shard.logits.shape == (
            entry.end_token_position - entry.start_token_position, 4
        )
        assert shard.indices.shape == shard.logits.shape


def test_generate_cache_is_resumable(tmp_path):
    """A second invocation must NOT re-process already-written shards."""
    corpus = _make_synthetic_corpus(tmp_path, n_tokens=300)
    output_dir = tmp_path / "cache_out"

    # First pass: cache part of the corpus.
    m1 = _cache_script.generate_teacher_cache(
        teacher_id="test-teacher",
        teacher_hf_model="x",
        tokenized_corpus=corpus,
        corpus_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        output_dir=output_dir,
        top_k=2,
        shard_tokens=50,
        end_token=100,
        batch_size=2,
        sequence_length=8,
        synthetic_teacher=True,
    )
    n_shards_after_first = len(m1.shards)
    tokens_after_first = m1.total_tokens()
    assert tokens_after_first == 100

    # Second pass: continue to end-of-corpus from start_token=0; resumability
    # should detect existing manifest covers 0..100 and start at 100.
    m2 = _cache_script.generate_teacher_cache(
        teacher_id="test-teacher",
        teacher_hf_model="x",
        tokenized_corpus=corpus,
        corpus_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        output_dir=output_dir,
        top_k=2,
        shard_tokens=50,
        end_token=300,
        batch_size=2,
        sequence_length=8,
        synthetic_teacher=True,
    )
    # All 300 tokens now covered.
    assert m2.total_tokens() == 300
    # And we appended without overwriting the first batch's shards.
    assert len(m2.shards) > n_shards_after_first


def test_two_teachers_coexist_in_same_output_dir(tmp_path):
    """Each teacher writes its own manifest file ({teacher_id}_manifest.json),
    so two different teachers can share an output_dir without collisions.
    """
    corpus = _make_synthetic_corpus(tmp_path, n_tokens=100)
    output_dir = tmp_path / "cache_out"

    m_a = _cache_script.generate_teacher_cache(
        teacher_id="teacher-A",
        teacher_hf_model="x",
        tokenized_corpus=corpus,
        corpus_sha256="0" * 64,
        tokenizer_sha256="0" * 64,
        output_dir=output_dir,
        top_k=2,
        shard_tokens=50,
        batch_size=2,
        sequence_length=8,
        synthetic_teacher=True,
    )
    m_b = _cache_script.generate_teacher_cache(
        teacher_id="teacher-B",
        teacher_hf_model="x",
        tokenized_corpus=corpus,
        corpus_sha256="0" * 64,
        tokenizer_sha256="0" * 64,
        output_dir=output_dir,
        top_k=2,
        shard_tokens=50,
        batch_size=2,
        sequence_length=8,
        synthetic_teacher=True,
    )
    assert m_a.teacher_id == "teacher-A"
    assert m_b.teacher_id == "teacher-B"
    assert (output_dir / "teacher-A_manifest.json").exists()
    assert (output_dir / "teacher-B_manifest.json").exists()


def test_generate_cache_rejects_corrupted_manifest_with_wrong_teacher_id(tmp_path):
    """If a manifest file's *internal* teacher_id doesn't match the
    teacher_id we're trying to write (e.g. file was manually renamed or
    corrupted), the script must refuse to overwrite it.
    """
    import json

    corpus = _make_synthetic_corpus(tmp_path, n_tokens=100)
    output_dir = tmp_path / "cache_out"
    output_dir.mkdir(parents=True)

    # Hand-craft a manifest file for "teacher-Z" but named for "teacher-A"
    # — simulates a corrupted / mis-renamed file on disk.
    (output_dir / "teacher-A_manifest.json").write_text(json.dumps({
        "format_version": 1,
        "teacher_id": "teacher-Z",
        "corpus_sha256": "0" * 64,
        "tokenizer_sha256": "0" * 64,
        "top_k": 2,
        "total_tokens": 0,
        "shards": [],
    }))

    with pytest.raises(ValueError, match="refusing to mix"):
        _cache_script.generate_teacher_cache(
            teacher_id="teacher-A",
            teacher_hf_model="x",
            tokenized_corpus=corpus,
            corpus_sha256="0" * 64,
            tokenizer_sha256="0" * 64,
            output_dir=output_dir,
            top_k=2,
            shard_tokens=50,
            batch_size=2,
            sequence_length=8,
            synthetic_teacher=True,
        )


def test_real_teacher_path_raises_not_implemented(tmp_path):
    """The real-teacher path must explicitly fail with a guiding error
    until vLLM integration lands — not silently fall back to fake data.
    """
    corpus = _make_synthetic_corpus(tmp_path, n_tokens=20)
    with pytest.raises(NotImplementedError, match="vLLM"):
        _cache_script.generate_teacher_cache(
            teacher_id="real-teacher",
            teacher_hf_model="org/real-model",
            tokenized_corpus=corpus,
            corpus_sha256="0" * 64,
            tokenizer_sha256="0" * 64,
            output_dir=tmp_path / "cache",
            top_k=2,
            shard_tokens=50,
            batch_size=2,
            sequence_length=8,
            synthetic_teacher=False,
        )
