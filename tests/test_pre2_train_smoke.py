from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from myllm_pre2.config import load_dense_config  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pre2_train.py"
SPEC = importlib.util.spec_from_file_location("pre2_train", SCRIPT)
pre2_train = importlib.util.module_from_spec(SPEC)
sys.modules["pre2_train"] = pre2_train
SPEC.loader.exec_module(pre2_train)


def test_pre2_synthetic_smoke_runs_one_step():
    base_cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")
    tiny_cfg = pre2_train.make_tiny_smoke_config(base_cfg)

    result = pre2_train.run_synthetic_smoke(
        tiny_cfg,
        steps=1,
        batch_size=1,
        sequence_length=8,
        device="cpu",
    )

    assert result["steps"] == 1
    assert result["device"] == "cpu"
    assert result["last_loss"] > 0
    assert result["tokens_consumed"] == 8
    assert result["precision"] == "fp32"
    assert result["grad_clip_global_norm"] == 1.0
    assert result["last_grad_norm"] > 0
    assert result["last_lr"] == 2.0e-4
    assert result["peak_cuda_memory_mb"] is None
    assert result["parameter_count"] < 1_000_000


def test_pre2_synthetic_smoke_supports_bf16_autocast():
    base_cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")
    tiny_cfg = pre2_train.make_tiny_smoke_config(base_cfg)

    result = pre2_train.run_synthetic_smoke(
        tiny_cfg,
        steps=1,
        batch_size=1,
        sequence_length=8,
        device="cpu",
        precision="bf16",
    )

    assert result["precision"] == "bf16"
    assert result["last_loss"] > 0
    assert result["last_grad_norm"] > 0


def test_pre2_synthetic_smoke_config_precision_uses_training_dtype():
    base_cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")
    tiny_cfg = pre2_train.make_tiny_smoke_config(base_cfg)

    result = pre2_train.run_synthetic_smoke(
        tiny_cfg,
        steps=1,
        batch_size=1,
        sequence_length=8,
        device="cpu",
        precision="config",
    )

    assert result["precision"] == "bf16"


def test_pre2_synthetic_smoke_writes_checkpoint(tmp_path):
    base_cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")
    tiny_cfg = pre2_train.make_tiny_smoke_config(base_cfg)

    result = pre2_train.run_synthetic_smoke(
        tiny_cfg,
        steps=1,
        batch_size=1,
        sequence_length=8,
        device="cpu",
        checkpoint_dir=tmp_path / "ckpt",
    )

    assert result["checkpoint_dir"] == str(tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "checkpoint.pt").exists()
    assert (tmp_path / "ckpt" / "manifest.json").exists()


def test_pre2_synthetic_smoke_resumes_from_checkpoint(tmp_path):
    base_cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")
    tiny_cfg = pre2_train.make_tiny_smoke_config(base_cfg)
    checkpoint_dir = tmp_path / "ckpt"

    first = pre2_train.run_synthetic_smoke(
        tiny_cfg,
        steps=1,
        batch_size=1,
        sequence_length=8,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
    )
    resumed = pre2_train.run_synthetic_smoke(
        tiny_cfg,
        steps=1,
        batch_size=1,
        sequence_length=8,
        device="cpu",
        resume_from_checkpoint=checkpoint_dir,
    )

    assert first["total_steps"] == 1
    assert resumed["start_step"] == 1
    assert resumed["total_steps"] == 2
    assert resumed["tokens_consumed"] == 16
    assert resumed["resumed_from_checkpoint"] == str(checkpoint_dir)
    assert resumed["last_loss"] > 0


def test_pre2_packed_corpus_smoke_runs_one_step(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("structlog")
    np = pytest.importorskip("numpy")

    from myllm.data.packed_corpus import DocSpan, PackedCorpusWriter, write_corpus_manifest

    root = tmp_path / "corpus"
    writer = PackedCorpusWriter(
        root,
        sequence_length=5,
        sequences_per_shard=2,
        tokenizer_sha256="tok-sha",
    )
    writer.append_sequence(
        np.array([1, 2, 3, 4, 5], dtype=np.uint32),
        [
            DocSpan(
                doc_span_id=-1,
                sequence_id=-1,
                source_id="source_a",
                doc_id_hash=1,
                dataset_revision_id="rev",
                token_start_in_sequence=0,
                token_end_in_sequence=2,
                text_hash=11,
            ),
            DocSpan(
                doc_span_id=-1,
                sequence_id=-1,
                source_id="source_b",
                doc_id_hash=2,
                dataset_revision_id="rev",
                token_start_in_sequence=2,
                token_end_in_sequence=5,
                text_hash=22,
            ),
        ],
    )
    writer.close()
    write_corpus_manifest(
        root,
        corpus_name="tiny-real",
        tokenizer_sha256="tok-sha",
        sequence_length=5,
        sequences_per_shard=2,
        source_revisions={"source_a": "rev", "source_b": "rev"},
        target_source_share={"source_a": 0.5, "source_b": 0.5},
    )

    base_cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")
    tiny_cfg = pre2_train.make_tiny_smoke_config(base_cfg)

    result = pre2_train.run_packed_corpus_smoke(
        tiny_cfg,
        packed_corpus_root=root,
        steps=1,
        batch_size=1,
        device="cpu",
        expected_tokenizer_sha256="tok-sha",
    )

    assert result["mode"] == "packed_corpus"
    assert result["steps"] == 1
    assert result["next_sequence_id"] == 1
    assert result["tokens_consumed"] == 4
    assert result["precision"] == "fp32"
    assert result["last_grad_norm"] > 0
    assert result["last_loss"] > 0
