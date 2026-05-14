from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myllm_pre2.data import PackedTorchDataLoader  # noqa: E402


@dataclass(frozen=True)
class _FakeManifest:
    tokenizer_sha256: str = "tok-sha"


class _FakeReader:
    total_sequences = 5
    manifest = _FakeManifest()

    def get_sequence(self, sequence_id: int):
        base = sequence_id * 10
        return np.array([base, base + 1, base + 2, base + 3, base + 4], dtype=np.uint32)

    def get_segment_ids(self, sequence_id: int):
        return np.array([0, 0, 1, 1, 1], dtype=np.int32)


def test_pre2_packed_loader_batches_and_shifts_labels():
    loader = PackedTorchDataLoader(reader=_FakeReader(), batch_size=2)

    batch = next(loader.iter_batches())

    assert batch.sequence_ids.tolist() == [0, 1]
    assert batch.input_ids.tolist() == [[0, 1, 2, 3], [10, 11, 12, 13]]
    assert batch.labels.tolist() == [[1, 2, 3, 4], [11, 12, 13, 14]]
    assert batch.segment_ids.tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert batch.loss_mask.tolist() == [[1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]
    assert batch.next_sequence_id == 2


def test_pre2_packed_loader_resume_by_sequence_id_is_exact():
    loader = PackedTorchDataLoader(reader=_FakeReader(), batch_size=2)
    first = next(loader.iter_batches())
    resumed = next(loader.iter_batches(start_sequence_id=first.next_sequence_id))

    assert resumed.sequence_ids.tolist() == [2, 3]
    assert resumed.input_ids.tolist()[0] == [20, 21, 22, 23]


def test_pre2_packed_loader_rejects_tokenizer_hash_mismatch():
    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        PackedTorchDataLoader(
            reader=_FakeReader(),
            batch_size=1,
            expected_tokenizer_sha256="other",
        )


def test_pre2_packed_loader_reads_real_packed_corpus_fixture(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("structlog")

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

    loader = PackedTorchDataLoader(
        root=root,
        batch_size=1,
        expected_tokenizer_sha256="tok-sha",
    )
    batch = next(loader.iter_batches())

    assert batch.sequence_ids.tolist() == [0]
    assert batch.input_ids.tolist() == [[1, 2, 3, 4]]
    assert batch.labels.tolist() == [[2, 3, 4, 5]]
    assert batch.segment_ids.tolist() == [[0, 0, 1, 1]]
    assert batch.loss_mask.tolist() == [[1.0, 0.0, 1.0, 1.0]]
