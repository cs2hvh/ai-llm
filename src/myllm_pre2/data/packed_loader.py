"""PyTorch adapter for the existing packed-corpus data plane."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor

@dataclass(frozen=True)
class PackedTorchBatch:
    sequence_ids: Tensor
    input_ids: Tensor
    labels: Tensor
    segment_ids: Tensor
    loss_mask: Tensor
    next_sequence_id: int


class PackedTorchDataLoader:
    """Small deterministic batcher over a packed corpus.

    This is intentionally not a ``torch.utils.data.DataLoader`` yet. Exact
    resume is easier to reason about with a simple sequence-id cursor, and the
    TorchTitan integration can wrap this contract later.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        reader: Any | None = None,
        batch_size: int,
        device: str | torch.device = "cpu",
        expected_tokenizer_sha256: str | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if reader is None:
            if root is None:
                raise ValueError("either root or reader must be provided")
            from myllm.data.packed_corpus import PackedCorpusReader

            reader = PackedCorpusReader(root)

        self.reader = reader
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.drop_last = bool(drop_last)

        manifest = getattr(reader, "manifest", None)
        tokenizer_sha = getattr(manifest, "tokenizer_sha256", None)
        if expected_tokenizer_sha256 is not None and tokenizer_sha != expected_tokenizer_sha256:
            raise ValueError(
                "packed corpus tokenizer hash mismatch: "
                f"expected {expected_tokenizer_sha256}, got {tokenizer_sha}"
            )

    def iter_batches(self, *, start_sequence_id: int = 0) -> Iterator[PackedTorchBatch]:
        if start_sequence_id < 0:
            raise ValueError("start_sequence_id must be >= 0")

        sequence_ids: list[int] = []
        inputs: list[list[int]] = []
        labels: list[list[int]] = []
        segments: list[list[int]] = []
        masks: list[list[int]] = []

        sid = start_sequence_id
        while sid < self.reader.total_sequences:
            tokens = [int(token) for token in self.reader.get_sequence(sid)]
            seg_ids = [int(segment) for segment in self.reader.get_segment_ids(sid)]
            if len(tokens) < 2:
                sid += 1
                continue
            input_ids = tokens[:-1]
            label_ids = tokens[1:]
            input_segments = seg_ids[:-1]
            label_segments = seg_ids[1:]
            loss_mask = [
                1 if (a == b and a != -1) else 0
                for a, b in zip(input_segments, label_segments, strict=False)
            ]
            sequence_ids.append(sid)
            inputs.append(input_ids)
            labels.append(label_ids)
            segments.append(input_segments)
            masks.append(loss_mask)
            sid += 1

            if len(inputs) == self.batch_size:
                yield self._make_batch(sequence_ids, inputs, labels, segments, masks, sid)
                sequence_ids, inputs, labels, segments, masks = [], [], [], [], []

        if inputs and not self.drop_last:
            yield self._make_batch(sequence_ids, inputs, labels, segments, masks, sid)

    def _make_batch(
        self,
        sequence_ids: list[int],
        inputs: list[list[int]],
        labels: list[list[int]],
        segments: list[list[int]],
        masks: list[list[int]],
        next_sequence_id: int,
    ) -> PackedTorchBatch:
        return PackedTorchBatch(
            sequence_ids=torch.tensor(sequence_ids, dtype=torch.long, device=self.device),
            input_ids=torch.tensor(inputs, dtype=torch.long, device=self.device),
            labels=torch.tensor(labels, dtype=torch.long, device=self.device),
            segment_ids=torch.tensor(segments, dtype=torch.long, device=self.device),
            loss_mask=torch.tensor(masks, dtype=torch.float32, device=self.device),
            next_sequence_id=next_sequence_id,
        )
