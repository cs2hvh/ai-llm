"""Sequence packing.

Pretraining sees fixed-length token sequences. Packing concatenates many
shorter documents into single sequences, separated by a configurable token
(typically EOS), so we don't waste compute on padding. This is the standard
GPT-style pretraining recipe.

The packer emits both tokens AND segment_ids (a parallel int array where
each token carries its document index within the pack). Segment_ids serve
two purposes downstream:

  1. Intra-document attention masking (R2): the model's attention layer
     uses segment_ids to build a mask that prevents tokens in document A
     from attending to tokens in document B even when they sit in the
     same packed sequence.

  2. Loss masking at boundaries: when `input[i]` is the last token of doc
     A and `label[i]` is the first token of doc B, predicting doc B's
     first token from doc A's content is impossible, so we mask the loss
     at those positions. ``make_input_label_pairs`` returns the loss_mask
     alongside input/label.

2026-05-12 audit P0-2 fix: previously the packer emitted token lists only
and downstream code never produced segment_ids, so the model's segment_ids
support was unused — packed documents DID attend across boundaries despite
the dossier claiming R2 was implemented.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from myllm.utils.exceptions import DataPipelineError


@dataclass(frozen=True)
class PackedSequence:
    """A packed sequence of tokens plus per-token document indices.

    Both lists have length ``sequence_length``. ``segment_ids[i]`` is the
    document index (0, 1, 2, ...) within this packed sequence that token
    ``tokens[i]`` belongs to. The EOS token between documents is
    conventionally counted as the END of the preceding document (same
    segment_id), since EOS marks the boundary.
    """

    tokens: list[int]
    segment_ids: list[int]


@dataclass
class SequencePacker:
    """Pack a stream of variable-length token lists into fixed-length sequences.

    Args:
        sequence_length: target output sequence length (e.g. 4096).
        eos_token_id: token inserted between documents.
        drop_last: if True, the trailing partial sequence is discarded.
            If False, it is padded with ``pad_token_id`` and returned.
        pad_token_id: only used if ``drop_last`` is False.

    Yields ``PackedSequence`` objects. Use ``.tokens`` for the legacy
    token-list view.
    """

    sequence_length: int
    eos_token_id: int
    drop_last: bool = True
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise DataPipelineError("sequence_length must be >= 2")

    def pack(
        self, token_streams: Iterable[Iterable[int]]
    ) -> Iterator[PackedSequence]:
        """Consume an iterable of token-id sequences; yield ``PackedSequence``."""
        buf: list[int] = []
        seg_buf: list[int] = []
        doc_idx = 0  # increments at every document boundary (EOS)
        for tokens in token_streams:
            had_tokens = False
            for t in tokens:
                buf.append(t)
                seg_buf.append(doc_idx)
                had_tokens = True
            if had_tokens:
                # EOS belongs to the same segment as the document it ends.
                buf.append(self.eos_token_id)
                seg_buf.append(doc_idx)
                doc_idx += 1
            while len(buf) >= self.sequence_length:
                yield PackedSequence(
                    tokens=buf[: self.sequence_length],
                    segment_ids=seg_buf[: self.sequence_length],
                )
                buf = buf[self.sequence_length :]
                seg_buf = seg_buf[self.sequence_length :]
        if buf and not self.drop_last:
            pad_count = self.sequence_length - len(buf)
            # Padding tokens get a sentinel segment_id (-1) so consumers can
            # detect and ignore them. They MUST NOT share a segment with
            # any real document, or attention/loss masks would let them
            # influence real positions.
            buf = buf + [self.pad_token_id] * pad_count
            seg_buf = seg_buf + [-1] * pad_count
            yield PackedSequence(tokens=buf, segment_ids=seg_buf)
