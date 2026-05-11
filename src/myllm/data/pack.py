"""Sequence packing.

Pretraining sees fixed-length token sequences. Packing concatenates many
shorter documents into single sequences, separated by a configurable token
(typically EOS), so we don't waste compute on padding. This is the standard
GPT-style pretraining recipe.

Variants:
    - ``concat-with-eos``: classic GPT packing. Documents are simply joined
      with EOS; cross-document attention is allowed. Cheapest, slightly
      noisier learning signal.
    - ``concat-with-attn-mask``: also returns a per-sequence attention mask
      that prevents attention from crossing document boundaries. More
      expensive but cleaner. Implemented as a future flag.

This file ships ``concat-with-eos`` only; attn-mask packing lands when the
training step needs it.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from myllm.utils.exceptions import DataPipelineError


@dataclass
class SequencePacker:
    """Pack a stream of variable-length token lists into fixed-length sequences.

    Args:
        sequence_length: target output sequence length (e.g. 4096).
        eos_token_id: token inserted between documents.
        drop_last: if True, the trailing partial sequence is discarded.
            If False, it is padded with ``pad_token_id`` and returned.
        pad_token_id: only used if ``drop_last`` is False.
    """

    sequence_length: int
    eos_token_id: int
    drop_last: bool = True
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise DataPipelineError("sequence_length must be >= 2")

    def pack(self, token_streams: Iterable[Iterable[int]]) -> Iterator[list[int]]:
        """Consume an iterable of token-id sequences; yield packed sequences."""
        buf: list[int] = []
        for tokens in token_streams:
            for t in tokens:
                buf.append(t)
            buf.append(self.eos_token_id)
            while len(buf) >= self.sequence_length:
                yield buf[: self.sequence_length]
                buf = buf[self.sequence_length :]
        if buf and not self.drop_last:
            pad_count = self.sequence_length - len(buf)
            buf = buf + [self.pad_token_id] * pad_count
            yield buf
