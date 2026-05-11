"""Tokenize documents into integer token streams.

Loads a HuggingFace ``tokenizers.Tokenizer`` from a saved ``tokenizer.json``
and converts a stream of ``Document`` objects into integer-token lists,
ready for ``SequencePacker``.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from myllm.data.types import Document
from myllm.utils import get_logger
from myllm.utils.exceptions import DataPipelineError

log = get_logger(__name__)


def load_tokenizer(path: str | Path) -> Any:
    """Load a tokenizers.Tokenizer from a saved ``tokenizer.json`` file."""
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError(
            "tokenizers not installed; pip install tokenizers"
        ) from e
    p = Path(path)
    if not p.exists():
        raise DataPipelineError(f"tokenizer file not found: {p}")
    log.info("tokenizer_load", path=str(p))
    return Tokenizer.from_file(str(p))


def tokenize_documents(
    docs: Iterable[Document],
    tokenizer: Any,
) -> Iterator[list[int]]:
    """Tokenize each document's text; yield list[int]. Empty results dropped."""
    for doc in docs:
        if not doc.text:
            continue
        ids = tokenizer.encode(doc.text).ids
        if ids:
            yield ids


def make_input_label_pairs(
    packed_sequences: Iterable[Any],
) -> Iterator[tuple[list[int], list[int], list[int], list[int]]]:
    """Convert packed sequences into ``(input_ids, labels, segment_ids, loss_mask)``.

    For next-token prediction, ``labels`` is ``input_ids`` shifted by one:
    given a packed ``[t0, t1, ..., t_{N-1}]`` we emit
    ``input=[t0..t_{N-2}], label=[t1..t_{N-1}]``.

    Accepts either:
      - ``PackedSequence`` objects (from the current ``SequencePacker.pack``;
        carry ``tokens`` + ``segment_ids``)
      - bare ``list[int]`` (legacy / synthetic-data paths; segment_ids are
        synthesized as all-zero meaning "one document per pack")

    Returns four parallel lists of length ``len(seq) - 1``:
      - ``input_ids``:   input tokens
      - ``labels``:      next-token targets
      - ``segment_ids``: doc-index for each INPUT position (used by the
        model's intra-doc attention mask)
      - ``loss_mask``:   1 where the label is in the same document as the
        input (so the next-token prediction is meaningful), 0 at document
        boundaries and at padding positions (segment_id == -1).
    """
    for item in packed_sequences:
        # Support both PackedSequence and bare list[int] for back-compat.
        if hasattr(item, "tokens") and hasattr(item, "segment_ids"):
            tokens = item.tokens
            seg_ids = item.segment_ids
        else:
            tokens = list(item)
            seg_ids = [0] * len(tokens)  # synthetic: single segment per pack
        if len(tokens) < 2:
            continue
        input_ids = tokens[:-1]
        labels = tokens[1:]
        input_segments = seg_ids[:-1]
        label_segments = seg_ids[1:]
        # Loss is computed only where input + label share a segment AND
        # neither is padding (-1). At doc boundaries the prediction is
        # meaningless (can't predict doc B's first token from doc A's content).
        loss_mask = [
            1 if (a == b and a != -1) else 0
            for a, b in zip(input_segments, label_segments)
        ]
        yield input_ids, labels, input_segments, loss_mask
