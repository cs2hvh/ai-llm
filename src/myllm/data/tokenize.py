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
    packed_sequences: Iterable[list[int]],
) -> Iterator[tuple[list[int], list[int]]]:
    """Convert packed sequences into ``(input_ids, labels)`` pairs.

    For next-token prediction, ``labels`` is ``input_ids`` shifted by one:
    given a packed ``[t0, t1, ..., t_{N-1}]`` we emit
    ``input=[t0..t_{N-2}], label=[t1..t_{N-1}]``.
    """
    for seq in packed_sequences:
        if len(seq) < 2:
            continue
        yield seq[:-1], seq[1:]
