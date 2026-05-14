"""Hugging Face streaming dataset wrapper.

Wraps ``datasets.load_dataset(..., streaming=True)`` with:
    - retry on transient network errors (tenacity)
    - optional ``skip_first`` counter for coarse resume / smoke-test offsets
      (NOT token-exact byte-offset checkpointing — that lives in the offline
      packed-corpus path; see ``src/myllm/data/packed_corpus.py`` §4 / B2)
    - normalisation into our ``Document`` type
    - optional sample limit for smoke tests

Designed to run inside a training pod or a CPU data-prep pod. The HF auth
token is read from ``HF_TOKEN`` env var.

2026-05-12 docstring fix: prior wording claimed "per-shard checkpointing
of byte offsets (resumable on pod failure)". The code is and has always
been a streaming wrapper with retry + a ``skip_first`` index counter.
Token-exact resume comes from the packed-corpus reader, not this module.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from myllm.data.types import Document, DocumentSource
from myllm.utils import get_logger
from myllm.utils.exceptions import DataPipelineError

log = get_logger(__name__)


@dataclass
class HFStreamLoader:
    """Stream documents from a Hugging Face dataset.

    ``text_field`` is the field on each example that contains the raw text.
    ``id_field`` is optional; when absent, a stable id is synthesised from the
    monotonic record index within the shard.
    """

    dataset: str
    category: DocumentSource
    # ``text_field`` can be a string (single field) or a list of strings
    # (concatenated with "\n\n" in listed order). The list form was added
    # 2026-05-12 for sources like stack-exchange where the useful content
    # spans multiple fields (question + accepted answer). Keeping the
    # default a single string preserves back-compat for all existing
    # source configs.
    text_field: str | list[str] = "text"
    id_field: str | None = None
    split: str = "train"
    config_name: str | None = None
    sample_limit: int | None = None
    skip_first: int = 0
    trust_remote_code: bool = False  # mc4 + a few others ship a custom loader script
    revision: str | None = None  # HF dataset commit SHA / version tag — B2 requirement

    def __iter__(self) -> Iterator[Document]:
        return self._iter_with_retry()

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _iter_with_retry(self) -> Iterator[Document]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise DataPipelineError(
                "datasets library not installed; pip install datasets"
            ) from e

        token = os.environ.get("HF_TOKEN")
        log.info(
            "hf_stream_open",
            dataset=self.dataset,
            split=self.split,
            config=self.config_name,
        )
        load_kwargs: dict = {
            "name": self.config_name,
            "split": self.split,
            "streaming": True,
            "token": token,
        }
        if self.trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if self.revision is not None:
            load_kwargs["revision"] = self.revision
        ds = load_dataset(self.dataset, **load_kwargs)

        for idx, row in enumerate(ds):
            if idx < self.skip_first:
                continue
            if self.sample_limit is not None and idx - self.skip_first >= self.sample_limit:
                break
            try:
                if isinstance(self.text_field, list):
                    # Multi-field source: concatenate the listed fields in
                    # order, separated by a blank line. Used for e.g.
                    # stack-exchange where question + answer both carry
                    # useful signal but live in different columns.
                    parts: list[str] = []
                    for field in self.text_field:
                        val = row.get(field)
                        if val is None:
                            continue
                        parts.append(val if isinstance(val, str) else str(val))
                    text = "\n\n".join(parts)
                else:
                    text = row[self.text_field]
            except KeyError as e:
                raise DataPipelineError(
                    f"text_field '{self.text_field}' missing from {self.dataset} row"
                ) from e
            if not isinstance(text, str):
                # Some datasets store text as bytes or as nested objects; coerce or skip.
                text = str(text) if text is not None else ""
            doc_id = (
                str(row[self.id_field])
                if self.id_field and self.id_field in row
                else f"{self.dataset}#{idx}"
            )
            yield Document(
                text=text,
                source=self.category,
                dataset=self.dataset,
                doc_id=doc_id,
                language=row.get("language") if isinstance(row, dict) else None,
                metadata={},
            )
