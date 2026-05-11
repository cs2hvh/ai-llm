"""Streaming loaders, filters, dedupe, tokenize, mixture sampling, sequence packing."""

from myllm.data.filters import (
    Filter,
    FilterChain,
    LengthFilter,
    PIIRedactor,
    RepetitionFilter,
    SymbolRatioFilter,
)
from myllm.data.mixture import MixtureSampler, SourceWeight
from myllm.data.pack import SequencePacker
from myllm.data.special_tokens import (
    OPTIONAL as OPTIONAL_SPECIAL_TOKENS,
    REQUIRED as REQUIRED_SPECIAL_TOKENS,
    SpecialTokens,
    all_special_token_strings,
    get_required_ids,
    verify_tokenizer_has_required,
)
from myllm.data.synthetic import make_synthetic_data_iter
from myllm.data.tokenize import load_tokenizer, make_input_label_pairs, tokenize_documents
from myllm.data.types import (
    Document,
    DocumentSource,
    FilterDecision,
    ProcessedShardManifest,
    ShardSpec,
)

__all__ = [
    # types
    "Document",
    "DocumentSource",
    "FilterDecision",
    "ShardSpec",
    "ProcessedShardManifest",
    # filters
    "Filter",
    "FilterChain",
    "LengthFilter",
    "RepetitionFilter",
    "SymbolRatioFilter",
    "PIIRedactor",
    # mixture & packing
    "MixtureSampler",
    "SourceWeight",
    "SequencePacker",
    # tokenize
    "load_tokenizer",
    "tokenize_documents",
    "make_input_label_pairs",
    # synthetic
    "make_synthetic_data_iter",
    # special tokens
    "SpecialTokens",
    "REQUIRED_SPECIAL_TOKENS",
    "OPTIONAL_SPECIAL_TOKENS",
    "all_special_token_strings",
    "verify_tokenizer_has_required",
    "get_required_ids",
]
