"""Vector and result value objects for search infrastructure."""

from __future__ import annotations

import enum
import math
import numbers
from dataclasses import dataclass, field
from typing import Optional, Sequence

from rag_core.search.sparse_channels import merge_sparse_channels


class ContentType(str, enum.Enum):
    DOCUMENT = "document"
    CODE = "code"


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation (BM25 indices + values)."""

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                "SparseVector.indices and values must have the same length"
            )
        for index in self.indices:
            if not _is_sparse_index(index):
                raise ValueError("SparseVector.indices must be non-negative integers")
        for value in self.values:
            if not _is_finite_real_number(value):
                raise ValueError("SparseVector.values must be finite numbers")


@dataclass(frozen=True)
class TextualRepresentation:
    """Metadata header + content for a chunk, ready for embedding."""

    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class VectorPoint:
    """A point to upsert into the vector store."""

    id: str
    dense_vector: list[float]
    sparse_vector: SparseVector
    payload: dict[str, object]
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.id, "VectorPoint.id")
        _validate_dense_vector(self.dense_vector, "VectorPoint.dense_vector")

    def all_sparse_vectors(self) -> dict[str, SparseVector]:
        """Return sparse vectors keyed by channel name (always includes bm25)."""
        return merge_sparse_channels(self.sparse_vector, self.sparse_vectors)


@dataclass(frozen=True)
class SearchResult:
    """A single search result from any source.

    Example: SearchResult(id="uuid5-hex", text="# Metadata\\n...\\n# Content\\n...",
             score=0.87, content_type="document", source_type="file",
             document_id="doc_123", corpus_id="help_center",
             document_key="docs/report.pdf", title="Q1 Report",
             chunk_index=3, section_title="Introduction")
    """

    id: str
    text: str
    score: float
    content_type: str
    source_type: str
    document_id: Optional[str] = None
    corpus_id: Optional[str] = None
    document_key: Optional[str] = None
    content_sha256: Optional[str] = None
    title: Optional[str] = None
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    document_path: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_word_count: Optional[int] = None
    chunk_token_estimate: Optional[int] = None
    embedding_model: Optional[str] = None
    chunker_strategy: Optional[str] = None
    result_type: Optional[str] = None
    figure_id: Optional[str] = None
    figure_thumbnail_url: Optional[str] = None
    namespace: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_id(self.id, "SearchResult.id")


def _validate_dense_vector(values: Sequence[object], field_name: str) -> None:
    for value in values:
        if not _is_finite_real_number(value):
            raise ValueError(f"{field_name} must contain finite numbers")


def _validate_non_empty_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _is_sparse_index(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Integral)
        and int(value) >= 0
    )


def _is_finite_real_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )
