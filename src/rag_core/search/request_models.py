"""Search request, delete, rerank, and stored-record models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from rag_core.search.filters import Filter
from rag_core.search.sparse_channels import merge_sparse_channels
from rag_core.search.vector_models import SparseVector, _validate_dense_vector

if TYPE_CHECKING:
    from rag_core.search.query_plan import QueryPlan


@dataclass(frozen=True)
class StoredDocumentRecord:
    document_id: str
    namespace: str
    corpus_id: str
    document_key: str | None = None
    content_sha256: str | None = None
    processing_version: str | None = None
    chunk_count: int = 0


@dataclass(frozen=True)
class RerankResult:
    """Result of reranking a document."""

    index: int
    score: float
    text: str


@dataclass(frozen=True)
class RerankBudget:
    """Per-query limits for provider-backed reranking."""

    candidate_count: int | None = None
    timeout_seconds: float | None = None
    fallback_on_error: bool = True
    max_output: int | None = None

    def __post_init__(self) -> None:
        if self.candidate_count is not None and (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count <= 0
        ):
            raise ValueError("RerankBudget.candidate_count must be positive")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
            or not math.isfinite(float(self.timeout_seconds))
        ):
            raise ValueError("RerankBudget.timeout_seconds must be positive")
        if self.max_output is not None and (
            isinstance(self.max_output, bool)
            or not isinstance(self.max_output, int)
            or self.max_output <= 0
        ):
            raise ValueError("RerankBudget.max_output must be positive")


@dataclass
class SearchQuery:
    """Parameters for a search query."""

    dense_vector: list[float]
    sparse_vector: SparseVector
    namespace: str
    corpus_ids: list[str]
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    limit: int = 20
    query_plan: "QueryPlan | None" = None
    metadata_filter: Filter | None = None

    def __post_init__(self) -> None:
        _validate_dense_vector(self.dense_vector, "SearchQuery.dense_vector")
        _require_non_blank_string(self.namespace, "SearchQuery.namespace")
        _require_positive_int(self.limit, "SearchQuery.limit")

    def all_sparse_vectors(self) -> dict[str, SparseVector]:
        """Return query sparse vectors keyed by channel name (always includes bm25)."""
        return merge_sparse_channels(self.sparse_vector, self.sparse_vectors)

    def has_empty_allowlist(self) -> bool:
        return (
            self.content_types == []
            or self.document_ids == []
            or self.corpus_ids == []
        )


@dataclass(frozen=True)
class SearchSidecarQuery:
    """Portable search request for optional lexical/exact sidecars."""

    query: str
    namespace: str
    corpus_ids: list[str]
    limit: int = 20
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    metadata_filter: Filter | None = None

    def __post_init__(self) -> None:
        _require_non_blank_string(self.namespace, "SearchSidecarQuery.namespace")
        _require_positive_int(self.limit, "SearchSidecarQuery.limit")


@dataclass
class DeleteFilter:
    """Filter for deleting points from the vector store."""

    namespace: Optional[str] = None
    corpus_id: Optional[str] = None
    document_id: Optional[str] = None


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_blank_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
