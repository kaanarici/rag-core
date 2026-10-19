"""Scoped vector queries, deletion, rerank budgets, and stored document identity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.filters import Filter
from rag_core.search.vector_models import SparseVector, _validate_dense_vector


SearchMode = Literal["dense", "hybrid", "lexical"]
SEARCH_MODES = ("dense", "hybrid", "lexical")


def validate_search_mode(mode: str) -> None:
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}")


@dataclass(frozen=True)
class StoredDocumentRecord:
    document_id: str
    tenant_id: str
    collection: str
    document_key: str | None = None
    content_sha256: str | None = None
    ingest_fingerprint: str | None = None
    processing_version: str | None = None
    chunk_count: int = 0
    revision: str | None = None
    expected_chunk_count: int | None = None
    committed: bool = False
    deleted: bool = False
    retired_revisions: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return (
            self.committed
            and not self.deleted
            and self.revision is not None
            and self.expected_chunk_count == self.chunk_count
        )


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
    fallback_on_error: bool = False
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
    """Vector-store execution query after embedding and pipeline planning."""

    dense_vector: list[float]
    tenant_id: str
    collections: list[str]
    sparse_vectors: dict[str, SparseVector] = field(default_factory=dict)
    content_types: Optional[list[str]] = None
    document_ids: Optional[list[str]] = None
    limit: int = DEFAULT_SEARCH_LIMIT
    metadata_filter: Filter | None = None
    lexical_query: str | None = None
    mode: SearchMode = "dense"
    candidate_limit: int | None = None
    excluded_revisions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_search_mode(self.mode)
        _validate_dense_vector(self.dense_vector, "SearchQuery.dense_vector")
        _require_non_blank_string(self.tenant_id, "SearchQuery.tenant_id")
        _require_non_blank_string_items(self.collections, "SearchQuery.collections")
        _require_non_blank_string_items(self.content_types, "SearchQuery.content_types")
        _require_non_blank_string_items(self.document_ids, "SearchQuery.document_ids")
        _require_positive_int(self.limit, "SearchQuery.limit")
        if self.candidate_limit is None:
            self.candidate_limit = (
                max(20, self.limit * 4) if self.mode == "hybrid" else self.limit
            )
        _require_positive_int(self.candidate_limit, "SearchQuery.candidate_limit")
        if self.candidate_limit < self.limit:
            raise ValueError("SearchQuery.candidate_limit must be at least limit")
        if self.lexical_query is not None and not isinstance(self.lexical_query, str):
            raise ValueError("SearchQuery.lexical_query must be a string")

    def has_empty_allowlist(self) -> bool:
        return (
            self.content_types == []
            or self.document_ids == []
            or self.collections == []
        )


@dataclass
class DeleteFilter:
    """Filter for deleting points from the vector store.

    Tier-widening footgun guard: an empty-string ``document_id`` or
    ``collection`` previously silently widened the delete (e.g. a caller
    intended to remove a single document, mis-formatted the id, and ended
    up clearing the collection or tenant_id). Empty strings are now rejected;
    callers wanting a collection-wide or tenant-wide delete must pass
    ``None`` deliberately (and the engine routes those through the
    explicit ``delete_collection`` / ``delete_tenant`` helpers on
    ``Engine``).
    """

    tenant_id: Optional[str] = None
    collection: Optional[str] = None
    document_id: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            self.tenant_id is None
            or not isinstance(self.tenant_id, str)
            or not self.tenant_id.strip()
        ):
            raise ValueError("DeleteFilter.tenant_id must be a non-empty string")
        if self.collection is not None and (
            not isinstance(self.collection, str) or not self.collection.strip()
        ):
            raise ValueError(
                "DeleteFilter.collection must be omitted (None) or a non-empty string; "
                "empty string would silently widen the delete"
            )
        if self.document_id is not None and (
            not isinstance(self.document_id, str) or not self.document_id.strip()
        ):
            raise ValueError(
                "DeleteFilter.document_id must be omitted (None) or a non-empty string; "
                "empty string would silently widen the delete"
            )
        if self.collection is None and self.document_id is None:
            raise ValueError(
                "DeleteFilter requires at least one of collection or document_id; "
                "use delete_tenant() for a deliberate tenant-wide delete"
            )


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_blank_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_blank_string_items(
    values: list[str] | None,
    field_name: str,
) -> None:
    if values is None:
        return
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must contain non-empty strings")
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
