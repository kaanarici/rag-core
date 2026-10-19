"""Provider and vector-store protocols for search infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.request_models import (
    DeleteFilter,
    RerankResult,
    SearchQuery,
    StoredDocumentRecord,
)
from rag_core.search.vector_models import SearchResult, SparseVector, VectorPoint

ProviderHealth = dict[str, object]


def provider_name(provider: object | None) -> str:
    if provider is None:
        return "none"
    name = getattr(provider, "provider_name", None)
    if isinstance(name, str) and name:
        return name
    return type(provider).__name__


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for dense embedding providers (OpenAI, Voyage, etc.)."""

    @property
    def dimensions(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class ProviderHealthCheck(Protocol):
    async def check_health(self) -> ProviderHealth: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Sparse encoder with a stable model/vocabulary identity for ingest reuse."""

    @property
    def model_name(self) -> str: ...

    def embed_texts(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, query: str) -> SparseVector: ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Protocol for reranking providers such as Cohere and Voyage."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RerankResult]: ...


@dataclass(frozen=True)
class MetadataFilterCapabilities:
    """Metadata filter shapes a vector store can translate faithfully."""

    term: bool = False
    in_: bool = False
    numeric_range: bool = False
    string_range: bool = False
    geo: bool = False
    boolean: bool = False


@dataclass(frozen=True)
class StoreCapabilities:
    """What a ``VectorStore`` adapter promises to support.

    Each flag corresponds to a method on ``VectorStore`` whose semantics are
    optional. Callers gate on these flags rather than ``isinstance`` so a
    third-party adapter can declare partial support without inheriting an
    implementation marker.

    Flags are added alongside the engine code that consumes them.
    """

    per_point_delete: bool
    dense_vector_dimensions: int | None = None
    hybrid: bool = False
    lexical: bool = False
    native_lexical: bool = False
    metadata_filter: MetadataFilterCapabilities = field(
        default_factory=MetadataFilterCapabilities
    )


@runtime_checkable
class VectorStore(Protocol):
    """Vendor-neutral baseline every vector store adapter must satisfy.

    Per-point deletion is gated by ``capabilities.per_point_delete``.
    Document publication, lookup, scoped search and scoped deletion are required.
    Raw upsert is a transport primitive; managed ingest uses replace_document.
    """

    @property
    def capabilities(self) -> StoreCapabilities: ...

    def bind_embedding_space(self, model_name: str) -> None:
        """Bind the adapter to one model; readiness must reject incompatible storage."""
        ...

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def replace_document(
        self, record: StoredDocumentRecord, points: Sequence[VectorPoint]
    ) -> None:
        """Publish a complete document without overwriting its live chunks.

        Failed staging preserves the previous version. A publication error can
        mean a lost acknowledgement; callers must reconcile rather than delete.
        Search exposes only published versions. Lookups report actual and
        expected chunk counts so incomplete versions cannot be skipped.
        """
        ...

    async def search(self, query: SearchQuery) -> list[SearchResult]: ...

    async def delete(self, filter: DeleteFilter) -> None: ...

    async def ensure_collection(self) -> None: ...

    async def check_health(self) -> ProviderHealth: ...

    async def close(self) -> None: ...

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None: ...

    async def get_document_record(
        self,
        *,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None: ...
