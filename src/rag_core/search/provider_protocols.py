"""Provider and vector-store protocols for search infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from rag_core.retrieval_defaults import DEFAULT_SEARCH_LIMIT
from rag_core.search.request_models import (
    DeleteFilter,
    RerankResult,
    SearchQuery,
    SearchSidecarQuery,
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
    """Protocol for sparse embedding (BM25 via FastEmbed)."""

    def embed_texts(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, query: str) -> SparseVector: ...

    def embed_query_multi(self, query: str) -> dict[str, SparseVector]: ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Protocol for reranking providers such as Cohere, Voyage, and ZeroEntropy."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[RerankResult]: ...


@runtime_checkable
class SearchSidecar(Protocol):
    """Protocol for optional lexical/exact-match sidecars."""

    def upsert_records(self, records: Sequence[object]) -> None: ...

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        collection: str | None = None,
    ) -> None: ...

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]: ...


@dataclass(frozen=True)
class QueryPlanCapabilities:
    """Query-plan stages a vector store can execute without silent downgrade."""

    dense: bool = False
    sparse: bool = False
    hybrid_rrf: bool = False
    hybrid_dbsf: bool = False
    hybrid_weighted_rrf: bool = False
    mmr: bool = False
    boost: bool = False
    nested_prefetch: bool = False

    @property
    def hybrid(self) -> bool:
        return self.hybrid_rrf or self.hybrid_dbsf or self.hybrid_weighted_rrf


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
    document_record_lookup: bool
    chunk_index_lookup: bool = False
    dense_vector_dimensions: int | None = None
    query_plan: QueryPlanCapabilities = field(default_factory=QueryPlanCapabilities)
    metadata_filter: MetadataFilterCapabilities = field(
        default_factory=MetadataFilterCapabilities
    )


@runtime_checkable
class VectorStore(Protocol):
    """Vendor-neutral baseline every vector store adapter must satisfy.

    Methods ``delete_point_ids``, ``get_document_record``, and
    ``get_chunks_by_index`` are conditionally supported: callers must check
    ``capabilities`` before invoking them. An adapter that declares the
    matching capability flag as ``False`` may raise ``NotImplementedError``
    from the corresponding method.
    """

    @property
    def capabilities(self) -> StoreCapabilities: ...

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def search(self, query: SearchQuery) -> list[SearchResult]: ...

    async def delete(self, filter: DeleteFilter) -> None: ...

    async def ensure_collection(self) -> None: ...

    async def check_health(self) -> ProviderHealth: ...

    async def close(self) -> None: ...

    async def delete_point_ids(self, point_ids: Sequence[str]) -> None: ...

    async def get_document_record(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
    ) -> StoredDocumentRecord | None: ...

    async def get_chunks_by_index(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        chunk_indices: Sequence[int],
    ) -> list[SearchResult]: ...
