"""Provider-author search contracts and shared result value objects."""

from __future__ import annotations

from rag_core.search.filters import (
    And as And,
    Filter as Filter,
    Geo as Geo,
    In as In,
    Not as Not,
    Or as Or,
    Range as Range,
    Term as Term,
)
from rag_core.search.provider_protocols import (
    EmbeddingProvider as EmbeddingProvider,
    MetadataFilterCapabilities as MetadataFilterCapabilities,
    QueryPlanCapabilities as QueryPlanCapabilities,
    RerankerProvider as RerankerProvider,
    SearchSidecar as SearchSidecar,
    SparseEmbedder as SparseEmbedder,
    StoreCapabilities as StoreCapabilities,
    VectorStore as VectorStore,
)
from rag_core.search.request_models import (
    DeleteFilter as DeleteFilter,
    RerankBudget as RerankBudget,
    RerankResult as RerankResult,
    SearchQuery as SearchQuery,
    SearchSidecarQuery as SearchSidecarQuery,
    StoredDocumentRecord as StoredDocumentRecord,
)
from rag_core.search.vector_models import (
    ContentType as ContentType,
    SEARCH_RESULT_TYPE_TEXT as SEARCH_RESULT_TYPE_TEXT,
    SearchResult as SearchResult,
    SparseVector as SparseVector,
    TextualRepresentation as TextualRepresentation,
    VectorPoint as VectorPoint,
)

__all__ = (
    "And",
    "ContentType",
    "DeleteFilter",
    "EmbeddingProvider",
    "Filter",
    "Geo",
    "In",
    "MetadataFilterCapabilities",
    "Not",
    "Or",
    "QueryPlanCapabilities",
    "Range",
    "RerankBudget",
    "RerankResult",
    "RerankerProvider",
    "SEARCH_RESULT_TYPE_TEXT",
    "SearchQuery",
    "SearchResult",
    "SearchSidecar",
    "SearchSidecarQuery",
    "SparseEmbedder",
    "SparseVector",
    "StoreCapabilities",
    "StoredDocumentRecord",
    "Term",
    "TextualRepresentation",
    "VectorPoint",
    "VectorStore",
)
