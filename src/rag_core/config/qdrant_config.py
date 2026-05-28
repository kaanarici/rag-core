from __future__ import annotations

from dataclasses import dataclass


DEFAULT_QDRANT_COLLECTION = "rag_core_chunks"
DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION = True
QDRANT_COLLECTION_ENV = "RAG_CORE_QDRANT_COLLECTION"
QDRANT_DIMENSION_AWARE_COLLECTION_ENV = "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION"
QDRANT_LOCATION_ENV = "RAG_CORE_QDRANT_LOCATION"
QDRANT_URL_ENV = "RAG_CORE_QDRANT_URL"


@dataclass(frozen=True)
class QdrantConfig:
    url: str | None = None
    location: str | None = None
    api_key: str | None = None
    collection: str = DEFAULT_QDRANT_COLLECTION
    dimension_aware_collection: bool = DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION

    def __post_init__(self) -> None:
        if self.api_key is not None:
            object.__setattr__(self, "api_key", self.api_key.strip() or None)
        # Both unset means the caller supplies a vector store directly.
        if self.url and self.location:
            raise ValueError(
                "QdrantConfig requires exactly one of url or location, not both"
            )
