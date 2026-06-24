from __future__ import annotations

from dataclasses import dataclass


DEFAULT_QDRANT_COLLECTION = "rag_core_chunks"
DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION = True
QDRANT_COLLECTION_ENV = "RAG_CORE_QDRANT_COLLECTION"
QDRANT_DIMENSION_AWARE_COLLECTION_ENV = "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION"
QDRANT_LOCATION_ENV = "RAG_CORE_QDRANT_LOCATION"
QDRANT_URL_ENV = "RAG_CORE_QDRANT_URL"
_QDRANT_TARGET_REMEDIATION = (
    "use QdrantConfig(location=':memory:'), pass url=..., "
    "or inject vector_store=... into Engine"
)


@dataclass(frozen=True)
class QdrantConfig:
    url: str | None = None
    location: str | None = None
    api_key: str | None = None
    store_collection: str = DEFAULT_QDRANT_COLLECTION
    dimension_aware_collection: bool = DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION

    def __post_init__(self) -> None:
        if self.url is not None:
            object.__setattr__(self, "url", self.url.strip() or None)
        if self.location is not None:
            object.__setattr__(self, "location", self.location.strip() or None)
        if self.api_key is not None:
            object.__setattr__(self, "api_key", self.api_key.strip() or None)
        # Neither url nor location is valid: the caller may inject a vector
        # store directly. The store factory raises the teachable error when
        # a Qdrant target is actually required.
        if self.url and self.location:
            raise ValueError(
                "QdrantConfig requires exactly one of url or location; got both. "
                f"{_QDRANT_TARGET_REMEDIATION}."
            )
