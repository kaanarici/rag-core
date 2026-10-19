from __future__ import annotations
from dataclasses import dataclass, field
from .qdrant_config import QdrantConfig

QDRANT_VECTOR_STORE_PROVIDER = "qdrant"
DEFAULT_VECTOR_STORE_PROVIDER = QDRANT_VECTOR_STORE_PROVIDER
VECTOR_STORE_ENV = "RAG_CORE_VECTOR_STORE"
SUPPORTED_VECTOR_STORE_PROVIDERS = (QDRANT_VECTOR_STORE_PROVIDER,)


@dataclass(frozen=True)
class VectorStoreConfig:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    provider: str = QDRANT_VECTOR_STORE_PROVIDER

    def __post_init__(self) -> None:
        if self.provider != QDRANT_VECTOR_STORE_PROVIDER:
            raise ValueError(
                "Only Qdrant is built in; inject another VectorStore into Engine"
            )
