from __future__ import annotations

from dataclasses import dataclass

DEFAULT_EMBEDDING_PROVIDER = "openai"
DEMO_EMBEDDING_PROVIDER = "demo"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEMO_EMBEDDING_MODEL = "demo-dense-v1"
DEFAULT_EMBEDDING_BATCH_SIZE = 50
EMBEDDING_BATCH_SIZE_ENV = "RAG_CORE_EMBEDDING_BATCH_SIZE"
EMBEDDING_DIMENSIONS_ENV = "RAG_CORE_EMBEDDING_DIMENSIONS"
EMBEDDING_MODEL_ENV = "RAG_CORE_EMBEDDING_MODEL"
EMBEDDING_PROVIDER_ENV = "RAG_CORE_EMBEDDING_PROVIDER"


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("EmbeddingConfig.provider must be non-empty")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions <= 0
        ):
            raise ValueError("EmbeddingConfig.dimensions must be a positive integer")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("EmbeddingConfig.batch_size must be a positive integer")
