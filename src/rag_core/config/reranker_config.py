from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RERANKER_PROVIDER = "none"
RERANKER_MODEL_ENV = "RAG_CORE_RERANKER_MODEL"
RERANKER_PROVIDER_ENV = "RAG_CORE_RERANKER_PROVIDER"


@dataclass(frozen=True)
class RerankerConfig:
    provider: str = DEFAULT_RERANKER_PROVIDER
    model: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("RerankerConfig.provider must be a non-empty string")
        object.__setattr__(self, "provider", self.provider.strip().lower())
