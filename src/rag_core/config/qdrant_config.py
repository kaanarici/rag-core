from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QdrantConfig:
    url: str | None = None
    location: str | None = None
    api_key: str | None = None
    collection: str = "rag_core_chunks"
    dimension_aware_collection: bool = True

    def __post_init__(self) -> None:
        if self.api_key is not None:
            object.__setattr__(self, "api_key", self.api_key.strip() or None)
        # Both unset means the caller supplies a vector store directly.
        if self.url and self.location:
            raise ValueError(
                "QdrantConfig requires exactly one of url or location, not both"
            )
