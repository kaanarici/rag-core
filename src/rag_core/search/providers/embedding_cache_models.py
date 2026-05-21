"""Embedding cache keys and protocol contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbedCacheKey:
    """Key for a dense embedding cache entry."""

    provider: str
    provider_config_fingerprint: str
    model: str
    dimensions: int
    input_type: str
    normalization: str
    processing_fingerprint: str
    content_sha256: str

    def stringify(self) -> str:
        return json.dumps(
            {
                "content_sha256": self.content_sha256,
                "dimensions": self.dimensions,
                "input_type": self.input_type,
                "model": self.model,
                "normalization": self.normalization,
                "processing_fingerprint": self.processing_fingerprint,
                "provider": self.provider,
                "provider_config_fingerprint": self.provider_config_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@runtime_checkable
class EmbeddingCache(Protocol):
    """Look up and store dense embeddings keyed by ``EmbedCacheKey``."""

    async def get(self, key: EmbedCacheKey) -> list[float] | None: ...

    async def put(self, key: EmbedCacheKey, vector: list[float]) -> None: ...


def sha256_text(text: str) -> str:
    """Hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "EmbedCacheKey",
    "EmbeddingCache",
    "sha256_text",
]
