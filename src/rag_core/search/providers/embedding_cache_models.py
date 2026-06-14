"""Embedding cache keys and protocol contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbedCacheKey:
    """Key for a dense embedding cache entry.

    Scope (``namespace`` / ``corpus_id`` / ``document_id``) is part of the
    cache key so a ``delete_by_document_scope`` purge actually removes the
    bytes derived from a deleted document. Sensitive paraphrases cannot survive
    a right-to-forget by sitting in an un-scoped content-hashed key. The
    trade-off: identical text ingested into two documents stops sharing a cache
    hit. Deployments can still disable cache entirely for sensitive corpora.

    All scope fields default to ``""`` so callers that don't yet know the
    scope (legacy plumbing tests) keep working, but production wiring
    through ``CachedEmbeddingKeyBuilder`` always populates them.
    """

    provider: str
    provider_config_fingerprint: str
    model: str
    dimensions: int
    input_type: str
    normalization: str
    processing_fingerprint: str
    content_sha256: str
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""

    def stringify(self) -> str:
        return json.dumps(
            {
                "content_sha256": self.content_sha256,
                "corpus_id": self.corpus_id,
                "dimensions": self.dimensions,
                "document_id": self.document_id,
                "input_type": self.input_type,
                "model": self.model,
                "namespace": self.namespace,
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


@runtime_checkable
class ScopedDeletableEmbeddingCache(Protocol):
    """Optional capability surfaced by caches that support scoped purge.

    The default :class:`EmbeddingCache` protocol stays minimal. The delete
    facade probes for this method via ``getattr`` so a third-party cache that
    cannot scope-delete (e.g. An external Redis whose keys don't carry scope)
    is acceptable. The resulting ``DeleteDocumentResult.embedding_cache_purged``
    just stays ``None`` for that surface.
    """

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str,
    ) -> int: ...


def sha256_text(text: str) -> str:
    """Hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "EmbedCacheKey",
    "EmbeddingCache",
    "ScopedDeletableEmbeddingCache",
    "sha256_text",
]
