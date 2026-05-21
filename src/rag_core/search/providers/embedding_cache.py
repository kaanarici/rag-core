"""Embedding and chunk-context cache providers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, ClassVar

from rag_core.private_files import (
    harden_private_file,
    prepare_private_file_for_open,
)
from rag_core.search.providers.chunk_context_cache import (
    ChunkContextCache,
    ChunkContextKey,
    InMemoryChunkContextCache,
    NoChunkContextCache,
    SqliteChunkContextCache,
)
from rag_core.search.providers.registry import (
    CHUNK_CONTEXT_CACHES,
    EMBEDDING_CACHES,
)
from rag_core.search.providers.embedding_sqlite_cache import (
    EmbeddingCacheWrite,
    EXPECTED_EMBEDDING_CACHE_SCHEMA,
    ensure_embedding_cache_schema,
    read_embedding_cache_vector,
    read_embedding_cache_vectors,
    write_embedding_cache_vector,
    write_embedding_cache_vectors,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey,
    EmbeddingCache,
    sha256_text,
)
from rag_core.search.providers.embedding_memory_cache import InMemoryCache, NoCache


class SqliteCache:
    """Disk-backed embedding cache via stdlib :mod:`sqlite3`.

    Vectors are stored as packed float32 blobs to keep the schema small. The
    connection is opened lazily and reused.
    """

    _EXPECTED_SCHEMA: ClassVar[frozenset[tuple[str, str, int, int]]] = frozenset(
        EXPECTED_EMBEDDING_CACHE_SCHEMA
    )

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            path = Path(self._path)
            prepare_private_file_for_open(path, reject_symlink=True)
            connection = sqlite3.connect(self._path)
            harden_private_file(path)
            self._ensure_schema(connection)
            connection.commit()
            self._connection = connection
        return self._connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        ensure_embedding_cache_schema(connection)

    async def get(self, key: EmbedCacheKey) -> list[float] | None:
        return read_embedding_cache_vector(
            self._connect(),
            cache_key=key.stringify(),
        )

    async def get_many(
        self,
        keys: list[EmbedCacheKey],
    ) -> dict[EmbedCacheKey, list[float]]:
        vectors = read_embedding_cache_vectors(
            self._connect(),
            cache_keys=[key.stringify() for key in keys],
        )
        return {
            key: vectors[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in vectors
        }

    async def put(self, key: EmbedCacheKey, vector: list[float]) -> None:
        write_embedding_cache_vector(
            self._connect(),
            cache_key=key.stringify(),
            vector=vector,
            provider=key.provider,
            provider_config_fingerprint=key.provider_config_fingerprint,
            model=key.model,
            dimensions=key.dimensions,
            input_type=key.input_type,
            normalization=key.normalization,
            processing_fingerprint=key.processing_fingerprint,
        )

    async def put_many(
        self,
        items: dict[EmbedCacheKey, list[float]],
    ) -> None:
        write_embedding_cache_vectors(
            self._connect(),
            entries=[
                EmbeddingCacheWrite(
                    cache_key=key.stringify(),
                    vector=vector,
                    provider=key.provider,
                    provider_config_fingerprint=key.provider_config_fingerprint,
                    model=key.model,
                    dimensions=key.dimensions,
                    input_type=key.input_type,
                    normalization=key.normalization,
                    processing_fingerprint=key.processing_fingerprint,
                )
                for key, vector in items.items()
            ],
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def _build_no_embedding_cache(**_: Any) -> NoCache:
    return NoCache()


def _build_in_memory_embedding_cache(**kwargs: Any) -> InMemoryCache:
    return InMemoryCache(**kwargs)


def _build_sqlite_embedding_cache(**kwargs: Any) -> SqliteCache:
    return SqliteCache(**kwargs)


def _build_no_chunk_context_cache(**_: Any) -> NoChunkContextCache:
    return NoChunkContextCache()


def _build_in_memory_chunk_context_cache(**_: Any) -> InMemoryChunkContextCache:
    return InMemoryChunkContextCache()


def _build_sqlite_chunk_context_cache(**kwargs: Any) -> SqliteChunkContextCache:
    return SqliteChunkContextCache(**kwargs)


def create_embedding_cache(
    provider: str | None = None,
    **kwargs: Any,
) -> EmbeddingCache:
    """Resolve the EmbeddingCache provider category from a config name.

    ``None`` resolves to ``"none"`` (the no-op :class:`NoCache`).
    """
    return EMBEDDING_CACHES.create(provider or "none", **kwargs)


def create_chunk_context_cache(
    provider: str | None = None,
    **kwargs: Any,
) -> ChunkContextCache:
    """Resolve the ChunkContextCache provider category from a config name.

    ``None`` resolves to ``"none"`` (the no-op :class:`NoChunkContextCache`).
    """
    return CHUNK_CONTEXT_CACHES.create(provider or "none", **kwargs)


EMBEDDING_CACHES.register("none", _build_no_embedding_cache)
EMBEDDING_CACHES.register("in_memory", _build_in_memory_embedding_cache)
EMBEDDING_CACHES.register("sqlite", _build_sqlite_embedding_cache)

CHUNK_CONTEXT_CACHES.register("none", _build_no_chunk_context_cache)
CHUNK_CONTEXT_CACHES.register("in_memory", _build_in_memory_chunk_context_cache)
CHUNK_CONTEXT_CACHES.register("sqlite", _build_sqlite_chunk_context_cache)


__all__ = [
    "ChunkContextCache",
    "ChunkContextKey",
    "EmbedCacheKey",
    "EmbeddingCache",
    "InMemoryCache",
    "InMemoryChunkContextCache",
    "NoCache",
    "NoChunkContextCache",
    "SqliteCache",
    "SqliteChunkContextCache",
    "create_chunk_context_cache",
    "create_embedding_cache",
    "sha256_text",
]
