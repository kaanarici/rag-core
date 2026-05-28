"""Concrete embedding cache providers and cache factory registrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, ClassVar

from rag_core.private_files import (
    harden_private_file,
    prepare_private_file_for_open,
)
from rag_core.search.providers.cache_provider_names import (
    NO_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.registry import (
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
    EmbedCacheKey as _EmbedCacheKey,
    EmbeddingCache as _EmbeddingCache,
)
from rag_core.search.providers.embedding_memory_cache import InMemoryCache, NoCache


class SqliteCache:
    """Disk-backed embedding cache via stdlib :mod:`sqlite3`.

    Vectors are stored as packed float32 blobs to keep the schema small. The
    connection is opened lazily and reused.
    """

    provider_name = SQLITE_CACHE_PROVIDER

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

    async def get(self, key: _EmbedCacheKey) -> list[float] | None:
        return read_embedding_cache_vector(
            self._connect(),
            cache_key=key.stringify(),
        )

    async def get_many(
        self,
        keys: list[_EmbedCacheKey],
    ) -> dict[_EmbedCacheKey, list[float]]:
        vectors = read_embedding_cache_vectors(
            self._connect(),
            cache_keys=[key.stringify() for key in keys],
        )
        return {
            key: vectors[cache_key]
            for key in keys
            if (cache_key := key.stringify()) in vectors
        }

    async def put(self, key: _EmbedCacheKey, vector: list[float]) -> None:
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
        items: dict[_EmbedCacheKey, list[float]],
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


DEFAULT_EMBEDDING_CACHE_PROVIDER = NO_CACHE_PROVIDER


def _build_no_embedding_cache(**_: Any) -> NoCache:
    return NoCache()


def _build_in_memory_embedding_cache(**kwargs: Any) -> InMemoryCache:
    return InMemoryCache(**kwargs)


def _build_sqlite_embedding_cache(**kwargs: Any) -> SqliteCache:
    return SqliteCache(**kwargs)


def create_embedding_cache(
    provider: str | None = None,
    **kwargs: Any,
) -> _EmbeddingCache:
    """Resolve the EmbeddingCache provider category from a config name.

    ``None`` resolves to ``"none"`` (the no-op :class:`NoCache`).
    """
    return EMBEDDING_CACHES.create(
        provider or DEFAULT_EMBEDDING_CACHE_PROVIDER,
        **kwargs,
    )


EMBEDDING_CACHES.register(
    DEFAULT_EMBEDDING_CACHE_PROVIDER,
    _build_no_embedding_cache,
)
EMBEDDING_CACHES.register(InMemoryCache.provider_name, _build_in_memory_embedding_cache)
EMBEDDING_CACHES.register(SqliteCache.provider_name, _build_sqlite_embedding_cache)

__all__ = [
    "DEFAULT_EMBEDDING_CACHE_PROVIDER",
    "InMemoryCache",
    "NoCache",
    "SqliteCache",
    "create_embedding_cache",
]
