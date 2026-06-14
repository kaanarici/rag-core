"""Shared cache provider names for embedding and chunk-context caches."""

from __future__ import annotations

NO_CACHE_PROVIDER = "none"
IN_MEMORY_CACHE_PROVIDER = "in_memory"
SQLITE_CACHE_PROVIDER = "sqlite"
CACHE_PROVIDER_ORDER = (
    NO_CACHE_PROVIDER,
    IN_MEMORY_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)

__all__ = [
    "CACHE_PROVIDER_ORDER",
    "IN_MEMORY_CACHE_PROVIDER",
    "NO_CACHE_PROVIDER",
    "SQLITE_CACHE_PROVIDER",
]
