"""Typed provider registry for name-resolved provider categories.

A registry holds named factories for one provider category. Built-in adapters
register themselves when their owning module is imported; the singletons load
those owning modules lazily on first lookup so the public aggregator never
pulls optional deps unless the caller asks for a specific provider.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Generic, TypeVar

T = TypeVar("T")
ProviderFactory = Callable[..., T]


class ProviderRegistry(Generic[T]):
    """Typed registry of named provider factories for one category.

    Lookup is case-insensitive after .strip().lower(). Built-ins register at
    module-import time of the owning module; the registry imports that module
    lazily so optional-dep providers are only loaded when actually used.
    """

    def __init__(self, category: str) -> None:
        self._category = category
        self._factories: dict[str, ProviderFactory[T]] = {}
        self._builtin_modules: tuple[str, ...] = ()
        self._builtins_loaded = False

    def with_builtins(self, *modules: str) -> "ProviderRegistry[T]":
        """Declare modules whose import-time side effects register built-ins.

        Lazy: imports happen on the first lookup, not now. Returns self for
        single-expression module-level construction.
        """

        self._builtin_modules = modules
        return self

    def register(self, name: str, factory: ProviderFactory[T]) -> None:
        key = self._normalize(name)
        if not key:
            raise ValueError(f"{self._category} provider name must be non-empty")
        if key in self._factories:
            raise ValueError(f"{self._category} provider already registered: {key}")
        self._factories[key] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(self._normalize(name), None)

    def create(self, name: str, /, **kwargs: object) -> T:
        self._ensure_builtins_loaded()
        key = self._normalize(name)
        factory = self._factories.get(key)
        if factory is None:
            known = ", ".join(sorted(self._factories)) or "(none)"
            raise ValueError(
                f"Unknown {self._category} provider: {name!r}. Known: {known}"
            )
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        self._ensure_builtins_loaded()
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        self._ensure_builtins_loaded()
        return self._normalize(name) in self._factories

    def _ensure_builtins_loaded(self) -> None:
        if self._builtins_loaded:
            return
        for module_name in self._builtin_modules:
            import_module(module_name)
        self._builtins_loaded = True

    @staticmethod
    def _normalize(name: str) -> str:
        return (name or "").strip().lower()


if TYPE_CHECKING:
    from rag_core.documents.ocr import OcrProvider
    from rag_core.search.providers.chunk_context_cache import ChunkContextCache
    from rag_core.search.providers.embedding_cache_models import EmbeddingCache
    from rag_core.search.types import (
        EmbeddingProvider,
        RerankerProvider,
        SearchSidecar,
        SparseEmbedder,
        VectorStore,
    )


EMBEDDING_PROVIDERS: ProviderRegistry["EmbeddingProvider"] = ProviderRegistry(
    "embedding"
).with_builtins("rag_core.search.providers.embedding")
RERANKER_PROVIDERS: ProviderRegistry["RerankerProvider"] = ProviderRegistry(
    "reranker"
).with_builtins("rag_core.search.providers.reranker")
SPARSE_EMBEDDERS: ProviderRegistry["SparseEmbedder"] = ProviderRegistry(
    "sparse"
).with_builtins("rag_core.search.providers.sparse")
OCR_PROVIDERS: ProviderRegistry["OcrProvider"] = ProviderRegistry(
    "ocr"
).with_builtins("rag_core.documents.ocr")
VECTOR_STORES: ProviderRegistry["VectorStore"] = ProviderRegistry(
    "vector_store"
).with_builtins(
    "rag_core.search.providers.qdrant_store",
    "rag_core.search.providers.memory_store",
    "rag_core.search.providers.turbopuffer_store",
)
SEARCH_SIDECARS: ProviderRegistry["SearchSidecar"] = ProviderRegistry(
    "search_sidecar"
).with_builtins("rag_core.search.lexical_sidecar")
EMBEDDING_CACHES: ProviderRegistry["EmbeddingCache"] = ProviderRegistry(
    "embedding_cache"
).with_builtins("rag_core.search.providers.embedding_cache")
CHUNK_CONTEXT_CACHES: ProviderRegistry["ChunkContextCache"] = ProviderRegistry(
    "chunk_context_cache"
).with_builtins("rag_core.search.providers.embedding_cache")
