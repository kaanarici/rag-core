"""Provider diagnostic category names."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

ProviderDiagnosticCategory: TypeAlias = Literal[
    "embedding",
    "sparse",
    "reranker",
    "ocr",
    "contextualizer",
    "embedding_cache",
    "chunk_context_cache",
    "search_sidecar",
    "event_sink",
    "vector_store",
]

EMBEDDING_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "embedding"
SPARSE_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "sparse"
RERANKER_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "reranker"
OCR_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "ocr"
CONTEXTUALIZER_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "contextualizer"
EMBEDDING_CACHE_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = (
    "embedding_cache"
)
CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = (
    "chunk_context_cache"
)
SEARCH_SIDECAR_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "search_sidecar"
EVENT_SINK_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "event_sink"
VECTOR_STORE_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "vector_store"

MODEL_PROVIDER_DIAGNOSTIC_CATEGORIES: Final[tuple[ProviderDiagnosticCategory, ...]] = (
    EMBEDDING_PROVIDER_CATEGORY,
    SPARSE_PROVIDER_CATEGORY,
    RERANKER_PROVIDER_CATEGORY,
    OCR_PROVIDER_CATEGORY,
    CONTEXTUALIZER_PROVIDER_CATEGORY,
    EMBEDDING_CACHE_PROVIDER_CATEGORY,
    CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY,
    SEARCH_SIDECAR_PROVIDER_CATEGORY,
    EVENT_SINK_PROVIDER_CATEGORY,
)

RUNTIME_PROVIDER_DIAGNOSTIC_CATEGORIES: Final[
    tuple[ProviderDiagnosticCategory, ...]
] = (
    SPARSE_PROVIDER_CATEGORY,
    OCR_PROVIDER_CATEGORY,
    CONTEXTUALIZER_PROVIDER_CATEGORY,
    EMBEDDING_CACHE_PROVIDER_CATEGORY,
    CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY,
    SEARCH_SIDECAR_PROVIDER_CATEGORY,
    EVENT_SINK_PROVIDER_CATEGORY,
)

__all__ = [
    "CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY",
    "CONTEXTUALIZER_PROVIDER_CATEGORY",
    "EMBEDDING_CACHE_PROVIDER_CATEGORY",
    "EMBEDDING_PROVIDER_CATEGORY",
    "EVENT_SINK_PROVIDER_CATEGORY",
    "MODEL_PROVIDER_DIAGNOSTIC_CATEGORIES",
    "OCR_PROVIDER_CATEGORY",
    "ProviderDiagnosticCategory",
    "RERANKER_PROVIDER_CATEGORY",
    "RUNTIME_PROVIDER_DIAGNOSTIC_CATEGORIES",
    "SEARCH_SIDECAR_PROVIDER_CATEGORY",
    "SPARSE_PROVIDER_CATEGORY",
    "VECTOR_STORE_PROVIDER_CATEGORY",
]
