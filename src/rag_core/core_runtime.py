from __future__ import annotations

import re
from typing import Any

from rag_core.core_models import ProcessingFingerprint, RAGCoreConfig
from rag_core.documents.pdf_inspector import describe_pdf_inspector_runtime
from rag_core.runtime_metadata import describe_runtime_metadata
from rag_core.search.providers.model_provider_diagnostics import (
    describe_model_provider_diagnostics,
)
from rag_core.search.query_plan_presets import describe_retrieval_profiles
from rag_core.search.types import QueryPlanCapabilities, StoreCapabilities

# ``create_reranker`` attaches these optional diagnostics when it resolves a
# configured provider. External rerankers can omit them.
RERANKER_REQUESTED_ATTR = "_rag_core_provider_requested"
RERANKER_EFFECTIVE_ATTR = "_rag_core_provider_effective"
RERANKER_FALLBACK_REASON_ATTR = "_rag_core_fallback_reason"
STANDARD_INGEST_SOURCE_TYPES = ("file", "url", "archive")


def resolve_collection_name(
    *,
    base_name: str,
    model_name: str,
    dimensions: int,
    dimension_aware: bool,
) -> str:
    if not dimension_aware:
        return base_name
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.strip().lower()).strip("_")
    return f"{base_name}__{model_slug}_{dimensions}d"


def resolve_runtime_collection_name(
    *,
    config: RAGCoreConfig,
    model_name: str,
    dimensions: int,
) -> str:
    if config.vector_store.provider == "qdrant":
        return resolve_collection_name(
            base_name=config.qdrant.collection,
            model_name=model_name,
            dimensions=dimensions,
            dimension_aware=config.qdrant.dimension_aware_collection,
        )
    if config.vector_store.provider == "turbopuffer":
        namespace = config.vector_store.turbopuffer.namespace
        if not namespace:
            raise ValueError(
                "TurboPuffer requires --turbopuffer-namespace or "
                "RAG_CORE_TURBOPUFFER_NAMESPACE"
            )
        return namespace
    raise ValueError(
        f"Unsupported vector store provider: {config.vector_store.provider}"
    )


def build_runtime_description(
    *,
    config: RAGCoreConfig,
    collection_name: str | None,
    embedding_provider: Any,
    sparse_embedder: Any,
    vector_store: Any,
    reranker: Any,
    ocr_provider: Any,
    processing_version: ProcessingFingerprint,
    search_sidecar: Any = None,
    event_sink: Any = None,
    chunk_contextualizer: Any = None,
    chunk_context_cache: Any = None,
    embedding_cache: Any = None,
) -> dict[str, object]:
    return {
        "runtime": describe_runtime_metadata(),
        "collection_name": collection_name,
        "processing_version": processing_version.serialize(),
        "source_processing_versions": describe_source_processing_versions(
            processing_version
        ),
        "embedding": {
            "provider": _provider_name(embedding_provider),
            "model": getattr(embedding_provider, "model_name", None),
            "dimensions": getattr(embedding_provider, "dimensions", None),
        },
        "sparse": {
            "provider": _provider_name(sparse_embedder),
        },
        "vector_store": {
            "provider": _provider_name(vector_store),
            "capabilities": describe_store_capabilities(vector_store.capabilities),
        },
        "retrieval": describe_retrieval_profiles(),
        "reranker": {
            "provider": _provider_name(reranker),
            "requested": getattr(reranker, RERANKER_REQUESTED_ATTR, None),
            "effective": getattr(reranker, RERANKER_EFFECTIVE_ATTR, None),
            "fallback_reason": getattr(reranker, RERANKER_FALLBACK_REASON_ATTR, None),
        },
        "ocr": (
            {
                "provider": _provider_name(ocr_provider),
                "model": getattr(ocr_provider, "model_name", None),
                "supports_page_selection": getattr(
                    ocr_provider, "supports_page_selection", False
                ),
            }
            if ocr_provider is not None
            else None
        ),
        "providers": describe_model_provider_diagnostics(
            config=config,
            embedding_dimensions=_embedding_dimensions(embedding_provider),
            sparse_provider_name=_provider_name(sparse_embedder),
            ocr_provider_name=_optional_provider_name(ocr_provider),
            contextualizer_name=_contextualizer_name(chunk_contextualizer),
            embedding_cache_name=_optional_provider_name(embedding_cache),
            chunk_context_cache_name=_optional_provider_name(chunk_context_cache),
            search_sidecar_name=_optional_provider_name(search_sidecar),
            event_sink_name=_optional_provider_name(event_sink),
        ),
        "pdf_inspector": describe_pdf_inspector_runtime(),
    }


def describe_store_capabilities(
    capabilities: StoreCapabilities,
) -> dict[str, object]:
    return {
        "per_point_delete": capabilities.per_point_delete,
        "document_record_lookup": capabilities.document_record_lookup,
        "dense_vector_dimensions": capabilities.dense_vector_dimensions,
        "query_plan": describe_query_plan_capabilities(capabilities.query_plan),
    }


def describe_query_plan_capabilities(
    capabilities: QueryPlanCapabilities,
) -> dict[str, bool]:
    return {
        "dense": capabilities.dense,
        "sparse": capabilities.sparse,
        "hybrid": capabilities.hybrid,
        "hybrid_rrf": capabilities.hybrid_rrf,
        "hybrid_dbsf": capabilities.hybrid_dbsf,
        "hybrid_weighted_rrf": capabilities.hybrid_weighted_rrf,
        "mmr": capabilities.mmr,
        "boost": capabilities.boost,
        "nested_prefetch": capabilities.nested_prefetch,
    }


def resolve_processing_version(
    *,
    configured_version: str,
    source_type: str,
    contextualizer_id: str | None = None,
) -> ProcessingFingerprint:
    base_version = (configured_version or "").strip() or "rag_core_processing_v1"
    normalized_source_type = (source_type or "").strip() or "file"
    normalized_contextualizer_id = _processing_contextualizer_id(contextualizer_id)
    return ProcessingFingerprint(
        base_version=base_version,
        source_type=normalized_source_type,
        contextualizer_id=normalized_contextualizer_id,
    )


def describe_source_processing_versions(
    default_processing_version: ProcessingFingerprint,
) -> dict[str, str]:
    versions = {
        source_type: ProcessingFingerprint(
            base_version=default_processing_version.base_version,
            source_type=source_type,
            contextualizer_id=default_processing_version.contextualizer_id,
        ).serialize()
        for source_type in STANDARD_INGEST_SOURCE_TYPES
    }
    versions["default"] = default_processing_version.serialize()
    return versions


def _provider_name(provider: Any) -> str:
    explicit = getattr(provider, "provider_name", None)
    if explicit:
        return str(explicit)
    return type(provider).__name__


def _optional_provider_name(provider: Any) -> str | None:
    if provider is None:
        return None
    return _provider_name(provider)


def _contextualizer_name(provider: Any) -> str | None:
    if provider is None:
        return None
    contextualizer_id = getattr(provider, "contextualizer_id", None)
    if contextualizer_id:
        return str(contextualizer_id)
    return _provider_name(provider)


def _processing_contextualizer_id(contextualizer_id: str | None) -> str | None:
    if contextualizer_id is None:
        return None
    normalized = str(contextualizer_id).strip()
    if not normalized or normalized == "noop":
        return None
    return normalized


def _embedding_dimensions(provider: Any) -> int:
    dimensions = getattr(provider, "dimensions", None)
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions <= 0
    ):
        raise ValueError("embedding provider must expose positive integer dimensions")
    return dimensions
