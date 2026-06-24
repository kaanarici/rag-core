from __future__ import annotations

import re
from typing import Any

from rag_core.config.vector_store_config import (
    PGVECTOR_VECTOR_STORE_PROVIDER,
    QDRANT_VECTOR_STORE_PROVIDER,
    TURBOPUFFER_NAMESPACE_ENV,
    TURBOPUFFER_VECTOR_STORE_PROVIDER,
)
from rag_core.config.ingest_config import (
    DEFAULT_INGEST_SOURCE_TYPE,
    DEFAULT_PROCESSING_VERSION,
    STANDARD_INGEST_SOURCE_TYPES,
)
from rag_core.core_models import ProcessingFingerprint, Config
from rag_core.documents.contextualizer_provider_names import NOOP_CONTEXTUALIZER_ID
from rag_core.documents.pdf_inspector import describe_pdf_inspector_runtime
from rag_core.events.sinks import describe_event_sink_status
from rag_core.runtime_metadata import describe_runtime_metadata
from rag_core.search.providers.provider_diagnostics import (
    describe_model_provider_diagnostics,
)
from rag_core.search.providers.provider_category_names import SPARSE_PROVIDER_CATEGORY
from rag_core.search.providers.reranker_runtime import (
    RERANKER_EFFECTIVE_ATTR,
    RERANKER_FALLBACK_REASON_ATTR,
    RERANKER_REQUESTED_ATTR,
)
from rag_core.search.providers.vector_store_capabilities import (
    describe_metadata_filter_capabilities,
    describe_query_plan_capabilities,
)
from rag_core.search.planning import describe_search_profile_catalog
from rag_core.search.provider_protocols import StoreCapabilities, provider_name

def resolve_collection_name(
    *,
    base_name: str,
    model_name: str,
    dimensions: int,
    dimension_aware: bool,
    collection_slug: str | None = None,
) -> str:
    # Per-tier physical isolation: when CollectionPolicy binds the process to one
    # logical collection, the slug lands between the base and the embed-model marker
    # so 'rag_core_chunks__restricted__text_embedding_3_large_3072d' cannot
    # collide with the public tier. Multi-collection or unrestricted processes pass
    # ``collection_slug=None`` and keep the legacy single store collection name.
    if not dimension_aware:
        if collection_slug:
            return f"{base_name}__{collection_slug}"
        return base_name
    model_slug = re.sub(r"[^a-z0-9]+", "_", model_name.strip().lower()).strip("_")
    if collection_slug:
        return f"{base_name}__{collection_slug}__{model_slug}_{dimensions}d"
    return f"{base_name}__{model_slug}_{dimensions}d"


def resolve_runtime_collection_name(
    *,
    config: Config,
    model_name: str,
    dimensions: int,
) -> str:
    if config.vector_store.provider == QDRANT_VECTOR_STORE_PROVIDER:
        qdrant_collection_slug: str | None = None
        if config.collection_policy is not None:
            qdrant_collection_slug = config.collection_policy.store_collection_slug
        return resolve_collection_name(
            base_name=config.qdrant.store_collection,
            model_name=model_name,
            dimensions=dimensions,
            dimension_aware=config.qdrant.dimension_aware_collection,
            collection_slug=qdrant_collection_slug,
        )
    if config.vector_store.provider == PGVECTOR_VECTOR_STORE_PROVIDER:
        table_name = config.vector_store.pgvector.table
        if table_name:
            return table_name
        pgvector_collection_slug: str | None = None
        if config.collection_policy is not None:
            pgvector_collection_slug = config.collection_policy.store_collection_slug
        return resolve_collection_name(
            base_name=config.qdrant.store_collection,
            model_name=model_name,
            dimensions=dimensions,
            dimension_aware=config.qdrant.dimension_aware_collection,
            collection_slug=pgvector_collection_slug,
        )
    if config.vector_store.provider == TURBOPUFFER_VECTOR_STORE_PROVIDER:
        namespace = config.vector_store.turbopuffer.namespace
        if not namespace:
            raise ValueError(
                "TurboPuffer requires --turbopuffer-namespace or "
                f"{TURBOPUFFER_NAMESPACE_ENV}"
            )
        return namespace
    raise ValueError(
        f"Unsupported vector store provider: {config.vector_store.provider}"
    )


def build_runtime_description(
    *,
    config: Config,
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
    store_capabilities = vector_store.capabilities
    return {
        "runtime": describe_runtime_metadata(),
        "collection_name": collection_name,
        "pipeline_version": processing_version.serialize(),
        "source_pipeline_versions": describe_source_processing_versions(
            processing_version
        ),
        "embedding": {
            "provider": provider_name(embedding_provider),
            "model": getattr(embedding_provider, "model_name", None),
            "dimensions": getattr(embedding_provider, "dimensions", None),
        },
        SPARSE_PROVIDER_CATEGORY: {
            "provider": provider_name(sparse_embedder),
        },
        "vector_store": {
            "provider": provider_name(vector_store),
            "capabilities": describe_store_capabilities(store_capabilities),
        },
        "search": describe_search_profile_catalog(
            capabilities=store_capabilities.query_plan
        ),
        "reranker": {
            "provider": provider_name(reranker),
            "requested": getattr(reranker, RERANKER_REQUESTED_ATTR, None),
            "effective": getattr(reranker, RERANKER_EFFECTIVE_ATTR, None),
            "fallback_reason": getattr(reranker, RERANKER_FALLBACK_REASON_ATTR, None),
            "configured": reranker is not None,
        },
        "ocr": (
            {
                "provider": provider_name(ocr_provider),
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
            sparse_provider_name=provider_name(sparse_embedder),
            ocr_provider_name=_optional_provider_name(ocr_provider),
            contextualizer_name=_contextualizer_name(chunk_contextualizer),
            embedding_cache_name=_optional_provider_name(embedding_cache),
            chunk_context_cache_name=_optional_provider_name(chunk_context_cache),
            search_sidecar_name=_optional_provider_name(search_sidecar),
            event_sink_name=_optional_provider_name(event_sink),
        ),
        # Sink-side observability surface: ``kind`` is the configured sink's
        # provider_name (or class name for user-supplied sinks); ``failure_count``
        # is the running total of swallowed sink errors. Surfaced here so an
        # operator inspecting ``/v1/runtime`` can see degraded telemetry
        # without parsing logs. Readiness mirrors the same field.
        "event_sink": describe_event_sink_status(event_sink),
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
        "metadata_filter": describe_metadata_filter_capabilities(
            capabilities.metadata_filter
        ),
    }


def resolve_processing_version(
    *,
    configured_version: str,
    source_type: str,
    contextualizer_id: str | None = None,
    contextualizer_chunk_cap: int | None = None,
) -> ProcessingFingerprint:
    base_version = (configured_version or "").strip() or DEFAULT_PROCESSING_VERSION
    normalized_source_type = (source_type or "").strip() or DEFAULT_INGEST_SOURCE_TYPE
    normalized_contextualizer_id = _processing_contextualizer_id(contextualizer_id)
    normalized_chunk_cap = _processing_contextualizer_chunk_cap(contextualizer_chunk_cap)
    return ProcessingFingerprint(
        base_version=base_version,
        source_type=normalized_source_type,
        contextualizer_id=normalized_contextualizer_id,
        contextualizer_chunk_cap=normalized_chunk_cap,
    )


def describe_source_processing_versions(
    default_processing_version: ProcessingFingerprint,
) -> dict[str, str]:
    versions = {
        source_type: ProcessingFingerprint(
            base_version=default_processing_version.base_version,
            source_type=source_type,
            contextualizer_id=default_processing_version.contextualizer_id,
            contextualizer_chunk_cap=(
                default_processing_version.contextualizer_chunk_cap
            ),
        ).serialize()
        for source_type in STANDARD_INGEST_SOURCE_TYPES
    }
    versions["default"] = default_processing_version.serialize()
    return versions


def _optional_provider_name(provider: Any) -> str | None:
    if provider is None:
        return None
    return provider_name(provider)


def _contextualizer_name(provider: Any) -> str | None:
    if provider is None:
        return None
    contextualizer_id = getattr(provider, "contextualizer_id", None)
    if contextualizer_id:
        return str(contextualizer_id)
    return provider_name(provider)


def _processing_contextualizer_id(contextualizer_id: str | None) -> str | None:
    if contextualizer_id is None:
        return None
    normalized = str(contextualizer_id).strip()
    if not normalized or normalized == NOOP_CONTEXTUALIZER_ID:
        return None
    return normalized


def _processing_contextualizer_chunk_cap(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("contextualizer_chunk_cap must be a positive integer")
    return value


def _embedding_dimensions(provider: Any) -> int:
    dimensions = getattr(provider, "dimensions", None)
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions <= 0
    ):
        raise ValueError("embedding provider must expose positive integer dimensions")
    return dimensions
