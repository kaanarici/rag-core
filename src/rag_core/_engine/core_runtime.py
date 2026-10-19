from __future__ import annotations
import re
from dataclasses import asdict, replace
from typing import Any
from rag_core.config.chunking_config import ChunkingConfig
from rag_core.config.ingest_config import (
    DEFAULT_PROCESSING_VERSION,
    DEFAULT_INGEST_SOURCE_TYPE,
    STANDARD_INGEST_SOURCE_TYPES,
)
from rag_core.core_models import Config, ProcessingFingerprint
from rag_core.documents.contextualizer_provider_names import NOOP_CONTEXTUALIZER_ID
from rag_core.events.sinks import describe_event_sink_status
from rag_core.runtime_metadata import describe_runtime_metadata
from rag_core.search.provider_protocols import StoreCapabilities, provider_name


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


def describe_store_capabilities(
    capabilities: StoreCapabilities,
) -> dict[str, object]:
    return {
        "per_point_delete": capabilities.per_point_delete,
        "dense_vector_dimensions": capabilities.dense_vector_dimensions,
        "hybrid": capabilities.hybrid,
        "lexical": capabilities.lexical,
        "native_lexical": capabilities.native_lexical,
        "metadata_filter": asdict(capabilities.metadata_filter),
    }


def resolve_processing_version(
    *,
    configured_version: str,
    source_type: str,
    contextualizer_id: str | None = None,
    contextualizer_chunk_cap: int | None = None,
    chunking: ChunkingConfig = ChunkingConfig(),
    sparse_index: bool = False,
    sparse_model: str | None = None,
) -> ProcessingFingerprint:
    base_version = (configured_version or "").strip() or DEFAULT_PROCESSING_VERSION
    normalized_source_type = (source_type or "").strip() or DEFAULT_INGEST_SOURCE_TYPE
    normalized_contextualizer_id = _processing_contextualizer_id(contextualizer_id)
    normalized_chunk_cap = _processing_contextualizer_chunk_cap(
        contextualizer_chunk_cap
    )
    return ProcessingFingerprint(
        base_version=base_version,
        source_type=normalized_source_type,
        contextualizer_id=normalized_contextualizer_id,
        contextualizer_chunk_cap=normalized_chunk_cap,
        chunking=chunking,
        sparse_index=sparse_index,
        sparse_model=sparse_model,
    )


def describe_source_processing_versions(
    default_processing_version: ProcessingFingerprint,
) -> dict[str, str]:
    versions = {
        source_type: replace(
            default_processing_version, source_type=source_type
        ).serialize()
        for source_type in STANDARD_INGEST_SOURCE_TYPES
    }
    versions["default"] = default_processing_version.serialize()
    return versions


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


def resolve_runtime_collection_name(
    *, config: Config, model_name: str, dimensions: int
) -> str:
    target = config.vector_store.qdrant
    return resolve_collection_name(
        base_name=target.store_collection,
        model_name=model_name,
        dimensions=dimensions,
        dimension_aware=target.dimension_aware_collection,
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
    event_sink: Any = None,
    chunk_contextualizer: Any = None,
) -> dict[str, object]:
    def describe(value: Any) -> dict[str, object] | None:
        if value is None:
            return None
        return {
            "provider": provider_name(value),
            "model": getattr(value, "model_name", None),
        }

    return {
        "runtime": describe_runtime_metadata(),
        "collection_name": collection_name,
        "pipeline_version": processing_version.serialize(),
        "source_pipeline_versions": describe_source_processing_versions(
            processing_version
        ),
        "embedding": {
            **(describe(embedding_provider) or {}),
            "dimensions": embedding_provider.dimensions,
        },
        "sparse": describe(sparse_embedder),
        "vector_store": {
            "provider": provider_name(vector_store),
            "capabilities": describe_store_capabilities(vector_store.capabilities),
        },
        "search": {"mode": "hybrid" if config.hybrid else "dense"},
        "reranker": describe(reranker),
        "ocr": describe(ocr_provider),
        "event_sink": describe_event_sink_status(event_sink),
    }
