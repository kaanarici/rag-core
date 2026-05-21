from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

from rag_core.cli_doctor_output import emit_doctor
from rag_core.cli_doctor_store import exercise_doctor_store
from rag_core.cli_inputs import cli_redacted_url, cli_store_location_label
from rag_core.core_models import RAGCoreConfig
from rag_core.core_runtime import (
    describe_source_processing_versions,
    resolve_processing_version,
    resolve_runtime_collection_name,
)
from rag_core.documents.pdf_inspector import describe_pdf_inspector_runtime
from rag_core.runtime_metadata import describe_runtime_metadata
from rag_core.search.providers.embedding_models import resolve_embedding_dimensions
from rag_core.search.providers.model_provider_diagnostics import (
    describe_model_provider_diagnostics,
)
from rag_core.search.providers.reranker import resolve_reranker_provider
from rag_core.search.providers.vector_store_diagnostics import (
    describe_vector_store_diagnostics,
)
from rag_core.search.query_plan_presets import describe_retrieval_profiles

if TYPE_CHECKING:
    from rag_core.core import RAGCore


async def run_doctor_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., "RAGCore"],
) -> int:
    config = RAGCoreConfig.from_cli(args)
    payload = _planned_runtime_payload(config)
    exit_code = 0
    if args.check_store or args.fix:
        store_outcome = await exercise_doctor_store(
            config,
            core_factory=core_factory,
            capture_dimension_mismatch=args.fix,
        )
        payload["store_health"] = store_outcome.health
        if not bool(store_outcome.health.get("healthy", False)):
            exit_code = 1
        if args.fix:
            payload["fix"] = store_outcome.fix_summary
            if store_outcome.fix_summary["status"] == "dimension_mismatch":
                exit_code = 1
    emit_doctor(payload, as_json=args.json, fix=args.fix)
    return exit_code


def _planned_runtime_payload(config: RAGCoreConfig) -> dict[str, object]:
    dimensions = resolve_embedding_dimensions(
        provider=config.embedding.provider,
        model=config.embedding.model,
        dimensions=config.embedding.dimensions,
    )
    requested, fallback_reason = resolve_reranker_provider(
        config.reranker.provider,
        api_key=config.reranker.api_key,
    )
    collection_name = resolve_runtime_collection_name(
        config=config,
        model_name=config.embedding.model,
        dimensions=dimensions,
    )
    processing_version = resolve_processing_version(
        configured_version=config.ingest.processing_version,
        source_type=config.ingest.source_type,
    )
    return {
        "runtime": describe_runtime_metadata(),
        "collection_name": collection_name,
        "processing_version": processing_version.serialize(),
        "source_processing_versions": describe_source_processing_versions(
            processing_version
        ),
        "embedding": {
            "provider": config.embedding.provider,
            "model": config.embedding.model,
            "dimensions": dimensions,
            "batch_size": config.embedding.batch_size,
        },
        "sparse": {
            "provider": "fastembed",
        },
        "reranker": {
            "provider": config.reranker.provider,
            "requested": config.reranker.provider,
            "effective": requested,
            "fallback_reason": fallback_reason,
            "model": config.reranker.model,
        },
        "qdrant": {
            "url": cli_redacted_url(config.qdrant.url),
            "location": cli_store_location_label(config.qdrant.location),
        },
        "vector_store": describe_vector_store_diagnostics(
            config=config,
            collection_name=collection_name,
        ),
        "retrieval": describe_retrieval_profiles(),
        "providers": describe_model_provider_diagnostics(
            config=config,
            embedding_dimensions=dimensions,
        ),
        "pdf_inspector": describe_pdf_inspector_runtime(),
    }
