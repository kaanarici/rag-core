from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TYPE_CHECKING

from rag_core.cli.doctor_output import emit_doctor
from rag_core.cli.doctor_providers import exercise_doctor_model_providers
from rag_core.cli.doctor_store import exercise_doctor_store
from rag_core.cli.inputs import cli_redacted_url, cli_store_location_label
from rag_core.core_models import Config
from rag_core.documents.converters.format_support import format_support_payloads
from rag_core._engine.core_runtime import (
    describe_source_processing_versions,
    resolve_processing_version,
    resolve_runtime_collection_name,
)
from rag_core.documents.pdf_inspector import describe_pdf_inspector_runtime
from rag_core.runtime_metadata import describe_runtime_metadata
from rag_core.search.providers.embedding_models import resolve_embedding_dimensions
from rag_core.search.providers.provider_diagnostics import (
    describe_model_provider_diagnostics,
)
from rag_core.search.providers.reranker import resolve_reranker_provider
from rag_core.search.providers.sparse import DEFAULT_SPARSE_EMBEDDER_PROVIDER
from rag_core.search.providers.vector_store_diagnostics import (
    VECTOR_STORE_RUNTIME_FAILED,
    VECTOR_STORE_RUNTIME_HEALTHY,
    describe_vector_store_diagnostics,
)
from rag_core.search.planning import describe_search_profile_catalog

if TYPE_CHECKING:
    from rag_core.core import Engine


async def run_doctor_command(
    args: argparse.Namespace,
    *,
    core_factory: Callable[..., "Engine"],
) -> int:
    config = Config.from_cli(args)
    payload = _planned_core_payload(config)
    exit_code = 0
    if args.check_store or args.fix:
        store_outcome = await exercise_doctor_store(
            config,
            core_factory=core_factory,
            capture_dimension_mismatch=args.fix,
        )
        payload["store_health"] = store_outcome.health
        _mark_vector_store_runtime_validation(
            payload,
            provider=config.vector_store.provider,
            healthy=bool(store_outcome.health.get("healthy", False)),
        )
        if not bool(store_outcome.health.get("healthy", False)):
            exit_code = 1
        if args.fix:
            payload["fix"] = store_outcome.fix_summary
            if store_outcome.fix_summary["status"] == "dimension_mismatch":
                exit_code = 1
        provider_health = await exercise_doctor_model_providers(config)
        if provider_health:
            payload["provider_health"] = provider_health
        if any(
            not bool(health.get("healthy", False))
            for health in provider_health.values()
        ):
            exit_code = 1
    emit_doctor(payload, as_json=args.json, fix=args.fix)
    return exit_code


def _mark_vector_store_runtime_validation(
    payload: dict[str, object],
    *,
    provider: str,
    healthy: bool,
) -> None:
    vector_store = payload.get("vector_store")
    if not isinstance(vector_store, dict):
        return
    providers = vector_store.get("providers")
    if not isinstance(providers, dict):
        return
    provider_payload = providers.get(provider)
    if not isinstance(provider_payload, dict):
        return
    provider_payload["runtime_validated"] = healthy
    provider_payload["runtime_validation"] = (
        VECTOR_STORE_RUNTIME_HEALTHY if healthy else VECTOR_STORE_RUNTIME_FAILED
    )


def _planned_core_payload(config: Config) -> dict[str, object]:
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
        "pipeline_version": processing_version.serialize(),
        "source_pipeline_versions": describe_source_processing_versions(
            processing_version
        ),
        "embedding": {
            "provider": config.embedding.provider,
            "model": config.embedding.model,
            "dimensions": dimensions,
            "batch_size": config.embedding.batch_size,
        },
        "sparse": {
            "provider": DEFAULT_SPARSE_EMBEDDER_PROVIDER,
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
        "search": describe_search_profile_catalog(),
        "providers": describe_model_provider_diagnostics(
            config=config,
            embedding_dimensions=dimensions,
        ),
        "pdf_inspector": describe_pdf_inspector_runtime(),
        "formats": format_support_payloads(),
    }
