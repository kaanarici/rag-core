from __future__ import annotations

import json

from rag_core.cli_output import require_mapping
from rag_core.config.ingest_config import STANDARD_INGEST_SOURCE_TYPES
from rag_core.documents.contextualizer_provider_names import (
    CONTEXTUALIZER_PROVIDER_ORDER,
)
from rag_core.documents.ocr_provider_names import OCR_PROVIDER_ORDER
from rag_core.events.sinks import EVENT_SINK_PROVIDER_ORDER
from rag_core.retrieval_channels import (
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
)
from rag_core.search.lexical_sidecar import SEARCH_SIDECAR_PROVIDER_ORDER
from rag_core.search.providers.cache_provider_names import CACHE_PROVIDER_ORDER
from rag_core.search.providers.model_provider_diagnostics import (
    EMBEDDING_PROVIDER_ORDER,
    RERANKER_PROVIDER_ORDER,
)
from rag_core.search.providers.provider_category_names import (
    CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY,
    CONTEXTUALIZER_PROVIDER_CATEGORY,
    EMBEDDING_CACHE_PROVIDER_CATEGORY,
    EMBEDDING_PROVIDER_CATEGORY,
    EVENT_SINK_PROVIDER_CATEGORY,
    OCR_PROVIDER_CATEGORY,
    RERANKER_PROVIDER_CATEGORY,
    SEARCH_SIDECAR_PROVIDER_CATEGORY,
    SPARSE_PROVIDER_CATEGORY,
)
from rag_core.search.providers.sparse import SPARSE_EMBEDDER_PROVIDER_ORDER
from rag_core.search.providers.vector_store_capabilities import (
    BUILTIN_VECTOR_STORE_PROVIDER_ORDER,
    QUERY_PLAN_STAGE_CAPABILITY_FIELDS,
)

_QUERY_PLAN_STAGE_DISPLAY_ORDER = (
    DENSE_RETRIEVAL_CHANNEL,
    SPARSE_RETRIEVAL_CHANNEL,
    *(
        field
        for field in QUERY_PLAN_STAGE_CAPABILITY_FIELDS
        if field not in (DENSE_RETRIEVAL_CHANNEL, SPARSE_RETRIEVAL_CHANNEL)
    ),
)


def emit_doctor(payload: dict[str, object], *, as_json: bool, fix: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    embedding = require_mapping(payload.get("embedding"))
    reranker = require_mapping(payload.get("reranker"))
    qdrant = require_mapping(payload.get("qdrant"))
    runtime = require_mapping(payload.get("runtime"))
    if runtime:
        print(
            "Runtime: "
            f"{runtime.get('package_name')} {runtime.get('package_version') or 'unknown'} "
            f"/ Python {runtime.get('python_version')}"
        )
    print(f"Collection: {payload.get('collection_name')}")
    print(f"Processing Version: {payload.get('processing_version')}")
    source_versions = payload.get("source_processing_versions")
    if isinstance(source_versions, dict):
        print("Source Processing Versions:")
        for source_type in STANDARD_INGEST_SOURCE_TYPES:
            print(f"  {source_type}: {source_versions.get(source_type)}")
    print(
        "Embedding: "
        f"{embedding.get('provider')} / {embedding.get('model')} / "
        f"{embedding.get('dimensions')}d / batch={embedding.get('batch_size')}"
    )
    print(
        "Reranker: "
        f"requested={reranker.get('requested')} "
        f"effective={reranker.get('effective')} "
        f"reason={reranker.get('fallback_reason') or 'none'}"
    )
    model_providers = require_mapping(payload.get("providers"))
    if model_providers:
        _emit_model_provider_summary(model_providers)
    search = require_mapping(payload.get("search"))
    if search:
        _emit_search_summary(search)
    print(f"Qdrant URL: {qdrant.get('url') or 'none'}")
    print(f"Qdrant Location: {qdrant.get('location') or 'none'}")
    vector_store = require_mapping(payload.get("vector_store"))
    if vector_store:
        print(
            "Vector Store: "
            f"configured={vector_store.get('configured')} "
            f"default={vector_store.get('default')}"
        )
        _emit_vector_store_provider_summary(vector_store)
    _emit_next_steps(payload)
    store_health = payload.get("store_health")
    if isinstance(store_health, dict):
        print(
            "Store Health: "
            f"healthy={store_health.get('healthy')} "
            f"points={store_health.get('points_count', 0)}"
        )
    fix_summary = payload.get("fix")
    if fix and isinstance(fix_summary, dict):
        status = fix_summary.get("status")
        if status == "dimension_mismatch":
            print(
                "Fix: dimension_mismatch "
                f"expected={fix_summary.get('expected')} "
                f"actual={fix_summary.get('actual')}"
            )
            print(f"  -> {fix_summary.get('message')}")
        else:
            print(f"Fix: {status} (expected={fix_summary.get('expected')})")


def _emit_vector_store_provider_summary(vector_store: dict[str, object]) -> None:
    providers = require_mapping(vector_store.get("providers"))
    if not providers:
        return
    print("Vector Store Providers:")
    for provider_name in BUILTIN_VECTOR_STORE_PROVIDER_ORDER:
        provider = require_mapping(providers.get(provider_name))
        if not provider:
            continue
        marker = "*" if provider.get("configured") is True else "-"
        support_level = provider.get("support_level") or "unknown"
        query_plan = _query_plan_stage_summary(provider.get("query_plan"))
        print(
            f"  {marker} {provider_name}: "
            f"support={support_level} query_plan={query_plan}"
        )


def _emit_model_provider_summary(providers: dict[str, object]) -> None:
    print("Model Providers:")
    embedding = require_mapping(providers.get(EMBEDDING_PROVIDER_CATEGORY))
    reranker = require_mapping(providers.get(RERANKER_PROVIDER_CATEGORY))
    _emit_provider_category_summary(
        EMBEDDING_PROVIDER_CATEGORY,
        embedding,
        EMBEDDING_PROVIDER_ORDER,
    )
    _emit_provider_category_summary(
        RERANKER_PROVIDER_CATEGORY,
        reranker,
        RERANKER_PROVIDER_ORDER,
    )
    _emit_runtime_provider_summary(providers)


def _emit_runtime_provider_summary(providers: dict[str, object]) -> None:
    print("Provider Categories:")
    category_order = (
        (SPARSE_PROVIDER_CATEGORY, SPARSE_EMBEDDER_PROVIDER_ORDER),
        (OCR_PROVIDER_CATEGORY, OCR_PROVIDER_ORDER),
        (CONTEXTUALIZER_PROVIDER_CATEGORY, CONTEXTUALIZER_PROVIDER_ORDER),
        (EMBEDDING_CACHE_PROVIDER_CATEGORY, CACHE_PROVIDER_ORDER),
        (CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY, CACHE_PROVIDER_ORDER),
        (SEARCH_SIDECAR_PROVIDER_CATEGORY, SEARCH_SIDECAR_PROVIDER_ORDER),
        (EVENT_SINK_PROVIDER_CATEGORY, EVENT_SINK_PROVIDER_ORDER),
    )
    for category, provider_order in category_order:
        _emit_provider_category_summary(
            category,
            require_mapping(providers.get(category)),
            provider_order,
        )


def _emit_provider_category_summary(
    category: str,
    diagnostics: dict[str, object],
    provider_order: tuple[str, ...],
) -> None:
    provider_payloads = require_mapping(diagnostics.get("providers"))
    configured = diagnostics.get("configured")
    effective = diagnostics.get("effective")
    fallback_reason = diagnostics.get("fallback_reason")
    ordered = list(provider_order)
    if isinstance(configured, str) and configured not in ordered:
        ordered.append(configured)
    for provider_name in ordered:
        provider = require_mapping(provider_payloads.get(provider_name))
        if not provider:
            continue
        marker = "*" if provider.get("configured") is True else "-"
        support_level = provider.get("support_level") or "unknown"
        package = _yes_no(provider.get("package_available"))
        api_key = _yes_no(provider.get("api_key_configured"))
        suffix = ""
        if category == RERANKER_PROVIDER_CATEGORY and provider.get("configured") is True:
            suffix = f" effective={effective} reason={fallback_reason or 'none'}"
        print(
            f"  {marker} {category}/{provider_name}: "
            f"support={support_level} package={package} api_key={api_key}{suffix}"
        )


def _emit_search_summary(search: dict[str, object]) -> None:
    profiles = require_mapping(search.get("search_profiles"))
    if not profiles:
        return
    default_profile = search.get("default_search_profile")
    print("Search Profiles:")
    for profile_name, profile_value in profiles.items():
        profile = require_mapping(profile_value)
        marker = "*" if profile_name == default_profile else "-"
        print(
            f"  {marker} {profile_name}: "
            f"preset={profile.get('preset')} "
            f"latency={profile.get('latency')} "
            f"quality={profile.get('quality')} "
            f"use={profile.get('summary')}"
        )
    note = search.get("default_search_profile_note")
    if isinstance(note, str) and note:
        print(f"  default: {note}")


def _emit_next_steps(payload: dict[str, object]) -> None:
    steps = _doctor_next_steps(payload)
    if not steps:
        return
    print("Next Steps:")
    for step in steps:
        print(f"  - {step}")


def _doctor_next_steps(payload: dict[str, object]) -> list[str]:
    steps: list[str] = []
    providers = require_mapping(payload.get("providers"))
    embedding = require_mapping(providers.get(EMBEDDING_PROVIDER_CATEGORY))
    configured_embedding = embedding.get("configured")
    embedding_providers = require_mapping(embedding.get("providers"))
    embedding_provider = require_mapping(
        embedding_providers.get(configured_embedding)
        if isinstance(configured_embedding, str)
        else None
    )
    vector_store = require_mapping(payload.get("vector_store"))
    configured_store = vector_store.get("configured")
    store_providers = require_mapping(vector_store.get("providers"))
    store_provider = require_mapping(
        store_providers.get(configured_store) if isinstance(configured_store, str) else None
    )

    if _embedding_api_key_missing(configured_embedding, embedding_provider):
        steps.append(
            "For no-key smoke, run "
            '`rag-core local-search examples/demo_corpus "How can invoices be paid?"`.'
        )
        env_names = _env_display(embedding_provider.get("api_key_env"))
        provider_name = str(configured_embedding)
        steps.append(
            f"For configured {provider_name} embeddings, set {env_names} "
            "or use `--embedding-provider demo --embedding-dimensions 64`."
        )
    if configured_store == "qdrant" and store_provider.get("connection_configured") is False:
        steps.append(
            "For Qdrant, pass `--qdrant-location :memory:` for local smoke "
            "or `--qdrant-url http://127.0.0.1:6333` for a running service."
        )
    return steps


def _embedding_api_key_missing(
    configured_embedding: object,
    embedding_provider: dict[str, object],
) -> bool:
    return (
        isinstance(configured_embedding, str)
        and embedding_provider.get("configured") is True
        and embedding_provider.get("api_key_configured") is False
        and bool(embedding_provider.get("api_key_env"))
    )


def _env_display(value: object) -> str:
    if isinstance(value, str) and value:
        return f"`{value}`"
    if isinstance(value, list):
        names = [f"`{item}`" for item in value if isinstance(item, str) and item]
        if names:
            return " or ".join(names)
    return "the provider API key"


def _yes_no(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _query_plan_stage_summary(query_plan: object) -> str:
    stages = require_mapping(query_plan)
    if not stages:
        return "none"
    supported = [
        stage for stage in _QUERY_PLAN_STAGE_DISPLAY_ORDER if stages.get(stage) is True
    ]
    return ",".join(supported) if supported else "none"
