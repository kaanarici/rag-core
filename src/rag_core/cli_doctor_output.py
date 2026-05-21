from __future__ import annotations

import json

from rag_core.cli_output import require_mapping


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
        for source_type in ("file", "url", "archive"):
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
    retrieval = require_mapping(payload.get("retrieval"))
    if retrieval:
        _emit_retrieval_summary(retrieval)
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
    for provider_name in ("qdrant", "turbopuffer"):
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
    embedding = require_mapping(providers.get("embedding"))
    reranker = require_mapping(providers.get("reranker"))
    _emit_provider_category_summary(
        "embedding",
        embedding,
        ("openai", "voyage", "zeroentropy"),
    )
    _emit_provider_category_summary(
        "reranker",
        reranker,
        ("none", "cohere", "voyage", "zeroentropy"),
    )
    _emit_runtime_provider_summary(providers)


def _emit_runtime_provider_summary(providers: dict[str, object]) -> None:
    print("Provider Categories:")
    category_order = (
        ("sparse", ("fastembed",)),
        ("ocr", ("mistral", "gemini")),
        ("contextualizer", ("noop", "anthropic")),
        ("embedding_cache", ("none", "in_memory", "sqlite")),
        ("chunk_context_cache", ("none", "in_memory", "sqlite")),
        ("search_sidecar", ("portable_lexical",)),
        ("event_sink", ("none", "logging", "jsonl", "buffer", "opentelemetry")),
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
        if category == "reranker" and provider.get("configured") is True:
            suffix = f" effective={effective} reason={fallback_reason or 'none'}"
        print(
            f"  {marker} {category}/{provider_name}: "
            f"support={support_level} package={package} api_key={api_key}{suffix}"
        )


def _emit_retrieval_summary(retrieval: dict[str, object]) -> None:
    profiles = require_mapping(retrieval.get("search_profiles"))
    if not profiles:
        return
    default_profile = retrieval.get("default_search_profile")
    print("Retrieval Profiles:")
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
    ordered = (
        "dense",
        "sparse",
        "hybrid_rrf",
        "hybrid_dbsf",
        "hybrid_weighted_rrf",
        "mmr",
        "nested_prefetch",
        "boost",
    )
    supported = [stage for stage in ordered if stages.get(stage) is True]
    return ",".join(supported) if supported else "none"
