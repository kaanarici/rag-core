from __future__ import annotations

from rag_core.search.providers.diagnostic_support import (
    FIELD_API_KEY_CONFIGURED,
    FIELD_API_KEY_ENV,
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_PROVIDERS,
    FIELD_READINESS_SCOPE,
    FIELD_REGISTERED,
    FIELD_RUNTIME_CONFIG,
    FIELD_MATURITY,
)
from rag_core.search.providers.provider_category_names import (
    CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY,
    CONTEXTUALIZER_PROVIDER_CATEGORY,
    EMBEDDING_CACHE_PROVIDER_CATEGORY,
    EMBEDDING_PROVIDER_CATEGORY,
    EVENT_SINK_PROVIDER_CATEGORY,
    MODEL_PROVIDER_DIAGNOSTIC_CATEGORIES,
    OCR_PROVIDER_CATEGORY,
    RERANKER_PROVIDER_CATEGORY,
    RUNTIME_PROVIDER_DIAGNOSTIC_CATEGORIES,
    SEARCH_SIDECAR_PROVIDER_CATEGORY,
    SPARSE_PROVIDER_CATEGORY,
    VECTOR_STORE_PROVIDER_CATEGORY,
)
from rag_core.search.providers.provider_result_values import safe_provider_value_type

from tests.support.source_graph import (
    defining_modules,
    import_graph,
    modules_assigning_value,
    symbol_module,
)

PROVIDER_RESULT_VALUES = "rag_core.search.providers.provider_result_values"
PROVIDER_CATEGORY_NAMES = "rag_core.search.providers.provider_category_names"
DIAGNOSTIC_SUPPORT = "rag_core.search.providers.diagnostic_support"


def test_provider_result_value_type_labels_have_single_owner() -> None:
    # The value-type classifier is a single function with one home; consumers
    # import it rather than reimplementing a per-module `_safe_value_type`.
    assert symbol_module(safe_provider_value_type) == PROVIDER_RESULT_VALUES
    assert defining_modules("src/rag_core", name="safe_provider_value_type") == {
        PROVIDER_RESULT_VALUES
    }
    assert defining_modules("src/rag_core", name="_safe_value_type") == set()

    graph = import_graph("src/rag_core")
    for consumer in (
        "rag_core.search.providers.embedding_results",
        "rag_core.search.providers.rerank_results",
    ):
        assert (
            f"{PROVIDER_RESULT_VALUES}.safe_provider_value_type"
            in graph[consumer]
        )


def test_provider_diagnostic_category_names_have_single_owner() -> None:
    assert EMBEDDING_PROVIDER_CATEGORY == "embedding"
    assert SPARSE_PROVIDER_CATEGORY == "sparse"
    assert RERANKER_PROVIDER_CATEGORY == "reranker"
    assert OCR_PROVIDER_CATEGORY == "ocr"
    assert CONTEXTUALIZER_PROVIDER_CATEGORY == "contextualizer"
    assert EMBEDDING_CACHE_PROVIDER_CATEGORY == "embedding_cache"
    assert CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY == "chunk_context_cache"
    assert SEARCH_SIDECAR_PROVIDER_CATEGORY == "search_sidecar"
    assert EVENT_SINK_PROVIDER_CATEGORY == "event_sink"
    assert VECTOR_STORE_PROVIDER_CATEGORY == "vector_store"
    assert MODEL_PROVIDER_DIAGNOSTIC_CATEGORIES == (
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
    assert RUNTIME_PROVIDER_DIAGNOSTIC_CATEGORIES == (
        SPARSE_PROVIDER_CATEGORY,
        OCR_PROVIDER_CATEGORY,
        CONTEXTUALIZER_PROVIDER_CATEGORY,
        EMBEDDING_CACHE_PROVIDER_CATEGORY,
        CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY,
        SEARCH_SIDECAR_PROVIDER_CATEGORY,
        EVENT_SINK_PROVIDER_CATEGORY,
    )

    # Each category constant and aggregate tuple lives only in
    # provider_category_names. ("embedding"/"reranker" literals also appear as
    # distinct provider_health KIND constants, so ownership is the AST definition
    # site of the category name, not the shared literal value.)
    for category in (
        "EMBEDDING_PROVIDER_CATEGORY",
        "SPARSE_PROVIDER_CATEGORY",
        "RERANKER_PROVIDER_CATEGORY",
        "OCR_PROVIDER_CATEGORY",
        "CONTEXTUALIZER_PROVIDER_CATEGORY",
        "EMBEDDING_CACHE_PROVIDER_CATEGORY",
        "CHUNK_CONTEXT_CACHE_PROVIDER_CATEGORY",
        "SEARCH_SIDECAR_PROVIDER_CATEGORY",
        "EVENT_SINK_PROVIDER_CATEGORY",
        "VECTOR_STORE_PROVIDER_CATEGORY",
        "MODEL_PROVIDER_DIAGNOSTIC_CATEGORIES",
        "RUNTIME_PROVIDER_DIAGNOSTIC_CATEGORIES",
    ):
        assert defining_modules("src/rag_core", name=category) == {
            PROVIDER_CATEGORY_NAMES
        }, category

    # The registry, diagnostics, doctor CLI, and runtime consume the named
    # category constants from the owner instead of raw ProviderRegistry("...")
    # literals or inline category tuples.
    graph = import_graph("src/rag_core")
    for consumer in (
        "rag_core.search.providers.registry",
        "rag_core.search.providers.provider_diagnostics",
        "rag_core.cli.doctor_output",
    ):
        assert (
            f"{PROVIDER_CATEGORY_NAMES}.EMBEDDING_PROVIDER_CATEGORY"
            in graph[consumer]
        ), consumer
    assert (
        f"{PROVIDER_CATEGORY_NAMES}.RERANKER_PROVIDER_CATEGORY"
        in graph["rag_core.cli.doctor_output"]
    )
    assert (
        f"{PROVIDER_CATEGORY_NAMES}.SPARSE_PROVIDER_CATEGORY"
        in graph["rag_core._engine.core_runtime"]
    )


def test_provider_diagnostic_payload_fields_have_single_owner() -> None:
    field_values = {
        "FIELD_API_KEY_CONFIGURED": FIELD_API_KEY_CONFIGURED,
        "FIELD_API_KEY_ENV": FIELD_API_KEY_ENV,
        "FIELD_CONFIGURED": FIELD_CONFIGURED,
        "FIELD_PACKAGE_AVAILABLE": FIELD_PACKAGE_AVAILABLE,
        "FIELD_PROVIDERS": FIELD_PROVIDERS,
        "FIELD_READINESS_SCOPE": FIELD_READINESS_SCOPE,
        "FIELD_REGISTERED": FIELD_REGISTERED,
        "FIELD_RUNTIME_CONFIG": FIELD_RUNTIME_CONFIG,
        "FIELD_MATURITY": FIELD_MATURITY,
    }
    assert field_values == {
        "FIELD_API_KEY_CONFIGURED": "api_key_configured",
        "FIELD_API_KEY_ENV": "api_key_env",
        "FIELD_CONFIGURED": "configured",
        "FIELD_PACKAGE_AVAILABLE": "package_available",
        "FIELD_PROVIDERS": "providers",
        "FIELD_READINESS_SCOPE": "readiness_scope",
        "FIELD_REGISTERED": "registered",
        "FIELD_RUNTIME_CONFIG": "runtime_config",
        "FIELD_MATURITY": "maturity",
    }

    # diagnostic_support is the sole assigner of every payload-field literal: a
    # consumer re-hardcoding e.g. "maturity" would add a second owner module.
    for symbol, value in field_values.items():
        assert modules_assigning_value("src/rag_core", value=value) == {
            DIAGNOSTIC_SUPPORT: [symbol]
        }, symbol

    # Diagnostics consumers import the named fields from the owner.
    graph = import_graph("src/rag_core")
    for consumer in (
        "rag_core.search.providers.provider_category_helpers",
        "rag_core.search.providers.event_sink_category_diagnostics",
        "rag_core.search.providers.model_provider_specs",
        "rag_core.search.providers.provider_diagnostics",
        "rag_core.search.providers.vector_store_diagnostics",
    ):
        imported_fields = {
            i.rsplit(".", 1)[-1]
            for i in graph.get(consumer, set())
            if i.startswith(f"{DIAGNOSTIC_SUPPORT}.FIELD_")
        }
        assert imported_fields, consumer
