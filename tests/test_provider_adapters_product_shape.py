from __future__ import annotations

from rag_core.config.vector_store_config import (
    PGVECTOR_VECTOR_STORE_PROVIDER,
    QDRANT_VECTOR_STORE_PROVIDER,
    TURBOPUFFER_VECTOR_STORE_PROVIDER,
)
from rag_core.search.providers.vector_store_capabilities import (
    BUILTIN_VECTOR_STORE_PROVIDER_ORDER,
    METADATA_FILTER_CAPABILITY_BOOLEAN,
    METADATA_FILTER_CAPABILITY_GEO,
    METADATA_FILTER_CAPABILITY_IN,
    METADATA_FILTER_CAPABILITY_NUMERIC_RANGE,
    METADATA_FILTER_CAPABILITY_STRING_RANGE,
    METADATA_FILTER_CAPABILITY_TERM,
    MEMORY_VECTOR_STORE_PROVIDER,
    QUERY_PLAN_CAPABILITY_BOOST,
    QUERY_PLAN_CAPABILITY_DENSE,
    QUERY_PLAN_CAPABILITY_HYBRID,
    QUERY_PLAN_CAPABILITY_HYBRID_DBSF,
    QUERY_PLAN_CAPABILITY_HYBRID_RRF,
    QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF,
    QUERY_PLAN_CAPABILITY_MMR,
    QUERY_PLAN_CAPABILITY_NESTED_PREFETCH,
    QUERY_PLAN_CAPABILITY_SPARSE,
    describe_metadata_filter_capabilities,
    describe_query_plan_capabilities,
)
from rag_core.search.providers.vector_store_diagnostics import (
    VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM,
    VECTOR_STORE_RUNTIME_FAILED,
    VECTOR_STORE_RUNTIME_HEALTHY,
    VECTOR_STORE_RUNTIME_NOT_REQUESTED,
)

from tests.support.source_graph import (
    defining_modules,
    import_graph,
    modules_importing,
    symbol_module,
    under_module,
)

VECTOR_STORE_CONFIG = "rag_core.config.vector_store_config"
CAPABILITIES = "rag_core.search.providers.vector_store_capabilities"
DIAGNOSTICS = "rag_core.search.providers.vector_store_diagnostics"
SEARCH_TYPES = "rag_core.search.types"

PROVIDER_ROOTS = (
    "src/rag_core/search/providers",
    "src/rag_core/_engine",
    "src/rag_core/cli",
)


def _single_owner(name: str, owner: str) -> None:
    assert defining_modules("src/rag_core", name=name) == {owner}, name


def _imports(graph: dict[str, set[str]], module: str, qualified: str) -> bool:
    return qualified in graph.get(module, set())


def test_vector_store_provider_order_uses_capability_specs() -> None:
    # ---- behavioral / value contract (preserved verbatim) ----
    assert BUILTIN_VECTOR_STORE_PROVIDER_ORDER == (
        QDRANT_VECTOR_STORE_PROVIDER,
        PGVECTOR_VECTOR_STORE_PROVIDER,
        TURBOPUFFER_VECTOR_STORE_PROVIDER,
        MEMORY_VECTOR_STORE_PROVIDER,
    )
    assert QDRANT_VECTOR_STORE_PROVIDER == "qdrant"
    assert PGVECTOR_VECTOR_STORE_PROVIDER == "pgvector"
    assert TURBOPUFFER_VECTOR_STORE_PROVIDER == "turbopuffer"
    assert MEMORY_VECTOR_STORE_PROVIDER == "memory"
    assert QUERY_PLAN_CAPABILITY_DENSE == "dense"
    assert QUERY_PLAN_CAPABILITY_SPARSE == "sparse"
    assert QUERY_PLAN_CAPABILITY_HYBRID == "hybrid"
    assert QUERY_PLAN_CAPABILITY_HYBRID_RRF == "hybrid_rrf"
    assert QUERY_PLAN_CAPABILITY_HYBRID_DBSF == "hybrid_dbsf"
    assert QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF == "hybrid_weighted_rrf"
    assert QUERY_PLAN_CAPABILITY_MMR == "mmr"
    assert QUERY_PLAN_CAPABILITY_NESTED_PREFETCH == "nested_prefetch"
    assert QUERY_PLAN_CAPABILITY_BOOST == "boost"
    assert METADATA_FILTER_CAPABILITY_TERM == "term"
    assert METADATA_FILTER_CAPABILITY_IN == "in"
    assert METADATA_FILTER_CAPABILITY_NUMERIC_RANGE == "numeric_range"
    assert METADATA_FILTER_CAPABILITY_STRING_RANGE == "string_range"
    assert METADATA_FILTER_CAPABILITY_GEO == "geo"
    assert METADATA_FILTER_CAPABILITY_BOOLEAN == "boolean"
    assert VECTOR_STORE_RUNTIME_NOT_REQUESTED == "not_requested"
    assert VECTOR_STORE_RUNTIME_HEALTHY == "healthy"
    assert VECTOR_STORE_RUNTIME_FAILED == "failed"
    assert VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM == "adapter_maximum"

    # ---- single-ownership of the named constants/tuples ----
    # Provider identity strings live in config; the capabilities module owns the
    # "memory" sentinel and the builtin provider order.
    for name in (
        "QDRANT_VECTOR_STORE_PROVIDER",
        "PGVECTOR_VECTOR_STORE_PROVIDER",
        "TURBOPUFFER_VECTOR_STORE_PROVIDER",
    ):
        _single_owner(name, VECTOR_STORE_CONFIG)
    for name in (
        "MEMORY_VECTOR_STORE_PROVIDER",
        "BUILTIN_VECTOR_STORE_PROVIDER_ORDER",
        "QUERY_PLAN_CAPABILITY_HYBRID",
        "QUERY_PLAN_CAPABILITY_HYBRID_RRF",
        "QUERY_PLAN_CAPABILITY_HYBRID_DBSF",
        "QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF",
        "QUERY_PLAN_CAPABILITY_MMR",
        "QUERY_PLAN_CAPABILITY_NESTED_PREFETCH",
        "QUERY_PLAN_CAPABILITY_BOOST",
        "QUERY_PLAN_CAPABILITY_DENSE",
        "QUERY_PLAN_CAPABILITY_SPARSE",
        "METADATA_FILTER_CAPABILITY_TERM",
        "METADATA_FILTER_CAPABILITY_IN",
        "METADATA_FILTER_CAPABILITY_NUMERIC_RANGE",
        "METADATA_FILTER_CAPABILITY_STRING_RANGE",
        "METADATA_FILTER_CAPABILITY_GEO",
        "METADATA_FILTER_CAPABILITY_BOOLEAN",
    ):
        _single_owner(name, CAPABILITIES)
    for name in (
        "VECTOR_STORE_RUNTIME_NOT_REQUESTED",
        "VECTOR_STORE_RUNTIME_HEALTHY",
        "VECTOR_STORE_RUNTIME_FAILED",
        "VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM",
    ):
        _single_owner(name, DIAGNOSTICS)

    # The capability-describer functions have one home and are imported by the
    # runtime rather than re-implemented there (the old test asserted both the
    # `def ... not in runtime` and the call site live).
    assert symbol_module(describe_query_plan_capabilities) == CAPABILITIES
    assert symbol_module(describe_metadata_filter_capabilities) == CAPABILITIES
    _single_owner("describe_query_plan_capabilities", CAPABILITIES)
    _single_owner("describe_metadata_filter_capabilities", CAPABILITIES)

    graph = import_graph(*PROVIDER_ROOTS)

    # ---- consumers reference the named owners (not inline literals) ----
    runtime = "rag_core._engine.core_runtime"
    factory = "rag_core._engine.core_vector_store_factory"
    for module in (runtime, factory):
        for name in (
            "QDRANT_VECTOR_STORE_PROVIDER",
            "PGVECTOR_VECTOR_STORE_PROVIDER",
            "TURBOPUFFER_VECTOR_STORE_PROVIDER",
        ):
            assert _imports(graph, module, f"{VECTOR_STORE_CONFIG}.{name}"), (module, name)
    for name in (
        "describe_query_plan_capabilities",
        "describe_metadata_filter_capabilities",
    ):
        assert _imports(graph, runtime, f"{CAPABILITIES}.{name}")

    diagnostics_module = DIAGNOSTICS
    doctor_output = "rag_core.cli.doctor_output"
    doctor = "rag_core.cli.commands.doctor"
    assert _imports(
        graph, diagnostics_module, f"{CAPABILITIES}.BUILTIN_VECTOR_STORE_PROVIDER_ORDER"
    )
    assert _imports(
        graph, doctor_output, f"{CAPABILITIES}.BUILTIN_VECTOR_STORE_PROVIDER_ORDER"
    )
    assert _imports(
        graph, doctor_output, f"{CAPABILITIES}.QUERY_PLAN_STAGE_CAPABILITY_FIELDS"
    )
    assert _imports(graph, doctor, f"{DIAGNOSTICS}.VECTOR_STORE_RUNTIME_HEALTHY")
    assert _imports(graph, doctor, f"{DIAGNOSTICS}.VECTOR_STORE_RUNTIME_FAILED")

    # ---- each vector-store adapter carries identity via its provider spec ----
    for module, spec in (
        ("rag_core.search.providers.memory_store", "MEMORY_VECTOR_STORE_PROVIDER_SPEC"),
        ("rag_core.search.providers.qdrant_collection", "QDRANT_VECTOR_STORE_PROVIDER_SPEC"),
        (
            "rag_core.search.providers.turbopuffer_client",
            "TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC",
        ),
        ("rag_core.search.providers.pgvector_store", "PGVECTOR_VECTOR_STORE_PROVIDER_SPEC"),
    ):
        assert _imports(graph, module, f"{CAPABILITIES}.{spec}"), module

    # ---- layering: diagnostics must not import up into the private engine ----
    assert (
        modules_importing(
            "src/rag_core/search/providers",
            predicate=under_module("rag_core._engine"),
        )
        == {}
    )


def _assert_no_stale_catch_all(*relative_roots: str) -> None:
    assert (
        modules_importing(*relative_roots, predicate=under_module(SEARCH_TYPES)) == {}
    )


def test_qdrant_adapter_imports_search_contract_owners_directly() -> None:
    _assert_no_stale_catch_all("src/rag_core/search/providers")
    graph = import_graph("src/rag_core/search/providers")
    assert any(
        i.startswith("rag_core.search.filters")
        for i in graph["rag_core.search.providers.qdrant_filters"]
    )
    assert any(
        i.startswith("rag_core.search.request_models")
        for i in graph["rag_core.search.providers.qdrant_store"]
    )
    assert any(
        i.startswith("rag_core.search.vector_models")
        for i in graph["rag_core.search.providers.qdrant_payloads"]
    )


def test_turbopuffer_adapter_imports_search_contract_owners_directly() -> None:
    _assert_no_stale_catch_all("src/rag_core/search/providers")
    graph = import_graph("src/rag_core/search/providers")
    assert any(
        i.startswith("rag_core.search.filters")
        for i in graph["rag_core.search.providers.turbopuffer_payloads"]
    )
    store = graph["rag_core.search.providers.turbopuffer_store"]
    assert any(i.startswith("rag_core.search.provider_protocols") for i in store)
    assert any(i.startswith("rag_core.search.request_models") for i in store)
    assert any(
        i.startswith("rag_core.search.vector_models")
        for i in graph["rag_core.search.providers.turbopuffer_payloads"]
    )


def test_pgvector_adapter_imports_search_contract_owners_directly() -> None:
    _assert_no_stale_catch_all("src/rag_core/search/providers")
    graph = import_graph("src/rag_core/search/providers")
    assert any(
        i.startswith("rag_core.search.filters")
        for i in graph["rag_core.search.providers.pgvector_filters"]
    )
    store = graph["rag_core.search.providers.pgvector_store"]
    assert any(i.startswith("rag_core.search.provider_protocols") for i in store)
    assert any(i.startswith("rag_core.search.request_models") for i in store)
    assert any(
        i.startswith("rag_core.search.vector_models")
        for i in graph["rag_core.search.providers.pgvector_payloads"]
    )


def test_memory_adapter_imports_search_contract_owners_directly() -> None:
    _assert_no_stale_catch_all("src/rag_core/search/providers")
    graph = import_graph("src/rag_core/search/providers")
    store = graph["rag_core.search.providers.memory_store"]
    assert any(i.startswith("rag_core.search.provider_protocols") for i in store)
    assert any(i.startswith("rag_core.search.request_models") for i in store)
    assert any(
        i.startswith("rag_core.search.request_models")
        for i in graph["rag_core.search.providers.memory_filters"]
    )
    assert any(
        i.startswith("rag_core.search.vector_models")
        for i in graph["rag_core.search.providers.memory_query_scoring"]
    )


def test_model_provider_modules_import_search_contract_owners_directly() -> None:
    _assert_no_stale_catch_all("src/rag_core/search/providers")
    graph = import_graph("src/rag_core/search/providers")
    for name in ("cohere", "rerank_results", "reranker", "voyage", "zeroentropy"):
        module = f"rag_core.search.providers.{name}"
        assert any(
            i.startswith("rag_core.search.request_models") for i in graph[module]
        ), name
    for name in ("sparse", "vector_dimensions"):
        module = f"rag_core.search.providers.{name}"
        assert any(
            i.startswith("rag_core.search.vector_models") for i in graph[module]
        ), name
