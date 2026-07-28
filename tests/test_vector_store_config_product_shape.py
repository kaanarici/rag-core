from __future__ import annotations

from rag_core.config import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
    DEFAULT_VECTOR_STORE_PROVIDER,
)

from tests.support.source_graph import (
    defining_modules,
    import_graph,
    modules_assigning_value,
)

VECTOR_STORE_CONFIG = "rag_core.config.vector_store_config"
QDRANT_CONFIG = "rag_core.config.qdrant_config"
RERANKER_CONFIG = "rag_core.config.reranker_config"


def _consumers_importing(symbol: str, *modules: str) -> None:
    """Assert each module imports ``symbol`` (the named default) somewhere upstream.

    Replaces the old "forbidden literal copy not in source" scrapes: the durable
    invariant is that config consumers reference the single named default instead
    of re-deriving the value, which survives renames of the import path.
    """
    graph = import_graph("src/rag_core")
    for module in modules:
        hits = [i for i in graph.get(module, set()) if i.endswith(f".{symbol}")]
        assert hits, (module, symbol)


def test_turbopuffer_distance_metric_default_has_single_config_owner() -> None:
    assert DEFAULT_TURBOPUFFER_DISTANCE_METRIC == "cosine_distance"

    # vector_store_config is the sole owner: it defines the named default and is
    # the only module that binds the "cosine_distance" literal to a constant.
    assert defining_modules(
        "src/rag_core", name="DEFAULT_TURBOPUFFER_DISTANCE_METRIC"
    ) == {VECTOR_STORE_CONFIG}
    assert modules_assigning_value("src/rag_core", value="cosine_distance") == {
        VECTOR_STORE_CONFIG: ["DEFAULT_TURBOPUFFER_DISTANCE_METRIC"]
    }
    _consumers_importing(
        "DEFAULT_TURBOPUFFER_DISTANCE_METRIC",
        "rag_core.cli.parsers.config",
        "rag_core._engine.core_config_cli",
        "rag_core.search.providers.turbopuffer_store",
    )


def test_vector_store_default_has_single_config_owner() -> None:
    assert DEFAULT_VECTOR_STORE_PROVIDER == "qdrant"

    # The provider default is an alias of QDRANT_VECTOR_STORE_PROVIDER and lives
    # only in vector_store_config; the "qdrant" literal itself has one owner.
    assert defining_modules(
        "src/rag_core", name="DEFAULT_VECTOR_STORE_PROVIDER"
    ) == {VECTOR_STORE_CONFIG}
    assert modules_assigning_value("src/rag_core", value="qdrant") == {
        VECTOR_STORE_CONFIG: ["QDRANT_VECTOR_STORE_PROVIDER"]
    }
    _consumers_importing(
        "DEFAULT_VECTOR_STORE_PROVIDER",
        "rag_core.cli.parsers.config",
        "rag_core._engine.core_config_cli",
        "rag_core.search.providers.vector_store_diagnostics",
    )


def test_qdrant_collection_default_has_single_config_owner() -> None:
    assert DEFAULT_QDRANT_COLLECTION == "rag_core_chunks"

    # qdrant_config owns both the named default and the "rag_core_chunks" literal.
    assert defining_modules(
        "src/rag_core", name="DEFAULT_QDRANT_COLLECTION"
    ) == {QDRANT_CONFIG}
    assert modules_assigning_value("src/rag_core", value="rag_core_chunks") == {
        QDRANT_CONFIG: ["DEFAULT_QDRANT_COLLECTION"]
    }
    _consumers_importing(
        "DEFAULT_QDRANT_COLLECTION",
        "rag_core.cli.parsers.config",
        "rag_core._engine.core_config_cli",
    )


def test_qdrant_dimension_aware_default_has_single_config_owner() -> None:
    assert DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION is True

    # Boolean default: a literal-value scan would match every `= True` in the
    # tree, so ownership is the AST definition site -- only qdrant_config binds it.
    assert defining_modules(
        "src/rag_core", name="DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION"
    ) == {QDRANT_CONFIG}
    _consumers_importing(
        "DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION",
        "rag_core._engine.core_config_cli",
    )


def test_reranker_default_provider_has_single_config_owner() -> None:
    assert DEFAULT_RERANKER_PROVIDER == "none"

    # "none" is a shared disabled-sentinel across several provider domains, so
    # ownership here is the AST definition site (not the shared literal): only
    # reranker_config defines DEFAULT_RERANKER_PROVIDER, and consumers import it.
    assert defining_modules(
        "src/rag_core", name="DEFAULT_RERANKER_PROVIDER"
    ) == {RERANKER_CONFIG}
    _consumers_importing(
        "DEFAULT_RERANKER_PROVIDER",
        "rag_core.cli.parsers.config",
        "rag_core._engine.core_config_cli",
        "rag_core.search.providers.reranker",
        "rag_core.search.providers.reranker_resolution",
        "rag_core.search.providers.provider_diagnostics",
    )
