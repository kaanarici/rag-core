from __future__ import annotations

from pathlib import Path

from rag_core.config.vector_store_config import (
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
)
from rag_core.search.providers.vector_store_diagnostics import (
    VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM,
    VECTOR_STORE_RUNTIME_FAILED,
    VECTOR_STORE_RUNTIME_HEALTHY,
    VECTOR_STORE_RUNTIME_NOT_REQUESTED,
)

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_vector_store_provider_order_uses_capability_specs() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/vector_store_config.py",
            "src/rag_core/search/providers/vector_store_capabilities.py",
            "src/rag_core/search/providers/vector_store_diagnostics.py",
            "src/rag_core/search/providers/memory_store.py",
            "src/rag_core/search/providers/qdrant_health.py",
            "src/rag_core/search/providers/qdrant_lifecycle.py",
            "src/rag_core/search/providers/qdrant_runtime.py",
            "src/rag_core/search/providers/qdrant_store_guards.py",
            "src/rag_core/search/providers/qdrant_write_logging.py",
            "src/rag_core/search/providers/qdrant_store.py",
            "src/rag_core/search/providers/turbopuffer_health.py",
            "src/rag_core/search/providers/turbopuffer_search.py",
            "src/rag_core/search/providers/turbopuffer_store.py",
            "src/rag_core/search/providers/turbopuffer_write.py",
            "src/rag_core/_engine/core_runtime.py",
            "src/rag_core/search/planning.py",
            "src/rag_core/search/query_plan_presets.py",
            "src/rag_core/_engine/core_vector_store_factory.py",
            "src/rag_core/cli_doctor.py",
            "src/rag_core/cli_doctor_output.py",
            "tests/test_cli.py",
            "tests/test_runtime_query_plan_diagnostics.py",
        )
    }

    assert BUILTIN_VECTOR_STORE_PROVIDER_ORDER == (
        QDRANT_VECTOR_STORE_PROVIDER,
        TURBOPUFFER_VECTOR_STORE_PROVIDER,
        MEMORY_VECTOR_STORE_PROVIDER,
    )
    capabilities = sources["src/rag_core/search/providers/vector_store_capabilities.py"]
    assert QDRANT_VECTOR_STORE_PROVIDER == "qdrant"
    assert TURBOPUFFER_VECTOR_STORE_PROVIDER == "turbopuffer"
    assert MEMORY_VECTOR_STORE_PROVIDER == "memory"
    assert (
        'QDRANT_VECTOR_STORE_PROVIDER = "qdrant"'
        in sources["src/rag_core/config/vector_store_config.py"]
    )
    assert (
        'TURBOPUFFER_VECTOR_STORE_PROVIDER = "turbopuffer"'
        in sources["src/rag_core/config/vector_store_config.py"]
    )
    assert 'MEMORY_VECTOR_STORE_PROVIDER = "memory"' in capabilities
    assert "BUILTIN_VECTOR_STORE_PROVIDER_ORDER = tuple(" in capabilities
    assert 'name="qdrant"' not in capabilities
    assert 'name="turbopuffer"' not in capabilities
    assert 'name="memory"' not in capabilities
    diagnostics = sources["src/rag_core/search/providers/vector_store_diagnostics.py"]
    runtime = sources["src/rag_core/_engine/core_runtime.py"]
    factory = sources["src/rag_core/_engine/core_vector_store_factory.py"]
    cli_doctor = sources["src/rag_core/cli_doctor.py"]
    doctor_output = sources["src/rag_core/cli_doctor_output.py"]
    assert "BUILTIN_VECTOR_STORE_PROVIDER_ORDER" in diagnostics
    assert "BUILTIN_VECTOR_STORE_PROVIDER_ORDER" in doctor_output
    assert "def describe_query_plan_capabilities(" in capabilities
    assert "def describe_metadata_filter_capabilities(" in capabilities
    assert "def describe_query_plan_capabilities(" not in runtime
    assert "def describe_metadata_filter_capabilities(" not in runtime
    assert "describe_query_plan_capabilities" in runtime
    assert "describe_metadata_filter_capabilities" in runtime
    for path in (
        "src/rag_core/_engine/core_runtime.py",
        "src/rag_core/search/planning.py",
        "src/rag_core/search/query_plan_presets.py",
        "src/rag_core/search/providers/qdrant_health.py",
        "src/rag_core/search/providers/vector_store_capabilities.py",
    ):
        source = sources[path]
        assert "rag_core.search.provider_protocols" in source
        assert "from rag_core.search.types import QueryPlanCapabilities" not in source
        assert "from rag_core.search.types import StoreCapabilities" not in source
        assert "from rag_core.search.types import MetadataFilterCapabilities" not in source
    assert "from rag_core._engine.core_runtime import" not in diagnostics
    assert "QUERY_PLAN_STAGE_CAPABILITY_FIELDS" in doctor_output
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
    for definition in (
        "QUERY_PLAN_CAPABILITY_DENSE: Final[str] = DENSE_RETRIEVAL_CHANNEL",
        "QUERY_PLAN_CAPABILITY_SPARSE: Final[str] = SPARSE_RETRIEVAL_CHANNEL",
        'QUERY_PLAN_CAPABILITY_HYBRID: Final[str] = "hybrid"',
        'QUERY_PLAN_CAPABILITY_HYBRID_RRF: Final[str] = "hybrid_rrf"',
        'QUERY_PLAN_CAPABILITY_HYBRID_DBSF: Final[str] = "hybrid_dbsf"',
        'QUERY_PLAN_CAPABILITY_HYBRID_WEIGHTED_RRF: Final[str] = "hybrid_weighted_rrf"',
        'QUERY_PLAN_CAPABILITY_MMR: Final[str] = "mmr"',
        'QUERY_PLAN_CAPABILITY_NESTED_PREFETCH: Final[str] = "nested_prefetch"',
        'QUERY_PLAN_CAPABILITY_BOOST: Final[str] = "boost"',
        'METADATA_FILTER_CAPABILITY_TERM: Final[str] = "term"',
        'METADATA_FILTER_CAPABILITY_IN: Final[str] = "in"',
        'METADATA_FILTER_CAPABILITY_NUMERIC_RANGE: Final[str] = "numeric_range"',
        'METADATA_FILTER_CAPABILITY_STRING_RANGE: Final[str] = "string_range"',
        'METADATA_FILTER_CAPABILITY_GEO: Final[str] = "geo"',
        'METADATA_FILTER_CAPABILITY_BOOLEAN: Final[str] = "boolean"',
    ):
        assert capabilities.count(definition) == 1
    for definition in (
        "VECTOR_STORE_RUNTIME_NOT_REQUESTED: Final[VectorStoreRuntimeValidation] = (",
        'VECTOR_STORE_RUNTIME_HEALTHY: Final[VectorStoreRuntimeValidation] = "healthy"',
        'VECTOR_STORE_RUNTIME_FAILED: Final[VectorStoreRuntimeValidation] = "failed"',
        "VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM: Final[VectorStoreQueryPlanScope]",
    ):
        assert diagnostics.count(definition) == 1
    assert '"package_present":' not in diagnostics
    assert '"credential_present":' not in diagnostics
    label_consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/search/providers/vector_store_diagnostics.py"
    )
    for symbol in (
        "VECTOR_STORE_RUNTIME_NOT_REQUESTED",
        "VECTOR_STORE_RUNTIME_HEALTHY",
        "VECTOR_STORE_RUNTIME_FAILED",
        "VECTOR_STORE_QUERY_PLAN_SCOPE_ADAPTER_MAXIMUM",
    ):
        assert symbol in label_consumers
    for duplicate in (
        '"runtime_validation": "not_requested"',
        'runtime_validation"] == "not_requested"',
        '"healthy" if healthy else "failed"',
        'runtime_validation"] == "healthy"',
        'runtime_validation"] == "failed"',
        '"query_plan_scope": "adapter_maximum"',
        'query_plan_scope"] == "adapter_maximum"',
        '"package_present":',
        '"credential_present":',
        'package_present"]',
        'credential_present"]',
    ):
        assert duplicate not in label_consumers
    for duplicate in (
        '"hybrid": capabilities.hybrid',
        '"hybrid_rrf": capabilities.hybrid_rrf',
        '"hybrid_dbsf": capabilities.hybrid_dbsf',
        '"hybrid_weighted_rrf": capabilities.hybrid_weighted_rrf',
        '"mmr": capabilities.mmr',
        '"nested_prefetch": capabilities.nested_prefetch',
        '"boost": capabilities.boost',
        '"term": capabilities.term',
        '"in": capabilities.in_',
        '"numeric_range": capabilities.numeric_range',
        '"string_range": capabilities.string_range',
        '"geo": capabilities.geo',
        '"boolean": capabilities.boolean',
        'field != "hybrid"',
    ):
        assert duplicate not in capabilities
    assert "QDRANT_VECTOR_STORE_PROVIDER" in runtime
    assert "TURBOPUFFER_VECTOR_STORE_PROVIDER" in runtime
    assert "QDRANT_VECTOR_STORE_PROVIDER" in factory
    assert "TURBOPUFFER_VECTOR_STORE_PROVIDER" in factory
    assert "VECTOR_STORE_RUNTIME_HEALTHY" in cli_doctor
    assert "VECTOR_STORE_RUNTIME_FAILED" in cli_doctor
    assert 'for provider_name in ("qdrant", "turbopuffer", "memory")' not in (
        doctor_output
    )
    assert (
        'VECTOR_STORES.register("memory"'
        not in sources["src/rag_core/search/providers/memory_store.py"]
    )
    assert (
        'VECTOR_STORES.register("qdrant"'
        not in sources["src/rag_core/search/providers/qdrant_store.py"]
    )
    assert (
        '"turbopuffer",\n    lambda **kw: TurboPufferVectorStore(**kw),'
        not in sources["src/rag_core/search/providers/turbopuffer_store.py"]
    )
    assert '"providers": {\n            "qdrant":' not in diagnostics
    assert '"extra": "turbopuffer"' not in diagnostics
    provider_identity_consumers = "\n".join(
        sources[path]
        for path in (
            "src/rag_core/search/providers/memory_store.py",
            "src/rag_core/search/providers/qdrant_health.py",
            "src/rag_core/search/providers/qdrant_lifecycle.py",
            "src/rag_core/search/providers/qdrant_runtime.py",
            "src/rag_core/search/providers/qdrant_store.py",
            "src/rag_core/search/providers/qdrant_store_guards.py",
            "src/rag_core/search/providers/qdrant_write_logging.py",
            "src/rag_core/search/providers/turbopuffer_health.py",
            "src/rag_core/search/providers/turbopuffer_search.py",
            "src/rag_core/search/providers/turbopuffer_write.py",
        )
    )
    for symbol in (
        "MEMORY_VECTOR_STORE_PROVIDER_SPEC.name",
        "QDRANT_VECTOR_STORE_PROVIDER_SPEC.name",
        "TURBOPUFFER_VECTOR_STORE_PROVIDER_SPEC.name",
    ):
        assert symbol in provider_identity_consumers
    for duplicate in (
        '"adapter": "memory"',
        '"adapter": "qdrant"',
        '"adapter": "turbopuffer"',
        'provider_name="qdrant"',
        'provider_name="turbopuffer"',
        "provider=qdrant",
        "provider=turbopuffer",
    ):
        assert duplicate not in provider_identity_consumers
    for source in (runtime, factory):
        assert 'config.vector_store.provider == "qdrant"' not in source
        assert 'config.vector_store.provider == "turbopuffer"' not in source
        assert 'vector_stores.create(\n            "qdrant"' not in source
        assert 'vector_stores.create(\n            "turbopuffer"' not in source




def test_qdrant_adapter_imports_search_contract_owners_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    provider_root = root / "src" / "rag_core" / "search" / "providers"
    qdrant_sources = {
        path: path.read_text(encoding="utf-8")
        for path in provider_root.glob("qdrant_*.py")
    }

    assert qdrant_sources
    for path, source in qdrant_sources.items():
        assert "from rag_core.search.types import" not in source, path
    assert "rag_core.search.filters" in qdrant_sources[
        provider_root / "qdrant_metadata_filters.py"
    ]
    assert "rag_core.search.request_models" in qdrant_sources[
        provider_root / "qdrant_store.py"
    ]
    assert "rag_core.search.vector_models" in qdrant_sources[
        provider_root / "qdrant_payloads.py"
    ]




def test_turbopuffer_adapter_imports_search_contract_owners_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    provider_root = root / "src" / "rag_core" / "search" / "providers"
    turbopuffer_sources = {
        path: path.read_text(encoding="utf-8")
        for path in provider_root.glob("turbopuffer_*.py")
    }

    assert turbopuffer_sources
    for path, source in turbopuffer_sources.items():
        assert "from rag_core.search.types import" not in source, path
    assert "rag_core.search.filters" in turbopuffer_sources[
        provider_root / "turbopuffer_filters.py"
    ]
    assert "rag_core.search.provider_protocols" in turbopuffer_sources[
        provider_root / "turbopuffer_store.py"
    ]
    assert "rag_core.search.request_models" in turbopuffer_sources[
        provider_root / "turbopuffer_store.py"
    ]
    assert "rag_core.search.vector_models" in turbopuffer_sources[
        provider_root / "turbopuffer_payloads.py"
    ]




def test_memory_adapter_imports_search_contract_owners_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    provider_root = root / "src" / "rag_core" / "search" / "providers"
    memory_sources = {
        path: path.read_text(encoding="utf-8")
        for path in provider_root.glob("memory_*.py")
    }

    assert memory_sources
    for path, source in memory_sources.items():
        assert "from rag_core.search.types import" not in source, path
    assert "rag_core.search.provider_protocols" in memory_sources[
        provider_root / "memory_store.py"
    ]
    assert "rag_core.search.request_models" in memory_sources[
        provider_root / "memory_store.py"
    ]
    assert "rag_core.search.request_models" in memory_sources[
        provider_root / "memory_filters.py"
    ]
    assert "rag_core.search.vector_models" in memory_sources[
        provider_root / "memory_query_scoring.py"
    ]




def test_model_provider_modules_import_search_contract_owners_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    provider_root = root / "src" / "rag_core" / "search" / "providers"
    relative_paths = (
        "cohere.py",
        "rerank_results.py",
        "reranker.py",
        "sparse.py",
        "vector_dimensions.py",
        "voyage.py",
        "zeroentropy.py",
    )
    sources = {
        path: (provider_root / path).read_text(encoding="utf-8")
        for path in relative_paths
    }

    for path, source in sources.items():
        assert "from rag_core.search.types import" not in source, path
    for path in (
        "cohere.py",
        "rerank_results.py",
        "reranker.py",
        "voyage.py",
        "zeroentropy.py",
    ):
        assert "rag_core.search.request_models" in sources[path], path
    for path in ("sparse.py", "vector_dimensions.py"):
        assert "rag_core.search.vector_models" in sources[path], path
