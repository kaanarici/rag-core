from __future__ import annotations

from pathlib import Path

from rag_core.search.providers.diagnostic_support import (
    FIELD_API_KEY_CONFIGURED,
    FIELD_API_KEY_ENV,
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_PROVIDERS,
    FIELD_READINESS_SCOPE,
    FIELD_REGISTERED,
    FIELD_RUNTIME_CONFIG,
    FIELD_SUPPORT_LEVEL,
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


def test_provider_result_value_type_labels_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/provider_result_values.py",
            "src/rag_core/search/providers/embedding_results.py",
            "src/rag_core/search/providers/rerank_results.py",
        )
    }

    owner = sources["src/rag_core/search/providers/provider_result_values.py"]
    assert owner.count("def safe_provider_value_type") == 1
    assert 'return "none"' in owner
    for path in (
        "src/rag_core/search/providers/embedding_results.py",
        "src/rag_core/search/providers/rerank_results.py",
    ):
        assert "safe_provider_value_type" in sources[path]
        assert "def _safe_value_type" not in sources[path]
        assert 'return "none"' not in sources[path]




def test_provider_diagnostic_category_names_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/search/providers/provider_category_names.py",
            "src/rag_core/search/providers/registry.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "src/rag_core/cli_doctor_output.py",
            "src/rag_core/_engine/core_runtime.py",
        )
    }
    owner = sources["src/rag_core/search/providers/provider_category_names.py"]
    registry = sources["src/rag_core/search/providers/registry.py"]
    diagnostics = sources["src/rag_core/search/providers/model_provider_diagnostics.py"]
    doctor_output = sources["src/rag_core/cli_doctor_output.py"]

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
    for assignment in (
        'EMBEDDING_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "embedding"',
        'SPARSE_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "sparse"',
        'RERANKER_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "reranker"',
        'OCR_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "ocr"',
        (
            "CONTEXTUALIZER_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "
            '"contextualizer"'
        ),
        '"embedding_cache"',
        '"chunk_context_cache"',
        'SEARCH_SIDECAR_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "search_sidecar"',
        'EVENT_SINK_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "event_sink"',
        'VECTOR_STORE_PROVIDER_CATEGORY: Final[ProviderDiagnosticCategory] = "vector_store"',
    ):
        assert assignment in owner
    for raw_registry in (
        'ProviderRegistry("embedding")',
        'ProviderRegistry("reranker")',
        'ProviderRegistry("sparse")',
        'ProviderRegistry("ocr")',
        'ProviderRegistry("contextualizer")',
        'ProviderRegistry("vector_store")',
        'ProviderRegistry("search_sidecar")',
        'ProviderRegistry("embedding_cache")',
        'ProviderRegistry("chunk_context_cache")',
    ):
        assert raw_registry not in registry
    assert "EMBEDDING_PROVIDER_CATEGORY" in diagnostics
    assert "SEARCH_SIDECAR_PROVIDER_CATEGORY" in diagnostics
    assert "SPARSE_PROVIDER_CATEGORY" in sources["src/rag_core/_engine/core_runtime.py"]
    assert 'providers.get("embedding")' not in doctor_output
    assert '_emit_provider_category_summary("embedding"' not in doctor_output
    assert '("sparse", SPARSE_EMBEDDER_PROVIDER_ORDER)' not in doctor_output
    assert "RERANKER_PROVIDER_CATEGORY" in doctor_output




def test_provider_diagnostic_payload_fields_have_single_owner() -> None:
    owner_path = "src/rag_core/search/providers/diagnostic_support.py"
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            owner_path,
            "src/rag_core/search/providers/provider_category_helpers.py",
            "src/rag_core/search/providers/event_sink_category_diagnostics.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "src/rag_core/search/providers/vector_store_diagnostics.py",
        )
    }

    field_values = {
        "FIELD_API_KEY_CONFIGURED": FIELD_API_KEY_CONFIGURED,
        "FIELD_API_KEY_ENV": FIELD_API_KEY_ENV,
        "FIELD_CONFIGURED": FIELD_CONFIGURED,
        "FIELD_PACKAGE_AVAILABLE": FIELD_PACKAGE_AVAILABLE,
        "FIELD_PROVIDERS": FIELD_PROVIDERS,
        "FIELD_READINESS_SCOPE": FIELD_READINESS_SCOPE,
        "FIELD_REGISTERED": FIELD_REGISTERED,
        "FIELD_RUNTIME_CONFIG": FIELD_RUNTIME_CONFIG,
        "FIELD_SUPPORT_LEVEL": FIELD_SUPPORT_LEVEL,
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
        "FIELD_SUPPORT_LEVEL": "support_level",
    }

    owner = sources[owner_path]
    for symbol, value in field_values.items():
        assert f'{symbol}: ProviderDiagnosticField = "{value}"' in owner
        assert f'"{symbol}"' in owner

    consumers = "\n".join(
        source for path, source in sources.items() if path != owner_path
    )
    for symbol in field_values:
        assert symbol in consumers
    for value in field_values.values():
        assert f'"{value}":' not in consumers
