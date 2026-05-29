from __future__ import annotations

from pathlib import Path

from rag_core.config import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION,
    DEFAULT_RERANKER_PROVIDER,
    DEFAULT_TURBOPUFFER_DISTANCE_METRIC,
    DEFAULT_VECTOR_STORE_PROVIDER,
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


def test_turbopuffer_distance_metric_default_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/vector_store_config.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/_engine/core_config_cli.py",
            "src/rag_core/search/providers/turbopuffer_config.py",
            "src/rag_core/search/providers/turbopuffer_store.py",
        )
    }

    assert DEFAULT_TURBOPUFFER_DISTANCE_METRIC == "cosine_distance"
    assert (
        sources["src/rag_core/config/vector_store_config.py"].count(
            'DEFAULT_TURBOPUFFER_DISTANCE_METRIC = "cosine_distance"'
        )
        == 1
    )
    assert (
        'distance_metric: str = "cosine_distance"'
        not in sources["src/rag_core/config/vector_store_config.py"]
    )
    forbidden_default_copies = (
        'default="cosine_distance"',
        'or "cosine_distance"',
        'distance_metric: str = "cosine_distance"',
        '_SUPPORTED_DISTANCE_METRICS = frozenset({"cosine_distance", "euclidean_squared"})',
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/vector_store_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source




def test_vector_store_default_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/vector_store_config.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/_engine/core_config_cli.py",
            "src/rag_core/search/providers/vector_store_diagnostics.py",
            "tests/support/fakes.py",
        )
    }

    assert DEFAULT_VECTOR_STORE_PROVIDER == "qdrant"
    assert (
        sources["src/rag_core/config/vector_store_config.py"].count(
            "DEFAULT_VECTOR_STORE_PROVIDER = QDRANT_VECTOR_STORE_PROVIDER"
        )
        == 1
    )
    forbidden_default_copies = (
        'env_or_default("RAG_CORE_VECTOR_STORE", "qdrant")',
        '_arg(args, "vector_store", default="qdrant")',
        'vector_store_provider: str = "qdrant"',
        '"default": "qdrant"',
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/vector_store_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source




def test_qdrant_collection_default_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/qdrant_config.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/_engine/core_config_cli.py",
            "tests/support/fakes.py",
        )
    }

    assert DEFAULT_QDRANT_COLLECTION == "rag_core_chunks"
    assert (
        sources["src/rag_core/config/qdrant_config.py"].count(
            'DEFAULT_QDRANT_COLLECTION = "rag_core_chunks"'
        )
        == 1
    )
    forbidden_default_copies = (
        'collection: str = "rag_core_chunks"',
        'env_or_default("RAG_CORE_QDRANT_COLLECTION", "rag_core_chunks")',
        '_arg(args, "qdrant_collection", default="rag_core_chunks")',
        'qdrant_collection: str = "rag_core_chunks"',
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/qdrant_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source




def test_qdrant_dimension_aware_default_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/qdrant_config.py",
            "src/rag_core/_engine/core_config_cli.py",
            "tests/support/fakes.py",
        )
    }

    assert DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION is True
    assert (
        sources["src/rag_core/config/qdrant_config.py"].count(
            "DEFAULT_QDRANT_DIMENSION_AWARE_COLLECTION = True"
        )
        == 1
    )
    forbidden_default_copies = (
        "dimension_aware_collection: bool = True",
        "default=True",
        "qdrant_dimension_aware_collection: bool = True",
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/qdrant_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source




def test_reranker_default_provider_has_single_config_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/reranker_config.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/_engine/core_config_cli.py",
            "src/rag_core/search/providers/reranker.py",
            "src/rag_core/search/providers/reranker_resolution.py",
            "src/rag_core/search/providers/model_provider_diagnostics.py",
            "tests/support/fakes.py",
        )
    }

    assert DEFAULT_RERANKER_PROVIDER == "none"
    assert (
        sources["src/rag_core/config/reranker_config.py"].count(
            'DEFAULT_RERANKER_PROVIDER = "none"'
        )
        == 1
    )
    forbidden_default_copies = (
        'provider: str = "none"',
        'env_or_default(RERANKER_PROVIDER_ENV, "none")',
        '_arg(args, "reranker_provider", default="none")',
        'reranker_provider: str = "none"',
        'provider or "none"',
    )
    for path, source in sources.items():
        if path == "src/rag_core/config/reranker_config.py":
            continue
        for forbidden in forbidden_default_copies:
            assert forbidden not in source
