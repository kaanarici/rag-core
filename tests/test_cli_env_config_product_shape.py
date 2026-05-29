from __future__ import annotations

from pathlib import Path

from rag_core.config import (
    CLI_MANIFEST_DIR_ENV,
    EMBEDDING_BATCH_SIZE_ENV,
    EMBEDDING_DIMENSIONS_ENV,
    EMBEDDING_MODEL_ENV,
    EMBEDDING_PROVIDER_ENV,
    PROCESSING_VERSION_ENV,
    RERANKER_MODEL_ENV,
    RERANKER_PROVIDER_ENV,
)
from rag_core.fetch_security import (
    FETCH_ALLOW_HTTP_ENV,
    FETCH_ALLOW_PRIVATE_ADDRESSES_ENV,
    FETCH_MAX_BYTES_ENV,
    FETCH_MAX_REDIRECTS_ENV,
    FETCH_TIMEOUT_SECONDS_ENV,
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


def test_cli_config_env_names_have_config_owners() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/config/embedding_config.py",
            "src/rag_core/config/ingest_config.py",
            "src/rag_core/config/reranker_config.py",
            "src/rag_core/config/__init__.py",
            "src/rag_core/cli_config_parser.py",
            "src/rag_core/_engine/core_config_cli.py",
            "src/rag_core/cli_parser.py",
        )
    }

    assert EMBEDDING_PROVIDER_ENV == "RAG_CORE_EMBEDDING_PROVIDER"
    assert EMBEDDING_MODEL_ENV == "RAG_CORE_EMBEDDING_MODEL"
    assert EMBEDDING_DIMENSIONS_ENV == "RAG_CORE_EMBEDDING_DIMENSIONS"
    assert EMBEDDING_BATCH_SIZE_ENV == "RAG_CORE_EMBEDDING_BATCH_SIZE"
    assert RERANKER_PROVIDER_ENV == "RAG_CORE_RERANKER_PROVIDER"
    assert RERANKER_MODEL_ENV == "RAG_CORE_RERANKER_MODEL"
    assert PROCESSING_VERSION_ENV == "RAG_CORE_PROCESSING_VERSION"
    assert CLI_MANIFEST_DIR_ENV == "RAG_CORE_MANIFEST_DIR"

    embedding_owner = sources["src/rag_core/config/embedding_config.py"]
    assert (
        embedding_owner.count('EMBEDDING_PROVIDER_ENV = "RAG_CORE_EMBEDDING_PROVIDER"')
        == 1
    )
    assert (
        embedding_owner.count('EMBEDDING_MODEL_ENV = "RAG_CORE_EMBEDDING_MODEL"') == 1
    )
    assert (
        embedding_owner.count(
            'EMBEDDING_DIMENSIONS_ENV = "RAG_CORE_EMBEDDING_DIMENSIONS"'
        )
        == 1
    )
    assert (
        embedding_owner.count(
            'EMBEDDING_BATCH_SIZE_ENV = "RAG_CORE_EMBEDDING_BATCH_SIZE"'
        )
        == 1
    )
    reranker_owner = sources["src/rag_core/config/reranker_config.py"]
    assert (
        reranker_owner.count('RERANKER_PROVIDER_ENV = "RAG_CORE_RERANKER_PROVIDER"')
        == 1
    )
    assert reranker_owner.count('RERANKER_MODEL_ENV = "RAG_CORE_RERANKER_MODEL"') == 1
    ingest_owner = sources["src/rag_core/config/ingest_config.py"]
    assert (
        ingest_owner.count('PROCESSING_VERSION_ENV = "RAG_CORE_PROCESSING_VERSION"')
        == 1
    )
    assert ingest_owner.count('CLI_MANIFEST_DIR_ENV = "RAG_CORE_MANIFEST_DIR"') == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path
        not in {
            "src/rag_core/config/embedding_config.py",
            "src/rag_core/config/ingest_config.py",
            "src/rag_core/config/reranker_config.py",
        }
    )
    for symbol in (
        "EMBEDDING_PROVIDER_ENV",
        "EMBEDDING_MODEL_ENV",
        "EMBEDDING_DIMENSIONS_ENV",
        "EMBEDDING_BATCH_SIZE_ENV",
        "RERANKER_PROVIDER_ENV",
        "RERANKER_MODEL_ENV",
        "PROCESSING_VERSION_ENV",
        "CLI_MANIFEST_DIR_ENV",
    ):
        assert symbol in consumers
    for duplicate in (
        '"RAG_CORE_EMBEDDING_PROVIDER"',
        '"RAG_CORE_EMBEDDING_MODEL"',
        '"RAG_CORE_EMBEDDING_DIMENSIONS"',
        '"RAG_CORE_EMBEDDING_BATCH_SIZE"',
        '"RAG_CORE_RERANKER_PROVIDER"',
        '"RAG_CORE_RERANKER_MODEL"',
        '"RAG_CORE_PROCESSING_VERSION"',
        '"RAG_CORE_MANIFEST_DIR"',
    ):
        assert duplicate not in consumers





def test_fetch_env_names_have_fetch_security_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/fetch_security.py",
            "src/rag_core/cli_remote_fetch.py",
        )
    }

    assert FETCH_ALLOW_HTTP_ENV == "RAG_CORE_FETCH_ALLOW_HTTP"
    assert FETCH_ALLOW_PRIVATE_ADDRESSES_ENV == "RAG_CORE_FETCH_ALLOW_PRIVATE_ADDRESSES"
    assert FETCH_MAX_BYTES_ENV == "RAG_CORE_FETCH_MAX_BYTES"
    assert FETCH_TIMEOUT_SECONDS_ENV == "RAG_CORE_FETCH_TIMEOUT_SECONDS"
    assert FETCH_MAX_REDIRECTS_ENV == "RAG_CORE_FETCH_MAX_REDIRECTS"

    owner = sources["src/rag_core/fetch_security.py"]
    assert owner.count('FETCH_ALLOW_HTTP_ENV = "RAG_CORE_FETCH_ALLOW_HTTP"') == 1
    assert (
        owner.count(
            'FETCH_ALLOW_PRIVATE_ADDRESSES_ENV = "RAG_CORE_FETCH_ALLOW_PRIVATE_ADDRESSES"'
        )
        == 1
    )
    assert owner.count('FETCH_MAX_BYTES_ENV = "RAG_CORE_FETCH_MAX_BYTES"') == 1
    assert (
        owner.count('FETCH_TIMEOUT_SECONDS_ENV = "RAG_CORE_FETCH_TIMEOUT_SECONDS"') == 1
    )
    assert owner.count('FETCH_MAX_REDIRECTS_ENV = "RAG_CORE_FETCH_MAX_REDIRECTS"') == 1

    consumer = sources["src/rag_core/cli_remote_fetch.py"]
    for symbol in (
        "FETCH_ALLOW_HTTP_ENV",
        "FETCH_ALLOW_PRIVATE_ADDRESSES_ENV",
        "FETCH_MAX_BYTES_ENV",
        "FETCH_TIMEOUT_SECONDS_ENV",
        "FETCH_MAX_REDIRECTS_ENV",
    ):
        assert symbol in consumer
    for duplicate in (
        '"RAG_CORE_FETCH_ALLOW_HTTP"',
        '"RAG_CORE_FETCH_ALLOW_PRIVATE_ADDRESSES"',
        '"RAG_CORE_FETCH_MAX_BYTES"',
        '"RAG_CORE_FETCH_TIMEOUT_SECONDS"',
        '"RAG_CORE_FETCH_MAX_REDIRECTS"',
    ):
        assert duplicate not in consumer
