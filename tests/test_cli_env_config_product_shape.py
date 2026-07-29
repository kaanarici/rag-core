from __future__ import annotations

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

from tests.support.source_graph import modules_assigning_value


def test_cli_config_env_names_have_config_owners() -> None:
    assert EMBEDDING_PROVIDER_ENV == "RAG_CORE_EMBEDDING_PROVIDER"
    assert EMBEDDING_MODEL_ENV == "RAG_CORE_EMBEDDING_MODEL"
    assert EMBEDDING_DIMENSIONS_ENV == "RAG_CORE_EMBEDDING_DIMENSIONS"
    assert EMBEDDING_BATCH_SIZE_ENV == "RAG_CORE_EMBEDDING_BATCH_SIZE"
    assert RERANKER_PROVIDER_ENV == "RAG_CORE_RERANKER_PROVIDER"
    assert RERANKER_MODEL_ENV == "RAG_CORE_RERANKER_MODEL"
    assert PROCESSING_VERSION_ENV == "RAG_CORE_PROCESSING_VERSION"
    assert CLI_MANIFEST_DIR_ENV == "RAG_CORE_MANIFEST_DIR"

    # Each env-var literal is assigned in exactly one config module; consumers
    # must import the named constant rather than re-typing the string. Asserting
    # on the literal's single owning module (across the whole package) survives
    # file moves and renames where the old per-file ``count()`` / ``not in
    # consumers`` scrape would have frozen the layout.
    owners = {
        EMBEDDING_PROVIDER_ENV: ("rag_core.config.embedding_config", "EMBEDDING_PROVIDER_ENV"),
        EMBEDDING_MODEL_ENV: ("rag_core.config.embedding_config", "EMBEDDING_MODEL_ENV"),
        EMBEDDING_DIMENSIONS_ENV: (
            "rag_core.config.embedding_config",
            "EMBEDDING_DIMENSIONS_ENV",
        ),
        EMBEDDING_BATCH_SIZE_ENV: (
            "rag_core.config.embedding_config",
            "EMBEDDING_BATCH_SIZE_ENV",
        ),
        RERANKER_PROVIDER_ENV: ("rag_core.config.reranker_config", "RERANKER_PROVIDER_ENV"),
        RERANKER_MODEL_ENV: ("rag_core.config.reranker_config", "RERANKER_MODEL_ENV"),
        PROCESSING_VERSION_ENV: ("rag_core.config.ingest_config", "PROCESSING_VERSION_ENV"),
        CLI_MANIFEST_DIR_ENV: ("rag_core.config.ingest_config", "CLI_MANIFEST_DIR_ENV"),
    }
    for literal, (owner_module, owner_name) in owners.items():
        assert modules_assigning_value("src/rag_core", value=literal) == {
            owner_module: [owner_name]
        }


def test_fetch_env_names_have_fetch_security_owner() -> None:
    assert FETCH_ALLOW_HTTP_ENV == "RAG_CORE_FETCH_ALLOW_HTTP"
    assert FETCH_ALLOW_PRIVATE_ADDRESSES_ENV == "RAG_CORE_FETCH_ALLOW_PRIVATE_ADDRESSES"
    assert FETCH_MAX_BYTES_ENV == "RAG_CORE_FETCH_MAX_BYTES"
    assert FETCH_TIMEOUT_SECONDS_ENV == "RAG_CORE_FETCH_TIMEOUT_SECONDS"
    assert FETCH_MAX_REDIRECTS_ENV == "RAG_CORE_FETCH_MAX_REDIRECTS"

    owners = {
        FETCH_ALLOW_HTTP_ENV: "FETCH_ALLOW_HTTP_ENV",
        FETCH_ALLOW_PRIVATE_ADDRESSES_ENV: "FETCH_ALLOW_PRIVATE_ADDRESSES_ENV",
        FETCH_MAX_BYTES_ENV: "FETCH_MAX_BYTES_ENV",
        FETCH_TIMEOUT_SECONDS_ENV: "FETCH_TIMEOUT_SECONDS_ENV",
        FETCH_MAX_REDIRECTS_ENV: "FETCH_MAX_REDIRECTS_ENV",
    }
    for literal, owner_name in owners.items():
        assert modules_assigning_value("src/rag_core", value=literal) == {
            "rag_core.fetch_security": [owner_name]
        }
