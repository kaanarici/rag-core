from __future__ import annotations

import importlib.machinery
from collections.abc import Callable
from typing import Any, cast

import pytest

import rag_core.search.providers.model_provider_diagnostics as diagnostics_module
import rag_core.search.providers.provider_category_diagnostics as category_diagnostics_module
from rag_core.cli_doctor_output import emit_doctor
from rag_core.config import EmbeddingConfig, IngestConfig, RerankerConfig
from rag_core.core_models import RAGCoreConfig
from rag_core.search.providers.model_provider_diagnostics import (
    describe_model_provider_diagnostics,
)


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _find_spec_for(
    *available: str,
) -> Callable[[str], importlib.machinery.ModuleSpec | None]:
    available_names = set(available)

    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        if name in available_names:
            return importlib.machinery.ModuleSpec(name, loader=None)
        return None

    return find_spec


def _raise_missing_parent_for(name_to_raise: str) -> Callable[[str], importlib.machinery.ModuleSpec | None]:
    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        if name == name_to_raise:
            raise ModuleNotFoundError(name)
        return None

    return find_spec


def test_model_provider_diagnostics_report_readiness_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("COHERE_API_KEY", "cohere-secret")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setattr(
        diagnostics_module.importlib.util,
        "find_spec",
        _find_spec_for("openai", "cohere", "fastembed", "anthropic"),
    )
    monkeypatch.setattr(
        category_diagnostics_module.importlib.util,
        "find_spec",
        _find_spec_for("openai", "cohere", "fastembed", "anthropic"),
    )
    config = RAGCoreConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-4-lite",
            dimensions=512,
            api_key="voyage-secret",
            batch_size=24,
        ),
        reranker=RerankerConfig(provider="cohere", model="rerank-v3.5"),
        ingest=IngestConfig(
            enable_lexical_search=True,
            embedding_cache_provider="sqlite",
        ),
    )

    payload = describe_model_provider_diagnostics(
        config=config,
        embedding_dimensions=512,
    )

    embedding = _mapping(payload["embedding"])
    embedding_providers = _mapping(embedding["providers"])
    voyage = _mapping(embedding_providers["voyage"])
    openai = _mapping(embedding_providers["openai"])
    assert embedding["configured"] == "voyage"
    assert voyage["configured"] is True
    assert voyage["support_level"] == "first_party_optional"
    assert voyage["package_available"] is False
    assert voyage["api_key_configured"] is True
    assert voyage["dimensions"] == 512
    assert voyage["allowed_dimensions"] == [256, 512, 1024, 2048]
    assert openai["api_key_configured"] is True

    reranker = _mapping(payload["reranker"])
    reranker_providers = _mapping(reranker["providers"])
    cohere = _mapping(reranker_providers["cohere"])
    assert reranker["configured"] == "cohere"
    assert reranker["effective"] == "cohere"
    assert reranker["fallback_reason"] is None
    assert cohere["configured"] is True
    assert cohere["package_available"] is True
    assert cohere["api_key_configured"] is True

    sparse = _mapping(payload["sparse"])
    fastembed = _mapping(_mapping(sparse["providers"])["fastembed"])
    assert sparse["registered"] == ["fastembed"]
    assert fastembed["support_level"] == "default"
    assert fastembed["configured"] is True
    assert fastembed["package_available"] is True
    assert fastembed["readiness_scope"] == "package_and_env"
    sparse_channels = _mapping(fastembed["channels"])
    splade = _mapping(sparse_channels["splade"])
    assert splade["live_ready"] is None
    assert splade["load_status"] == "unknown_until_sparse_embedding_runs"

    ocr = _mapping(payload["ocr"])
    mistral = _mapping(_mapping(ocr["providers"])["mistral"])
    gemini = _mapping(_mapping(ocr["providers"])["gemini"])
    assert {"gemini", "mistral"}.issubset(set(ocr["registered"]))
    assert mistral["api_key_configured"] is True
    assert mistral["supports_page_selection"] is True
    assert gemini["supports_page_selection"] is False

    contextualizer = _mapping(payload["contextualizer"])
    anthropic = _mapping(_mapping(contextualizer["providers"])["anthropic"])
    assert contextualizer["configured"] == "none"
    assert anthropic["package_available"] is True
    assert anthropic["api_key_configured"] is True

    embedding_cache = _mapping(payload["embedding_cache"])
    sqlite_cache = _mapping(_mapping(embedding_cache["providers"])["sqlite"])
    assert embedding_cache["configured"] == "sqlite"
    assert sqlite_cache["configured"] is True

    chunk_context_cache = _mapping(payload["chunk_context_cache"])
    assert {"in_memory", "none", "sqlite"}.issubset(
        set(chunk_context_cache["registered"])
    )

    search_sidecar = _mapping(payload["search_sidecar"])
    portable_lexical = _mapping(_mapping(search_sidecar["providers"])["portable_lexical"])
    assert search_sidecar["configured"] == "portable_lexical"
    assert portable_lexical["configured"] is True

    event_sink = _mapping(payload["event_sink"])
    otel = _mapping(_mapping(event_sink["providers"])["opentelemetry"])
    assert event_sink["configured"] == "none"
    assert otel["support_level"] == "first_party_optional"
    assert "secret" not in repr(payload)


def test_cohere_reranker_diagnostics_accept_standard_co_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CO_API_KEY", "cohere-secret")
    config = RAGCoreConfig(
        reranker=RerankerConfig(provider="cohere", model="rerank-v3.5"),
    )

    payload = describe_model_provider_diagnostics(
        config=config,
        embedding_dimensions=1536,
    )

    reranker = _mapping(payload["reranker"])
    cohere = _mapping(_mapping(reranker["providers"])["cohere"])
    assert reranker["effective"] == "cohere"
    assert reranker["fallback_reason"] is None
    assert cohere["api_key_env"] == ["COHERE_API_KEY", "CO_API_KEY"]
    assert cohere["api_key_configured"] is True
    assert "cohere-secret" not in repr(payload)


def test_embedding_diagnostics_normalize_blank_openai_base_url() -> None:
    blank_payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(embedding=EmbeddingConfig(base_url="   ")),
        embedding_dimensions=1536,
    )
    trimmed_payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(
            embedding=EmbeddingConfig(base_url=" https://example.test/v1 "),
        ),
        embedding_dimensions=1536,
    )

    blank_openai = _mapping(
        _mapping(_mapping(blank_payload["embedding"])["providers"])["openai"]
    )
    trimmed_openai = _mapping(
        _mapping(_mapping(trimmed_payload["embedding"])["providers"])["openai"]
    )
    assert blank_openai["base_url_configured"] is False
    assert trimmed_openai["base_url_configured"] is True
    assert "example.test" not in repr(trimmed_payload)


def test_doctor_human_output_summarizes_model_provider_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "cohere-secret")
    monkeypatch.setattr(
        diagnostics_module.importlib.util,
        "find_spec",
        _find_spec_for("openai", "cohere"),
    )
    monkeypatch.setattr(
        category_diagnostics_module.importlib.util,
        "find_spec",
        _find_spec_for("openai", "cohere"),
    )
    config = RAGCoreConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-4-lite",
            dimensions=512,
            api_key="voyage-secret",
            batch_size=24,
        ),
        reranker=RerankerConfig(provider="cohere", model="rerank-v3.5"),
    )
    provider_payload = describe_model_provider_diagnostics(
        config=config,
        embedding_dimensions=512,
    )

    emit_doctor(
        {
            "runtime": {},
            "collection_name": "docs",
            "processing_version": "v1",
            "source_processing_versions": {},
            "embedding": {
                "provider": "voyage",
                "model": "voyage-4-lite",
                "dimensions": 512,
                "batch_size": 24,
            },
            "reranker": {
                "requested": "cohere",
                "effective": "cohere",
                "fallback_reason": None,
            },
            "qdrant": {"url": None, "location": ":memory:"},
            "vector_store": {},
            "providers": provider_payload,
        },
        as_json=False,
        fix=False,
    )

    output = capsys.readouterr().out
    assert "Model Providers:" in output
    assert "Provider Categories:" in output
    assert (
        "* embedding/voyage: support=first_party_optional package=no api_key=yes"
        in output
    )
    assert (
        "* reranker/cohere: support=first_party_optional package=yes api_key=yes "
        "effective=cohere reason=none"
    ) in output
    assert "* sparse/fastembed: support=default package=no api_key=n/a" in output
    assert "* contextualizer/noop: support=default_noop package=yes api_key=n/a" in output
    assert "opentelemetry: support=first_party_optional" in output
    assert "secret" not in output


def test_provider_category_diagnostics_handles_missing_dotted_package_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        category_diagnostics_module.importlib.util,
        "find_spec",
        _raise_missing_parent_for("opentelemetry.trace"),
    )

    payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(),
        embedding_dimensions=1536,
    )

    event_sink = _mapping(payload["event_sink"])
    opentelemetry = _mapping(_mapping(event_sink["providers"])["opentelemetry"])
    assert opentelemetry["package_available"] is False
