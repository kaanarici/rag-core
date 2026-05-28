from __future__ import annotations

import importlib.machinery
from collections.abc import Callable
from typing import Any, cast

import pytest

import rag_core.search.providers.model_provider_diagnostics as diagnostics_module
import rag_core.search.providers.provider_category_diagnostics as category_diagnostics_module
import rag_core.search.providers.provider_category_helpers as category_helpers_module
from rag_core.cli_doctor_output import emit_doctor
from rag_core.config import EmbeddingConfig, IngestConfig, RerankerConfig
from rag_core.core_models import RAGCoreConfig
from rag_core.documents.contextualizer_provider_names import (
    CONTEXTUALIZER_DISABLED_ALIAS,
    NOOP_CONTEXTUALIZER_ID,
)
from rag_core.events.sinks import (
    DEFAULT_EVENT_SINK_PROVIDER,
    LOGGING_EVENT_SINK_PROVIDER,
    MULTI_EVENT_SINK_PROVIDER,
    OPENTELEMETRY_EVENT_SINK_PROVIDER,
)
from rag_core.provider_api_keys import COHERE_API_KEY_ENVS
from rag_core.search.providers.diagnostic_support import (
    PROVIDER_DIAGNOSTIC_READINESS_SCOPES,
    PROVIDER_DIAGNOSTIC_SUPPORT_LEVELS,
    READINESS_PACKAGE_AND_ENV,
    SUPPORT_DEFAULT,
    SUPPORT_DEFAULT_NOOP,
    SUPPORT_FIRST_PARTY_OPTIONAL,
    SUPPORT_FIRST_PARTY_UTILITY,
    SUPPORT_INJECTED,
)
from rag_core.search.providers.cache_provider_names import (
    CACHE_PROVIDER_ORDER,
    IN_MEMORY_CACHE_PROVIDER,
    SQLITE_CACHE_PROVIDER,
)
from rag_core.search.providers.model_provider_diagnostics import (
    describe_model_provider_diagnostics,
)
from rag_core.search.providers.sparse import SPLADE_LOAD_UNKNOWN_UNTIL_RUN


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _support_levels(value: object) -> set[str]:
    levels: set[str] = set()
    if isinstance(value, dict):
        support_level = value.get("support_level")
        if isinstance(support_level, str):
            levels.add(support_level)
        for child in value.values():
            levels.update(_support_levels(child))
    elif isinstance(value, list | tuple):
        for child in value:
            levels.update(_support_levels(child))
    return levels


def _readiness_scopes(value: object) -> set[str]:
    scopes: set[str] = set()
    if isinstance(value, dict):
        readiness_scope = value.get("readiness_scope")
        if isinstance(readiness_scope, str):
            scopes.add(readiness_scope)
        for child in value.values():
            scopes.update(_readiness_scopes(child))
    elif isinstance(value, list | tuple):
        for child in value:
            scopes.update(_readiness_scopes(child))
    return scopes


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
        category_helpers_module.importlib.util,
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
            embedding_cache_provider=SQLITE_CACHE_PROVIDER,
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
    assert voyage["support_level"] == SUPPORT_FIRST_PARTY_OPTIONAL
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
    assert fastembed["support_level"] == SUPPORT_DEFAULT
    assert fastembed["configured"] is True
    assert fastembed["package_available"] is True
    assert fastembed["readiness_scope"] == READINESS_PACKAGE_AND_ENV
    sparse_channels = _mapping(fastembed["channels"])
    splade = _mapping(sparse_channels["splade"])
    assert splade["live_ready"] is None
    assert splade["load_status"] == SPLADE_LOAD_UNKNOWN_UNTIL_RUN

    ocr = _mapping(payload["ocr"])
    mistral = _mapping(_mapping(ocr["providers"])["mistral"])
    gemini = _mapping(_mapping(ocr["providers"])["gemini"])
    assert {"gemini", "mistral"}.issubset(set(ocr["registered"]))
    assert mistral["api_key_configured"] is True
    assert mistral["supports_page_selection"] is True
    assert gemini["supports_page_selection"] is False

    contextualizer = _mapping(payload["contextualizer"])
    anthropic = _mapping(_mapping(contextualizer["providers"])["anthropic"])
    assert contextualizer["configured"] == NOOP_CONTEXTUALIZER_ID
    assert anthropic["package_available"] is True
    assert anthropic["api_key_configured"] is True

    embedding_cache = _mapping(payload["embedding_cache"])
    sqlite_cache = _mapping(_mapping(embedding_cache["providers"])[SQLITE_CACHE_PROVIDER])
    assert embedding_cache["configured"] == SQLITE_CACHE_PROVIDER
    assert sqlite_cache["configured"] is True

    chunk_context_cache = _mapping(payload["chunk_context_cache"])
    assert set(CACHE_PROVIDER_ORDER).issubset(set(chunk_context_cache["registered"]))

    search_sidecar = _mapping(payload["search_sidecar"])
    portable_lexical = _mapping(_mapping(search_sidecar["providers"])["portable_lexical"])
    assert search_sidecar["configured"] == "portable_lexical"
    assert portable_lexical["configured"] is True

    event_sink = _mapping(payload["event_sink"])
    otel = _mapping(_mapping(event_sink["providers"])[OPENTELEMETRY_EVENT_SINK_PROVIDER])
    multi = _mapping(_mapping(event_sink["providers"])[MULTI_EVENT_SINK_PROVIDER])
    assert event_sink["configured"] == DEFAULT_EVENT_SINK_PROVIDER
    assert multi["support_level"] == SUPPORT_FIRST_PARTY_UTILITY
    assert otel["support_level"] == SUPPORT_FIRST_PARTY_OPTIONAL
    assert "secret" not in repr(payload)


def test_provider_diagnostics_use_shared_support_level_vocabulary() -> None:
    payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(),
        embedding_dimensions=1536,
        sparse_provider_name="custom_sparse",
        ocr_provider_name="custom_ocr",
        contextualizer_name="custom_contextualizer",
        embedding_cache_name="custom_embedding_cache",
        chunk_context_cache_name="custom_chunk_context_cache",
        search_sidecar_name="custom_sidecar",
        event_sink_name="custom_event_sink",
    )

    levels = _support_levels(payload)
    expected = set(PROVIDER_DIAGNOSTIC_SUPPORT_LEVELS)
    assert levels <= expected
    assert expected <= levels


def test_provider_diagnostics_use_shared_readiness_scope_vocabulary() -> None:
    payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(),
        embedding_dimensions=1536,
    )

    scopes = _readiness_scopes(payload)
    expected = set(PROVIDER_DIAGNOSTIC_READINESS_SCOPES)
    assert scopes <= expected
    assert READINESS_PACKAGE_AND_ENV in scopes


def test_contextualizer_diagnostics_use_noop_as_disabled_provider() -> None:
    payload = category_diagnostics_module.describe_contextualizer_diagnostics()

    assert payload["configured"] == NOOP_CONTEXTUALIZER_ID
    assert _mapping(payload["providers"])[NOOP_CONTEXTUALIZER_ID]["configured"] is True


def test_contextualizer_diagnostics_accept_none_as_disabled_alias() -> None:
    payload = category_diagnostics_module.describe_contextualizer_diagnostics(
        runtime_provider=CONTEXTUALIZER_DISABLED_ALIAS,
    )

    assert payload["configured"] == NOOP_CONTEXTUALIZER_ID
    assert CONTEXTUALIZER_DISABLED_ALIAS not in _mapping(payload["providers"])
    assert _mapping(payload["providers"])[NOOP_CONTEXTUALIZER_ID]["configured"] is True


def test_provider_category_diagnostics_treat_class_names_as_injected_names() -> None:
    payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(ingest=IngestConfig(enable_lexical_search=False)),
        embedding_dimensions=1536,
        sparse_provider_name="FastEmbedSparseEmbedder",
        contextualizer_name="NoOpContextualizer",
        embedding_cache_name="InMemoryCache",
        chunk_context_cache_name="SqliteChunkContextCache",
        search_sidecar_name="PortableLexicalSidecar",
        event_sink_name="LoggingSink",
    )

    expected = {
        "sparse": "fastembedsparseembedder",
        "contextualizer": "noopcontextualizer",
        "embedding_cache": "inmemorycache",
        "chunk_context_cache": "sqlitechunkcontextcache",
        "search_sidecar": "portablelexicalsidecar",
        "event_sink": "loggingsink",
    }
    for category, configured in expected.items():
        diagnostics = _mapping(payload[category])
        providers = _mapping(diagnostics["providers"])
        injected = _mapping(providers[configured])
        assert diagnostics["configured"] == configured
        assert injected["support_level"] == SUPPORT_INJECTED
        assert injected["configured"] is True


def test_provider_category_diagnostics_accept_canonical_builtin_provider_names() -> None:
    payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(ingest=IngestConfig(enable_lexical_search=False)),
        embedding_dimensions=1536,
        sparse_provider_name="fastembed",
        contextualizer_name=NOOP_CONTEXTUALIZER_ID,
        embedding_cache_name=IN_MEMORY_CACHE_PROVIDER,
        chunk_context_cache_name=SQLITE_CACHE_PROVIDER,
        search_sidecar_name="portable_lexical",
        event_sink_name=LOGGING_EVENT_SINK_PROVIDER,
    )

    assert _mapping(payload["sparse"])["configured"] == "fastembed"
    assert _mapping(payload["contextualizer"])["configured"] == NOOP_CONTEXTUALIZER_ID
    assert _mapping(payload["embedding_cache"])["configured"] == IN_MEMORY_CACHE_PROVIDER
    assert _mapping(payload["chunk_context_cache"])["configured"] == SQLITE_CACHE_PROVIDER
    assert _mapping(payload["search_sidecar"])["configured"] == "portable_lexical"
    assert _mapping(payload["event_sink"])["configured"] == LOGGING_EVENT_SINK_PROVIDER


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
    assert cohere["api_key_env"] == list(COHERE_API_KEY_ENVS)
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
        category_helpers_module.importlib.util,
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
        f"* embedding/voyage: support={SUPPORT_FIRST_PARTY_OPTIONAL} "
        "package=no api_key=yes"
        in output
    )
    assert (
        f"* reranker/cohere: support={SUPPORT_FIRST_PARTY_OPTIONAL} "
        "package=yes api_key=yes effective=cohere reason=none"
    ) in output
    assert f"* sparse/fastembed: support={SUPPORT_DEFAULT} package=no api_key=n/a" in output
    assert (
        f"* contextualizer/noop: support={SUPPORT_DEFAULT_NOOP} "
        "package=yes api_key=n/a"
    ) in output
    assert f"event_sink/multi: support={SUPPORT_FIRST_PARTY_UTILITY}" in output
    assert f"opentelemetry: support={SUPPORT_FIRST_PARTY_OPTIONAL}" in output
    assert "secret" not in output


def test_provider_category_diagnostics_handles_missing_dotted_package_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        category_helpers_module.importlib.util,
        "find_spec",
        _raise_missing_parent_for("opentelemetry.trace"),
    )

    payload = describe_model_provider_diagnostics(
        config=RAGCoreConfig(),
        embedding_dimensions=1536,
    )

    event_sink = _mapping(payload["event_sink"])
    opentelemetry = _mapping(
        _mapping(event_sink["providers"])[OPENTELEMETRY_EVENT_SINK_PROVIDER]
    )
    assert opentelemetry["package_available"] is False
