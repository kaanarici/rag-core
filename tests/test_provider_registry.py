from __future__ import annotations

from typing import Any

import pytest

from rag_core.search.providers.embedding import create_embedding_provider
import rag_core.search.providers.sparse as sparse_module
from rag_core.search.providers.registry import (
    CHUNK_CONTEXT_CACHES,
    EMBEDDING_CACHES,
    EMBEDDING_PROVIDERS,
    OCR_PROVIDERS,
    RERANKER_PROVIDERS,
    SEARCH_SIDECARS,
    SPARSE_EMBEDDERS,
    ProviderRegistry,
)


class _FakeProvider:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _registry() -> ProviderRegistry[_FakeProvider]:
    return ProviderRegistry("test")


def test_registry_rejects_empty_name() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", lambda **kw: _FakeProvider(**kw))
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("   ", lambda **kw: _FakeProvider(**kw))


def test_registry_rejects_double_registration() -> None:
    registry = _registry()
    registry.register("foo", lambda **kw: _FakeProvider(**kw))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("foo", lambda **kw: _FakeProvider(**kw))
    # Case-insensitive collision is also rejected.
    with pytest.raises(ValueError, match="already registered"):
        registry.register("FOO", lambda **kw: _FakeProvider(**kw))


def test_registry_round_trips_register_and_create() -> None:
    registry = _registry()
    registry.register("foo", lambda **kw: _FakeProvider(**kw))
    instance = registry.create("foo", a=1, b="two")
    assert isinstance(instance, _FakeProvider)
    assert instance.kwargs == {"a": 1, "b": "two"}


def test_registry_lookup_is_case_insensitive() -> None:
    registry = _registry()
    registry.register("Foo", lambda **kw: _FakeProvider(**kw))
    assert isinstance(registry.create("foo"), _FakeProvider)
    assert isinstance(registry.create("FOO"), _FakeProvider)
    assert isinstance(registry.create("  Foo  "), _FakeProvider)
    assert "foo" in registry
    assert "FOO" in registry
    assert "missing" not in registry


def test_registry_create_unknown_raises_with_known_names() -> None:
    registry = _registry()
    registry.register("foo", lambda **kw: _FakeProvider(**kw))
    registry.register("bar", lambda **kw: _FakeProvider(**kw))
    with pytest.raises(ValueError, match="bar, foo"):
        registry.create("missing")


def test_registry_names_returns_sorted() -> None:
    registry = _registry()
    registry.register("zeta", lambda **kw: _FakeProvider(**kw))
    registry.register("alpha", lambda **kw: _FakeProvider(**kw))
    assert registry.names() == ("alpha", "zeta")


def test_registry_contains_rejects_non_string() -> None:
    registry = _registry()
    registry.register("foo", lambda **kw: _FakeProvider(**kw))
    assert 42 not in registry
    assert None not in registry


def test_registry_retries_builtin_import_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ProviderRegistry[_FakeProvider]("test").with_builtins("tests.fake.provider")
    attempts = 0

    def flaky_import(module_name: str) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError(f"boom: {module_name}")
        registry.register("retryable", lambda **kw: _FakeProvider(**kw))
        return object()

    monkeypatch.setattr("rag_core.search.providers.registry.import_module", flaky_import)

    with pytest.raises(RuntimeError, match="boom"):
        registry.create("retryable")

    created = registry.create("retryable", attempt=2)
    assert isinstance(created, _FakeProvider)
    assert created.kwargs == {"attempt": 2}
    assert attempts == 2


@pytest.mark.parametrize(
    "registry,expected",
    [
        (EMBEDDING_PROVIDERS, ("openai", "voyage", "zeroentropy")),
        (RERANKER_PROVIDERS, ("none", "cohere", "voyage", "zeroentropy")),
        (SPARSE_EMBEDDERS, ("fastembed",)),
        (OCR_PROVIDERS, ("mistral", "gemini")),
        (SEARCH_SIDECARS, ("portable_lexical",)),
        (EMBEDDING_CACHES, ("none", "in_memory", "sqlite")),
        (CHUNK_CONTEXT_CACHES, ("none", "in_memory", "sqlite")),
    ],
    ids=[
        "embedding",
        "reranker",
        "sparse",
        "ocr",
        "search_sidecar",
        "embedding_cache",
        "chunk_context_cache",
    ],
)
def test_builtin_providers_are_registered(
    registry: ProviderRegistry[Any], expected: tuple[str, ...]
) -> None:
    for name in expected:
        assert name in registry, f"expected built-in provider {name!r} in registry"


def test_sparse_provider_registry_introspection_does_not_import_fastembed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import() -> Any:
        raise AssertionError("runtime import should only happen on create")

    monkeypatch.setattr(sparse_module, "_import_sparse_text_embedding", fail_import)

    assert "fastembed" in SPARSE_EMBEDDERS.names()
    with pytest.raises(AssertionError, match="runtime import"):
        SPARSE_EMBEDDERS.create("fastembed")


def test_create_embedding_provider_openai_uses_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeOpenAIClient:
        def __init__(self, **_: object) -> None:
            self.embeddings = object()

    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_async_openai",
        lambda: _FakeOpenAIClient,
    )

    provider = create_embedding_provider(provider="openai", model="text-embedding-3-small")

    assert provider.model_name == "text-embedding-3-small"
    assert provider.dimensions == 1536


def test_user_registered_embedding_provider_works_end_to_end() -> None:
    """Adding a custom adapter externally is a register() call, not a fork."""

    class _CustomEmbeddingProvider:
        def __init__(self, *, model: str = "custom-1", dimensions: int = 8, **_: Any) -> None:
            self._model = model
            self._dimensions = dimensions

        @property
        def model_name(self) -> str:
            return self._model

        @property
        def dimensions(self) -> int:
            return self._dimensions

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(t))] * self._dimensions for t in texts]

        async def embed_query(self, query: str) -> list[float]:
            return [float(len(query))] * self._dimensions

    name = "custom-test-provider"
    EMBEDDING_PROVIDERS.register(
        name, lambda **kwargs: _CustomEmbeddingProvider(**kwargs)
    )
    try:
        provider = create_embedding_provider(provider=name, model="x", dimensions=4)
        assert isinstance(provider, _CustomEmbeddingProvider)
        assert provider.model_name == "x"
        assert provider.dimensions == 4
    finally:
        EMBEDDING_PROVIDERS._factories.pop(name, None)
