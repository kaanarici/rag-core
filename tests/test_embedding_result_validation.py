from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag_core.search.providers.embedding import OpenAIEmbeddingProvider
from rag_core.search.providers.cached_embedding import CachedEmbeddingProvider
from rag_core.search.providers.embedding_cache import (
    InMemoryCache,
)
from rag_core.search.providers.embedding_cache_models import (
    EmbedCacheKey,
    sha256_text,
)
from rag_core.search.providers.embedding_results import (
    safe_indexed_embedding_vectors,
    safe_ordered_embedding_vectors,
)

SECRET = "sk-test-secret"


class _DangerousVector:
    def __repr__(self) -> str:
        return f"repr leaked {SECRET}\nTraceback (most recent call last):"


DangerousTypeName = type(
    f"EmbeddingTypeNameLeak_{SECRET}_Traceback",
    (),
    {},
)


class _FakeEmbeddingsAPI:
    def __init__(self, data: list[SimpleNamespace]) -> None:
        self._data = data

    async def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(data=self._data)


def _patch_async_openai(
    monkeypatch: pytest.MonkeyPatch,
    api: _FakeEmbeddingsAPI,
) -> None:
    class _FakeClient:
        def __init__(self, **_: object) -> None:
            self.embeddings = api

    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_async_openai",
        lambda: _FakeClient,
    )


@pytest.mark.parametrize(
    ("rows", "expected_count", "message"),
    [
        ([(f"idx-{SECRET}", [0.1])], 1, "invalid embedding index"),
        ([(True, [0.1])], 1, "invalid embedding index"),
        ([(0, [0.1]), (0, [0.2])], 2, "duplicate embedding index"),
        ([(0, [0.1])], 2, "embedding count mismatch"),
        ([(0, _DangerousVector())], 1, "invalid embedding vector"),
        ([(0, [])], 1, "empty embedding vector"),
        ([(0, [False])], 1, "invalid embedding value"),
        ([(0, [DangerousTypeName()])], 1, "invalid embedding value"),
        ([(0, [float("nan")])], 1, "non-finite embedding value"),
    ],
)
def test_indexed_embedding_validation_errors_are_sanitized(
    rows: list[tuple[object, object]],
    expected_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        safe_indexed_embedding_vectors(
            rows=rows,
            expected_count=expected_count,
            expected_dimensions=None,
            provider_name="FakeEmbeddingProvider",
        )

    text = str(exc_info.value)
    assert message in text
    assert SECRET not in text
    assert "Traceback" not in text
    assert "repr leaked" not in text
    assert "EmbeddingTypeNameLeak" not in text
    assert "_DangerousVector" not in text


def test_ordered_embedding_validation_rejects_wrong_dimensions_safely() -> None:
    with pytest.raises(ValueError) as exc_info:
        safe_ordered_embedding_vectors(
            rows=[[0.1], [0.2, 0.3, 0.4]],
            expected_count=2,
            expected_dimensions=2,
            provider_name="FakeEmbeddingProvider",
        )

    text = str(exc_info.value)
    assert "embedding dimension mismatch" in text
    assert "expected 2 dimensions, got 1" in text


def test_openai_embedding_provider_validates_indexed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeEmbeddingsAPI(
        [
            SimpleNamespace(index=1, embedding=[0.2, 0.3]),
            SimpleNamespace(index=0, embedding=[0.1, 0.4]),
        ]
    )
    _patch_async_openai(monkeypatch, api)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dimensions=2)

    assert asyncio.run(provider.embed_texts(["alpha", "beta"])) == [
        [0.1, 0.4],
        [0.2, 0.3],
    ]


def test_openai_embedding_provider_errors_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeEmbeddingsAPI(
        [
            SimpleNamespace(index=f"idx-{SECRET}", embedding=[0.2]),
        ]
    )
    _patch_async_openai(monkeypatch, api)
    provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(provider.embed_texts(["alpha"]))

    text = str(exc_info.value)
    assert "OpenAIEmbeddingProvider returned invalid embedding index" in text
    assert SECRET not in text
    assert "Traceback" not in text


def test_openai_embedding_provider_rejects_wrong_dimensions_before_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeEmbeddingsAPI(
        [
            SimpleNamespace(index=0, embedding=[0.2]),
        ]
    )
    _patch_async_openai(monkeypatch, api)
    cache = InMemoryCache()
    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dimensions=2)
    cached = CachedEmbeddingProvider(provider, cache)

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(cached.embed_texts(["alpha"]))

    text = str(exc_info.value)
    assert "OpenAIEmbeddingProvider returned embedding dimension mismatch" in text
    assert asyncio.run(
        cache.get(
            EmbedCacheKey(
                provider="openai",
                provider_config_fingerprint="",
                model="text-embedding-3-small",
                dimensions=2,
                input_type="document",
                normalization="text_sha256_utf8",
                processing_fingerprint="",
                content_sha256=sha256_text("alpha"),
            )
        )
    ) is None
