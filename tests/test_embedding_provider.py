from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag_core.search.providers.cohere import CohereEmbeddingProvider
from rag_core.search.providers.embedding import (
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from rag_core.search.providers.cached_embedding import CachedEmbeddingProvider
from rag_core.search.providers.embedding_cache import InMemoryCache


class _FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        input_rows = kwargs["input"]
        assert isinstance(input_rows, list)
        raw_dimensions = kwargs.get("dimensions")
        dimensions = raw_dimensions if isinstance(raw_dimensions, int) else 1536
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index)] * dimensions)
                for index, _ in enumerate(input_rows)
            ]
        )


def _patch_async_openai(monkeypatch: pytest.MonkeyPatch, api: _FakeEmbeddingsAPI) -> None:
    class _FakeClient:
        def __init__(self, **_: object) -> None:
            self.embeddings = api

    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_async_openai",
        lambda: _FakeClient,
    )


def test_openai_embedding_provider_omits_dimensions_for_ada_002(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeEmbeddingsAPI()
    _patch_async_openai(monkeypatch, api)

    provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")
    rows = asyncio.run(provider.embed_texts(["alpha", "beta"]))

    assert len(rows) == 2
    assert [row[0] for row in rows] == [0.0, 1.0]
    assert all(len(row) == 1536 for row in rows)
    assert api.calls == [
        {
            "model": "text-embedding-ada-002",
            "input": ["alpha", "beta"],
        }
    ]


def test_openai_embedding_provider_keeps_dimensions_for_text_embedding_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeEmbeddingsAPI()
    _patch_async_openai(monkeypatch, api)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dimensions=512)
    asyncio.run(provider.embed_texts(["alpha"]))

    assert api.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["alpha"],
            "dimensions": 512,
        }
    ]


def test_openai_embedding_provider_keeps_dimensions_for_unknown_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeEmbeddingsAPI()
    _patch_async_openai(monkeypatch, api)

    provider = OpenAIEmbeddingProvider(model="custom-embedding-model", dimensions=1536)
    asyncio.run(provider.embed_texts(["alpha"]))

    assert api.calls == [
        {
            "model": "custom-embedding-model",
            "input": ["alpha"],
            "dimensions": 1536,
        }
    ]


def test_openai_embedding_provider_cache_identity_tracks_base_url_not_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_async_openai(monkeypatch, _FakeEmbeddingsAPI())

    default_provider = OpenAIEmbeddingProvider(api_key="secret-a")
    endpoint_a = OpenAIEmbeddingProvider(
        api_key="secret-a",
        base_url="https://embeddings-a.example.test/v1",
    )
    endpoint_a_other_key = OpenAIEmbeddingProvider(
        api_key="secret-b",
        base_url="https://embeddings-a.example.test/v1",
    )
    endpoint_a_spaced = OpenAIEmbeddingProvider(
        api_key="secret-a",
        base_url=" https://embeddings-a.example.test/v1 ",
    )
    blank_endpoint = OpenAIEmbeddingProvider(
        api_key="secret-a",
        base_url="   ",
    )
    endpoint_b = OpenAIEmbeddingProvider(
        api_key="secret-a",
        base_url="https://embeddings-b.example.test/v1",
    )

    assert default_provider.cache_identity == ""
    assert blank_endpoint.cache_identity == ""
    assert endpoint_a.cache_identity
    assert endpoint_a.cache_identity == endpoint_a_other_key.cache_identity
    assert endpoint_a.cache_identity == endpoint_a_spaced.cache_identity
    assert endpoint_a.cache_identity != endpoint_b.cache_identity
    assert "secret" not in endpoint_a.cache_identity
    assert "embeddings-a" not in endpoint_a.cache_identity


def test_openai_embedding_provider_trimmed_base_url_shares_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeEmbeddingsAPI()
    _patch_async_openai(monkeypatch, api)
    cache = InMemoryCache()
    trimmed = CachedEmbeddingProvider(
        OpenAIEmbeddingProvider(base_url="https://embeddings-a.example.test/v1"),
        cache,
    )
    untrimmed = CachedEmbeddingProvider(
        OpenAIEmbeddingProvider(base_url=" https://embeddings-a.example.test/v1 "),
        cache,
    )

    first = asyncio.run(trimmed.embed_texts(["alpha"]))
    second = asyncio.run(untrimmed.embed_texts(["alpha"]))

    assert second == first
    assert len(api.calls) == 1


def test_create_embedding_provider_voyage_does_not_import_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeVoyageProvider:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def _raise_if_openai_imported() -> object:
        raise AssertionError("openai import should not happen for voyage provider")

    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_async_openai",
        _raise_if_openai_imported,
    )
    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_voyage_embedding_provider",
        lambda: _FakeVoyageProvider,
    )

    provider = create_embedding_provider(provider="voyage", model="voyage-4-lite", dimensions=256)

    assert isinstance(provider, _FakeVoyageProvider)
    assert provider.kwargs == {
        "model": "voyage-4-lite",
        "dimensions": 256,
        "api_key": None,
    }


def test_create_embedding_provider_cohere_uses_default_model_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCohereProvider:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def _raise_if_openai_imported() -> object:
        raise AssertionError("openai import should not happen for cohere provider")

    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_async_openai",
        _raise_if_openai_imported,
    )
    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_cohere_embedding_provider",
        lambda: _FakeCohereProvider,
    )

    provider = create_embedding_provider(provider="cohere")

    assert isinstance(provider, _FakeCohereProvider)
    assert provider.kwargs == {
        "model": "embed-v4.0",
        "dimensions": 1536,
        "api_key": None,
    }


def test_cohere_embedding_provider_batches_by_sdk_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCohereClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def embed(self, **kwargs: object) -> SimpleNamespace:
            self.calls.append(dict(kwargs))
            texts = kwargs["texts"]
            assert isinstance(texts, list)
            dimensions = kwargs["output_dimension"]
            assert isinstance(dimensions, int)
            return SimpleNamespace(
                embeddings=SimpleNamespace(
                    float_=[
                        [float(index)] * dimensions
                        for index, _text in enumerate(texts)
                    ]
                )
            )

    client = _FakeCohereClient()
    monkeypatch.setattr(
        "rag_core.search.providers.cohere._import_cohere",
        lambda: SimpleNamespace(AsyncClientV2=lambda **_: client),
    )
    provider = CohereEmbeddingProvider(dimensions=2)

    vectors = asyncio.run(provider.embed_texts([f"doc {index}" for index in range(97)]))
    empty = asyncio.run(provider.embed_texts([]))

    batch_sizes: list[int] = []
    for call in client.calls:
        texts = call["texts"]
        assert isinstance(texts, list)
        batch_sizes.append(len(texts))

    assert len(vectors) == 97
    assert empty == []
    assert batch_sizes == [96, 1]
    assert all(call["input_type"] == "search_document" for call in client.calls)
    assert all(call["embedding_types"] == ["float"] for call in client.calls)


def test_openai_embedding_provider_missing_sdk_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_async_openai",
        lambda: (_ for _ in ()).throw(ImportError("openai package is required")),
    )

    with pytest.raises(ImportError, match="openai package is required"):
        OpenAIEmbeddingProvider()


def test_openai_embedding_provider_normalizes_blank_api_key_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))
            self.embeddings = _FakeEmbeddingsAPI()

    monkeypatch.setattr(
        "rag_core.search.providers.embedding._import_async_openai",
        lambda: _FakeClient,
    )

    OpenAIEmbeddingProvider(api_key="   ", base_url="  ")
    OpenAIEmbeddingProvider(api_key=" secret ", base_url=" https://example.test/v1 ")

    assert calls == [{}, {"api_key": "secret", "base_url": "https://example.test/v1"}]
