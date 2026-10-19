from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag_core.search.providers.embedding import (
    OpenAIEmbeddingProvider,
)


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


def _patch_async_openai(
    monkeypatch: pytest.MonkeyPatch, api: _FakeEmbeddingsAPI
) -> None:
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
