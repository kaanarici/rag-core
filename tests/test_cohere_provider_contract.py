"""Cohere rerank provider replay from recorded fixtures."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_core.search.providers.cohere import CohereEmbeddingProvider, CohereReranker
from tests.support.provider_fixtures import (
    cohere_embed_response_from_fixture,
    load_provider_fixture,
    record_mode,
    rerank_response_from_fixture,
)

pytestmark = [pytest.mark.provider_contract]


@pytest.fixture(autouse=True)
def _no_network_replay() -> None:
    assert record_mode() in {"none", "once", "new_episodes"}


def test_cohere_rerank_replays_order_and_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = load_provider_fixture("cohere", "rerank.json")
    response = rerank_response_from_fixture(payload)

    client = MagicMock()
    client.rerank = AsyncMock(return_value=response)
    monkeypatch.setattr("rag_core.search.providers.cohere._import_cohere", lambda: MagicMock(AsyncClientV2=lambda **_: client))

    reranker = CohereReranker(api_key="test-key")
    results = asyncio.run(
        reranker.rerank(
            query="invoice payment",
            documents=["ACH billing", "shipping tracking"],
            top_k=2,
        )
    )

    assert [result.index for result in results] == [1, 0]
    assert results[0].score == pytest.approx(0.91)
    call = client.rerank.await_args.kwargs
    assert call["top_n"] == 2


def test_cohere_embed_texts_replays_document_input_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = load_provider_fixture("cohere", "embed.json")
    response = cohere_embed_response_from_fixture(payload)

    client = MagicMock()
    client.embed = AsyncMock(return_value=response)
    monkeypatch.setattr(
        "rag_core.search.providers.cohere._import_cohere",
        lambda: SimpleNamespace(AsyncClientV2=lambda **_: client),
    )

    provider = CohereEmbeddingProvider(dimensions=4, api_key="test-key")
    vectors = asyncio.run(provider.embed_texts(["doc one", "doc two"]))

    assert vectors == payload["embeddings"]["float"]
    call = client.embed.await_args.kwargs
    assert call["input_type"] == "search_document"
    assert call["output_dimension"] == 4
    assert call["embedding_types"] == ["float"]


def test_cohere_embed_query_replays_query_input_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = load_provider_fixture("cohere", "embed.json")
    response = cohere_embed_response_from_fixture(
        {"embeddings": {"float": [payload["embeddings"]["float"][0]]}}
    )

    client = MagicMock()
    client.embed = AsyncMock(return_value=response)
    monkeypatch.setattr(
        "rag_core.search.providers.cohere._import_cohere",
        lambda: SimpleNamespace(AsyncClientV2=lambda **_: client),
    )

    provider = CohereEmbeddingProvider(dimensions=4, api_key="test-key")
    vector = asyncio.run(provider.embed_query("billing webhook"))

    assert vector == payload["embeddings"]["float"][0]
    assert client.embed.await_args.kwargs["input_type"] == "search_query"
