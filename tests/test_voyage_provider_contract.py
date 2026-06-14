"""Voyage embedding provider replay from recorded fixtures."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_core.search.providers.voyage import VoyageEmbeddingProvider
from tests.support.provider_fixtures import (
    load_provider_fixture,
    record_mode,
    voyage_embed_response_from_fixture,
)

pytestmark = [pytest.mark.provider_contract]


@pytest.fixture(autouse=True)
def _no_network_replay() -> None:
    assert record_mode() in {"none", "once", "new_episodes"}


def test_voyage_embed_texts_replays_document_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = load_provider_fixture("voyage", "embed.json")
    response = voyage_embed_response_from_fixture(payload)

    client = MagicMock()
    client.embed = MagicMock(return_value=response)
    voyage_module = SimpleNamespace(Client=lambda **_: client)
    monkeypatch.setattr(
        "rag_core.search.providers.voyage._import_voyageai",
        lambda: voyage_module,
    )

    provider = VoyageEmbeddingProvider(model="voyage-4", dimensions=4, api_key="test-key")
    vectors = asyncio.run(provider.embed_texts(["doc one", "doc two"]))

    assert vectors == payload["embeddings"]
    assert client.embed.call_args.kwargs["input_type"] == "document"


def test_voyage_embed_query_replays_query_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = load_provider_fixture("voyage", "embed.json")
    response = voyage_embed_response_from_fixture({"embeddings": [payload["embeddings"][0]]})

    client = MagicMock()
    client.embed = MagicMock(return_value=response)
    voyage_module = SimpleNamespace(Client=lambda **_: client)
    monkeypatch.setattr(
        "rag_core.search.providers.voyage._import_voyageai",
        lambda: voyage_module,
    )

    provider = VoyageEmbeddingProvider(model="voyage-4", dimensions=4, api_key="test-key")
    vector = asyncio.run(provider.embed_query("billing webhook"))

    assert vector == payload["embeddings"][0]
    assert client.embed.call_args.kwargs["input_type"] == "query"
