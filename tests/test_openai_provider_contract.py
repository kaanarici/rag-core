"""OpenAI embedding provider replay from recorded fixtures."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_core.search.providers.embedding import OpenAIEmbeddingProvider
from tests.support.provider_fixtures import (
    embeddings_response_from_fixture,
    load_provider_fixture,
    record_mode,
)

pytestmark = [pytest.mark.provider_contract]


@pytest.fixture(autouse=True)
def _no_network_replay() -> None:
    assert record_mode() in {"none", "once", "new_episodes"}


def test_openai_embed_texts_replays_batch_shape_and_order(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = load_provider_fixture("openai", "embeddings_create.json")
    response = embeddings_response_from_fixture(payload)

    embeddings_api = MagicMock()
    embeddings_api.create = AsyncMock(return_value=response)
    client = SimpleNamespace(embeddings=embeddings_api)

    monkeypatch.setattr(
        "rag_core.search.providers.embedding.build_openai_client",
        lambda *_args, **_kwargs: client,
    )

    provider = OpenAIEmbeddingProvider(
        model="text-embedding-3-small",
        dimensions=4,
        api_key="test-key",
    )
    vectors = asyncio.run(provider.embed_texts(["alpha", "beta"]))

    assert len(vectors) == 2
    assert vectors[0] == payload["data"][0]["embedding"]
    assert vectors[1] == payload["data"][1]["embedding"]
    request = embeddings_api.create.await_args.kwargs
    assert request["model"] == "text-embedding-3-small"
    assert request["dimensions"] == 4
