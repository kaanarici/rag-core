"""Cohere rerank provider replay from recorded fixtures."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_core.search.providers.cohere import CohereReranker
from tests.support.provider_fixtures import load_provider_fixture, record_mode, rerank_response_from_fixture

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
