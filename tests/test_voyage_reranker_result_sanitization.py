from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import pytest

from rag_core.search.providers.voyage import VoyageReranker
from rag_core.search.providers.cohere import CohereReranker
from rag_core.search.providers.rerank_results import rerank_provider_result_count

SECRET = "sk-live-super-secret"
LOGGER_NAME = "rag_core.search.providers.rerank_results"


class _FakeRow:
    def __init__(self, index: object, relevance_score: object) -> None:
        self.index = index
        self.relevance_score = relevance_score


class _FakeResponse:
    def __init__(self, results: Sequence[_FakeRow]) -> None:
        self.results = list(results)


class _DangerousValue:
    def __repr__(self) -> str:
        return f"repr leaked {SECRET}\\nTraceback (most recent call last):"


DangerousTypeName = type(
    f"TypeNameLeak_{SECRET}_Traceback",
    (),
    {},
)


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, list[str], str, int]] = []

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_k: int,
    ) -> _FakeResponse:
        self.calls.append((query, list(documents), model, top_k))
        return self._response


def test_voyage_reranker_preserves_valid_rows_and_sanitizes_invalid_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    documents = ["alpha", "beta", "gamma"]
    fake_response = _FakeResponse(
        results=[
            _FakeRow(2, "0.85"),
            _FakeRow(True, 0.5),
            _FakeRow(f"idx-{SECRET}", 0.9),
            _FakeRow(0, False),
            _FakeRow(1, _DangerousValue()),
            _FakeRow(0, DangerousTypeName()),
            _FakeRow(0, "not-a-number"),
            _FakeRow(0, "nan"),
            _FakeRow(1, 1),
        ]
    )
    fake_client = _FakeClient(fake_response)

    reranker = object.__new__(VoyageReranker)
    reranker._client = fake_client
    reranker._model = "rerank-2.5-lite"

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = asyncio.run(
            reranker.rerank(
                query="what is relevant",
                documents=documents,
                top_k=5,
            )
        )

    assert fake_client.calls == [
        (
                "what is relevant",
                documents,
                "rerank-2.5-lite",
                3,
            )
        ]

    assert [(row.index, row.score, row.text) for row in results] == [
        (1, 1.0, "beta"),
        (2, 0.85, "gamma"),
    ]

    assert "VoyageReranker returned invalid rerank index" in caplog.text
    assert "VoyageReranker returned invalid rerank score" in caplog.text
    assert "VoyageReranker returned non-finite rerank score" in caplog.text
    assert "reason=invalid_type" in caplog.text
    assert "reason=invalid_value" in caplog.text
    assert "value_type=str" in caplog.text
    assert "value_type=bool" in caplog.text
    assert "value_type=object" in caplog.text

    assert SECRET not in caplog.text
    assert "Traceback (most recent call last):" not in caplog.text
    assert "repr leaked" not in caplog.text
    assert "TypeNameLeak_" not in caplog.text
    assert "_DangerousValue" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_voyage_reranker_non_positive_top_k_returns_empty_without_provider_call() -> None:
    fake_client = _FakeClient(_FakeResponse([]))
    reranker = object.__new__(VoyageReranker)
    reranker._client = fake_client
    reranker._model = "rerank-2.5-lite"

    assert asyncio.run(reranker.rerank("query", ["alpha"], top_k=0)) == []
    assert asyncio.run(reranker.rerank("query", ["alpha"], top_k=-1)) == []
    assert fake_client.calls == []


def test_voyage_reranker_bounds_top_k_and_slices_validated_results() -> None:
    fake_client = _FakeClient(
        _FakeResponse([_FakeRow(0, 0.9), _FakeRow(1, 0.8), _FakeRow(2, 0.7)])
    )
    reranker = object.__new__(VoyageReranker)
    reranker._client = fake_client
    reranker._model = "rerank-2.5-lite"

    results = asyncio.run(reranker.rerank("query", ["alpha", "beta", "gamma"], top_k=2))

    assert fake_client.calls[0][3] == 2
    assert [result.index for result in results] == [0, 1]
    assert rerank_provider_result_count(results) == 3


class _FakeCohereClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    async def rerank(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(dict(kwargs))
        return self._response


def test_cohere_reranker_non_positive_top_k_returns_empty_without_provider_call() -> None:
    fake_client = _FakeCohereClient(_FakeResponse([]))
    reranker = object.__new__(CohereReranker)
    reranker._client = fake_client
    reranker._model = "rerank-v3.5"

    assert asyncio.run(reranker.rerank("query", ["alpha"], top_k=0)) == []
    assert asyncio.run(reranker.rerank("query", ["alpha"], top_k=-1)) == []
    assert fake_client.calls == []


def test_cohere_reranker_bounds_top_k_and_slices_validated_results() -> None:
    fake_client = _FakeCohereClient(
        _FakeResponse([_FakeRow(0, 0.9), _FakeRow(1, 0.8), _FakeRow(2, 0.7)])
    )
    reranker = object.__new__(CohereReranker)
    reranker._client = fake_client
    reranker._model = "rerank-v3.5"

    results = asyncio.run(reranker.rerank("query", ["alpha", "beta", "gamma"], top_k=2))

    assert fake_client.calls[0]["top_n"] == 2
    assert [result.index for result in results] == [0, 1]
    assert rerank_provider_result_count(results) == 3
