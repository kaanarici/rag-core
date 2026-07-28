from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from rag_core.search.providers.zeroentropy import ZeroEntropyReranker
from rag_core.search.providers.rerank_results import rerank_provider_result_count
from tests.support import TEST_API_SECRET, assert_caplog_omits_private

LOGGER_NAME = "rag_core.search.providers.rerank_results"

SECRET = TEST_API_SECRET


DangerousTypeName = type(
    f"TypeNameShouldNeverBeLogged_{SECRET}_Traceback",
    (),
    {},
)


class _FakeModels:
    calls: list[dict[str, object]]

    def __init__(self) -> None:
        self.calls = []

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> SimpleNamespace:
        assert model == "zerank-test"
        assert query == "private query"
        assert documents == ["alpha", "beta"]
        assert top_n == 2
        self.calls.append(
            {"model": model, "query": query, "documents": documents, "top_n": top_n}
        )
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=2, relevance_score=1.0),
                SimpleNamespace(index=True, relevance_score=0.9),
                SimpleNamespace(index=f"idx-{SECRET}", relevance_score=0.9),
                SimpleNamespace(index=0, relevance_score=True),
                SimpleNamespace(index=1, relevance_score=False),
                SimpleNamespace(index=0, relevance_score=DangerousTypeName()),
                SimpleNamespace(index=0, relevance_score="not-a-number"),
                SimpleNamespace(index=0, relevance_score="nan"),
                SimpleNamespace(index=1, relevance_score="0.8"),
                SimpleNamespace(index=0, relevance_score=0.7),
            ]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


def test_zeroentropy_reranker_result_validation_logs_are_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reranker = ZeroEntropyReranker.__new__(ZeroEntropyReranker)
    reranker._client = _FakeClient()
    reranker._model = "zerank-test"

    async def run() -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            results = await reranker.rerank(
                "private query",
                ["alpha", "beta"],
                top_k=10,
            )

        assert [(result.index, result.score, result.text) for result in results] == [
            (1, 0.8, "beta"),
            (0, 0.7, "alpha"),
        ]

    asyncio.run(run())

    assert "ZeroEntropyReranker returned invalid rerank index" in caplog.text
    assert "ZeroEntropyReranker returned invalid rerank score" in caplog.text
    assert "ZeroEntropyReranker returned non-finite rerank score" in caplog.text
    assert "reason=invalid_type" in caplog.text
    assert "reason=invalid_value" in caplog.text
    assert "value_type=bool" in caplog.text
    assert "value_type=str" in caplog.text
    assert "value_type=object" in caplog.text
    assert_caplog_omits_private(caplog, "TypeNameShouldNeverBeLogged")


def test_zeroentropy_top_k_non_positive_returns_empty_without_provider_call() -> None:
    calls: list[int] = []
    reranker = ZeroEntropyReranker.__new__(ZeroEntropyReranker)
    reranker._client = SimpleNamespace(
        models=SimpleNamespace(
            rerank=lambda **_kwargs: calls.append(1),
        )
    )
    reranker._model = "zerank-test"

    async def run() -> None:
        assert await reranker.rerank("query", ["alpha", "beta"], top_k=0) == []
        assert await reranker.rerank("query", ["alpha", "beta"], top_k=-2) == []

    asyncio.run(run())
    assert calls == []


def test_zeroentropy_top_k_applies_after_result_validation() -> None:
    calls: list[dict[str, object]] = []

    def rerank(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=3, relevance_score=1.0),
                SimpleNamespace(index=0, relevance_score="nan"),
                SimpleNamespace(index=1, relevance_score=0.8),
                SimpleNamespace(index=0, relevance_score=0.7),
            ]
        )

    reranker = ZeroEntropyReranker.__new__(ZeroEntropyReranker)
    reranker._client = SimpleNamespace(
        models=SimpleNamespace(rerank=rerank)
    )
    reranker._model = "zerank-test"

    async def run() -> None:
        results = await reranker.rerank(
            "query",
            ["alpha", "beta"],
            top_k=1,
        )
        assert [(result.index, result.score, result.text) for result in results] == [
            (1, 0.8, "beta")
        ]
        assert rerank_provider_result_count(results) == 4

    asyncio.run(run())
    assert calls[0]["top_n"] == 1
