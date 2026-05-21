from __future__ import annotations

import logging

import pytest

from rag_core.search.providers.rerank_results import safe_indexed_rerank_results

LOGGER_NAME = "rag_core.search.providers.rerank_results"
SECRET = "sk-test-secret"


class _DangerousValue:
    def __repr__(self) -> str:
        return f"repr should never be logged {SECRET}\nTraceback (most recent call last):"


DangerousTypeName = type(
    f"TypeNameShouldNeverBeLogged_{SECRET}_Traceback",
    (),
    {},
)


def test_reranker_result_validation_logs_are_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [
        (f"idx-{SECRET}", 0.5),
        (True, 0.5),
        (0, f"score-{SECRET}\nTraceback (most recent call last):"),
        (0, False),
        (0, _DangerousValue()),
        (0, DangerousTypeName()),
        (0, "nan"),
        (0, 0.9),
    ]

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = safe_indexed_rerank_results(
            rows=rows,
            documents=["alpha"],
            provider_name="FakeRerankerProvider",
        )

    assert len(results) == 1
    assert results[0].index == 0
    assert results[0].score == 0.9
    assert results[0].text == "alpha"

    assert "FakeRerankerProvider returned invalid rerank index" in caplog.text
    assert "FakeRerankerProvider returned invalid rerank score" in caplog.text
    assert "FakeRerankerProvider returned non-finite rerank score" in caplog.text
    assert "value_type=str" in caplog.text
    assert "value_type=bool" in caplog.text
    assert "value_type=object" in caplog.text
    assert "reason=invalid_type" in caplog.text
    assert "reason=invalid_value" in caplog.text
    assert SECRET not in caplog.text
    assert "Traceback (most recent call last):" not in caplog.text
    assert "repr should never be logged" not in caplog.text
    assert "_DangerousValue" not in caplog.text
    assert "TypeNameShouldNeverBeLogged" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_reranker_result_validation_drops_duplicate_indices(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = safe_indexed_rerank_results(
            rows=[(0, 0.9), (0, 0.8), (1, 0.7)],
            documents=["alpha", "beta"],
            provider_name="FakeRerankerProvider",
        )

    assert [(row.index, row.score, row.text) for row in results] == [
        (0, 0.9, "alpha"),
        (1, 0.7, "beta"),
    ]
    assert "FakeRerankerProvider returned duplicate rerank index" in caplog.text


def test_reranker_result_validation_sorts_valid_results_by_score_desc() -> None:
    results = safe_indexed_rerank_results(
        rows=[(0, 0.1), (1, "0.9"), (2, 0.5)],
        documents=["alpha", "beta", "gamma"],
        provider_name="FakeRerankerProvider",
    )

    assert [(row.index, row.score, row.text) for row in results] == [
        (1, 0.9, "beta"),
        (2, 0.5, "gamma"),
        (0, 0.1, "alpha"),
    ]
