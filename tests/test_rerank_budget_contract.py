from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.search.request_models import RerankBudget


def test_rerank_budget_accepts_positive_limits() -> None:
    budget = RerankBudget(candidate_count=10, timeout_seconds=2.5, max_output=5)

    assert budget.candidate_count == 10
    assert budget.timeout_seconds == 2.5
    assert budget.max_output == 5


@pytest.mark.parametrize("value", [0, -1, True, cast(Any, 1.5), cast(Any, "3")])
def test_rerank_budget_rejects_invalid_candidate_count(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        RerankBudget(candidate_count=cast(Any, value))

    assert str(exc_info.value) == "RerankBudget.candidate_count must be positive"


@pytest.mark.parametrize(
    "value",
    [0, -1, True, float("nan"), float("inf"), cast(Any, "3")],
)
def test_rerank_budget_rejects_invalid_timeout_seconds(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        RerankBudget(timeout_seconds=cast(Any, value))

    assert str(exc_info.value) == "RerankBudget.timeout_seconds must be positive"


@pytest.mark.parametrize("value", [0, -1, True, cast(Any, 1.5), cast(Any, "3")])
def test_rerank_budget_rejects_invalid_max_output(value: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        RerankBudget(max_output=cast(Any, value))

    assert str(exc_info.value) == "RerankBudget.max_output must be positive"
