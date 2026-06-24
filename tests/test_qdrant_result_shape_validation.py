from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from rag_core.search.policy import DEFAULT_POLICY
from rag_core.search.providers.qdrant_payloads import (
    _point_to_result,
    _score_result_value,
)

SECRET = "sk-test-secret"


class _SecretId:
    def __str__(self) -> str:
        return "point " + SECRET


def _point(**overrides: object) -> Any:
    point = SimpleNamespace(
        id="00000000-0000-4000-8000-000000000001",
        score=0.75,
        payload={
            "text": "alpha",
            "content_type": "document",
            "source_type": "file",
        },
    )
    for key, value in overrides.items():
        setattr(point, key, value)
    return point


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(True, id="bool"),
        pytest.param("0.625", id="numeric-string"),
        pytest.param("nan", id="nan-string"),
        pytest.param(float("nan"), id="nan-float"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(-float("inf"), id="negative-inf"),
        pytest.param(10**10000, id="huge-int"),
        pytest.param(object(), id="object"),
        pytest.param("not-a-number " + SECRET, id="secret-string"),
    ],
)
def test_qdrant_point_to_result_rejects_malformed_scores_without_leaking(
    score: object,
) -> None:
    point = _point(score=score)
    point.payload["text"] = "private " + SECRET

    with pytest.raises(ValueError) as exc_info:
        _point_to_result(point, policy=DEFAULT_POLICY)

    message = str(exc_info.value)
    assert message == "qdrant result point returned invalid field: score"
    assert SECRET not in message


@pytest.mark.parametrize("score", [pytest.param(None, id="none")])
def test_qdrant_point_to_result_reports_missing_scores_without_leaking(
    score: object,
) -> None:
    point = _point(score=score)
    point.payload["text"] = "private " + SECRET

    with pytest.raises(ValueError) as exc_info:
        _point_to_result(point, policy=DEFAULT_POLICY)

    message = str(exc_info.value)
    assert message == "qdrant result point missing required field: score"
    assert SECRET not in message


def test_qdrant_point_to_result_reports_absent_score_without_leaking() -> None:
    point = _point()
    delattr(point, "score")
    point.payload["text"] = "private " + SECRET

    with pytest.raises(ValueError) as exc_info:
        _point_to_result(point, policy=DEFAULT_POLICY)

    message = str(exc_info.value)
    assert message == "qdrant result point missing required field: score"
    assert SECRET not in message


def test_qdrant_point_to_result_accepts_valid_float_score() -> None:
    result = _point_to_result(_point(score=0.625), policy=DEFAULT_POLICY)

    assert result.score == 0.625


def test_qdrant_point_to_result_accepts_valid_int_score() -> None:
    result = _point_to_result(_point(score=1), policy=DEFAULT_POLICY)

    assert result.score == 1.0


def test_qdrant_point_to_result_reports_missing_id_safely() -> None:
    point = _point(score=0.5)
    delattr(point, "id")
    point.payload["text"] = "private " + SECRET

    with pytest.raises(ValueError) as exc_info:
        _point_to_result(point, policy=DEFAULT_POLICY)

    message = str(exc_info.value)
    assert message == "qdrant result point missing required field: id"
    assert SECRET not in message


@pytest.mark.parametrize(
    "point_id",
    [
        pytest.param("", id="empty-string"),
        pytest.param("   ", id="blank-string"),
        pytest.param("point-valid", id="non-uuid-string"),
        pytest.param(True, id="bool"),
        pytest.param(object(), id="object"),
        pytest.param(_SecretId(), id="secret-stringifier"),
    ],
)
def test_qdrant_point_to_result_reports_malformed_id_safely(point_id: object) -> None:
    point = _point(id=point_id)
    point.payload["text"] = "private " + SECRET

    with pytest.raises(ValueError) as exc_info:
        _point_to_result(point, policy=DEFAULT_POLICY)

    message = str(exc_info.value)
    assert message == "qdrant result point missing required field: id"
    assert SECRET not in message


def test_qdrant_point_to_result_accepts_non_empty_string_id() -> None:
    point_id = "00000000-0000-4000-8000-000000000011"
    result = _point_to_result(_point(id=point_id), policy=DEFAULT_POLICY)

    assert result.id == point_id


def test_qdrant_point_to_result_normalizes_provider_native_uuid_ids() -> None:
    point_id = uuid4()
    result = _point_to_result(_point(id=point_id), policy=DEFAULT_POLICY)

    assert result.id == str(point_id)


def test_qdrant_point_to_result_rejects_provider_native_int_ids() -> None:
    with pytest.raises(ValueError, match="missing required field: id"):
        _point_to_result(_point(id=123), policy=DEFAULT_POLICY)


def test_qdrant_score_result_value_preserves_negative_finite_scores() -> None:
    assert _score_result_value(-0.25) == -0.25
