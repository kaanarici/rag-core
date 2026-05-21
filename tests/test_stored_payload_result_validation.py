from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.search.stored_payload import build_stored_payload, payload_to_result
from rag_core.search.types import ContentType

SECRET = "sk-test-secret"


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": "hello",
        "content_type": "document",
        "source_type": "file",
    }
    payload.update(overrides)
    return payload


def test_payload_to_result_ignores_malformed_optional_ints_without_leaking_values() -> None:
    result = payload_to_result(
        point_id="point-1",
        payload=_base_payload(
            chunk_index=f"idx-{SECRET}",
            chunk_word_count=False,
            chunk_token_estimate="not-a-number\nTraceback (most recent call last):",
        ),
        score=0.5,
    )

    assert result.chunk_index is None
    assert result.chunk_word_count is None
    assert result.chunk_token_estimate is None


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(True, id="bool"),
        pytest.param(cast(Any, object()), id="object"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(-float("inf"), id="negative-inf"),
    ],
)
def test_payload_to_result_sanitizes_malformed_scores(score: float) -> None:
    result = payload_to_result(
        point_id="point-1",
        payload=_base_payload(),
        score=score,
    )

    assert result.score == 0.0
    assert isinstance(result.score, float)


def test_payload_to_result_preserves_valid_finite_score() -> None:
    result = payload_to_result(
        point_id="point-1",
        payload=_base_payload(),
        score=0.75,
    )

    assert result.score == 0.75


def test_build_stored_payload_ignores_non_string_filter_metadata_keys() -> None:
    payload = build_stored_payload(
        namespace="acme",
        corpus_id="help",
        document_id="doc-1",
        document_key="doc.md",
        content_sha256="sha",
        processing_version="v1",
        filename="doc.md",
        mime_type="text/markdown",
        source_type="file",
        document_path=None,
        chunk_index=0,
        chunk_text="hello",
        chunk_token_count=1,
        payload_text="hello",
        content_type=ContentType.DOCUMENT,
        embedding_model="text-embedding-3-small",
        chunker_strategy="markdown",
        title=None,
        filter_metadata={cast(Any, 1): "bad-key", "category": "guide"},
    )

    assert "category" in payload
    assert 1 not in payload


def test_build_stored_payload_preserves_filterable_geo_points() -> None:
    payload = build_stored_payload(
        namespace="acme",
        corpus_id="help",
        document_id="doc-1",
        document_key="doc.md",
        content_sha256="sha",
        processing_version="v1",
        filename="doc.md",
        mime_type="text/markdown",
        source_type="file",
        document_path=None,
        chunk_index=0,
        chunk_text="hello",
        chunk_token_count=1,
        payload_text="hello",
        content_type=ContentType.DOCUMENT,
        embedding_model="text-embedding-3-small",
        chunker_strategy="markdown",
        title=None,
        filter_metadata={
            "location": {"lat": 40.7484, "lon": -73.9857},
            "bad_location": {"lat": True, "lon": -73.9857},
            "nested": {"category": "guide"},
        },
    )

    assert payload["location"] == {"lat": 40.7484, "lon": -73.9857}
    assert "bad_location" not in payload
    assert "nested" not in payload


@pytest.mark.parametrize("missing_field", ["content_type", "text"])
def test_payload_to_result_reports_missing_required_fields_safely(
    missing_field: str,
) -> None:
    payload = _base_payload(text=f"private {SECRET}")
    del payload[missing_field]

    with pytest.raises(ValueError) as exc_info:
        payload_to_result(point_id="point-1", payload=payload, score=0.5)

    message = str(exc_info.value)
    assert message == f"search payload missing required field: {missing_field}"
    assert SECRET not in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", {"nested": "object"}),
        ("content_type", True),
    ],
)
def test_payload_to_result_rejects_malformed_required_string_fields(
    field: str,
    value: object,
) -> None:
    payload = _base_payload(**{field: value})

    with pytest.raises(ValueError) as exc_info:
        payload_to_result(point_id="point-1", payload=payload, score=0.5)

    assert str(exc_info.value) == f"search payload field must be a string: {field}"
