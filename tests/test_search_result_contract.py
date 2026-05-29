from __future__ import annotations

from typing import Any, cast

import pytest

from rag_core.search.types import SearchResult


def _result(result_id: object) -> SearchResult:
    return SearchResult(
        id=cast(Any, result_id),
        text="alpha",
        score=0.5,
        content_type="document",
        source_type="file",
    )


def test_search_result_accepts_non_empty_string_id() -> None:
    result = _result("point-1")

    assert result.id == "point-1"


def test_search_result_positional_fields_keep_document_identity_order() -> None:
    result = SearchResult(
        "point-1",
        "alpha",
        0.5,
        "document",
        "file",
        "doc-1",
        "corpus-1",
        "docs/a.md",
    )

    assert result.document_id == "doc-1"
    assert result.corpus_id == "corpus-1"
    assert result.document_key == "docs/a.md"
    assert result.namespace is None


def test_search_result_text_is_clean_chunk_content_by_contract() -> None:
    result = SearchResult(
        id="point-1",
        text="Clean chunk body.",
        score=0.42,
        content_type="document",
        source_type="file",
        metadata={
            "team": "support",
            "rerank": {"provider_score": 0.98, "search_score": 0.42},
        },
    )

    assert result.text == "Clean chunk body."
    assert result.score == 0.42
    assert result.metadata["team"] == "support"
    assert result.metadata["rerank"] == {
        "provider_score": 0.98,
        "search_score": 0.42,
    }


@pytest.mark.parametrize(
    "result_id",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param(True, id="bool"),
        pytest.param(123, id="int"),
        pytest.param(object(), id="object"),
    ],
)
def test_search_result_rejects_invalid_ids(result_id: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        _result(result_id)

    assert str(exc_info.value) == "SearchResult.id must be a non-empty string"
