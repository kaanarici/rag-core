from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_core.evals import EvalCase, load_cases


def test_load_cases_validates_and_loads_jsonl_rows(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": ["doc-a#chunk-1", "doc-a#chunk-2"],
                "expected_grades": {"doc-a#chunk-1": 3, "doc-a#chunk-2": 1},
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    assert load_cases(path) == [
        EvalCase(
            query="billing policy",
            namespace="acme",
            corpus_ids=("help",),
            expected_chunk_ids=("doc-a#chunk-1", "doc-a#chunk-2"),
            expected_grades={"doc-a#chunk-1": 3, "doc-a#chunk-2": 1},
        )
    ]
    assert load_cases(path)[0].expected_ids == ("doc-a#chunk-1", "doc-a#chunk-2")


def test_load_cases_accepts_legacy_expected_chunk_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "billing policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["doc-a"],
            }
        ),
        encoding="utf-8",
    )

    [case] = load_cases(path)

    assert case.expected_ids == ("doc-a",)
    assert case.expected_chunk_ids == ("doc-a",)


def test_load_cases_strips_stable_identifier_fields(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": " case-1 ",
                "query": " billing policy ",
                "namespace": " acme ",
                "corpus_ids": [" help "],
                "expected_ids": [" doc-a#chunk-1 "],
                "expected_grades": {" doc-a#chunk-1 ": 3},
            }
        ),
        encoding="utf-8",
    )

    [case] = load_cases(path)

    assert case == EvalCase(
        case_id="case-1",
        query="billing policy",
        namespace="acme",
        corpus_ids=("help",),
        expected_chunk_ids=("doc-a#chunk-1",),
        expected_grades={"doc-a#chunk-1": 3},
    )


def test_load_cases_rejects_duplicate_case_id_after_stripping(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case-1",
                        "query": "q1",
                        "namespace": "acme",
                        "corpus_ids": ["help"],
                        "expected_ids": ["doc"],
                    }
                ),
                json.dumps(
                    {
                        "case_id": " case-1 ",
                        "query": "q2",
                        "namespace": "acme",
                        "corpus_ids": ["help"],
                        "expected_ids": ["doc"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case_id 'case-1'"):
        load_cases(path)


@pytest.mark.parametrize(
    ("row", "message"),
    (
        (["not", "an", "object"], "case must be a JSON object"),
        (
            {
                "query": "",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": ["doc"],
            },
            "query must be a non-empty string",
        ),
        (
            {
                "query": "q",
                "namespace": "acme",
                "corpus_ids": "help",
                "expected_ids": ["doc"],
            },
            "corpus_ids must be a non-empty string array",
        ),
        (
            {
                "query": "q",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": [],
            },
            "expected_ids must be a non-empty string array",
        ),
        (
            {
                "query": "q",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": ["doc-a#chunk-secret", " doc-a#chunk-secret "],
            },
            "expected_ids must not contain duplicate ids",
        ),
        (
            {
                "query": "q",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": ["doc"],
                "expected_grades": {"doc-a#chunk-secret": True},
            },
            "expected_grades values must be non-negative integers",
        ),
        (
            {
                "query": "q",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": ["doc"],
                "expected_grades": {"other-doc": 1},
            },
            "expected_grades positive ids must match expected_ids",
        ),
        (
            {
                "query": "q",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": ["doc"],
                "expected_grades": {"doc": 0},
            },
            "expected_grades positive ids must match expected_ids",
        ),
        (
            {
                "query": "q",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_ids": ["doc"],
                "expected_chunk_ids": ["doc"],
            },
            "use expected_ids or expected_chunk_ids, not both",
        ),
    ),
)
def test_load_cases_rejects_malformed_rows_without_echoing_payload(
    tmp_path: Path,
    row: object,
    message: str,
) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)

    error = str(exc_info.value)
    assert f"{path}:1:" in error
    assert message in error
    assert "not an object" not in error
    assert "doc-a#chunk" not in error


def test_load_cases_rejects_invalid_json_without_echoing_line(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"query": "secret customer question"', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)

    error = str(exc_info.value)
    assert f"{path}:1:" in error
    assert "invalid JSON" in error
    assert "secret customer question" not in error


def test_load_cases_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no eval cases found"):
        load_cases(path)


def test_load_cases_rejects_all_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(" \n\t\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no eval cases found"):
        load_cases(path)


def test_load_cases_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"query":"q1","query":"q2","namespace":"acme","corpus_ids":["help"],"expected_ids":["doc"]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate object key 'query'"):
        load_cases(path)
