from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_core.evals import EvalCase, EvalResult, load_cases
from rag_core.evals.reporting import eval_result_payload


def test_load_cases_accepts_optional_case_id(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "billing/refund-policy",
                "query": "refund policy",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["refunds.md#chunk-1"],
            }
        ),
        encoding="utf-8",
    )

    [case] = load_cases(path)

    assert case.case_id == "billing/refund-policy"


def test_load_cases_rejects_blank_case_id_without_echoing_payload(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "",
                "query": "secret refund question",
                "namespace": "acme",
                "corpus_ids": ["help"],
                "expected_chunk_ids": ["secret-doc#chunk-1"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)

    error = str(exc_info.value)
    assert f"{path}:1:" in error
    assert "case_id must be a non-empty string when set" in error
    assert "secret refund question" not in error
    assert "secret-doc" not in error


def test_load_cases_rejects_duplicate_case_id_with_line_context(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "case_id": "billing/refund-policy",
                        "query": "refund policy",
                        "namespace": "acme",
                        "corpus_ids": ["help"],
                        "expected_chunk_ids": ["refunds.md#chunk-1"],
                    }
                ),
                json.dumps(
                    {
                        "case_id": "billing/refund-policy",
                        "query": "another refund policy",
                        "namespace": "acme",
                        "corpus_ids": ["help"],
                        "expected_chunk_ids": ["refunds.md#chunk-2"],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_cases(path)

    error = str(exc_info.value)
    assert f"{path}:2:" in error
    assert "duplicate case_id 'billing/refund-policy' (first seen at line 1)" in error


def test_eval_result_payload_includes_stable_case_id() -> None:
    result = EvalResult(
        case=EvalCase(
            query="refund policy",
            namespace="acme",
            corpus_ids=("help",),
            expected_chunk_ids=("refunds.md#chunk-1",),
            case_id="billing/refund-policy",
        ),
        retrieved_ids=("refunds.md#chunk-1",),
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        ndcg_at_10=1.0,
        latency_ms=2.5,
    )

    payload = eval_result_payload(result)

    assert payload["case_id"] == "billing/refund-policy"
    assert payload["expected_ids"] == ["refunds.md#chunk-1"]
    assert payload["expected_chunk_ids"] == ["refunds.md#chunk-1"]
