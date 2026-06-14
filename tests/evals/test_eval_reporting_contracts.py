from __future__ import annotations

from typing import cast

from rag_core.evals import (
    EvalCase,
    EvalResult,
    eval_report,
    redact_eval_report,
)


def test_redact_eval_report_strips_retrieved_identifiers() -> None:
    case = EvalCase(
        query="secret query",
        namespace="rt",
        corpus_ids=("docs",),
        expected_ids=("doc-secret",),
    )
    result = EvalResult(
        case=case,
        retrieved_ids=("doc-secret", "local:private/path.md#source:abc"),
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=1.0,
        ndcg_at_10=1.0,
        latency_ms=10.0,
    )

    redacted = redact_eval_report(eval_report([result]))
    [case_payload] = cast(list[dict[str, object]], redacted["cases"])

    assert case_payload["case_label"] == "case-1"
    assert "retrieved_ids" not in case_payload
    assert "expected_ids" not in case_payload
    assert "query" not in case_payload
