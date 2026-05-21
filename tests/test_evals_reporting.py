from __future__ import annotations

import math
from typing import cast

import pytest

from rag_core.evals import EvalCase, EvalResult
from rag_core.evals.reporting import (
    add_quality_gate,
    eval_comparison_report,
    eval_exit_code,
    eval_result_payload,
    eval_profile_comparison_report,
    eval_report,
    format_eval_comparison_report,
    format_eval_profile_comparison_report,
    format_eval_report,
)


def test_eval_report_is_available_from_evals_namespace() -> None:
    report = eval_report([_result("refunds")])

    assert report["case_count"] == 1


def test_eval_report_builds_case_payload_and_mean_metrics() -> None:
    report = eval_report(
        [
            _result("refunds", recall_at_5=1.0, recall_at_10=1.0, mrr=1.0, ndcg_at_10=1.0),
            _result(
                "pricing",
                recall_at_5=0.0,
                recall_at_10=0.5,
                mrr=0.25,
                ndcg_at_10=0.5,
                latency_ms=4.0,
            ),
        ]
    )

    assert report["case_count"] == 2
    assert report["failure_count"] == 0
    assert report["metrics"] == {
        "recall_at_5": 0.5,
        "recall_at_10": 0.75,
        "mrr": 0.625,
        "ndcg_at_10": 0.75,
        "latency_ms": 3.0,
        "latency_p95_ms": 4.0,
        "throughput_qps": 333.3333333333333,
    }
    assert report["cases"] == [
        {
            "case_id": "case/refunds",
            "case_ordinal": 1,
            "query": "refunds query",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "expected_chunk_ids": ["refunds#chunk"],
            "ok": True,
            "retrieved_ids": ["refunds#chunk"],
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "mrr": 1.0,
            "ndcg_at_10": 1.0,
            "latency_ms": 2.0,
        },
        {
            "case_id": "case/pricing",
            "case_ordinal": 2,
            "query": "pricing query",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "expected_chunk_ids": ["pricing#chunk"],
            "ok": True,
            "retrieved_ids": ["pricing#chunk"],
            "recall_at_5": 0.0,
            "recall_at_10": 0.5,
            "mrr": 0.25,
            "ndcg_at_10": 0.5,
            "latency_ms": 4.0,
        },
    ]


def test_eval_result_payload_includes_expected_grades_when_present() -> None:
    payload = eval_result_payload(
        EvalResult(
            case=EvalCase(
                query="refunds query",
                namespace="acme",
                corpus_ids=("help",),
                expected_chunk_ids=("refunds#chunk",),
                expected_grades={"refunds#chunk": 3},
                case_id="case/refunds",
            ),
            retrieved_ids=("refunds#chunk",),
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            ndcg_at_10=1.0,
            latency_ms=2.0,
        )
    )

    assert payload["expected_grades"] == {"refunds#chunk": 3}


def test_eval_comparison_report_includes_metric_deltas() -> None:
    report = eval_comparison_report(
        [_result("refunds", recall_at_5=0.0, recall_at_10=0.5, mrr=0.25)],
        [_result("refunds", recall_at_5=1.0, recall_at_10=1.0, mrr=1.0)],
    )

    assert report["metric_deltas"] == {
        "recall_at_5": 1.0,
        "recall_at_10": 0.5,
        "mrr": 0.75,
        "ndcg_at_10": 0.0,
        "latency_ms": 0.0,
        "latency_p95_ms": 0.0,
        "throughput_qps": 0.0,
    }
    assert format_eval_comparison_report(report) == [
        "Cases: 1",
        "Baseline Mean: recall@5=0.000 recall@10=0.500 mrr=0.250 ndcg@10=1.000 latency_ms=2.000 latency_p95_ms=2.000 throughput_qps=500.000",
        "Reranked Mean: recall@5=1.000 recall@10=1.000 mrr=1.000 ndcg@10=1.000 latency_ms=2.000 latency_p95_ms=2.000 throughput_qps=500.000",
        "Delta: recall@5=1.000 recall@10=0.500 mrr=0.750 ndcg@10=0.000 latency_ms=0.000 latency_p95_ms=0.000 throughput_qps=0.000",
    ]


def test_eval_comparison_report_rejects_case_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"baseline has 2 case\(s\), reranked has 1",
    ):
        eval_comparison_report(
            [_result("refunds"), _result("pricing")],
            [_result("refunds")],
        )


def test_eval_comparison_report_rejects_case_order_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="case 1 is baseline='case/refunds', reranked='case/pricing'",
    ):
        eval_comparison_report(
            [_result("refunds"), _result("pricing")],
            [_result("pricing"), _result("refunds")],
        )


def test_eval_comparison_report_rejects_case_payload_mismatch_for_same_case_id() -> None:
    with pytest.raises(
        ValueError,
        match="matching case payloads; case 1 differs for case_id 'case/refunds'",
    ):
        eval_comparison_report(
            [_result("refunds")],
            [_result("refunds", query="updated refunds query")],
        )


def test_eval_comparison_report_compares_implicit_case_identity() -> None:
    baseline = _result_without_case_id("refunds")
    candidate = _result_without_case_id("refunds", mrr=0.5)

    report = eval_comparison_report([baseline], [candidate])

    assert report["case_count"] == 1
    metric_deltas = cast("dict[str, float]", report["metric_deltas"])
    assert metric_deltas["mrr"] == -0.5


def test_eval_comparison_report_rejects_implicit_case_mismatch() -> None:
    with pytest.raises(ValueError, match="eval comparison requires the same ordered case_ids"):
        eval_comparison_report(
            [_result_without_case_id("refunds")],
            [_result_without_case_id("pricing")],
        )


def test_eval_comparison_report_does_not_echo_implicit_case_payloads() -> None:
    with pytest.raises(ValueError) as exc_info:
        eval_comparison_report(
            [_result_without_case_id("refunds")],
            [_result_without_case_id("pricing")],
        )

    message = str(exc_info.value)
    assert "<implicit case 1>" in message
    assert "refunds query" not in message
    assert "pricing query" not in message
    assert "acme" not in message
    assert "help" not in message
    assert "refunds#chunk" not in message
    assert "pricing#chunk" not in message


def test_eval_profile_comparison_report_uses_stable_baseline_profile() -> None:
    report = eval_profile_comparison_report(
        {
            "fast": [_result("refunds", recall_at_5=0.0, recall_at_10=0.5, mrr=0.25)],
            "balanced": [_result("refunds", recall_at_5=1.0, recall_at_10=1.0, mrr=1.0)],
        }
    )

    assert report["baseline_profile"] == "balanced"
    assert report["metric_deltas"] == {
        "balanced": {
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
            "latency_ms": 0.0,
            "latency_p95_ms": 0.0,
            "throughput_qps": 0.0,
        },
        "fast": {
            "recall_at_5": -1.0,
            "recall_at_10": -0.5,
            "mrr": -0.75,
            "ndcg_at_10": 0.0,
            "latency_ms": 0.0,
            "latency_p95_ms": 0.0,
            "throughput_qps": 0.0,
        },
    }
    assert format_eval_profile_comparison_report(report) == [
        "Cases: 1",
        "Baseline Profile: balanced",
        "balanced Mean: recall@5=1.000 recall@10=1.000 mrr=1.000 ndcg@10=1.000 latency_ms=2.000 latency_p95_ms=2.000 throughput_qps=500.000",
        "fast Mean: recall@5=0.000 recall@10=0.500 mrr=0.250 ndcg@10=1.000 latency_ms=2.000 latency_p95_ms=2.000 throughput_qps=500.000",
        "fast Delta: recall@5=-1.000 recall@10=-0.500 mrr=-0.750 ndcg@10=0.000 latency_ms=0.000 latency_p95_ms=0.000 throughput_qps=0.000",
    ]


def test_eval_profile_comparison_report_accepts_explicit_baseline_profile() -> None:
    report = eval_profile_comparison_report(
        {
            "fast": [_result("refunds", recall_at_5=0.0, recall_at_10=0.5, mrr=0.25)],
            "balanced": [_result("refunds", recall_at_5=1.0, recall_at_10=1.0, mrr=1.0)],
        },
        baseline_profile="fast",
    )

    assert report["baseline_profile"] == "fast"
    metric_deltas = cast("dict[str, dict[str, float]]", report["metric_deltas"])
    assert metric_deltas["fast"]["mrr"] == 0.0
    assert metric_deltas["balanced"]["mrr"] == 0.75


def test_eval_profile_comparison_report_rejects_profile_case_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="case 1 is balanced='case/refunds', fast='case/pricing'",
    ):
        eval_profile_comparison_report(
            {
                "balanced": [_result("refunds")],
                "fast": [_result("pricing")],
            }
        )


def test_eval_profile_comparison_report_rejects_profile_case_payload_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="matching case payloads; case 1 differs for case_id 'case/refunds'",
    ):
        eval_profile_comparison_report(
            {
                "balanced": [_result("refunds")],
                "fast": [_result("refunds", query="updated refunds query")],
            }
        )


def test_quality_gate_sets_exit_code_and_formats_failures() -> None:
    report = eval_report([_result("refunds", recall_at_5=0.5, latency_ms=30.0)])

    add_quality_gate(
        report,
        {"eval": report},
        {
            "recall_at_5": {"minimum": 0.75},
            "latency_ms": {"maximum": 20.0},
            "latency_p95_ms": {"maximum": 20.0},
            "throughput_qps": {"minimum": 100.0},
        },
    )

    assert eval_exit_code(report) == 1
    assert format_eval_report(report)[-5:] == [
        "Gate: fail",
        "- eval recall_at_5: 0.500 >= 0.750",
        "- eval latency_ms: 30.000 <= 20.000",
        "- eval latency_p95_ms: 30.000 <= 20.000",
        "- eval throughput_qps: 33.333 >= 100.000",
    ]


def test_quality_gate_pass_formats_without_failures() -> None:
    report = eval_report([_result("refunds", recall_at_5=1.0, latency_ms=5.0)])

    add_quality_gate(
        report,
        {"eval": report},
        {
            "recall_at_5": {"minimum": 0.75},
            "latency_ms": {"maximum": 20.0},
        },
    )

    assert eval_exit_code(report) == 0
    assert format_eval_report(report)[-1] == "Gate: pass"


def test_quality_gate_fails_when_eval_has_search_failures() -> None:
    report = eval_report(
        [
            EvalResult(
                case=EvalCase(
                    query="refunds query",
                    namespace="acme",
                    corpus_ids=("help",),
                    expected_chunk_ids=("refunds#chunk",),
                    case_id="case/refunds",
                ),
                retrieved_ids=(),
                recall_at_5=0.0,
                recall_at_10=0.0,
                mrr=0.0,
                ndcg_at_10=0.0,
                latency_ms=5.0,
                error_type="RuntimeError",
            )
        ]
    )

    add_quality_gate(report, {"eval": report}, {"latency_ms": {"maximum": 20.0}})

    assert eval_exit_code(report) == 1
    assert format_eval_report(report)[-2:] == [
        "Gate: fail",
        "- eval search_failures: 1.000 == 0.000",
    ]


def test_quality_gate_fails_on_malformed_metric_values() -> None:
    report = {
        "case_count": 1,
        "metrics": {"recall_at_5": "not-a-number"},
        "cases": [],
    }

    add_quality_gate(report, {"eval": report}, {"recall_at_5": {"maximum": 0.0}})

    assert eval_exit_code(report) == 1
    assert format_eval_report(report)[-2:] == [
        "Gate: fail",
        "- eval recall_at_5: invalid is numeric",
    ]


def test_quality_gate_fails_on_non_finite_metric_values() -> None:
    report = {
        "case_count": 1,
        "metrics": {
            "recall_at_5": math.nan,
            "latency_ms": math.inf,
            "mrr": -math.inf,
        },
        "cases": [],
    }

    add_quality_gate(
        report,
        {"eval": report},
        {
            "recall_at_5": {"maximum": 1.0},
            "latency_ms": {"maximum": 100.0},
            "mrr": {"minimum": 0.0},
        },
    )

    quality_gate = cast("dict[str, object]", report["quality_gate"])
    failures = cast("list[dict[str, object]]", quality_gate["failures"])
    assert quality_gate["passed"] is False
    assert failures == [
        {
            "scope": "eval",
            "metric": "recall_at_5",
            "actual": "invalid",
            "operator": "is",
            "threshold": "numeric",
        },
        {
            "scope": "eval",
            "metric": "latency_ms",
            "actual": "invalid",
            "operator": "is",
            "threshold": "numeric",
        },
        {
            "scope": "eval",
            "metric": "mrr",
            "actual": "invalid",
            "operator": "is",
            "threshold": "numeric",
        },
    ]


def test_eval_report_rejects_non_finite_case_metrics() -> None:
    with pytest.raises(ValueError, match="eval metric recall_at_5 must be a finite number"):
        eval_report(
            [
                _result(
                    "refunds",
                    recall_at_5=math.nan,
                )
            ]
        )


def _result(
    case_id: str,
    *,
    query: str | None = None,
    recall_at_5: float = 1.0,
    recall_at_10: float = 1.0,
    mrr: float = 1.0,
    ndcg_at_10: float = 1.0,
    latency_ms: float = 2.0,
) -> EvalResult:
    return EvalResult(
        case=EvalCase(
            query=query or f"{case_id} query",
            namespace="acme",
            corpus_ids=("help",),
            expected_chunk_ids=(f"{case_id}#chunk",),
            case_id=f"case/{case_id}",
        ),
        retrieved_ids=(f"{case_id}#chunk",),
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mrr=mrr,
        ndcg_at_10=ndcg_at_10,
        latency_ms=latency_ms,
    )


def _result_without_case_id(
    case_id: str,
    *,
    mrr: float = 1.0,
) -> EvalResult:
    return EvalResult(
        case=EvalCase(
            query=f"{case_id} query",
            namespace="acme",
            corpus_ids=("help",),
            expected_chunk_ids=(f"{case_id}#chunk",),
        ),
        retrieved_ids=(f"{case_id}#chunk",),
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr=mrr,
        ndcg_at_10=1.0,
        latency_ms=2.0,
    )
