"""Minimal aggregate eval reports for examples and tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_core.evals.runner import EvalResult

from .report_models import EvalReport, EvalRunMetadata, finite_metric_value


def eval_result_payload(result: EvalResult) -> EvalReport:
    case = result.case
    payload: EvalReport = {
        "case_id": case.case_id,
        "query": case.query,
        "namespace": case.namespace,
        "corpus_ids": list(case.corpus_ids),
        "expected_ids": list(case.expected_ids),
        "expected_chunk_ids": list(case.expected_chunk_ids),
        "ok": result.error_type is None,
        "retrieved_ids": list(result.retrieved_ids),
        "recall_at_5": finite_metric_value(result.recall_at_5, name="recall_at_5"),
        "recall_at_10": finite_metric_value(result.recall_at_10, name="recall_at_10"),
        "mrr": finite_metric_value(result.mrr, name="mrr"),
        "ndcg_at_10": finite_metric_value(result.ndcg_at_10, name="ndcg_at_10"),
        "latency_ms": finite_metric_value(result.latency_ms, name="latency_ms"),
    }
    if case.expected_grades is not None:
        payload["expected_grades"] = dict(case.expected_grades)
    if result.error_type is not None:
        payload["error_type"] = result.error_type
    return payload


def eval_report(
    results: Sequence[EvalResult],
    *,
    run: EvalRunMetadata | None = None,
) -> EvalReport:
    result_payloads = []
    for case_ordinal, result in enumerate(results, start=1):
        payload = eval_result_payload(result)
        payload["case_ordinal"] = case_ordinal
        result_payloads.append(payload)
    report: EvalReport = {
        "case_count": len(result_payloads),
        "failure_count": sum(1 for payload in result_payloads if payload.get("ok") is False),
        "metrics": _aggregate_metrics(result_payloads),
        "cases": result_payloads,
    }
    if run:
        report["run"] = dict(run)
    return report


def redact_eval_report(report: Mapping[str, object]) -> EvalReport:
    redacted: EvalReport = {}
    for key, value in report.items():
        if key == "cases" and isinstance(value, Sequence) and not isinstance(value, str):
            redacted[key] = [_redacted_case_payload(item) for item in value]
        elif isinstance(value, Mapping):
            redacted[key] = redact_eval_report(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_eval_report(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            redacted[key] = value
    return redacted


def _redacted_case_payload(payload: object) -> EvalReport:
    if not isinstance(payload, Mapping):
        return {}
    redacted = dict(payload)
    case_ordinal = redacted.get("case_ordinal")
    if isinstance(case_ordinal, int):
        redacted["case_label"] = f"case-{case_ordinal}"
    redacted.pop("case_id", None)
    redacted.pop("query", None)
    redacted.pop("expected_ids", None)
    redacted.pop("expected_chunk_ids", None)
    redacted.pop("expected_grades", None)
    return redacted


def _aggregate_metrics(payloads: Sequence[Mapping[str, object]]) -> EvalReport:
    return {
        "recall_at_5": _mean_payload_value(payloads, "recall_at_5"),
        "recall_at_10": _mean_payload_value(payloads, "recall_at_10"),
        "mrr": _mean_payload_value(payloads, "mrr"),
        "ndcg_at_10": _mean_payload_value(payloads, "ndcg_at_10"),
        "latency_ms": _mean_payload_value(payloads, "latency_ms"),
        "latency_p95_ms": _latency_percentile(payloads, percentile=0.95),
        "throughput_qps": _throughput_qps(payloads),
    }


def _mean_payload_value(payloads: Sequence[Mapping[str, object]], key: str) -> float:
    if not payloads:
        return 0.0
    return sum(
        finite_metric_value(payload[key], name=key) for payload in payloads
    ) / len(payloads)


def _latency_percentile(
    payloads: Sequence[Mapping[str, object]],
    *,
    percentile: float,
) -> float:
    if not payloads:
        return 0.0
    values = sorted(
        finite_metric_value(payload["latency_ms"], name="latency_ms")
        for payload in payloads
    )
    rank = max(0, min(len(values) - 1, int(len(values) * percentile + 0.999999) - 1))
    return values[rank]


def _throughput_qps(payloads: Sequence[Mapping[str, object]]) -> float:
    total_latency_ms = sum(
        finite_metric_value(payload["latency_ms"], name="latency_ms")
        for payload in payloads
    )
    if not payloads or total_latency_ms <= 0.0:
        return 0.0
    return len(payloads) * 1000.0 / total_latency_ms
