"""Minimal aggregate eval reports for examples and tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from rag_core.evals.runner import EvalResult

from .report_models import EvalReport, EvalRunMetadata, finite_metric_value


def eval_result_payload(result: EvalResult) -> EvalReport:
    case = result.case
    payload: EvalReport = {
        "case_id": case.case_id,
        "query": case.query,
        "namespace": case.namespace,
        "collections": list(case.collections),
        "expected_ids": list(case.expected_ids),
        "ok": result.error_type is None,
        "retrieved_ids": list(result.retrieved_ids),
        "recall_at_5": finite_metric_value(result.recall_at_5, name="recall_at_5"),
        "recall_at_10": finite_metric_value(result.recall_at_10, name="recall_at_10"),
        "mrr": finite_metric_value(result.mrr, name="mrr"),
        "ndcg_at_10": finite_metric_value(result.ndcg_at_10, name="ndcg_at_10"),
        "latency_ms": finite_metric_value(result.latency_ms, name="latency_ms"),
    }
    if _case_has_context_expectations(case):
        payload.update(
            {
                "expected_context_contains": list(case.expected_context_contains),
                "forbidden_context_contains": list(case.forbidden_context_contains),
                "forbidden_private_identifiers": list(
                    case.forbidden_private_identifiers
                ),
                "expected_citation_count_min": case.expected_citation_count_min,
                "expected_source_count_min": case.expected_source_count_min,
                "max_context_chars": case.max_context_chars,
                "max_context_tokens": case.max_context_tokens,
                "context_recall": finite_metric_value(
                    result.context_recall,
                    name="context_recall",
                ),
                "citation_count": result.citation_count,
                "source_count": result.source_count,
                "forbidden_leak_count": result.forbidden_leak_count,
                "context_token_estimate": result.context_token_estimate,
                "context_char_count": result.context_char_count,
                "context_contains_pass": result.context_contains_pass,
                "prompt_safety_pass": result.prompt_safety_pass,
            }
        )
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
    run_payload = dict(run) if run else None
    result_payloads = []
    for case_ordinal, result in enumerate(results, start=1):
        payload = eval_result_payload(result)
        payload["case_ordinal"] = case_ordinal
        result_payloads.append(payload)
    report: EvalReport = {
        "case_count": len(result_payloads),
        "failure_count": sum(1 for payload in result_payloads if payload.get("ok") is False),
        "metrics": _aggregate_metrics(result_payloads, run=run_payload),
        "cases": result_payloads,
    }
    if run_payload:
        report["run"] = run_payload
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
    redacted.pop("expected_grades", None)
    redacted.pop("expected_context_contains", None)
    redacted.pop("forbidden_context_contains", None)
    redacted.pop("forbidden_private_identifiers", None)
    redacted.pop("retrieved_ids", None)
    return redacted


def _aggregate_metrics(
    payloads: Sequence[Mapping[str, object]],
    *,
    run: Mapping[str, object] | None = None,
) -> EvalReport:
    metrics: EvalReport = {
        "recall_at_5": _mean_payload_value(payloads, "recall_at_5"),
        "recall_at_10": _mean_payload_value(payloads, "recall_at_10"),
        "mrr": _mean_payload_value(payloads, "mrr"),
        "ndcg_at_10": _mean_payload_value(payloads, "ndcg_at_10"),
        "latency_ms": _mean_payload_value(payloads, "latency_ms"),
        "latency_p95_ms": _latency_percentile(payloads, percentile=0.95),
        "throughput_qps": _throughput_qps(payloads, run=run),
        "context_recall": _mean_optional_payload_value(payloads, "context_recall"),
        "prompt_safety_pass_rate": _mean_bool_payload_value(
            payloads,
            "prompt_safety_pass",
        ),
        "forbidden_leak_count": _sum_payload_int(payloads, "forbidden_leak_count"),
    }
    if _uses_wall_clock_throughput(run):
        metrics["serial_latency_qps"] = _serial_latency_qps(payloads)
    return metrics


def _mean_payload_value(payloads: Sequence[Mapping[str, object]], key: str) -> float:
    if not payloads:
        return 0.0
    return sum(
        finite_metric_value(payload[key], name=key) for payload in payloads
    ) / len(payloads)


def _mean_optional_payload_value(
    payloads: Sequence[Mapping[str, object]],
    key: str,
) -> float:
    values = [payload[key] for payload in payloads if key in payload]
    if not values:
        return 0.0
    return sum(finite_metric_value(value, name=key) for value in values) / len(values)


def _mean_bool_payload_value(
    payloads: Sequence[Mapping[str, object]],
    key: str,
) -> float:
    values = [payload[key] for payload in payloads if isinstance(payload.get(key), bool)]
    if not values:
        return 0.0
    return sum(1.0 for value in values if value is True) / len(values)


def _sum_payload_int(payloads: Sequence[Mapping[str, object]], key: str) -> int:
    total = 0
    for payload in payloads:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
    return total


def _case_has_context_expectations(case: object) -> bool:
    return bool(
        getattr(case, "expected_context_contains", ())
        or getattr(case, "forbidden_context_contains", ())
        or getattr(case, "forbidden_private_identifiers", ())
        or getattr(case, "expected_citation_count_min", 0)
        or getattr(case, "expected_source_count_min", 0)
        or getattr(case, "max_context_chars", None) is not None
        or getattr(case, "max_context_tokens", None) is not None
    )


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


def _throughput_qps(
    payloads: Sequence[Mapping[str, object]],
    *,
    run: Mapping[str, object] | None,
) -> float:
    if payloads and _uses_wall_clock_throughput(run):
        wall_clock_seconds = _positive_float_metadata(run, "wall_clock_seconds")
        if wall_clock_seconds is not None:
            return len(payloads) / wall_clock_seconds
    return _serial_latency_qps(payloads)


def _serial_latency_qps(payloads: Sequence[Mapping[str, object]]) -> float:
    total_latency_ms = sum(
        finite_metric_value(payload["latency_ms"], name="latency_ms")
        for payload in payloads
    )
    if not payloads or total_latency_ms <= 0.0:
        return 0.0
    return len(payloads) * 1000.0 / total_latency_ms


def _uses_wall_clock_throughput(run: Mapping[str, object] | None) -> bool:
    return (
        _int_metadata(run, "max_concurrency") > 1
        and _positive_float_metadata(run, "wall_clock_seconds") is not None
    )


def _int_metadata(run: Mapping[str, object] | None, key: str) -> int:
    if run is None:
        return 0
    value = run.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _positive_float_metadata(
    run: Mapping[str, object] | None,
    key: str,
) -> float | None:
    if run is None:
        return None
    value = run.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number > 0.0:
            return number
    return None
