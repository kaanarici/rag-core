"""Report builders and quality gates for retrieval eval runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_core.evals.runner import EvalResult

from .quality_gates import add_quality_gate, eval_exit_code, quality_gate_report
from .report_formatting import (
    format_eval_comparison_report,
    format_eval_metrics,
    format_eval_profile_comparison_report,
    format_eval_report,
    format_quality_gate,
    format_run_metadata,
)
from .report_models import (
    METRIC_KEYS,
    EvalReport,
    EvalRunMetadata,
    EvalThresholds,
    finite_metric_value,
    mapping,
)


def eval_result_payload(result: EvalResult) -> EvalReport:
    """Return the stable JSON payload for one eval case result."""
    case = result.case
    payload: EvalReport = {
        "case_id": case.case_id,
        "query": case.query,
        "namespace": case.namespace,
        "corpus_ids": list(case.corpus_ids),
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
    """Build the aggregate report for one retrieval eval run."""
    result_payloads = []
    for case_ordinal, result in enumerate(results, start=1):
        payload = eval_result_payload(result)
        payload["case_ordinal"] = case_ordinal
        result_payloads.append(payload)
    metrics = _aggregate_metrics(result_payloads)
    report: EvalReport = {
        "case_count": len(result_payloads),
        "failure_count": sum(1 for payload in result_payloads if payload.get("ok") is False),
        "metrics": metrics,
        "cases": result_payloads,
    }
    if run:
        report["run"] = dict(run)
    return report


def redact_eval_report(report: Mapping[str, object]) -> EvalReport:
    """Return a JSON-safe eval report without raw case/query identifiers."""
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


def eval_comparison_report(
    baseline_results: Sequence[EvalResult],
    reranked_results: Sequence[EvalResult],
    *,
    baseline_run: EvalRunMetadata | None = None,
    reranked_run: EvalRunMetadata | None = None,
) -> EvalReport:
    """Build a baseline-vs-reranked report with metric deltas."""
    _require_matching_case_ids(
        "baseline",
        baseline_results,
        "reranked",
        reranked_results,
    )
    baseline = eval_report(baseline_results, run=baseline_run)
    reranked = eval_report(reranked_results, run=reranked_run)
    return {
        "case_count": baseline.get("case_count"),
        "failure_count": _failure_count(baseline) + _failure_count(reranked),
        "baseline": baseline,
        "reranked": reranked,
        "metric_deltas": metric_delta_payload(
            mapping(baseline.get("metrics")),
            mapping(reranked.get("metrics")),
        ),
    }


def eval_profile_comparison_report(
    profile_results: Mapping[str, Sequence[EvalResult]],
    *,
    profile_runs: Mapping[str, EvalRunMetadata] | None = None,
    baseline_profile: str | None = None,
) -> EvalReport:
    """Build a report comparing named search profiles against a stable baseline."""
    ordered_profile_names = tuple(sorted(profile_results))
    if not ordered_profile_names:
        return {
            "baseline_profile": "",
            "profiles": {},
            "metric_deltas": {},
        }
    if baseline_profile is not None and baseline_profile not in profile_results:
        raise ValueError(
            f"baseline profile {baseline_profile!r} is not present in profile results"
        )
    baseline_name = baseline_profile or ordered_profile_names[0]
    baseline_results = profile_results[baseline_name]
    for profile_name in ordered_profile_names:
        _require_matching_case_ids(
            baseline_name,
            baseline_results,
            profile_name,
            profile_results[profile_name],
        )
    profiles = {
        name: eval_report(
            profile_results[name],
            run=profile_runs.get(name) if profile_runs else None,
        )
        for name in ordered_profile_names
    }
    baseline_metrics = (
        mapping(mapping(profiles.get(baseline_name)).get("metrics"))
        if baseline_name
        else {}
    )
    return {
        "case_count": mapping(profiles.get(baseline_name)).get("case_count"),
        "failure_count": sum(
            _failure_count(mapping(profile)) for profile in profiles.values()
        ),
        "baseline_profile": baseline_name,
        "profiles": profiles,
        "metric_deltas": {
            name: metric_delta_payload(
                baseline_metrics,
                mapping(mapping(payload).get("metrics")),
            )
            for name, payload in profiles.items()
        },
    }


def metric_delta_payload(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, float]:
    """Return candidate-minus-baseline metric deltas."""
    return {
        key: finite_metric_value(candidate.get(key), name=key)
        - finite_metric_value(baseline.get(key), name=key)
        for key in METRIC_KEYS
    }


def _aggregate_metrics(payloads: Sequence[Mapping[str, object]]) -> EvalReport:
    metrics: EvalReport = {
        "recall_at_5": _mean_payload_value(payloads, "recall_at_5"),
        "recall_at_10": _mean_payload_value(payloads, "recall_at_10"),
        "mrr": _mean_payload_value(payloads, "mrr"),
        "ndcg_at_10": _mean_payload_value(payloads, "ndcg_at_10"),
        "latency_ms": _mean_payload_value(payloads, "latency_ms"),
        "latency_p95_ms": _latency_percentile(payloads, percentile=0.95),
        "throughput_qps": _throughput_qps(payloads),
    }
    return metrics


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


def _require_matching_case_ids(
    left_label: str,
    left_results: Sequence[EvalResult],
    right_label: str,
    right_results: Sequence[EvalResult],
) -> None:
    left_cases = _comparison_cases(left_results)
    right_cases = _comparison_cases(right_results)
    left_ids = tuple(case["id"] for case in left_cases)
    right_ids = tuple(case["id"] for case in right_cases)
    if left_ids == right_ids:
        _require_matching_case_shapes(
            left_label=left_label,
            left_cases=left_cases,
            right_label=right_label,
            right_cases=right_cases,
        )
        return
    if len(left_ids) != len(right_ids):
        raise ValueError(
            "eval comparison requires the same ordered case_ids; "
            f"{left_label} has {len(left_ids)} case(s), "
            f"{right_label} has {len(right_ids)}"
        )
    for index, (left_id, right_id) in enumerate(
        zip(left_ids, right_ids, strict=True),
        start=1,
    ):
        if left_id != right_id:
            left_display_id = left_cases[index - 1]["display_id"]
            right_display_id = right_cases[index - 1]["display_id"]
            raise ValueError(
                "eval comparison requires the same ordered case_ids; "
                f"case {index} is {left_label}={left_display_id!r}, "
                f"{right_label}={right_display_id!r}"
            )
    _require_matching_case_shapes(
        left_label=left_label,
        left_cases=left_cases,
        right_label=right_label,
        right_cases=right_cases,
    )


def _require_matching_case_shapes(
    *,
    left_label: str,
    left_cases: Sequence[Mapping[str, str]],
    right_label: str,
    right_cases: Sequence[Mapping[str, str]],
) -> None:
    for index, (left_case, right_case) in enumerate(
        zip(left_cases, right_cases, strict=True),
        start=1,
    ):
        if left_case["shape"] != right_case["shape"]:
            _raise_case_shape_mismatch(
                index=index,
                display_id=left_case["display_id"],
                left_label=left_label,
                right_label=right_label,
            )


def _raise_case_shape_mismatch(
    *,
    index: int,
    display_id: str,
    left_label: str,
    right_label: str,
) -> None:
    raise ValueError(
        "eval comparison requires matching case payloads; "
        f"case {index} differs for case_id {display_id!r} "
        f"between {left_label} and {right_label}"
    )


def _comparison_cases(results: Sequence[EvalResult]) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "id": _case_key(result),
            "display_id": _case_display_id(result, index),
            "shape": _case_shape(result),
        }
        for index, result in enumerate(results, start=1)
    )


def _case_key(result: EvalResult) -> str:
    case = result.case
    if case.case_id is not None:
        return case.case_id
    return _case_shape(result)


def _case_display_id(result: EvalResult, index: int) -> str:
    case_id = result.case.case_id
    if case_id is not None:
        return case_id
    return f"<implicit case {index}>"


def _case_shape(result: EvalResult) -> str:
    case = result.case
    expected_grades = (
        tuple(sorted(case.expected_grades.items()))
        if case.expected_grades is not None
        else ()
    )
    return repr(
        (
            case.query,
            case.namespace,
            tuple(case.corpus_ids),
            tuple(case.expected_chunk_ids),
            expected_grades,
        )
    )


def _failure_count(report: Mapping[str, object]) -> int:
    count = report.get("failure_count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    return 0


def _redacted_case_payload(value: object) -> EvalReport:
    if not isinstance(value, Mapping):
        return {}
    payload: EvalReport = {}
    case_ordinal = value.get("case_ordinal")
    if isinstance(case_ordinal, int) and not isinstance(case_ordinal, bool):
        payload["case_ordinal"] = case_ordinal
        payload["case_label"] = f"case-{case_ordinal}"
    for key in (
        "ok",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "latency_ms",
        "error_type",
    ):
        if key in value:
            payload[key] = value[key]
    payload["corpus_count"] = _sequence_count(value.get("corpus_ids"))
    payload["expected_chunk_count"] = _sequence_count(value.get("expected_chunk_ids"))
    payload["retrieved_count"] = _sequence_count(value.get("retrieved_ids"))
    if "expected_grades" in value:
        grades = value.get("expected_grades")
        payload["expected_grade_count"] = len(grades) if isinstance(grades, Mapping) else 0
    return payload


def _sequence_count(value: object) -> int:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return len(value)
    return 0


__all__ = [
    "EvalReport",
    "EvalRunMetadata",
    "EvalThresholds",
    "add_quality_gate",
    "eval_comparison_report",
    "eval_exit_code",
    "eval_profile_comparison_report",
    "eval_report",
    "eval_result_payload",
    "format_eval_comparison_report",
    "format_eval_metrics",
    "format_eval_profile_comparison_report",
    "format_eval_report",
    "format_quality_gate",
    "format_run_metadata",
    "metric_delta_payload",
    "quality_gate_report",
    "redact_eval_report",
]
