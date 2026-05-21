from __future__ import annotations

from collections.abc import Mapping

from .report_models import float_value, mapping


def format_eval_report(report: Mapping[str, object]) -> list[str]:
    """Return human-readable CLI lines for a single eval report."""
    metrics = mapping(report.get("metrics"))
    return [
        *format_run_metadata("Run", report.get("run")),
        f"Cases: {report.get('case_count')}",
        *format_failure_count(report),
        f"Mean: {format_eval_metrics(metrics)}",
        *format_quality_gate(report),
    ]


def format_eval_comparison_report(report: Mapping[str, object]) -> list[str]:
    """Return human-readable CLI lines for a baseline-vs-reranked report."""
    baseline = mapping(report.get("baseline"))
    reranked = mapping(report.get("reranked"))
    deltas = mapping(report.get("metric_deltas"))
    return [
        *format_run_metadata("Baseline Run", baseline.get("run")),
        *format_run_metadata("Reranked Run", reranked.get("run")),
        f"Cases: {baseline.get('case_count')}",
        *format_failure_count(baseline, label="Baseline search failures"),
        *format_failure_count(reranked, label="Reranked search failures"),
        f"Baseline Mean: {format_eval_metrics(mapping(baseline.get('metrics')))}",
        f"Reranked Mean: {format_eval_metrics(mapping(reranked.get('metrics')))}",
        f"Delta: {format_eval_metrics(deltas)}",
        *format_quality_gate(report),
    ]


def format_eval_profile_comparison_report(report: Mapping[str, object]) -> list[str]:
    """Return human-readable CLI lines for a search-profile comparison report."""
    profiles = mapping(report.get("profiles"))
    deltas = mapping(report.get("metric_deltas"))
    baseline_profile = str(report.get("baseline_profile") or "")
    baseline = mapping(profiles.get(baseline_profile)) if baseline_profile else {}
    lines = [
        f"Cases: {baseline.get('case_count')}",
        f"Baseline Profile: {baseline_profile}",
    ]
    for name, raw_profile in profiles.items():
        profile = mapping(raw_profile)
        metrics = mapping(profile.get("metrics"))
        delta = mapping(deltas.get(name))
        lines.append(f"{name} Mean: {format_eval_metrics(metrics)}")
        lines.extend(format_failure_count(profile, label=f"{name} search failures"))
        if name != baseline_profile:
            lines.append(f"{name} Delta: {format_eval_metrics(delta)}")
    lines.extend(format_quality_gate(report))
    return lines


def format_failure_count(
    report: Mapping[str, object],
    *,
    label: str = "Search failures",
) -> list[str]:
    count = report.get("failure_count")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return [f"{label}: {count}"]
    return []


def format_quality_gate(report: Mapping[str, object]) -> list[str]:
    """Return human-readable quality-gate lines for an eval report."""
    gate = report.get("quality_gate")
    if not isinstance(gate, dict):
        return []
    failures = gate.get("failures")
    if gate.get("passed") is True:
        return ["Gate: pass"]
    lines = ["Gate: fail"]
    if not isinstance(failures, list):
        return lines
    for raw_failure in failures:
        failure = mapping(raw_failure)
        scope = str(failure.get("scope") or "eval")
        metric = str(failure.get("metric") or "")
        operator = str(failure.get("operator") or "")
        lines.append(
            f"- {scope} {metric}: "
            f"{_format_gate_value(failure.get('actual'))} {operator} "
            f"{_format_gate_value(failure.get('threshold'))}"
        )
    return lines


def format_run_metadata(label: str, run: object) -> list[str]:
    """Return a compact CLI line describing the eval run shape."""
    if not isinstance(run, dict):
        return []
    parts: list[str] = []
    for key in (
        "mode",
        "rag_core_version",
        "vector_store",
        "qdrant_collection",
        "turbopuffer_namespace",
        "search_profile",
        "query_plan_preset",
        "rerank",
        "reranker_provider",
        "embedding_model",
        "embedding_dimensions",
    ):
        value = run.get(key)
        if value is None:
            continue
        parts.append(f"{key}={_format_run_value(value)}")
    return [f"{label}: {' '.join(parts)}"] if parts else []


def format_eval_metrics(metrics: Mapping[str, object]) -> str:
    """Return the compact metric string used by CLI eval output."""
    return (
        f"recall@5={float_value(metrics.get('recall_at_5')):.3f} "
        f"recall@10={float_value(metrics.get('recall_at_10')):.3f} "
        f"mrr={float_value(metrics.get('mrr')):.3f} "
        f"ndcg@10={float_value(metrics.get('ndcg_at_10')):.3f} "
        f"latency_ms={float_value(metrics.get('latency_ms')):.3f} "
        f"latency_p95_ms={float_value(metrics.get('latency_p95_ms')):.3f} "
        f"throughput_qps={float_value(metrics.get('throughput_qps')):.3f}"
    )


def _format_run_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _format_gate_value(value: object) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float_value(value):.3f}"
    return str(value)
