from __future__ import annotations

from collections.abc import Mapping
import math

from .report_models import EvalReport, EvalThresholds, mapping


def add_quality_gate(
    report: EvalReport,
    targets: Mapping[str, Mapping[str, object]],
    thresholds: EvalThresholds,
) -> None:
    """Attach quality-gate results to ``report`` when thresholds are configured."""
    if thresholds:
        report["quality_gate"] = quality_gate_report(targets, thresholds)


def quality_gate_report(
    targets: Mapping[str, Mapping[str, object]],
    thresholds: EvalThresholds,
) -> EvalReport:
    """Build pass/fail details for metric thresholds across one or more targets."""
    failures: list[EvalReport] = []
    for scope, target in targets.items():
        failure_count = _failure_count(target)
        if failure_count > 0:
            failures.append(
                {
                    "scope": scope,
                    "metric": "search_failures",
                    "actual": failure_count,
                    "operator": "==",
                    "threshold": 0,
                }
            )
        metrics = mapping(target.get("metrics"))
        for metric, threshold in thresholds.items():
            actual = _metric_float(metrics.get(metric))
            if actual is None:
                failures.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "actual": "invalid",
                        "operator": "is",
                        "threshold": "numeric",
                    }
                )
                continue
            minimum = threshold.get("minimum")
            maximum = threshold.get("maximum")
            if minimum is not None and actual < minimum:
                failures.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "actual": actual,
                        "operator": ">=",
                        "threshold": minimum,
                    }
                )
            if maximum is not None and actual > maximum:
                failures.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "actual": actual,
                        "operator": "<=",
                        "threshold": maximum,
                    }
                )
    return {
        "passed": not failures,
        "thresholds": {metric: dict(threshold) for metric, threshold in thresholds.items()},
        "failures": failures,
    }


def eval_exit_code(report: Mapping[str, object]) -> int:
    """Return a process exit code for an eval report with an optional gate."""
    if _has_eval_failures(report):
        return 1
    gate = report.get("quality_gate")
    if isinstance(gate, dict) and gate.get("passed") is False:
        return 1
    return 0


def _has_eval_failures(report: Mapping[str, object]) -> bool:
    failure_count = report.get("failure_count")
    if isinstance(failure_count, int) and not isinstance(failure_count, bool):
        return failure_count > 0
    for key in ("baseline", "reranked"):
        nested = report.get(key)
        if isinstance(nested, dict) and _has_eval_failures(nested):
            return True
    profiles = report.get("profiles")
    if isinstance(profiles, dict):
        return any(
            _has_eval_failures(profile)
            for profile in profiles.values()
            if isinstance(profile, dict)
        )
    return False


def _metric_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _failure_count(report: Mapping[str, object]) -> int:
    count = report.get("failure_count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    return 0
