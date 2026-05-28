from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.evals import eval_exit_code, redact_eval_report
from rag_core.evals.report_models import EvalReport, EvalThresholds
from rag_core.local_eval_runner import LocalEvalCore, LocalEvalCoreFactory, run_local_eval

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


async def run_local_eval_command(
    args: argparse.Namespace,
    *,
    event_sink: "EventSink | None",
) -> int:
    core_factory: LocalEvalCoreFactory | None = None
    if event_sink is not None:
        from rag_core.demo import build_demo_core
        from rag_core.local_search_models import DEFAULT_LOCAL_SEARCH_COLLECTION

        def core_factory() -> LocalEvalCore:
            return build_demo_core(
                collection=DEFAULT_LOCAL_SEARCH_COLLECTION,
                event_sink=event_sink,
            )

    result = await run_local_eval(
        path=Path(args.path),
        cases_path=Path(args.cases),
        max_files=args.max_files,
        search_profile_name=args.search_profile,
        thresholds=_thresholds_from_args(args),
        core_factory=core_factory,
    )
    payload = redact_eval_report(result.report)
    _emit_local_eval(payload, as_json=args.json)
    return eval_exit_code(result.report)


def _emit_local_eval(payload: EvalReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    run = _mapping(payload.get("run"))
    metrics = _mapping(payload.get("metrics"))
    print(f"Indexed: {_payload_int(run, 'indexed_count')} files")
    print(
        "Skipped: "
        f"{_payload_int(run, 'skipped_count')} files "
        f"(unsupported={_payload_int(run, 'skipped_unsupported_count')}, "
        f"empty={_payload_int(run, 'skipped_empty_count')}, "
        f"failed={_payload_int(run, 'skipped_failed_count')})"
    )
    if run.get("truncated") is True:
        print("Truncated: yes; rerun with --max-files to include more supported files")
    print(f"Corpus: {run.get('namespace')}/{run.get('corpus_id')}")
    print(f"Cases: {_payload_int(payload, 'case_count')}")
    print(
        "Metrics: "
        f"recall@5={_payload_float(metrics, 'recall_at_5'):.3f} "
        f"mrr={_payload_float(metrics, 'mrr'):.3f} "
        f"latency_p95_ms={_payload_float(metrics, 'latency_p95_ms'):.1f}"
    )
    gate = payload.get("quality_gate")
    if isinstance(gate, dict):
        print(f"Quality gate: {'passed' if gate.get('passed') is True else 'failed'}")


def _thresholds_from_args(args: argparse.Namespace) -> EvalThresholds:
    thresholds: dict[str, dict[str, float]] = {}
    _add_min_threshold(
        thresholds,
        "recall_at_5",
        _unit_interval_value(args.min_recall_at_5, flag="--min-recall-at-5"),
    )
    _add_min_threshold(
        thresholds,
        "mrr",
        _unit_interval_value(args.min_mrr, flag="--min-mrr"),
    )
    _add_max_threshold(
        thresholds,
        "latency_p95_ms",
        _positive_value(args.max_latency_p95_ms, flag="--max-latency-p95-ms"),
    )
    return thresholds


def _add_min_threshold(
    thresholds: dict[str, dict[str, float]],
    metric: str,
    value: float | None,
) -> None:
    if value is not None:
        thresholds.setdefault(metric, {})["minimum"] = value


def _add_max_threshold(
    thresholds: dict[str, dict[str, float]],
    metric: str,
    value: float | None,
) -> None:
    if value is not None:
        thresholds.setdefault(metric, {})["maximum"] = value


def _unit_interval_value(value: float | None, *, flag: str) -> float | None:
    if value is None:
        return None
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{flag} must be between 0 and 1")
    return value


def _positive_value(value: float | None, *, flag: str) -> float | None:
    if value is None:
        return None
    if value <= 0.0:
        raise ValueError(f"{flag} must be positive")
    return value


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _payload_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


__all__ = ["run_local_eval_command"]
