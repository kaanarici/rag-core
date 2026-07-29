from __future__ import annotations

from collections.abc import Mapping
import math

EvalReport = dict[str, object]
EvalRunMetadata = Mapping[str, object]

METRIC_KEYS = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
    "latency_ms",
    "latency_p95_ms",
    "throughput_qps",
    "duplicate_result_rate",
    "unique_document_count",
    "off_topic_result_rate",
    "suppressed_duplicate_count",
)


def float_value(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return 0.0


def finite_metric_value(value: object, *, name: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    raise ValueError(f"eval metric {name} must be a finite number")
