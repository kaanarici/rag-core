"""Payload shaping and redaction for built-in event sinks."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, TypeAlias

from rag_core.events.event_types import SEARCH_COMPLETED_EVENT
from rag_core.events.sink_field_policy import (
    EVENT_SINK_SAFE_LABEL_FIELDS,
    EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS,
    EVENT_SINK_SAFE_STAGE_LABEL_FIELDS,
    EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS,
    EVENT_SINK_SENSITIVE_LOG_FIELDS,
    EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS,
)
from rag_core.events.trace_payload_fields import (
    safe_trace_label,
    safe_trace_label_sequence,
)
from rag_core.events.trace_summary_models import safe_search_id
from rag_core.events.types import Event

_OTEL_PRIMITIVE_TYPES = (str, bool, int, float)
OtelAttributeValue: TypeAlias = (
    str | bool | int | float | list[str | bool | int | float]
)


def event_to_dict(event: Event) -> dict[str, Any]:
    payload = asdict(event)
    if "corpus_ids" in payload and isinstance(payload["corpus_ids"], tuple):
        payload["corpus_ids"] = list(payload["corpus_ids"])
    return payload


def event_to_jsonl_dict(event: Event) -> dict[str, Any]:
    fields = event_to_dict(event)
    event_type = str(fields.get("event_type", ""))
    return {
        key: _sanitize_visible_value(key, value)
        for key, value in fields.items()
        if _jsonl_field_is_visible(event_type=event_type, key=key, value=value)
    }


def summarize_event(event: Event) -> str:
    fields = event_to_dict(event)
    event_type = str(fields.pop("event_type", ""))
    visible_fields = (
        (key, _sanitize_visible_value(key, value))
        for key, value in fields.items()
        if _log_field_is_visible(event_type=event_type, key=key, value=value)
    )
    summary = ", ".join(f"{key}={value!r}" for key, value in visible_fields)
    if not summary:
        return "fields=omitted"
    return summary


def event_to_otel_attributes(
    event: Event,
    *,
    include_sensitive_attributes: bool = False,
) -> dict[str, OtelAttributeValue]:
    attributes: dict[str, OtelAttributeValue] = {}
    for key, value in event_to_dict(event).items():
        if key == "event_type":
            continue
        if key == "search_id" and _safe_search_id(value) == "":
            continue
        if (
            not include_sensitive_attributes
            and key in EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS
        ):
            continue
        normalized = _to_otel_attribute_value(_sanitize_visible_value(key, value))
        if normalized is not None:
            attributes[f"rag_core.{key}"] = normalized
    return attributes


def _to_otel_attribute_value(value: Any) -> OtelAttributeValue | None:
    if value is None:
        return None
    if isinstance(value, _OTEL_PRIMITIVE_TYPES):
        return value
    if isinstance(value, (list, tuple)):
        normalized = [
            item if isinstance(item, _OTEL_PRIMITIVE_TYPES) else str(item)
            for item in value
            if item is not None
        ]
        if not normalized:
            return None
        kinds = {_otel_primitive_kind(item) for item in normalized}
        if len(kinds) == 1:
            return normalized
        return [str(item) for item in normalized]
    return str(value)


def _empty_summary_value(value: Any) -> bool:
    return value is None or value == "" or value == () or value == [] or value == {}


def _log_field_is_visible(*, event_type: str, key: str, value: Any) -> bool:
    if event_type == SEARCH_COMPLETED_EVENT and key == "succeeded" and value is True:
        return False
    if _empty_summary_value(value):
        return False
    if key not in EVENT_SINK_SENSITIVE_LOG_FIELDS:
        return True
    if not isinstance(value, str):
        return False
    return value in EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS.get(
        (event_type, key),
        frozenset(),
    )


def _jsonl_field_is_visible(*, event_type: str, key: str, value: Any) -> bool:
    if key == "event_type":
        return True
    if key == "search_id":
        return _safe_search_id(value) != ""
    if event_type == SEARCH_COMPLETED_EVENT and key == "succeeded" and value is True:
        return False
    return _log_field_is_visible(event_type=event_type, key=key, value=value)


def _sanitize_visible_value(key: str, value: Any) -> Any:
    if key == "search_id":
        return _safe_search_id(value)
    if key in EVENT_SINK_SAFE_STAGE_LABEL_FIELDS:
        return safe_trace_label(value, stage=True)
    if key in EVENT_SINK_SAFE_LABEL_FIELDS:
        return safe_trace_label(value, stage=False)
    if key in EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS:
        return safe_trace_label_sequence(value, stage=True)
    if key == "channels":
        return safe_trace_label_sequence(value, stage=False)
    return _json_safe_value(value)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return value


def _otel_primitive_kind(value: str | bool | int | float) -> type[object]:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str


def _safe_search_id(value: object) -> str:
    return safe_search_id(value)
