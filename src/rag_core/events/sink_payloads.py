"""Sink payload rendering and the sensitive/safe field policy that shapes it."""

from __future__ import annotations

from typing import Final
from rag_core.events.event_types import INGEST_SKIPPED_EVENT, SEARCH_PLANNED_EVENT
from rag_core.events.trace_payload_fields import TRACE_ABSENT_LABEL
from dataclasses import asdict
import math
from typing import Any, TypeAlias
from rag_core.events.event_types import SEARCH_COMPLETED_EVENT
from rag_core.events.trace_payload_fields import (
    safe_trace_label,
    safe_trace_label_sequence,
)
from rag_core.events.trace_summary_models import safe_search_id
from rag_core.events.types import Event


EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        # Audit correlation. ``actor`` is a human/service identity; ``request_id``
        # and ``ingest_id`` can be guessable (UUID4 is fine, but a gateway
        # might use a short slug). Keep them off OTel by default so they
        # don't bleed into shared monitoring backends. The gateway has the
        # audit log already.
        "actor",
        "content_sha256",
        "collection",
        "collections",
        "document_id",
        "document_key",
        "error",
        "filename",
        "ingest_id",
        "message",
        "namespace",
        "ocr_page_indices",
        "quality_details",
        "redacted_url",
        "request_id",
        "returned_document_ids",
        "search_id",
    }
)

EVENT_SINK_SAFE_LABEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "boost",
        "error_type",
        "fallback_reason",
        "fusion",
        "metadata_filter",
        "model",
        "parser",
        "plan_rerank",
        "provider",
        "reason",
        "role",
        "search_profile",
        "truncation_reason",
    }
)

EVENT_SINK_SAFE_STAGE_LABEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "fuse_stage",
        "rerank_stage",
        "retrieve_stage",
        "stage",
        "stage_name",
    }
)

EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {"postprocesses", "query_transforms"}
)

EVENT_SINK_SENSITIVE_LOG_FIELDS: Final[frozenset[str]] = (
    EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS
    | frozenset(
        {
            "boost",
            "metadata_filter",
            "reason",
        }
    )
)

EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS: Final[dict[tuple[str, str], frozenset[str]]] = {
    (INGEST_SKIPPED_EVENT, "reason"): frozenset({"content_unchanged"}),
    (SEARCH_PLANNED_EVENT, "boost"): frozenset(
        {TRACE_ABSENT_LABEL, "linear_decay", "exp_decay", "gauss_decay", "raw"}
    ),
    (SEARCH_PLANNED_EVENT, "metadata_filter"): frozenset(
        {TRACE_ABSENT_LABEL, "Term", "In", "Range", "Geo", "And", "Or", "Not"}
    ),
}


_OTEL_PRIMITIVE_TYPES = (str, bool, int, float)

OtelAttributeValue: TypeAlias = (
    str | bool | int | float | list[str | bool | int | float]
)

_AUDIT_CORRELATION_FIELDS = frozenset({"actor", "ingest_id", "request_id", "search_id"})

_TIMESTAMP_FIELDS = frozenset({"emitted_at_ns", "wall_clock_ns"})

def event_to_dict(event: Event) -> dict[str, Any]:
    payload = asdict(event)
    if "collections" in payload and isinstance(payload["collections"], tuple):
        payload["collections"] = list(payload["collections"])
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
        if key in _AUDIT_CORRELATION_FIELDS and safe_search_id(value) == "":
            continue
        # ``emitted_at_ns``/``wall_clock_ns`` are stamped by ``emit_event``;
        # when an event reaches the OTel sink directly (without going through
        # the helper) the value is the dataclass default ``0`` and is not
        # useful as an attribute. Hide rather than ship a zero timestamp.
        if key in _TIMESTAMP_FIELDS and value == 0:
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
    # ``emitted_at_ns``/``wall_clock_ns`` default to ``0``; hide the default so
    # an event constructed without going through ``emit_event`` (e.g. a sink
    # receiving the event directly in a test) doesn't bloat the log line. A
    # real stamped timestamp is far larger than 0 and remains visible.
    if key in _TIMESTAMP_FIELDS and value == 0:
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
    if key in _AUDIT_CORRELATION_FIELDS:
        return safe_search_id(value) != ""
    return _log_field_is_visible(event_type=event_type, key=key, value=value)

def _sanitize_visible_value(key: str, value: Any) -> Any:
    if key in _AUDIT_CORRELATION_FIELDS:
        # ``actor``/``request_id``/``ingest_id``/``search_id`` are caller-
        # supplied opaque tokens. Same sanitization as ``search_id`` (label
        # whitelist) so a malicious caller cannot inject control characters
        # or newlines into a log line.
        return safe_search_id(value)
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
