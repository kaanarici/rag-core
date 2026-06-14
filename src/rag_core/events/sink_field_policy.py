"""Field policy for built-in event sink payloads."""

from __future__ import annotations

from typing import Final

from rag_core.events.event_types import INGEST_SKIPPED_EVENT, SEARCH_PLANNED_EVENT
from rag_core.events.trace_payload_fields import TRACE_ABSENT_LABEL

EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        # Audit correlation. ``actor`` is a human/service identity; ``request_id``
        # and ``ingest_id`` can be guessable (UUID4 is fine, but a gateway
        # might use a short slug). Keep them off OTel by default so they
        # don't bleed into shared monitoring backends. The gateway has the
        # audit log already.
        "actor",
        "content_sha256",
        "corpus_id",
        "corpus_ids",
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
