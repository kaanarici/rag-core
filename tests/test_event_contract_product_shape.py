"""Single-owner invariants for the event-type, sink-policy, and sink-provider contracts.

These check the contracts that matter -- the canonical event-type labels, the
sink field-policy sets, the retrieval-hit export field names and their public
docs/behavior honesty, and the sink-provider order -- via runtime values,
module-level ownership, and the import graph, so they survive file merges,
renames, and reformatting. (Previously these were asserted by reading
hand-pinned source-file lists and scanning for forbidden literal substrings,
which froze the file layout.) The doc/behavior honesty asserts on ``export.py``
and ``traces.mdx`` are preserved verbatim because they guard the public export
contract, not internal module layout.
"""

from __future__ import annotations

from pathlib import Path

from rag_core.events.event_types import (
    CHUNK_PRODUCED_EVENT,
    CONTEXTUALIZE_COMPLETED_EVENT,
    CONTEXTUALIZE_STARTED_EVENT,
    EMBED_COMPLETED_EVENT,
    EMBED_REQUESTED_EVENT,
    FETCH_COMPLETED_EVENT,
    FETCH_FAILED_EVENT,
    FETCH_STARTED_EVENT,
    INDEX_DELETED_EVENT,
    INDEX_UPSERTED_EVENT,
    INGEST_BATCH_COMPLETED_EVENT,
    INGEST_BATCH_FAILED_EVENT,
    INGEST_BATCH_PROGRESS_EVENT,
    INGEST_BATCH_STARTED_EVENT,
    INGEST_COMPLETED_EVENT,
    INGEST_SKIPPED_EVENT,
    INGEST_STARTED_EVENT,
    OCR_APPLIED_EVENT,
    PARSE_COMPLETED_EVENT,
    RERANK_APPLIED_EVENT,
    SEARCH_COMPLETED_EVENT,
    SEARCH_PLANNED_EVENT,
    SEARCH_STAGE_COMPLETED_EVENT,
    SEARCH_STARTED_EVENT,
    SIDECAR_APPLIED_EVENT,
    STAGE_ERROR_EVENT,
)
from rag_core.events.export import (
    RETRIEVAL_HIT_CONTENT_FIELD,
    RETRIEVAL_HIT_CORE_FIELDS,
    RETRIEVAL_HIT_OPTIONAL_FIELDS,
)
from rag_core.events.sink_payloads import (
    EVENT_SINK_SAFE_LABEL_FIELDS,
    EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS,
    EVENT_SINK_SAFE_STAGE_LABEL_FIELDS,
    EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS,
    EVENT_SINK_SENSITIVE_LOG_FIELDS,
    EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS,
)
from rag_core.events.sinks import (
    BUFFER_EVENT_SINK_PROVIDER,
    DEFAULT_EVENT_SINK_PROVIDER,
    EVENT_SINK_PROVIDER_ORDER,
    JSONL_EVENT_SINK_PROVIDER,
    LOGGING_EVENT_SINK_PROVIDER,
    MULTI_EVENT_SINK_PROVIDER,
    NOOP_EVENT_SINK_PROVIDER,
    OPENTELEMETRY_EVENT_SINK_PROVIDER,
)

from tests.support.source_graph import (
    defining_modules,
    modules_importing,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_EVENT_ROOTS = (
    "src/rag_core/events",
    "src/rag_core/search",
    "src/rag_core/_engine",
    "src/rag_core/cli",
)


def test_event_type_labels_have_single_events_owner() -> None:
    expected = {
        "INGEST_STARTED_EVENT": INGEST_STARTED_EVENT,
        "INGEST_SKIPPED_EVENT": INGEST_SKIPPED_EVENT,
        "INGEST_BATCH_STARTED_EVENT": INGEST_BATCH_STARTED_EVENT,
        "INGEST_BATCH_PROGRESS_EVENT": INGEST_BATCH_PROGRESS_EVENT,
        "INGEST_BATCH_COMPLETED_EVENT": INGEST_BATCH_COMPLETED_EVENT,
        "INGEST_BATCH_FAILED_EVENT": INGEST_BATCH_FAILED_EVENT,
        "INGEST_COMPLETED_EVENT": INGEST_COMPLETED_EVENT,
        "FETCH_STARTED_EVENT": FETCH_STARTED_EVENT,
        "FETCH_COMPLETED_EVENT": FETCH_COMPLETED_EVENT,
        "FETCH_FAILED_EVENT": FETCH_FAILED_EVENT,
        "PARSE_COMPLETED_EVENT": PARSE_COMPLETED_EVENT,
        "OCR_APPLIED_EVENT": OCR_APPLIED_EVENT,
        "CHUNK_PRODUCED_EVENT": CHUNK_PRODUCED_EVENT,
        "CONTEXTUALIZE_STARTED_EVENT": CONTEXTUALIZE_STARTED_EVENT,
        "CONTEXTUALIZE_COMPLETED_EVENT": CONTEXTUALIZE_COMPLETED_EVENT,
        "EMBED_REQUESTED_EVENT": EMBED_REQUESTED_EVENT,
        "EMBED_COMPLETED_EVENT": EMBED_COMPLETED_EVENT,
        "INDEX_UPSERTED_EVENT": INDEX_UPSERTED_EVENT,
        "INDEX_DELETED_EVENT": INDEX_DELETED_EVENT,
        "SEARCH_STARTED_EVENT": SEARCH_STARTED_EVENT,
        "SEARCH_PLANNED_EVENT": SEARCH_PLANNED_EVENT,
        "SEARCH_STAGE_COMPLETED_EVENT": SEARCH_STAGE_COMPLETED_EVENT,
        "SEARCH_COMPLETED_EVENT": SEARCH_COMPLETED_EVENT,
        "RERANK_APPLIED_EVENT": RERANK_APPLIED_EVENT,
        "SIDECAR_APPLIED_EVENT": SIDECAR_APPLIED_EVENT,
        "STAGE_ERROR_EVENT": STAGE_ERROR_EVENT,
    }

    assert expected == {
        "INGEST_STARTED_EVENT": "ingest.started",
        "INGEST_SKIPPED_EVENT": "ingest.skipped",
        "INGEST_BATCH_STARTED_EVENT": "ingest.batch.started",
        "INGEST_BATCH_PROGRESS_EVENT": "ingest.batch.progress",
        "INGEST_BATCH_COMPLETED_EVENT": "ingest.batch.completed",
        "INGEST_BATCH_FAILED_EVENT": "ingest.batch.failed",
        "INGEST_COMPLETED_EVENT": "ingest.completed",
        "FETCH_STARTED_EVENT": "fetch.started",
        "FETCH_COMPLETED_EVENT": "fetch.completed",
        "FETCH_FAILED_EVENT": "fetch.failed",
        "PARSE_COMPLETED_EVENT": "parse.completed",
        "OCR_APPLIED_EVENT": "ocr.applied",
        "CHUNK_PRODUCED_EVENT": "chunk.produced",
        "CONTEXTUALIZE_STARTED_EVENT": "contextualize.started",
        "CONTEXTUALIZE_COMPLETED_EVENT": "contextualize.completed",
        "EMBED_REQUESTED_EVENT": "embed.requested",
        "EMBED_COMPLETED_EVENT": "embed.completed",
        "INDEX_UPSERTED_EVENT": "index.upserted",
        "INDEX_DELETED_EVENT": "index.deleted",
        "SEARCH_STARTED_EVENT": "search.started",
        "SEARCH_PLANNED_EVENT": "search.planned",
        "SEARCH_STAGE_COMPLETED_EVENT": "search.stage.completed",
        "SEARCH_COMPLETED_EVENT": "search.completed",
        "RERANK_APPLIED_EVENT": "rerank.applied",
        "SIDECAR_APPLIED_EVENT": "sidecar.applied",
        "STAGE_ERROR_EVENT": "stage.error",
    }

    # Exactly one module binds each event-type constant, so no other event module
    # can re-derive the literal under its own name.
    for symbol in expected:
        assert defining_modules(*_EVENT_ROOTS, name=symbol) == {
            "rag_core.events.event_types"
        }


def test_event_sink_field_policy_has_single_owner() -> None:
    assert EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS == frozenset(
        {
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
    assert {"boost", "metadata_filter", "reason"}.issubset(
        EVENT_SINK_SENSITIVE_LOG_FIELDS
    )
    assert EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS == frozenset(
        {"postprocesses", "query_transforms"}
    )
    assert {"provider", "model", "parser", "role"}.issubset(
        EVENT_SINK_SAFE_LABEL_FIELDS
    )
    assert {"stage", "stage_name", "retrieve_stage"}.issubset(
        EVENT_SINK_SAFE_STAGE_LABEL_FIELDS
    )
    assert EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS[
        (INGEST_SKIPPED_EVENT, "reason")
    ] == frozenset({"content_unchanged"})

    # Each policy set has exactly one owner: the sink payload renderer module that
    # consumes them, with no private ``_SAFE_*``/``_SENSITIVE_*`` copy elsewhere.
    for name in (
        "EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS",
        "EVENT_SINK_SAFE_LABEL_FIELDS",
        "EVENT_SINK_SAFE_STAGE_LABEL_FIELDS",
        "EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS",
        "EVENT_SINK_SENSITIVE_LOG_FIELDS",
        "EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS",
    ):
        assert defining_modules(*_EVENT_ROOTS, name=name) == {
            "rag_core.events.sink_payloads"
        }


def test_retrieval_hit_export_fields_have_single_owner() -> None:
    assert RETRIEVAL_HIT_CORE_FIELDS == ("id", "content", "score")
    assert RETRIEVAL_HIT_CONTENT_FIELD == "content"
    assert "metadata" in RETRIEVAL_HIT_OPTIONAL_FIELDS

    for name in (
        "RETRIEVAL_HIT_ID_FIELD",
        "RETRIEVAL_HIT_CONTENT_FIELD",
        "RETRIEVAL_HIT_SCORE_FIELD",
        "RETRIEVAL_HIT_DOCUMENT_ID_FIELD",
        "RETRIEVAL_HIT_DOCUMENT_KEY_FIELD",
        "RETRIEVAL_HIT_METADATA_FIELD",
    ):
        assert defining_modules(*_EVENT_ROOTS, name=name) == {
            "rag_core.events.export"
        }

    # Public behavior/doc honesty for the export contract: the export renames the
    # result body to a stable ``content`` field and never emits the raw internal
    # field names verbatim. These guard the contract, not the file layout.
    export = (REPO_ROOT / "src/rag_core/events/export.py").read_text(encoding="utf-8")
    assert "``id``, ``content``, ``score``" in export
    assert "``id``, ``text``, ``score``" not in export
    assert '"content": result.text' not in export
    assert 'document["metadata"]' not in export

    docs = (REPO_ROOT / "docs-site/content/docs/traces.mdx").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(docs.split())
    assert "to_retrieval_hits" in docs
    assert "renames the result body from `.text` to a `content` field" in normalized
    assert "keeps `id` and `score`" in normalized
    assert "emit the same\nlogical fields" not in docs


def test_event_sink_provider_defaults_have_single_sink_owner() -> None:
    assert DEFAULT_EVENT_SINK_PROVIDER == "none"
    assert EVENT_SINK_PROVIDER_ORDER == (
        NOOP_EVENT_SINK_PROVIDER,
        LOGGING_EVENT_SINK_PROVIDER,
        JSONL_EVENT_SINK_PROVIDER,
        BUFFER_EVENT_SINK_PROVIDER,
        MULTI_EVENT_SINK_PROVIDER,
        OPENTELEMETRY_EVENT_SINK_PROVIDER,
    )

    # The provider names, default, and canonical order all live in one sink
    # module; the diagnostics/doctor surfaces import the shared order instead of
    # re-listing the provider tuple.
    for name in (
        "NOOP_EVENT_SINK_PROVIDER",
        "LOGGING_EVENT_SINK_PROVIDER",
        "JSONL_EVENT_SINK_PROVIDER",
        "BUFFER_EVENT_SINK_PROVIDER",
        "MULTI_EVENT_SINK_PROVIDER",
        "OPENTELEMETRY_EVENT_SINK_PROVIDER",
        "DEFAULT_EVENT_SINK_PROVIDER",
        "EVENT_SINK_PROVIDER_ORDER",
    ):
        assert defining_modules(*_EVENT_ROOTS, name=name) == {"rag_core.events.sinks"}

    order_consumers = modules_importing(
        "src/rag_core/search",
        "src/rag_core/cli",
        predicate=lambda module: module
        == "rag_core.events.sinks.EVENT_SINK_PROVIDER_ORDER",
    )
    assert {
        "rag_core.search.providers.event_sink_category_diagnostics",
        "rag_core.cli.doctor_output",
    } <= set(order_consumers)

    diagnostics_consumers = modules_importing(
        "src/rag_core/search",
        predicate=lambda module: module
        == "rag_core.search.providers.event_sink_category_diagnostics"
        ".describe_event_sink_provider_diagnostics",
    )
    assert (
        "rag_core.search.providers.provider_diagnostics"
        in diagnostics_consumers
    )
