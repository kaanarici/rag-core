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
from rag_core.events.retrieval_hit_fields import (
    RETRIEVAL_HIT_CONTENT_FIELD,
    RETRIEVAL_HIT_CORE_FIELDS,
    RETRIEVAL_HIT_OPTIONAL_FIELDS,
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
from rag_core.events.sink_field_policy import (
    EVENT_SINK_SAFE_LABEL_FIELDS,
    EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS,
    EVENT_SINK_SAFE_STAGE_LABEL_FIELDS,
    EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS,
    EVENT_SINK_SENSITIVE_LOG_FIELDS,
    EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS,
)

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_event_type_labels_have_single_events_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/events/event_types.py",
            "src/rag_core/events/document_events.py",
            "src/rag_core/events/ingest_events.py",
            "src/rag_core/events/search_events.py",
            "src/rag_core/events/embedding_trace_summary.py",
            "src/rag_core/events/trace_payloads.py",
            "src/rag_core/events/sink_payloads.py",
        )
    }

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

    owner = sources["src/rag_core/events/event_types.py"]
    for symbol, value in expected.items():
        assert symbol in owner
        assert owner.count(f'"{value}"') >= 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/events/event_types.py"
    )
    for symbol in expected:
        assert symbol in consumers
    for value in expected.values():
        assert f'event_type == "{value}"' not in consumers
        assert f'event_type="{value}"' not in consumers
        assert f'= "{value}"' not in consumers




def test_event_sink_field_policy_has_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/events/sink_field_policy.py",
            "src/rag_core/events/sink_payloads.py",
        )
    }

    assert EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS == frozenset(
        {
            "content_sha256",
            "corpus_id",
            "corpus_ids",
            "document_id",
            "document_key",
            "error",
            "filename",
            "message",
            "namespace",
            "ocr_page_indices",
            "quality_details",
            "redacted_url",
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

    owner = sources["src/rag_core/events/sink_field_policy.py"]
    for symbol in (
        "EVENT_SINK_SENSITIVE_OTEL_ATTRIBUTE_FIELDS",
        "EVENT_SINK_SAFE_LABEL_FIELDS",
        "EVENT_SINK_SAFE_STAGE_LABEL_FIELDS",
        "EVENT_SINK_SAFE_STAGE_LABEL_SEQUENCE_FIELDS",
        "EVENT_SINK_SENSITIVE_LOG_FIELDS",
        "EVENT_SINK_SAFE_LOG_VALUE_ALLOWLISTS",
    ):
        assert symbol in owner
        assert symbol in sources["src/rag_core/events/sink_payloads.py"]

    consumer = sources["src/rag_core/events/sink_payloads.py"]
    for old_local_name in (
        "_SENSITIVE_OTEL_ATTRIBUTE_FIELDS",
        "_SAFE_LABEL_FIELDS",
        "_SAFE_STAGE_LABEL_FIELDS",
        "_SAFE_STAGE_LABEL_SEQUENCE_FIELDS",
        "_SENSITIVE_LOG_FIELDS",
        "_SAFE_LOG_VALUE_ALLOWLISTS",
    ):
        assert f"{old_local_name} =" not in consumer




def test_retrieval_hit_export_fields_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/events/retrieval_hit_fields.py",
            "src/rag_core/events/export.py",
            "docs/expectations.md",
            "tests/test_events_export.py",
        )
    }

    assert RETRIEVAL_HIT_CORE_FIELDS == ("id", "content", "score")
    assert RETRIEVAL_HIT_CONTENT_FIELD == "content"
    assert "metadata" in RETRIEVAL_HIT_OPTIONAL_FIELDS

    owner = sources["src/rag_core/events/retrieval_hit_fields.py"]
    for symbol in (
        "RETRIEVAL_HIT_ID_FIELD",
        "RETRIEVAL_HIT_CONTENT_FIELD",
        "RETRIEVAL_HIT_SCORE_FIELD",
        "RETRIEVAL_HIT_DOCUMENT_ID_FIELD",
        "RETRIEVAL_HIT_DOCUMENT_KEY_FIELD",
        "RETRIEVAL_HIT_METADATA_FIELD",
    ):
        assert owner.count(f"{symbol}: Final[str]") == 1
        assert symbol in sources["src/rag_core/events/export.py"]
        assert symbol in sources["tests/test_events_export.py"]

    export = sources["src/rag_core/events/export.py"]
    assert "``id``, ``content``, ``score``" in export
    assert "``id``, ``text``, ``score``" not in export
    assert '"content": result.text' not in export
    assert 'document["metadata"]' not in export

    docs = sources["docs/expectations.md"]
    assert "| `text` | `text` | `content` | Retrieved chunk body |" in docs
    assert "`SearchResult.text` to an observability-friendly `content` field" in docs
    assert "emit the same\nlogical fields" not in docs




def test_event_sink_provider_defaults_have_single_sink_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/events/sinks.py",
            "src/rag_core/search/providers/event_sink_category_diagnostics.py",
            "src/rag_core/search/providers/provider_category_diagnostics.py",
            "src/rag_core/cli_doctor_output.py",
        )
    }

    assert DEFAULT_EVENT_SINK_PROVIDER == "none"
    assert EVENT_SINK_PROVIDER_ORDER == (
        NOOP_EVENT_SINK_PROVIDER,
        LOGGING_EVENT_SINK_PROVIDER,
        JSONL_EVENT_SINK_PROVIDER,
        BUFFER_EVENT_SINK_PROVIDER,
        MULTI_EVENT_SINK_PROVIDER,
        OPENTELEMETRY_EVENT_SINK_PROVIDER,
    )
    event_sinks = sources["src/rag_core/events/sinks.py"]
    assert event_sinks.count('NOOP_EVENT_SINK_PROVIDER = "none"') == 1
    assert event_sinks.count('LOGGING_EVENT_SINK_PROVIDER = "logging"') == 1
    assert event_sinks.count('JSONL_EVENT_SINK_PROVIDER = "jsonl"') == 1
    assert event_sinks.count('BUFFER_EVENT_SINK_PROVIDER = "buffer"') == 1
    assert event_sinks.count('MULTI_EVENT_SINK_PROVIDER = "multi"') == 1
    assert event_sinks.count('OPENTELEMETRY_EVENT_SINK_PROVIDER = "opentelemetry"') == 1
    assert "DEFAULT_EVENT_SINK_PROVIDER = NOOP_EVENT_SINK_PROVIDER" in event_sinks
    assert "EVENT_SINK_PROVIDER_ORDER = (" in event_sinks
    for provider_literal in (
        'provider_name = "none"',
        'provider_name = "logging"',
        'provider_name = "jsonl"',
        'provider_name = "buffer"',
        'provider_name = "multi"',
        'provider_name = "opentelemetry"',
    ):
        assert provider_literal not in event_sinks
    event_diagnostics = sources[
        "src/rag_core/search/providers/event_sink_category_diagnostics.py"
    ]
    category_diagnostics = sources[
        "src/rag_core/search/providers/provider_category_diagnostics.py"
    ]
    doctor_output = sources["src/rag_core/cli_doctor_output.py"]
    assert "describe_event_sink_provider_diagnostics" in category_diagnostics
    assert "EVENT_SINK_PROVIDER_ORDER" in event_diagnostics
    assert "EVENT_SINK_PROVIDER_ORDER" in doctor_output
    assert "_EVENT_SINK_PROVIDER_ALIASES" not in event_diagnostics
    assert '("none", "logging", "jsonl", "buffer", "multi", "opentelemetry")' not in (
        category_diagnostics + event_diagnostics + doctor_output
    )
