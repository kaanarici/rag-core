from __future__ import annotations

import logging
import json
from pathlib import Path

import pytest

from rag_core.events import LoggingSink
from rag_core.events.sinks import JsonlSink
from rag_core.events.types import (
    EmbedRequested,
    FetchFailed,
    IndexDeleted,
    IngestStarted,
    IngestSkipped,
    ParseCompleted,
    RerankApplied,
    SearchPlanned,
    SearchStarted,
    StageError,
)

from tests.support import TEST_API_SECRET

LOGGER_NAME = "rag_core.events"
SECRET = TEST_API_SECRET
TOKEN_SHAPED_SECRET = "ghp_abcdefghijklmnopqrstuvwxyz123456"
AWS_ACCESS_KEY_LABEL = "AKIA1234567890ABCDEF"
PREFIXED_OPENAI_SECRET = "openai:sk-proj-abcdefghijklmnopqrstuvwxyz123456"
PREFIXED_ANTHROPIC_SECRET = "anthropic:sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
PREFIXED_SLACK_XOXC_SECRET = (
    "slack:xoxc-123456789012-123456789012-abcdefghijklmnopqrstuvwxyz"
)


def _messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def _assert_private_context_omitted(
    caplog: pytest.LogCaptureFixture,
    message: str,
) -> None:
    assert SECRET not in message
    assert "/private/customer/report.pdf" not in message
    assert "https://docs.example.com/private" not in message
    assert "tenant-acme" not in message
    assert "corpus-finance" not in message
    assert "doc-payroll" not in message
    assert "Traceback" not in message
    assert all(record.exc_info is None for record in caplog.records)


def test_logging_sink_omits_identifiers_while_preserving_safe_ingest_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = IngestStarted(
        namespace=f"tenant-acme-{SECRET}",
        corpus_id=f"corpus-finance-{SECRET}",
        document_id=f"doc-payroll-{SECRET}",
        filename=f"/private/customer/report.pdf-{SECRET}",
        mime_type="application/pdf",
        content_sha256=f"sha-{SECRET}",
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        LoggingSink().emit(event)

    message = _messages(caplog)
    assert message.startswith("ingest.started")
    assert "mime_type='application/pdf'" in message
    assert "namespace=" not in message
    assert "corpus_id=" not in message
    assert "document_id=" not in message
    assert "filename=" not in message
    assert "content_sha256=" not in message
    _assert_private_context_omitted(caplog, message)


def test_logging_sink_omits_search_scope_but_keeps_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = SearchStarted(
        namespace=f"tenant-acme-{SECRET}",
        corpus_ids=(f"corpus-finance-{SECRET}",),
        query_length=42,
        limit=5,
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        LoggingSink().emit(event)

    message = _messages(caplog)
    assert message.startswith("search.started")
    assert "query_length=42" in message
    assert "limit=5" in message
    assert "namespace=" not in message
    assert "corpus_ids=" not in message
    _assert_private_context_omitted(caplog, message)


def test_logging_sink_omits_error_messages_urls_and_quality_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        LoggingSink().emit(
            FetchFailed(
                namespace=f"tenant-acme-{SECRET}",
                corpus_id=f"corpus-finance-{SECRET}",
                redacted_url=f"https://docs.example.com/private?token={SECRET}",
                error_type="TimeoutError",
                duration_ms=12.5,
            )
        )
        LoggingSink().emit(
            ParseCompleted(
                filename=f"/private/customer/report.pdf-{SECRET}",
                parser="local:pdf_inspector",
                quality_details=f"raw extraction failed for {SECRET}",
                char_count=128,
            )
        )
        LoggingSink().emit(
            StageError(
                stage="index",
                error_type="RuntimeError",
                message=f"raw failure for /private/customer/report.pdf {SECRET}",
            )
        )

    message = _messages(caplog)
    assert "fetch.failed" in message
    assert "error_type='TimeoutError'" in message
    assert "duration_ms=12.5" in message
    assert "parse.completed" in message
    assert "parser='local:pdf_inspector'" in message
    assert "char_count=128" in message
    assert "stage.error" in message
    assert "stage='index'" in message
    assert "error_type='RuntimeError'" in message
    assert "redacted_url=" not in message
    assert "quality_details=" not in message
    assert "message=" not in message
    _assert_private_context_omitted(caplog, message)


def test_logging_sink_reports_omitted_fields_when_only_sensitive_values_exist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = IndexDeleted(
        namespace=f"tenant-acme-{SECRET}",
        corpus_id=f"corpus-finance-{SECRET}",
        document_id=f"doc-payroll-{SECRET}",
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        LoggingSink().emit(event)

    message = _messages(caplog)
    assert message == "index.deleted fields=omitted"
    _assert_private_context_omitted(caplog, message)


def test_logging_sink_keeps_known_safe_normalized_labels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        LoggingSink().emit(
            IngestSkipped(
                namespace=f"tenant-acme-{SECRET}",
                corpus_id=f"corpus-finance-{SECRET}",
                document_id=f"doc-payroll-{SECRET}",
                reason="content_unchanged",
            )
        )
        LoggingSink().emit(
            SearchPlanned(
                namespace=f"tenant-acme-{SECRET}",
                corpus_ids=(f"corpus-finance-{SECRET}",),
                boost="linear_decay",
                metadata_filter="Term",
            )
        )

    message = _messages(caplog)
    assert "ingest.skipped reason='content_unchanged'" in message
    assert "search.planned" in message
    assert "boost='linear_decay'" in message
    assert "metadata_filter='Term'" in message
    assert "namespace=" not in message
    assert "corpus_id=" not in message
    assert "corpus_ids=" not in message
    assert "document_id=" not in message
    _assert_private_context_omitted(caplog, message)


def test_logging_sink_omits_unrecognized_sensitive_labels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        LoggingSink().emit(IngestSkipped(reason=SECRET))
        LoggingSink().emit(SearchPlanned(boost=SECRET, metadata_filter=SECRET))

    message = _messages(caplog)
    assert SECRET not in message
    assert "reason=" not in message
    assert "boost=" not in message
    assert "metadata_filter=" not in message
    assert all(record.exc_info is None for record in caplog.records)


def test_logging_sink_sanitizes_provider_model_and_stage_labels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        LoggingSink().emit(EmbedRequested(provider=f"openai/{SECRET}", model=SECRET))
        LoggingSink().emit(
            EmbedRequested(
                provider=PREFIXED_OPENAI_SECRET,
                model=PREFIXED_ANTHROPIC_SECRET,
            )
        )
        LoggingSink().emit(
            EmbedRequested(
                provider=PREFIXED_SLACK_XOXC_SECRET,
                model=PREFIXED_SLACK_XOXC_SECRET,
            )
        )
        LoggingSink().emit(
            EmbedRequested(provider=TOKEN_SHAPED_SECRET, model=TOKEN_SHAPED_SECRET)
        )
        LoggingSink().emit(
            EmbedRequested(provider=AWS_ACCESS_KEY_LABEL, model=AWS_ACCESS_KEY_LABEL)
        )
        LoggingSink().emit(
            SearchPlanned(
                retrieve_stage="secret_stage_token_abc123",
                postprocesses=("private_postprocess_secret",),
            )
        )

    message = _messages(caplog)
    assert SECRET not in message
    assert TOKEN_SHAPED_SECRET not in message
    assert AWS_ACCESS_KEY_LABEL not in message
    assert PREFIXED_OPENAI_SECRET not in message
    assert PREFIXED_ANTHROPIC_SECRET not in message
    assert PREFIXED_SLACK_XOXC_SECRET not in message
    assert "secret_stage_token_abc123" not in message
    assert "private_postprocess_secret" not in message
    assert "provider='unknown'" in message
    assert "model='unknown'" in message
    assert "retrieve_stage='unknown'" in message
    assert "postprocesses=['unknown']" in message


def test_jsonl_sink_redacts_private_fields_and_sanitizes_labels(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    sink = JsonlSink(log)

    sink.emit(IngestStarted(namespace="tenant", corpus_id="corp", document_id="doc", filename="/private/file.pdf"))
    sink.emit(
        EmbedRequested(
            provider=PREFIXED_OPENAI_SECRET,
            model=PREFIXED_ANTHROPIC_SECRET,
        )
    )
    sink.emit(
        EmbedRequested(
            provider=PREFIXED_SLACK_XOXC_SECRET,
            model=PREFIXED_SLACK_XOXC_SECRET,
        )
    )
    sink.emit(EmbedRequested(provider=TOKEN_SHAPED_SECRET, model=TOKEN_SHAPED_SECRET))
    sink.emit(EmbedRequested(provider=AWS_ACCESS_KEY_LABEL, model=AWS_ACCESS_KEY_LABEL))
    sink.emit(SearchPlanned(retrieve_stage="secret_stage_token_abc123"))

    payloads = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    rendered = repr(payloads)
    assert "tenant" not in rendered
    assert "corpus_id" not in payloads[0]
    assert "corpus_ids" not in rendered
    assert "'corp'" not in rendered
    assert "'document_id':" not in rendered
    assert "/private/file.pdf" not in rendered
    assert SECRET not in rendered
    assert TOKEN_SHAPED_SECRET not in rendered
    assert AWS_ACCESS_KEY_LABEL not in rendered
    assert PREFIXED_OPENAI_SECRET not in rendered
    assert PREFIXED_ANTHROPIC_SECRET not in rendered
    assert PREFIXED_SLACK_XOXC_SECRET not in rendered
    embedded_payloads = [payload for payload in payloads if payload["event_type"] == "embed.requested"]
    assert all(payload["provider"] == "unknown" for payload in embedded_payloads)
    assert all(payload["model"] == "unknown" for payload in embedded_payloads)
    assert payloads[-1]["retrieve_stage"] == "unknown"


def test_jsonl_sink_normalizes_non_finite_numbers(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    sink = JsonlSink(log)

    sink.emit(
        RerankApplied(
            provider="cohere",
            model="rerank-v3.5",
            provider_score_min=float("nan"),
            provider_score_max=float("inf"),
            search_score_min=float("-inf"),
            search_score_max=0.7,
        )
    )

    raw = log.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["provider_score_min"] is None
    assert payload["provider_score_max"] is None
    assert payload["search_score_min"] is None
    assert payload["search_score_max"] == 0.7
    assert "NaN" not in raw
    assert "Infinity" not in raw
