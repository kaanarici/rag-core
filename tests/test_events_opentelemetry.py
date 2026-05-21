from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from typing import Any, Literal, cast

import pytest

import rag_core
from rag_core.events import OpenTelemetrySink
from rag_core.events.types import (
    EmbedCompleted,
    ParseCompleted,
    RerankApplied,
    SearchPlanned,
    StageError,
)

AWS_ACCESS_KEY_LABEL = "AKIA1234567890ABCDEF"
PREFIXED_OPENAI_SECRET = "openai:sk-proj-abcdefghijklmnopqrstuvwxyz123456"
PREFIXED_ANTHROPIC_SECRET = "anthropic:sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456"
PREFIXED_SLACK_XOXC_SECRET = (
    "slack:xoxc-123456789012-123456789012-abcdefghijklmnopqrstuvwxyz"
)


class _FakeSpan:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def add_event(self, name: str, *, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))


def _install_fake_opentelemetry(
    monkeypatch: pytest.MonkeyPatch,
    span: Any,
) -> None:
    package = types.ModuleType("opentelemetry")
    trace_module = types.ModuleType("opentelemetry.trace")
    trace_module.get_current_span = lambda: span  # type: ignore[attr-defined]
    package.trace = trace_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", package)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_module)


def test_open_telemetry_sink_lives_under_events_namespace() -> None:
    assert "OpenTelemetrySink" not in rag_core.__all__
    assert rag_core.events.OpenTelemetrySink is OpenTelemetrySink


def test_open_telemetry_sink_emits_prefixed_safe_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    OpenTelemetrySink().emit(
        SearchPlanned(
            search_id="customer-search-123",
            namespace="acme",
            corpus_ids=("help", "docs"),
            limit=5,
            channels=("dense:dense:primary", "sparse:bm25:bm25"),
            prefetch_limits=(10, 20),
            query_transforms=(),
            metadata_filter="Term",
            rerank_timeout_ms=1500.0,
        )
    )

    assert span.events == [
        (
            "search.planned",
                {
                    "rag_core.limit": 5,
                    "rag_core.final_limit": 0,
                    "rag_core.corpus_count": 2,
                    "rag_core.channels": ["dense:dense:primary", "sparse:bm25:bm25"],
                "rag_core.prefetch_limits": [10, 20],
                "rag_core.fusion": "",
                "rag_core.plan_rerank": "",
                "rag_core.boost": "",
                "rag_core.metadata_filter": "Term",
                "rag_core.content_type_count": 0,
                "rag_core.document_id_count": 0,
                "rag_core.rerank_candidate_count": 0,
                "rag_core.rerank_timeout_ms": 1500.0,
                "rag_core.rerank_max_output": 0,
                "rag_core.rerank_fallback_on_error": True,
                "rag_core.use_lexical_search": False,
                "rag_core.retrieve_stage": "",
                "rag_core.fuse_stage": "",
                "rag_core.rerank_stage": "",
            },
        )
    ]


def test_open_telemetry_sink_omits_sensitive_attributes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    OpenTelemetrySink().emit(
        StageError(
            stage="index",
            error_type="RuntimeError",
            message="raw internal failure for /private/file.pdf",
        )
    )

    _, attributes = span.events[0]
    assert attributes == {
        "rag_core.stage": "index",
        "rag_core.error_type": "RuntimeError",
    }


def test_open_telemetry_sink_emits_embedding_cache_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    OpenTelemetrySink().emit(
        EmbedCompleted(
            provider="openai",
            model="text-embedding-3-small",
            text_count=3,
            role="dense",
            duration_ms=4.5,
            cache_hits=2,
            cache_misses=1,
            cache_writes=1,
            cache_bypasses=0,
        )
    )

    assert span.events == [
        (
            "embed.completed",
            {
                "rag_core.provider": "openai",
                "rag_core.model": "text-embedding-3-small",
                "rag_core.text_count": 3,
                "rag_core.role": "dense",
                "rag_core.duration_ms": 4.5,
                "rag_core.cache_hits": 2,
                "rag_core.cache_misses": 1,
                "rag_core.cache_writes": 1,
                "rag_core.cache_bypasses": 0,
            },
        )
    ]


def test_open_telemetry_sink_sanitizes_provider_model_and_stage_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    sink = OpenTelemetrySink()
    sink.emit(
        EmbedCompleted(
            provider=PREFIXED_OPENAI_SECRET,
            model=PREFIXED_ANTHROPIC_SECRET,
        )
    )
    sink.emit(
        EmbedCompleted(
            provider=PREFIXED_SLACK_XOXC_SECRET,
            model=PREFIXED_SLACK_XOXC_SECRET,
        )
    )
    sink.emit(
        EmbedCompleted(
            provider="ghp_abcdefghijklmnopqrstuvwxyz123456",
            model="ghp_abcdefghijklmnopqrstuvwxyz123456",
        )
    )
    sink.emit(EmbedCompleted(provider=AWS_ACCESS_KEY_LABEL, model=AWS_ACCESS_KEY_LABEL))
    sink.emit(
        SearchPlanned(
            retrieve_stage="secret_stage_token_abc123",
            postprocesses=("private_postprocess_secret",),
        )
    )

    assert span.events[0][1]["rag_core.provider"] == "unknown"
    assert span.events[0][1]["rag_core.model"] == "unknown"
    assert span.events[1][1]["rag_core.provider"] == "unknown"
    assert span.events[1][1]["rag_core.model"] == "unknown"
    assert span.events[2][1]["rag_core.provider"] == "unknown"
    assert span.events[2][1]["rag_core.model"] == "unknown"
    assert span.events[3][1]["rag_core.provider"] == "unknown"
    assert span.events[3][1]["rag_core.model"] == "unknown"
    assert span.events[4][1]["rag_core.retrieve_stage"] == "unknown"
    assert span.events[4][1]["rag_core.postprocesses"] == ["unknown"]


def test_open_telemetry_sink_emits_rerank_rank_and_score_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    OpenTelemetrySink().emit(
        RerankApplied(
            provider="cohere",
            model="rerank-v3.5",
            input_count=6,
            candidate_count=4,
            result_count=3,
            top_k=3,
            fallback_reason="",
            truncation_reason="candidate_budget",
            duration_ms=1.5,
            succeeded=True,
            provider_result_count=5,
            accepted_count=3,
            dropped_count=2,
            rank_changed_count=3,
            rank_promoted_count=1,
            rank_demoted_count=2,
            max_rank_gain=2,
            max_rank_loss=1,
            provider_score_min=0.12,
            provider_score_max=0.91,
            search_score_min=0.2,
            search_score_max=0.9,
        )
    )

    assert span.events == [
        (
            "rerank.applied",
            {
                "rag_core.provider": "cohere",
                "rag_core.model": "rerank-v3.5",
                "rag_core.input_count": 6,
                "rag_core.candidate_count": 4,
                "rag_core.result_count": 3,
                "rag_core.top_k": 3,
                "rag_core.fallback_reason": "",
                "rag_core.truncation_reason": "candidate_budget",
                "rag_core.duration_ms": 1.5,
                "rag_core.succeeded": True,
                "rag_core.provider_result_count": 5,
                "rag_core.accepted_count": 3,
                "rag_core.dropped_count": 2,
                "rag_core.rank_changed_count": 3,
                "rag_core.rank_promoted_count": 1,
                "rag_core.rank_demoted_count": 2,
                "rag_core.max_rank_gain": 2,
                "rag_core.max_rank_loss": 1,
                "rag_core.provider_score_min": 0.12,
                "rag_core.provider_score_max": 0.91,
                "rag_core.search_score_min": 0.2,
                "rag_core.search_score_max": 0.9,
            },
        )
    ]
    _, attributes = span.events[0]
    assert not any("query" in key for key in attributes)
    assert not any("document" in key for key in attributes)
    assert not any("content" in key for key in attributes)
    assert "rag_core.search_id" not in attributes


def test_open_telemetry_sink_can_emit_sensitive_attributes_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    OpenTelemetrySink(include_sensitive_attributes=True).emit(
        StageError(
            stage="index",
            error_type="RuntimeError",
            message="raw internal failure for /private/file.pdf",
        )
    )

    _, attributes = span.events[0]
    assert attributes["rag_core.message"] == "raw internal failure for /private/file.pdf"


def test_open_telemetry_sink_omits_none_empty_and_sensitive_parse_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    OpenTelemetrySink().emit(
        ParseCompleted(
            filename="scan.pdf",
            parser="local:pdf_inspector",
            quality_details="low extraction quality",
            ocr_page_indices=(0, 2),
            extraction_ratio=None,
        )
    )

    _, attributes = span.events[0]
    assert attributes["rag_core.parser"] == "local:pdf_inspector"
    assert "rag_core.ocr_page_indices" not in attributes
    assert "rag_core.extraction_ratio" not in attributes
    assert "rag_core.filename" not in attributes
    assert "rag_core.quality_details" not in attributes


def test_open_telemetry_sink_coerces_mixed_sequences_to_string_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    _install_fake_opentelemetry(monkeypatch, span)

    @dataclass(frozen=True)
    class _Event:
        values: tuple[str, int, bool] = ("a", 1, True)
        event_type: Literal["test.mixed"] = "test.mixed"

    OpenTelemetrySink().emit(cast(Any, _Event()))

    _, attributes = span.events[0]
    assert attributes["rag_core.values"] == ["a", "1", "True"]


def test_open_telemetry_sink_swallows_span_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSpan:
        def add_event(self, name: str, *, attributes: dict[str, Any]) -> None:
            raise RuntimeError("span closed")

    _install_fake_opentelemetry(monkeypatch, BrokenSpan())

    sink = OpenTelemetrySink()
    sink.emit(StageError(stage="search", error_type="RuntimeError"))

    assert sink.failure_count == 1


def test_open_telemetry_sink_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> types.ModuleType:
        if name == "opentelemetry.trace":
            raise ImportError("missing")
        return original_import_module(name)

    original_import_module = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ImportError, match=r"rag-core\[opentelemetry\]"):
        OpenTelemetrySink()


def test_open_telemetry_sink_accepts_real_sdk_span() -> None:
    pytest.importorskip("opentelemetry.trace")
    sdk_trace: Any = pytest.importorskip("opentelemetry.sdk.trace")
    sdk_export: Any = pytest.importorskip("opentelemetry.sdk.trace.export")
    exported_spans: list[Any] = []

    class _Exporter:
        def export(self, spans: list[Any]) -> object:
            exported_spans.extend(spans)
            return sdk_export.SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_export.SimpleSpanProcessor(_Exporter()))
    tracer = provider.get_tracer("rag-core-tests")
    with tracer.start_as_current_span("search"):
        OpenTelemetrySink().emit(
            SearchPlanned(
                namespace="acme",
                corpus_ids=("help",),
                channels=("dense:dense:primary",),
                prefetch_limits=(10,),
            )
        )
    provider.shutdown()

    [event] = exported_spans[0].events
    assert event.name == "search.planned"
    assert dict(event.attributes)["rag_core.channels"] == ("dense:dense:primary",)
