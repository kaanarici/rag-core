"""Engine lifecycle events and sinks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "AUDIT_EVENT_TYPES",
    "AuditContext",
    "ChunkProduced",
    "ContextualizeCompleted",
    "ContextualizeStarted",
    "EmbedCompleted",
    "EmbedRequested",
    "EmbeddingTraceSummary",
    "Event",
    "EventBuffer",
    "EventSink",
    "FetchCompleted",
    "FetchFailed",
    "FetchStarted",
    "IndexDeleted",
    "IndexUpserted",
    "IngestBatchCompleted",
    "IngestBatchFailed",
    "IngestBatchProgress",
    "IngestBatchStarted",
    "IngestCompleted",
    "IngestSkipped",
    "IngestStarted",
    "JsonlSink",
    "LexicalSidecarBoundExceeded",
    "LoggingSink",
    "MultiSink",
    "NeighborExpandSkipped",
    "NoOpSink",
    "OcrApplied",
    "OpenTelemetrySink",
    "ParseCompleted",
    "RerankApplied",
    "SearchCompleted",
    "SearchPlanned",
    "SearchStarted",
    "SearchStageCompleted",
    "SearchStageTraceSummary",
    "SearchTraceSummary",
    "SidecarApplied",
    "StageError",
    "summarize_embedding_trace",
    "summarize_embedding_trace_payloads",
    "summarize_search_trace",
    "summarize_search_trace_payloads",
    "summarize_search_trace_payload_runs",
    "summarize_search_trace_runs",
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AUDIT_EVENT_TYPES": ("rag_core.events.types", "AUDIT_EVENT_TYPES"),
    "AuditContext": ("rag_core.events.types", "AuditContext"),
    "ChunkProduced": ("rag_core.events.types", "ChunkProduced"),
    "ContextualizeCompleted": ("rag_core.events.types", "ContextualizeCompleted"),
    "ContextualizeStarted": ("rag_core.events.types", "ContextualizeStarted"),
    "EmbedCompleted": ("rag_core.events.types", "EmbedCompleted"),
    "EmbedRequested": ("rag_core.events.types", "EmbedRequested"),
    "EmbeddingTraceSummary": (
        "rag_core.events.embedding_trace_summary",
        "EmbeddingTraceSummary",
    ),
    "Event": ("rag_core.events.types", "Event"),
    "FetchCompleted": ("rag_core.events.types", "FetchCompleted"),
    "FetchFailed": ("rag_core.events.types", "FetchFailed"),
    "FetchStarted": ("rag_core.events.types", "FetchStarted"),
    "IndexDeleted": ("rag_core.events.types", "IndexDeleted"),
    "IndexUpserted": ("rag_core.events.types", "IndexUpserted"),
    "IngestBatchCompleted": ("rag_core.events.types", "IngestBatchCompleted"),
    "IngestBatchFailed": ("rag_core.events.types", "IngestBatchFailed"),
    "IngestBatchProgress": ("rag_core.events.types", "IngestBatchProgress"),
    "IngestBatchStarted": ("rag_core.events.types", "IngestBatchStarted"),
    "IngestCompleted": ("rag_core.events.types", "IngestCompleted"),
    "IngestSkipped": ("rag_core.events.types", "IngestSkipped"),
    "IngestStarted": ("rag_core.events.types", "IngestStarted"),
    "LexicalSidecarBoundExceeded": (
        "rag_core.events.types",
        "LexicalSidecarBoundExceeded",
    ),
    "NeighborExpandSkipped": ("rag_core.events.types", "NeighborExpandSkipped"),
    "OcrApplied": ("rag_core.events.types", "OcrApplied"),
    "ParseCompleted": ("rag_core.events.types", "ParseCompleted"),
    "RerankApplied": ("rag_core.events.types", "RerankApplied"),
    "SearchCompleted": ("rag_core.events.types", "SearchCompleted"),
    "SearchPlanned": ("rag_core.events.types", "SearchPlanned"),
    "SearchStarted": ("rag_core.events.types", "SearchStarted"),
    "SearchStageCompleted": ("rag_core.events.types", "SearchStageCompleted"),
    "SearchStageTraceSummary": ("rag_core.events.traces", "SearchStageTraceSummary"),
    "SearchTraceSummary": ("rag_core.events.traces", "SearchTraceSummary"),
    "SidecarApplied": ("rag_core.events.types", "SidecarApplied"),
    "StageError": ("rag_core.events.types", "StageError"),
    "summarize_embedding_trace": (
        "rag_core.events.embedding_trace_summary",
        "summarize_embedding_trace",
    ),
    "summarize_embedding_trace_payloads": (
        "rag_core.events.embedding_trace_summary",
        "summarize_embedding_trace_payloads",
    ),
    "summarize_search_trace": ("rag_core.events.traces", "summarize_search_trace"),
    "summarize_search_trace_payloads": (
        "rag_core.events.traces",
        "summarize_search_trace_payloads",
    ),
    "summarize_search_trace_payload_runs": (
        "rag_core.events.traces",
        "summarize_search_trace_payload_runs",
    ),
    "summarize_search_trace_runs": (
        "rag_core.events.traces",
        "summarize_search_trace_runs",
    ),
    "EventSink": ("rag_core.events.sink", "EventSink"),
    "EventBuffer": ("rag_core.events.sinks", "EventBuffer"),
    "JsonlSink": ("rag_core.events.sinks", "JsonlSink"),
    "LoggingSink": ("rag_core.events.sinks", "LoggingSink"),
    "MultiSink": ("rag_core.events.sinks", "MultiSink"),
    "NoOpSink": ("rag_core.events.sinks", "NoOpSink"),
    "OpenTelemetrySink": ("rag_core.events.sinks", "OpenTelemetrySink"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, symbol = target
    return getattr(import_module(module_name), symbol)


def __dir__() -> list[str]:
    return list(__all__)
