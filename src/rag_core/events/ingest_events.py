"""Ingest and fetch lifecycle event records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IngestStarted:
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""
    filename: str = ""
    mime_type: str = ""
    content_sha256: str = ""
    event_type: Literal["ingest.started"] = "ingest.started"


@dataclass(frozen=True)
class IngestSkipped:
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""
    reason: str = ""
    event_type: Literal["ingest.skipped"] = "ingest.skipped"


@dataclass(frozen=True)
class IngestBatchStarted:
    namespace: str = ""
    corpus_id: str = ""
    planned_count: int = 0
    event_type: Literal["ingest.batch.started"] = "ingest.batch.started"


@dataclass(frozen=True)
class IngestBatchProgress:
    namespace: str = ""
    corpus_id: str = ""
    planned_count: int = 0
    completed_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    current_index: int = 0
    filename: str = ""
    document_key: str = ""
    content_sha256: str = ""
    manifest_status: str = ""
    manifest_reason: str = ""
    status: Literal["succeeded", "failed"] = "succeeded"
    ingest_state: str = ""
    error: str = ""
    event_type: Literal["ingest.batch.progress"] = "ingest.batch.progress"


@dataclass(frozen=True)
class IngestBatchCompleted:
    namespace: str = ""
    corpus_id: str = ""
    planned_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    duration_ms: float = 0.0
    event_type: Literal["ingest.batch.completed"] = "ingest.batch.completed"


@dataclass(frozen=True)
class IngestBatchFailed:
    namespace: str = ""
    corpus_id: str = ""
    planned_count: int = 0
    completed_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    duration_ms: float = 0.0
    error: str = ""
    event_type: Literal["ingest.batch.failed"] = "ingest.batch.failed"


@dataclass(frozen=True)
class IngestCompleted:
    namespace: str = ""
    corpus_id: str = ""
    document_id: str = ""
    chunk_count: int = 0
    duration_ms: float = 0.0
    event_type: Literal["ingest.completed"] = "ingest.completed"


@dataclass(frozen=True)
class FetchStarted:
    namespace: str = ""
    corpus_id: str = ""
    redacted_url: str = ""
    event_type: Literal["fetch.started"] = "fetch.started"


@dataclass(frozen=True)
class FetchCompleted:
    namespace: str = ""
    corpus_id: str = ""
    redacted_url: str = ""
    status_code: int = 0
    content_type: str = ""
    content_length: int | None = None
    byte_count: int = 0
    content_sha256: str = ""
    redirect_count: int = 0
    duration_ms: float = 0.0
    event_type: Literal["fetch.completed"] = "fetch.completed"


@dataclass(frozen=True)
class FetchFailed:
    namespace: str = ""
    corpus_id: str = ""
    redacted_url: str = ""
    error_type: str = ""
    duration_ms: float = 0.0
    event_type: Literal["fetch.failed"] = "fetch.failed"
