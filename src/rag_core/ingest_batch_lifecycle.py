from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import (
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchProgress,
    IngestBatchStarted,
)
from rag_core.ingest_progress_statuses import IngestProgressStatus

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class IngestBatchProgressPayload:
    filename: str
    document_key: str
    content_sha256: str
    manifest_status: str
    manifest_reason: str
    status: IngestProgressStatus
    ingest_state: str = ""
    error: str = ""


class IngestBatchLifecycle(Generic[RecordT]):
    def __init__(
        self,
        *,
        event_sink: EventSink | None,
        namespace: str,
        corpus_id: str,
        planned_count: int,
        is_success: Callable[[RecordT], bool],
        error_type: Callable[[Exception], str],
    ) -> None:
        self._event_sink = event_sink
        self._namespace = namespace
        self._corpus_id = corpus_id
        self._planned_count = planned_count
        self._is_success = is_success
        self._error_type = error_type
        self._started_ms = now_ms()
        self._records: list[RecordT | None] = [None for _ in range(planned_count)]
        self._lock = asyncio.Lock()
        self._completed_count = 0
        self._succeeded_count = 0
        self._failed_count = 0

    def started(self) -> None:
        emit_event(
            self._event_sink,
            IngestBatchStarted(
                namespace=self._namespace,
                corpus_id=self._corpus_id,
                planned_count=self._planned_count,
            ),
        )

    async def record(
        self,
        *,
        position: int,
        record: RecordT,
        progress: IngestBatchProgressPayload,
    ) -> None:
        async with self._lock:
            self._records[position] = record
            self._completed_count += 1
            if self._is_success(record):
                self._succeeded_count += 1
                error = ""
            else:
                self._failed_count += 1
                error = progress.error
            emit_event(
                self._event_sink,
                IngestBatchProgress(
                    namespace=self._namespace,
                    corpus_id=self._corpus_id,
                    planned_count=self._planned_count,
                    completed_count=self._completed_count,
                    succeeded_count=self._succeeded_count,
                    failed_count=self._failed_count,
                    current_index=self._completed_count,
                    filename=progress.filename,
                    document_key=progress.document_key,
                    content_sha256=progress.content_sha256,
                    manifest_status=progress.manifest_status,
                    manifest_reason=progress.manifest_reason,
                    status=progress.status,
                    ingest_state=progress.ingest_state,
                    error=error,
                ),
            )

    @property
    def records(self) -> tuple[RecordT, ...]:
        return tuple(record for record in self._records if record is not None)

    async def records_snapshot(self) -> tuple[RecordT, ...]:
        async with self._lock:
            return self.records

    def completed(self, records: Sequence[RecordT] | None = None) -> None:
        succeeded_count, failed_count = self._counts(records)
        emit_event(
            self._event_sink,
            IngestBatchCompleted(
                namespace=self._namespace,
                corpus_id=self._corpus_id,
                planned_count=self._planned_count,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                duration_ms=now_ms() - self._started_ms,
            ),
        )

    def failed(
        self,
        *,
        error: Exception,
        records: Sequence[RecordT] | None = None,
    ) -> None:
        active_records = records if records is not None else self.records
        succeeded_count, failed_count = self._counts(active_records)
        emit_event(
            self._event_sink,
            IngestBatchFailed(
                namespace=self._namespace,
                corpus_id=self._corpus_id,
                planned_count=self._planned_count,
                completed_count=len(active_records),
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                duration_ms=now_ms() - self._started_ms,
                error=self._error_type(error),
            ),
        )

    def _counts(self, records: Sequence[RecordT] | None) -> tuple[int, int]:
        active_records = self.records if records is None else records
        succeeded_count = sum(1 for record in active_records if self._is_success(record))
        return succeeded_count, len(active_records) - succeeded_count
