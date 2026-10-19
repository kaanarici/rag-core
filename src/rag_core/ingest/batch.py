"""Bounded ingestion workers and correlated batch progress."""

from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, TypeVar, Any
from pathlib import Path
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import (
    IngestBatchStarted,
    IngestBatchProgress,
    IngestBatchCompleted,
    IngestBatchFailed,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.ingest.local import LocalIngestSuccess, LocalIngestFailure

T = TypeVar("T")


async def bounded_map(
    items: Sequence[T],
    run: Callable[[int, T], Awaitable[None]],
    *,
    max_concurrency: int,
) -> None:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    indexed = iter(enumerate(items))

    async def worker() -> None:
        for position, item in indexed:
            await run(position, item)

    tasks = [
        asyncio.create_task(worker()) for _ in range(min(max_concurrency, len(items)))
    ]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class BatchEvents:
    def __init__(
        self,
        *,
        event_sink: EventSink | None,
        tenant_id: str,
        collection: str,
        planned_count: int,
    ) -> None:
        self.sink = event_sink
        self.scope: dict[str, Any] = {
            "tenant_id": tenant_id,
            "collection": collection,
            "planned_count": planned_count,
        }
        self._records: list[LocalIngestSuccess | LocalIngestFailure | None] = [
            None
        ] * planned_count
        self.start = now_ms()
        self.terminal = False
        self.done = self.success = self.failure = 0

    @property
    def records(self) -> tuple[LocalIngestSuccess | LocalIngestFailure, ...]:
        return tuple(r for r in self._records if r is not None)

    def started(self) -> None:
        emit_event(self.sink, IngestBatchStarted(**self.scope))

    def record(
        self,
        position: int,
        record: LocalIngestSuccess | LocalIngestFailure,
        *,
        error: str = "",
    ) -> None:
        self._records[position] = record
        self.done += 1
        if hasattr(record, "ingest_state"):
            self.success += 1
        else:
            self.failure += 1
        emit_event(
            self.sink,
            IngestBatchProgress(
                **self.scope,
                succeeded_count=self.success,
                failed_count=self.failure,
                completed_count=self.done,
                current_index=self.done,
                filename=getattr(record, "filename", Path(record.path).name),
                document_key=record.document_key,
                content_sha256=record.content_sha256 or "",
                status="failed" if error else "succeeded",
                ingest_state=getattr(record, "ingest_state", ""),
                error=error,
            ),
        )

    def completed(self) -> None:
        if not self.terminal:
            self.terminal = True
            emit_event(
                self.sink,
                IngestBatchCompleted(
                    **self.scope,
                    succeeded_count=self.success,
                    failed_count=self.failure,
                    duration_ms=now_ms() - self.start,
                ),
            )

    def failed(self, error: BaseException) -> None:
        if not self.terminal:
            self.terminal = True
            emit_event(
                self.sink,
                IngestBatchFailed(
                    **self.scope,
                    succeeded_count=self.success,
                    failed_count=self.failure,
                    completed_count=self.done,
                    duration_ms=now_ms() - self.start,
                    error=type(error).__name__,
                ),
            )
