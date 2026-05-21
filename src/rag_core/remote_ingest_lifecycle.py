from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import (
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchStarted,
)
from rag_core.remote_ingest_records import (
    remote_ingest_error_type,
    remote_ingest_record_counts,
)
from rag_core.remote_ingest_results import (
    RemoteUrlIngestFailure,
    RemoteUrlIngestSuccess,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.fetching import FetchClient
    from rag_core.remote_ingest_models import RemoteUrlIngestPlan, RemoteUrlIngestRequest


RemoteUrlIngestRecord = RemoteUrlIngestSuccess | RemoteUrlIngestFailure


class RemoteUrlIngestBatchLifecycle:
    def __init__(
        self,
        *,
        event_sink: EventSink | None,
        plan: RemoteUrlIngestPlan,
    ) -> None:
        self._event_sink = event_sink
        self._plan = plan
        self._started_ms = now_ms()

    def started(self) -> None:
        emit_event(
            self._event_sink,
            IngestBatchStarted(
                namespace=self._plan.namespace,
                corpus_id=self._plan.corpus_id,
                planned_count=self._plan.url_count,
            ),
        )

    def completed(self, records: Sequence[RemoteUrlIngestRecord]) -> None:
        succeeded_count, failed_count = remote_ingest_record_counts(records)
        emit_event(
            self._event_sink,
            IngestBatchCompleted(
                namespace=self._plan.namespace,
                corpus_id=self._plan.corpus_id,
                planned_count=self._plan.url_count,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                duration_ms=now_ms() - self._started_ms,
            ),
        )

    def failed(
        self,
        *,
        records: Sequence[RemoteUrlIngestRecord],
        error: Exception,
    ) -> None:
        succeeded_count, failed_count = remote_ingest_record_counts(records)
        emit_event(
            self._event_sink,
            IngestBatchFailed(
                namespace=self._plan.namespace,
                corpus_id=self._plan.corpus_id,
                planned_count=self._plan.url_count,
                completed_count=len(records),
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                duration_ms=now_ms() - self._started_ms,
                error=remote_ingest_error_type(error),
            ),
        )


def validate_remote_fetch_configuration(
    request: RemoteUrlIngestRequest,
    fetch_client: FetchClient | None,
) -> None:
    if fetch_client is None:
        return
    if request.fetch_policy is not None:
        raise ValueError("fetch_client cannot be combined with request fetch_policy")
    if request.fetch_limits is not None:
        raise ValueError("fetch_client cannot be combined with request fetch_limits")
