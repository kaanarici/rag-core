from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from rag_core.events.emit import emit_event
from rag_core.events.types import IngestBatchProgress
from rag_core.remote_ingest_models import (
    RemoteManifestStatus,
    RemoteUrlIngestFailure,
    RemoteUrlIngestPlan,
    RemoteUrlIngestSuccess,
    RemoteUrlSourceItem,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink


RemoteIngestProgressStatus = Literal["succeeded", "failed"]


class RemoteUrlIngestProgress:
    def __init__(
        self,
        *,
        event_sink: EventSink | None,
        plan: RemoteUrlIngestPlan,
    ) -> None:
        self._event_sink = event_sink
        self._plan = plan
        self._lock = asyncio.Lock()
        self._completed_count = 0
        self._succeeded_count = 0
        self._failed_count = 0

    async def record(
        self,
        *,
        item: RemoteUrlSourceItem,
        record: RemoteUrlIngestSuccess | RemoteUrlIngestFailure,
        document_key: str,
        content_sha256: str,
        manifest_status: RemoteManifestStatus,
        manifest_reason: str,
        status: RemoteIngestProgressStatus,
        ingest_state: str,
        error: str,
    ) -> None:
        async with self._lock:
            self._completed_count += 1
            if isinstance(record, RemoteUrlIngestSuccess):
                self._succeeded_count += 1
                event_error = ""
            else:
                self._failed_count += 1
                event_error = error
            emit_event(
                self._event_sink,
                IngestBatchProgress(
                    namespace=self._plan.namespace,
                    corpus_id=self._plan.corpus_id,
                    planned_count=self._plan.url_count,
                    completed_count=self._completed_count,
                    succeeded_count=self._succeeded_count,
                    failed_count=self._failed_count,
                    current_index=self._completed_count,
                    filename=item.redacted_url,
                    document_key=document_key,
                    content_sha256=content_sha256,
                    manifest_status=manifest_status,
                    manifest_reason=manifest_reason,
                    status=status,
                    ingest_state=ingest_state,
                    error=event_error,
                ),
            )
