"""Server-sent ingest job status events for the optional HTTP runtime."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Final

from rag_core.runtime.jobs import (
    IngestJobRecord,
    IngestJobStore,
    JobStatus,
    ingest_job_status_payload,
    is_terminal_job_status,
)

_POLL_INTERVAL_SECONDS: Final[float] = 0.3
_HEARTBEAT_INTERVAL_SECONDS: Final[float] = 15.0


def ingest_job_event_stream_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }


async def stream_ingest_job_events(
    jobs: IngestJobStore,
    job_id: str,
    *,
    initial_record: IngestJobRecord | None = None,
) -> AsyncIterator[str]:
    last_status: JobStatus | None = None
    next_heartbeat = time.monotonic() + _HEARTBEAT_INTERVAL_SECONDS
    record = initial_record

    while True:
        if record is None:
            record = jobs.get(job_id)
        if record is None:
            break

        now = time.monotonic()
        if record.status != last_status:
            last_status = record.status
            yield _status_event(ingest_job_status_payload(record))
            if is_terminal_job_status(record.status):
                break
            next_heartbeat = now + _HEARTBEAT_INTERVAL_SECONDS
        elif now >= next_heartbeat:
            yield ": heartbeat\n\n"
            next_heartbeat = now + _HEARTBEAT_INTERVAL_SECONDS

        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        record = None


def _status_event(payload: dict[str, object]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return f"event: status\ndata: {data}\n\n"
