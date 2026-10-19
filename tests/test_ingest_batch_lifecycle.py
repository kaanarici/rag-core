from __future__ import annotations
from typing import NoReturn

import asyncio
import hashlib
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest
from rag_core.ingest.archive import ingest_zip_archive_with_core
from rag_core.core_models import IngestedDocument
from rag_core.events import (
    EventBuffer,
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchProgress,
)
from rag_core.ingest.batch import BatchEvents
from rag_core.ingest.local import LocalIngestRequest, run_local_ingest_with_core


class _OutOfOrderLocalCore:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.all_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def ensure_ready(self) -> None:
        return None

    async def add_file(
        self,
        file_path: Path,
        *,
        tenant_id: str,
        collection: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        self.started.append(file_path.name)
        if len(self.started) == 3:
            self.all_started.set()
        if file_path.name == "a.md":
            await self.release_first.wait()
        if file_path.name == "b.md":
            raise RuntimeError("parse failed")
        return IngestedDocument(
            document_id=f"doc-{file_path.stem}",
            tenant_id=tenant_id,
            collection=collection,
            chunk_count=1,
            filename=file_path.name,
            mime_type="text/markdown",
            document_key=document_key,
            content_sha256=f"hash-{file_path.stem}",
            ingest_state="created",
        )

    async def close(self) -> None:
        return None

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")


class _OutOfOrderRemoteCore:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.all_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def ensure_ready(self) -> None:
        return None

    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        self.started.append(url)
        if len(self.started) == 3:
            self.all_started.set()
        if url.endswith("/a"):
            await self.release_first.wait()
        if url.endswith("/b"):
            raise RuntimeError("fetch failed")
        slug = url.rsplit("/", 1)[-1]
        return IngestedDocument(
            document_id=f"doc-{slug}",
            tenant_id=kwargs["tenant_id"],
            collection=kwargs["collection"],
            chunk_count=1,
            filename=f"{slug}.txt",
            mime_type="text/plain",
            document_key=f"url:{url}",
            content_sha256=f"hash-{slug}",
            ingest_state="created",
            metadata={"source_url": url},
        )

    async def close(self) -> None:
        return None


class _CancellationCore:
    def __init__(self) -> None:
        self.call_count = 0
        self.second_started = asyncio.Event()
        self.closed = False

    async def ensure_ready(self) -> None:
        return None

    async def _ingest(
        self,
        *,
        filename: str,
        tenant_id: str,
        collection: str,
        document_key: str | None,
        file_bytes: bytes,
    ) -> IngestedDocument:
        self.call_count += 1
        if self.call_count == 2:
            self.second_started.set()
            await asyncio.Event().wait()
        return IngestedDocument(
            document_id=f"doc-{self.call_count}",
            tenant_id=tenant_id,
            collection=collection,
            chunk_count=1,
            filename=filename,
            mime_type="text/markdown",
            document_key=document_key,
            content_sha256=hashlib.sha256(file_bytes).hexdigest(),
            ingest_state="created",
        )

    async def add_bytes(self, **kwargs: object) -> IngestedDocument:
        return await self._ingest(
            filename=cast(str, kwargs["filename"]),
            tenant_id=cast(str, kwargs["tenant_id"]),
            collection=cast(str, kwargs["collection"]),
            document_key=cast(str | None, kwargs["document_key"]),
            file_bytes=cast(bytes, kwargs["file_bytes"]),
        )

    async def add_file(
        self,
        file_path: Path,
        *,
        tenant_id: str,
        collection: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        del metadata, force_reindex
        return await self._ingest(
            filename=file_path.name,
            tenant_id=tenant_id,
            collection=collection,
            document_key=document_key,
            file_bytes=pre_read_bytes or file_path.read_bytes(),
        )

    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        return await self._ingest(
            filename=f"{url.rsplit('/', 1)[-1]}.md",
            tenant_id=cast(str, kwargs["tenant_id"]),
            collection=cast(str, kwargs["collection"]),
            document_key=f"url:{url}",
            file_bytes=url.encode(),
        )

    async def close(self) -> None:
        self.closed = True

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")












def test_batch_lifecycle_emits_only_first_terminal_event() -> None:
    events = EventBuffer()
    lifecycle = BatchEvents(
        event_sink=events,
        tenant_id="acme",
        collection="help",
        planned_count=1,
    )

    lifecycle.started()
    lifecycle.failed(error=RuntimeError("failed"))
    lifecycle.completed()

    assert [event.event_type for event in events.events] == [
        "ingest.batch.started",
        "ingest.batch.failed",
    ]












async def _assert_cancelled_archive_lifecycle(tmp_path: Path) -> None:
    archive_path = tmp_path / "cancel.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a.md", "# A")
        archive.writestr("b.md", "# B")
    core = _CancellationCore()
    events = EventBuffer()
    task = asyncio.create_task(
        ingest_zip_archive_with_core(
            core=core,
            archive_path=archive_path,
            tenant_id="acme",
            collection="help",
            max_concurrency=1,
            event_sink=events,
        )
    )
    await asyncio.wait_for(core.second_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_cancelled_lifecycle_events(events)


async def _assert_cancelled_local_lifecycle(tmp_path: Path) -> None:
    docs = tmp_path / "cancel-local"
    docs.mkdir()
    (docs / "a.md").write_text("# A", encoding="utf-8")
    (docs / "b.md").write_text("# B", encoding="utf-8")
    core = _CancellationCore()
    events = EventBuffer()
    task = asyncio.create_task(
        run_local_ingest_with_core(
            LocalIngestRequest(
                path=docs,
                tenant_id="acme",
                collection="help",
                max_concurrency=1,
            ),
            core=core,
            event_sink=events,
        )
    )
    await asyncio.wait_for(core.second_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert core.closed is False
    _assert_cancelled_lifecycle_events(events)




def _assert_cancelled_lifecycle_events(events: EventBuffer) -> None:
    assert [event.event_type for event in events.events] == [
        "ingest.batch.started",
        "ingest.batch.progress",
        "ingest.batch.failed",
    ]
    failed = events.events[-1]
    assert isinstance(failed, IngestBatchFailed)
    assert failed.planned_count == 2
    assert failed.completed_count == 1
    assert failed.succeeded_count == 1
    assert failed.failed_count == 0
    assert failed.error == "CancelledError"


def _assert_failed_terminal(
    events: EventBuffer,
    *,
    planned_count: int,
    completed_count: int,
    succeeded_count: int,
    error: str,
    has_progress: bool = True,
) -> None:
    expected = ["ingest.batch.started"]
    if has_progress:
        expected.append("ingest.batch.progress")
    expected.append("ingest.batch.failed")
    assert [event.event_type for event in events.events] == expected
    failed = events.events[-1]
    assert isinstance(failed, IngestBatchFailed)
    assert failed.planned_count == planned_count
    assert failed.completed_count == completed_count
    assert failed.succeeded_count == succeeded_count
    assert failed.failed_count == completed_count - succeeded_count
    assert failed.error == error


async def _assert_local_lifecycle(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (docs / name).write_text(name, encoding="utf-8")
    core = _OutOfOrderLocalCore()
    events = EventBuffer()

    task = asyncio.create_task(
        run_local_ingest_with_core(
            LocalIngestRequest(
                path=docs,
                tenant_id="acme",
                collection="help",
                max_concurrency=3,
            ),
            core=core,
            event_sink=events,
        )
    )
    try:
        await asyncio.wait_for(core.all_started.wait(), timeout=1.0)
        await _wait_for_progress(events, count=2)
    finally:
        core.release_first.set()
    result = await task

    assert [Path(record.path).name for record in result.records] == [
        "a.md",
        "b.md",
        "c.md",
    ]
    _assert_shared_lifecycle_events(events)




async def _wait_for_progress(events: EventBuffer, *, count: int) -> None:
    for _ in range(100):
        progress = [
            event for event in events.events if isinstance(event, IngestBatchProgress)
        ]
        if len(progress) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected at least {count} progress events")


def _assert_shared_lifecycle_events(events: EventBuffer) -> None:
    progress = [
        event for event in events.events if isinstance(event, IngestBatchProgress)
    ]
    completed = [
        event for event in events.events if isinstance(event, IngestBatchCompleted)
    ]

    assert [event.current_index for event in progress] == [1, 2, 3]
    assert [event.completed_count for event in progress] == [1, 2, 3]
    assert [event.succeeded_count for event in progress] == sorted(
        event.succeeded_count for event in progress
    )
    assert [event.failed_count for event in progress] == sorted(
        event.failed_count for event in progress
    )
    assert len(completed) == 1
    assert completed[0].succeeded_count == 2
    assert completed[0].failed_count == 1










@pytest.mark.parametrize("source", ["local", "archive"])
def test_cancellation_joins_workers_and_reports_partial_progress(tmp_path: Path, source: str) -> None:
    asyncio.run(_assert_cancelled_local_lifecycle(tmp_path) if source == "local" else _assert_cancelled_archive_lifecycle(tmp_path))


def test_local_batch_out_of_order_progress_and_record_order(tmp_path: Path) -> None:
    asyncio.run(_assert_local_lifecycle(tmp_path))
