from __future__ import annotations

import asyncio
import hashlib
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest
from rag_core._engine.core_archive_ingest import ingest_zip_archive_with_core
from rag_core.core_models import IngestedDocument
from rag_core.events import (
    EventBuffer,
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchProgress,
)
from rag_core.ingest.lifecycle import IngestBatchLifecycle
from rag_core.ingest.local import LocalIngestRequest, run_local_ingest
from rag_core.ingest.local.models import LocalIngestSuccess
from rag_core.ingest.urls import RemoteUrlIngestRequest, run_remote_url_ingest


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
        namespace: str,
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
            namespace=namespace,
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
            namespace=kwargs["namespace"],
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
        namespace: str,
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
            namespace=namespace,
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
            namespace=cast(str, kwargs["namespace"]),
            collection=cast(str, kwargs["collection"]),
            document_key=cast(str | None, kwargs["document_key"]),
            file_bytes=cast(bytes, kwargs["file_bytes"]),
        )

    async def add_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        collection: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        del metadata, force_reindex
        return await self._ingest(
            filename=file_path.name,
            namespace=namespace,
            collection=collection,
            document_key=document_key,
            file_bytes=pre_read_bytes or file_path.read_bytes(),
        )

    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        return await self._ingest(
            filename=f"{url.rsplit('/', 1)[-1]}.md",
            namespace=cast(str, kwargs["namespace"]),
            collection=cast(str, kwargs["collection"]),
            document_key=f"url:{url}",
            file_bytes=url.encode(),
        )

    async def close(self) -> None:
        self.closed = True


class _CleanupCore(_CancellationCore):
    def __init__(
        self,
        *,
        block_close: bool = False,
        close_error: Exception | None = None,
        ready_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.block_close = block_close
        self.close_error = close_error
        self.ready_error = ready_error
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def ensure_ready(self) -> None:
        if self.ready_error is not None:
            raise self.ready_error

    async def close(self) -> None:
        self.close_started.set()
        if self.block_close:
            await self.release_close.wait()
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


def test_local_and_remote_batch_lifecycle_contracts_match(tmp_path: Path) -> None:
    async def scenario() -> None:
        await _assert_local_lifecycle(tmp_path)
        await _assert_remote_lifecycle(tmp_path)

    asyncio.run(scenario())


def test_owned_core_cleanup_cancellation_finishes_close_and_fails_batch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        await _assert_cleanup_cancellation(tmp_path, source="local")
        await _assert_cleanup_cancellation(tmp_path, source="url")

    asyncio.run(scenario())


def test_owned_core_cleanup_exception_fails_batch_without_completion(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        await _assert_cleanup_exception(tmp_path, source="local")
        await _assert_cleanup_exception(tmp_path, source="url")

    asyncio.run(scenario())


def test_owned_core_cleanup_does_not_replace_primary_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        await _assert_primary_and_cleanup_failure(tmp_path, source="local")
        await _assert_primary_and_cleanup_failure(tmp_path, source="url")

    asyncio.run(scenario())


def test_batch_lifecycle_emits_only_first_terminal_event() -> None:
    events = EventBuffer()
    lifecycle = IngestBatchLifecycle[LocalIngestSuccess](
        event_sink=events,
        namespace="acme",
        collection="help",
        planned_count=1,
        is_success=lambda _record: True,
        error_type=lambda error: type(error).__name__,
    )

    lifecycle.started()
    lifecycle.failed(error=RuntimeError("failed"))
    lifecycle.cancelled(error=asyncio.CancelledError())
    lifecycle.completed()

    assert [event.event_type for event in events.events] == [
        "ingest.batch.started",
        "ingest.batch.failed",
    ]


def test_cancelled_batches_emit_failure_with_partial_counts_for_all_sources(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        await _assert_cancelled_archive_lifecycle(tmp_path)
        await _assert_cancelled_local_lifecycle(tmp_path)
        await _assert_cancelled_remote_lifecycle(tmp_path)

    asyncio.run(scenario())


async def _assert_cleanup_cancellation(
    tmp_path: Path,
    *,
    source: str,
) -> None:
    core = _CleanupCore(block_close=True)
    events = EventBuffer()
    task = asyncio.create_task(
        _run_single_owned_batch(
            tmp_path,
            source=source,
            core=core,
            events=events,
        )
    )
    await asyncio.wait_for(core.close_started.wait(), timeout=1.0)
    task.cancel()
    core.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert core.closed is True
    _assert_failed_terminal(
        events,
        planned_count=1,
        completed_count=1,
        succeeded_count=1,
        error="CancelledError",
    )


async def _assert_cleanup_exception(
    tmp_path: Path,
    *,
    source: str,
) -> None:
    core = _CleanupCore(close_error=RuntimeError("close failed"))
    events = EventBuffer()

    with pytest.raises(RuntimeError, match="close failed"):
        await _run_single_owned_batch(
            tmp_path,
            source=source,
            core=core,
            events=events,
        )

    assert core.closed is False
    _assert_failed_terminal(
        events,
        planned_count=1,
        completed_count=1,
        succeeded_count=1,
        error="RuntimeError",
    )


async def _assert_primary_and_cleanup_failure(
    tmp_path: Path,
    *,
    source: str,
) -> None:
    core = _CleanupCore(
        ready_error=ValueError("primary ingest failure"),
        close_error=RuntimeError("cleanup failure"),
    )
    events = EventBuffer()

    with pytest.raises(ValueError, match="primary ingest failure"):
        await _run_single_owned_batch(
            tmp_path,
            source=source,
            core=core,
            events=events,
        )

    assert core.closed is False
    _assert_failed_terminal(
        events,
        planned_count=1,
        completed_count=0,
        succeeded_count=0,
        error="ValueError",
        has_progress=False,
    )


async def _run_single_owned_batch(
    tmp_path: Path,
    *,
    source: str,
    core: _CleanupCore,
    events: EventBuffer,
) -> object:
    if source == "local":
        path = tmp_path / f"cleanup-{id(core)}.md"
        path.write_text("# Complete before cleanup", encoding="utf-8")
        return await run_local_ingest(
            LocalIngestRequest(
                path=path,
                namespace="acme",
                collection="help",
            ),
            core_factory=lambda: core,
            event_sink=events,
        )

    url_file = tmp_path / f"cleanup-{id(core)}.txt"
    url_file.write_text("https://example.com/complete\n", encoding="utf-8")
    return await run_remote_url_ingest(
        RemoteUrlIngestRequest(
            url_file=url_file,
            namespace="acme",
            collection="help",
        ),
        core_factory=lambda: core,
        event_sink=events,
    )


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
            namespace="acme",
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
        run_local_ingest(
            LocalIngestRequest(
                path=docs,
                namespace="acme",
                collection="help",
                max_concurrency=1,
            ),
            core_factory=lambda: core,
            event_sink=events,
        )
    )
    await asyncio.wait_for(core.second_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert core.closed is True
    _assert_cancelled_lifecycle_events(events)


async def _assert_cancelled_remote_lifecycle(tmp_path: Path) -> None:
    url_file = tmp_path / "cancel-urls.txt"
    url_file.write_text(
        "https://example.com/a\nhttps://example.com/b\n",
        encoding="utf-8",
    )
    core = _CancellationCore()
    events = EventBuffer()
    task = asyncio.create_task(
        run_remote_url_ingest(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="acme",
                collection="help",
                max_concurrency=1,
            ),
            core_factory=lambda: core,
            event_sink=events,
        )
    )
    await asyncio.wait_for(core.second_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert core.closed is True
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
        run_local_ingest(
            LocalIngestRequest(
                path=docs,
                namespace="acme",
                collection="help",
                max_concurrency=3,
            ),
            core_factory=lambda: core,
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


async def _assert_remote_lifecycle(tmp_path: Path) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ]
        ),
        encoding="utf-8",
    )
    core = _OutOfOrderRemoteCore()
    events = EventBuffer()

    task = asyncio.create_task(
        run_remote_url_ingest(
            RemoteUrlIngestRequest(
                url_file=url_file,
                namespace="acme",
                collection="help",
                max_concurrency=3,
            ),
            core_factory=lambda: core,
            event_sink=events,
        )
    )
    try:
        await asyncio.wait_for(core.all_started.wait(), timeout=1.0)
        await _wait_for_progress(events, count=2)
    finally:
        core.release_first.set()
    result = await task

    assert [record.requested_url for record in result.records] == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
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
