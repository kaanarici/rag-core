from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rag_core.core_models import IngestedDocument
from rag_core.events import EventBuffer, IngestBatchCompleted, IngestBatchProgress
from rag_core.local_ingest import LocalIngestRequest, run_local_ingest
from rag_core.remote_ingest import RemoteUrlIngestRequest, run_remote_url_ingest


class _OutOfOrderLocalCore:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.all_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def ensure_ready(self) -> None:
        return None

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
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
            corpus_id=corpus_id,
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

    async def ingest_url(self, url: str, **kwargs: Any) -> IngestedDocument:
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
            corpus_id=kwargs["corpus_id"],
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


def test_local_and_remote_batch_lifecycle_contracts_match(tmp_path: Path) -> None:
    async def scenario() -> None:
        await _assert_local_lifecycle(tmp_path)
        await _assert_remote_lifecycle(tmp_path)

    asyncio.run(scenario())


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
                corpus_id="help",
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
                corpus_id="help",
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
