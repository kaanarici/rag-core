from __future__ import annotations
from typing import NoReturn

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from rag_core.cli.inputs import parse_metadata_fields
from rag_core.safe_messages import error_message
from rag_core.file_io import compute_content_sha256
from rag_core.core_models import CollectionManifestEntry, IngestedDocument
from rag_core.events.sinks import EventBuffer
from rag_core.events.types import (
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchProgress,
    IngestBatchStarted,
)
from rag_core.manifest.persistence import write_entry
from rag_core.ingest.sources.local import document_key as local_document_key
from rag_core.ingest.local import (
    LocalIngestRequest,
    build_local_ingest_plan,
    run_local_ingest_with_core,
)


@dataclass
class _IngestCall:
    file_path: Path
    tenant_id: str
    collection: str
    document_key: str
    metadata: dict[str, str] | None
    force_reindex: bool


class _FakeLocalIngestCore:
    def __init__(
        self,
        *,
        fail_paths: set[str] | None = None,
        ready_error: Exception | None = None,
    ) -> None:
        self.fail_paths = fail_paths or set()
        self.ready_error = ready_error
        self.ensure_ready_calls = 0
        self.closed = False
        self.ingest_calls: list[_IngestCall] = []

    async def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1
        if self.ready_error is not None:
            raise self.ready_error

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
        self.ingest_calls.append(
            _IngestCall(
                file_path=file_path,
                tenant_id=tenant_id,
                collection=collection,
                document_key=document_key,
                metadata=metadata,
                force_reindex=force_reindex,
            )
        )
        if str(file_path) in self.fail_paths:
            raise ValueError("failed to parse")
        return IngestedDocument(
            document_id=f"doc-{file_path.stem}",
            collection=collection,
            tenant_id=tenant_id,
            chunk_count=1,
            filename=file_path.name,
            mime_type="text/plain",
            document_key=document_key,
            content_sha256=compute_content_sha256(pre_read_bytes)
            if pre_read_bytes is not None
            else None,
            ingest_state="created",
        )

    async def add_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
    ) -> IngestedDocument:
        file_path = Path(path or filename)
        self.ingest_calls.append(
            _IngestCall(
                file_path=file_path,
                tenant_id=tenant_id,
                collection=collection,
                document_key=document_key or filename,
                metadata=metadata,
                force_reindex=force_reindex,
            )
        )
        if str(file_path) in self.fail_paths:
            raise ValueError("failed to parse")
        return IngestedDocument(
            document_id=document_id or f"doc-{file_path.stem}",
            collection=collection,
            tenant_id=tenant_id,
            chunk_count=1,
            filename=filename,
            mime_type=mime_type,
            document_key=document_key or filename,
            content_sha256=compute_content_sha256(file_bytes),
            ingest_state="created",
        )

    async def close(self) -> None:
        self.closed = True

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")


class _FakeOpenAIError(Exception):
    __module__ = "openai"


class _BootstrapErrorLocalIngestCore(_FakeLocalIngestCore):
    def __init__(self, *, fail_path: Path) -> None:
        super().__init__()
        self.fail_path = fail_path

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
        if file_path == self.fail_path:
            raise _FakeOpenAIError("raw api_key client option OPENAI_API_KEY secret")
        return await super().add_file(
            file_path,
            tenant_id=tenant_id,
            collection=collection,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
            pre_read_bytes=pre_read_bytes,
        )

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")


class _SiblingAbortLocalIngestCore(_FakeLocalIngestCore):
    def __init__(self, *, fail_path: Path) -> None:
        super().__init__()
        self.fail_path = fail_path
        self.active = 0
        self.closed_while_active = False
        self.sibling_started = asyncio.Event()
        self.release_sibling = asyncio.Event()

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
        self.active += 1
        try:
            if file_path == self.fail_path:
                await self.sibling_started.wait()
                raise _FakeOpenAIError(
                    "raw api_key client option OPENAI_API_KEY secret"
                )
            self.sibling_started.set()
            await self.release_sibling.wait()
            return await super().add_file(
                file_path,
                tenant_id=tenant_id,
                collection=collection,
                document_key=document_key,
                metadata=metadata,
                force_reindex=force_reindex,
                pre_read_bytes=pre_read_bytes,
            )
        finally:
            self.active -= 1

    async def close(self) -> None:
        self.closed_while_active = self.active > 0
        self.release_sibling.set()
        await super().close()

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")


class _GatedLocalIngestCore(_FakeLocalIngestCore):
    def __init__(self) -> None:
        super().__init__()
        self.started: list[Path] = []
        self.two_started = asyncio.Event()
        self.release = asyncio.Event()

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
        self.started.append(file_path)
        if len(self.started) == 2:
            self.two_started.set()
        await self.release.wait()
        return await super().add_file(
            file_path,
            tenant_id=tenant_id,
            collection=collection,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
            pre_read_bytes=pre_read_bytes,
        )

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")


class _DelayedFirstLocalIngestCore(_FakeLocalIngestCore):
    def __init__(self) -> None:
        super().__init__()
        self.started: list[Path] = []
        self.all_started = asyncio.Event()
        self.release_first = asyncio.Event()

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
        self.started.append(file_path)
        if len(self.started) == 3:
            self.all_started.set()
        if file_path.name == "a.md":
            await self.release_first.wait()
        return await super().add_file(
            file_path,
            tenant_id=tenant_id,
            collection=collection,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
            pre_read_bytes=pre_read_bytes,
        )

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")


def test_build_local_ingest_plan_truncates_supported_files_when_max_files_set(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.txt"
    first.write_text("a", encoding="utf-8")
    (docs / "b.txt").write_text("b", encoding="utf-8")

    plan = build_local_ingest_plan(
        LocalIngestRequest(
            path=docs,
            tenant_id="acme",
            collection="help",
            max_files=1,
        )
    )

    assert plan.truncated is True
    assert [item.path for item in plan.documents] == [first]
    assert [item.document_key for item in plan.documents] == [
        local_document_key(docs, first)
    ]


def test_build_local_ingest_plan_expands_paths_with_portable_keys_and_hashes(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.md"
    first.write_text("a", encoding="utf-8")
    second = docs / "nested" / "b.txt"
    second.parent.mkdir()
    second.write_text("b", encoding="utf-8")
    (docs / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    plan = build_local_ingest_plan(
        LocalIngestRequest(path=docs, tenant_id="acme", collection="help")
    )

    assert plan.document_count == 2
    first_hash = compute_content_sha256(b"a")
    second_hash = compute_content_sha256(b"b")
    assert [
        (item.path, item.document_key, item.content_sha256, item.source_error)
        for item in plan.documents
    ] == [
        (first, local_document_key(docs, first), first_hash, ""),
        (second, local_document_key(docs, second), second_hash, ""),
    ]


def test_build_local_ingest_plan_uses_file_and_glob_key_roots(tmp_path) -> None:
    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    file_path = docs / "single.md"
    file_path.write_text("single", encoding="utf-8")
    nested_file = nested / "item.md"
    nested_file.write_text("nested", encoding="utf-8")

    file_plan = build_local_ingest_plan(
        LocalIngestRequest(path=file_path, tenant_id="acme", collection="help")
    )
    glob_plan = build_local_ingest_plan(
        LocalIngestRequest(
            path=str(docs / "**" / "*.md"),
            tenant_id="acme",
            collection="help",
        )
    )

    assert [item.document_key for item in file_plan.documents] == [
        local_document_key(docs, file_path)
    ]
    assert [item.document_key for item in glob_plan.documents] == [
        local_document_key(docs, nested_file),
        local_document_key(docs, file_path),
    ]


def test_build_local_ingest_plan_rejects_hardlinked_file_path(tmp_path: Path) -> None:
    if not hasattr(os, "link"):
        pytest.skip("hardlinks are unavailable on this platform")
    source = tmp_path / "source.md"
    source.write_text("secret", encoding="utf-8")
    alias = tmp_path / "alias.md"
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink support unavailable: {exc}")

    with pytest.raises(ValueError, match="does not allow multi-link"):
        build_local_ingest_plan(
            LocalIngestRequest(path=alias, tenant_id="acme", collection="help")
        )


def test_build_local_ingest_plan_rejects_empty_supported_set_before_core_creation(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="no supported files matched"):
        build_local_ingest_plan(
            LocalIngestRequest(path=docs, tenant_id="acme", collection="help")
        )


def test_build_local_ingest_plan_rejects_invalid_concurrency(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide", encoding="utf-8")

    with pytest.raises(ValueError, match="max_concurrency"):
        build_local_ingest_plan(
            LocalIngestRequest(
                path=docs,
                tenant_id="acme",
                collection="help",
                max_concurrency=0,
            )
        )


def test_run_local_ingest_reports_partial_failures_and_closes_core(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    good = docs / "billing.md"
    good.write_text("Invoices are monthly.", encoding="utf-8")
    bad = docs / "broken.txt"
    bad.write_text("broken", encoding="utf-8")

    core = _FakeLocalIngestCore(fail_paths={str(bad)})
    buffer = EventBuffer()
    good_hash = compute_content_sha256(b"Invoices are monthly.")
    bad_hash = compute_content_sha256(b"broken")
    good_key = local_document_key(docs, good)
    bad_key = local_document_key(docs, bad)
    manifest_dir = tmp_path / "manifest"
    write_entry(
        manifest_dir,
        CollectionManifestEntry(
            document_id="doc-billing",
            tenant_id="acme",
            collection="help",
            document_key=good_key,
            content_sha256=good_hash,
            filename="billing.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    )
    write_entry(
        manifest_dir,
        CollectionManifestEntry(
            document_id="doc-broken",
            tenant_id="acme",
            collection="help",
            document_key=bad_key,
            content_sha256="old-hash",
            filename="broken.txt",
            mime_type="text/plain",
            chunk_count=1,
        ),
    )

    result = asyncio.run(
        run_local_ingest_with_core(
            LocalIngestRequest(
                path=docs,
                tenant_id="acme",
                collection="help",
                metadata={"source": "seed"},
                force_reindex=True,
            ),
            core=core,
            event_sink=buffer,
            manifest_dir=manifest_dir,
        )
    )

    payload = result.to_payload(include_private=True)
    records = cast(list[dict[str, object]], payload["records"])
    assert payload["planned_count"] == 2
    assert payload["written_count"] == 1
    assert payload["failed_count"] == 1
    assert records == [
        {
            "ok": True,
            "path": str(good),
            "document_key": good_key,
            "content_sha256": good_hash,
            "document_id": "doc-billing",
            "filename": "billing.md",
            "chunk_count": 1,
            "ingest_state": "created",
            "replaced_existing": False,
        },
        {
            "ok": False,
            "path": str(bad),
            "document_key": bad_key,
            "content_sha256": bad_hash,
            "error": "failed to parse",
        },
    ]
    assert core.ensure_ready_calls == 1
    assert core.ingest_calls == [
        _IngestCall(
            file_path=good,
            tenant_id="acme",
            collection="help",
            document_key=good_key,
            metadata={"source": "seed"},
            force_reindex=True,
        ),
        _IngestCall(
            file_path=bad,
            tenant_id="acme",
            collection="help",
            document_key=bad_key,
            metadata={"source": "seed"},
            force_reindex=True,
        ),
    ]
    assert core.closed is False
    started = [
        event for event in buffer.events if isinstance(event, IngestBatchStarted)
    ]
    progress = [
        event for event in buffer.events if isinstance(event, IngestBatchProgress)
    ]
    completed = [
        event for event in buffer.events if isinstance(event, IngestBatchCompleted)
    ]
    assert len(started) == 1
    assert (started[0].tenant_id, started[0].collection, started[0].planned_count) == (
        "acme",
        "help",
        2,
    )
    assert [(e.filename, e.status, e.content_sha256) for e in progress] == [
        ("billing.md", "succeeded", good_hash),
        ("broken.txt", "failed", bad_hash),
    ]
    assert [
        (e.current_index, e.completed_count, e.succeeded_count, e.failed_count)
        for e in progress
    ] == [(1, 1, 1, 0), (2, 2, 1, 1)]
    assert progress[1].error == "ValueError"
    assert "failed to parse" not in str(progress[1])
    assert len(completed) == 1
    assert completed[0].planned_count == 2
    assert completed[0].succeeded_count == 1
    assert completed[0].failed_count == 1
    assert completed[0].duration_ms >= 0.0


def test_run_local_ingest_respects_max_concurrency_and_preserves_record_order(
    tmp_path,
) -> None:
    asyncio.run(_run_local_ingest_respects_max_concurrency(tmp_path))


async def _run_local_ingest_respects_max_concurrency(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.md"
    second = docs / "b.md"
    third = docs / "c.md"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    third.write_text("c", encoding="utf-8")
    core = _GatedLocalIngestCore()

    task = asyncio.create_task(
        run_local_ingest_with_core(
            LocalIngestRequest(
                path=docs,
                tenant_id="acme",
                collection="help",
                max_concurrency=2,
            ),
            core=core,
        )
    )
    try:
        await asyncio.wait_for(core.two_started.wait(), timeout=1.0)
    except Exception:
        core.release.set()
        await task
        raise
    assert sorted(path.name for path in core.started) == ["a.md", "b.md"]

    core.release.set()
    result = await task

    assert [record.document_key for record in result.records] == [
        local_document_key(docs, first),
        local_document_key(docs, second),
        local_document_key(docs, third),
    ]
    assert result.written_count == 3
    assert core.closed is False


def test_run_local_ingest_keeps_progress_counts_monotonic_with_out_of_order_completion(
    tmp_path,
) -> None:
    asyncio.run(_run_local_ingest_tracks_out_of_order_progress(tmp_path))


async def _run_local_ingest_tracks_out_of_order_progress(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    paths = [docs / name for name in ("a.md", "b.md", "c.md")]
    for path in paths:
        path.write_text(path.name, encoding="utf-8")
    core = _DelayedFirstLocalIngestCore()
    buffer = EventBuffer()

    task = asyncio.create_task(
        run_local_ingest_with_core(
            LocalIngestRequest(
                path=docs,
                tenant_id="acme",
                collection="help",
                max_concurrency=3,
            ),
            core=core,
            event_sink=buffer,
        )
    )
    try:
        await asyncio.wait_for(core.all_started.wait(), timeout=1.0)
        await _wait_for_ingest_progress(buffer, count=2)
        early_progress = [
            event for event in buffer.events if isinstance(event, IngestBatchProgress)
        ]
        assert {event.filename for event in early_progress} == {"b.md", "c.md"}
        core.release_first.set()
        result = await task
    except Exception:
        core.release_first.set()
        await task
        raise

    progress = [
        event for event in buffer.events if isinstance(event, IngestBatchProgress)
    ]
    assert [event.filename for event in progress] != ["a.md", "b.md", "c.md"]
    assert [event.current_index for event in progress] == [1, 2, 3]
    assert [event.completed_count for event in progress] == [1, 2, 3]
    assert [record.document_key for record in result.records] == [
        *(local_document_key(docs, path) for path in paths),
    ]


async def _wait_for_ingest_progress(buffer: EventBuffer, *, count: int) -> None:
    for _ in range(100):
        progress = [
            event for event in buffer.events if isinstance(event, IngestBatchProgress)
        ]
        if len(progress) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected at least {count} progress events")


def test_run_local_ingest_returns_all_failures_without_raising(tmp_path) -> None:
    file_path = tmp_path / "broken.txt"
    file_path.write_text("broken", encoding="utf-8")
    core = _FakeLocalIngestCore(fail_paths={str(file_path)})
    buffer = EventBuffer()

    result = asyncio.run(
        run_local_ingest_with_core(
            LocalIngestRequest(path=file_path, tenant_id="acme", collection="help"),
            core=core,
            event_sink=buffer,
        )
    )

    assert result.planned_count == 1
    assert result.written_count == 0
    assert result.failed_count == 1
    assert result.failed[0].document_key == local_document_key(
        file_path.parent, file_path
    )
    assert result.failed[0].content_sha256 == compute_content_sha256(b"broken")
    assert core.closed is False
    completed = [
        event for event in buffer.events if isinstance(event, IngestBatchCompleted)
    ]
    assert completed[0].succeeded_count == 0
    assert completed[0].failed_count == 1


def test_single_file_ingest_reads_fresh_bytes_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_core.ingest.local as runner_module

    file_path = tmp_path / "guide.txt"
    file_path.write_text("guide", encoding="utf-8")
    core = _FakeLocalIngestCore()
    read_paths: list[Path] = []

    async def read_once(path: Path, *, max_bytes: int | None = None) -> bytes:
        read_paths.append(path)
        return path.read_bytes()

    monkeypatch.setattr(runner_module, "read_file_bytes", read_once)

    result = asyncio.run(
        run_local_ingest_with_core(
            LocalIngestRequest(path=file_path, tenant_id="acme", collection="help"),
            core=core,
        )
    )

    assert read_paths == [file_path]
    assert result.written[0].content_sha256 == compute_content_sha256(b"guide")
    assert core.ingest_calls == [
        _IngestCall(
            file_path=file_path,
            tenant_id="acme",
            collection="help",
            document_key=local_document_key(file_path.parent, file_path),
            metadata=None,
            force_reindex=False,
        )
    ]


def test_local_ingest_source_read_failures_do_not_reach_the_core(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_core.ingest.local as runner_module
    import rag_core.ingest.sources.local as sources_module

    file_path = tmp_path / "blocked.txt"
    file_path.write_text("blocked", encoding="utf-8")
    manifest_dir = tmp_path / "manifest"
    document_key = local_document_key(file_path.parent, file_path)
    write_entry(
        manifest_dir,
        CollectionManifestEntry(
            document_id="doc-blocked",
            tenant_id="acme",
            collection="help",
            document_key=document_key,
            content_sha256=compute_content_sha256(b"blocked"),
            filename="blocked.txt",
            mime_type="text/plain",
            chunk_count=1,
        ),
    )

    def fail_hash(path: Path) -> str:
        if path == file_path:
            raise PermissionError("blocked")
        return compute_content_sha256(path.read_bytes())

    monkeypatch.setattr(sources_module, "file_content_sha256", fail_hash)
    plan = build_local_ingest_plan(
        LocalIngestRequest(path=file_path, tenant_id="acme", collection="help")
    )
    assert plan.documents[0].content_sha256 is None
    assert plan.documents[0].source_error == "blocked"


    async def fail_read(path: Path, *, max_bytes: int | None = None) -> bytes:
        if path == file_path:
            raise PermissionError("blocked")
        return path.read_bytes()

    monkeypatch.setattr(runner_module, "read_file_bytes", fail_read)

    core = _FakeLocalIngestCore()
    buffer = EventBuffer()
    result = asyncio.run(
        run_local_ingest_with_core(
            LocalIngestRequest(path=file_path, tenant_id="acme", collection="help"),
            core=core,
            event_sink=buffer,
            manifest_dir=manifest_dir,
        )
    )

    assert result.failed[0].content_sha256 is None
    assert core.ingest_calls == []
    progress = [
        event for event in buffer.events if isinstance(event, IngestBatchProgress)
    ]
    assert progress[0].content_sha256 == ""


def test_run_local_ingest_success_uses_actual_ingested_hash(tmp_path) -> None:
    file_path = tmp_path / "changed.txt"
    file_path.write_text("old", encoding="utf-8")
    old_hash = compute_content_sha256(b"old")
    new_hash = compute_content_sha256(b"new")
    manifest_dir = tmp_path / "manifest"
    document_key = local_document_key(file_path.parent, file_path)
    write_entry(
        manifest_dir,
        CollectionManifestEntry(
            document_id="doc-changed",
            tenant_id="acme",
            collection="help",
            document_key=document_key,
            content_sha256=old_hash,
            filename="changed.txt",
            mime_type="text/plain",
            chunk_count=1,
        ),
    )

    class _MutatingCore:
        def __init__(self) -> None:
            self.closed = False

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
            file_path.write_text("new", encoding="utf-8")
            return IngestedDocument(
                document_id="doc-changed",
                collection=collection,
                tenant_id=tenant_id,
                chunk_count=1,
                filename=file_path.name,
                mime_type="text/plain",
                document_key=document_key,
                content_sha256=new_hash,
                ingest_state="replaced",
            )

        async def close(self) -> None:
            self.closed = True

        async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
            raise AssertionError("prune_missing is disabled in this test")

    core = _MutatingCore()
    buffer = EventBuffer()

    result = asyncio.run(
        run_local_ingest_with_core(
            LocalIngestRequest(path=file_path, tenant_id="acme", collection="help"),
            core=core,
            event_sink=buffer,
            manifest_dir=manifest_dir,
        )
    )

    assert result.written[0].content_sha256 == new_hash
    progress = [
        event for event in buffer.events if isinstance(event, IngestBatchProgress)
    ]
    assert progress[0].content_sha256 == new_hash
    assert core.closed is False


def test_run_local_ingest_emits_batch_failed_when_setup_fails(tmp_path) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("billing", encoding="utf-8")
    core = _FakeLocalIngestCore(ready_error=RuntimeError("vector store unavailable"))
    buffer = EventBuffer()

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        asyncio.run(
            run_local_ingest_with_core(
                LocalIngestRequest(path=file_path, tenant_id="acme", collection="help"),
                core=core,
                event_sink=buffer,
            )
        )

    assert core.ensure_ready_calls == 1
    assert core.ingest_calls == []
    assert core.closed is False
    assert [event.event_type for event in buffer.events] == [
        "ingest.batch.started",
        "ingest.batch.failed",
    ]
    failed = buffer.events[1]
    assert isinstance(failed, IngestBatchFailed)
    assert failed.tenant_id == "acme"
    assert failed.collection == "help"
    assert failed.planned_count == 1
    assert failed.completed_count == 0
    assert failed.succeeded_count == 0
    assert failed.failed_count == 0
    assert failed.error == "RuntimeError"
    assert "vector store unavailable" not in str(failed)
    assert failed.duration_ms >= 0.0


def test_run_local_ingest_batch_failed_preserves_partial_provider_abort_counts(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.md"
    first.write_text("alpha", encoding="utf-8")
    second = docs / "b.md"
    second.write_text("beta", encoding="utf-8")
    core = _BootstrapErrorLocalIngestCore(fail_path=second)
    buffer = EventBuffer()

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            run_local_ingest_with_core(
                LocalIngestRequest(
                    path=docs,
                    tenant_id="acme",
                    collection="help",
                    max_concurrency=1,
                ),
                core=core,
                event_sink=buffer,
            )
        )

    assert "provider failed during ingest" in str(exc_info.value)
    assert "OPENAI_API_KEY" not in str(exc_info.value)
    progress = [
        event for event in buffer.events if isinstance(event, IngestBatchProgress)
    ]
    assert len(progress) == 1
    assert progress[0].status == "succeeded"
    failed = [event for event in buffer.events if isinstance(event, IngestBatchFailed)]
    assert len(failed) == 1
    assert failed[0].planned_count == 2
    assert failed[0].completed_count == 1
    assert failed[0].succeeded_count == 1
    assert failed[0].failed_count == 0
    assert failed[0].error == "ProviderCliError"
    assert "OPENAI_API_KEY" not in str(failed[0])
    assert core.closed is False


def test_run_local_ingest_drains_sibling_workers_before_close(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.md"
    first.write_text("alpha", encoding="utf-8")
    second = docs / "b.md"
    second.write_text("beta", encoding="utf-8")
    core = _SiblingAbortLocalIngestCore(fail_path=first)

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            run_local_ingest_with_core(
                LocalIngestRequest(
                    path=docs,
                    tenant_id="acme",
                    collection="help",
                    max_concurrency=2,
                ),
                core=core,
            )
        )

    assert "provider failed during ingest" in str(exc_info.value)
    assert core.closed is False
    assert core.closed_while_active is False


def test_parse_metadata_fields_requires_key_value() -> None:
    assert parse_metadata_fields(["source=seed", "owner=rag"]) == {
        "source": "seed",
        "owner": "rag",
    }

    with pytest.raises(ValueError, match="metadata entries must use KEY=VALUE"):
        parse_metadata_fields(["broken"])


def test_cli_error_message_for_file_not_found() -> None:
    assert "file not found" in error_message(FileNotFoundError("missing.txt"))


def test_folder_ingest_reads_each_file_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_core.ingest.local as runner_module
    import rag_core.ingest.sources.local as sources_module

    docs = tmp_path / "corpus"
    docs.mkdir()
    file_a = docs / "a.txt"
    file_b = docs / "b.txt"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")

    read_paths: list[Path] = []
    hash_paths: list[Path] = []

    original_read = runner_module.read_file_bytes
    original_hash = sources_module.file_content_sha256

    async def counting_read(path: Path, *, max_bytes: int | None = None) -> bytes:
        read_paths.append(path)
        return await original_read(path, max_bytes=max_bytes)

    def counting_hash(path: Path) -> str:
        hash_paths.append(path)
        return original_hash(path)

    monkeypatch.setattr(runner_module, "read_file_bytes", counting_read)
    monkeypatch.setattr(sources_module, "file_content_sha256", counting_hash)

    core = _FakeLocalIngestCore()
    result = asyncio.run(
        run_local_ingest_with_core(
            LocalIngestRequest(path=docs, tenant_id="acme", collection="help"),
            core=core,
        )
    )

    assert result.written_count == 2

    # Discovery only scans names; the worker reads the bytes once.
    assert hash_paths == []
    # Runner read: exactly one read call per file. No second read inside ingest_file.
    assert sorted(read_paths) == sorted([file_a, file_b])


def test_single_file_changed_after_discovery_uses_fresh_bytes(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"old contents")
    received: list[bytes | None] = []

    class ChangingCore(_FakeLocalIngestCore):
        async def ensure_ready(self) -> None:
            path.write_bytes(b"fresh contents")
            await super().ensure_ready()

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
            received.append(pre_read_bytes)
            return await super().add_file(
                file_path,
                tenant_id=tenant_id,
                collection=collection,
                document_key=document_key,
                metadata=metadata,
                force_reindex=force_reindex,
                pre_read_bytes=pre_read_bytes,
            )

        async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
            raise AssertionError("prune_missing is disabled in this test")

    result = asyncio.run(
        run_local_ingest_with_core(
            LocalIngestRequest(path=path, tenant_id="n", collection="c"),
            core=ChangingCore(),
        )
    )
    assert received == [b"fresh contents"]
    assert result.written[0].content_sha256 == compute_content_sha256(b"fresh contents")
