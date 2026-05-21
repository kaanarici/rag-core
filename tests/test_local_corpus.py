from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from rag_core.cli_inputs import cli_error_message, parse_metadata_fields
from rag_core.core_lifecycle import compute_content_sha256
from rag_core.core_models import CorpusManifestEntry, IngestedDocument
from rag_core.events.sinks import EventBuffer
from rag_core.events.types import (
    IngestBatchCompleted,
    IngestBatchFailed,
    IngestBatchProgress,
    IngestBatchStarted,
)
from rag_core.manifest_persistence import ManifestSource, write_entry
from rag_core.sources import document_key as local_document_key
from rag_core.local_corpus import (
    LocalIngestRequest,
    LocalSearchRequest,
    ManifestPreviewRequest,
    build_local_ingest_plan,
    preview_manifest,
    reconcile_local_ingest_plan,
    run_local_ingest,
    run_local_search,
)
from rag_core.search.types import SearchResult
from tests.support import make_search_result


@dataclass
class _SearchCall:
    query: str
    namespace: str
    corpus_ids: list[str]
    limit: int
    rerank: bool


@dataclass
class _IngestCall:
    file_path: Path
    namespace: str
    corpus_id: str
    document_key: str
    metadata: dict[str, str] | None
    force_reindex: bool


class _FakeLocalSearchCore:
    def __init__(self, *, fail_paths: set[str] | None = None) -> None:
        self.fail_paths = fail_paths or set()
        self.ensure_ready_calls = 0
        self.closed = False
        self.ingest_calls: list[tuple[Path, str]] = []
        self.search_calls: list[_SearchCall] = []

    async def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
    ) -> IngestedDocument:
        self.ingest_calls.append((file_path, document_key))
        if str(file_path) in self.fail_paths:
            raise ValueError("failed to parse")
        return IngestedDocument(
            document_id=f"doc-{file_path.stem}",
            corpus_id=corpus_id,
            namespace=namespace,
            chunk_count=1,
            filename=file_path.name,
            mime_type="text/plain",
            document_key=document_key,
        )

    async def search(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int,
        rerank: bool,
    ) -> list[SearchResult]:
        self.search_calls.append(
            _SearchCall(
                query=query,
                namespace=namespace,
                corpus_ids=corpus_ids,
                limit=limit,
                rerank=rerank,
            )
        )
        return [
            make_search_result(
                text="Invoices can be paid monthly.",
                document_id="doc-billing",
                document_key="billing.md",
            )
        ]

    async def close(self) -> None:
        self.closed = True


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

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> IngestedDocument:
        self.ingest_calls.append(
            _IngestCall(
                file_path=file_path,
                namespace=namespace,
                corpus_id=corpus_id,
                document_key=document_key,
                metadata=metadata,
                force_reindex=force_reindex,
            )
        )
        if str(file_path) in self.fail_paths:
            raise ValueError("failed to parse")
        return IngestedDocument(
            document_id=f"doc-{file_path.stem}",
            corpus_id=corpus_id,
            namespace=namespace,
            chunk_count=1,
            filename=file_path.name,
            mime_type="text/plain",
            document_key=document_key,
            ingest_state="created",
        )

    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
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
                namespace=namespace,
                corpus_id=corpus_id,
                document_key=document_key or filename,
                metadata=metadata,
                force_reindex=force_reindex,
            )
        )
        if str(file_path) in self.fail_paths:
            raise ValueError("failed to parse")
        return IngestedDocument(
            document_id=document_id or f"doc-{file_path.stem}",
            corpus_id=corpus_id,
            namespace=namespace,
            chunk_count=1,
            filename=filename,
            mime_type=mime_type,
            document_key=document_key or filename,
            content_sha256=compute_content_sha256(file_bytes),
            ingest_state="created",
        )

    async def close(self) -> None:
        self.closed = True


class _FakeOpenAIError(Exception):
    __module__ = "openai"


class _BootstrapErrorLocalIngestCore(_FakeLocalIngestCore):
    def __init__(self, *, fail_path: Path) -> None:
        super().__init__()
        self.fail_path = fail_path

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> IngestedDocument:
        if file_path == self.fail_path:
            raise _FakeOpenAIError("raw api_key client option OPENAI_API_KEY secret")
        return await super().ingest_file(
            file_path,
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
        )


class _GatedLocalIngestCore(_FakeLocalIngestCore):
    def __init__(self) -> None:
        super().__init__()
        self.started: list[Path] = []
        self.two_started = asyncio.Event()
        self.release = asyncio.Event()

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> IngestedDocument:
        self.started.append(file_path)
        if len(self.started) == 2:
            self.two_started.set()
        await self.release.wait()
        return await super().ingest_file(
            file_path,
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
        )


class _DelayedFirstLocalIngestCore(_FakeLocalIngestCore):
    def __init__(self) -> None:
        super().__init__()
        self.started: list[Path] = []
        self.all_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def ingest_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> IngestedDocument:
        self.started.append(file_path)
        if len(self.started) == 3:
            self.all_started.set()
        if file_path.name == "a.md":
            await self.release_first.wait()
        return await super().ingest_file(
            file_path,
            namespace=namespace,
            corpus_id=corpus_id,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
        )


def test_run_local_search_applies_file_policy_document_keys_and_payload(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    readme = docs / "readme.txt"
    readme.write_text("rag retrieval smoke text", encoding="utf-8")
    nested = docs / "faq" / "billing.md"
    nested.parent.mkdir()
    nested.write_text("Invoices are monthly.", encoding="utf-8")
    (docs / "__init__.py").write_text("", encoding="utf-8")
    pycache = docs / "__pycache__"
    pycache.mkdir()
    (pycache / "readme.cpython-314.pyc").write_bytes(b"generated")
    (docs / "~$draft.docx").write_bytes(b"not a real docx")
    (docs / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    core = _FakeLocalSearchCore()

    result = asyncio.run(
        run_local_search(
            LocalSearchRequest(path=docs, query="retrieval", max_files=200),
            core_factory=lambda: core,
        )
    )

    payload = result.to_payload()
    indexed = cast(list[dict[str, object]], payload["indexed"])
    assert [item["document_key"] for item in indexed] == [
        local_document_key(docs, nested),
        local_document_key(docs, readme),
    ]
    assert payload["corpus_id"] == "docs"
    assert payload["indexed_count"] == 2
    assert payload["skipped_count"] == 2
    assert payload["skipped_empty_count"] == 0
    assert payload["skipped_unsupported_count"] == 2
    assert payload["skipped_failed"] == []
    assert payload["truncated"] is False
    assert core.ingest_calls == [
        (nested, local_document_key(docs, nested)),
        (readme, local_document_key(docs, readme)),
    ]
    assert core.search_calls[0] == _SearchCall(
        query="retrieval",
        namespace="local",
        corpus_ids=["docs"],
        limit=5,
        rerank=False,
    )


def test_run_local_search_truncates_supported_files_before_indexing(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.txt"
    first.write_text("a", encoding="utf-8")
    (docs / "b.txt").write_text("b", encoding="utf-8")

    core = _FakeLocalSearchCore()

    result = asyncio.run(
        run_local_search(
            LocalSearchRequest(path=docs, query="a", max_files=1),
            core_factory=lambda: core,
        )
    )

    payload = result.to_payload()
    assert payload["indexed_count"] == 1
    assert payload["truncated"] is True
    assert core.ingest_calls == [(first, local_document_key(docs, first))]


def test_run_local_search_uses_single_file_document_key_and_corpus_id(tmp_path) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("guide", encoding="utf-8")

    core = _FakeLocalSearchCore()

    result = asyncio.run(
        run_local_search(
            LocalSearchRequest(path=file_path, query="guide"),
            core_factory=lambda: core,
        )
    )

    assert result.to_payload()["corpus_id"] == "guide"
    assert core.ingest_calls == [
        (file_path, local_document_key(file_path.parent, file_path))
    ]


def test_run_local_search_rejects_empty_supported_set_before_core_creation(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    created_core = False

    def core_factory() -> _FakeLocalSearchCore:
        nonlocal created_core
        created_core = True
        return _FakeLocalSearchCore()

    with pytest.raises(ValueError, match="no supported files found"):
        asyncio.run(
            run_local_search(
                LocalSearchRequest(path=docs, query="scan"), core_factory=core_factory
            )
        )

    assert created_core is False


def test_run_local_search_reports_only_empty_supported_files_before_core_creation(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "empty.txt").write_text("", encoding="utf-8")
    created_core = False

    def core_factory() -> _FakeLocalSearchCore:
        nonlocal created_core
        created_core = True
        return _FakeLocalSearchCore()

    with pytest.raises(ValueError, match="only empty supported files found"):
        asyncio.run(
            run_local_search(
                LocalSearchRequest(path=docs, query="empty"), core_factory=core_factory
            )
        )

    assert created_core is False


def test_run_local_search_indexes_files_reports_failed_and_returns_hits(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    good = docs / "billing.md"
    good.write_text("Invoices are monthly.", encoding="utf-8")
    bad = docs / "broken.txt"
    bad.write_text("broken", encoding="utf-8")

    core = _FakeLocalSearchCore(fail_paths={str(bad)})

    result = asyncio.run(
        run_local_search(
            LocalSearchRequest(path=docs, query="invoices", namespace="local", limit=3),
            core_factory=lambda: core,
        )
    )

    payload = result.to_payload()
    hits = cast(list[dict[str, object]], payload["hits"])
    skipped_failed = cast(list[dict[str, str]], payload["skipped_failed"])
    assert payload["indexed_count"] == 1
    assert payload["skipped_count"] == 1
    assert skipped_failed == [{"path": str(bad), "error": "failed to parse"}]
    assert hits[0]["document_key"] == "billing.md"
    assert core.search_calls[0].corpus_ids == ["docs"]
    assert core.search_calls[0].rerank is False
    assert core.closed is True


def test_run_local_search_raises_when_every_ingest_fails(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    file_path = docs / "broken.txt"
    file_path.write_text("broken", encoding="utf-8")

    core = _FakeLocalSearchCore(fail_paths={str(file_path)})

    with pytest.raises(ValueError, match="no files could be indexed"):
        asyncio.run(
            run_local_search(
                LocalSearchRequest(path=docs, query="invoices"),
                core_factory=lambda: core,
            )
        )

    assert core.closed is True


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
        LocalIngestRequest(path=docs, namespace="acme", corpus_id="help")
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
    assert plan.manifest_sources == (
        ManifestSource(document_key=local_document_key(docs, first), content_sha256=first_hash),
        ManifestSource(document_key=local_document_key(docs, second), content_sha256=second_hash),
    )
    assert plan.to_payload()["documents"] == [
        {
            "path": "<local-file>",
            "filename": "a.md",
            "content_sha256_available": True,
            "source_error": "",
        },
        {
            "path": "<local-file>",
            "filename": "b.txt",
            "content_sha256_available": True,
            "source_error": "",
        },
    ]
    assert plan.to_payload(include_private=True)["documents"] == [
        {
            "path": str(first),
            "document_key": local_document_key(docs, first),
            "content_sha256": first_hash,
            "source_error": "",
        },
        {
            "path": str(second),
            "document_key": local_document_key(docs, second),
            "content_sha256": second_hash,
            "source_error": "",
        },
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
        LocalIngestRequest(path=file_path, namespace="acme", corpus_id="help")
    )
    glob_plan = build_local_ingest_plan(
        LocalIngestRequest(
            path=str(docs / "**" / "*.md"),
            namespace="acme",
            corpus_id="help",
        )
    )

    assert [item.document_key for item in file_plan.documents] == [
        local_document_key(docs, file_path)
    ]
    assert [item.document_key for item in glob_plan.documents] == [
        local_document_key(docs, nested_file),
        local_document_key(docs, file_path),
    ]


def test_reconcile_local_ingest_plan_uses_content_hashes(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    unchanged = docs / "unchanged.md"
    unchanged.write_text("same", encoding="utf-8")
    changed = docs / "changed.md"
    changed.write_text("new", encoding="utf-8")
    unchanged_hash = compute_content_sha256(b"same")
    changed_hash = compute_content_sha256(b"new")
    manifest_dir = tmp_path / "manifest"
    plan = build_local_ingest_plan(
        LocalIngestRequest(path=docs, namespace="acme", corpus_id="help")
    )
    unchanged_key = local_document_key(docs, unchanged)
    changed_key = local_document_key(docs, changed)

    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id="doc-unchanged",
            namespace="acme",
            corpus_id="help",
            document_key=unchanged_key,
            content_sha256=unchanged_hash,
            filename="unchanged.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    )
    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id="doc-changed",
            namespace="acme",
            corpus_id="help",
            document_key=changed_key,
            content_sha256="old-hash",
            filename="changed.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    )
    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id="doc-orphan",
            namespace="acme",
            corpus_id="help",
            document_key="removed.md",
            content_sha256="orphan-hash",
            filename="removed.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    )

    reconciliation = reconcile_local_ingest_plan(plan, manifest_dir=manifest_dir)

    assert [(item.document_key, item.content_sha256) for item in plan.manifest_sources] == [
        (changed_key, changed_hash),
        (unchanged_key, unchanged_hash),
    ]
    assert [(item.document_key, item.reason) for item in reconciliation.changed] == [
        (changed_key, "content_sha256_changed")
    ]
    assert [(item.document_key, item.reason) for item in reconciliation.unchanged] == [
        (unchanged_key, "content_sha256_match")
    ]
    assert [(item.document_key, item.reason) for item in reconciliation.orphaned] == [
        ("removed.md", "manifest_entry_without_source")
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
            LocalIngestRequest(path=alias, namespace="acme", corpus_id="help")
        )


def test_build_local_ingest_plan_rejects_empty_supported_set_before_core_creation(
    tmp_path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="no supported files matched"):
        build_local_ingest_plan(
            LocalIngestRequest(path=docs, namespace="acme", corpus_id="help")
        )


def test_build_local_ingest_plan_rejects_invalid_concurrency(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide", encoding="utf-8")

    with pytest.raises(ValueError, match="max_concurrency"):
        build_local_ingest_plan(
            LocalIngestRequest(
                path=docs,
                namespace="acme",
                corpus_id="help",
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
        CorpusManifestEntry(
            document_id="doc-billing",
            namespace="acme",
            corpus_id="help",
            document_key=good_key,
            content_sha256=good_hash,
            filename="billing.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    )
    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id="doc-broken",
            namespace="acme",
            corpus_id="help",
            document_key=bad_key,
            content_sha256="old-hash",
            filename="broken.txt",
            mime_type="text/plain",
            chunk_count=1,
        ),
    )

    result = asyncio.run(
        run_local_ingest(
            LocalIngestRequest(
                path=docs,
                namespace="acme",
                corpus_id="help",
                metadata={"source": "seed"},
                force_reindex=True,
            ),
            core_factory=lambda: core,
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
            "manifest_status": "unchanged",
            "manifest_reason": "content_sha256_match",
        },
        {
            "ok": False,
            "path": str(bad),
            "document_key": bad_key,
            "content_sha256": bad_hash,
            "error": "failed to parse",
            "manifest_status": "changed",
            "manifest_reason": "content_sha256_changed",
        },
    ]
    assert core.ensure_ready_calls == 1
    assert core.ingest_calls == [
        _IngestCall(
            file_path=good,
            namespace="acme",
            corpus_id="help",
            document_key=good_key,
            metadata={"source": "seed"},
            force_reindex=True,
        ),
        _IngestCall(
            file_path=bad,
            namespace="acme",
            corpus_id="help",
            document_key=bad_key,
            metadata={"source": "seed"},
            force_reindex=True,
        ),
    ]
    assert core.closed is True
    started = [event for event in buffer.events if isinstance(event, IngestBatchStarted)]
    progress = [event for event in buffer.events if isinstance(event, IngestBatchProgress)]
    completed = [
        event for event in buffer.events if isinstance(event, IngestBatchCompleted)
    ]
    assert len(started) == 1
    assert (started[0].namespace, started[0].corpus_id, started[0].planned_count) == (
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
    assert (progress[0].manifest_status, progress[0].manifest_reason) == (
        "unchanged",
        "content_sha256_match",
    )
    assert (progress[1].manifest_status, progress[1].manifest_reason) == (
        "changed",
        "content_sha256_changed",
    )
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
        run_local_ingest(
            LocalIngestRequest(
                path=docs,
                namespace="acme",
                corpus_id="help",
                max_concurrency=2,
            ),
            core_factory=lambda: core,
        )
    )
    try:
        await asyncio.wait_for(core.two_started.wait(), timeout=1.0)
    except Exception:
        core.release.set()
        await task
        raise
    assert [path.name for path in core.started] == ["a.md", "b.md"]

    core.release.set()
    result = await task

    assert [record.document_key for record in result.records] == [
        local_document_key(docs, first),
        local_document_key(docs, second),
        local_document_key(docs, third),
    ]
    assert result.written_count == 3
    assert core.closed is True


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
        run_local_ingest(
            LocalIngestRequest(
                path=docs,
                namespace="acme",
                corpus_id="help",
                max_concurrency=3,
            ),
            core_factory=lambda: core,
            event_sink=buffer,
        )
    )
    try:
        await asyncio.wait_for(core.all_started.wait(), timeout=1.0)
        await _wait_for_ingest_progress(buffer, count=2)
        early_progress = [
            event
            for event in buffer.events
            if isinstance(event, IngestBatchProgress)
        ]
        assert {event.filename for event in early_progress} == {"b.md", "c.md"}
        core.release_first.set()
        result = await task
    except Exception:
        core.release_first.set()
        await task
        raise

    progress = [event for event in buffer.events if isinstance(event, IngestBatchProgress)]
    assert [event.filename for event in progress] != ["a.md", "b.md", "c.md"]
    assert [event.current_index for event in progress] == [1, 2, 3]
    assert [event.completed_count for event in progress] == [1, 2, 3]
    assert [record.document_key for record in result.records] == [
        *(local_document_key(docs, path) for path in paths),
    ]


async def _wait_for_ingest_progress(buffer: EventBuffer, *, count: int) -> None:
    for _ in range(100):
        progress = [
            event
            for event in buffer.events
            if isinstance(event, IngestBatchProgress)
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
        run_local_ingest(
            LocalIngestRequest(path=file_path, namespace="acme", corpus_id="help"),
            core_factory=lambda: core,
            event_sink=buffer,
        )
    )

    assert result.planned_count == 1
    assert result.written_count == 0
    assert result.failed_count == 1
    assert result.failed[0].document_key == local_document_key(file_path.parent, file_path)
    assert result.failed[0].content_sha256 == compute_content_sha256(b"broken")
    assert result.failed[0].manifest_status == "unknown"
    assert result.failed[0].manifest_reason == "manifest_not_checked"
    assert core.closed is True
    completed = [
        event for event in buffer.events if isinstance(event, IngestBatchCompleted)
    ]
    assert completed[0].succeeded_count == 0
    assert completed[0].failed_count == 1


def test_run_local_ingest_single_file_reuses_initial_read_for_ingest_bytes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_core import local_corpus as local_corpus_module

    file_path = tmp_path / "guide.txt"
    file_path.write_text("guide", encoding="utf-8")
    core = _FakeLocalIngestCore()
    read_paths: list[Path] = []

    async def read_once(path: Path) -> bytes:
        read_paths.append(path)
        return path.read_bytes()

    monkeypatch.setattr(local_corpus_module, "read_file_bytes", read_once)

    result = asyncio.run(
        run_local_ingest(
            LocalIngestRequest(path=file_path, namespace="acme", corpus_id="help"),
            core_factory=lambda: core,
        )
    )

    assert read_paths == [file_path]
    assert result.written[0].content_sha256 == compute_content_sha256(b"guide")
    assert core.ingest_calls == [
        _IngestCall(
            file_path=file_path,
            namespace="acme",
            corpus_id="help",
            document_key=local_document_key(file_path.parent, file_path),
            metadata=None,
            force_reindex=False,
        )
    ]


def test_local_ingest_source_read_failures_are_unknown_manifest_status(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_core import local_corpus as local_corpus_module
    from rag_core import sources as sources_module

    file_path = tmp_path / "blocked.txt"
    file_path.write_text("blocked", encoding="utf-8")
    manifest_dir = tmp_path / "manifest"
    document_key = local_document_key(file_path.parent, file_path)
    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id="doc-blocked",
            namespace="acme",
            corpus_id="help",
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
        LocalIngestRequest(path=file_path, namespace="acme", corpus_id="help")
    )
    reconciliation = reconcile_local_ingest_plan(plan, manifest_dir=manifest_dir)

    assert plan.to_payload(reconciliation=reconciliation)["documents"] == [
        {
            "path": "<local-file>",
            "filename": "blocked.txt",
            "content_sha256_available": False,
            "source_error": "source read failed",
            "manifest_status": "unknown",
            "manifest_reason": "source_read_failed",
        }
    ]
    assert plan.to_payload(
        reconciliation=reconciliation,
        include_private=True,
    )["documents"] == [
        {
            "path": str(file_path),
            "document_key": document_key,
            "content_sha256": None,
            "source_error": "blocked",
            "manifest_status": "unknown",
            "manifest_reason": "source_read_failed",
        }
    ]

    async def fail_read(path: Path) -> bytes:
        if path == file_path:
            raise PermissionError("blocked")
        return path.read_bytes()

    monkeypatch.setattr(local_corpus_module, "read_file_bytes", fail_read)

    core = _FakeLocalIngestCore()
    buffer = EventBuffer()
    result = asyncio.run(
        run_local_ingest(
            LocalIngestRequest(path=file_path, namespace="acme", corpus_id="help"),
            core_factory=lambda: core,
            event_sink=buffer,
            manifest_dir=manifest_dir,
        )
    )

    assert result.failed[0].content_sha256 is None
    assert result.failed[0].manifest_status == "unknown"
    assert result.failed[0].manifest_reason == "source_read_failed"
    assert core.ingest_calls == []
    progress = [event for event in buffer.events if isinstance(event, IngestBatchProgress)]
    assert progress[0].content_sha256 == ""
    assert progress[0].manifest_status == "unknown"
    assert progress[0].manifest_reason == "source_read_failed"


def test_run_local_ingest_success_uses_actual_ingested_hash(tmp_path) -> None:
    file_path = tmp_path / "changed.txt"
    file_path.write_text("old", encoding="utf-8")
    old_hash = compute_content_sha256(b"old")
    new_hash = compute_content_sha256(b"new")
    manifest_dir = tmp_path / "manifest"
    document_key = local_document_key(file_path.parent, file_path)
    write_entry(
        manifest_dir,
        CorpusManifestEntry(
            document_id="doc-changed",
            namespace="acme",
            corpus_id="help",
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

        async def ingest_file(
            self,
            file_path: Path,
            *,
            namespace: str,
            corpus_id: str,
            document_key: str,
            metadata: dict[str, str] | None = None,
            force_reindex: bool = False,
        ) -> IngestedDocument:
            file_path.write_text("new", encoding="utf-8")
            return IngestedDocument(
                document_id="doc-changed",
                corpus_id=corpus_id,
                namespace=namespace,
                chunk_count=1,
                filename=file_path.name,
                mime_type="text/plain",
                document_key=document_key,
                content_sha256=new_hash,
                ingest_state="replaced",
            )

        async def close(self) -> None:
            self.closed = True

    core = _MutatingCore()
    buffer = EventBuffer()

    result = asyncio.run(
        run_local_ingest(
            LocalIngestRequest(path=file_path, namespace="acme", corpus_id="help"),
            core_factory=lambda: core,
            event_sink=buffer,
            manifest_dir=manifest_dir,
        )
    )

    assert result.written[0].content_sha256 == new_hash
    assert result.written[0].manifest_status == "changed"
    assert result.written[0].manifest_reason == "content_sha256_changed"
    progress = [event for event in buffer.events if isinstance(event, IngestBatchProgress)]
    assert progress[0].content_sha256 == new_hash
    assert progress[0].manifest_status == "changed"
    assert progress[0].manifest_reason == "content_sha256_changed"
    assert core.closed is True


def test_run_local_ingest_emits_batch_failed_when_setup_fails(tmp_path) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("billing", encoding="utf-8")
    core = _FakeLocalIngestCore(ready_error=RuntimeError("vector store unavailable"))
    buffer = EventBuffer()

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        asyncio.run(
            run_local_ingest(
                LocalIngestRequest(path=file_path, namespace="acme", corpus_id="help"),
                core_factory=lambda: core,
                event_sink=buffer,
            )
        )

    assert core.ensure_ready_calls == 1
    assert core.ingest_calls == []
    assert core.closed is True
    assert [event.event_type for event in buffer.events] == [
        "ingest.batch.started",
        "ingest.batch.failed",
    ]
    failed = buffer.events[1]
    assert isinstance(failed, IngestBatchFailed)
    assert failed.namespace == "acme"
    assert failed.corpus_id == "help"
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
            run_local_ingest(
                LocalIngestRequest(
                    path=docs,
                    namespace="acme",
                    corpus_id="help",
                    max_concurrency=1,
                ),
                core_factory=lambda: core,
                event_sink=buffer,
            )
        )

    assert "provider failed during ingest" in str(exc_info.value)
    assert "OPENAI_API_KEY" not in str(exc_info.value)
    progress = [event for event in buffer.events if isinstance(event, IngestBatchProgress)]
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
    assert core.closed is True


def test_preview_manifest_builds_preview_payload(tmp_path) -> None:
    file_path = tmp_path / "guide.txt"
    file_path.write_text("billing docs stay easy to find", encoding="utf-8")

    result = asyncio.run(
        preview_manifest(
            ManifestPreviewRequest(
                path=file_path,
                namespace="acme",
                corpus_id="help-center",
                metadata={"source": "seed"},
            )
        )
    )

    payload = result.to_payload()
    document_payload = cast(dict[str, object], payload["document"])
    manifest_entry_payload = cast(dict[str, object], payload["manifest_entry"])
    manifest_metadata = cast(dict[str, object], manifest_entry_payload["metadata"])

    assert document_payload["ingest_state"] == "preview"
    assert manifest_entry_payload["parser"] == "local:text"
    assert manifest_metadata["source"] == "seed"


def test_parse_metadata_fields_requires_key_value() -> None:
    assert parse_metadata_fields(["source=seed", "owner=rag"]) == {
        "source": "seed",
        "owner": "rag",
    }

    with pytest.raises(ValueError, match="metadata entries must use KEY=VALUE"):
        parse_metadata_fields(["broken"])


def test_cli_error_message_for_file_not_found() -> None:
    assert "file not found" in cli_error_message(FileNotFoundError("missing.txt"))
