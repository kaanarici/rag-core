from __future__ import annotations

import asyncio
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from rag_core import RAGCore
from rag_core.core_archive_ingest import ingest_zip_archive_with_core
from rag_core.core_models import IngestedDocument
from rag_core.events import EventBuffer
from rag_core.local_ingest_models import LocalIngestResult
from rag_core.events import IngestBatchCompleted, IngestBatchProgress, IngestBatchStarted
from rag_core.sources import archive_document_key

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)

pytestmark = [pytest.mark.plumbing]


def test_rag_core_ingest_archive_indexes_supported_zip_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "docs.zip"
    manifest_dir = tmp_path / "manifest"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", "# Guide\n\nAlpha")
        archive.writestr("docs/reference.md", "# Reference\n\nBeta")
        archive.writestr("assets/logo.png", b"\x89PNG\r\n\x1a\n")

    async def scenario() -> tuple[
        LocalIngestResult,
        LocalIngestResult,
        RecordingVectorStore,
        EventBuffer,
    ]:
        store = RecordingVectorStore()
        events = EventBuffer()
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_archive_ingest",
                embedding_dimensions=4,
                manifest_directory=manifest_dir,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=events,
        )
        try:
            result = await core.ingest_archive(
                archive_path,
                namespace="acme",
                corpus_id="help",
                metadata={"team": "docs"},
                max_concurrency=2,
            )
            rerun_result = await core.ingest_archive(
                archive_path,
                namespace="acme",
                corpus_id="help",
                metadata={"team": "docs"},
                max_concurrency=2,
            )
        finally:
            await core.close()
        return result, rerun_result, store, events

    result, rerun_result, store, events = asyncio.run(scenario())

    assert result.succeeded_count == 2
    assert result.failed_count == 0
    assert {record.manifest_status for record in result.succeeded} == {"missing"}
    assert [record.document_key for record in result.succeeded] == [
        archive_document_key(archive_path, "docs/guide.md"),
        archive_document_key(archive_path, "docs/reference.md"),
    ]
    assert rerun_result.succeeded_count == 2
    assert rerun_result.skipped_count == 2
    assert {record.manifest_status for record in rerun_result.succeeded} == {
        "unchanged"
    }
    assert [record.path for record in result.succeeded] == [
        "docs.zip!/docs/guide.md",
        "docs.zip!/docs/reference.md",
    ]
    assert str(tmp_path) not in repr(result.succeeded)
    points = [point for call in store.upsert_calls for point in call]
    assert {point.payload["source_type"] for point in points} == {"archive"}
    assert {
        json.loads(str(point.payload["processing_version"]))["source_type"]
        for point in points
    } == {"archive"}
    assert store.close_calls == 1
    assert isinstance(events.events[0], IngestBatchStarted)
    progress = events.by_type("ingest.batch.progress")
    assert len(progress) == 4
    assert all(isinstance(event, IngestBatchProgress) for event in progress)
    typed_progress = cast(list[IngestBatchProgress], progress)
    assert [event.current_index for event in typed_progress] == [1, 2, 1, 2]
    assert isinstance(events.events[-1], IngestBatchCompleted)


def test_rag_core_ingest_archive_failure_preserves_sanitized_message(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", "# Guide")

    class FailingCore:
        async def ingest_bytes(self, **_kwargs: object) -> Any:
            raise ValueError("failed to parse archive member")

    async def scenario() -> tuple[LocalIngestResult, EventBuffer]:
        events = EventBuffer()
        result = await ingest_zip_archive_with_core(
            core=FailingCore(),
            archive_path=archive_path,
            namespace="acme",
            corpus_id="help",
            event_sink=events,
        )
        return result, events

    result, events = asyncio.run(scenario())

    assert result.failed_count == 1
    assert result.failed[0].error == "failed to parse archive member"
    progress = events.by_type("ingest.batch.progress")
    assert len(progress) == 1
    assert isinstance(progress[0], IngestBatchProgress)
    assert progress[0].current_index == 1
    assert progress[0].error == "ValueError"
    assert "failed to parse archive member" not in str(progress[0])


def test_archive_ingest_reuses_planned_member_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("docs/guide.md", b"# Guide")
        archive.writestr("docs/reference.md", b"# Reference")
        archive.writestr("assets/logo.png", b"\x89PNG\r\n\x1a\n")

    opened_members: list[str] = []
    original_open = zipfile.ZipFile.open

    def counting_open(
        self: zipfile.ZipFile,
        name: str | zipfile.ZipInfo,
        mode: str = "r",
        pwd: bytes | None = None,
        *,
        force_zip64: bool = False,
    ) -> Any:
        if isinstance(name, zipfile.ZipInfo):
            opened_members.append(name.filename)
        else:
            opened_members.append(name)
        return original_open(self, name, mode, pwd, force_zip64=force_zip64)

    class RecordingCore:
        def __init__(self) -> None:
            self.file_bytes: list[bytes] = []

        async def ingest_bytes(self, **kwargs: object) -> IngestedDocument:
            file_bytes = cast(bytes, kwargs["file_bytes"])
            self.file_bytes.append(file_bytes)
            return IngestedDocument(
                document_id=f"doc-{len(self.file_bytes)}",
                namespace=cast(str, kwargs["namespace"]),
                corpus_id=cast(str, kwargs["corpus_id"]),
                chunk_count=1,
                filename=cast(str, kwargs["filename"]),
                mime_type=cast(str, kwargs["mime_type"]),
                document_key=cast(str | None, kwargs["document_key"]),
                content_sha256=hashlib.sha256(file_bytes).hexdigest(),
            )

    monkeypatch.setattr(zipfile.ZipFile, "open", counting_open)
    core = RecordingCore()

    result = asyncio.run(
        ingest_zip_archive_with_core(
            core=core,
            archive_path=archive_path,
            namespace="acme",
            corpus_id="help",
            max_concurrency=2,
        )
    )

    assert result.succeeded_count == 2
    assert core.file_bytes == [b"# Guide", b"# Reference"]
    assert opened_members == ["docs/guide.md", "docs/reference.md"]
