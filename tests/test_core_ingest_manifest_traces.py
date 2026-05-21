from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

import rag_core.core_ingest as core_ingest
from rag_core.core_ingest import CoreIngestor
from rag_core.core_models import (
    CorpusManifestEntry,
    IngestedDocument,
    OcrRoutingSignal,
    PreparedChunk,
    PreparedDocument,
    ProcessingFingerprint,
)
from rag_core.core_lifecycle import compute_content_sha256
from rag_core.events.sink import EventSink
from rag_core.events.sinks import EventBuffer
from rag_core.events.types import StageError
from rag_core.search.indexer_models import IndexResult
from rag_core.manifest_persistence import read_entries, write_entry
from rag_core.search.types import SearchResult, SearchSidecarQuery, StoredDocumentRecord
from tests.support import RecordingVectorStore


class RecordingIndexer:
    def __init__(self) -> None:
        self.index_document_calls = 0
        self.delete_document_calls = 0

    async def index_document(self, req: object) -> IndexResult:
        self.index_document_calls += 1
        return IndexResult(
            document_id="doc-1",
            chunk_count=1,
            point_ids=["point-1"],
            point_payloads=[],
            document_key="doc.md",
            content_sha256="indexed-content-sha256",
        )

    async def delete_document(
        self,
        document_id: str,
        namespace: str,
        *,
        corpus_id: str,
    ) -> None:
        self.delete_document_calls += 1


class FailingIndexer(RecordingIndexer):
    async def index_document(self, req: object) -> IndexResult:
        self.index_document_calls += 1
        raise RuntimeError("index failed")


class DeleteFailingIndexer(RecordingIndexer):
    async def delete_document(
        self,
        document_id: str,
        namespace: str,
        *,
        corpus_id: str,
    ) -> None:
        self.delete_document_calls += 1
        raise RuntimeError("delete failed")


class EmptyPayloadIndexer(RecordingIndexer):
    async def index_document(self, req: object) -> IndexResult:
        self.index_document_calls += 1
        return IndexResult(
            document_id="doc-1",
            chunk_count=1,
            point_ids=[],
            point_payloads=[],
            document_key="doc.md",
            content_sha256="indexed-content-sha256",
        )


class SevenChunkIndexer(RecordingIndexer):
    async def index_document(self, req: object) -> IndexResult:
        self.index_document_calls += 1
        return IndexResult(
            document_id="doc-1",
            chunk_count=7,
            point_ids=["point-1"],
            point_payloads=[],
            document_key="doc.md",
            content_sha256="indexed-content-sha256",
        )


class FailingSidecar:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def upsert_records(self, records: object) -> None:
        raise RuntimeError("sidecar failed")

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        corpus_id: str | None = None,
    ) -> None:
        self.deleted.append((namespace, document_id))

    async def search(self, query: SearchSidecarQuery) -> list[SearchResult]:
        return []


class RecordingSidecar(FailingSidecar):
    def upsert_records(self, records: object) -> None:
        return None


class DeleteFailingSidecar(FailingSidecar):
    def upsert_records(self, records: object) -> None:
        return None

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        corpus_id: str | None = None,
    ) -> None:
        self.deleted.append((namespace, document_id))
        raise RuntimeError("sidecar delete failed")


def test_manifest_write_failure_emits_sanitized_stage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_entry(directory: Path, entry: object) -> None:
        raise RuntimeError("secret manifest path /tmp/private/acme/help.jsonl")

    monkeypatch.setattr(core_ingest, "write_entry", fail_write_entry)

    async def run() -> tuple[EventBuffer, RecordingIndexer]:
        events = EventBuffer()
        ingestor, indexer = _make_ingestor(
            event_sink=events,
            manifest_directory=tmp_path / "manifest",
        )
        with pytest.raises(RuntimeError) as excinfo:
            await ingestor.ingest_bytes(
                file_bytes=b"hello",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
            )
        assert "secret manifest path" in str(excinfo.value)
        return events, indexer

    events, indexer = asyncio.run(run())

    assert indexer.index_document_calls == 1
    assert indexer.delete_document_calls == 1
    errors = [event for event in events.events if isinstance(event, StageError)]
    assert len(errors) == 1
    assert errors[0].stage == "manifest"
    assert errors[0].error_type == "RuntimeError"
    assert errors[0].message == ""
    assert "secret manifest path" not in str(errors[0])


def test_existing_document_manifest_write_failure_surfaces_after_reindex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_entry(directory: Path, entry: object) -> None:
        raise OSError("disk full at /tmp/private/acme/help.jsonl")

    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=1,
            )
        }
    )
    monkeypatch.setattr(core_ingest, "write_entry", fail_write_entry)

    async def run() -> tuple[EventBuffer, RecordingIndexer]:
        events = EventBuffer()
        ingestor, indexer = _make_ingestor(
            event_sink=events,
            manifest_directory=tmp_path / "manifest",
            store=store,
        )
        with pytest.raises(OSError):
            await ingestor.ingest_bytes(
                file_bytes=b"hello",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
                force_reindex=True,
            )
        return events, indexer

    events, indexer = asyncio.run(run())

    assert indexer.index_document_calls == 0
    assert indexer.delete_document_calls == 0
    errors = [event for event in events.events if isinstance(event, StageError)]
    assert [error.stage for error in errors] == ["manifest", "ingest"]
    assert "disk full" not in str(errors)


def test_existing_document_with_missing_manifest_rolls_back_speculative_entry(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=3,
            )
        }
    )

    async def run() -> FailingIndexer:
        events = EventBuffer()
        indexer = FailingIndexer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
            indexer=indexer,
        )
        with pytest.raises(RuntimeError, match="index failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"new content",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
                force_reindex=True,
            )
        return indexer

    indexer = asyncio.run(run())
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert indexer.index_document_calls == 1
    assert [(entry.document_id, entry.content_sha256, entry.chunk_count) for entry in entries] == [
        ("doc-1", "old-content", 3)
    ]


def test_unchanged_ingest_without_manifest_preserves_indexed_source_identity(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    content_hash = compute_content_sha256(b"same")
    manifest_directory = tmp_path / "manifest"
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="old.pdf",
                content_sha256=content_hash,
                processing_version=processing_version,
                chunk_count=4,
            )
        }
    )

    async def run() -> IngestedDocument:
        events = EventBuffer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
        )
        return await ingestor.ingest_bytes(
            file_bytes=b"same",
            filename="new.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
        )

    result = asyncio.run(run())
    [entry] = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert result.document_key == "old.pdf"
    assert result.content_sha256 == content_hash
    assert result.filename == "old.pdf"
    assert result.chunk_count == 4
    assert entry.document_key == "old.pdf"
    assert entry.content_sha256 == content_hash
    assert entry.filename == "old.pdf"
    assert entry.chunk_count == 4


def test_unchanged_ingest_uses_safe_filename_from_source_document_key(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    content_hash = compute_content_sha256(b"same")
    manifest_directory = tmp_path / "manifest"
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="local:docs/guide.md#source:deadbeef",
                content_sha256=content_hash,
                processing_version=processing_version,
                chunk_count=4,
            )
        }
    )

    async def run() -> IngestedDocument:
        events = EventBuffer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
        )
        return await ingestor.ingest_bytes(
            file_bytes=b"same",
            filename="new.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
        )

    result = asyncio.run(run())
    [entry] = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert result.filename == "guide.md"
    assert entry.filename == "guide.md"


def test_failed_reindex_restores_previous_manifest_shape(tmp_path: Path) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    previous_entry = CorpusManifestEntry(
        document_id="doc-1",
        namespace="acme",
        corpus_id="help",
        document_key="doc.md",
        content_sha256="old-content",
        filename="doc.md",
        mime_type="text/markdown",
        chunk_count=3,
        parser="local:text",
        needs_ocr=True,
        metadata={"title": "First Title", "ocr_page_indices": [0, 2]},
    )
    write_entry(manifest_directory, previous_entry)
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=3,
            )
        }
    )

    async def run() -> None:
        events = EventBuffer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
            indexer=FailingIndexer(),
        )
        with pytest.raises(RuntimeError, match="index failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"new content",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
                force_reindex=True,
            )

    asyncio.run(run())
    [entry] = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert entry.document_id == "doc-1"
    assert entry.content_sha256 == "old-content"
    assert entry.chunk_count == 3
    assert entry.parser == "local:text"
    assert entry.needs_ocr is True
    assert entry.metadata == {"title": "First Title", "ocr_page_indices": [0, 2]}


def test_failed_reindex_without_previous_manifest_restores_indexed_source_identity(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="old.pdf",
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=3,
            )
        }
    )

    async def run() -> None:
        events = EventBuffer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
            indexer=FailingIndexer(),
        )
        with pytest.raises(RuntimeError, match="index failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"new content",
                filename="new.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
                force_reindex=True,
            )

    asyncio.run(run())
    [entry] = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert entry.document_key == "old.pdf"
    assert entry.content_sha256 == "old-content"
    assert entry.filename == "old.pdf"
    assert entry.chunk_count == 3


def test_successful_existing_document_reindex_writes_actual_index_result(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=3,
            )
        }
    )

    async def run() -> SevenChunkIndexer:
        events = EventBuffer()
        indexer = SevenChunkIndexer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
            indexer=indexer,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"new content",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
            force_reindex=True,
        )
        return indexer

    indexer = asyncio.run(run())
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert indexer.index_document_calls == 1
    assert [(entry.document_id, entry.content_sha256, entry.chunk_count) for entry in entries] == [
        ("doc-1", "indexed-content-sha256", 7)
    ]


def test_unchanged_ingest_heals_null_document_key_and_preserves_prepare_metadata(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    write_entry(
        manifest_directory,
        CorpusManifestEntry(
            document_id="doc-1",
            namespace="acme",
            corpus_id="help",
            document_key=None,
            content_sha256=compute_content_sha256(b"hello"),
            filename="doc.md",
            mime_type="text/markdown",
            chunk_count=1,
        ),
    )
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key=None,
                content_sha256=compute_content_sha256(b"hello"),
                processing_version=processing_version,
                chunk_count=1,
            )
        }
    )

    async def run() -> tuple[RecordingIndexer, PreparedDocument]:
        events = EventBuffer()
        prepared = PreparedDocument(
            filename="doc.md",
            mime_type="text/markdown",
            path=None,
            markdown="unchanged content",
            chunks=[
                PreparedChunk(
                    chunk_index=0,
                    text="unchanged content",
                    embedding_text="unchanged content",
                    word_count=2,
                )
            ],
            metadata={
                "parser": "local:pdf",
                "needs_ocr": True,
                "ocr": {"provider": "tesseract", "page_count": 2},
            },
            ocr=OcrRoutingSignal(needed=True),
        )

        async def prepare_bytes(
            *,
            file_bytes: bytes,
            filename: str,
            mime_type: str,
            path: str | None = None,
        ) -> PreparedDocument:
            return prepared

        indexer = EmptyPayloadIndexer()
        ingestor = CoreIngestor(
            collection_name="rag_core_chunks",
            source_type="file",
            embedding_model="fake-embedding",
            processing_version=ProcessingFingerprint(
                base_version="rag_core_processing_v1",
                source_type="file",
            ),
            store=store,
            indexer=cast(Any, indexer),
            sidecar=None,
            prepare_bytes=prepare_bytes,
            event_sink=events,
            manifest_directory=manifest_directory,
        )
        result = await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
        )
        assert result.ingest_state == "unchanged"
        assert result.metadata["parser"] == "local:pdf"
        assert result.metadata["needs_ocr"] is True
        return indexer, prepared

    indexer, prepared = asyncio.run(run())

    assert indexer.index_document_calls == 0
    [entry] = read_entries(manifest_directory, namespace="acme", corpus_id="help")
    assert entry.document_key == "doc.md"
    assert entry.parser == "local:pdf"
    assert entry.needs_ocr is True
    assert entry.metadata.get("ocr") == prepared.metadata["ocr"]


def test_failed_reindex_repairs_null_document_key_from_resolved_identity(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    write_entry(
        manifest_directory,
        CorpusManifestEntry(
            document_id="doc-1",
            namespace="acme",
            corpus_id="help",
            document_key=None,
            content_sha256="old-content",
            filename="doc.md",
            mime_type="text/markdown",
            chunk_count=3,
        ),
    )
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key=None,
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=3,
            )
        }
    )

    async def run() -> None:
        events = EventBuffer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
            indexer=FailingIndexer(),
        )
        with pytest.raises(RuntimeError, match="index failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"new content",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
                force_reindex=True,
            )

    asyncio.run(run())
    [entry] = read_entries(manifest_directory, namespace="acme", corpus_id="help")
    assert entry.document_key == "doc.md"
    assert entry.content_sha256 == "old-content"


def test_existing_document_final_manifest_retry_does_not_report_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=3,
            )
        }
    )
    original_write_entry = core_ingest.write_entry
    write_calls = 0

    def fail_final_write(directory: Path, entry: object) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("final manifest write failed")
        original_write_entry(directory, cast(Any, entry))

    monkeypatch.setattr(core_ingest, "write_entry", fail_final_write)

    async def run() -> SevenChunkIndexer:
        events = EventBuffer()
        indexer = SevenChunkIndexer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
            indexer=indexer,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"new content",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
            force_reindex=True,
        )
        return indexer

    indexer = asyncio.run(run())
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert indexer.index_document_calls == 1
    assert indexer.delete_document_calls == 0
    assert write_calls == 3
    assert [(entry.document_id, entry.content_sha256, entry.chunk_count) for entry in entries] == [
        ("doc-1", "indexed-content-sha256", 7)
    ]


def test_sidecar_sync_failure_is_not_reported_as_success(
    tmp_path: Path,
) -> None:
    manifest_directory = tmp_path / "manifest"

    async def run() -> tuple[EventBuffer, RecordingIndexer, FailingSidecar]:
        events = EventBuffer()
        indexer = EmptyPayloadIndexer()
        sidecar = FailingSidecar()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            indexer=indexer,
            sidecar=sidecar,
        )
        with pytest.raises(RuntimeError, match="sidecar failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"hello",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
            )
        return events, indexer, sidecar

    events, indexer, sidecar = asyncio.run(run())
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")
    errors = [event for event in events.events if isinstance(event, StageError)]

    assert indexer.index_document_calls == 1
    assert indexer.delete_document_calls == 1
    assert entries == []
    assert sidecar.deleted == [("acme", "doc-1"), ("acme", "doc-1")]
    assert [(error.stage, error.error_type) for error in errors] == [
        ("sidecar", "RuntimeError")
    ]


def test_existing_document_sidecar_failure_does_not_delete_reindexed_document(
    tmp_path: Path,
) -> None:
    processing_version = ProcessingFingerprint(
        base_version="rag_core_processing_v1",
        source_type="file",
    ).serialize()
    manifest_directory = tmp_path / "manifest"
    store = RecordingVectorStore(
        document_records={
            ("acme", "help", "doc-1"): StoredDocumentRecord(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
                document_key="doc.md",
                content_sha256="old-content",
                processing_version=processing_version,
                chunk_count=3,
            )
        }
    )

    async def run() -> tuple[EmptyPayloadIndexer, FailingSidecar]:
        events = EventBuffer()
        indexer = EmptyPayloadIndexer()
        sidecar = FailingSidecar()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            store=store,
            indexer=indexer,
            sidecar=sidecar,
        )
        async def prepare_bytes(
            *,
            file_bytes: bytes,
            filename: str,
            mime_type: str,
            path: str | None = None,
        ) -> PreparedDocument:
            del file_bytes
            return PreparedDocument(
                filename=filename,
                mime_type=mime_type,
                path=path,
                markdown="ocr text",
                chunks=[
                    PreparedChunk(
                        chunk_index=0,
                        text="ocr text",
                        embedding_text="ocr text",
                        word_count=2,
                    )
                ],
                metadata={
                    "parser": "local:pdf",
                    "needs_ocr": True,
                    "ocr_page_indices": [0, 2],
                },
            )

        cast(Any, ingestor)._prepare_bytes = prepare_bytes
        with pytest.raises(RuntimeError, match="sidecar failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"new content",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
                document_id="doc-1",
                force_reindex=True,
            )
        return indexer, sidecar

    indexer, sidecar = asyncio.run(run())
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")

    assert indexer.index_document_calls == 1
    assert indexer.delete_document_calls == 0
    assert sidecar.deleted == [("acme", "doc-1")]
    assert [(entry.document_id, entry.content_sha256, entry.chunk_count) for entry in entries] == [
        ("doc-1", "indexed-content-sha256", 1)
    ]
    [entry] = entries
    assert entry.parser == "local:pdf"
    assert entry.needs_ocr is True
    assert entry.metadata["ocr_page_indices"] == [0, 2]


def test_manifest_write_failure_reports_failed_index_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_entry(directory: Path, entry: object) -> None:
        raise OSError("manifest write failed")

    monkeypatch.setattr(core_ingest, "write_entry", fail_write_entry)

    async def run() -> DeleteFailingIndexer:
        events = EventBuffer()
        indexer = DeleteFailingIndexer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=tmp_path / "manifest",
            indexer=indexer,
        )
        with pytest.raises(ExceptionGroup) as excinfo:
            await ingestor.ingest_bytes(
                file_bytes=b"hello",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help",
            )
        assert [type(exc).__name__ for exc in excinfo.value.exceptions] == [
            "OSError",
            "RuntimeError",
        ]
        return indexer

    indexer = asyncio.run(run())

    assert indexer.index_document_calls == 1
    assert indexer.delete_document_calls == 1


def test_delete_document_removes_manifest_entry(tmp_path: Path) -> None:
    manifest_directory = tmp_path / "manifest"

    async def run() -> None:
        events = EventBuffer()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
        )
        await ingestor.delete_document(
            document_id="doc-1",
            namespace="acme",
            corpus_id="help",
        )

    asyncio.run(run())

    assert read_entries(manifest_directory, namespace="acme", corpus_id="help") == []


def test_delete_document_keeps_manifest_when_index_delete_fails(
    tmp_path: Path,
) -> None:
    manifest_directory = tmp_path / "manifest"

    async def run() -> tuple[DeleteFailingIndexer, RecordingSidecar]:
        events = EventBuffer()
        indexer = DeleteFailingIndexer()
        sidecar = RecordingSidecar()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            indexer=indexer,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
        )
        ingestor._sidecar = cast(Any, sidecar)
        sidecar.deleted.clear()
        with pytest.raises(RuntimeError, match="delete failed"):
            await ingestor.delete_document(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
            )
        return indexer, sidecar

    indexer, sidecar = asyncio.run(run())

    assert indexer.delete_document_calls == 1
    assert sidecar.deleted == []
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")
    assert [entry.document_id for entry in entries] == ["doc-1"]


def test_delete_document_keeps_manifest_when_manifest_delete_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_directory = tmp_path / "manifest"

    def fail_delete_entry(*args: object, **kwargs: object) -> bool:
        raise OSError("manifest delete failed")

    monkeypatch.setattr(core_ingest, "delete_entry", fail_delete_entry)

    async def run() -> tuple[RecordingIndexer, RecordingSidecar]:
        events = EventBuffer()
        indexer = RecordingIndexer()
        sidecar = RecordingSidecar()
        ingestor, _ = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
            indexer=indexer,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
        )
        ingestor._sidecar = cast(Any, sidecar)
        indexer.delete_document_calls = 0
        sidecar.deleted.clear()
        with pytest.raises(OSError, match="manifest delete failed"):
            await ingestor.delete_document(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
            )
        return indexer, sidecar

    indexer, sidecar = asyncio.run(run())

    assert indexer.delete_document_calls == 1
    assert sidecar.deleted == [("acme", "doc-1")]
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")
    assert [entry.document_id for entry in entries] == ["doc-1"]


def test_delete_document_keeps_manifest_aligned_when_sidecar_delete_fails(
    tmp_path: Path,
) -> None:
    manifest_directory = tmp_path / "manifest"

    async def run() -> tuple[DeleteFailingSidecar, RecordingIndexer]:
        events = EventBuffer()
        sidecar = DeleteFailingSidecar()
        ingestor, indexer = _make_ingestor(
            event_sink=events,
            manifest_directory=manifest_directory,
        )
        await ingestor.ingest_bytes(
            file_bytes=b"hello",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            corpus_id="help",
            document_id="doc-1",
        )
        ingestor._sidecar = cast(Any, sidecar)
        sidecar.deleted.clear()
        with pytest.raises(RuntimeError, match="sidecar delete failed"):
            await ingestor.delete_document(
                document_id="doc-1",
                namespace="acme",
                corpus_id="help",
            )
        return sidecar, indexer

    sidecar, indexer = asyncio.run(run())

    assert sidecar.deleted == [("acme", "doc-1")]
    assert indexer.delete_document_calls == 1
    entries = read_entries(manifest_directory, namespace="acme", corpus_id="help")
    assert [entry.document_id for entry in entries] == ["doc-1"]


def _make_ingestor(
    *,
    event_sink: EventSink,
    manifest_directory: Path,
    store: RecordingVectorStore | None = None,
    indexer: RecordingIndexer | None = None,
    sidecar: object | None = None,
) -> tuple[CoreIngestor, RecordingIndexer]:
    indexer = indexer or RecordingIndexer()

    async def prepare_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
    ) -> PreparedDocument:
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            path=path,
            markdown="hi",
            chunks=[
                PreparedChunk(
                    chunk_index=0,
                    text="hi",
                    embedding_text="hi",
                    word_count=1,
                )
            ],
        )

    return (
        CoreIngestor(
            collection_name="rag_core_chunks",
            source_type="file",
            embedding_model="fake-embedding",
            processing_version=ProcessingFingerprint(
                base_version="rag_core_processing_v1",
                source_type="file",
            ),
            store=store or RecordingVectorStore(),
            indexer=cast(Any, indexer),
            sidecar=cast(Any, sidecar),
            prepare_bytes=prepare_bytes,
            event_sink=event_sink,
            manifest_directory=manifest_directory,
        ),
        indexer,
    )
