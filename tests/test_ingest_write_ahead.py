"""Torn-write rollback + write-ahead journal recovery.

Three guarantees:

1. ``kw_only=True`` on the public frozen dataclasses: you cannot
   construct ``PreparedChunk``, ``IngestedDocument``, ``DeleteDocumentResult``,
   ``CollectionManifestEntry``, ``CollectionManifest``, ``OcrMetadata`` or
   ``ProcessingFingerprint`` by accident with positional args.
2. The index-failure rollback in ``CoreIngestor`` calls
   ``indexer.delete_document`` BEFORE restoring the manifest, so a torn Qdrant
   batch upsert cannot leave residual chunks under the new content_sha256.
3. The write-ahead journal records ``upserted_pending_manifest`` before
   ``indexer.upsert`` and ``manifest_written`` after manifest commit. A crash
   between those steps leaves a pending entry; the next ingest of the same
   ``(namespace, collection, document_id)`` triple replays the purge via
   ``resume_pending_write_ahead`` before fresh content lands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from rag_core._engine.core_ingest import CoreIngestor
from rag_core._engine.core_ingest_write_ahead import (
    STATE_MANIFEST_WRITTEN,
    STATE_UPSERTED_PENDING_MANIFEST,
    IngestWriteAheadJournal,
    WriteAheadEntry,
    resume_pending_write_ahead,
    write_ahead_journal_for,
)
from rag_core.core_models import (
    CollectionManifest,
    CollectionManifestEntry,
    DeleteDocumentResult,
    IngestedDocument,
    OcrMetadata,
    PreparedChunk,
    PreparedDocument,
    ProcessingFingerprint,
)
from rag_core.events.sinks import EventBuffer
from rag_core.search.indexer_models import DeleteAck, IndexResult
from rag_core.manifest.persistence import read_entries, write_entry
from tests.support import RecordingVectorStore


pytestmark = [pytest.mark.plumbing]


# -- kw_only enforcement --------------------------------------------------


def test_prepared_chunk_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        PreparedChunk(0, "text", "text", 1)  # type: ignore[misc]


def test_ingested_document_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        IngestedDocument(  # type: ignore[misc]
            "doc-1",
            "corpus",
            "ns",
            1,
            "doc.md",
            "text/markdown",
        )


def test_delete_document_result_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        DeleteDocumentResult("doc-1", "ns", "corpus", True)  # type: ignore[misc]


def test_corpus_manifest_entry_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        CollectionManifestEntry(  # type: ignore[misc]
            "doc-1",
            "ns",
            "corpus",
            "doc.md",
            "sha",
            "doc.md",
            "text/markdown",
            1,
        )


def test_corpus_manifest_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        CollectionManifest(  # type: ignore[misc]
            "ns",
            "corpus",
            "rag_core_chunks",
            "fake-provider",
            "fake-model",
            128,
            0,
            0,
            (),
            0,
            0,
            (),
        )


def test_ocr_metadata_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        OcrMetadata("provider", "model")  # type: ignore[misc]


def test_processing_fingerprint_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        ProcessingFingerprint("rag-core-v1", "file")  # type: ignore[misc]


# -- index-failure rollback fires delete-by-document-id -------------------


class _FailingThenCountingIndexer:
    """Indexer whose first ``index_document`` raises, recording rollback calls.

    Lets the test assert that the index-failure branch in ``CoreIngestor``
    calls ``delete_document`` before restoring the manifest.
    """

    def __init__(self) -> None:
        self.index_document_calls = 0
        self.delete_document_calls = 0
        self.delete_targets: list[tuple[str, str, str]] = []

    async def index_document(self, req: object, *, event_sink: object = None) -> IndexResult:
        self.index_document_calls += 1
        raise RuntimeError("index failed mid-upsert")

    async def delete_document(
        self,
        *,
        document_id: str,
        namespace: str,
        collection: str,
    ) -> DeleteAck:
        self.delete_document_calls += 1
        self.delete_targets.append((namespace, collection, document_id))
        return DeleteAck(succeeded=True, deleted_point_count=-1)


def _make_test_ingestor(
    *,
    manifest_directory: Path,
    indexer: _FailingThenCountingIndexer,
    event_sink: EventBuffer,
) -> CoreIngestor:
    async def prepare_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
        namespace: str = "",
        collection: str = "",
        document_id: str = "",
    ) -> PreparedDocument:
        del namespace, collection, document_id
        return PreparedDocument(
            filename=filename,
            mime_type=mime_type,
            path=path,
            markdown="alpha beta",
            chunks=[
                PreparedChunk(
                    chunk_index=0,
                    text="alpha beta",
                    embedding_text="alpha beta",
                    word_count=2,
                ),
            ],
        )

    return CoreIngestor(
        collection_name="rag_core_chunks",
        source_type="file",
        embedding_model="fake-embedding",
        processing_version=ProcessingFingerprint(
            base_version="rag_core_processing_v1",
            source_type="file",
        ),
        store=RecordingVectorStore(),
        indexer=cast(Any, indexer),
        sidecar=None,
        prepare_bytes=prepare_bytes,
        event_sink=event_sink,
        manifest_directory=manifest_directory,
    )


def test_index_failure_triggers_best_effort_rollback_delete(tmp_path: Path) -> None:
    """A torn upsert is purged before manifest restore.

    The rollback delete is best-effort. It must run even when no prior
    manifest entry exists, so the new content_sha256 cannot leave residue
    behind.
    """
    manifest_directory = tmp_path / "manifest"
    indexer = _FailingThenCountingIndexer()
    events = EventBuffer()
    ingestor = _make_test_ingestor(
        manifest_directory=manifest_directory,
        indexer=indexer,
        event_sink=events,
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="index failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"alpha beta",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                collection="help",
                document_id="doc-torn",
            )

    asyncio.run(run())

    assert indexer.index_document_calls == 1
    assert indexer.delete_document_calls == 1
    assert indexer.delete_targets == [("acme", "help", "doc-torn")]
    # No manifest entry should survive a fresh-ingest torn write.
    assert read_entries(manifest_directory, namespace="acme", collection="help") == []


# -- write-ahead journal: record_pending and record_committed -------------


def test_write_ahead_journal_pending_entry_is_resumable(tmp_path: Path) -> None:
    """A pending entry is what ``resume_pending_write_ahead`` re-runs."""
    journal = IngestWriteAheadJournal(directory=tmp_path / "manifest")
    pending = journal.record_pending(
        namespace="acme",
        collection="restricted",
        document_id="doc-pending",
        content_sha256="sha-pending",
        expected_chunk_count=3,
    )
    assert pending.state == STATE_UPSERTED_PENDING_MANIFEST
    pending_entries = journal.pending_entries()
    assert len(pending_entries) == 1
    assert pending_entries[0].document_id == "doc-pending"


def test_write_ahead_journal_committed_entry_is_not_pending(tmp_path: Path) -> None:
    journal = IngestWriteAheadJournal(directory=tmp_path / "manifest")
    journal.record_pending(
        namespace="acme",
        collection="restricted",
        document_id="doc-fresh",
        content_sha256="sha-fresh",
        expected_chunk_count=2,
    )
    journal.record_committed(
        namespace="acme",
        collection="restricted",
        document_id="doc-fresh",
        content_sha256="sha-fresh",
        expected_chunk_count=2,
    )
    assert journal.pending_entries() == []
    latest = journal.latest_entry(
        namespace="acme",
        collection="restricted",
        document_id="doc-fresh",
    )
    assert latest is not None
    assert latest.state == STATE_MANIFEST_WRITTEN


def test_resume_pending_write_ahead_purges_orphan_chunks(tmp_path: Path) -> None:
    """A crash between upsert and manifest.write leaves a pending entry;
    the next ingest replays the purge before fresh content lands.
    """
    manifest_directory = tmp_path / "manifest"
    # Simulate a crashed prior ingest: journal entry exists, manifest does not.
    journal = IngestWriteAheadJournal(directory=manifest_directory)
    journal.record_pending(
        namespace="acme",
        collection="restricted",
        document_id="doc-orphan",
        content_sha256="sha-orphan",
        expected_chunk_count=4,
    )

    class _ResumeIndexer:
        def __init__(self) -> None:
            self.delete_calls: list[tuple[str, str, str]] = []

        async def delete_document(
            self,
            *,
            document_id: str,
            namespace: str,
            collection: str,
        ) -> DeleteAck:
            self.delete_calls.append((namespace, collection, document_id))
            return DeleteAck(succeeded=True, deleted_point_count=-1)

    indexer = _ResumeIndexer()
    events = EventBuffer()

    async def run() -> bool:
        return await resume_pending_write_ahead(
            indexer=cast(Any, indexer),
            event_sink=events,
            manifest_directory=manifest_directory,
            namespace="acme",
            collection="restricted",
            document_id="doc-orphan",
        )

    resolved = asyncio.run(run())
    assert resolved is True
    assert indexer.delete_calls == [("acme", "restricted", "doc-orphan")]
    # The journal is closed by the resume; no pending entries remain.
    assert journal.pending_entries() == []


def test_resume_pending_write_ahead_is_noop_when_committed(tmp_path: Path) -> None:
    manifest_directory = tmp_path / "manifest"
    journal = IngestWriteAheadJournal(directory=manifest_directory)
    journal.record_pending(
        namespace="acme",
        collection="public",
        document_id="doc-ok",
        content_sha256="sha-ok",
        expected_chunk_count=1,
    )
    journal.record_committed(
        namespace="acme",
        collection="public",
        document_id="doc-ok",
        content_sha256="sha-ok",
        expected_chunk_count=1,
    )

    class _NoTouchIndexer:
        async def delete_document(
            self,
            *,
            document_id: str,
            namespace: str,
            collection: str,
        ) -> DeleteAck:
            raise AssertionError("delete should not be called for a committed entry")

    async def run() -> bool:
        return await resume_pending_write_ahead(
            indexer=cast(Any, _NoTouchIndexer()),
            event_sink=None,
            manifest_directory=manifest_directory,
            namespace="acme",
            collection="public",
            document_id="doc-ok",
        )

    assert asyncio.run(run()) is False


def _committed_manifest_entry(
    *, document_id: str, namespace: str, collection: str, content_sha256: str, chunk_count: int
) -> CollectionManifestEntry:
    return CollectionManifestEntry(
        document_id=document_id,
        namespace=namespace,
        collection=collection,
        document_key=None,
        content_sha256=content_sha256,
        filename="doc.txt",
        mime_type="text/plain",
        chunk_count=chunk_count,
    )


def test_resume_purges_even_when_manifest_matches_pending_content(
    tmp_path: Path,
) -> None:
    """A staged manifest entry carrying the NEW content_sha256 is written BEFORE
    the upsert on an existing-doc reindex, so a torn upsert leaves a manifest
    entry that matches the pending write-ahead content. Replay must STILL purge:
    the manifest match is not a commit signal (it could be the pre-upsert staged
    entry), and purging self-heals because resolve_ingest_decision reads the
    store, so the same-triple ingest that follows re-indexes from scratch.

    Regression: an earlier read-repair trusted the manifest content match and
    skipped the purge here, stranding the torn partial chunks.
    """
    manifest_directory = tmp_path / "manifest"
    journal = IngestWriteAheadJournal(directory=manifest_directory)
    journal.record_pending(
        namespace="acme",
        collection="help",
        document_id="doc-staged",
        content_sha256="sha-new",
        expected_chunk_count=4,
    )
    write_entry(
        manifest_directory,
        _committed_manifest_entry(
            document_id="doc-staged",
            namespace="acme",
            collection="help",
            content_sha256="sha-new",
            chunk_count=4,
        ),
    )

    class _CountingIndexer:
        def __init__(self) -> None:
            self.deletes: list[tuple[str, str, str]] = []

        async def delete_document(
            self, *, document_id: str, namespace: str, collection: str
        ) -> DeleteAck:
            self.deletes.append((namespace, collection, document_id))
            return DeleteAck(succeeded=True, deleted_point_count=-1)

    indexer = _CountingIndexer()

    async def run() -> bool:
        return await resume_pending_write_ahead(
            indexer=cast(Any, indexer),
            event_sink=None,
            manifest_directory=manifest_directory,
            namespace="acme",
            collection="help",
            document_id="doc-staged",
        )

    assert asyncio.run(run()) is True
    assert indexer.deletes == [("acme", "help", "doc-staged")]


def test_resume_pending_write_ahead_is_noop_when_no_journal(tmp_path: Path) -> None:
    class _NoTouchIndexer:
        async def delete_document(
            self,
            *,
            document_id: str,
            namespace: str,
            collection: str,
        ) -> DeleteAck:
            raise AssertionError("delete should not be called without a journal")

    async def run() -> bool:
        return await resume_pending_write_ahead(
            indexer=cast(Any, _NoTouchIndexer()),
            event_sink=None,
            manifest_directory=None,
            namespace="acme",
            collection="public",
            document_id="doc-anything",
        )

    assert asyncio.run(run()) is False


def test_write_ahead_journal_records_state_through_successful_ingest(
    tmp_path: Path,
) -> None:
    """Full ingest writes ``pending`` then ``committed`` for the same triple."""
    manifest_directory = tmp_path / "manifest"

    class _SuccessIndexer:
        def __init__(self) -> None:
            self.index_document_calls = 0
            self.delete_document_calls = 0

        async def index_document(
            self,
            req: object,
            *,
            event_sink: object = None,
        ) -> IndexResult:
            self.index_document_calls += 1
            return IndexResult(
                document_id="doc-ok",
                chunk_count=1,
                point_ids=["point-1"],
                point_payloads=[],
                document_key="doc.md",
                content_sha256="indexed-content-sha256",
            )

        async def delete_document(
            self,
            *,
            document_id: str,
            namespace: str,
            collection: str,
        ) -> DeleteAck:
            self.delete_document_calls += 1
            return DeleteAck(succeeded=True, deleted_point_count=-1)

    indexer = _SuccessIndexer()
    events = EventBuffer()

    async def prepare_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        path: str | None = None,
        namespace: str = "",
        collection: str = "",
        document_id: str = "",
    ) -> PreparedDocument:
        del namespace, collection, document_id
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
                ),
            ],
        )

    ingestor = CoreIngestor(
        collection_name="rag_core_chunks",
        source_type="file",
        embedding_model="fake-embedding",
        processing_version=ProcessingFingerprint(
            base_version="rag_core_processing_v1",
            source_type="file",
        ),
        store=RecordingVectorStore(),
        indexer=cast(Any, indexer),
        sidecar=None,
        prepare_bytes=prepare_bytes,
        event_sink=events,
        manifest_directory=manifest_directory,
    )

    async def run() -> None:
        await ingestor.ingest_bytes(
            file_bytes=b"hi",
            filename="doc.md",
            mime_type="text/markdown",
            namespace="acme",
            collection="restricted",
            document_id="doc-ok",
        )

    asyncio.run(run())

    journal = write_ahead_journal_for(manifest_directory)
    assert journal is not None
    latest = journal.latest_entry(
        namespace="acme",
        collection="restricted",
        document_id="doc-ok",
    )
    assert latest is not None
    # The committed marker is the latest record for the triple.
    assert latest.state == STATE_MANIFEST_WRITTEN
    # Pending entries from the same ingest must be cleared by the commit.
    assert journal.pending_entries() == []
    # And the manifest got written, so the ingest landed normally.
    [entry] = read_entries(manifest_directory, namespace="acme", collection="restricted")
    assert entry.document_id == "doc-ok"
    assert entry.chunk_count == 1
    assert indexer.delete_document_calls == 0


def test_write_ahead_pending_entry_persisted_when_index_fails(tmp_path: Path) -> None:
    """The pending entry is on disk even after the index-failure rollback.

    The rollback delete is best-effort and the journal entry remains
    ``upserted_pending_manifest`` so the next ingest of the same triple
    re-runs the purge. Without this property, a flaky run could leave the
    store quietly populated and the operator with no journal trail.
    """
    manifest_directory = tmp_path / "manifest"
    indexer = _FailingThenCountingIndexer()
    events = EventBuffer()
    ingestor = _make_test_ingestor(
        manifest_directory=manifest_directory,
        indexer=indexer,
        event_sink=events,
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="index failed"):
            await ingestor.ingest_bytes(
                file_bytes=b"alpha beta",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="acme",
                collection="help",
                document_id="doc-pending-after-fail",
            )

    asyncio.run(run())

    journal = write_ahead_journal_for(manifest_directory)
    assert journal is not None
    latest = journal.latest_entry(
        namespace="acme",
        collection="help",
        document_id="doc-pending-after-fail",
    )
    assert latest is not None
    assert latest.state == STATE_UPSERTED_PENDING_MANIFEST


def test_write_ahead_entry_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        WriteAheadEntry(  # type: ignore[misc]
            "acme",
            "public",
            "doc",
            "sha",
            1,
            "state",
        )


def test_latest_entry_uses_file_order_not_timestamps(tmp_path: Path) -> None:
    # Monotonic clocks reset across OS reboots: a pre-reboot pending entry can
    # carry a larger timestamp than a post-reboot committed entry. File append
    # order, not the timestamp, must decide the latest state.
    journal = IngestWriteAheadJournal(directory=tmp_path)
    journal._append(
        WriteAheadEntry(
            namespace="acme",
            collection="help",
            document_id="doc-1",
            content_sha256="sha-old",
            expected_chunk_count=3,
            state=STATE_UPSERTED_PENDING_MANIFEST,
            created_at_ns=10**18,
            updated_at_ns=10**18,
        )
    )
    journal._append(
        WriteAheadEntry(
            namespace="acme",
            collection="help",
            document_id="doc-1",
            content_sha256="sha-new",
            expected_chunk_count=3,
            state=STATE_MANIFEST_WRITTEN,
            created_at_ns=1,
            updated_at_ns=1,
        )
    )

    latest = journal.latest_entry(
        namespace="acme", collection="help", document_id="doc-1"
    )
    assert latest is not None
    assert latest.state == STATE_MANIFEST_WRITTEN
    assert journal.pending_entries() == []


def test_record_committed_compacts_resolved_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_core._engine import core_ingest_write_ahead as module

    monkeypatch.setattr(module, "_COMPACT_MIN_BYTES", 1)
    journal = IngestWriteAheadJournal(directory=tmp_path)
    journal.record_pending(
        namespace="acme",
        collection="help",
        document_id="doc-resolved",
        content_sha256="sha-1",
        expected_chunk_count=2,
    )
    journal.record_committed(
        namespace="acme",
        collection="help",
        document_id="doc-resolved",
        content_sha256="sha-1",
        expected_chunk_count=2,
    )
    journal.record_pending(
        namespace="acme",
        collection="help",
        document_id="doc-pending",
        content_sha256="sha-2",
        expected_chunk_count=2,
    )

    lines = [
        line
        for line in journal.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    pending = journal.pending_entries()
    assert [entry.document_id for entry in pending] == ["doc-pending"]
