"""Torn-write rollback on index failure.

Two guarantees:

1. ``kw_only=True`` on the public frozen dataclasses: you cannot
   construct ``PreparedChunk``, ``IngestedDocument``, ``DeleteDocumentResult``,
   ``CollectionManifestEntry``, ``CollectionManifest``, ``OcrMetadata`` or
   ``ProcessingFingerprint`` by accident with positional args.
2. The index-failure rollback in ``CoreIngestor`` calls
   ``indexer.delete_document`` BEFORE restoring the manifest, so a torn Qdrant
   batch upsert cannot leave residual chunks under the new content_sha256.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from rag_core._engine.core_ingest import CoreIngestor
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
from rag_core.manifest.persistence import read_entries
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
