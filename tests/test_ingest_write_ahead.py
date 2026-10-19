"""Public keyword-only models and non-destructive ingest failure behavior."""

from __future__ import annotations
from rag_core.events.sink import EventSink

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from rag_core.ingest.document import CoreIngestor
from rag_core.core_models import (
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


def test_ocr_metadata_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        OcrMetadata("provider", "model")  # type: ignore[misc]


def test_processing_fingerprint_rejects_positional_args() -> None:
    with pytest.raises(TypeError):
        ProcessingFingerprint("rag-core-v1", "file")  # type: ignore[misc]


# -- index failures leave publication and cleanup to the store ------------


class _FailingThenCountingIndexer:
    """Fail indexing and record any unsafe broad cleanup attempt."""

    def __init__(self) -> None:
        self.index_document_calls = 0
        self.delete_document_calls = 0
        self.delete_targets: list[tuple[str, str, str]] = []

    async def index_document(
        self, req: object, *, event_sink: object = None
    ) -> IndexResult:
        self.index_document_calls += 1
        raise RuntimeError("index failed mid-upsert")

    async def delete_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
        collection: str,
    ) -> DeleteAck:
        self.delete_document_calls += 1
        self.delete_targets.append((tenant_id, collection, document_id))
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
        tenant_id: str = "",
        collection: str = "",
        document_id: str = "",
        event_sink: EventSink | None = None,
    ) -> PreparedDocument:
        del tenant_id, collection, document_id
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
        prepare_bytes=prepare_bytes,
        event_sink=event_sink,
        manifest_directory=manifest_directory,
    )


def test_index_failure_does_not_attempt_broad_rollback_delete(tmp_path: Path) -> None:
    """CoreIngestor must not delete a previously published document on failure."""
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
                tenant_id="acme",
                collection="help",
                document_id="doc-torn",
            )

    asyncio.run(run())

    assert indexer.index_document_calls == 1
    assert indexer.delete_document_calls == 0
    assert indexer.delete_targets == []
    # No manifest entry should survive a fresh-ingest torn write.
    assert read_entries(manifest_directory, tenant_id="acme", collection="help") == []
