import asyncio
from dataclasses import replace
from typing import Any, Callable

import pytest

from rag_core.config.ingest_config import DEFAULT_PROCESSING_VERSION
from rag_core.core import Engine
from rag_core.config import (
    SKIP_UNCHANGED_FAST,
    SKIP_UNCHANGED_MATERIALIZE,
    SkipUnchangedMode,
)
from rag_core.ingest.manifest import (
    build_manifest_entry,
)
from rag_core.file_io import compute_content_sha256
from rag_core._engine.core_runtime import resolve_processing_version
from rag_core.search.request_models import (
    StoredDocumentRecord,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


def _processing_version(
    *,
    base_version: str = DEFAULT_PROCESSING_VERSION,
    source_type: str = "file",
) -> str:
    return resolve_processing_version(
        configured_version=base_version,
        source_type=source_type,
        sparse_index=True,
        sparse_model=FakeSparseEmbedder.model_name,
    ).serialize()


def _stored_record(
    *,
    document_id: str = "doc_unchanged",
    tenant_id: str = "team-space",
    collection: str = "corpus-1",
    document_key: str = "/docs/guide.txt",
    content: bytes = b"same bytes",
    processing_version: str | None = None,
    chunk_count: int = 1,
) -> StoredDocumentRecord:
    return StoredDocumentRecord(
        document_id=document_id,
        tenant_id=tenant_id,
        collection=collection,
        document_key=document_key,
        content_sha256=compute_content_sha256(content),
        processing_version=processing_version
        if processing_version is not None
        else _processing_version(),
        chunk_count=chunk_count,
        expected_chunk_count=chunk_count,
        revision="1" * 32,
        committed=True,
    )


def test_ingest_uses_stable_document_key_and_builds_manifest() -> None:
    async def _run() -> None:
        core, store = _make_core()
        try:
            doc = await core.add_bytes(
                file_bytes=b"alpha fox query",
                filename="guide.txt",
                mime_type="text/plain",
                tenant_id="team-space",
                collection="corpus-1",
                path="/docs/guide.txt",
            )
            entry = build_manifest_entry(doc)
        finally:
            await core.close()

        assert doc.document_key == "/docs/guide.txt"
        assert doc.document_id.startswith("doc_")
        assert doc.content_sha256 is not None
        assert doc.ingest_state == "created"
        assert doc.collection_name == "rag_core_chunks__fake_embedding_4d"
        assert doc.processing_version == _processing_version()
        assert store.get_document_record_calls == [
            ("team-space", "corpus-1", doc.document_id, None),
            ("team-space", "corpus-1", None, "/docs/guide.txt"),
        ]
        stored = store.document_records[("team-space", "corpus-1", doc.document_id)]
        assert stored.processing_version == _processing_version()
        assert entry.document_key == doc.document_key
        assert entry.content_sha256 == doc.content_sha256

    asyncio.run(_run())


def test_ingest_skips_reindex_when_content_is_unchanged() -> None:
    async def _run() -> None:
        existing = _stored_record(chunk_count=3)
        core, store = _make_core(
            RecordingVectorStore(
                document_records={
                    ("team-space", "corpus-1", "doc_unchanged"): existing
                },
            ),
            skip_unchanged=SKIP_UNCHANGED_MATERIALIZE,
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                tenant_id="team-space",
                collection="corpus-1",
                document_id="doc_unchanged",
                path="/docs/guide.txt",
            )
        finally:
            await core.close()

        assert doc.ingest_state == "unchanged"
        assert doc.chunk_count == existing.chunk_count
        assert doc.metadata["parser"] == "local:text"
        assert doc.ocr.needed is False
        assert store.upsert_calls == []

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("scenario", "title"),
    [("created", "Created Title"), ("unchanged", "Guide Title")],
)
def test_ingest_returns_caller_metadata(scenario: str, title: str) -> None:
    async def _run() -> None:
        existing = _stored_record(document_id="doc-existing")
        core, _ = _make_core(
            RecordingVectorStore(
                document_records={("team-space", "corpus-1", "doc-existing"): existing},
            )
        )

        try:
            if scenario == "created":
                doc = await core.add_bytes(
                    file_bytes=b"new bytes",
                    filename="created.txt",
                    mime_type="text/plain",
                    tenant_id="team-space",
                    collection="corpus-1",
                    path="/docs/created.txt",
                    metadata={"title": title},
                )
            else:
                doc = await core.add_bytes(
                    file_bytes=b"same bytes",
                    filename="guide.txt",
                    mime_type="text/plain",
                    tenant_id="team-space",
                    collection="corpus-1",
                    document_id="doc-existing",
                    path="/docs/guide.txt",
                    metadata={"title": title},
                )
        finally:
            await core.close()

        assert doc.metadata["title"] == title

    asyncio.run(_run())


def test_ingest_marks_existing_document_as_replaced_when_hash_changes() -> None:
    async def _run() -> None:
        # An old content_sha256 forces the replace path on the next ingest.
        stale = StoredDocumentRecord(
            document_id="doc_existing",
            tenant_id="team-space",
            collection="corpus-1",
            document_key="/docs/guide.txt",
            content_sha256="old-hash",
            chunk_count=1,
            expected_chunk_count=1,
            revision="1" * 32,
            committed=True,
        )
        core, store = _make_core(
            RecordingVectorStore(
                document_records={("team-space", "corpus-1", "doc_existing"): stale},
            )
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"new bytes",
                filename="guide.txt",
                mime_type="text/plain",
                tenant_id="team-space",
                collection="corpus-1",
                document_id="doc_existing",
                path="/docs/guide.txt",
            )
        finally:
            await core.close()

        assert doc.ingest_state == "replaced"
        assert doc.replaced_existing is True
        assert store.operations[:2] == ["get_document_record", "upsert"]

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("trigger", "build_existing"),
    [
        pytest.param(
            "force_reindex",
            lambda: _stored_record(),
            id="force-reindex",
        ),
        pytest.param(
            "processing_version_drift",
            lambda: _stored_record(
                processing_version=_processing_version(
                    base_version="rag_core_processing_old"
                ),
            ),
            id="processing-version-drift",
        ),
    ],
)
def test_unchanged_content_reindexes_when_trigger_present(
    trigger: str,
    build_existing: Callable[[], StoredDocumentRecord],
) -> None:
    async def _run() -> None:
        existing = build_existing()
        core, store = _make_core(
            RecordingVectorStore(
                document_records={
                    ("team-space", "corpus-1", "doc_unchanged"): existing
                },
            )
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                tenant_id="team-space",
                collection="corpus-1",
                document_id="doc_unchanged",
                path="/docs/guide.txt",
                force_reindex=(trigger == "force_reindex"),
            )
        finally:
            await core.close()

        assert doc.ingest_state == "reindexed"
        assert doc.replaced_existing is True
        assert doc.processing_version == _processing_version()
        assert store.operations[:2] == ["get_document_record", "upsert"]

    asyncio.run(_run())


def test_unchanged_v3_processing_version_reindexes_after_v4_default() -> None:
    async def _run() -> None:
        existing = _stored_record(
            processing_version=_processing_version(
                base_version="rag_core_processing_v3"
            )
        )
        core, store = _make_core(
            RecordingVectorStore(
                document_records={
                    ("team-space", "corpus-1", "doc_unchanged"): existing
                },
            )
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                tenant_id="team-space",
                collection="corpus-1",
                document_id="doc_unchanged",
                path="/docs/guide.txt",
            )
        finally:
            await core.close()

        assert doc.ingest_state == "reindexed"
        assert doc.processing_version == _processing_version()
        assert store.operations[:2] == ["get_document_record", "upsert"]

    asyncio.run(_run())


def test_delete_document_requires_collection() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(
                ValueError, match="collection must be a non-empty string"
            ):
                await core.delete_document(
                    document_id="doc-1", tenant_id="team-space", collection=""
                )
        finally:
            await core.close()

    asyncio.run(_run())


def test_explicit_document_ids_are_scoped_by_collection_for_existing_checks() -> None:
    async def _run() -> None:
        # Shared document_id exists in a different collection -> must not be
        # treated as existing.
        other_corpus_record = StoredDocumentRecord(
            document_id="doc-shared",
            tenant_id="team-space",
            collection="corpus-2",
            document_key="/docs/guide.txt",
            content_sha256=compute_content_sha256(b"same bytes"),
            chunk_count=5,
        )
        core, store = _make_core(
            RecordingVectorStore(
                document_records={
                    ("team-space", "corpus-2", "doc-shared"): other_corpus_record,
                }
            )
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                tenant_id="team-space",
                collection="corpus-1",
                document_id="doc-shared",
                path="/docs/guide.txt",
            )
        finally:
            await core.close()

        assert doc.ingest_state == "created"
        assert store.get_document_record_calls == [
            ("team-space", "corpus-1", "doc-shared", None)
        ]

    asyncio.run(_run())


class _CloseResource:
    def __init__(
        self, *, name: str, fail: bool = False, async_close: bool = False
    ) -> None:
        self.name = name
        self.fail = fail
        self.async_close = async_close
        self.close_calls = 0

    def close(self) -> Any:
        self.close_calls += 1
        if self.async_close:
            return self._close_async()
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")
        return None

    async def _close_async(self) -> None:
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")


def _make_core(
    store: RecordingVectorStore | None = None,
    *,
    skip_unchanged: SkipUnchangedMode = SKIP_UNCHANGED_FAST,
) -> tuple[Engine, RecordingVectorStore]:
    store = store if store is not None else RecordingVectorStore()
    base_config = make_test_config(
        embedding_model="text-embedding-3-small",
        embedding_dimensions=4,
    )
    core = Engine(
        replace(
            base_config,
            ingest=replace(base_config.ingest, skip_unchanged=skip_unchanged),
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )
    return core, store
