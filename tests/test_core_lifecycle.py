import asyncio
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from rag_core import Engine
from rag_core.config import (
    SKIP_UNCHANGED_FAST,
    SKIP_UNCHANGED_MATERIALIZE,
    SkipUnchangedMode,
)
from rag_core._engine.core_builders import OCR_METADATA_KEY
from rag_core.file_io import compute_content_sha256
from rag_core.core_models import (
    IngestedDocument,
    OcrMetadata,
    OcrRoutingSignal,
    PreparedChunk,
    PreparedDocument,
)
from rag_core._engine.core_runtime import resolve_processing_version
from rag_core.documents.ocr_provider_names import (
    DEFAULT_MISTRAL_OCR_MODEL,
    MISTRAL_OCR_PROVIDER,
)
from rag_core.search.lexical_sidecar import LexicalSidecarRecord, PortableLexicalSidecar
from rag_core.search.provider_protocols import SearchSidecar
from rag_core.search.request_models import (
    SearchSidecarQuery,
    StoredDocumentRecord,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeSearchSidecar,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


def _processing_version(
    *,
    base_version: str = "rag_core_processing_v3",
    source_type: str = "file",
) -> str:
    return resolve_processing_version(
        configured_version=base_version,
        source_type=source_type,
    ).serialize()


def _make_core(
    store: RecordingVectorStore | None = None,
    *,
    search_sidecar: SearchSidecar | None = None,
    skip_unchanged: SkipUnchangedMode = SKIP_UNCHANGED_FAST,
) -> tuple[Engine, RecordingVectorStore]:
    store = store if store is not None else RecordingVectorStore()
    base_config = make_test_config(
        embedding_model="text-embedding-3-small",
        embedding_dimensions=4,
    )
    core = Engine(
        replace(base_config, ingest=replace(base_config.ingest, skip_unchanged=skip_unchanged)),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        search_sidecar=search_sidecar,
    )
    return core, store


def _stored_record(
    *,
    document_id: str = "doc_unchanged",
    namespace: str = "team-space",
    collection: str = "corpus-1",
    document_key: str = "/docs/guide.txt",
    content: bytes = b"same bytes",
    processing_version: str | None = None,
    chunk_count: int = 1,
) -> StoredDocumentRecord:
    return StoredDocumentRecord(
        document_id=document_id,
        namespace=namespace,
        collection=collection,
        document_key=document_key,
        content_sha256=compute_content_sha256(content),
        processing_version=processing_version if processing_version is not None else _processing_version(),
        chunk_count=chunk_count,
    )


def test_ingest_uses_stable_document_key_and_builds_manifest() -> None:
    async def _run() -> None:
        core, store = _make_core()
        try:
            doc = await core.add_bytes(
                file_bytes=b"alpha fox query",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
                collection="corpus-1",
                path="/docs/guide.txt",
            )
            manifest = core.build_collection_manifest(
                namespace="team-space",
                collection="corpus-1",
                documents=[doc],
            )
        finally:
            await core.close()

        assert doc.document_key == "/docs/guide.txt"
        assert doc.document_id.startswith("doc_")
        assert doc.content_sha256 is not None
        assert doc.ingest_state == "created"
        assert doc.collection_name == "rag_core_chunks__fake_embedding_4d"
        assert doc.processing_version == _processing_version()
        assert store.get_document_record_calls == [
            ("team-space", "corpus-1", doc.document_id, None)
        ]
        stored = store.document_records[("team-space", "corpus-1", doc.document_id)]
        assert stored.processing_version == _processing_version()
        assert manifest.document_count == 1
        assert manifest.chunk_count == doc.chunk_count
        assert manifest.source_document_ids == (doc.document_id,)
        assert manifest.embedding_model == "fake-embedding"
        [entry] = manifest.entries
        assert entry.document_key == doc.document_key
        assert entry.content_sha256 == doc.content_sha256

    asyncio.run(_run())


def test_ingest_skips_reindex_when_content_is_unchanged() -> None:
    async def _run() -> None:
        existing = _stored_record(chunk_count=3)
        core, store = _make_core(
            RecordingVectorStore(
                document_records={("team-space", "corpus-1", "doc_unchanged"): existing},
            ),
            skip_unchanged=SKIP_UNCHANGED_MATERIALIZE,
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
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
                    namespace="team-space",
                    collection="corpus-1",
                    path="/docs/created.txt",
                    metadata={"title": title},
                )
            else:
                doc = await core.add_bytes(
                    file_bytes=b"same bytes",
                    filename="guide.txt",
                    mime_type="text/plain",
                    namespace="team-space",
                    collection="corpus-1",
                    document_id="doc-existing",
                    path="/docs/guide.txt",
                    metadata={"title": title},
                )
        finally:
            await core.close()

        assert doc.metadata["title"] == title

    asyncio.run(_run())


def test_unchanged_ingest_does_not_mutate_sidecar_records() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar()
        store = RecordingVectorStore(
            document_records={("team-space", "corpus-1", "doc_unchanged"): _stored_record(chunk_count=3)},
        )
        core, _ = _make_core(store, search_sidecar=sidecar)

        try:
            await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
                collection="corpus-1",
                document_id="doc_unchanged",
                path="/docs/guide.txt",
            )
        finally:
            await core.close()

        assert sidecar.upserted == []
        assert sidecar.deleted == []

    asyncio.run(_run())


def test_ingest_marks_existing_document_as_replaced_when_hash_changes() -> None:
    async def _run() -> None:
        # An old content_sha256 forces the replace path on the next ingest.
        stale = StoredDocumentRecord(
            document_id="doc_existing",
            namespace="team-space",
            collection="corpus-1",
            document_key="/docs/guide.txt",
            content_sha256="old-hash",
            chunk_count=1,
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
                namespace="team-space",
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
                document_records={("team-space", "corpus-1", "doc_unchanged"): existing},
            )
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
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


def test_unchanged_v2_processing_version_reindexes_after_v3_default() -> None:
    async def _run() -> None:
        existing = _stored_record(
            processing_version=_processing_version(
                base_version="rag_core_processing_v2"
            )
        )
        core, store = _make_core(
            RecordingVectorStore(
                document_records={("team-space", "corpus-1", "doc_unchanged"): existing},
            )
        )

        try:
            doc = await core.add_bytes(
                file_bytes=b"same bytes",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
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


def test_ingest_and_delete_keep_sidecar_in_sync() -> None:
    async def _run() -> None:
        sidecar = FakeSearchSidecar()
        core, _ = _make_core(search_sidecar=sidecar)

        try:
            doc = await core.add_bytes(
                file_bytes=b"alpha fox query",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
                collection="corpus-1",
                path="/docs/guide.txt",
            )
            await core.delete_document(
                document_id=doc.document_id,
                namespace="team-space",
                collection="corpus-1",
            )
        finally:
            await core.close()

        assert len(sidecar.upserted) == doc.chunk_count
        first_record = cast(LexicalSidecarRecord, sidecar.upserted[0])
        assert first_record.result.document_key == doc.document_key
        assert first_record.result.content_sha256 == doc.content_sha256
        assert first_record.result.chunker_strategy == "prechunked"
        assert first_record.result.result_type == "text"
        # delete fires once on initial ingest replace-check, once on explicit delete.
        assert sidecar.deleted == [
            ("team-space", doc.document_id),
            ("team-space", doc.document_id),
        ]

    asyncio.run(_run())


def test_delete_document_requires_collection() -> None:
    async def _run() -> None:
        core, _ = _make_core()
        try:
            with pytest.raises(ValueError, match="collection must be a non-empty string"):
                await core.delete_document(
                    document_id="doc-1", namespace="team-space", collection=""
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
            namespace="team-space",
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
                namespace="team-space",
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


def test_manifest_file_is_preview_only() -> None:
    async def _run() -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        try:
            handle.write(b"preview only")
            handle.flush()
            file_path = Path(handle.name)
        finally:
            handle.close()

        core, store = _make_core()
        try:
            entry = await core.manifest_file(
                file_path, namespace="team-space", collection="corpus-1"
            )
        finally:
            await core.close()
            file_path.unlink(missing_ok=True)

        assert entry.document_key == str(file_path)
        assert store.upsert_calls == []
        assert store.delete_calls == []

    asyncio.run(_run())


def test_reingest_shrinks_sidecar_results_without_stale_chunks() -> None:
    async def _run() -> None:
        sidecar = PortableLexicalSidecar([])
        core, _ = _make_core(search_sidecar=sidecar)

        async def fake_prepare_bytes(
            *,
            file_bytes: bytes,
            filename: str,
            mime_type: str,
            path: str | None = None,
            namespace: str = "",
            collection: str = "",
            document_id: str = "",
        ) -> PreparedDocument:
            chunks = (
                [
                    PreparedChunk(chunk_index=0, text="alpha", embedding_text="alpha", word_count=1),
                    PreparedChunk(chunk_index=1, text="beta", embedding_text="beta", word_count=1),
                ]
                if file_bytes == b"first"
                else [
                    PreparedChunk(chunk_index=0, text="alpha", embedding_text="alpha", word_count=1),
                ]
            )
            return PreparedDocument(
                filename=filename,
                mime_type=mime_type,
                markdown="alpha\n\nbeta" if file_bytes == b"first" else "alpha",
                chunks=chunks,
                metadata={"parser": "local:text"},
                path=path,
                ocr=OcrRoutingSignal(),
            )

        core.prepare_bytes = fake_prepare_bytes  # type: ignore[method-assign]

        try:
            first = await core.add_bytes(
                file_bytes=b"first",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
                collection="corpus-1",
                path="/docs/guide.txt",
            )
            second = await core.add_bytes(
                file_bytes=b"second",
                filename="guide.txt",
                mime_type="text/plain",
                namespace="team-space",
                collection="corpus-1",
                document_id=first.document_id,
                path="/docs/guide.txt",
            )
            removed = await sidecar.search(
                SearchSidecarQuery(
                    query="beta",
                    namespace="team-space",
                    collections=["corpus-1"],
                )
            )
        finally:
            await core.close()

        assert second.chunk_count == 1
        # Stale "beta" chunk must be gone from sidecar.
        assert removed == []

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("pages_used", "explicit_page_count", "expected_ocr_pages"),
    [
        pytest.param((0, 2), 2, 2, id="pages-used-only"),
        pytest.param((0,), 4, 4, id="explicit-page-count-wins"),
    ],
)
def test_corpus_manifest_counts_ocr_usage(
    pages_used: tuple[int, ...],
    explicit_page_count: int,
    expected_ocr_pages: int,
) -> None:
    core, _ = _make_core()
    document = IngestedDocument(
        document_id="doc-1",
        namespace="team-space",
        collection="corpus-1",
        chunk_count=1,
        filename="scan.pdf",
        mime_type="application/pdf",
        document_key="/docs/scan.pdf",
        content_sha256="hash-1",
        metadata={
            "parser": "local:pdf_inspector",
            "needs_ocr": False,
            OCR_METADATA_KEY: asdict(
                OcrMetadata(
                    provider=MISTRAL_OCR_PROVIDER,
                    model=DEFAULT_MISTRAL_OCR_MODEL,
                    pages_used=pages_used,
                    page_count=explicit_page_count,
                    merge_mode="append",
                )
            ),
        },
        ocr=OcrRoutingSignal(needed=False, page_indices=[]),
    )
    try:
        entry = core.build_manifest_entry(document=document)
        manifest = core.build_collection_manifest(
            namespace="team-space",
            collection="corpus-1",
            documents=[document],
        )
    finally:
        asyncio.run(core.close())

    # OCR provider presence (not the older needs_ocr flag) drives the count.
    assert entry.needs_ocr is True
    assert manifest.ocr_document_count == 1
    assert manifest.ocr_page_count == expected_ocr_pages


class _CloseResource:
    def __init__(self, *, name: str, fail: bool = False, async_close: bool = False) -> None:
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


def test_close_attempts_all_resources_and_aggregates_failures() -> None:
    async def _run() -> None:
        store = RecordingVectorStore()
        embedding_cache = _CloseResource(name="embedding_cache", fail=True)
        chunk_context_cache = _CloseResource(name="chunk_context_cache", async_close=True)

        async def fail_store_close() -> None:
            store.operations.append("close")
            store.close_calls += 1
            raise RuntimeError("vector_store close failed")

        store.close = fail_store_close  # type: ignore[method-assign]

        core = Engine(
            make_test_config(
                embedding_model="text-embedding-3-small",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            embedding_cache=cast(Any, embedding_cache),
            chunk_context_cache=cast(Any, chunk_context_cache),
        )

        with pytest.raises(ExceptionGroup) as excinfo:
            await core.close()

        assert store.close_calls == 1
        assert embedding_cache.close_calls == 1
        assert chunk_context_cache.close_calls == 1
        assert [str(exc) for exc in excinfo.value.exceptions] == [
            "vector_store close failed",
            "embedding_cache close failed",
        ]
        notes = [
            note
            for exc in excinfo.value.exceptions
            for note in getattr(exc, "__notes__", [])
        ]
        assert "while closing Engine resource: vector_store" in notes
        assert "while closing Engine resource: embedding_cache" in notes

    asyncio.run(_run())


def test_close_re_raises_single_error_after_other_resources_are_closed() -> None:
    async def _run() -> None:
        store = RecordingVectorStore()
        embedding_cache = _CloseResource(name="embedding_cache")
        chunk_context_cache = _CloseResource(name="chunk_context_cache", async_close=True)

        async def fail_store_close() -> None:
            store.operations.append("close")
            store.close_calls += 1
            raise RuntimeError("vector_store close failed")

        store.close = fail_store_close  # type: ignore[method-assign]

        core = Engine(
            make_test_config(
                embedding_model="text-embedding-3-small",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            embedding_cache=cast(Any, embedding_cache),
            chunk_context_cache=cast(Any, chunk_context_cache),
        )

        with pytest.raises(RuntimeError, match="vector_store close failed") as excinfo:
            await core.close()

        assert store.close_calls == 1
        assert embedding_cache.close_calls == 1
        assert chunk_context_cache.close_calls == 1
        assert "while closing Engine resource: vector_store" in getattr(
            excinfo.value, "__notes__", []
        )

    asyncio.run(_run())
