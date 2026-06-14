import asyncio

import pytest

from rag_core.core_models import PreparedChunk
from rag_core.search.context_pack import build_context_pack
from rag_core.search.indexer import DocumentIndexer, IndexRequest
from rag_core.search.indexer_embeddings import prepare_index_data
from rag_core.search.indexer_points import make_point_id
from rag_core.search.stored_payload import payload_to_result
from rag_core.search.provider_protocols import (
    SparseEmbedder,
    StoreCapabilities,
)

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    FakeSparseEmbedderNoMulti,
    RecordingVectorStore,
)


class _DimensionedRecordingVectorStore(RecordingVectorStore):
    def __init__(self, *, dense_dimensions: int) -> None:
        super().__init__()
        self.capabilities = StoreCapabilities(
            per_point_delete=True,
            document_record_lookup=True,
            dense_vector_dimensions=dense_dimensions,
        )


class _AccidentallyDimensionedRecordingVectorStore(RecordingVectorStore):
    # Has a dense_dimensions attribute but does NOT advertise it via capabilities,
    # so the indexer must ignore it.
    dense_dimensions = 2


class _WrongDimensionEmbeddingProvider(FakeEmbeddingProvider):
    @property
    def dimensions(self) -> int:
        return 3

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def _index_request(**overrides: object) -> IndexRequest:
    base: dict[str, object] = {
        "document_id": "doc-1",
        "corpus_id": "corpus-1",
        "namespace": "team-space",
        "text": "unused",
        "filename": "report.txt",
        "mime_type": "text/plain",
        "source_type": "file",
    }
    base.update(overrides)
    return IndexRequest(**base)  # type: ignore[arg-type]


def _make_indexer(
    store: RecordingVectorStore | None = None,
    *,
    embedding: FakeEmbeddingProvider | None = None,
    sparse: SparseEmbedder | None = None,
) -> tuple[DocumentIndexer, RecordingVectorStore]:
    store = store if store is not None else RecordingVectorStore()
    indexer = DocumentIndexer(
        embedding_provider=embedding or FakeEmbeddingProvider(),
        sparse_embedder=sparse or FakeSparseEmbedder(include_extra_channel=False),
        vector_store=store,
    )
    return indexer, store


def test_prepare_index_data_rejects_dense_vector_dimension_mismatch() -> None:
    async def _run() -> None:
        with pytest.raises(
            ValueError,
            match=(
                "Dense embedding dimension mismatch at chunk index 0.*"
                "expected 3 dimensions, got 2"
            ),
        ):
            await prepare_index_data(
                req=_index_request(pre_chunked_texts=["fox query"]),
                embedding_provider=_WrongDimensionEmbeddingProvider(),
                sparse_embedder=FakeSparseEmbedder(include_extra_channel=False),
            )

    asyncio.run(_run())


def test_index_document_rejects_store_embedding_dimension_mismatch_before_upsert() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider(vocabulary=("fox", "query", "context"))
        indexer, store = _make_indexer(
            _DimensionedRecordingVectorStore(dense_dimensions=2),
            embedding=embedding,
        )

        with pytest.raises(
            ValueError,
            match="embedding provider produces 3 dimensions, but vector store expects 2",
        ):
            await indexer.index_document(_index_request(pre_chunked_texts=["fox query"]))

        assert store.upsert_calls == []
        assert store.operations == []

    asyncio.run(_run())


def test_index_document_deletes_empty_prepared_document_before_dimension_check() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer(
            _DimensionedRecordingVectorStore(dense_dimensions=2),
            embedding=FakeEmbeddingProvider(vocabulary=("fox", "query", "context")),
        )

        result = await indexer.index_document(
            _index_request(text="", filename="empty.txt"),
        )

        assert result.chunk_count == 0
        assert store.upsert_calls == []
        assert [call.document_id for call in store.delete_calls] == ["doc-1"]
        assert store.operations == ["delete"]

    asyncio.run(_run())


def test_index_document_ignores_accidental_dense_dimensions_without_capability() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer(
            _AccidentallyDimensionedRecordingVectorStore(),
            embedding=FakeEmbeddingProvider(vocabulary=("fox", "query", "context")),
        )

        await indexer.index_document(_index_request(pre_chunked_texts=["fox query"]))

        assert len(store.upsert_calls) == 1
        assert store.operations == ["upsert"]

    asyncio.run(_run())


def test_index_document_uses_payload_chunks_and_contextual_dense_text() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider(vocabulary=("original", "context", "fox"))
        sparse = FakeSparseEmbedder()
        indexer, store = _make_indexer(embedding=embedding, sparse=sparse)

        result = await indexer.index_document(
            _index_request(
                namespace=" team-space ",
                path="/docs/report.txt",
                extra_fields={"team": "search"},
                processing_version="processing-v-test",
                pre_chunked_texts=["original fox"],
                embedding_chunk_texts=["context fox"],
            )
        )

        point = store.upsert_calls[0][0]
        assert result.chunk_count == 1
        assert store.operations == ["upsert"]
        assert store.delete_calls == []
        assert point.payload["namespace"] == "team-space"
        assert point.payload["processing_version"] == "processing-v-test"
        assert point.payload["team"] == "search"
        payload_text = str(point.payload["text"])
        assert payload_text == "original fox"
        assert "# Metadata" not in payload_text
        assert "# Content" not in payload_text
        assert "context fox" not in payload_text
        assert "source_type: file" in embedding.embed_texts_calls[0][0]
        assert "name: report.txt" in embedding.embed_texts_calls[0][0]
        assert "# Metadata" not in embedding.embed_texts_calls[0][0]
        assert "# Content" not in embedding.embed_texts_calls[0][0]
        # Dense vector uses embedding_chunk_texts (context-only counts), payload uses raw chunk.
        assert point.dense_vector == [0.0, 1.0, 1.0]
        assert "context fox" in sparse.embed_texts_multi_calls[0][0]
        assert "original fox" not in sparse.embed_texts_multi_calls[0][0]

    asyncio.run(_run())


def test_index_document_persists_filterable_chunk_metadata() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer()

        await indexer.index_document(
            _index_request(
                extra_fields={"team": "support", "title": "Display Title"},
                pre_chunked_texts=["page one"],
                chunk_metadata=[
                    {
                        "language": "en",
                        "published_at": 1700.0,
                        "featured": True,
                        "text": "should not overwrite payload text",
                        "nested": {"ignored": True},
                    }
                ],
            )
        )

        payload = store.upsert_calls[0][0].payload
        result = payload_to_result(point_id="point-1", payload=payload, score=0.8)
        assert payload["team"] == "support"
        assert payload["language"] == "en"
        assert payload["published_at"] == 1700.0
        assert payload["featured"] is True
        assert payload["text"] == "page one"
        assert "should not overwrite payload text" not in str(payload["text"])
        assert "nested" not in payload
        assert result.metadata["team"] == "support"
        assert result.metadata["language"] == "en"
        assert result.text == "page one"

    asyncio.run(_run())


def test_index_document_rejects_embedding_chunk_lengths_that_do_not_match() -> None:
    async def _run() -> None:
        embedding = FakeEmbeddingProvider(vocabulary=("original", "context", "fox"))
        indexer, store = _make_indexer(embedding=embedding, sparse=FakeSparseEmbedder())

        with pytest.raises(ValueError, match="embedding_texts length mismatch"):
            await indexer.index_document(
                _index_request(
                    pre_chunked_texts=["original fox"],
                    embedding_chunk_texts=["context fox", "extra"],
                )
            )

        assert store.upsert_calls == []

    asyncio.run(_run())


def test_index_document_falls_back_to_bm25_only_when_multi_channel_is_unavailable() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer(sparse=FakeSparseEmbedderNoMulti())

        await indexer.index_document(_index_request(pre_chunked_texts=["fox query"]))

        point = store.upsert_calls[0][0]
        assert set(point.sparse_vectors) == {"bm25"}
        assert point.sparse_vector == point.sparse_vectors["bm25"]

    asyncio.run(_run())


def test_index_document_builds_section_payload_from_mappings() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer()

        await indexer.index_document(
            _index_request(
                filename="slides.pdf",
                mime_type="application/pdf",
                pre_chunked_texts=["page one", "page two"],
                section_mappings=[
                    {
                        "chunk_index": 1,
                        "section_id": "sec-2",
                        "section_path": "Intro > Details",
                        "page_number": 3,
                        "result_type": "image",
                        "thumbnail_url": "thumb.png",
                    }
                ],
            )
        )

        point = store.upsert_calls[0][1]
        assert point.payload["section_id"] == "sec-2"
        assert point.payload["section_path"] == "Intro > Details"
        assert point.payload["section_title"] == "Details"
        assert point.payload["page_number"] == 3
        assert point.payload["result_type"] == "image"
        assert point.payload["thumbnail_url"] == "thumb.png"

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("filename", "mime_type", "pre_chunked_texts", "chunk_metadata", "expected"),
    [
        pytest.param(
            "report.pdf",
            "application/pdf",
            ["page one", "page two"],
            [
                {"page_number": 1, "page_index": 0},
                {"page_number": 2, "page_index": 1},
            ],
            [{"page_number": 1, "page_index": 0}, {"page_number": 2, "page_index": 1}],
            id="pdf-page-locator",
        ),
        pytest.param(
            "review.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ["## Slide 1\n\n### Figures\n\nSlide 1 Figure 1."],
            [
                {
                    "section_path": "Slide 1 > Figures",
                    "section_title": "Figures",
                    "slide_number": 1,
                    "figure_id": "fig:slide:1:1",
                    "figure_caption": "Architecture diagram",
                }
            ],
            [
                {
                    "slide_number": 1,
                    "figure_id": "fig:slide:1:1",
                    "figure_caption": "Architecture diagram",
                }
            ],
            id="pptx-figure-locator",
        ),
    ],
)
def test_index_document_preserves_prepared_chunk_locators(
    filename: str,
    mime_type: str,
    pre_chunked_texts: list[str],
    chunk_metadata: list[dict[str, object]],
    expected: list[dict[str, object]],
) -> None:
    async def _run() -> None:
        indexer, store = _make_indexer()

        await indexer.index_document(
            _index_request(
                filename=filename,
                mime_type=mime_type,
                pre_chunked_texts=pre_chunked_texts,
                chunk_metadata=chunk_metadata,
            )
        )

        points = store.upsert_calls[0]
        assert len(points) == len(expected)
        for point, expected_fields in zip(points, expected, strict=True):
            for key, value in expected_fields.items():
                assert point.payload[key] == value

    asyncio.run(_run())


def test_index_document_does_not_store_local_document_path() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer()

        await indexer.index_document(
            _index_request(
                path="/Users/private/workspace/notes.md",
                document_path="/Users/private/workspace/notes.md",
                pre_chunked_texts=["private local note"],
            )
        )

        payload = store.upsert_calls[0][0].payload
        assert payload["document_path"] is None
        assert "/Users/private" not in repr(payload)

    asyncio.run(_run())


def test_index_document_preserves_redacted_url_document_path() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer()

        await indexer.index_document(
            _index_request(
                source_type="url",
                path="https://example.com/docs?redacted",
                document_path="https://example.com/docs?redacted",
                document_key="url:https://example.com/docs?redacted|query_sha256:abc",
                pre_chunked_texts=["remote note"],
            )
        )

        payload = store.upsert_calls[0][0].payload
        assert payload["document_path"] == "https://example.com/docs?redacted"

    asyncio.run(_run())


def test_index_document_emits_figure_locator_through_context_pack() -> None:
    """End-to-end: figure locator survives indexer -> stored payload -> context pack."""

    async def _run() -> None:
        indexer, store = _make_indexer()
        text = "## Slide 1\n\n### Figures\n\nSlide 1 Figure 1."
        metadata = {
            "section_path": "Slide 1 > Figures",
            "section_title": "Figures",
            "slide_number": 1,
            "figure_id": "fig:slide:1:1",
            "figure_caption": "Architecture diagram",
            "figure_thumbnail_url": "https://cdn.example.com/fig-1.png",
        }

        await indexer.index_document(
            _index_request(
                filename="review.pptx",
                mime_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.presentation"
                ),
                text=text,
                prepared_chunks=[
                    PreparedChunk(
                        chunk_index=0,
                        text=text,
                        embedding_text=text,
                        word_count=len(text.split()),
                        start_char=0,
                        end_char=len(text),
                        metadata=metadata,
                    )
                ],
            )
        )

        point = store.upsert_calls[0][0]
        assert point.payload["figure_thumbnail_url"] == "https://cdn.example.com/fig-1.png"
        result = payload_to_result(point_id=point.id, payload=point.payload, score=0.8)
        assert result.figure_thumbnail_url == "https://cdn.example.com/fig-1.png"
        pack = build_context_pack([result], query="diagram")
        assert pack.snippets[0].locator.figure_id == "fig:slide:1:1"
        assert (
            pack.snippets[0].locator.figure_thumbnail_url
            == "https://cdn.example.com/fig-1.png"
        )

    asyncio.run(_run())


def test_index_document_derives_locators_from_raw_text() -> None:
    """When chunk_metadata is absent, the router-derived locators must reach payload."""

    async def _run() -> None:
        indexer, store = _make_indexer()

        await indexer.index_document(
            _index_request(
                text=(
                    "## Page 1\n\nAlpha page content for retrieval citations.\n\n"
                    "## Page 2\n\nBeta page content for retrieval citations.\n"
                ),
                filename="report.pdf",
                mime_type="application/pdf",
            )
        )

        first, second = store.upsert_calls[0]
        assert (first.payload["page_number"], first.payload["page_index"]) == (1, 0)
        assert (second.payload["page_number"], second.payload["page_index"]) == (2, 1)

    asyncio.run(_run())


def test_index_document_prefers_metadata_title_for_display_title() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer()

        await indexer.index_document(
            _index_request(
                extra_fields={"title": "Quarterly Report"},
                pre_chunked_texts=["page one"],
            )
        )

        assert store.upsert_calls[0][0].payload["title"] == "Quarterly Report"

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("operation", "expected_message"),
    [
        ("index", "namespace is required for indexing"),
        ("delete", "namespace is required for delete_document"),
    ],
)
def test_namespace_must_be_non_blank(operation: str, expected_message: str) -> None:
    async def _run() -> None:
        indexer, _ = _make_indexer(sparse=FakeSparseEmbedder())
        with pytest.raises(ValueError, match=expected_message):
            if operation == "index":
                await indexer.index_document(
                    _index_request(namespace="   ", pre_chunked_texts=["fox query"]),
                )
            else:
                await indexer.delete_document(
                    document_id="doc-1", namespace=" ", corpus_id="corpus-1"
                )

    asyncio.run(_run())


def test_index_document_deletes_stale_tail_chunks_before_upsert() -> None:
    async def _run() -> None:
        indexer, store = _make_indexer(sparse=FakeSparseEmbedder())

        result = await indexer.index_document(
            _index_request(existing_chunk_count=3, pre_chunked_texts=["page one"]),
        )

        expected_stale_ids = [
            make_point_id(
                namespace="team-space",
                corpus_id="corpus-1",
                document_id="doc-1",
                chunk_index=index,
            )
            for index in (1, 2)
        ]
        assert result.chunk_count == 1
        assert store.operations == ["delete_point_ids", "upsert"]
        assert store.delete_calls == []
        assert store.delete_point_ids_calls == [expected_stale_ids]

    asyncio.run(_run())


def test_make_point_id_scopes_namespace_and_corpus() -> None:
    def _id(namespace: str, corpus_id: str) -> str:
        return make_point_id(
            namespace=namespace,
            corpus_id=corpus_id,
            document_id="doc-1",
            chunk_index=0,
        )

    first = _id("team-a", "corpus-1")
    second = _id("team-b", "corpus-1")
    third = _id("team-a", "corpus-2")

    assert first != second
    assert first != third
    assert second != third


def test_build_index_request_carries_prepare_time_chunk_spans() -> None:
    """The real ingest path must hand prepare-time chunks (verbatim spans and
    real strategy) to ``resolve_chunks`` rather than re-zeroing them as
    ``prechunked``."""
    from rag_core._engine.core_builders import build_index_request
    from rag_core.core_models import (
        PreparedChunk,
        PreparedDocument,
        ProcessingFingerprint,
    )
    from rag_core.search.indexer_texts import resolve_chunks

    markdown = "# Title\n\nFirst section body.\n\nSecond section body.\n"
    chunks = [
        PreparedChunk(
            chunk_index=0,
            text="First section body.",
            embedding_text="First section body.",
            word_count=3,
            start_char=9,
            end_char=28,
            token_count=5,
            chunking_strategy="markdown",
            metadata={"chunking_strategy": "markdown", "section_path": "Title"},
        ),
        PreparedChunk(
            chunk_index=1,
            text="Second section body.",
            embedding_text="Second section body.",
            word_count=3,
            start_char=30,
            end_char=50,
            token_count=5,
            chunking_strategy="markdown",
            metadata={"chunking_strategy": "markdown", "section_path": "Title"},
        ),
    ]
    prepared = PreparedDocument(
        filename="guide.md",
        mime_type="text/markdown",
        markdown=markdown,
        chunks=chunks,
    )

    request = build_index_request(
        prepared=prepared,
        document_id="doc-1",
        document_key="guide.md",
        content_sha256="sha",
        processing_version=ProcessingFingerprint(
            base_version="v1", source_type="file"
        ),
        existing=None,
        corpus_id="corpus-1",
        namespace="team-a",
        source_type="file",
        metadata=None,
        embedding_model="demo",
    )

    resolved = resolve_chunks(request)

    assert resolved == chunks
    assert [c.start_char for c in resolved] == [9, 30]
    assert [c.end_char for c in resolved] == [28, 50]
    assert [c.chunking_strategy for c in resolved] == ["markdown", "markdown"]
