from __future__ import annotations

import asyncio
from collections.abc import Sequence

from rag_core.events import EventBuffer
from rag_core.events.types import NeighborExpandSkipped
from rag_core.search.vector_models import SearchResult
from rag_core.search.context_pack import (
    CONTEXT_EXPANSION_AFTER_METADATA_KEY,
    CONTEXT_EXPANSION_BEFORE_METADATA_KEY,
)
from rag_core.search.pipeline import PipelineContext, PipelineQuery
from rag_core.search.pipeline.stages.neighbor_expand import NeighborExpandPostprocess
from rag_core.search.provider_protocols import VectorStore
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
)
from tests.support.turbopuffer_fake import TurboPufferFakeNamespace


def _query() -> PipelineQuery:
    return PipelineQuery(
        query="amberglint",
        namespace="space-1",
        corpus_ids=["corpus-1"],
        limit=5,
    )


def _ctx(store: VectorStore, event_sink: EventBuffer | None = None) -> PipelineContext:
    return PipelineContext(
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        event_sink=event_sink,
    )


def _chunk(index: int, text: str = ""):
    return make_search_result(
        id=f"chunk-{index}",
        text=text or f"chunk {index}",
        document_id="doc-1",
        corpus_id="corpus-1",
        namespace="space-1",
        chunk_index=index,
    )


def test_neighbor_expand_handles_first_and_last_chunk_edges() -> None:
    async def _run() -> None:
        chunks = [_chunk(0, "zero"), _chunk(1, "one"), _chunk(2, "two")]
        store = RecordingVectorStore(search_results=chunks)
        stage = NeighborExpandPostprocess(window=1)

        first = await stage.postprocess([chunks[0]], _query(), _ctx(store))
        last = await stage.postprocess([chunks[2]], _query(), _ctx(store))

        assert first[0].metadata[CONTEXT_EXPANSION_BEFORE_METADATA_KEY] == []
        assert first[0].metadata[CONTEXT_EXPANSION_AFTER_METADATA_KEY] == ["one"]
        assert last[0].metadata[CONTEXT_EXPANSION_BEFORE_METADATA_KEY] == ["one"]
        assert last[0].metadata[CONTEXT_EXPANSION_AFTER_METADATA_KEY] == []
        assert store.get_chunks_by_index_calls[0] == (
            "space-1",
            "corpus-1",
            "doc-1",
            (0, 1),
        )
        assert store.get_chunks_by_index_calls[1] == (
            "space-1",
            "corpus-1",
            "doc-1",
            (1, 2, 3),
        )

    asyncio.run(_run())


def test_neighbor_expand_caps_hits_and_dedupes_overlapping_neighbors() -> None:
    async def _run() -> None:
        chunks = [_chunk(0, "zero"), _chunk(1, "one"), _chunk(2, "two")]
        store = RecordingVectorStore(search_results=chunks)
        stage = NeighborExpandPostprocess(window=1, max_hits=2)

        expanded = await stage.postprocess(chunks, _query(), _ctx(store))

        assert expanded[0].metadata[CONTEXT_EXPANSION_AFTER_METADATA_KEY] == ["one"]
        assert expanded[1].metadata[CONTEXT_EXPANSION_BEFORE_METADATA_KEY] == []
        assert expanded[1].metadata[CONTEXT_EXPANSION_AFTER_METADATA_KEY] == ["two"]
        assert CONTEXT_EXPANSION_BEFORE_METADATA_KEY not in expanded[2].metadata
        assert store.get_chunks_by_index_calls == [
            ("space-1", "corpus-1", "doc-1", (0, 1, 2))
        ]

    asyncio.run(_run())


def test_neighbor_expand_drops_store_rows_outside_requested_scope() -> None:
    async def _run() -> None:
        class MaliciousStore(RecordingVectorStore):
            async def get_chunks_by_index(
                self,
                *,
                namespace: str,
                corpus_id: str,
                document_id: str,
                chunk_indices: Sequence[int],
            ) -> list[SearchResult]:
                self.get_chunks_by_index_calls.append(
                    (namespace, corpus_id, document_id, tuple(chunk_indices))
                )
                return [
                    make_search_result(
                        id="wrong-doc",
                        text="wrong before",
                        namespace=namespace,
                        corpus_id=corpus_id,
                        document_id="doc-2",
                        chunk_index=0,
                    ),
                    make_search_result(
                        id="wrong-index",
                        text="wrong index",
                        namespace=namespace,
                        corpus_id=corpus_id,
                        document_id=document_id,
                        chunk_index=99,
                    ),
                    make_search_result(
                        id="right-after",
                        text="right after",
                        namespace=namespace,
                        corpus_id=corpus_id,
                        document_id=document_id,
                        chunk_index=2,
                    ),
                ]

        hit = _chunk(1, "center")
        expanded = await NeighborExpandPostprocess(window=1).postprocess(
            [hit],
            _query(),
            _ctx(MaliciousStore()),
        )

        assert expanded[0].metadata[CONTEXT_EXPANSION_BEFORE_METADATA_KEY] == []
        assert expanded[0].metadata[CONTEXT_EXPANSION_AFTER_METADATA_KEY] == [
            "right after"
        ]

    asyncio.run(_run())


def test_neighbor_expand_refuses_turbopuffer_with_one_event() -> None:
    async def _run() -> None:
        buffer = EventBuffer()
        namespace = TurboPufferFakeNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=4,
            namespace_client=namespace,
        )
        stage = NeighborExpandPostprocess()
        hit = _chunk(1, "one")

        expanded = await stage.postprocess([hit], _query(), _ctx(store, buffer))

        assert expanded == [hit]
        skipped = [event for event in buffer.events if isinstance(event, NeighborExpandSkipped)]
        assert len(skipped) == 1
        assert skipped[0].reason == "unsupported_store"
        assert skipped[0].input_count == 1
        assert namespace.query_calls == []

    asyncio.run(_run())
