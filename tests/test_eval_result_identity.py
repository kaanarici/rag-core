from __future__ import annotations

import asyncio

from rag_core import RAGCore
from rag_core.evals import EvalCase, run_eval
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def test_eval_matches_chunk_result_ids_before_document_fallback() -> None:
    async def scenario() -> tuple[tuple[str, ...], float, float]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="doc-a#chunk-2",
                    text="specific billing policy",
                    score=0.9,
                    document_id="doc-a",
                    corpus_id="docs",
                ),
                make_search_result(
                    id="doc-a#chunk-1",
                    text="general billing overview",
                    score=0.8,
                    document_id="doc-a",
                    corpus_id="docs",
                ),
            ]
        )
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_eval_chunk_identity",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            [result] = await run_eval(
                core,
                [
                    EvalCase(
                        query="billing policy",
                        namespace="rt",
                        corpus_ids=("docs",),
                        expected_chunk_ids=("doc-a#chunk-2",),
                    )
                ],
            )
            return result.retrieved_ids, result.recall_at_5, result.mrr
        finally:
            await core.close()

    retrieved_ids, recall, reciprocal_rank = asyncio.run(scenario())

    assert retrieved_ids[:2] == ("doc-a#chunk-2", "doc-a")
    assert recall == 1.0
    assert reciprocal_rank == 1.0


def test_eval_still_matches_document_ids_when_cases_are_document_level() -> None:
    async def scenario() -> tuple[tuple[str, ...], float]:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="doc-a#chunk-2",
                    text="specific billing policy",
                    score=0.9,
                    document_id="doc-a",
                    corpus_id="docs",
                )
            ]
        )
        core = RAGCore(
            make_test_config(
                qdrant_collection="rag_core_eval_document_identity",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            [result] = await run_eval(
                core,
                [
                    EvalCase(
                        query="billing policy",
                        namespace="rt",
                        corpus_ids=("docs",),
                        expected_chunk_ids=("doc-a",),
                    )
                ],
            )
            return result.retrieved_ids, result.recall_at_5
        finally:
            await core.close()

    retrieved_ids, recall = asyncio.run(scenario())

    assert retrieved_ids == ("doc-a",)
    assert recall == 1.0
