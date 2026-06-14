from __future__ import annotations

import asyncio
from typing import cast

from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.turbopuffer_payloads import TURBOPUFFER_BM25_TEXT_FIELD
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from rag_core.search.request_models import SearchQuery
from rag_core.search.vector_models import SparseVector, VectorPoint
from tests.support.turbopuffer_fake import TurboPufferFakeNamespace


def test_turbopuffer_contextual_bm25_field_is_rank_only_not_result_payload() -> None:
    async def _run() -> None:
        namespace = TurboPufferFakeNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )
        await store.upsert(
            [
                VectorPoint(
                    id="contextual-point",
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=SparseVector(indices=[], values=[]),
                    sparse_text="leasecontext\n\nclean chunk",
                    payload={
                        "namespace": "team-space",
                        "corpus_id": "corpus-a",
                        "document_id": "doc-ctx",
                        "content_type": "document",
                        "source_type": "file",
                        "text": "clean chunk",
                    },
                )
            ]
        )

        rows = cast(list[dict[str, object]], namespace.write_calls[0]["upsert_rows"])
        assert rows[0]["text"] == "clean chunk"
        assert rows[0][TURBOPUFFER_BM25_TEXT_FIELD] == "leasecontext\n\nclean chunk"

        results = await store.search(
            SearchQuery(
                dense_vector=[0.0, 0.0, 0.0],
                sparse_vector=SparseVector(indices=[1], values=[1.0]),
                namespace="team-space",
                corpus_ids=["corpus-a"],
                limit=1,
                lexical_query="leasecontext",
                query_plan=query_plan_preset("sparse_only", limit=1),
            )
        )

        assert namespace.query_calls[-1]["rank_by"] == (
            TURBOPUFFER_BM25_TEXT_FIELD,
            "BM25",
            "leasecontext",
        )
        assert len(results) == 1
        assert results[0].text == "clean chunk"
        assert TURBOPUFFER_BM25_TEXT_FIELD not in results[0].metadata

    asyncio.run(_run())


def test_turbopuffer_rejects_reserved_bm25_text_payload_field() -> None:
    async def _run() -> None:
        namespace = TurboPufferFakeNamespace()
        store = TurboPufferVectorStore(
            namespace="docs",
            dense_dimensions=3,
            namespace_client=namespace,
        )
        try:
            await store.upsert(
                [
                    VectorPoint(
                        id="collide-point",
                        dense_vector=[1.0, 0.0, 0.0],
                        sparse_vector=SparseVector(indices=[], values=[]),
                        sparse_text="ctx\n\nclean chunk",
                        payload={
                            "namespace": "team-space",
                            "corpus_id": "corpus-a",
                            "document_id": "doc-collide",
                            "content_type": "document",
                            "source_type": "file",
                            "text": "clean chunk",
                            TURBOPUFFER_BM25_TEXT_FIELD: "user data that would be lost",
                        },
                    )
                ]
            )
        except ValueError as exc:
            assert TURBOPUFFER_BM25_TEXT_FIELD in str(exc)
            assert "reserved" in str(exc)
            return
        raise AssertionError("expected reserved-field collision to raise")

    asyncio.run(_run())
