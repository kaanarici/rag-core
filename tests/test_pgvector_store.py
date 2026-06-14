from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from rag_core.search.filters import And, Geo, In, Not, Or, Range, Term
from rag_core.search.pipeline import HybridRetrieve, PipelineContext, PipelineQuery
from rag_core.search.planning import query_plan_preset
from rag_core.search.providers.pgvector_store import (
    PgVectorExtensionError,
    PgVectorVectorStore,
)
from rag_core.search.query_plan import DenseChannel, Prefetch, QueryPlan, UnsupportedQueryStage
from rag_core.search.request_models import DeleteFilter, SearchQuery
from rag_core.search.vector_models import SparseVector, VectorPoint
from tests.support import FakeEmbeddingProvider, FakeSparseEmbedder
from tests.support.pgvector_fake import PgVectorFakeConnection, PgVectorFakePool


def test_pgvector_store_declares_dense_sql_capabilities() -> None:
    store = _store()

    capabilities = store.capabilities

    assert capabilities.dense_vector_dimensions == 3
    assert capabilities.query_plan.dense is True
    assert capabilities.query_plan.sparse is False
    assert capabilities.query_plan.hybrid is False
    assert capabilities.metadata_filter.geo is True
    assert capabilities.chunk_index_lookup is True


def test_pgvector_ensure_collection_creates_extension_table_and_index() -> None:
    connection = PgVectorFakeConnection(extension_available=False)
    pool = PgVectorFakePool(connection)
    store = _store(pool=pool)

    asyncio.run(store.ensure_collection())

    executed = [call.sql for call in connection.execute_calls]
    assert executed[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert 'CREATE SCHEMA IF NOT EXISTS "public"' in executed
    assert any('CREATE TABLE IF NOT EXISTS "public"."docs"' in sql for sql in executed)
    assert any("USING hnsw (dense vector_cosine_ops)" in sql for sql in executed)


def test_pgvector_ensure_collection_rejects_existing_dense_dimension_mismatch() -> None:
    connection = PgVectorFakeConnection(dense_dimensions=2)
    store = _store(pool=PgVectorFakePool(connection))

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(store.ensure_collection())

    message = str(exc_info.value)
    assert "dense vector dimension mismatch" in message
    assert "expected 3" in message
    assert "found 2" in message
    assert "table_name" in message
    assert store._ready is False


def test_pgvector_extension_missing_raises_teachable_error() -> None:
    connection = PgVectorFakeConnection(extension_available=False)
    connection.fail_create_extension = True
    store = _store(pool=PgVectorFakePool(connection))

    with pytest.raises(PgVectorExtensionError, match="Install the pgvector extension"):
        asyncio.run(store.ensure_collection())


def test_pgvector_upsert_uses_parameterized_jsonb_and_hot_columns() -> None:
    connection = PgVectorFakeConnection()
    store = _store(pool=PgVectorFakePool(connection))

    asyncio.run(store.upsert([_point("point-1", section="intro")]))

    command, rows = connection.executemany_calls[0]
    assert "VALUES ($1, $2::vector, $3::jsonb" in command
    assert "intro" not in command
    assert rows[0][0] == "point-1"
    assert rows[0][1] == [1.0, 0.0, 0.0]
    assert rows[0][3:12] == (
        "team-space",
        "corpus-a",
        "doc-1",
        "/docs/doc-1.md",
        "sha-1",
        "v1",
        "document",
        "file",
        0,
    )


def test_pgvector_search_parameterizes_nested_metadata_filter_sql() -> None:
    connection = PgVectorFakeConnection()
    store = _store(pool=PgVectorFakePool(connection))

    asyncio.run(
        store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=_empty_sparse(),
                namespace="team-space",
                corpus_ids=["corpus-a"],
                limit=5,
                metadata_filter=And(
                    (
                        Term(field="section", value="intro"),
                        Or(
                            (
                                In(field="tag", values=("finance", "ops")),
                                Not(Range(field="score", gte=0.2, lt=0.9)),
                            )
                        ),
                        Geo(field="location", lat=40.7, lon=-74.0, radius_m=500.0),
                    )
                ),
            )
        )
    )

    call = connection.fetch_calls[0]
    assert "dense <=> $1::vector" in call.sql
    assert "ORDER BY dense <=> $1::vector" in call.sql
    assert "LIMIT $16::integer" in call.sql
    for unsafe_literal in ("intro", "finance", "ops", "section", "tag", "location"):
        assert unsafe_literal not in call.sql
    assert call.params[0] == [1.0, 0.0, 0.0]
    assert call.params[1] == "team-space"
    assert call.params[2] == ["corpus-a"]
    assert call.params[-1] == 5


def test_pgvector_rejects_unsafe_metadata_field_before_sql() -> None:
    connection = PgVectorFakeConnection()
    store = _store(pool=PgVectorFakePool(connection))

    with pytest.raises(ValueError, match="metadata filter field"):
        asyncio.run(
            store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="team-space",
                    corpus_ids=["corpus-a"],
                    metadata_filter=Term(field="section) OR TRUE --", value="intro"),
                )
            )
        )

    assert connection.fetch_calls == []


def test_pgvector_numeric_range_treats_dirty_payload_value_as_non_match() -> None:
    connection = PgVectorFakeConnection()
    store = _store(pool=PgVectorFakePool(connection))

    async def _run() -> list[str]:
        await store.upsert([_point("dirty-score", score="high")])
        results = await store.search(
            SearchQuery(
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=_empty_sparse(),
                namespace="team-space",
                corpus_ids=["corpus-a"],
                metadata_filter=Range(field="score", gte=0.2),
            )
        )
        return [result.id for result in results]

    assert asyncio.run(_run()) == []
    sql = connection.fetch_calls[-1].sql
    assert "jsonb_typeof" in sql
    assert "COALESCE" in sql


def test_pgvector_delete_paths_use_parameterized_sql() -> None:
    connection = PgVectorFakeConnection()
    store = _store(pool=PgVectorFakePool(connection))

    asyncio.run(
        store.delete(
            DeleteFilter(
                namespace="team-space",
                corpus_id="corpus-a",
                document_id="doc-1",
            )
        )
    )
    asyncio.run(store.delete_point_ids(["point-1", "point-2"]))

    delete_sql = connection.execute_calls[-2].sql
    assert 'DELETE FROM "public"."docs" WHERE' in delete_sql
    assert "team-space" not in delete_sql
    assert connection.execute_calls[-2].params == ("team-space", "corpus-a", "doc-1")
    assert connection.execute_calls[-1].sql.endswith("id = ANY($1::text[])")
    assert connection.execute_calls[-1].params == (["point-1", "point-2"],)


def test_pgvector_document_and_chunk_lookup_round_trip_against_fake() -> None:
    connection = PgVectorFakeConnection()
    store = _store(pool=PgVectorFakePool(connection))

    async def _run() -> None:
        await store.upsert(
            [
                _point("point-1", chunk_index=0, text="alpha opening"),
                _point("point-2", chunk_index=1, text="alpha details"),
            ]
        )
        record = await store.get_document_record(
            namespace="team-space",
            corpus_id="corpus-a",
            document_key="/docs/doc-1.md",
        )
        chunks = await store.get_chunks_by_index(
            namespace="team-space",
            corpus_id="corpus-a",
            document_id="doc-1",
            chunk_indices=[1, 0, 1],
        )
        assert record is not None
        assert record.document_id == "doc-1"
        assert record.chunk_count == 2
        assert [chunk.text for chunk in chunks] == ["alpha opening", "alpha details"]

    asyncio.run(_run())

    assert '"document_key"' in connection.fetchrow_calls[0].sql
    assert '"chunk_index" = ANY' in connection.fetch_calls[-1].sql


def test_pgvector_hybrid_query_plan_fails_before_embedding_or_sql() -> None:
    store = _store()
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder()

    async def _run() -> None:
        with pytest.raises(UnsupportedQueryStage, match="hybrid RRF query plans"):
            await HybridRetrieve().retrieve(
                PipelineQuery(
                    query="billing",
                    namespace="team-space",
                    corpus_ids=["corpus-a"],
                    query_plan=query_plan_preset("hybrid_rrf", limit=5),
                ),
                PipelineContext(
                    embedding_provider=embedding,
                    sparse_embedder=sparse,
                    vector_store=store,
                ),
            )

    asyncio.run(_run())
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == []


def test_pgvector_non_primary_dense_plan_fails_before_embedding_or_sql() -> None:
    store = _store()
    embedding = FakeEmbeddingProvider()
    sparse = FakeSparseEmbedder()

    async def _run() -> None:
        with pytest.raises(UnsupportedQueryStage, match="primary dense query vector"):
            await HybridRetrieve().retrieve(
                PipelineQuery(
                    query="billing",
                    namespace="team-space",
                    corpus_ids=["corpus-a"],
                    query_plan=QueryPlan(
                        prefetches=(
                            Prefetch(
                                channel=DenseChannel(vector_field="alternate"),
                                limit=5,
                            ),
                        ),
                        final_limit=5,
                    ),
                ),
                PipelineContext(
                    embedding_provider=embedding,
                    sparse_embedder=sparse,
                    vector_store=store,
                ),
            )

    asyncio.run(_run())
    assert embedding.embed_query_calls == []
    assert sparse.embed_query_multi_calls == []


@pytest.mark.live
def test_pgvector_live_round_trip_when_dsn_is_configured() -> None:
    dsn = os.environ.get("PGVECTOR_TEST_DSN", "").strip()
    if not dsn:
        pytest.skip("PGVECTOR_TEST_DSN is not set")

    async def _run() -> None:
        table_name = f"rag_core_live_{uuid.uuid4().hex}"
        store = PgVectorVectorStore(
            dsn=dsn,
            table_name=table_name,
            dense_dimensions=3,
        )
        try:
            await store.upsert(
                [
                    _point("live-1", chunk_index=0, section="intro", text="alpha opening"),
                    _point("live-2", chunk_index=1, section="details", text="alpha details"),
                ]
            )
            hits = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="team-space",
                    corpus_ids=["corpus-a"],
                    metadata_filter=Term(field="section", value="details"),
                )
            )
            record = await store.get_document_record(
                namespace="team-space",
                corpus_id="corpus-a",
                document_id="doc-1",
            )
            chunks = await store.get_chunks_by_index(
                namespace="team-space",
                corpus_id="corpus-a",
                document_id="doc-1",
                chunk_indices=[0, 1],
            )
            await store.delete_point_ids(["live-1"])
            remaining = await store.search(
                SearchQuery(
                    dense_vector=[1.0, 0.0, 0.0],
                    sparse_vector=_empty_sparse(),
                    namespace="team-space",
                    corpus_ids=["corpus-a"],
                )
            )

            assert [hit.text for hit in hits] == ["alpha details"]
            assert record is not None
            assert record.chunk_count == 2
            assert [chunk.chunk_index for chunk in chunks] == [0, 1]
            assert [hit.id for hit in remaining] == ["live-2"]
        finally:
            await store.close()

    asyncio.run(_run())


def _store(pool: PgVectorFakePool | None = None) -> PgVectorVectorStore:
    return PgVectorVectorStore(
        table_name="docs",
        dense_dimensions=3,
        pool=pool or PgVectorFakePool(),
    )


def _point(
    point_id: str,
    *,
    chunk_index: int = 0,
    section: str = "section",
    text: str = "alpha",
    score: object = 0.5,
) -> VectorPoint:
    return VectorPoint(
        id=point_id,
        dense_vector=[1.0, 0.0, 0.0],
        sparse_vector=_empty_sparse(),
        payload={
            "namespace": "team-space",
            "corpus_id": "corpus-a",
            "document_id": "doc-1",
            "document_key": "/docs/doc-1.md",
            "content_sha256": "sha-1",
            "processing_version": "v1",
            "content_type": "document",
            "source_type": "file",
            "text": text,
            "chunk_index": chunk_index,
            "section": section,
            "score": score,
            "location": {"lat": 40.7, "lon": -74.0},
        },
    )


def _empty_sparse() -> SparseVector:
    return SparseVector(indices=[], values=[])
