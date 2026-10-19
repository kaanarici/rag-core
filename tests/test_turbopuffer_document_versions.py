"""Version publication through the real TurboPuffer SDK and an HTTP fixture.

This checks wire contracts and recovery plumbing, not the hosted service.
"""

from __future__ import annotations
import asyncio
from dataclasses import replace
import json
from typing import Any
import pytest
from rag_core.core import Engine
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore
from tests.support import FakeEmbeddingProvider, make_test_config


def matches(row: dict[str, Any], condition: Any) -> bool:
    if condition is None:
        return True
    if condition[0] == "And":
        return all(matches(row, item) for item in condition[1])
    if condition[0] == "Not":
        return not matches(row, condition[1])
    key, operator, value = condition
    if operator == "Eq":
        return bool(row.get(key) == value)
    if operator == "In":
        return row.get(key) in value
    raise AssertionError(f"unexpected fixture filter {operator}")


@pytest.mark.integration
def test_real_sdk_document_publish_failure_delete_replay_and_recreate() -> None:
    sdk = pytest.importorskip("turbopuffer")
    import httpx

    async def run() -> None:
        rows: dict[str, dict[str, Any]] = {}
        calls: list[dict[str, Any]] = []
        fail_after: int | None = None
        staged_writes = 0

        def query(body):
            selected = [
                dict(row) for row in rows.values() if matches(row, body.get("filters"))
            ]
            if "aggregate_by" in body:
                return {
                    "aggregations": {key: len(selected) for key in body["aggregate_by"]}
                }
            rank = body.get("rank_by", ["id", "asc"])
            if rank[0] == "vector":
                selected = [row for row in selected if "vector" in row]
            elif rank[0] == "bm25_text":
                selected = [row for row in selected if row.get("bm25_text")]
            selected.sort(key=lambda row: row["id"])
            selected = selected[: body.get("limit", 10)]
            attrs = body.get("include_attributes", True)
            if attrs is not True:
                selected = [
                    {
                        key: value
                        for key, value in row.items()
                        if key == "id" or key in attrs
                    }
                    for row in selected
                ]
            return {"rows": [{**row, "$dist": 0.1} for row in selected]}

        def handler(request):
            nonlocal staged_writes, fail_after
            body = json.loads(request.content)
            calls.append(body)
            if request.url.path.endswith("/query"):
                return httpx.Response(
                    200,
                    json={"results": [query(q) for q in body["queries"]]}
                    if "queries" in body
                    else query(body),
                )
            for row in body.get("upsert_rows", []):
                if row.get("_rag_kind") == "chunk_v2":
                    staged_writes += 1
                    if fail_after is not None and staged_writes >= fail_after:
                        fail_after = None
                        return httpx.Response(
                            503, json={"error": "injected staging failure"}
                        )
                old = rows.get(row["id"])
                if old is None or matches(old, body.get("upsert_condition")):
                    rows[row["id"]] = dict(row)
            for row in body.get("patch_rows", []):
                old = rows.get(row["id"])
                if old is not None and matches(old, body.get("patch_condition")):
                    old.update(row)
            if "delete_by_filter" in body:
                for id in [
                    id
                    for id, row in rows.items()
                    if matches(row, body["delete_by_filter"])
                ]:
                    del rows[id]
            return httpx.Response(
                200, json={"rows_affected": 1, "rows_remaining": False}
            )

        client = sdk.AsyncTurbopuffer(
            api_key="fixture-key",
            region="gcp-us-central1",
            base_url="https://{region}.fixture.test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        store = TurboPufferVectorStore(
            namespace="docs", dense_dimensions=4, client=client, write_batch_size=1
        )
        core = Engine(
            replace(make_test_config(), hybrid=True),
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=store,
        )

        async def ingest(text):
            return await core.add_bytes(
                file_bytes=text,
                filename="bill.txt",
                mime_type="text/plain",
                tenant_id="acme",
                collection="docs",
                document_key="bill",
            )

        async def search():
            return await core.search(
                query="fox invoice", tenant_id="acme", collection="docs", rerank=False
            )

        try:
            first = await ingest(b"fox invoice original. " * 200)
            assert {hit.chunker_strategy for hit in await search()} == {"markdown"}
            original = await store.get_document_record(
                tenant_id="acme", collection="docs", document_id=first.document_id
            )
            assert (
                original is not None
                and original.is_complete
                and original.chunk_count > 1
            )
            from rag_core.search.filters import Term

            filtered = await core.search(
                query="fox",
                tenant_id="acme",
                collection="docs",
                metadata_filter=Term("chunk_index", 0),
                rerank=False,
            )
            assert len(filtered) == 1 and filtered[0].chunk_index == 0

            reopened = TurboPufferVectorStore(
                namespace="docs", dense_dimensions=4, client=client
            )
            reopened.bind_embedding_space(FakeEmbeddingProvider().model_name)
            read_heads = reopened._read_heads
            missed = False

            async def initially_missing(ids):
                nonlocal missed
                if not missed:
                    missed = True
                    return {}
                return await read_heads(ids)

            reopened._read_heads = initially_missing  # type: ignore[method-assign]
            await reopened.ensure_collection()

            class OtherModel(FakeEmbeddingProvider):
                @property
                def model_name(self) -> str:
                    return "different-model"

            other = Engine(
                replace(make_test_config(), hybrid=True),
                embedding_provider=OtherModel(),
                vector_store=TurboPufferVectorStore(
                    namespace="docs", dense_dimensions=4, client=client
                ),
            )
            before_rows = dict(rows)
            with pytest.raises(ValueError, match="different embedding model"):
                await other.add_bytes(
                    file_bytes=b"fox invoice original. " * 200,
                    filename="bill.txt",
                    mime_type="text/plain",
                    tenant_id="acme",
                    collection="docs",
                    document_key="bill",
                )
            with pytest.raises(ValueError, match="different embedding model"):
                await other.search(query="fox", tenant_id="acme", collection="docs")
            assert rows == before_rows
            initial_publish = next(
                call
                for call in calls
                if call.get("upsert_condition")
                and call["upsert_rows"][0].get("_rag_kind") == "head_v2"
            )
            staged_writes = 0
            fail_after = 2
            with pytest.raises(sdk.APIStatusError):
                await ingest(b"fox invoice revised. " * 200)
            assert all("original" in hit.text for hit in await search())
            retry = await ingest(b"fox invoice revised. " * 200)
            record = await store.get_document_record(
                tenant_id="acme", collection="docs", document_id=first.document_id
            )
            assert (
                record is not None
                and record.is_complete
                and record.chunk_count == retry.chunk_count
            )
            assert any(
                call.get("patch_condition")
                == ["_rag_revision", "Eq", original.revision]
                for call in calls
            )
            # Replay the original publication after replacement. Its insert-only
            # condition must leave the newer head alone.
            await store._namespace().write(**initial_publish)
            assert all("revised" in hit.text for hit in await search())
            await core.delete_document(
                tenant_id="acme", collection="docs", document_id=first.document_id
            )
            assert await search() == []
            assert (
                await store.get_document_record(
                    tenant_id="acme", collection="docs", document_id=first.document_id
                )
                is None
            )
            await store._namespace().write(**initial_publish)
            assert await search() == []
            await ingest(b"fox invoice recreated.")
            assert [hit.text for hit in await search()] == ["fox invoice recreated."]
        finally:
            await core.close()
            await client.close()

    asyncio.run(run())
