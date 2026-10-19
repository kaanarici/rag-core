"""Persisted document guarantees, using actual Qdrant and deterministic vectors."""

from __future__ import annotations
import asyncio
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
from qdrant_client import models as rest

from rag_core import Config
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider
from rag_core.search.document_versions import KIND, CHUNK
from rag_core.search.policy import CollectionPolicy, CollectionPolicyViolation
from rag_core.search.providers.qdrant_store import QdrantVectorStore

SCOPE = dict(tenant_id="acme", collection="docs")


def engine(path: Path | None = None, **kwargs: Any) -> Engine:
    return Engine(
        Config.local(path), embedding_provider=DemoEmbeddingProvider(), **kwargs
    )


async def add(core: Engine, text: bytes = b"Invoices accept card.", **scope: Any):
    return await core.add_bytes(
        file_bytes=text,
        filename="billing.txt",
        mime_type="text/plain",
        document_key="billing",
        **(scope or SCOPE),
    )


async def texts(core: Engine, **scope: Any) -> list[str]:
    return [
        hit.text
        for hit in await core.search(query="invoices", rerank=False, **(scope or SCOPE))
    ]


def test_failed_embedding_and_failed_partial_write_keep_previous_version() -> None:
    async def run() -> None:
        async with engine() as core:
            await add(core)
            original_embed = core._embedding.embed_texts

            async def fail_embed(texts: list[str]) -> list[list[float]]:
                raise RuntimeError("embedding failed")

            core._embedding.embed_texts = fail_embed  # type: ignore[method-assign]
            with pytest.raises(RuntimeError, match="embedding failed"):
                await add(core, b"Invoices require bank transfer.")
            assert await texts(core) == ["Invoices accept card."]
            core._embedding.embed_texts = original_embed  # type: ignore[method-assign]
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            original_upsert = store.upsert

            async def fail_write(points):
                await original_upsert(points[:1])
                raise RuntimeError("partial write")

            store.upsert = fail_write  # type: ignore[method-assign]
            with pytest.raises(RuntimeError, match="partial write"):
                await add(core, b"Invoices require bank transfer. " * 150)
            assert await texts(core) == ["Invoices accept card."]
            store.upsert = original_upsert  # type: ignore[method-assign]
            await add(core, b"Invoices require bank transfer.")
            assert await texts(core) == ["Invoices require bank transfer."]

    asyncio.run(run())


def test_partial_committed_version_is_reindexed_instead_of_skipped() -> None:
    async def run() -> None:
        async with engine() as core:
            source = b"Invoices accept card and bank transfer. " * 200
            first = await add(core, source)
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            rows, _ = await store._client.scroll(
                collection_name=store._config.collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key=KIND, match=rest.MatchValue(value=CHUNK)
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
            )
            assert len(rows) > 2
            await store.delete_point_ids([str(rows[-1].id)])
            record = await store.get_document_record(
                **SCOPE, document_id=first.document_id
            )
            assert record is not None and not record.is_complete
            repaired = await add(core, source)
            assert (
                repaired.ingest_state == "reindexed"
                and repaired.chunk_count == first.chunk_count
            )
            final = await store.get_document_record(
                **SCOPE, document_id=first.document_id
            )
            assert final is not None and final.is_complete

    asyncio.run(run())


def test_colliding_delimited_scopes_and_explicit_ids_remain_independent() -> None:
    async def run() -> None:
        async with engine() as core:
            a = dict(tenant_id="a", collection="b::c")
            b = dict(tenant_id="a::b", collection="c")
            first = await add(core, b"Invoices for first tenant.", **a)
            second = await add(core, b"Invoices for second tenant.", **b)
            assert first.document_id != second.document_id
            assert await texts(core, **a) == ["Invoices for first tenant."]
            assert await texts(core, **b) == ["Invoices for second tenant."]
            for scope, text in [(a, b"Explicit first."), (b, b"Explicit second.")]:
                await core.add_bytes(
                    file_bytes=text,
                    filename="x.txt",
                    mime_type="text/plain",
                    document_id="same::id",
                    document_key="explicit",
                    tenant_id=scope["tenant_id"],
                    collection=scope["collection"],
                )
            await core.delete_document(document_id="same::id", **a)
            assert "Explicit second." in await texts(core, **b)

    asyncio.run(run())


@pytest.mark.parametrize(
    "method", ["add_bytes", "add_file", "add", "add_archive", "add_url"]
)
def test_scope_refusal_precedes_io_and_provider_work(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support import RecordingVectorStore, FakeEmbeddingProvider

    store = RecordingVectorStore()
    config = replace(
        Config.local(),
        collection_policy=CollectionPolicy(
            bound_tenant_id="allowed", allowed_collections=frozenset({"docs"})
        ),
    )
    core = Engine(
        config, embedding_provider=FakeEmbeddingProvider(), vector_store=store
    )

    async def run() -> None:
        with pytest.raises(CollectionPolicyViolation):
            if method == "add_bytes":
                await core.add_bytes(
                    file_bytes=b"private",
                    filename="a.txt",
                    mime_type="text/plain",
                    tenant_id="other",
                    collection="private",
                )
            else:
                await getattr(core, method)(
                    "https://should-never-be-fetched.invalid/private"
                    if method == "add_url"
                    else "/missing/should-not-be-read",
                    tenant_id="other",
                    collection="private",
                )
        assert store.operations == []
        await core.close()

    asyncio.run(run())


def test_capitalized_folder_scope_has_safe_distinct_manifest_paths(
    tmp_path: Path,
) -> None:
    from rag_core.manifest.persistence import manifest_path
    from rag_core.local import local_collection

    folder = tmp_path / "ClientDocs"
    folder.mkdir()
    assert local_collection(folder) == "ClientDocs"
    upper = manifest_path(tmp_path / "manifest", "acme", "ClientDocs")
    lower = manifest_path(tmp_path / "manifest", "acme", "clientdocs")
    escaped = manifest_path(tmp_path / "manifest", "../escape", "A/B")
    assert upper != lower and upper.name.lower() != lower.name.lower()
    assert escaped.is_relative_to(tmp_path / "manifest")


@pytest.mark.parametrize("phase", ["mid_stage", "pre_publish", "post_publish"])
def test_process_death_preserves_a_complete_version_and_retry_recovers(
    tmp_path: Path, phase: str
) -> None:
    path = tmp_path / "store"

    async def initial() -> None:
        async with engine(path) as core:
            await add(core)

    asyncio.run(initial())
    script = """
import asyncio, os
from rag_core import Config
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider
async def main():
 async with Engine(Config.local(os.environ['INTEGRITY_STORE']), embedding_provider=DemoEmbeddingProvider()) as core:
  phase = os.environ['INTEGRITY_PHASE']
  original_publish = core._store._publish_head
  original_upsert = core._store.upsert
  async def die(*args, **kwargs):
   if phase == 'post_publish': await original_publish(*args, **kwargs)
   os._exit(91)
  async def mid_stage(points):
   await original_upsert(points[:1])
   os._exit(91)
  if phase == 'mid_stage': core._store.upsert = mid_stage
  else: core._store._publish_head = die
  await core.add_bytes(file_bytes=b"New invoice terms. " * 300, filename="billing.txt", mime_type="text/plain", document_key="billing", tenant_id="acme", collection="docs")
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, INTEGRITY_STORE=str(path), INTEGRITY_PHASE=phase),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 91, result.stderr

    async def reopen() -> None:
        async with engine(path) as core:
            before = await texts(core)
            if phase == "post_publish":
                assert before and all("New invoice terms." in hit for hit in before)
            else:
                assert before == ["Invoices accept card."]
            replacement = await add(core, b"New invoice terms. " * 300)
            assert replacement.ingest_state == (
                "unchanged" if phase == "post_publish" else "replaced"
            )
            record = await core._store.get_document_record(
                **SCOPE, document_id=replacement.document_id
            )
            assert record is not None and record.is_complete and record.chunk_count > 1

    asyncio.run(reopen())


def test_incomplete_staging_is_never_published_even_if_adapter_returns_success() -> (
    None
):
    async def run() -> None:
        async with engine() as core:
            await add(core)
            store = core._store
            original = store.upsert

            async def truncate(points):
                await original(points[:1])

            store.upsert = truncate  # type: ignore[method-assign]
            with pytest.raises(RuntimeError, match="incomplete staged"):
                await add(core, b"New invoice terms. " * 300)
            assert await texts(core) == ["Invoices accept card."]

    asyncio.run(run())


def test_concurrent_engines_publish_one_complete_winner() -> None:
    async def run() -> None:
        from rag_core.search.document_versions import ConcurrentDocumentUpdate

        core = engine()
        store = core._store
        assert isinstance(store, QdrantVectorStore)
        await add(core)
        other = Engine(
            Config.local(),
            embedding_provider=DemoEmbeddingProvider(),
            vector_store=store,
        )
        arrived = 0
        ready = asyncio.Event()
        original = store.upsert

        async def staged(points):
            nonlocal arrived
            await original(points)
            arrived += 1
            if arrived == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 3)

        store.upsert = staged  # type: ignore[method-assign]
        results = await asyncio.gather(
            add(core, b"Version alpha invoices. " * 200),
            add(other, b"Version beta invoices. " * 250),
            return_exceptions=True,
        )
        assert (
            sum(isinstance(result, ConcurrentDocumentUpdate) for result in results) == 1
        )
        hits = await texts(core)
        assert hits and (
            all("alpha" in hit for hit in hits) or all("beta" in hit for hit in hits)
        )
        winner = next(
            result for result in results if not isinstance(result, BaseException)
        )
        record = await store.get_document_record(
            **SCOPE, document_id=winner.document_id
        )
        assert (
            record is not None
            and record.is_complete
            and record.chunk_count == winner.chunk_count
        )
        await core.close()

    asyncio.run(run())


def test_lost_publish_ack_never_rolls_back_a_committed_version() -> None:
    async def run() -> None:
        async with engine() as core:
            await add(core)
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            original = store._publish_head

            async def lost(*args, **kwargs):
                await original(*args, **kwargs)
                raise TimeoutError("publish acknowledgement lost")

            store._publish_head = lost  # type: ignore[method-assign]
            with pytest.raises(TimeoutError):
                await add(core, b"New complete invoice terms.")
            assert await texts(core) == ["New complete invoice terms."]
            store._publish_head = original  # type: ignore[method-assign]
            retry = await add(core, b"New complete invoice terms.")
            assert retry.ingest_state == "unchanged"

    asyncio.run(run())


def test_replayed_publication_cannot_resurrect_retired_chunks() -> None:
    async def run() -> None:
        async with engine() as core:
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            original = store._publish_head
            calls = []

            async def record(*args, **kwargs):
                calls.append((args, kwargs))
                return await original(*args, **kwargs)

            store._publish_head = record  # type: ignore[method-assign]
            await add(core)
            await add(core, b"New invoice terms.")
            await add(core, b"Latest invoice terms.")
            for args, kwargs in calls[:2]:
                await original(*args, **kwargs)
            assert await texts(core) == ["Latest invoice terms."]

    asyncio.run(run())


def test_empty_index_request_has_a_durable_complete_record() -> None:
    from rag_core.search.indexer_models import IndexRequest

    async def run() -> None:
        async with engine() as core:
            await core._indexer.index_document(
                IndexRequest(
                    document_id="empty",
                    tenant_id="acme",
                    collection="docs",
                    text="",
                    filename="empty.txt",
                    mime_type="text/plain",
                    source_type="file",
                    prepared_chunks=[],
                )
            )
            record = await core._store.get_document_record(**SCOPE, document_id="empty")
            assert record is not None and record.chunk_count == 0 and record.is_complete

    asyncio.run(run())


def test_qdrant_deleted_head_blocks_replayed_creation_and_allows_new_ingest() -> None:
    async def run() -> None:
        async with engine() as core:
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            original = store._publish_head
            writes = []

            async def capture(*args, **kwargs):
                writes.append((args, kwargs))
                return await original(*args, **kwargs)

            store._publish_head = capture  # type: ignore[method-assign]
            document = await add(core)
            await core.delete_document(**SCOPE, document_id=document.document_id)
            args, kwargs = writes[0]
            await original(*args, **kwargs)
            assert await texts(core) == []
            assert (
                await store.get_document_record(
                    **SCOPE, document_id=document.document_id
                )
                is None
            )
            recreated = await add(core, b"Invoices recreated after delete.")
            assert recreated.ingest_state == "created"
            assert await texts(core) == ["Invoices recreated after delete."]

    asyncio.run(run())


def test_failed_delete_cleanup_is_retried_before_acknowledgement() -> None:
    async def run() -> None:
        async with engine() as core:
            document = await add(core)
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            original = store._delete_revision

            async def fail(*args, **kwargs):
                raise OSError("cleanup failed")

            store._delete_revision = fail  # type: ignore[method-assign]
            with pytest.raises(OSError, match="cleanup failed"):
                await core.delete_document(**SCOPE, document_id=document.document_id)
            assert await texts(core) == []
            store._delete_revision = original  # type: ignore[method-assign]
            await core.delete_document(**SCOPE, document_id=document.document_id)
            rows, _ = await store._client.scroll(
                collection_name=store._config.collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key=KIND, match=rest.MatchValue(value=CHUNK)
                        )
                    ]
                ),
                limit=100,
            )
            assert rows == []

    asyncio.run(run())


def test_successful_shrinking_replacement_removes_obsolete_physical_chunks() -> None:
    async def run() -> None:
        async with engine() as core:
            first = await add(core, b"Old invoices accept card. " * 400)
            assert first.chunk_count > 2
            latest = await add(core, b"New invoices accept ACH.")
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            rows, _ = await store._client.scroll(
                collection_name=store._config.collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key=KIND, match=rest.MatchValue(value=CHUNK)
                        )
                    ]
                ),
                limit=100,
            )
            assert len(rows) == latest.chunk_count == 1
            assert await texts(core) == ["New invoices accept ACH."]

    asyncio.run(run())
