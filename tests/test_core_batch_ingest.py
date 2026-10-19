from __future__ import annotations
from rag_core.events.sink import EventSink

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

from rag_core.core import Engine
from rag_core.config import SKIP_UNCHANGED_MATERIALIZE
from rag_core.core_models import IngestedDocument, PreparedDocument
from rag_core.events import EventBuffer
from rag_core.ingest.local import LocalIngestResult
from rag_core.events import IngestBatchCompleted, IngestBatchStarted
from rag_core.ingest.sources.local import document_key as local_document_key

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)




def test_rag_core_ingest_files_reuses_local_batch_ingest_without_closing(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "billing.md").write_text("Billing is paid by card or ACH.", encoding="utf-8")
    (docs / "shipping.md").write_text("Shipping takes three days.", encoding="utf-8")

    async def scenario() -> tuple[LocalIngestResult, RecordingVectorStore, EventBuffer]:
        store = RecordingVectorStore()
        events = EventBuffer()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_batch_ingest",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=events,
        )
        result = await core.add(
            docs,
            tenant_id="acme",
            collection="help",
            max_concurrency=2,
        )
        assert store.close_calls == 0
        await core.close()
        return result, store, events

    result, store, events = asyncio.run(scenario())

    assert isinstance(result, LocalIngestResult)
    assert result.succeeded_count == 2
    assert result.written_count == 2
    assert result.failed_count == 0
    assert sorted(record.document_key for record in result.succeeded) == [
        local_document_key(docs, docs / "billing.md"),
        local_document_key(docs, docs / "shipping.md"),
    ]
    assert store.close_calls == 1
    assert isinstance(events.events[0], IngestBatchStarted)
    assert isinstance(events.events[-1], IngestBatchCompleted)




def test_rag_core_ingest_file_uses_file_source_type_fingerprint(tmp_path: Path) -> None:
    doc = tmp_path / "billing.md"
    doc.write_text("Billing is paid by card or ACH.", encoding="utf-8")

    async def scenario() -> RecordingVectorStore:
        store = RecordingVectorStore()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_single_file_source_guard",
                embedding_dimensions=4,
                source_type="archive",
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            await core.add_file(
                doc,
                tenant_id="acme",
                collection="help",
            )
        finally:
            await core.close()
        return store

    store = asyncio.run(scenario())

    points = [point for call in store.upsert_calls for point in call]
    assert {point.payload["source_type"] for point in points} == {"file"}
    assert {
        json.loads(str(point.payload["processing_version"]))["source_type"]
        for point in points
    } == {"file"}


def test_unchanged_ingest_uses_fast_skip_without_reparsing(tmp_path: Path) -> None:
    doc = tmp_path / "billing.md"
    doc.write_text("Billing is paid by card or ACH.", encoding="utf-8")

    async def scenario() -> tuple[
        tuple[IngestedDocument, IngestedDocument], RecordingVectorStore
    ]:
        store = RecordingVectorStore()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_fast_skip",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            first = await core.add_file(doc, tenant_id="acme", collection="help")

            async def fail_prepare(
                *,
                file_bytes: bytes,
                filename: str,
                mime_type: str,
                path: str | None = None,
                tenant_id: str = "",
                collection: str = "",
                document_id: str = "",
                event_sink: EventSink | None = None,
            ) -> NoReturn:
                raise AssertionError("fast skip should not parse unchanged content")

            core._ingest._prepare_bytes = fail_prepare
            second = await core.add_file(doc, tenant_id="acme", collection="help")
            return (first, second), store
        finally:
            await core.close()

    (first, second), store = asyncio.run(scenario())

    assert first.ingest_state == "created"
    assert second.ingest_state == "unchanged"
    assert second.chunk_count == first.chunk_count
    assert second.metadata["skip_mode"] == "fast"
    assert [op for op in store.operations if op == "upsert"] == ["upsert"]


def test_materialized_skip_mode_preserves_previous_prepare_behavior(tmp_path: Path) -> None:
    doc = tmp_path / "billing.md"
    doc.write_text("Billing is paid by card or ACH.", encoding="utf-8")

    async def scenario() -> int:
        store = RecordingVectorStore()
        base = make_test_config(
            qdrant_collection="rag_core_materialized_skip",
            embedding_dimensions=4,
        )
        core = Engine(
            replace(
                base,
                ingest=replace(
                    base.ingest,
                    skip_unchanged=SKIP_UNCHANGED_MATERIALIZE,
                ),
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        try:
            await core.add_file(doc, tenant_id="acme", collection="help")
            prepare_calls = 0
            original_prepare = core._ingest._prepare_bytes

            async def count_prepare(
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
                nonlocal prepare_calls
                prepare_calls += 1
                return await original_prepare(
                    file_bytes=file_bytes,
                    filename=filename,
                    mime_type=mime_type,
                    path=path,
                    tenant_id=tenant_id,
                    collection=collection,
                    document_id=document_id,
                    event_sink=event_sink,
                )

            core._ingest._prepare_bytes = count_prepare
            await core.add_file(doc, tenant_id="acme", collection="help")
            return prepare_calls
        finally:
            await core.close()

    assert asyncio.run(scenario()) == 1
