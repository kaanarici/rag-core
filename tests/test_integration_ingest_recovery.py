"""Real-store proof for ingest torn-write rollback, write-ahead replay, and
delete-journal replay.

The recovery machinery (``core_ingest.py`` index-failure rollback, the
write-ahead journal, and the delete-recovery journal) is otherwise proven only
against recording-store and recording-indexer fakes. Per the repo trust model
fakes are not product proof, so these tests run the same scenarios against a
real embedded Qdrant (``location=":memory:"``) wrapped by a failure-injecting
proxy. They assert real store contents (search hits / point counts), not just
returned result objects.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Sequence

import pytest

from rag_core._engine.core_runtime import resolve_runtime_collection_name
from rag_core._engine.core_vector_store_factory import create_configured_vector_store
from rag_core.config import (
    DEFAULT_RERANKER_PROVIDER,
    EmbeddingConfig,
    QdrantConfig,
    RerankerConfig,
)
from rag_core.core import RAGCore, RAGCoreConfig
from rag_core.config.ingest_config import IngestConfig
from rag_core.demo import (
    _DEMO_COLLECTION_PREFIX,
    _DEMO_EMBEDDING_DIMENSIONS,
    DemoEmbeddingProvider,
    DemoSparseEmbedder,
)
from rag_core.search.provider_protocols import VectorStore
from rag_core.search.lexical_sidecar import PortableLexicalSidecar
from rag_core.search.providers.registry import VECTOR_STORES
from rag_core.search.request_models import DeleteFilter, SearchSidecarQuery
from rag_core.search.vector_models import SearchResult, VectorPoint

pytestmark = [pytest.mark.integration]


class _InjectedUpsertFailure(RuntimeError):
    """Raised by the proxy when an upsert is configured to fail."""


class _InjectedSidecarFailure(RuntimeError):
    """Raised by the sidecar when an upsert is configured to fail."""


class _FailingOnceSidecar(PortableLexicalSidecar):
    def __init__(self) -> None:
        super().__init__([])
        self.fail_next_upsert = False

    def upsert_records(self, records: Sequence[object]) -> None:
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise _InjectedSidecarFailure("injected sidecar failure")
        super().upsert_records(records)


class _FailingVectorStore:
    """Delegates every ``VectorStore`` method to a real store, but can be armed
    to raise on the Nth ``upsert`` (torn-write injection) or on the next
    ``delete`` (right-to-forget failure injection).

    A plain ``__getattr__`` proxy is used so the real adapter's properties and
    any methods not relevant to the test still pass through untouched, while the
    two write paths that recovery depends on stay interceptable.
    """

    def __init__(self, inner: VectorStore) -> None:
        self._inner = inner
        self._upsert_calls = 0
        self._fail_upsert_on_call: int | None = None
        self._fail_next_delete = False

    def fail_upsert_on_call(self, call_number: int | None) -> None:
        self._upsert_calls = 0
        self._fail_upsert_on_call = call_number

    def fail_next_delete(self, enabled: bool) -> None:
        self._fail_next_delete = enabled

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        self._upsert_calls += 1
        if (
            self._fail_upsert_on_call is not None
            and self._upsert_calls == self._fail_upsert_on_call
        ):
            raise _InjectedUpsertFailure("injected upsert failure")
        await self._inner.upsert(points)

    async def delete(self, filter: DeleteFilter) -> None:
        if self._fail_next_delete:
            self._fail_next_delete = False
            raise RuntimeError("injected delete failure")
        await self._inner.delete(filter)

    def __getattr__(self, name: str) -> Any:
        # Fires only for attributes not defined on the proxy itself, so the
        # intercepted ``upsert``/``delete`` above always win.
        return getattr(self._inner, name)


def _build_real_store(config: RAGCoreConfig, embedding: DemoEmbeddingProvider) -> VectorStore:
    collection_name = resolve_runtime_collection_name(
        config=config,
        model_name=embedding.model_name,
        dimensions=embedding.dimensions,
    )
    return create_configured_vector_store(
        config=config,
        collection_name=collection_name,
        dense_dimensions=embedding.dimensions,
        vector_stores=VECTOR_STORES,
        embedding_model=embedding.model_name,
    )


def _build_core_with_proxy(
    *, manifest_directory: Path | None
) -> tuple[RAGCore, _FailingVectorStore]:
    """Construct a demo-equivalent ``RAGCore`` whose single real Qdrant
    ``:memory:`` store is wrapped by the failure-injecting proxy.

    The proxy is injected at construction so the indexer, search runner, and
    ingestor all share the one wrapped store. No second in-memory database and
    no post-hoc attribute swapping.
    """
    config = RAGCoreConfig(
        qdrant=QdrantConfig(
            location=":memory:",
            collection=f"{_DEMO_COLLECTION_PREFIX}_recovery_{uuid.uuid4().hex}",
            dimension_aware_collection=False,
        ),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=_DEMO_EMBEDDING_DIMENSIONS,
        ),
        reranker=RerankerConfig(provider=DEFAULT_RERANKER_PROVIDER),
        ingest=IngestConfig(manifest_directory=manifest_directory),
    )
    embedding = DemoEmbeddingProvider()
    proxy = _FailingVectorStore(_build_real_store(config, embedding))
    core = RAGCore(
        config,
        embedding_provider=embedding,
        sparse_embedder=DemoSparseEmbedder(),
        vector_store=proxy,
    )
    return core, proxy


def _build_core_with_sidecar(
    *,
    manifest_directory: Path | None,
    sidecar: _FailingOnceSidecar,
) -> RAGCore:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(
            location=":memory:",
            collection=f"{_DEMO_COLLECTION_PREFIX}_sidecar_{uuid.uuid4().hex}",
            dimension_aware_collection=False,
        ),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=_DEMO_EMBEDDING_DIMENSIONS,
        ),
        reranker=RerankerConfig(provider=DEFAULT_RERANKER_PROVIDER),
        ingest=IngestConfig(manifest_directory=manifest_directory),
    )
    embedding = DemoEmbeddingProvider()
    return RAGCore(
        config,
        embedding_provider=embedding,
        sparse_embedder=DemoSparseEmbedder(),
        vector_store=_build_real_store(config, embedding),
        search_sidecar=sidecar,
    )


async def _search_hits(core: RAGCore, query: str, *, document_id: str | None = None) -> list[SearchResult]:
    hits = await core.search(
        query=query,
        namespace="recovery",
        corpus_ids=["docs"],
        limit=50,
        rerank=False,
    )
    if document_id is None:
        return hits
    return [hit for hit in hits if hit.document_id == document_id]


async def _sidecar_hits(
    sidecar: PortableLexicalSidecar,
    query: str,
) -> list[SearchResult]:
    return await sidecar.search(
        SearchSidecarQuery(query=query, namespace="recovery", corpus_ids=["docs"])
    )


def test_proxy_smoke_ingests_and_searches_against_real_store(tmp_path: Path) -> None:
    async def go() -> None:
        core, _proxy = _build_core_with_proxy(manifest_directory=tmp_path / "manifest")
        async with core:
            ingested = await core.ingest_bytes(
                file_bytes=b"Invoices can be paid by card or ACH bank transfer.",
                filename="billing.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-smoke",
            )
            assert ingested.chunk_count >= 1
            hits = await _search_hits(core, "How can invoices be paid?", document_id="doc-smoke")
            assert hits, "document must be searchable after a healthy ingest"

    asyncio.run(go())


def test_failed_upsert_rolls_back_real_store_state(tmp_path: Path) -> None:
    """A torn upsert under a new content_sha256 must leave no v2 (and no mixed
    v1/v2) content retrievable; a clean retry then serves only v2.
    """

    async def go() -> None:
        core, proxy = _build_core_with_proxy(manifest_directory=tmp_path / "manifest")
        async with core:
            await core.ingest_bytes(
                file_bytes=b"alpha invoices billing card payment original version one",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-roll",
            )
            v1_hits = await _search_hits(core, "alpha original version one", document_id="doc-roll")
            assert v1_hits, "v1 must be searchable before the torn re-ingest"

            proxy.fail_upsert_on_call(1)
            with pytest.raises(_InjectedUpsertFailure):
                await core.ingest_bytes(
                    file_bytes=b"beta refund cancellation gamma rewritten version two",
                    filename="doc.md",
                    mime_type="text/markdown",
                    namespace="recovery",
                    corpus_id="docs",
                    document_id="doc-roll",
                    force_reindex=True,
                )
            proxy.fail_upsert_on_call(None)

            # Real-store post-condition: rollback purged the document entirely
            # (deterministic point ids per chunk mean a partial upsert mixes
            # content irrecoverably; the engine chooses unsearchable-until-retry
            # over serving a mixed v1/v2 set).
            v2_hits = await _search_hits(core, "beta gamma rewritten version two", document_id="doc-roll")
            assert v2_hits == [], "no v2 content may be retrievable after rollback"
            residue = await _search_hits(core, "alpha original version one", document_id="doc-roll")
            assert residue == [], "rollback purges the torn document; no mixed set survives"

            # The write-ahead journal entry left by the torn ingest enables the
            # retry's resume path; the clean retry then serves only v2.
            await core.ingest_bytes(
                file_bytes=b"beta refund cancellation gamma rewritten version two",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-roll",
                force_reindex=True,
            )
            retried = await _search_hits(core, "beta gamma rewritten version two", document_id="doc-roll")
            assert retried, "v2 must be searchable after a clean retry"
            old = await _search_hits(core, "alpha original version one", document_id="doc-roll")
            for hit in old:
                assert "alpha" not in hit.text.lower(), "v1 content must not survive the v2 retry"

    asyncio.run(go())


def test_existing_reindex_sidecar_failure_without_journal_is_retry_healable() -> None:
    async def go() -> None:
        sidecar = _FailingOnceSidecar()
        core = _build_core_with_sidecar(manifest_directory=None, sidecar=sidecar)
        async with core:
            await core.ingest_bytes(
                file_bytes=b"alpha original invoice card payment version",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-sidecar",
            )

            sidecar.fail_next_upsert = True
            with pytest.raises(_InjectedSidecarFailure):
                await core.ingest_bytes(
                    file_bytes=b"theta renewal retry sidecar repair content",
                    filename="doc.md",
                    mime_type="text/markdown",
                    namespace="recovery",
                    corpus_id="docs",
                    document_id="doc-sidecar",
                    force_reindex=True,
                )
            assert await _sidecar_hits(sidecar, "theta renewal retry") == []

            retried = await core.ingest_bytes(
                file_bytes=b"theta renewal retry sidecar repair content",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-sidecar",
            )

            assert retried.ingest_state == "created"
            assert await _sidecar_hits(sidecar, "theta renewal retry")

    asyncio.run(go())


def test_pending_write_ahead_replays_on_next_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash AFTER the store upsert but BEFORE the final manifest write leaves
    a pending write-ahead entry plus orphan points on disk. The next healthy
    ingest of the same triple must replay the purge and land exactly one
    consistent copy. No duplicate points.

    Injection seam: ``core_ingest.write_final_manifest`` is monkeypatched to
    raise ``KeyboardInterrupt`` (a ``BaseException``, not caught by the
    ``except Exception`` rollback in ``_ingest_inside_fence``), so the upsert
    survives and ``record_committed`` is never reached. Exactly the torn state
    a real crash produces.
    """

    async def go() -> None:
        core, _proxy = _build_core_with_proxy(manifest_directory=tmp_path / "manifest")
        async with core:
            from rag_core._engine import core_ingest as core_ingest_module

            def _crash(**_kwargs: object) -> None:
                raise KeyboardInterrupt("simulated crash before manifest write")

            monkeypatch.setattr(core_ingest_module, "write_final_manifest", _crash)
            with pytest.raises(KeyboardInterrupt):
                await core.ingest_bytes(
                    file_bytes=b"delta shipping logistics warehouse fulfillment pending",
                    filename="doc.md",
                    mime_type="text/markdown",
                    namespace="recovery",
                    corpus_id="docs",
                    document_id="doc-wal",
                )
            monkeypatch.undo()

            # Replay path: the next healthy ingest purges the orphan via
            # resume_pending_write_ahead before landing fresh content.
            await core.ingest_bytes(
                file_bytes=b"delta shipping logistics warehouse fulfillment pending",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-wal",
            )

            hits = await _search_hits(core, "delta shipping logistics warehouse", document_id="doc-wal")
            assert hits, "document must be searchable after write-ahead replay"
            # No duplicate points: the resume purge removed the orphan copy, so
            # the real store holds exactly one record for the document.
            record = await core._store.get_document_record(
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-wal",
            )
            assert record is not None
            chunk_count = record.chunk_count
            assert chunk_count >= 1
            unique_chunk_indexes = {hit.chunk_index for hit in hits}
            assert len(unique_chunk_indexes) == chunk_count, (
                "replay must not leave duplicate points: "
                f"{len(unique_chunk_indexes)} distinct chunk indexes for "
                f"{chunk_count} stored chunks"
            )

    asyncio.run(go())


def test_failed_delete_journal_replays_on_reingest(tmp_path: Path) -> None:
    """A failed store delete must report the failure honestly via the
    ``DeleteDocumentResult`` tri-state contract and leave a delete-journal
    entry. Re-ingesting the same triple replays the purge so only the new
    content is searchable and the old content is fully gone.
    """

    async def go() -> None:
        core, proxy = _build_core_with_proxy(manifest_directory=tmp_path / "manifest")
        async with core:
            await core.ingest_bytes(
                file_bytes=b"epsilon onboarding tutorial getting started guide original",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-del",
            )
            assert await _search_hits(core, "epsilon onboarding tutorial guide", document_id="doc-del")

            proxy.fail_next_delete(True)
            with pytest.raises(RuntimeError, match="injected delete failure"):
                await core.delete_document(
                    document_id="doc-del",
                    namespace="recovery",
                    corpus_id="docs",
                )

            # A journal entry now tracks the partial delete so the next ingest
            # of the triple replays the purge. The store delete failed, so the
            # old content is still present until that replay runs.
            from rag_core._engine.core_ingest_delete_journal import DeleteRecoveryJournal

            journal = DeleteRecoveryJournal(directory=tmp_path / "manifest")
            latest = journal.latest_entry(
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-del",
            )
            assert latest is not None
            assert latest.completed is False

            proxy.fail_next_delete(False)
            await core.ingest_bytes(
                file_bytes=b"zeta security compliance audit logging replacement content",
                filename="doc.md",
                mime_type="text/markdown",
                namespace="recovery",
                corpus_id="docs",
                document_id="doc-del",
                force_reindex=True,
            )

            new_hits = await _search_hits(core, "zeta security compliance audit", document_id="doc-del")
            assert new_hits, "replacement content must be searchable after reingest"
            for hit in new_hits:
                assert "epsilon" not in hit.text.lower()
            old_hits = await _search_hits(core, "epsilon onboarding tutorial guide", document_id="doc-del")
            for hit in old_hits:
                assert "epsilon" not in hit.text.lower(), "old content must be fully gone after journal replay"

    asyncio.run(go())
