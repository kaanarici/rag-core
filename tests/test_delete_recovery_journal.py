"""Right-to-forget recovery journal: partial-delete replay across crashes.

The delete path runs vector store -> sidecar -> embedding cache -> chunk
context cache -> manifest. A crash between stages writes a journal entry;
the next ``ingest_bytes`` on the same ``(namespace, collection, document_id)``
triple replays the purge before writing new content.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from rag_core._engine.core_ingest_delete import (
    delete_ingested_document,
    resume_partial_delete,
)
from rag_core._engine.core_ingest_delete_journal import (
    DELETE_STAGES_IN_ORDER,
    DeleteRecoveryJournal,
    STAGE_LEXICAL_SIDECAR,
    STAGE_VECTOR_STORE,
)
from rag_core.search.indexer_models import DeleteAck


class _RecordingIndexer:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def delete_document(
        self,
        *,
        document_id: str,
        namespace: str,
        collection: str,
    ) -> DeleteAck:
        self.calls += 1
        if self.fail:
            raise RuntimeError("vector store delete failed")
        return DeleteAck(succeeded=True, deleted_point_count=-1)


class _RecordingSidecar:
    def __init__(self, *, fail: bool = False) -> None:
        self.deleted: list[tuple[str, str, str]] = []
        self.fail = fail

    def delete_document(
        self,
        *,
        namespace: str,
        document_id: str,
        collection: str | None = None,
    ) -> None:
        self.deleted.append((namespace, document_id, str(collection)))
        if self.fail:
            raise RuntimeError("sidecar delete failed")


class _RecordingCache:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.fail = fail

    async def delete_by_document_scope(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> int:
        self.calls.append((namespace, collection, document_id))
        if self.fail:
            raise RuntimeError("cache purge failed")
        return 0


def test_journal_records_partial_purge_when_sidecar_fails(tmp_path: Path) -> None:
    indexer = _RecordingIndexer()
    sidecar = _RecordingSidecar(fail=True)
    embedding_cache = _RecordingCache()
    chunk_context_cache = _RecordingCache()

    async def run() -> None:
        with pytest.raises(RuntimeError, match="sidecar delete failed"):
            await delete_ingested_document(
                indexer=cast(Any, indexer),
                sidecar=cast(Any, sidecar),
                event_sink=None,
                manifest_directory=tmp_path,
                document_id="doc-007",
                namespace="workspace-alpha",
                collection="restricted",
                embedding_cache=cast(Any, embedding_cache),
                chunk_context_cache=cast(Any, chunk_context_cache),
            )

    asyncio.run(run())

    # Vector store completed; later stages never ran because the canonical
    # right-to-forget order halts on the first failure.
    assert indexer.calls == 1
    assert sidecar.deleted == [("workspace-alpha", "doc-007", "restricted")]
    assert embedding_cache.calls == []
    assert chunk_context_cache.calls == []

    journal = DeleteRecoveryJournal(directory=tmp_path)
    latest = journal.latest_entry(
        namespace="workspace-alpha",
        collection="restricted",
        document_id="doc-007",
    )
    assert latest is not None
    assert latest.completed is False
    assert latest.last_error_type == "RuntimeError"
    assert latest.last_error_stage == STAGE_LEXICAL_SIDECAR
    assert latest.stages_completed == (STAGE_VECTOR_STORE,)


def test_journal_records_completed_on_full_purge(tmp_path: Path) -> None:
    indexer = _RecordingIndexer()
    sidecar = _RecordingSidecar()
    embedding_cache = _RecordingCache()
    chunk_context_cache = _RecordingCache()

    async def run() -> None:
        result = await delete_ingested_document(
            indexer=cast(Any, indexer),
            sidecar=cast(Any, sidecar),
            event_sink=None,
            manifest_directory=tmp_path,
            document_id="doc-007",
            namespace="ns",
            collection="public",
            embedding_cache=cast(Any, embedding_cache),
            chunk_context_cache=cast(Any, chunk_context_cache),
        )
        assert result.index_deleted is True
        assert result.vector_store_acked is True
        assert result.lexical_sidecar_purged is True
        assert result.embedding_cache_purged is True
        assert result.chunk_context_cache_purged is True
        # No manifest file created during this test, so manifest_removed is False.

    asyncio.run(run())

    journal = DeleteRecoveryJournal(directory=tmp_path)
    latest = journal.latest_entry(
        namespace="ns",
        collection="public",
        document_id="doc-007",
    )
    assert latest is not None
    assert latest.completed is True
    assert latest.stages_completed == DELETE_STAGES_IN_ORDER


def test_resume_partial_delete_completes_purge_after_restart(tmp_path: Path) -> None:
    # Simulate a crash: first attempt fails on sidecar; resume_partial_delete
    # then completes the remaining stages on a healthy sidecar.
    flaky_sidecar = _RecordingSidecar(fail=True)
    indexer = _RecordingIndexer()
    embedding_cache = _RecordingCache()
    chunk_context_cache = _RecordingCache()

    async def crash() -> None:
        with pytest.raises(RuntimeError):
            await delete_ingested_document(
                indexer=cast(Any, indexer),
                sidecar=cast(Any, flaky_sidecar),
                event_sink=None,
                manifest_directory=tmp_path,
                document_id="doc-A",
                namespace="ns",
                collection="public",
                embedding_cache=cast(Any, embedding_cache),
                chunk_context_cache=cast(Any, chunk_context_cache),
            )

    asyncio.run(crash())

    healed_sidecar = _RecordingSidecar()

    async def replay() -> None:
        result = await resume_partial_delete(
            indexer=cast(Any, indexer),
            sidecar=cast(Any, healed_sidecar),
            event_sink=None,
            manifest_directory=tmp_path,
            document_id="doc-A",
            namespace="ns",
            collection="public",
            embedding_cache=cast(Any, embedding_cache),
            chunk_context_cache=cast(Any, chunk_context_cache),
        )
        assert result is not None
        assert result.index_deleted is True
        assert result.lexical_sidecar_purged is True

    asyncio.run(replay())

    # Vector store delete is idempotent; healthy sidecar succeeded;
    # caches both ran the scoped purge.
    assert indexer.calls == 2
    assert healed_sidecar.deleted == [("ns", "doc-A", "public")]
    assert embedding_cache.calls == [("ns", "public", "doc-A")]
    assert chunk_context_cache.calls == [("ns", "public", "doc-A")]

    journal = DeleteRecoveryJournal(directory=tmp_path)
    latest = journal.latest_entry(
        namespace="ns",
        collection="public",
        document_id="doc-A",
    )
    assert latest is not None
    assert latest.completed is True


def test_resume_partial_delete_is_noop_when_no_journal_entry(tmp_path: Path) -> None:
    indexer = _RecordingIndexer()

    async def run() -> None:
        result = await resume_partial_delete(
            indexer=cast(Any, indexer),
            sidecar=None,
            event_sink=None,
            manifest_directory=tmp_path,
            document_id="doc-fresh",
            namespace="ns",
            collection="public",
        )
        # Nothing pending -> caller proceeds with normal ingest.
        assert result is None

    asyncio.run(run())
    assert indexer.calls == 0


def test_delete_journal_latest_entry_uses_file_order_not_timestamps(
    tmp_path: Path,
) -> None:
    # Monotonic clocks reset across OS reboots: a pre-reboot incomplete entry
    # can carry a larger timestamp than a post-reboot completed entry. File
    # append order, not the timestamp, must decide the latest state.
    journal = DeleteRecoveryJournal(directory=tmp_path)
    journal.record(
        namespace="acme",
        collection="help",
        document_id="doc-1",
        stages_completed=("vector_store",),
        completed=False,
        created_at_ns=10**18,
    )
    entry = journal.record(
        namespace="acme",
        collection="help",
        document_id="doc-1",
        stages_completed=DELETE_STAGES_IN_ORDER,
        completed=True,
        created_at_ns=1,
    )
    # Force the completed entry's stamps below the stale pending entry's.
    journal.path.write_text(
        journal.path.read_text(encoding="utf-8").replace(
            f'"updated_at_ns":{entry.updated_at_ns}', '"updated_at_ns":2'
        ),
        encoding="utf-8",
    )

    latest = journal.latest_entry(
        namespace="acme", collection="help", document_id="doc-1"
    )
    assert latest is not None
    assert latest.completed is True
    assert journal.pending_entries() == []


def test_delete_journal_compacts_completed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_core._engine import core_ingest_delete_journal as module

    monkeypatch.setattr(module, "_COMPACT_MIN_BYTES", 1)
    journal = DeleteRecoveryJournal(directory=tmp_path)
    journal.record(
        namespace="acme",
        collection="help",
        document_id="doc-resolved",
        stages_completed=("vector_store",),
        completed=False,
    )
    journal.record(
        namespace="acme",
        collection="help",
        document_id="doc-resolved",
        stages_completed=DELETE_STAGES_IN_ORDER,
        completed=True,
    )
    journal.record(
        namespace="acme",
        collection="help",
        document_id="doc-pending",
        stages_completed=("vector_store",),
        completed=False,
    )

    lines = [
        line
        for line in journal.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    pending = journal.pending_entries()
    assert [entry.document_id for entry in pending] == ["doc-pending"]
