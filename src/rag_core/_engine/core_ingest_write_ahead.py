"""Ingest write-ahead journal: torn upsert recovery across crashes.

Manifest reconciliation depends on the manifest entry's ``chunk_count``
matching what the vector store actually holds for a document. The ingest
sequence is ``prepare -> indexer.upsert -> manifest.write``; a crash between
``upsert`` and ``manifest.write`` strands chunks in the vector store with no
manifest record. Worse, a partial Qdrant batch upsert can land *some* chunks
before raising. The index-failure rollback in ``core_ingest`` repairs the
manifest by reading the store, but cannot infer whether the store still holds
torn-write residue under the new content_sha256.

The journal closes that window:

1. Before ``indexer.index_document(...)``, write an entry with intent
   (namespace, collection, document_id, content_sha256, expected chunk_count,
   state=``upserted_pending_manifest``).
2. After ``manifest.write`` succeeds, mark the entry committed
   (state=``manifest_written``).
3. On the next ingest of the same ``(namespace, collection, document_id)``
   triple, or on engine startup, ``resume_pending_write_ahead`` reads the
   journal: any entry stuck in ``upserted_pending_manifest`` is purged
   best-effort (``indexer.delete_document``) before fresh content lands.

This is parallel to the right-to-forget recovery journal
(``core_ingest_delete_journal``). One file each because the failure modes
differ. Delete journal records *intent to forget*, write-ahead records
*intent to land*. Both append-only JSONL under the manifest directory.

Deletion trigger: collapse into a unified ingest/delete recovery log once a
real adapter needs cross-file replay (today, the per-doc fence in
``core_ingest`` makes single-process replay sufficient).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast

from rag_core._engine.core_jsonl_journal import JsonlRecoveryJournal
from rag_core.events.emit import stage_guard

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.search.indexer import DocumentIndexer

# Canonical states for the write-ahead entry. ``upserted_pending_manifest``
# is the only state that requires recovery; ``manifest_written`` means the
# critical section completed and the entry is informational.
STATE_UPSERTED_PENDING_MANIFEST = "upserted_pending_manifest"
STATE_MANIFEST_WRITTEN = "manifest_written"

WRITE_AHEAD_FILE_NAME = ".ingest_write_ahead.jsonl"

# Compaction trigger: once the journal file exceeds this size, resolving a
# pending entry rewrites the file keeping only the still-pending tail. Keeps
# per-ingest replay reads O(active pending), not O(collection history).
_COMPACT_MIN_BYTES = 64 * 1024


@dataclass(frozen=True, kw_only=True)
class WriteAheadEntry:
    """One record of an in-flight ingest's progress through upsert/manifest."""

    namespace: str
    collection: str
    document_id: str
    content_sha256: str
    expected_chunk_count: int
    state: str
    created_at_ns: int = 0
    updated_at_ns: int = 0

    _CHARSET: ClassVar[str] = "utf-8"

    def to_jsonl(self) -> str:
        payload = {
            "namespace": self.namespace,
            "collection": self.collection,
            "document_id": self.document_id,
            "content_sha256": self.content_sha256,
            "expected_chunk_count": self.expected_chunk_count,
            "state": self.state,
            "created_at_ns": self.created_at_ns,
            "updated_at_ns": self.updated_at_ns,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class IngestWriteAheadJournal(JsonlRecoveryJournal[WriteAheadEntry]):
    """Append-only JSONL journal at ``<directory>/.ingest_write_ahead.jsonl``.

    The journal is opportunistic: a missing or torn file is not fatal.
    Recovery prefers a noisy fresh attempt over a hidden stuck state.
    ``record_pending(...)`` runs before ``indexer.upsert``;
    ``record_committed(...)`` runs after ``manifest.write``;
    ``pending_entries_for(...)`` and ``latest_entry(...)`` drive replay.
    """

    _FILE_NAME: ClassVar[str] = WRITE_AHEAD_FILE_NAME
    _CHARSET: ClassVar[str] = WriteAheadEntry._CHARSET

    def record_pending(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        content_sha256: str,
        expected_chunk_count: int,
    ) -> WriteAheadEntry:
        return self._record(
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            content_sha256=content_sha256,
            expected_chunk_count=expected_chunk_count,
            state=STATE_UPSERTED_PENDING_MANIFEST,
        )

    def record_committed(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        content_sha256: str,
        expected_chunk_count: int,
    ) -> WriteAheadEntry:
        return self._record(
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            content_sha256=content_sha256,
            expected_chunk_count=expected_chunk_count,
            state=STATE_MANIFEST_WRITTEN,
        )

    def _record(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
        content_sha256: str,
        expected_chunk_count: int,
        state: str,
    ) -> WriteAheadEntry:
        # Wall clock, informational only: entry ordering is file append order,
        # which survives process restarts and OS reboots (monotonic does not).
        now_ns = time.time_ns()
        entry = WriteAheadEntry(
            namespace=namespace,
            collection=collection,
            document_id=document_id,
            content_sha256=content_sha256,
            expected_chunk_count=expected_chunk_count,
            state=state,
            created_at_ns=now_ns,
            updated_at_ns=now_ns,
        )
        self._append(entry)
        if state == STATE_MANIFEST_WRITTEN:
            self._compact_if_oversized()
        return entry

    def _parse(self, payload: Mapping[str, object]) -> WriteAheadEntry | None:
        try:
            return WriteAheadEntry(
                namespace=str(payload.get("namespace", "")),
                collection=str(payload.get("collection", "")),
                document_id=str(payload.get("document_id", "")),
                content_sha256=str(payload.get("content_sha256", "")),
                expected_chunk_count=int(
                    cast(Any, payload.get("expected_chunk_count", 0))
                ),
                state=str(payload.get("state", "")),
                created_at_ns=int(cast(Any, payload.get("created_at_ns", 0))),
                updated_at_ns=int(cast(Any, payload.get("updated_at_ns", 0))),
            )
        except (TypeError, ValueError):
            return None

    def _is_pending(self, entry: WriteAheadEntry) -> bool:
        return entry.state == STATE_UPSERTED_PENDING_MANIFEST

    def _compact_min_bytes(self) -> int:
        return _COMPACT_MIN_BYTES


def write_ahead_journal_for(manifest_directory: Path | None) -> IngestWriteAheadJournal | None:
    """Build a journal under the manifest directory, when configured."""
    if manifest_directory is None:
        return None
    return IngestWriteAheadJournal(directory=manifest_directory)


async def best_effort_rollback_delete(
    *,
    indexer: "DocumentIndexer",
    event_sink: "EventSink | None",
    namespace: str,
    collection: str,
    document_id: str,
) -> None:
    """Purge any torn upsert residue before restoring the manifest.

    A Qdrant batch upsert can land *some* points before raising. Calling
    ``indexer.delete_document`` here means a half-written chunk set cannot
    leak into the caller's reconciler view as ``chunk_count == manifest``
    while the store holds a different number. The call is best-effort:
    failures are surfaced via the ``index`` ``StageError`` from
    ``stage_guard`` but never mask the original ``index`` failure that
    triggered the rollback.
    """
    try:
        with stage_guard(event_sink, stage="index"):
            await indexer.delete_document(
                document_id=document_id,
                namespace=namespace,
                collection=collection,
            )
    except Exception:
        # Best effort. Recovery proceeds with manifest repair and the
        # caller surfaces the original index failure.
        pass


async def resume_pending_write_ahead(
    *,
    indexer: "DocumentIndexer",
    event_sink: "EventSink | None",
    manifest_directory: Path | None,
    namespace: str,
    collection: str,
    document_id: str,
    journal: IngestWriteAheadJournal | None = None,
) -> bool:
    """Replay an interrupted ingest of the same triple before fresh content.

    Called from ``ingest_bytes`` BEFORE a fresh write goes in. If the
    write-ahead journal has a pending entry, meaning a prior ingest
    crashed between ``indexer.upsert`` and ``manifest.write``, best-effort
    purge the orphaned chunks so the new ingest cannot pile on top of a
    torn write. The journal is then closed by writing a committed entry
    against the same triple so a re-replay is a no-op.

    Returns ``True`` when a pending entry was resolved (caller may want to
    audit); ``False`` when there was nothing to do. Always returns instead
    of raising. Torn-write recovery is best effort, and the canonical
    error path is the next ``index_document`` call observing whatever
    state the store now has.
    """
    journal = journal or write_ahead_journal_for(manifest_directory)
    if journal is None:
        return False
    latest = journal.latest_entry(
        namespace=namespace,
        collection=collection,
        document_id=document_id,
    )
    if latest is None or latest.state != STATE_UPSERTED_PENDING_MANIFEST:
        return False
    # Always purge a pending entry. The manifest cannot be used to infer "this
    # is actually committed, skip the delete": for an existing document a
    # staged manifest entry carrying the NEW content_sha256 is written BEFORE
    # the upsert (see core_ingest), so a torn upsert leaves a matching manifest
    # entry that would falsely read as committed and strand the partial chunks.
    # Purging is also self-healing: resolve_ingest_decision reads the store, so
    # the same-triple ingest that follows this replay re-indexes from scratch.
    try:
        with stage_guard(event_sink, stage="index"):
            await indexer.delete_document(
                document_id=document_id,
                namespace=namespace,
                collection=collection,
            )
    except Exception:
        # Best-effort: the journal stays pending so a future attempt can
        # retry, but the new ingest must proceed. Refusing here would
        # strand the doc in a state operators cannot resolve without the
        # journal file. The rollback path already emits ``StageError``.
        return False
    # Close the entry: a fresh ingest will record its own pending /
    # committed pair. Mark the resume as "committed" against the
    # recovered content_sha256 so the latest-entry lookup returns a
    # ``manifest_written`` row and ``pending_entries`` skips it.
    journal.record_committed(
        namespace=namespace,
        collection=collection,
        document_id=document_id,
        content_sha256=latest.content_sha256,
        expected_chunk_count=latest.expected_chunk_count,
    )
    return True


__all__ = [
    "STATE_MANIFEST_WRITTEN",
    "STATE_UPSERTED_PENDING_MANIFEST",
    "WRITE_AHEAD_FILE_NAME",
    "IngestWriteAheadJournal",
    "WriteAheadEntry",
    "best_effort_rollback_delete",
    "resume_pending_write_ahead",
    "write_ahead_journal_for",
]
