"""Per-document concurrency fence for ingest and delete.

Two concurrent operations on the same ``(namespace, collection, document_id)``
triple are serialized through a shared :class:`asyncio.Lock` so the
read-stale-delete-then-upsert flow in :mod:`rag_core._engine.core_ingest`
and the right-to-forget delete in :mod:`rag_core._engine.core_ingest_delete`
cannot interleave and produce a torn state (e.g. Delete acks the vector
store but the parallel re-ingest had already started, leaving orphan
chunks).

Operations on *different* triples run concurrently. The fence is keyed by
the triple, not the collection.

Lock identity is the contract: callers that need to coordinate (the
ingestor and its delete mixin) must share the same :class:`DocumentFence`
instance. ``CoreIngestor.__init__`` constructs one and the delete mixin
reads it back off ``self._fence``.

Memory is bounded by reference counting active waiters: when the last
holder releases, the entry is dropped. Long-lived locks for inactive
triples don't accumulate, so a large collection does not leak memory.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

# Triple identity: ``(namespace, collection, document_id)``. The same tuple
# the indexer uses to derive point identity via
# :func:`rag_core.search.policy.VectorStorePolicy.make_document_id`. Keeping
# the keys identical is what guarantees the fence and the indexer agree on
# what "this document" means; if the triple components ever diverge (e.g.
# collection casing), they're a bug in the caller, not in the fence.
DocumentKey = tuple[str, str, str]


@dataclass
class _FenceEntry:
    lock: asyncio.Lock
    waiters: int


class DocumentFence:
    """Serialize ingest and delete on the same document triple.

    Use as ``async with fence.acquire(namespace=..., collection=..., document_id=...):``.
    """

    def __init__(self) -> None:
        # ``_registry_lock`` guards the dict mutations only. It's never held
        # across the await on the per-document lock, so it cannot serialize
        # different triples against each other.
        self._registry_lock = asyncio.Lock()
        self._entries: dict[DocumentKey, _FenceEntry] = {}

    @asynccontextmanager
    async def acquire(
        self,
        *,
        namespace: str,
        collection: str,
        document_id: str,
    ) -> AsyncIterator[None]:
        key: DocumentKey = (namespace, collection, document_id)
        entry = await self._reserve(key)
        try:
            # Acquire inside the try so a cancellation or error during the
            # await still drops this waiter's reservation. Acquiring outside
            # would leak the refcount and pin the entry in the registry
            # forever, fencing the triple against all future operations.
            await entry.lock.acquire()
        except BaseException:
            await self._release(key)
            raise
        try:
            yield
        finally:
            entry.lock.release()
            await self._release(key)

    async def _reserve(self, key: DocumentKey) -> _FenceEntry:
        async with self._registry_lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _FenceEntry(lock=asyncio.Lock(), waiters=0)
                self._entries[key] = entry
            entry.waiters += 1
            return entry

    async def _release(self, key: DocumentKey) -> None:
        async with self._registry_lock:
            entry = self._entries.get(key)
            if entry is None:
                # Should not happen. But if it does, the registry is already
                # in the dropped state we want.
                return
            entry.waiters -= 1
            if entry.waiters <= 0:
                # No remaining holders or waiters: drop the entry so the dict
                # doesn't grow with one lock per triple ever ingested.
                self._entries.pop(key, None)

    def active_keys(self) -> tuple[DocumentKey, ...]:
        """Snapshot of triples currently held or queued. Test/diagnostic only."""
        return tuple(self._entries.keys())


__all__ = ["DocumentFence", "DocumentKey"]
