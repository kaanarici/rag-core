"""Serialize ingest/delete for one document without retaining idle locks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

DocumentKey = tuple[str, str, str]


@dataclass
class _FenceEntry:
    lock: asyncio.Lock
    waiters: int = 0


@dataclass
class _CollectionGate:
    condition: asyncio.Condition
    readers: int = 0
    writer: bool = False
    waiting_writers: int = 0
    users: int = 0


class DocumentFence:
    def __init__(self) -> None:
        self._entries: dict[DocumentKey, _FenceEntry] = {}
        self._collections: dict[tuple[str, str], _CollectionGate] = {}

    @asynccontextmanager
    async def acquire(
        self, *, tenant_id: str, collection: str, document_id: str
    ) -> AsyncIterator[None]:
        async with self.collection(
            tenant_id=tenant_id, collection=collection, exclusive=False
        ):
            key = (tenant_id, collection, document_id)
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = _FenceEntry(asyncio.Lock())
            entry.waiters += 1
            try:
                async with entry.lock:
                    yield
            finally:
                entry.waiters -= 1
                if entry.waiters == 0:
                    del self._entries[key]

    @asynccontextmanager
    async def collection(
        self, *, tenant_id: str, collection: str, exclusive: bool = True
    ) -> AsyncIterator[None]:
        key = (tenant_id, collection)
        gate = self._collections.get(key)
        if gate is None:
            gate = self._collections[key] = _CollectionGate(asyncio.Condition())
        gate.users += 1
        entered = False
        try:
            async with gate.condition:
                if exclusive:
                    gate.waiting_writers += 1
                    try:
                        await gate.condition.wait_for(
                            lambda: not gate.writer and gate.readers == 0
                        )
                    finally:
                        gate.waiting_writers -= 1
                        gate.condition.notify_all()
                    gate.writer = True
                else:
                    await gate.condition.wait_for(
                        lambda: not gate.writer and gate.waiting_writers == 0
                    )
                    gate.readers += 1
                entered = True
            yield
        finally:
            if entered:
                async with gate.condition:
                    if exclusive:
                        gate.writer = False
                    else:
                        gate.readers -= 1
                    gate.condition.notify_all()
            gate.users -= 1
            if gate.users == 0:
                del self._collections[key]

    def active_keys(self) -> tuple[DocumentKey, ...]:
        return tuple(self._entries)
