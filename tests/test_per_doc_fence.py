"""Per-document concurrency fence.

Two concurrent operations on the same ``(namespace, corpus_id, document_id)``
triple must serialize; different triples run concurrently. The fence's job
is to prevent ingest and delete from interleaving on the same document, not
to throttle the whole engine.
"""

from __future__ import annotations

import asyncio

import pytest

from rag_core._engine.core_ingest_fence import DocumentFence

pytestmark = [pytest.mark.plumbing]


async def _hold(
    fence: DocumentFence,
    *,
    namespace: str,
    corpus_id: str,
    document_id: str,
    log: list[str],
    enter_tag: str,
    exit_tag: str,
    gate: asyncio.Event,
    release: asyncio.Event,
) -> None:
    async with fence.acquire(
        namespace=namespace, corpus_id=corpus_id, document_id=document_id,
    ):
        log.append(enter_tag)
        gate.set()
        await release.wait()
        log.append(exit_tag)


def test_fence_serializes_same_triple() -> None:
    async def _run() -> None:
        fence = DocumentFence()
        log: list[str] = []
        first_in = asyncio.Event()
        first_release = asyncio.Event()
        second_in = asyncio.Event()
        second_release = asyncio.Event()

        first = asyncio.create_task(
            _hold(
                fence,
                namespace="acme", corpus_id="public", document_id="doc-1",
                log=log,
                enter_tag="A-enter", exit_tag="A-exit",
                gate=first_in, release=first_release,
            )
        )
        await first_in.wait()

        second = asyncio.create_task(
            _hold(
                fence,
                namespace="acme", corpus_id="public", document_id="doc-1",
                log=log,
                enter_tag="B-enter", exit_tag="B-exit",
                gate=second_in, release=second_release,
            )
        )
        # Give the second task time to attempt acquisition; it must block.
        await asyncio.sleep(0.05)
        assert log == ["A-enter"]
        assert not second_in.is_set()

        first_release.set()
        await first

        await second_in.wait()
        second_release.set()
        await second

        assert log == ["A-enter", "A-exit", "B-enter", "B-exit"]

    asyncio.run(_run())


def test_fence_runs_different_triples_in_parallel() -> None:
    async def _run() -> None:
        fence = DocumentFence()
        log: list[str] = []
        a_in = asyncio.Event()
        a_release = asyncio.Event()
        b_in = asyncio.Event()
        b_release = asyncio.Event()

        a = asyncio.create_task(
            _hold(
                fence,
                namespace="acme", corpus_id="public", document_id="doc-1",
                log=log,
                enter_tag="A-enter", exit_tag="A-exit",
                gate=a_in, release=a_release,
            )
        )
        b = asyncio.create_task(
            _hold(
                fence,
                namespace="acme", corpus_id="public", document_id="doc-2",
                log=log,
                enter_tag="B-enter", exit_tag="B-exit",
                gate=b_in, release=b_release,
            )
        )
        # Both enter before either releases.
        await asyncio.wait_for(a_in.wait(), timeout=1.0)
        await asyncio.wait_for(b_in.wait(), timeout=1.0)
        assert set(log) == {"A-enter", "B-enter"}

        a_release.set()
        b_release.set()
        await asyncio.gather(a, b)

    asyncio.run(_run())


def test_fence_drops_entries_after_release() -> None:
    async def _run() -> None:
        fence = DocumentFence()
        async with fence.acquire(
            namespace="acme", corpus_id="public", document_id="doc-1",
        ):
            assert fence.active_keys() == (("acme", "public", "doc-1"),)
        # After the last holder releases, the registry is empty so we don't leak
        # one ``asyncio.Lock`` per triple ever ingested.
        assert fence.active_keys() == ()

    asyncio.run(_run())


def test_fence_drops_entry_only_when_no_waiters_remain() -> None:
    async def _run() -> None:
        fence = DocumentFence()
        enter_a = asyncio.Event()
        release_a = asyncio.Event()
        enter_b = asyncio.Event()
        release_b = asyncio.Event()
        log: list[str] = []

        a = asyncio.create_task(
            _hold(
                fence,
                namespace="acme", corpus_id="public", document_id="doc-x",
                log=log, enter_tag="A", exit_tag="A-done",
                gate=enter_a, release=release_a,
            )
        )
        await enter_a.wait()
        b = asyncio.create_task(
            _hold(
                fence,
                namespace="acme", corpus_id="public", document_id="doc-x",
                log=log, enter_tag="B", exit_tag="B-done",
                gate=enter_b, release=release_b,
            )
        )
        await asyncio.sleep(0.05)
        # Entry stays in the registry while B is queued behind A.
        assert ("acme", "public", "doc-x") in fence.active_keys()

        release_a.set()
        await a
        await enter_b.wait()
        release_b.set()
        await b
        assert fence.active_keys() == ()

    asyncio.run(_run())


def test_fence_lock_released_on_exception() -> None:
    async def _run() -> None:
        fence = DocumentFence()

        class _Boom(Exception):
            pass

        with pytest.raises(_Boom):
            async with fence.acquire(
                namespace="acme", corpus_id="public", document_id="doc-1",
            ):
                raise _Boom

        # After the exception unwinds the context, the lock is released and the
        # entry is dropped. The next acquire on the same triple proceeds.
        async with fence.acquire(
            namespace="acme", corpus_id="public", document_id="doc-1",
        ):
            assert fence.active_keys() == (("acme", "public", "doc-1"),)

    asyncio.run(_run())
