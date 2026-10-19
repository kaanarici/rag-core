"""Local storage and controlled workers prove ownership, not semantic quality."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import threading
import zipfile
from typing import Any

import pytest

from rag_core import EngineBusyError, EngineClosedError
from rag_core.core import Engine
from rag_core.core_models import Config, ParsedDocument
from rag_core._engine.runtime_operations import RuntimeOperations
from rag_core._async import run_sync
from tests.support import FakeEmbeddingProvider, make_test_config
from tests.support.local_store import LocalQdrantStore

pytestmark = pytest.mark.plumbing


class Store(LocalQdrantStore):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        await super().close()


class PausedEmbedding(FakeEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def embed_query(self, query: str) -> list[float]:
        self.started.set()
        await self.release.wait()
        return await super().embed_query(query)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.started.set()
        await self.release.wait()
        return await super().embed_texts(texts)


def core(*, embed: FakeEmbeddingProvider | None = None, **limits: Any) -> tuple[Engine, Store]:
    store = Store()
    return Engine(
        make_test_config(embedding_dimensions=4),
        embedding_provider=embed or FakeEmbeddingProvider(),
        vector_store=store,
        **limits,
    ), store


def operations(engine: Engine) -> dict[str, object]:
    snapshot = engine.describe_runtime()['operations']
    assert isinstance(snapshot, dict)
    return snapshot


async def until(check: Any) -> None:
    async with asyncio.timeout(3):
        while not check():
            await asyncio.sleep(0.001)


def test_close_drains_active_search_and_rejects_new_operations() -> None:
    async def scenario() -> None:
        embed = PausedEmbedding()
        engine, store = core(embed=embed)
        search = asyncio.create_task(engine.search(query='query', tenant_id='t', collection='c'))
        await asyncio.wait_for(embed.started.wait(), 3)
        assert operations(engine) == {
            'state': 'open', 'max_concurrent_ingests': 8,
            'max_concurrent_searches': 32, 'max_pending_operations': 64,
            'active_ingest_reservations': 0, 'active_searches': 1,
            'pending_operations': 0, 'active_operations': 1,
        }
        close = asyncio.create_task(engine.close())
        try:
            await until(lambda: operations(engine)['state'] == 'closing')
            assert not close.done() and store.close_calls == 0
            with pytest.raises(EngineClosedError):
                await engine.parse_bytes(file_bytes=b'hello', filename='a.txt', mime_type='text/plain')
            embed.release.set()
            assert await search == []
            await close
            await engine.close()
            assert store.close_calls == 1
            assert operations(engine)['state'] == 'closed'
        finally:
            embed.release.set()
            await asyncio.gather(search, close, return_exceptions=True)
            await engine.close()
    asyncio.run(scenario())


def test_capacity_one_nested_paths_and_active_batch_survive_closing(tmp_path: Path) -> None:
    async def scenario() -> None:
        folder = tmp_path / 'docs'
        folder.mkdir()
        for name in ('first', 'second'):
            (folder / f'{name}.txt').write_text(f'The original {name} query document.')
        embed = PausedEmbedding()
        engine, store = core(embed=embed, max_concurrent_ingests=1)
        batch = asyncio.create_task(engine.add(folder, tenant_id='t', collection='c'))
        await asyncio.wait_for(embed.started.wait(), 3)
        close = asyncio.create_task(engine.close())
        try:
            await until(lambda: operations(engine)['state'] == 'closing')
            embed.release.set()
            result = await asyncio.wait_for(batch, 3)
            assert result.succeeded_count == 2 and result.failed_count == 0
            await close
            assert store.close_calls == 1
        finally:
            embed.release.set()
            await asyncio.gather(batch, close, return_exceptions=True)
            await engine.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('source', ['bytes', 'file', 'archive', 'prepare'])
def test_capacity_one_direct_and_nested_ingest_paths(tmp_path: Path, source: str) -> None:
    async def scenario() -> None:
        engine, _ = core(max_concurrent_ingests=1)
        file = tmp_path / 'source.txt'
        file.write_text('An original query document about foxes.')
        archive = tmp_path / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('source.txt', file.read_text())
        try:
            if source == 'bytes':
                result = await engine.add_bytes(file_bytes=file.read_bytes(), filename=file.name,
                                                mime_type='text/plain', tenant_id='t', collection='c')
                assert result.chunk_count > 0
            elif source == 'file':
                assert (await engine.add_file(file, tenant_id='t', collection='c')).chunk_count > 0
            elif source == 'archive':
                assert (await engine.add_archive(archive, tenant_id='t', collection='c')).succeeded_count == 1
            else:
                assert (await engine.prepare_file(file)).chunks
        finally:
            await engine.close()
    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_shared_batch_reservations_block_before_source_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rag_core.ingest.archive as archive_module
    import rag_core.ingest.local as local_module
    plans: list[str] = []
    local_plan = local_module.build_local_ingest_plan
    archive_plan = archive_module.read_zip_archive

    def local(*args: Any, **kwargs: Any) -> Any:
        plans.append('local')
        return local_plan(*args, **kwargs)

    def archive(*args: Any, **kwargs: Any) -> Any:
        plans.append('archive')
        return archive_plan(*args, **kwargs)

    monkeypatch.setattr(local_module, 'build_local_ingest_plan', local)
    monkeypatch.setattr(archive_module, 'read_zip_archive', archive)

    async def scenario() -> None:
        folder = tmp_path / 'docs'
        folder.mkdir()
        for i in range(2):
            (folder / f'{i}.txt').write_text(f'Original document {i} about queries.')
        embed = PausedEmbedding()
        engine, store = core(embed=embed, max_concurrent_ingests=2, max_pending_operations=2)
        active = asyncio.create_task(engine.add(folder, tenant_id='t', collection='c', max_concurrency=2))
        await asyncio.wait_for(embed.started.wait(), 3)
        queued_zip = asyncio.create_task(engine.add_archive(tmp_path / 'never-read.zip', tenant_id='t', collection='c'))
        queued_bytes = asyncio.create_task(engine.add_bytes(file_bytes=b'Original byte document.', filename='b.txt',
                                                           mime_type='text/plain', tenant_id='t', collection='c'))
        close = None
        try:
            await until(lambda: operations(engine)['pending_operations'] == 2)
            assert plans == ['local']
            assert operations(engine)['active_ingest_reservations'] == 2
            with pytest.raises(EngineBusyError):
                await engine.add_file(tmp_path / 'never-read.txt', tenant_id='t', collection='c')
            close = asyncio.create_task(engine.close())
            queued_results = await asyncio.gather(queued_zip, queued_bytes, return_exceptions=True)
            assert all(isinstance(r, EngineClosedError) for r in queued_results)
            assert plans == ['local'] and store.close_calls == 0
            embed.release.set()
            assert (await active).succeeded_count == 2
            await close
            assert operations(engine)['pending_operations'] == operations(engine)['active_operations'] == 0
        finally:
            embed.release.set()
            await asyncio.gather(active, queued_zip, queued_bytes, return_exceptions=True)
            if close:
                await close
            await engine.close()
    asyncio.run(scenario())


def test_search_lane_not_blocked_by_ingest_queue_and_shared_pending_limit() -> None:
    async def scenario() -> None:
        owner = RuntimeOperations(max_concurrent_ingests=1, max_concurrent_searches=1, max_pending_operations=1)
        release = asyncio.Event()
        started = asyncio.Event()
        async def hold() -> None:
            async with owner.operation(ingest_slots=1):
                started.set()
                await release.wait()
        active = asyncio.create_task(hold())
        await started.wait()
        queued = asyncio.create_task(hold())
        try:
            await until(lambda: owner.snapshot()['pending_operations'] == 1)
            async with owner.operation(search_slots=1):
                with pytest.raises(EngineBusyError):
                    async with owner.operation(search_slots=1):
                        pytest.fail('queue overflow admitted')
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert owner.snapshot()['pending_operations'] == 0
        finally:
            release.set()
            await asyncio.gather(active, queued, return_exceptions=True)
    asyncio.run(scenario())


def test_close_joins_repeated_cancellation_and_shared_store_failure() -> None:
    async def scenario() -> None:
        engine, store = core()
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0
        async def failing_close() -> None:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            raise ValueError('controlled close failure')
        store.close = failing_close  # type: ignore[method-assign]
        first = asyncio.create_task(engine.close())
        await started.wait()
        second = asyncio.create_task(engine.close())
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert isinstance(results[1], ValueError)
        with pytest.raises(ValueError, match='controlled close failure'):
            await engine.close()
        assert calls == 1
        with pytest.raises(EngineClosedError):
            await engine.ensure_ready()
        await LocalQdrantStore.close(store)
    asyncio.run(scenario())


def test_cancelled_worker_retains_engine_lease_until_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_core.core as core_module
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def parse() -> None:
        started.set()
        try:
            if not release.wait(3):
                raise TimeoutError('worker release missing')
            raise ValueError('worker failure after cancellation')
        finally:
            finished.set()
    async def blocked_parse(**kwargs: Any) -> Any:
        await run_sync(parse)
        pytest.fail('cancelled operation proceeded past worker')
    monkeypatch.setattr(core_module, 'parse_document_bytes', blocked_parse)
    async def scenario() -> None:
        engine, store = core()
        task = asyncio.create_task(engine.parse_bytes(file_bytes=b'Original.', filename='a.txt', mime_type='text/plain'))
        await until(started.is_set)
        task.cancel()
        close = asyncio.create_task(engine.close())
        try:
            await until(lambda: operations(engine)['state'] == 'closing')
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not close.done() and store.close_calls == 0
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await close
            assert finished.is_set() and store.close_calls == 1
        finally:
            release.set()
            await asyncio.gather(task, close, return_exceptions=True)
            await engine.close()
    asyncio.run(scenario())


def test_sequential_loops_can_each_contend_then_close_on_another_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_core.core as core_module
    engine, store = core(max_concurrent_ingests=1)
    async def contend() -> None:
        release, started = asyncio.Event(), asyncio.Event()
        async def parse(**kwargs: Any) -> ParsedDocument:
            started.set()
            await release.wait()
            return ParsedDocument(filename='a.txt', mime_type='text/plain', markdown='original')
        monkeypatch.setattr(core_module, 'parse_document_bytes', parse)
        def request() -> Any:
            return engine.parse_bytes(file_bytes=b'original', filename='a.txt', mime_type='text/plain')
        first = asyncio.create_task(request())
        await started.wait()
        second = asyncio.create_task(request())
        await until(lambda: operations(engine)['pending_operations'] == 1)
        release.set()
        assert all(item.markdown == 'original' for item in await asyncio.gather(first, second))
    try:
        asyncio.run(contend())
        asyncio.run(contend())
    finally:
        asyncio.run(engine.close())
    assert store.close_calls == 1
    asyncio.run(engine.close())


def test_oversized_reservation_and_invalid_scope_do_not_read_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import rag_core.ingest.local as local_module
    def unexpected_read(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('invalid request reached source discovery')
    monkeypatch.setattr(local_module, 'build_local_ingest_plan', unexpected_read)
    async def scenario() -> None:
        engine, _ = core(max_concurrent_ingests=1)
        try:
            with pytest.raises(ValueError, match='max_concurrent_ingests'):
                await engine.add(tmp_path, tenant_id='t', collection='c', max_concurrency=2)
            with pytest.raises(ValueError):
                await engine.add(tmp_path, tenant_id='', collection='c')
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_direct_byte_paths_enforce_document_cap_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_core.core as core_module
    async def unexpected_parse(**kwargs: Any) -> Any:
        pytest.fail('oversized document reached parsing')
    monkeypatch.setattr(core_module, 'parse_document_bytes', unexpected_parse)
    monkeypatch.setattr(core_module, 'prepare_document_bytes', unexpected_parse)
    async def scenario() -> None:
        config: Config = make_test_config(embedding_dimensions=4)
        config = replace(config, ingest=replace(config.ingest, max_document_bytes=4))
        engine = Engine(config, embedding_provider=FakeEmbeddingProvider())
        try:
            for method in (engine.parse_bytes, engine.prepare_bytes):
                with pytest.raises(ValueError, match='max_document_bytes'):
                    await method(file_bytes=b'12345', filename='a.txt', mime_type='text/plain')
            with pytest.raises(ValueError, match='max_document_bytes'):
                await engine.add_bytes(file_bytes=b'12345', filename='a.txt', mime_type='text/plain',
                                       tenant_id='t', collection='c')
        finally:
            await engine.close()
    asyncio.run(scenario())
