"""The manifest follows committed storage; journal failure never destroys data."""

from __future__ import annotations
import asyncio
from dataclasses import replace
from pathlib import Path
import pytest
from rag_core import Config
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider
from rag_core.events import EventBuffer, StageError
from rag_core.manifest.persistence import read_entries


@pytest.mark.parametrize("replacement", [False, True])
def test_manifest_failure_preserves_published_data_and_retry_heals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    import rag_core.ingest.manifest as manifest

    original_write = manifest.write_entry

    def fail(*args, **kwargs):
        raise OSError("private manifest failure sk-test-secret")

    async def run() -> None:
        base = Config.local()
        config = replace(
            base, ingest=replace(base.ingest, manifest_directory=tmp_path / "manifest")
        )
        events = EventBuffer()
        async with Engine(
            config, embedding_provider=DemoEmbeddingProvider(), event_sink=events
        ) as core:

            async def ingest(content):
                return await core.add_bytes(
                    file_bytes=content,
                    filename="bill.txt",
                    mime_type="text/plain",
                    document_key="bill",
                    tenant_id="acme",
                    collection="docs",
                )

            if replacement:
                await ingest(b"Old invoice terms.")
            previous = read_entries(tmp_path / "manifest", "acme", "docs")
            monkeypatch.setattr(manifest, "write_entry", fail)
            with pytest.raises(OSError):
                await ingest(b"New invoice terms.")
            assert read_entries(tmp_path / "manifest", "acme", "docs") == previous
            hits = await core.search(
                query="invoice", tenant_id="acme", collection="docs", rerank=False
            )
            assert [hit.text for hit in hits] == ["New invoice terms."]
            assert "sk-test-secret" not in repr(
                [event for event in events.events if isinstance(event, StageError)]
            )
            monkeypatch.setattr(manifest, "write_entry", original_write)
            retry = await ingest(b"New invoice terms.")
            assert retry.ingest_state == "unchanged"
            entries = read_entries(tmp_path / "manifest", "acme", "docs")
            assert (
                len(entries) == 1 and entries[0].content_sha256 == retry.content_sha256
            )

    asyncio.run(run())


def test_collection_delete_waits_for_ingest_and_clears_manifest(tmp_path: Path) -> None:
    async def run() -> None:
        base = Config.local()
        config = replace(
            base, ingest=replace(base.ingest, manifest_directory=tmp_path / "manifest")
        )
        async with Engine(config, embedding_provider=DemoEmbeddingProvider()) as core:
            staged = asyncio.Event()
            release = asyncio.Event()
            original = core._store.upsert

            async def pause(points):
                await original(points)
                staged.set()
                await release.wait()

            core._store.upsert = pause  # type: ignore[method-assign]
            ingest = asyncio.create_task(
                core.add_bytes(
                    file_bytes=b"Invoice terms.",
                    filename="bill.txt",
                    mime_type="text/plain",
                    document_key="bill",
                    tenant_id="acme",
                    collection="docs",
                )
            )
            await asyncio.wait_for(staged.wait(), 3)
            deletion = asyncio.create_task(
                core.delete_collection(tenant_id="acme", collection="docs")
            )
            await asyncio.sleep(0)
            assert not deletion.done()
            release.set()
            await ingest
            await asyncio.wait_for(deletion, 3)
            assert read_entries(tmp_path / "manifest", "acme", "docs") == []
            assert (
                await core.search(
                    query="invoice", tenant_id="acme", collection="docs", rerank=False
                )
                == []
            )
            assert core._ingest._fence.active_keys() == ()

    asyncio.run(run())


@pytest.mark.parametrize(
    "tenant,collection", [("Acme", "docs"), ("acme", "Help Center")]
)
def test_encoded_manifest_scopes_roundtrip_and_reingest_unchanged(
    tmp_path: Path, tenant: str, collection: str
) -> None:
    async def run() -> None:
        base = Config.local()
        config = replace(
            base, ingest=replace(base.ingest, manifest_directory=tmp_path / "manifest")
        )
        async with Engine(config, embedding_provider=DemoEmbeddingProvider()) as core:

            async def ingest():
                return await core.add_bytes(
                    file_bytes=b"Invoice scope roundtrip.",
                    filename="bill.txt",
                    mime_type="text/plain",
                    document_key="bill",
                    tenant_id=tenant,
                    collection=collection,
                )

            await ingest()
            again = await ingest()
            assert again.ingest_state == "unchanged"
            entries = read_entries(tmp_path / "manifest", tenant, collection)
            assert len(entries) == 1 and (
                entries[0].tenant_id,
                entries[0].collection,
            ) == (tenant, collection)

    asyncio.run(run())


def test_padded_delete_id_uses_same_identity_for_fence_store_and_manifest(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        base = Config.local()
        config = replace(
            base, ingest=replace(base.ingest, manifest_directory=tmp_path / "manifest")
        )
        async with Engine(config, embedding_provider=DemoEmbeddingProvider()) as core:
            await core.add_bytes(
                file_bytes=b"Invoice terms.",
                filename="bill.txt",
                mime_type="text/plain",
                tenant_id="acme",
                collection="docs",
                document_id="bill",
            )
            fence = core._ingest._fence
            async with fence.acquire(
                tenant_id="acme", collection="docs", document_id="bill"
            ):
                deletion = asyncio.create_task(
                    core.delete_document(
                        document_id=" bill ", tenant_id="acme", collection="docs"
                    )
                )
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not deletion.done()
            result = await asyncio.wait_for(deletion, 3)
            assert result.document_id == "bill" and result.manifest_removed is True
            assert read_entries(tmp_path / "manifest", "acme", "docs") == []

    asyncio.run(run())
