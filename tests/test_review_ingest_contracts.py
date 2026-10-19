"""Independent-review regressions: filesystem, HTTP, journal and sparse identity."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from http.client import HTTPResponse
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import pytest

from rag_core import Config
from rag_core.core import Engine
from rag_core.core_models import CollectionManifestEntry
from rag_core.demo import DemoEmbeddingProvider
from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy, validate_fetch_url
from rag_core.fetching import (
    FetchError,
    FetchResponse,
    _FetchDeadline,
    read_success_fetch_body,
)
from rag_core.ingest.local import LocalIngestRequest, build_local_ingest_plan
from rag_core.manifest.persistence import compact_manifest, read_entries, write_entry
from rag_core.search.providers.qdrant_store import QdrantVectorStore
from rag_core.search.providers.sparse import FastEmbedSparseEmbedder
from rag_core.search.vector_models import SparseVector


class Scope(TypedDict):
    tenant_id: str
    collection: str


SCOPE: Scope = dict(tenant_id="acme", collection="docs")


@pytest.mark.integration
def test_discovery_stat_error_never_prunes_present_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    file = folder / "billing.md"
    file.write_text("Invoices accept ACH.")
    base = Config.local(tmp_path / "store")
    manifest = tmp_path / "manifest"
    config = replace(base, ingest=replace(base.ingest, manifest_directory=manifest))
    original_stat = Path.stat

    def fail_stat(self, *args, **kwargs):
        if self == file:
            raise OSError("injected discovery EIO")
        return original_stat(self, *args, **kwargs)

    async def run() -> None:
        async with Engine(config, embedding_provider=DemoEmbeddingProvider()) as core:
            await core.add(folder, **SCOPE, prune_missing=True)
            with monkeypatch.context() as patch:
                patch.setattr(Path, "stat", fail_stat)
                with pytest.raises(OSError, match="discovery EIO"):
                    await core.add(folder, **SCOPE, prune_missing=True)
            assert len(read_entries(manifest, **SCOPE)) == 1
            hits = await core.search(query="invoices", **SCOPE, rerank=False)
            assert [hit.text for hit in hits] == ["Invoices accept ACH."]

    asyncio.run(run())


def test_discovery_never_reads_package_contents_and_includes_empty_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "__init__.py").write_text("   \n")
    (tmp_path / "empty.md").touch()

    def forbidden(*args, **kwargs):
        raise AssertionError("discovery must not read source content")

    monkeypatch.setattr(Path, "read_text", forbidden)
    plan = build_local_ingest_plan(
        LocalIngestRequest(path=tmp_path, **SCOPE), hash_content=False
    )
    assert {item.path.name for item in plan.documents} == {"__init__.py", "empty.md"}


@pytest.mark.integration
def test_http_early_eof_rejected_at_read_and_injected_response_boundaries() -> None:
    class Socket:
        def makefile(self, *args, **kwargs):
            return io.BytesIO(
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nContent-Type: text/plain\r\n\r\nabc"
            )

    response = HTTPResponse(cast(Any, Socket()))
    response.begin()
    with pytest.raises(FetchError, match="length mismatch"):
        read_success_fetch_body(
            response,
            headers={"content-type": "text/plain", "content-length": "100"},
            limits=FetchLimits(),
            deadline=_FetchDeadline.from_timeout(1),
            redacted_url="https://example.com/docs",
        )

    url = validate_fetch_url("https://example.com/docs", policy=FetchSecurityPolicy())
    body = b"Invoices accept ACH."
    good = FetchResponse(
        url=url,
        status_code=200,
        content_type="text/plain",
        content_length=len(body),
        content_sha256=hashlib.sha256(body).hexdigest(),
        body=body,
        redirect_chain=(url,),
    )

    class Fetch:
        value = good

        def fetch(self, url: str) -> FetchResponse:
            return self.value

    fetch = Fetch()

    async def run() -> None:
        async with Engine(
            Config.local(), embedding_provider=DemoEmbeddingProvider()
        ) as core:
            await core.add_url("https://example.com/docs", **SCOPE, fetch_client=fetch)
            fetch.value = replace(
                good,
                body=b"abc",
                content_length=100,
                content_sha256=hashlib.sha256(b"abc").hexdigest(),
            )
            with pytest.raises(FetchError, match="length mismatch"):
                await core.add_url(
                    "https://example.com/docs", **SCOPE, fetch_client=fetch
                )
            assert [
                hit.text
                for hit in await core.search(query="invoices", **SCOPE, rerank=False)
            ] == [body.decode()]

    asyncio.run(run())


def test_manifest_compaction_does_not_reassign_historical_keys(tmp_path: Path) -> None:
    def entry(document: str, key: str) -> CollectionManifestEntry:
        return CollectionManifestEntry(
            content_sha256="hash",
            document_id=document,
            document_key=key,
            filename="a.txt",
            mime_type="text/plain",
            chunk_count=1,
            **SCOPE,
        )

    for item in (entry("A", "old"), entry("A", "new"), entry("B", "old")):
        write_entry(tmp_path, item)
    expected = {
        (entry.document_id, entry.document_key)
        for entry in read_entries(tmp_path, **SCOPE)
    }
    compact_manifest(tmp_path, **SCOPE)
    assert (
        {
            (entry.document_id, entry.document_key)
            for entry in read_entries(tmp_path, **SCOPE)
        }
        == expected
        == {("A", "new"), ("B", "old")}
    )


@pytest.mark.integration
def test_sparse_vocabulary_change_reindexes_instead_of_skipping() -> None:
    class Sparse:
        def __init__(self, index: int) -> None:
            self.index = index
            self.model_name = f"vocabulary-{index}"

        def embed_texts(self, texts: list[str]) -> list[SparseVector]:
            return [SparseVector(indices=[self.index], values=[1.0]) for _ in texts]

        def embed_query(self, text: str) -> SparseVector:
            return self.embed_texts([text])[0]

    async def run() -> None:
        dense = DemoEmbeddingProvider()
        store = QdrantVectorStore(
            url=None,
            api_key=None,
            location=":memory:",
            collection_name="sparse",
            dense_dimensions=dense.dimensions,
            sparse_enabled=True,
        )

        async def add(core: Engine):
            return await core.add_bytes(
                file_bytes=b"Invoices accept ACH.",
                filename="billing.txt",
                mime_type="text/plain",
                **SCOPE,
            )

        try:
            first = Engine(
                Config.local(hybrid=True),
                embedding_provider=dense,
                sparse_embedder=Sparse(1),
                vector_store=store,
            )
            await add(first)
            second = Engine(
                Config.local(hybrid=True),
                embedding_provider=dense,
                sparse_embedder=Sparse(2),
                vector_store=store,
            )
            assert (await add(second)).ingest_state == "reindexed"
            assert await second.search(query="invoices", mode="lexical", **SCOPE)
            assert (await add(second)).ingest_state == "unchanged"
        finally:
            await store.close()

    asyncio.run(run())


def test_bm25_queries_use_sdk_query_encoding() -> None:
    calls: list[str] = []

    class SDK:
        def embed(self, texts):
            calls.append("document")
            return [SimpleNamespace(indices=[1], values=[1.9]) for _ in texts]

        def query_embed(self, texts):
            calls.append("query")
            return [SimpleNamespace(indices=[1], values=[1.0]) for _ in texts]

    sparse = FastEmbedSparseEmbedder()
    sparse._bm25_model = SDK()
    assert sparse.embed_texts(["invoice"])[0].values == [1.9]
    assert sparse.embed_query("invoice invoice").values == [1.0]
    assert calls == ["document", "query"]
