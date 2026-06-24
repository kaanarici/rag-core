from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest

from rag_core import Engine, Config
from rag_core.config import (
    DEFAULT_RERANKER_PROVIDER,
    EmbeddingConfig,
    QdrantConfig,
    RerankerConfig,
)
from rag_core.config.ingest_config import IngestConfig
from rag_core.core_models import IngestedDocument
from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
from rag_core.fetch_security import (
    FetchLimits,
    FetchSecurityPolicy,
    validate_fetch_url,
)
from rag_core.fetching import FetchClient, FetchResponse
from rag_core.ingest.local import run_local_ingest_with_core
from rag_core.ingest.local.models import LocalIngestRequest
from rag_core.ingest.sources.local import document_key as local_document_key
from rag_core.manifest.persistence import read_entries
from rag_core.ingest.urls import run_remote_url_ingest_with_core
from rag_core.ingest.urls.models import RemoteUrlIngestRequest
from rag_core.search.vector_models import SparseVector

pytestmark = [pytest.mark.integration]

_NAMESPACE = "resume"
_COLLECTION = "docs"
_CRASH_AFTER_SUCCESSES = 2


class _FakeOpenAIError(Exception):
    __module__ = "openai"


class _SimulatedCrash(BaseException):
    pass


class _ResumeResult(Protocol):
    @property
    def succeeded_count(self) -> int: ...

    @property
    def skipped_count(self) -> int: ...

    @property
    def written_count(self) -> int: ...


class _CountingDemoEmbeddingProvider(DemoEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.text_batches: list[list[str]] = []
        self.fail_on_text_call: int | None = None

    @property
    def text_count(self) -> int:
        return sum(len(batch) for batch in self.text_batches)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_batches.append(list(texts))
        if self.fail_on_text_call == len(self.text_batches):
            raise _FakeOpenAIError("raw api_key client option OPENAI_API_KEY secret")
        return await super().embed_texts(texts)


class _CountingDemoSparseEmbedder(DemoSparseEmbedder):
    def __init__(self) -> None:
        self.text_batches: list[list[str]] = []

    @property
    def text_count(self) -> int:
        return sum(len(batch) for batch in self.text_batches)

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        self.text_batches.append(list(texts))
        return super().embed_texts(texts)


class _StaticFetchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        body = self.body_for(url)
        validated = validate_fetch_url(url)
        return FetchResponse(
            url=validated,
            status_code=200,
            content_type="text/plain",
            content_length=len(body),
            content_sha256=hashlib.sha256(body).hexdigest(),
            body=body,
            redirect_chain=(validated,),
        )

    @staticmethod
    def body_for(url: str) -> bytes:
        return f"Resume proof document for {url}".encode()


@dataclass
class _CrashAfterLocalCore:
    core: Engine
    crash_after_successes: int
    completed: int = 0

    async def ensure_ready(self) -> None:
        await self.core.ensure_ready()

    async def add_file(
        self,
        file_path: Path,
        *,
        namespace: str,
        collection: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        ingested = await self.core.add_file(
            file_path,
            namespace=namespace,
            collection=collection,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
            pre_read_bytes=pre_read_bytes,
        )
        self.completed += 1
        if self.completed == self.crash_after_successes:
            raise _SimulatedCrash("simulated crash after item completion")
        return ingested

    async def close(self) -> None:
        return None


@dataclass
class _CrashAfterRemoteCore:
    core: Engine
    crash_after_successes: int
    completed: int = 0

    async def ensure_ready(self) -> None:
        await self.core.ensure_ready()

    async def add_url(
        self,
        url: str,
        *,
        namespace: str,
        collection: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        fetch_client: FetchClient | None = None,
        fetch_policy: FetchSecurityPolicy | None = None,
        fetch_limits: FetchLimits | None = None,
    ) -> IngestedDocument:
        ingested = await self.core.add_url(
            url,
            namespace=namespace,
            collection=collection,
            metadata=metadata,
            force_reindex=force_reindex,
            fetch_client=fetch_client,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
        )
        self.completed += 1
        if self.completed == self.crash_after_successes:
            raise _SimulatedCrash("simulated crash after item completion")
        return ingested

    async def close(self) -> None:
        return None


def test_local_batch_provider_abort_rerun_skips_manifested_successes(
    tmp_path: Path,
) -> None:
    async def go() -> None:
        docs = _write_local_docs(tmp_path)
        manifest_dir = tmp_path / "manifest"
        core, dense, sparse = _build_core(manifest_dir)
        async with core:
            dense.fail_on_text_call = _CRASH_AFTER_SUCCESSES + 1
            with pytest.raises(ValueError, match="provider failed during ingest"):
                await core.add(
                    docs,
                    namespace=_NAMESPACE,
                    collection=_COLLECTION,
                    max_concurrency=1,
                    manifest_dir=manifest_dir,
                )
            dense.fail_on_text_call = None
            _assert_manifested_success_count(manifest_dir)

            dense_before = dense.text_count
            sparse_before = sparse.text_count
            result = await core.add(
                docs,
                namespace=_NAMESPACE,
                collection=_COLLECTION,
                max_concurrency=1,
                manifest_dir=manifest_dir,
            )

        _assert_rerun_finished_remaining(
            result,
            dense=dense,
            sparse=sparse,
            dense_before=dense_before,
            sparse_before=sparse_before,
        )
        _assert_local_manifest_complete(manifest_dir, docs)

    asyncio.run(go())


def test_local_batch_crash_rerun_skips_manifested_successes(tmp_path: Path) -> None:
    async def go() -> None:
        docs = _write_local_docs(tmp_path)
        manifest_dir = tmp_path / "manifest"
        core, dense, sparse = _build_core(manifest_dir)
        crashing_core = _CrashAfterLocalCore(
            core=core,
            crash_after_successes=_CRASH_AFTER_SUCCESSES,
        )
        async with core:
            with pytest.raises(_SimulatedCrash):
                await run_local_ingest_with_core(
                    LocalIngestRequest(
                        path=docs,
                        namespace=_NAMESPACE,
                        collection=_COLLECTION,
                        max_concurrency=1,
                    ),
                    core=crashing_core,
                    manifest_dir=manifest_dir,
                )
            _assert_manifested_success_count(manifest_dir)

            dense_before = dense.text_count
            sparse_before = sparse.text_count
            result = await core.add(
                docs,
                namespace=_NAMESPACE,
                collection=_COLLECTION,
                max_concurrency=1,
                manifest_dir=manifest_dir,
            )

        _assert_rerun_finished_remaining(
            result,
            dense=dense,
            sparse=sparse,
            dense_before=dense_before,
            sparse_before=sparse_before,
        )
        _assert_local_manifest_complete(manifest_dir, docs)

    asyncio.run(go())


def test_remote_batch_provider_abort_rerun_skips_manifested_successes(
    tmp_path: Path,
) -> None:
    async def go() -> None:
        urls = _remote_urls()
        manifest_dir = tmp_path / "manifest"
        fetch_client = _StaticFetchClient()
        core, dense, sparse = _build_core(manifest_dir)
        async with core:
            dense.fail_on_text_call = _CRASH_AFTER_SUCCESSES + 1
            with pytest.raises(ValueError, match="provider failed during ingest"):
                await core.add_urls(
                    urls=urls,
                    namespace=_NAMESPACE,
                    collection=_COLLECTION,
                    max_concurrency=1,
                    fetch_client=fetch_client,
                    manifest_dir=manifest_dir,
                )
            dense.fail_on_text_call = None
            _assert_manifested_success_count(manifest_dir)

            dense_before = dense.text_count
            sparse_before = sparse.text_count
            result = await core.add_urls(
                urls=urls,
                namespace=_NAMESPACE,
                collection=_COLLECTION,
                max_concurrency=1,
                fetch_client=fetch_client,
                manifest_dir=manifest_dir,
            )

        _assert_rerun_finished_remaining(
            result,
            dense=dense,
            sparse=sparse,
            dense_before=dense_before,
            sparse_before=sparse_before,
        )
        _assert_remote_manifest_complete(manifest_dir, urls)

    asyncio.run(go())


def test_remote_batch_crash_rerun_skips_manifested_successes(tmp_path: Path) -> None:
    async def go() -> None:
        urls = _remote_urls()
        manifest_dir = tmp_path / "manifest"
        fetch_client = _StaticFetchClient()
        core, dense, sparse = _build_core(manifest_dir)
        crashing_core = _CrashAfterRemoteCore(
            core=core,
            crash_after_successes=_CRASH_AFTER_SUCCESSES,
        )
        async with core:
            with pytest.raises(_SimulatedCrash):
                await run_remote_url_ingest_with_core(
                    RemoteUrlIngestRequest(
                        urls=urls,
                        namespace=_NAMESPACE,
                        collection=_COLLECTION,
                        max_concurrency=1,
                    ),
                    core=crashing_core,
                    fetch_client=fetch_client,
                    manifest_dir=manifest_dir,
                )
            _assert_manifested_success_count(manifest_dir)

            dense_before = dense.text_count
            sparse_before = sparse.text_count
            result = await core.add_urls(
                urls=urls,
                namespace=_NAMESPACE,
                collection=_COLLECTION,
                max_concurrency=1,
                fetch_client=fetch_client,
                manifest_dir=manifest_dir,
            )

        _assert_rerun_finished_remaining(
            result,
            dense=dense,
            sparse=sparse,
            dense_before=dense_before,
            sparse_before=sparse_before,
        )
        _assert_remote_manifest_complete(manifest_dir, urls)

    asyncio.run(go())


def _build_core(
    manifest_dir: Path,
) -> tuple[Engine, _CountingDemoEmbeddingProvider, _CountingDemoSparseEmbedder]:
    dense = _CountingDemoEmbeddingProvider()
    sparse = _CountingDemoSparseEmbedder()
    config = Config(
        qdrant=QdrantConfig(
            location=":memory:",
            store_collection=f"rag_core_resume_{uuid.uuid4().hex}",
            dimension_aware_collection=False,
        ),
        embedding=EmbeddingConfig(
            provider="demo",
            model=dense.model_name,
            dimensions=dense.dimensions,
        ),
        reranker=RerankerConfig(provider=DEFAULT_RERANKER_PROVIDER),
        ingest=IngestConfig(manifest_directory=manifest_dir),
    )
    return (
        Engine(config, embedding_provider=dense, sparse_embedder=sparse),
        dense,
        sparse,
    )


def _write_local_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (docs / name).write_text(
            f"Resume proof local document {name}.", encoding="utf-8"
        )
    return docs


def _remote_urls() -> tuple[str, str, str]:
    return (
        "https://example.com/docs/a",
        "https://example.com/docs/b",
        "https://example.com/docs/c",
    )


def _assert_local_manifest_complete(manifest_dir: Path, docs: Path) -> None:
    entries = read_entries(manifest_dir, _NAMESPACE, _COLLECTION)
    expected = {
        local_document_key(docs, path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(docs.glob("*.md"))
    }
    assert {entry.document_key: entry.content_sha256 for entry in entries} == expected
    assert all(entry.chunk_count >= 1 for entry in entries)


def _assert_manifested_success_count(manifest_dir: Path) -> None:
    assert len(read_entries(manifest_dir, _NAMESPACE, _COLLECTION)) == 2


def _assert_rerun_finished_remaining(
    result: _ResumeResult,
    *,
    dense: _CountingDemoEmbeddingProvider,
    sparse: _CountingDemoSparseEmbedder,
    dense_before: int,
    sparse_before: int,
) -> None:
    assert result.succeeded_count == 3
    assert result.skipped_count == 2
    assert result.written_count == 1
    assert dense.text_count - dense_before == 1
    assert sparse.text_count - sparse_before == 1


def _assert_remote_manifest_complete(
    manifest_dir: Path,
    urls: tuple[str, str, str],
) -> None:
    entries = read_entries(manifest_dir, _NAMESPACE, _COLLECTION)
    assert {entry.metadata["source_url"] for entry in entries} == set(urls)
    expected = {
        entry.document_key: hashlib.sha256(
            _StaticFetchClient.body_for(entry.metadata["source_url"])
        ).hexdigest()
        for entry in entries
    }
    assert len(entries) == len(urls)
    assert {entry.document_key: entry.content_sha256 for entry in entries} == expected
    assert all(entry.chunk_count >= 1 for entry in entries)
