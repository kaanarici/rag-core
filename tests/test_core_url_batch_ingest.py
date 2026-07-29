from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import rag_core
from rag_core import Engine
from rag_core.events import EventBuffer
from rag_core.ingest.urls.results import RemoteUrlIngestResult
from rag_core.events import IngestBatchCompleted, IngestBatchStarted
from rag_core.fetch_security import validate_fetch_url
from rag_core.fetching import FetchResponse

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)


class _StaticFetchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        body = f"Fetched document for {url.split('?', 1)[0]}".encode()
        return FetchResponse(
            url=validate_fetch_url(url),
            status_code=200,
            content_type="text/plain",
            content_length=len(body),
            content_sha256=hashlib.sha256(body).hexdigest(),
            body=body,
            redirect_chain=(validate_fetch_url(url),),
        )


def test_rag_core_ingest_urls_reuses_remote_batch_ingest_without_closing(
    tmp_path: Path,
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "\n".join(
            [
                "https://example.com/docs/guide?private=alpha",
                "https://example.com/docs/reference",
            ]
        ),
        encoding="utf-8",
    )

    async def scenario() -> tuple[
        RemoteUrlIngestResult,
        RecordingVectorStore,
        EventBuffer,
        _StaticFetchClient,
    ]:
        store = RecordingVectorStore()
        events = EventBuffer()
        fetch_client = _StaticFetchClient()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_url_batch_ingest",
                embedding_dimensions=4,
                source_type="url",
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=events,
        )
        result = await core.add_urls(
            url_file,
            namespace="acme",
            collection="help",
            metadata={"team": "docs"},
            max_concurrency=2,
            fetch_client=fetch_client,
        )
        assert store.close_calls == 0
        await core.close()
        return result, store, events, fetch_client

    result, store, events, fetch_client = asyncio.run(scenario())

    assert isinstance(result, RemoteUrlIngestResult)
    assert result.succeeded_count == 2
    assert result.written_count == 2
    assert result.failed_count == 0
    assert [record.requested_url for record in result.succeeded] == [
        "https://example.com/docs/guide?redacted",
        "https://example.com/docs/reference",
    ]
    assert [record.source_url for record in result.succeeded] == [
        "https://example.com/docs/guide?redacted",
        "https://example.com/docs/reference",
    ]
    assert "private=alpha" not in repr(result)
    assert "private=alpha" not in repr(events.events)
    assert fetch_client.calls == [
        "https://example.com/docs/guide?private=alpha",
        "https://example.com/docs/reference",
    ]
    assert store.close_calls == 1
    assert isinstance(events.events[0], IngestBatchStarted)
    assert isinstance(events.events[-1], IngestBatchCompleted)


def test_rag_core_ingest_urls_accepts_inline_url_sequence() -> None:
    async def scenario() -> tuple[RemoteUrlIngestResult, _StaticFetchClient]:
        fetch_client = _StaticFetchClient()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_inline_url_batch_ingest",
                embedding_dimensions=4,
                source_type="url",
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(),
        )
        try:
            result = await core.add_urls(
                urls=(
                    "https://example.com/docs/guide?private=alpha",
                    "https://example.com/docs/reference",
                ),
                namespace="acme",
                collection="help",
                fetch_client=fetch_client,
            )
        finally:
            await core.close()
        return result, fetch_client

    result, fetch_client = asyncio.run(scenario())

    assert result.succeeded_count == 2
    assert result.failed_count == 0
    assert [record.requested_url for record in result.succeeded] == [
        "https://example.com/docs/guide?redacted",
        "https://example.com/docs/reference",
    ]
    assert fetch_client.calls == [
        "https://example.com/docs/guide?private=alpha",
        "https://example.com/docs/reference",
    ]


def test_remote_url_ingest_result_lives_under_remote_ingest_namespace() -> None:
    assert RemoteUrlIngestResult.__name__ == "RemoteUrlIngestResult"
    assert "RemoteUrlIngestResult" not in rag_core.__all__
