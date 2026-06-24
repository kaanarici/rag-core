from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, replace
from threading import Lock

import pytest

import rag_core
from rag_core import Engine
from rag_core.core_models import IngestedDocument
from rag_core.events import (
    EventBuffer,
    FetchCompleted,
    FetchFailed,
    FetchStarted,
)
from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy, validate_fetch_url
from rag_core.fetch_security import safe_remote_document_key
from rag_core.fetching import FetchResponse
from rag_core._engine.core_remote import ingest_remote_url
from rag_core.ingest.urls.models import RemoteUrlSourceItem
from rag_core.ingest.urls.records import safe_remote_ingest_error

from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_test_config,
)

ALLOW_HTTP_PRIVATE_POLICY = FetchSecurityPolicy(
    allowed_schemes=("https", "http"),
    allow_private_addresses=True,
)


@dataclass
class FakeFetchClient:
    response: FetchResponse
    calls: list[str]

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        return self.response


@dataclass
class FailingFetchClient:
    calls: list[str]

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        raise RuntimeError(f"fetch exploded for {url}")


@dataclass
class EchoFetchClient:
    calls: list[str]

    def fetch(self, url: str) -> FetchResponse:
        self.calls.append(url)
        return _fetch_response(
            url=url,
            content_type="text/plain",
            body=b"same remote body",
        )


class SlowFetchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.active = 0
        self.peak_active = 0
        self._lock = Lock()

    def fetch(self, url: str) -> FetchResponse:
        with self._lock:
            self.calls.append(url)
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(0.05)
            return _fetch_response(
                url=url,
                content_type="text/plain",
                body=f"body for {url}".encode(),
            )
        finally:
            with self._lock:
                self.active -= 1


def _make_url_core(
    *,
    url: str = "https://example.com/docs/guide?private=alpha",
    body: bytes = b"remote guide fox query",
    content_type: str = "text/plain",
    source_type: str = "file",
    event_sink: EventBuffer | None = None,
) -> tuple[Engine, RecordingVectorStore, FakeFetchClient]:
    fetch_client = FakeFetchClient(
        response=_fetch_response(url=url, content_type=content_type, body=body),
        calls=[],
    )
    store = RecordingVectorStore()
    core = Engine(
        make_test_config(
            embedding_model="text-embedding-3-small",
            embedding_dimensions=4,
            source_type=source_type,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
        event_sink=event_sink,
    )
    return core, store, fetch_client


def test_fetch_controls_live_under_fetch_security_namespace() -> None:
    assert FetchSecurityPolicy.__name__ == "FetchSecurityPolicy"
    assert FetchLimits.__name__ == "FetchLimits"
    assert {"FetchSecurityPolicy", "FetchLimits"}.isdisjoint(set(rag_core.__all__))


def test_ingest_url_indexes_with_url_source_type_and_redacted_identity() -> None:
    async def _run() -> None:
        core, store, fetch_client = _make_url_core()

        try:
            document = await core.add_url(
                " https://example.com/docs/guide?private=alpha ",
                namespace="team-space",
                collection="corpus-1",
                metadata={"title": "Remote Guide", "source_type": "caller"},
                fetch_client=fetch_client,
            )
        finally:
            await core.close()

        expected_key = _url_key("https://example.com/docs/guide?private=alpha")
        assert fetch_client.calls == [" https://example.com/docs/guide?private=alpha "]
        assert document.document_key == expected_key
        assert document.filename == "guide.txt"
        assert document.processing_version == (
            '{"base_version":"rag_core_processing_v3","source_type":"url"}'
        )
        assert document.metadata["title"] == "Remote Guide"
        assert document.metadata["source_type"] == "url"
        assert (
            document.metadata["source_url"] == "https://example.com/docs/guide?redacted"
        )
        assert "private=alpha" not in repr(document)

        [points] = store.upsert_calls
        payload = points[0].payload
        assert payload["source_type"] == "url"
        assert payload["document_key"] == expected_key
        assert payload["processing_version"] == document.processing_version
        assert payload["document_path"] == "https://example.com/docs/guide?redacted"
        assert payload["title"] == "Remote Guide"
        assert "private=alpha" not in repr(payload)

    asyncio.run(_run())


def test_ingest_url_emits_redacted_fetch_events() -> None:
    async def _run() -> None:
        events = EventBuffer()
        core, _, fetch_client = _make_url_core(event_sink=events)

        try:
            await core.add_url(
                "https://example.com/docs/guide?private=alpha",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
        finally:
            await core.close()

        started = events.by_type("fetch.started")
        assert len(started) == 1
        assert isinstance(started[0], FetchStarted)
        assert started[0].namespace == "team-space"
        assert started[0].collection == "corpus-1"
        assert started[0].redacted_url == "https://example.com/?redacted"

        completed = events.by_type("fetch.completed")
        assert len(completed) == 1
        assert isinstance(completed[0], FetchCompleted)
        assert completed[0].namespace == "team-space"
        assert completed[0].collection == "corpus-1"
        assert completed[0].redacted_url == "https://example.com/docs/guide?redacted"
        assert completed[0].status_code == 200
        assert completed[0].content_type == "text/plain"
        assert completed[0].content_length == len(b"remote guide fox query")
        assert completed[0].byte_count == len(b"remote guide fox query")
        assert (
            completed[0].content_sha256
            == hashlib.sha256(b"remote guide fox query").hexdigest()
        )
        assert completed[0].redirect_count == 0
        assert completed[0].duration_ms >= 0.0

        assert events.by_type("fetch.failed") == []
        assert "private=alpha" not in repr(events.events)

    asyncio.run(_run())


def test_ingest_url_fetch_failure_event_omits_error_message() -> None:
    async def _run() -> None:
        events = EventBuffer()
        core, _, _ = _make_url_core(event_sink=events)
        fetch_client = FailingFetchClient(calls=[])

        try:
            with pytest.raises(RuntimeError, match="fetch exploded"):
                await core.add_url(
                    "https://example.com/docs/guide?private=alpha",
                    namespace="team-space",
                    collection="corpus-1",
                    fetch_client=fetch_client,
                )
        finally:
            await core.close()

        failed = events.by_type("fetch.failed")
        assert len(failed) == 1
        assert isinstance(failed[0], FetchFailed)
        assert failed[0].redacted_url == "https://example.com/?redacted"
        assert failed[0].error_type == "RuntimeError"
        assert failed[0].duration_ms >= 0.0
        assert "private=alpha" not in repr(failed)
        assert "fetch exploded" not in repr(failed)
        assert fetch_client.calls == ["https://example.com/docs/guide?private=alpha"]

    asyncio.run(_run())


def test_ingest_remote_url_offloads_blocking_fetches() -> None:
    fetch_client = SlowFetchClient()
    ingested: list[str] = []

    async def add_bytes(
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
    ) -> IngestedDocument:
        ingested.append(path or "")
        return IngestedDocument(
            document_id=document_id or filename,
            collection=collection,
            namespace=namespace,
            chunk_count=1,
            filename=filename,
            mime_type=mime_type,
        )

    async def scenario() -> None:
        await asyncio.gather(
            *(
                ingest_remote_url(
                    f"https://example.com/docs/{index}",
                    ingest_bytes=add_bytes,
                    namespace="acme",
                    collection="help",
                    fetch_client=fetch_client,
                )
                for index in range(8)
            )
        )

    asyncio.run(scenario())

    assert len(fetch_client.calls) == 8
    assert len(ingested) == 8
    assert fetch_client.peak_active > 1


def test_safe_remote_ingest_error_uses_generic_message_for_query_bearing_urls() -> None:
    item = RemoteUrlSourceItem(
        url="https://example.com/export?id=1&token=secret",
        redacted_url="https://example.com/export?redacted",
        document_key="url:https://example.com/export?redacted",
        query_sha256="hash",
        source_line=1,
        raw_query="id=1&token=secret",
    )

    message = safe_remote_ingest_error(
        RuntimeError("parser failed on token=secret"),
        item,
    )

    assert message == "RuntimeError while ingesting https://example.com/export?redacted"
    assert "token=secret" not in message


def test_safe_remote_ingest_error_sanitizes_non_query_runtime_messages() -> None:
    item = RemoteUrlSourceItem(
        url="https://example.com/docs/final",
        redacted_url="https://example.com/docs/final",
        document_key="url:https://example.com/docs/final",
        source_line=1,
    )

    message = safe_remote_ingest_error(
        RuntimeError("inner failure at /tmp/private-cache with api_key=secret"),
        item,
    )

    assert message == "RuntimeError while ingesting https://example.com/docs/final"
    assert "api_key=secret" not in message
    assert "/tmp/private-cache" not in message


def test_ingest_url_revalidates_custom_fetch_client_response_with_default_policy() -> (
    None
):
    async def _run() -> None:
        events = EventBuffer()
        fetch_client = FakeFetchClient(
            response=_fetch_response(
                url="http://127.0.0.1/docs?private=alpha",
                content_type="text/plain",
                body=b"private docs",
                policy=ALLOW_HTTP_PRIVATE_POLICY,
            ),
            calls=[],
        )
        store = RecordingVectorStore()
        core = Engine(
            make_test_config(
                embedding_model="text-embedding-3-small",
                embedding_dimensions=4,
                source_type="url",
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=events,
        )

        try:
            with pytest.raises(ValueError, match="HTTP requires explicit opt-in"):
                await core.add_url(
                    "https://example.com/docs?private=alpha",
                    namespace="team-space",
                    collection="corpus-1",
                    fetch_client=fetch_client,
                )
        finally:
            await core.close()

        assert fetch_client.calls == ["https://example.com/docs?private=alpha"]
        started = events.by_type("fetch.started")[0]
        failed = events.by_type("fetch.failed")[0]
        assert isinstance(started, FetchStarted)
        assert isinstance(failed, FetchFailed)
        assert started.redacted_url == "https://example.com/?redacted"
        assert failed.redacted_url == "https://example.com/?redacted"
        assert "private=alpha" not in repr(events.events)

    asyncio.run(_run())


def test_ingest_url_rejects_custom_fetch_client_with_explicit_policy() -> None:
    async def _run() -> None:
        fetch_client = FakeFetchClient(
            response=_fetch_response(
                url="http://127.0.0.1/docs?private=alpha",
                content_type="text/plain",
                body=b"private docs",
                policy=ALLOW_HTTP_PRIVATE_POLICY,
            ),
            calls=[],
        )
        core, _, _ = _make_url_core()

        try:
            with pytest.raises(
                ValueError, match="fetch_client cannot be combined with policy"
            ):
                await core.add_url(
                    "http://127.0.0.1/docs?private=alpha",
                    namespace="team-space",
                    collection="corpus-1",
                    fetch_client=fetch_client,
                    fetch_policy=ALLOW_HTTP_PRIVATE_POLICY,
                )
        finally:
            await core.close()

        assert fetch_client.calls == []

    asyncio.run(_run())


def test_ingest_url_can_build_default_fetch_client_with_policy_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        constructed: list[tuple[FetchSecurityPolicy | None, FetchLimits | None]] = []
        fetch_calls: list[str] = []

        class RecordingHttpFetchClient:
            def __init__(
                self,
                *,
                policy: FetchSecurityPolicy | None = None,
                limits: FetchLimits | None = None,
            ) -> None:
                constructed.append((policy, limits))
                self._policy = policy

            def fetch(self, url: str) -> FetchResponse:
                fetch_calls.append(url)
                return _fetch_response(
                    url=url,
                    content_type="text/plain",
                    body=b"private docs",
                    policy=self._policy,
                )

        monkeypatch.setattr(
            "rag_core.ingest.sources.remote.HttpFetchClient", RecordingHttpFetchClient
        )
        policy = ALLOW_HTTP_PRIVATE_POLICY
        limits = FetchLimits(max_bytes=1024, timeout_seconds=2.5, max_redirects=1)
        events = EventBuffer()
        store = RecordingVectorStore()
        core = Engine(
            make_test_config(
                embedding_model="text-embedding-3-small",
                embedding_dimensions=4,
                source_type="url",
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=events,
        )

        try:
            document = await core.add_url(
                "http://localhost:8000/docs?private=alpha",
                namespace="team-space",
                collection="corpus-1",
                fetch_policy=policy,
                fetch_limits=limits,
            )
        finally:
            await core.close()

        assert constructed == [(policy, limits)]
        assert fetch_calls == ["http://localhost:8000/docs?private=alpha"]
        assert document.metadata["source_url"] == "http://localhost:8000/docs?redacted"
        completed = events.by_type("fetch.completed")[0]
        assert isinstance(completed, FetchCompleted)
        assert completed.redacted_url == "http://localhost:8000/docs?redacted"
        assert "private=alpha" not in repr(events.events)

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("fetch_policy", "fetch_limits"),
    [
        (None, FetchLimits(max_bytes=1024)),
        (FetchSecurityPolicy(), FetchLimits(max_bytes=1024)),
    ],
)
def test_ingest_url_rejects_ambiguous_fetch_configuration(
    fetch_policy: FetchSecurityPolicy | None,
    fetch_limits: FetchLimits | None,
) -> None:
    async def _run() -> None:
        core, _, fetch_client = _make_url_core()

        try:
            with pytest.raises(
                ValueError, match="fetch_client cannot be combined with limits"
            ):
                await core.add_url(
                    "https://example.com/docs/guide",
                    namespace="team-space",
                    collection="corpus-1",
                    fetch_client=fetch_client,
                    fetch_policy=fetch_policy,
                    fetch_limits=fetch_limits,
                )
        finally:
            await core.close()

        assert fetch_client.calls == []

    asyncio.run(_run())


def test_ingest_url_uses_url_source_type_when_runtime_default_differs() -> None:
    async def _run() -> None:
        core, _, fetch_client = _make_url_core(
            url="https://example.com/docs/guide",
            body=b"remote",
            source_type="file",
        )
        try:
            document = await core.add_url(
                "https://example.com/docs/guide",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
        finally:
            await core.close()

        assert fetch_client.calls == ["https://example.com/docs/guide"]
        assert document.processing_version == (
            '{"base_version":"rag_core_processing_v3","source_type":"url"}'
        )
        assert document.metadata["source_type"] == "url"

    asyncio.run(_run())


def test_ingest_url_skips_when_same_remote_content_is_already_indexed() -> None:
    async def _run() -> None:
        core, store, fetch_client = _make_url_core()

        try:
            created = await core.add_url(
                "https://example.com/docs/guide?private=alpha",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
            repeated = await core.add_url(
                "https://example.com/docs/guide?private=alpha",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
        finally:
            await core.close()

        assert repeated.ingest_state == "unchanged"
        assert repeated.document_id == created.document_id
        assert repeated.document_key == created.document_key
        assert repeated.processing_version == created.processing_version
        assert len(store.upsert_calls) == 1
        assert fetch_client.calls == [
            "https://example.com/docs/guide?private=alpha",
            "https://example.com/docs/guide?private=alpha",
        ]

    asyncio.run(_run())


def test_ingest_url_distinguishes_redacted_query_identities() -> None:
    async def _run() -> None:
        store = RecordingVectorStore()
        core = Engine(
            make_test_config(
                embedding_model="text-embedding-3-small",
                embedding_dimensions=4,
                source_type="url",
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )
        fetch_client = EchoFetchClient(calls=[])

        try:
            first = await core.add_url(
                "https://example.com/export?id=1",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
            second = await core.add_url(
                "https://example.com/export?id=2",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
        finally:
            await core.close()

        assert first.document_key != second.document_key
        assert first.document_id != second.document_id
        assert second.ingest_state == "created"
        assert len(store.upsert_calls) == 2

    asyncio.run(_run())


def test_ingest_url_reindexes_when_stored_processing_version_drifts() -> None:
    async def _run() -> None:
        core, store, fetch_client = _make_url_core()

        try:
            created = await core.add_url(
                "https://example.com/docs/guide?private=alpha",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
            record_key = ("team-space", "corpus-1", created.document_id)
            store.document_records[record_key] = replace(
                store.document_records[record_key],
                processing_version='{"base_version":"old","source_type":"url"}',
            )
            reindexed = await core.add_url(
                "https://example.com/docs/guide?private=alpha",
                namespace="team-space",
                collection="corpus-1",
                fetch_client=fetch_client,
            )
        finally:
            await core.close()

        assert reindexed.ingest_state == "reindexed"
        assert reindexed.document_id == created.document_id
        assert reindexed.document_key == created.document_key
        assert reindexed.processing_version == created.processing_version
        assert len(store.upsert_calls) == 2

    asyncio.run(_run())


def _fetch_response(
    *,
    url: str,
    content_type: str,
    body: bytes,
    policy: FetchSecurityPolicy | None = None,
) -> FetchResponse:
    validated_url = validate_fetch_url(url, policy=policy)
    return FetchResponse(
        url=validated_url,
        status_code=200,
        content_type=content_type,
        content_length=len(body),
        content_sha256=hashlib.sha256(body).hexdigest(),
        body=body,
        redirect_chain=(validated_url,),
    )


def _url_key(url: str) -> str:
    return safe_remote_document_key(validate_fetch_url(url))
