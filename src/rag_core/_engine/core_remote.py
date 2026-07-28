from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from rag_core.core_models import IngestedDocument
from rag_core.events.emit import emit_event, now_ms
from rag_core.events.types import FetchCompleted, FetchFailed, FetchStarted
from rag_core.fetch_security import FetchSecurityPolicy, redact_fetch_url, validate_fetch_url
from rag_core.config import INGEST_SOURCE_TYPE_URL
from rag_core.fetch_security import safe_remote_event_url
from rag_core.ingest.sources.remote import RemoteUrlSourceReader

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient


class IngestBytes(Protocol):
    async def __call__(
        self,
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
    ) -> IngestedDocument: ...


async def ingest_remote_url(
    url: str,
    *,
    ingest_bytes: IngestBytes,
    namespace: str,
    collection: str,
    event_sink: "EventSink | None" = None,
    document_id: str | None = None,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    fetch_client: "FetchClient | None" = None,
    fetch_policy: "FetchSecurityPolicy | None" = None,
    fetch_limits: "FetchLimits | None" = None,
) -> IngestedDocument:
    reader = RemoteUrlSourceReader(
        fetch_client=fetch_client,
        policy=fetch_policy,
        limits=fetch_limits,
    )
    fetch_started_ms = now_ms()
    redacted_url = _safe_remote_event_url(url, policy=fetch_policy)
    try:
        emit_event(
            event_sink,
            FetchStarted(
                namespace=namespace,
                collection=collection,
                redacted_url=redacted_url,
            ),
        )
        remote_document = await asyncio.to_thread(reader.read, url)
    except Exception as exc:
        emit_event(
            event_sink,
            FetchFailed(
                namespace=namespace,
                collection=collection,
                redacted_url=redacted_url,
                error_type=type(exc).__name__,
                duration_ms=now_ms() - fetch_started_ms,
            ),
        )
        raise

    emit_event(
        event_sink,
        FetchCompleted(
            namespace=namespace,
            collection=collection,
            redacted_url=remote_document.redacted_url,
            status_code=remote_document.status_code,
            content_type=remote_document.mime_type,
            content_length=remote_document.content_length,
            byte_count=remote_document.byte_count,
            content_sha256=remote_document.content_sha256,
            redirect_count=remote_document.redirect_count,
            duration_ms=now_ms() - fetch_started_ms,
        ),
    )
    return await ingest_bytes(
        file_bytes=remote_document.file_bytes,
        filename=remote_document.filename,
        mime_type=remote_document.mime_type,
        namespace=namespace,
        collection=collection,
        document_id=document_id,
        document_key=remote_document.document_key,
        path=remote_document.redacted_url,
        metadata={**dict(metadata or {}), **remote_document.to_source_metadata()},
        force_reindex=force_reindex,
        source_type=INGEST_SOURCE_TYPE_URL,
    )
__all__ = ["ingest_remote_url"]


def _safe_remote_event_url(
    url: str,
    *,
    policy: FetchSecurityPolicy | None,
) -> str:
    try:
        return safe_remote_event_url(validate_fetch_url(url, policy=policy))
    except ValueError:
        return redact_fetch_url(url)
