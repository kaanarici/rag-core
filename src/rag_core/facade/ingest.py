from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.core_models import Config, DeleteDocumentResult, IngestedDocument
from rag_core.events.types import AuditContext
from rag_core.facade.ingest_batches import (
    ingest_archive_from_facade,
    ingest_files_from_facade,
    ingest_urls_from_facade,
)
from rag_core.facade.ingest_sources import (
    ingest_local_file_source,
    ingest_remote_url_source,
)
from rag_core.scope import normalize_collection, normalize_namespace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_core.ingest.sources.archive import ArchiveLimits
    from rag_core.events.sink import EventSink
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient
    from rag_core.ingest.local.models import LocalIngestResult
    from rag_core.ingest.urls.models import RemoteUrlIngestResult


class _IngestEngine(Protocol):
    async def ingest_bytes(
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
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument: ...

    async def delete_document(
        self,
        *,
        document_id: str,
        namespace: str,
        collection: str,
    ) -> DeleteDocumentResult: ...

    async def delete_collection(
        self,
        *,
        namespace: str,
        collection: str,
    ) -> None: ...

    async def delete_namespace(
        self,
        *,
        namespace: str,
    ) -> None: ...


class _EngineIngestMethods:
    _config: Config
    _event_sink: "EventSink | None"
    _ingest: _IngestEngine

    async def add_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        collection: str | None = None,
        namespace: str | None = None,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument:
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        return await self._ingest.ingest_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            namespace=resolved_namespace,
            collection=resolved_collection,
            document_id=document_id,
            document_key=document_key,
            path=path,
            metadata=metadata,
            force_reindex=force_reindex,
            source_type=source_type,
            audit_context=audit_context,
            ingest_id=ingest_id,
        )

    async def add_file(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        audit_context: AuditContext | None = None,
        ingest_id: str | None = None,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        return await ingest_local_file_source(
            path,
            ingest_bytes=self.add_bytes,
            namespace=resolved_namespace,
            collection=resolved_collection,
            document_id=document_id,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
            audit_context=audit_context,
            ingest_id=ingest_id,
            pre_read_bytes=pre_read_bytes,
        )

    async def add(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        manifest_dir: str | Path | None = None,
    ) -> "LocalIngestResult":
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        return await ingest_files_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            path=path,
            namespace=resolved_namespace,
            collection=resolved_collection,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            manifest_dir=manifest_dir,
        )

    async def add_archive(
        self,
        archive_path: str | Path,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        archive_limits: "ArchiveLimits | None" = None,
        manifest_dir: str | Path | None = None,
    ) -> "LocalIngestResult":
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        return await ingest_archive_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            archive_path=archive_path,
            namespace=resolved_namespace,
            collection=resolved_collection,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            archive_limits=archive_limits,
            manifest_dir=manifest_dir,
        )

    async def add_url(
        self,
        url: str,
        *,
        collection: str | None = None,
        namespace: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        fetch_client: "FetchClient | None" = None,
        fetch_policy: "FetchSecurityPolicy | None" = None,
        fetch_limits: "FetchLimits | None" = None,
    ) -> IngestedDocument:
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        return await ingest_remote_url_source(
            url,
            ingest_bytes=self.add_bytes,
            namespace=resolved_namespace,
            collection=resolved_collection,
            event_sink=self._event_sink,
            document_id=document_id,
            metadata=metadata,
            force_reindex=force_reindex,
            fetch_client=fetch_client,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
        )

    async def add_urls(
        self,
        url_file: str | Path | None = None,
        *,
        urls: "Sequence[str] | None" = None,
        collection: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        fetch_client: "FetchClient | None" = None,
        fetch_policy: "FetchSecurityPolicy | None" = None,
        fetch_limits: "FetchLimits | None" = None,
        manifest_dir: str | Path | None = None,
    ) -> "RemoteUrlIngestResult":
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        return await ingest_urls_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            url_file=url_file,
            urls=urls,
            namespace=resolved_namespace,
            collection=resolved_collection,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            fetch_client=fetch_client,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
            manifest_dir=manifest_dir,
        )

    async def delete_document(
        self,
        *,
        document_id: str,
        collection: str | None = None,
        namespace: str | None = None,
    ) -> DeleteDocumentResult:
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        return await self._ingest.delete_document(
            document_id=document_id,
            namespace=resolved_namespace,
            collection=resolved_collection,
        )

    async def delete_collection(
        self,
        *,
        collection: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """Explicit collection-wide delete (every document in the collection).

        Separate from ``delete_document`` so callers cannot reach a
        collection-wide delete by accident (e.g. by passing an empty
        ``document_id``). The ``DeleteFilter`` seam rejects the silent
        path; this is the deliberate one.
        """
        resolved_namespace = normalize_namespace(namespace)
        resolved_collection = normalize_collection(collection)
        await self._ingest.delete_collection(
            namespace=resolved_namespace,
            collection=resolved_collection,
        )

    async def delete_namespace(
        self,
        *,
        namespace: str,
    ) -> None:
        """Explicit namespace-wide delete (every document in the namespace).

        Reserved for tenant offboarding / "right to forget" workflows; rare
        and irreversible. Kept separate from the collection path so it has
        its own contract surface and its own audit hook.
        """
        await self._ingest.delete_namespace(namespace=namespace)
