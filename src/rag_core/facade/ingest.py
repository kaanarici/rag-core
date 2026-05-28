from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.core_models import DeleteDocumentResult, IngestedDocument, RAGCoreConfig
from rag_core.facade.ingest_batches import (
    ingest_archive_from_facade,
    ingest_files_from_facade,
    ingest_urls_from_facade,
)
from rag_core.facade.ingest_sources import (
    ingest_local_file_source,
    ingest_remote_url_source,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_core.archive_sources import ArchiveLimits
    from rag_core.events.sink import EventSink
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient
    from rag_core.local_ingest_models import LocalIngestResult
    from rag_core.remote_ingest_models import RemoteUrlIngestResult


class _IngestEngine(Protocol):
    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
    ) -> IngestedDocument: ...

    async def delete_document(
        self,
        *,
        document_id: str,
        namespace: str,
        corpus_id: str,
    ) -> DeleteDocumentResult: ...


class _RAGCoreIngestMethods:
    _config: RAGCoreConfig
    _event_sink: "EventSink | None"
    _ingest: _IngestEngine

    async def ingest_bytes(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
    ) -> IngestedDocument:
        return await self._ingest.ingest_bytes(
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            document_key=document_key,
            path=path,
            metadata=metadata,
            force_reindex=force_reindex,
            source_type=source_type,
        )

    async def ingest_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        document_key: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> IngestedDocument:
        return await ingest_local_file_source(
            path,
            ingest_bytes=self.ingest_bytes,
            namespace=namespace,
            corpus_id=corpus_id,
            document_id=document_id,
            document_key=document_key,
            metadata=metadata,
            force_reindex=force_reindex,
        )

    async def ingest_files(
        self,
        path: str | Path,
        *,
        namespace: str,
        corpus_id: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        manifest_dir: str | Path | None = None,
    ) -> "LocalIngestResult":
        return await ingest_files_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            path=path,
            namespace=namespace,
            corpus_id=corpus_id,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            manifest_dir=manifest_dir,
        )

    async def ingest_archive(
        self,
        archive_path: str | Path,
        *,
        namespace: str,
        corpus_id: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        archive_limits: "ArchiveLimits | None" = None,
        manifest_dir: str | Path | None = None,
    ) -> "LocalIngestResult":
        return await ingest_archive_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            archive_path=archive_path,
            namespace=namespace,
            corpus_id=corpus_id,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            archive_limits=archive_limits,
            manifest_dir=manifest_dir,
        )

    async def ingest_url(
        self,
        url: str,
        *,
        namespace: str,
        corpus_id: str,
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        fetch_client: "FetchClient | None" = None,
        fetch_policy: "FetchSecurityPolicy | None" = None,
        fetch_limits: "FetchLimits | None" = None,
    ) -> IngestedDocument:
        return await ingest_remote_url_source(
            url,
            ingest_bytes=self.ingest_bytes,
            namespace=namespace,
            corpus_id=corpus_id,
            event_sink=self._event_sink,
            document_id=document_id,
            metadata=metadata,
            force_reindex=force_reindex,
            fetch_client=fetch_client,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
        )

    async def ingest_urls(
        self,
        url_file: str | Path | None = None,
        *,
        urls: "Sequence[str] | None" = None,
        namespace: str,
        corpus_id: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
        fetch_client: "FetchClient | None" = None,
        fetch_policy: "FetchSecurityPolicy | None" = None,
        fetch_limits: "FetchLimits | None" = None,
        manifest_dir: str | Path | None = None,
    ) -> "RemoteUrlIngestResult":
        return await ingest_urls_from_facade(
            core=self,
            config=self._config,
            event_sink=self._event_sink,
            url_file=url_file,
            urls=urls,
            namespace=namespace,
            corpus_id=corpus_id,
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
        namespace: str,
        corpus_id: str,
    ) -> DeleteDocumentResult:
        return await self._ingest.delete_document(
            document_id=document_id,
            namespace=namespace,
            corpus_id=corpus_id,
        )
