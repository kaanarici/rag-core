from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.facade.ingest_sources import resolve_archive_manifest_dir
from rag_core.core_models import RAGCoreConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_core.archive_sources import ArchiveLimits
    from rag_core.core_archive_runner import ArchiveIngestCore
    from rag_core.events.sink import EventSink
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient
    from rag_core.local_ingest_models import LocalIngestResult
    from rag_core.local_ingest_runner import LocalIngestCore
    from rag_core.remote_ingest_models import RemoteUrlIngestResult
    from rag_core.remote_ingest_runner import RemoteUrlIngestCore


async def ingest_files_from_facade(
    *,
    core: object,
    config: RAGCoreConfig,
    event_sink: "EventSink | None",
    path: str | Path,
    namespace: str,
    corpus_id: str,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
    manifest_dir: str | Path | None = None,
) -> "LocalIngestResult":
    from rag_core.core_batch_ingest import ingest_files_with_core

    return await ingest_files_with_core(
        core=cast("LocalIngestCore", core),
        config=config,
        event_sink=event_sink,
        path=path,
        namespace=namespace,
        corpus_id=corpus_id,
        metadata=metadata,
        force_reindex=force_reindex,
        max_concurrency=max_concurrency,
        manifest_dir=manifest_dir,
    )


async def ingest_archive_from_facade(
    *,
    core: object,
    config: RAGCoreConfig,
    event_sink: "EventSink | None",
    archive_path: str | Path,
    namespace: str,
    corpus_id: str,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
    archive_limits: "ArchiveLimits | None" = None,
    manifest_dir: str | Path | None = None,
) -> "LocalIngestResult":
    from rag_core.core_archive_ingest import ingest_zip_archive_with_core

    return await ingest_zip_archive_with_core(
        core=cast("ArchiveIngestCore", core),
        archive_path=archive_path,
        namespace=namespace,
        corpus_id=corpus_id,
        metadata=metadata,
        force_reindex=force_reindex,
        max_concurrency=max_concurrency,
        limits=archive_limits,
        manifest_dir=resolve_archive_manifest_dir(
            config=config,
            manifest_dir=manifest_dir,
        ),
        event_sink=event_sink,
    )


async def ingest_urls_from_facade(
    *,
    core: object,
    config: RAGCoreConfig,
    event_sink: "EventSink | None",
    url_file: str | Path | None = None,
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
    from rag_core.core_batch_ingest import ingest_urls_with_core

    return await ingest_urls_with_core(
        core=cast("RemoteUrlIngestCore", core),
        config=config,
        event_sink=event_sink,
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
