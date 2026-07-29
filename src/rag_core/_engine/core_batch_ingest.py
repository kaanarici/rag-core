from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.config.ingest_config import DEFAULT_INGEST_MAX_CONCURRENCY
from rag_core.ingest.local import run_local_ingest_with_core
from rag_core.ingest.local.models import LocalIngestRequest
from rag_core.ingest.urls import (
    run_remote_url_ingest_with_core,
)
from rag_core.ingest.urls.models import RemoteUrlIngestRequest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_core.core_models import Config
    from rag_core.events.sink import EventSink
    from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
    from rag_core.fetching import FetchClient
    from rag_core.ingest.local import LocalIngestCore
    from rag_core.ingest.local.models import LocalIngestResult
    from rag_core.ingest.urls.results import RemoteUrlIngestResult
    from rag_core.ingest.urls.runner import RemoteUrlIngestCore


async def ingest_files_with_core(
    *,
    core: "LocalIngestCore",
    config: "Config",
    event_sink: "EventSink | None",
    path: str | Path,
    namespace: str,
    collection: str,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
    manifest_dir: str | Path | None = None,
) -> "LocalIngestResult":
    return await run_local_ingest_with_core(
        LocalIngestRequest(
            path=path,
            namespace=namespace,
            collection=collection,
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
        ),
        core=core,
        event_sink=event_sink,
        manifest_dir=_manifest_dir(config=config, manifest_dir=manifest_dir),
    )


async def ingest_urls_with_core(
    *,
    core: "RemoteUrlIngestCore",
    config: "Config",
    event_sink: "EventSink | None",
    url_file: str | Path | None = None,
    urls: "Sequence[str] | None" = None,
    namespace: str,
    collection: str,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    max_concurrency: int = DEFAULT_INGEST_MAX_CONCURRENCY,
    fetch_client: "FetchClient | None" = None,
    fetch_policy: "FetchSecurityPolicy | None" = None,
    fetch_limits: "FetchLimits | None" = None,
    manifest_dir: str | Path | None = None,
) -> "RemoteUrlIngestResult":
    return await run_remote_url_ingest_with_core(
        RemoteUrlIngestRequest(
            namespace=namespace,
            collection=collection,
            url_file=url_file,
            urls=tuple(urls or ()),
            metadata=metadata,
            force_reindex=force_reindex,
            max_concurrency=max_concurrency,
            fetch_policy=fetch_policy,
            fetch_limits=fetch_limits,
        ),
        core=core,
        event_sink=event_sink,
        fetch_client=fetch_client,
        manifest_dir=_manifest_dir(config=config, manifest_dir=manifest_dir),
    )


def _manifest_dir(
    *,
    config: "Config",
    manifest_dir: str | Path | None,
) -> Path | None:
    if manifest_dir is not None:
        return Path(manifest_dir)
    return config.ingest.manifest_directory
