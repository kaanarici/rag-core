from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

from rag_core.provider_errors import is_provider_bootstrap_error
from rag_core.core_models import IngestedDocument
from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
from rag_core.ingest.lifecycle import (
    IngestBatchLifecycle,
    IngestBatchProgressPayload,
)
from rag_core.ingest.progress import (
    INGEST_PROGRESS_FAILED,
    INGEST_PROGRESS_SUCCEEDED,
    IngestProgressStatus,
)
from rag_core.manifest.persistence import ManifestReconciliation
from rag_core.manifest.reconciliation.statuses import MANIFEST_STATUS_UNKNOWN
from rag_core.ingest.urls.models import (
    RemoteUrlIngestFailure,
    RemoteUrlIngestPlan,
    RemoteUrlIngestRequest,
    RemoteUrlIngestSuccess,
    RemoteUrlSourceItem,
)
from rag_core.ingest.urls.manifest import (
    remote_manifest_status_for_content,
    remote_source_reconciliation_by_key,
)
from rag_core.ingest.urls.records import (
    remote_ingest_error_type,
    remote_ingest_success_record,
    safe_remote_ingest_error,
)
from rag_core.ingest.urls.call import remote_url_core_ingest_kwargs

if TYPE_CHECKING:
    from rag_core.fetching import FetchClient


class RemoteUrlIngestCore(Protocol):
    async def ensure_ready(self) -> None: ...

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
    ) -> IngestedDocument: ...

    async def close(self) -> None: ...


RemoteUrlIngestCoreFactory = Callable[[], RemoteUrlIngestCore]


@dataclass(frozen=True)
class RemoteUrlIngestAborted(Exception):
    cause: Exception
    records: tuple[RemoteUrlIngestSuccess | RemoteUrlIngestFailure, ...]


async def ingest_remote_urls(
    core: RemoteUrlIngestCore,
    plan: RemoteUrlIngestPlan,
    request: RemoteUrlIngestRequest,
    *,
    lifecycle: IngestBatchLifecycle[RemoteUrlIngestSuccess | RemoteUrlIngestFailure],
    fetch_client: FetchClient | None,
    reconciliation: ManifestReconciliation | None = None,
) -> list[RemoteUrlIngestSuccess | RemoteUrlIngestFailure]:
    if not plan.urls:
        return []
    concurrency = min(request.max_concurrency, plan.url_count)
    url_iter = iter(enumerate(plan.urls))
    url_iter_lock = asyncio.Lock()
    reconciliation_by_key = remote_source_reconciliation_by_key(reconciliation)

    async def add_url(position: int, item: RemoteUrlSourceItem) -> None:
        manifest_status, manifest_reason = remote_manifest_status_for_content(
            document_key=item.document_key,
            content_sha256=None,
            reconciliation_by_key=reconciliation_by_key,
        )
        try:
            ingested = await _ingest_remote_url(
                core=core,
                item=item,
                plan=plan,
                request=request,
                fetch_client=fetch_client,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch running.
            if is_provider_bootstrap_error(exc):
                partial = await lifecycle.records_snapshot()
                raise RemoteUrlIngestAborted(exc, partial) from exc
            record: RemoteUrlIngestSuccess | RemoteUrlIngestFailure = (
                RemoteUrlIngestFailure(
                    requested_url=item.redacted_url,
                    document_key=item.document_key,
                    error=safe_remote_ingest_error(exc, item),
                    manifest_status=manifest_status,
                    manifest_reason=manifest_reason,
                )
            )
            status: IngestProgressStatus = INGEST_PROGRESS_FAILED
            content_sha256 = ""
            ingest_state = ""
            document_key = item.document_key
            progress_error = remote_ingest_error_type(exc)
        else:
            document_key = ingested.document_key or item.document_key
            manifest_status, manifest_reason = remote_manifest_status_for_content(
                document_key=document_key,
                content_sha256=ingested.content_sha256,
                reconciliation_by_key=reconciliation_by_key,
            )
            if (
                manifest_status == MANIFEST_STATUS_UNKNOWN
                and document_key != item.document_key
            ):
                manifest_status, manifest_reason = remote_manifest_status_for_content(
                    document_key=item.document_key,
                    content_sha256=ingested.content_sha256,
                    reconciliation_by_key=reconciliation_by_key,
                )
            record = remote_ingest_success_record(
                item=item,
                ingested=ingested,
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
            )
            status = INGEST_PROGRESS_SUCCEEDED
            content_sha256 = ingested.content_sha256 or ""
            ingest_state = ingested.ingest_state
            progress_error = ""

        await lifecycle.record(
            position=position,
            record=record,
            progress=IngestBatchProgressPayload(
                filename=item.redacted_url,
                document_key=document_key,
                content_sha256=content_sha256,
                manifest_status=manifest_status,
                manifest_reason=manifest_reason,
                status=status,
                ingest_state=ingest_state,
                error=progress_error,
            ),
        )

    async def worker() -> None:
        while True:
            async with url_iter_lock:
                try:
                    position, item = next(url_iter)
                except StopIteration:
                    return
            await add_url(position, item)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return list(lifecycle.records)


async def _ingest_remote_url(
    *,
    core: RemoteUrlIngestCore,
    item: RemoteUrlSourceItem,
    plan: RemoteUrlIngestPlan,
    request: RemoteUrlIngestRequest,
    fetch_client: FetchClient | None,
) -> IngestedDocument:
    return await core.add_url(
        item.url,
        **remote_url_core_ingest_kwargs(
            plan=plan,
            request=request,
            fetch_client=fetch_client,
        ),
    )


__all__ = [
    "RemoteUrlIngestCore",
    "RemoteUrlIngestCoreFactory",
    "ingest_remote_urls",
]
