from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_core.cli_provider_errors import (
    ProviderCliError,
    is_provider_bootstrap_error,
    provider_runtime_message,
)
from rag_core.ingest_batch_lifecycle import IngestBatchLifecycle
from rag_core.manifest_persistence import (
    ManifestReconciliation,
    read_entries,
    reconcile_entries,
    validate_manifest_scope,
)
from rag_core.remote_ingest_models import (
    RemoteUrlIngestFailure,
    RemoteUrlIngestPlan,
    RemoteUrlIngestRequest,
    RemoteUrlIngestSuccess,
)
from rag_core.remote_ingest_results import RemoteUrlIngestResult
from rag_core.remote_ingest_runner import (
    RemoteUrlIngestAborted,
    RemoteUrlIngestCore,
    RemoteUrlIngestCoreFactory,
    ingest_remote_urls as _ingest_remote_urls,
)
from rag_core.remote_ingest_records import remote_ingest_error_type
from rag_core.remote_ingest_sources import (
    remote_url_source_items,
    validate_unique_url_keys,
)

if TYPE_CHECKING:
    from rag_core.events.sink import EventSink
    from rag_core.fetching import FetchClient


RemoteUrlIngestRecord = RemoteUrlIngestSuccess | RemoteUrlIngestFailure

REMOTE_INGEST_MAX_CONCURRENCY_CAP = 64


def validate_remote_fetch_configuration(
    request: RemoteUrlIngestRequest,
    fetch_client: "FetchClient | None",
) -> None:
    if fetch_client is None:
        return
    if request.fetch_policy is not None:
        raise ValueError("fetch_client cannot be combined with request fetch_policy")
    if request.fetch_limits is not None:
        raise ValueError("fetch_client cannot be combined with request fetch_limits")


def build_remote_url_ingest_plan(
    request: RemoteUrlIngestRequest,
) -> RemoteUrlIngestPlan:
    validate_manifest_scope(request.namespace, request.corpus_id)
    if not (1 <= request.max_concurrency <= REMOTE_INGEST_MAX_CONCURRENCY_CAP):
        raise ValueError(
            f"max_concurrency must be between 1 and {REMOTE_INGEST_MAX_CONCURRENCY_CAP}"
            f" (got {request.max_concurrency}); pass --max-concurrency with a value in that range"
        )
    urls, url_source, url_file = remote_url_source_items(request)
    if not urls:
        if url_file is not None:
            raise ValueError(f"no URLs found in {url_file!r}")
        raise ValueError("no URLs found in inline URL list")
    validate_unique_url_keys(urls)
    return RemoteUrlIngestPlan(
        url_source=url_source,
        url_file=url_file,
        namespace=request.namespace,
        corpus_id=request.corpus_id,
        urls=urls,
    )


def reconcile_remote_url_ingest_plan(
    plan: RemoteUrlIngestPlan,
    *,
    manifest_dir: Path,
) -> ManifestReconciliation:
    entries = read_entries(
        manifest_dir,
        namespace=plan.namespace,
        corpus_id=plan.corpus_id,
    )
    return reconcile_entries(entries, plan.manifest_sources)


async def run_remote_url_ingest(
    request: RemoteUrlIngestRequest,
    *,
    core_factory: RemoteUrlIngestCoreFactory,
    event_sink: "EventSink | None" = None,
    fetch_client: "FetchClient | None" = None,
    manifest_dir: Path | None = None,
) -> RemoteUrlIngestResult:
    return await _run_remote_url_ingest(
        request,
        core_factory=core_factory,
        close_core=True,
        event_sink=event_sink,
        fetch_client=fetch_client,
        manifest_dir=manifest_dir,
    )


async def run_remote_url_ingest_with_core(
    request: RemoteUrlIngestRequest,
    *,
    core: RemoteUrlIngestCore,
    event_sink: "EventSink | None" = None,
    fetch_client: "FetchClient | None" = None,
    manifest_dir: Path | None = None,
) -> RemoteUrlIngestResult:
    return await _run_remote_url_ingest(
        request,
        core_factory=lambda: core,
        close_core=False,
        event_sink=event_sink,
        fetch_client=fetch_client,
        manifest_dir=manifest_dir,
    )


async def _run_remote_url_ingest(
    request: RemoteUrlIngestRequest,
    *,
    core_factory: RemoteUrlIngestCoreFactory,
    close_core: bool,
    event_sink: "EventSink | None" = None,
    fetch_client: "FetchClient | None" = None,
    manifest_dir: Path | None = None,
) -> RemoteUrlIngestResult:
    validate_remote_fetch_configuration(request, fetch_client)
    plan = build_remote_url_ingest_plan(request)
    reconciliation = (
        reconcile_remote_url_ingest_plan(plan, manifest_dir=manifest_dir)
        if manifest_dir is not None
        else None
    )
    lifecycle = IngestBatchLifecycle[RemoteUrlIngestRecord](
        event_sink=event_sink,
        namespace=plan.namespace,
        corpus_id=plan.corpus_id,
        planned_count=plan.url_count,
        is_success=lambda record: isinstance(record, RemoteUrlIngestSuccess),
        error_type=remote_ingest_error_type,
    )
    lifecycle.started()
    core: RemoteUrlIngestCore | None = None
    records: list[RemoteUrlIngestSuccess | RemoteUrlIngestFailure] = []
    batch_failed = False
    ready = False
    try:
        core = core_factory()
        await core.ensure_ready()
        ready = True
        records = await _ingest_remote_urls(
            core,
            plan,
            request,
            lifecycle=lifecycle,
            fetch_client=fetch_client,
            reconciliation=reconciliation,
        )
        return RemoteUrlIngestResult(
            namespace=plan.namespace,
            corpus_id=plan.corpus_id,
            records=tuple(records),
        )
    except RemoteUrlIngestAborted as exc:
        records = list(exc.records)
        cause = exc.cause
        batch_failed = True
        if ready and is_provider_bootstrap_error(cause):
            cli_error = ProviderCliError(
                provider_runtime_message(cause, action="ingest")
            )
            lifecycle.failed(error=cli_error, records=records)
            raise cli_error from cause
        lifecycle.failed(error=cause, records=records)
        raise cause from exc
    except Exception as exc:
        batch_failed = True
        if ready and is_provider_bootstrap_error(exc):
            cli_error = ProviderCliError(provider_runtime_message(exc, action="ingest"))
            lifecycle.failed(error=cli_error, records=records)
            raise cli_error from exc
        lifecycle.failed(error=exc, records=records)
        raise
    finally:
        if not batch_failed:
            lifecycle.completed()
        if close_core and core is not None:
            await core.close()


__all__ = [
    "RemoteUrlIngestFailure",
    "RemoteUrlIngestPlan",
    "RemoteUrlIngestRequest",
    "RemoteUrlIngestResult",
    "RemoteUrlIngestSuccess",
    "build_remote_url_ingest_plan",
    "reconcile_remote_url_ingest_plan",
    "run_remote_url_ingest",
    "run_remote_url_ingest_with_core",
]
