from __future__ import annotations

from typing import TypedDict

from rag_core.fetch_security import FetchLimits, FetchSecurityPolicy
from rag_core.fetching import FetchClient
from rag_core.ingest.urls.models import RemoteUrlIngestPlan, RemoteUrlIngestRequest


class RemoteUrlCoreIngestKwargs(TypedDict, total=False):
    namespace: str
    collection: str
    metadata: dict[str, str] | None
    force_reindex: bool
    fetch_client: FetchClient
    fetch_policy: FetchSecurityPolicy
    fetch_limits: FetchLimits


def remote_url_core_ingest_kwargs(
    *,
    plan: RemoteUrlIngestPlan,
    request: RemoteUrlIngestRequest,
    fetch_client: FetchClient | None,
) -> RemoteUrlCoreIngestKwargs:
    kwargs: RemoteUrlCoreIngestKwargs = {
        "namespace": plan.namespace,
        "collection": plan.collection,
        "metadata": request.metadata,
        "force_reindex": request.force_reindex,
    }
    if fetch_client is not None:
        kwargs["fetch_client"] = fetch_client
        if request.fetch_policy is not None:
            kwargs["fetch_policy"] = request.fetch_policy
        return kwargs
    if request.fetch_policy is not None:
        kwargs["fetch_policy"] = request.fetch_policy
    if request.fetch_limits is not None:
        kwargs["fetch_limits"] = request.fetch_limits
    return kwargs


__all__ = ["RemoteUrlCoreIngestKwargs", "remote_url_core_ingest_kwargs"]
