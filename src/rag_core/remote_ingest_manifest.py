from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.manifest_persistence import (
    ManifestReconciliation,
    ManifestReconciliationItem,
)
from rag_core.remote_ingest_results import RemoteManifestStatus

if TYPE_CHECKING:
    from rag_core.remote_ingest_models import RemoteUrlSourceItem


def remote_url_plan_payload(
    item: "RemoteUrlSourceItem",
    *,
    reconciliation_by_key: dict[str, ManifestReconciliationItem],
    include_private: bool = False,
) -> dict[str, object]:
    manifest_status, manifest_reason = remote_manifest_status_for_content(
        document_key=item.document_key,
        content_sha256=None,
        reconciliation_by_key=reconciliation_by_key,
    )
    payload = item.to_payload(include_private=include_private)
    if reconciliation_by_key:
        payload["manifest_status"] = manifest_status
        payload["manifest_reason"] = manifest_reason
    return payload


def remote_source_reconciliation_by_key(
    reconciliation: ManifestReconciliation | None,
) -> dict[str, ManifestReconciliationItem]:
    if reconciliation is None:
        return {}
    source_statuses = {"changed", "duplicate", "missing", "orphaned", "unchanged"}
    return {
        item.document_key: item
        for item in reconciliation.items
        if item.status in source_statuses
    }


def remote_manifest_status_for_content(
    *,
    document_key: str,
    content_sha256: str | None,
    reconciliation_by_key: dict[str, ManifestReconciliationItem],
) -> tuple[RemoteManifestStatus, str]:
    item = reconciliation_by_key.get(document_key)
    if item is None:
        return "unknown", "manifest_not_checked"
    if item.status == "missing" and content_sha256 is None:
        return "unknown_until_fetch", "canonical_url_unknown_until_fetch"
    if item.status == "duplicate":
        return "duplicate", item.reason
    if content_sha256 is not None and item.manifest_content_sha256 is not None:
        if content_sha256 == item.manifest_content_sha256:
            return "unchanged", "content_sha256_match"
        return "changed", "content_sha256_changed"
    return item.status, item.reason
