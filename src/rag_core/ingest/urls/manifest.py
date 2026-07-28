from __future__ import annotations

from typing import TYPE_CHECKING

from rag_core.manifest.persistence import (
    ManifestReconciliation,
    ManifestReconciliationItem,
)
from rag_core.manifest.reconciliation.reasons import (
    MANIFEST_REASON_CANONICAL_URL_UNKNOWN_UNTIL_FETCH,
    MANIFEST_REASON_CONTENT_SHA256_CHANGED,
    MANIFEST_REASON_CONTENT_SHA256_MATCH,
    MANIFEST_REASON_NOT_CHECKED,
)
from rag_core.manifest.reconciliation.statuses import (
    MANIFEST_RECONCILIATION_STATUSES,
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_DUPLICATE,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_UNKNOWN,
    MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH,
    MANIFEST_STATUS_UNCHANGED,
    RemoteManifestStatus,
)

if TYPE_CHECKING:
    from rag_core.ingest.urls.models import RemoteUrlSourceItem


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
    return {
        item.document_key: item
        for item in reconciliation.items
        if item.status in MANIFEST_RECONCILIATION_STATUSES
    }


def remote_manifest_status_for_content(
    *,
    document_key: str,
    content_sha256: str | None,
    reconciliation_by_key: dict[str, ManifestReconciliationItem],
) -> tuple[RemoteManifestStatus, str]:
    item = reconciliation_by_key.get(document_key)
    if item is None:
        return MANIFEST_STATUS_UNKNOWN, MANIFEST_REASON_NOT_CHECKED
    if item.status == MANIFEST_STATUS_MISSING and content_sha256 is None:
        return (
            MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH,
            MANIFEST_REASON_CANONICAL_URL_UNKNOWN_UNTIL_FETCH,
        )
    if item.status == MANIFEST_STATUS_DUPLICATE:
        return MANIFEST_STATUS_DUPLICATE, item.reason
    if content_sha256 is not None and item.manifest_content_sha256 is not None:
        if content_sha256 == item.manifest_content_sha256:
            return MANIFEST_STATUS_UNCHANGED, MANIFEST_REASON_CONTENT_SHA256_MATCH
        return MANIFEST_STATUS_CHANGED, MANIFEST_REASON_CONTENT_SHA256_CHANGED
    return item.status, item.reason
