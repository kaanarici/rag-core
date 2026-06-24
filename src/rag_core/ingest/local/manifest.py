from __future__ import annotations

from rag_core.manifest.persistence import (
    ManifestReconciliation,
    ManifestReconciliationItem,
)
from rag_core.manifest.reconciliation.reasons import (
    MANIFEST_REASON_CONTENT_SHA256_CHANGED,
    MANIFEST_REASON_CONTENT_SHA256_MATCH,
    MANIFEST_REASON_NOT_CHECKED,
    MANIFEST_REASON_SOURCE_READ_FAILED,
)
from rag_core.manifest.reconciliation.statuses import (
    LOCAL_MANIFEST_SOURCE_STATUSES,
    LocalManifestStatus,
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_UNKNOWN,
    MANIFEST_STATUS_UNCHANGED,
)
from rag_core.ingest.sources.local import LocalSourceItem


def source_reconciliation_by_key(
    reconciliation: ManifestReconciliation | None,
) -> dict[str, ManifestReconciliationItem]:
    if reconciliation is None:
        return {}
    return {
        item.document_key: item
        for item in reconciliation.items
        if item.status in LOCAL_MANIFEST_SOURCE_STATUSES
    }


def manifest_status_for_document(
    document: LocalSourceItem,
    reconciliation_by_key: dict[str, ManifestReconciliationItem],
) -> tuple[LocalManifestStatus, str]:
    if document.source_error:
        return MANIFEST_STATUS_UNKNOWN, MANIFEST_REASON_SOURCE_READ_FAILED
    return manifest_status_for_content(
        document_key=document.document_key,
        content_sha256=document.content_sha256,
        reconciliation_by_key=reconciliation_by_key,
    )


def manifest_status_for_content(
    *,
    document_key: str,
    content_sha256: str | None,
    reconciliation_by_key: dict[str, ManifestReconciliationItem],
) -> tuple[LocalManifestStatus, str]:
    item = reconciliation_by_key.get(document_key)
    if item is None:
        return MANIFEST_STATUS_UNKNOWN, MANIFEST_REASON_NOT_CHECKED
    if (
        item.status in (MANIFEST_STATUS_CHANGED, MANIFEST_STATUS_UNCHANGED)
        and content_sha256 is not None
        and item.manifest_content_sha256 is not None
    ):
        if content_sha256 == item.manifest_content_sha256:
            return MANIFEST_STATUS_UNCHANGED, MANIFEST_REASON_CONTENT_SHA256_MATCH
        return MANIFEST_STATUS_CHANGED, MANIFEST_REASON_CONTENT_SHA256_CHANGED
    return item.status, item.reason
