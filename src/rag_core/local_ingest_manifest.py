from __future__ import annotations

from typing import Literal

from rag_core.manifest_persistence import (
    ManifestReconciliation,
    ManifestReconciliationItem,
    ManifestReconciliationStatus,
)
from rag_core.sources import LocalSourceItem


def source_reconciliation_by_key(
    reconciliation: ManifestReconciliation | None,
) -> dict[str, ManifestReconciliationItem]:
    if reconciliation is None:
        return {}
    source_statuses = {"changed", "duplicate", "missing", "unchanged"}
    return {
        item.document_key: item
        for item in reconciliation.items
        if item.status in source_statuses
    }


def manifest_status_for_document(
    document: LocalSourceItem,
    reconciliation_by_key: dict[str, ManifestReconciliationItem],
) -> tuple[ManifestReconciliationStatus | Literal["unknown"], str]:
    if document.source_error:
        return "unknown", "source_read_failed"
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
) -> tuple[ManifestReconciliationStatus | Literal["unknown"], str]:
    item = reconciliation_by_key.get(document_key)
    if item is None:
        return "unknown", "manifest_not_checked"
    if (
        item.status in {"changed", "unchanged"}
        and content_sha256 is not None
        and item.manifest_content_sha256 is not None
    ):
        if content_sha256 == item.manifest_content_sha256:
            return "unchanged", "content_sha256_match"
        return "changed", "content_sha256_changed"
    return item.status, item.reason
