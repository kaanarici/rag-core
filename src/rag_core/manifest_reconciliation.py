from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from rag_core.core_models import CorpusManifestEntry
from rag_core.manifest_reconciliation_matching import build_reconciliation_items
from rag_core.manifest_reconciliation_statuses import (
    MANIFEST_NEEDS_REINDEX_STATUSES,
    MANIFEST_RECONCILIATION_STATUSES,
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_DUPLICATE,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_ORPHANED,
    MANIFEST_STATUS_UNCHANGED,
    ManifestReconciliationStatus,
)


@dataclass(frozen=True)
class ManifestSource:
    document_key: str
    content_sha256: str | None = None


@dataclass(frozen=True)
class ManifestReconciliationItem:
    status: ManifestReconciliationStatus
    document_key: str
    reason: str
    document_id: str | None = None
    manifest_content_sha256: str | None = None
    source_content_sha256: str | None = None


@dataclass(frozen=True)
class ManifestReconciliation:
    items: tuple[ManifestReconciliationItem, ...]

    @property
    def unchanged(self) -> tuple[ManifestReconciliationItem, ...]:
        return self.by_status(MANIFEST_STATUS_UNCHANGED)

    @property
    def changed(self) -> tuple[ManifestReconciliationItem, ...]:
        return self.by_status(MANIFEST_STATUS_CHANGED)

    @property
    def missing(self) -> tuple[ManifestReconciliationItem, ...]:
        return self.by_status(MANIFEST_STATUS_MISSING)

    @property
    def orphaned(self) -> tuple[ManifestReconciliationItem, ...]:
        return self.by_status(MANIFEST_STATUS_ORPHANED)

    @property
    def duplicate(self) -> tuple[ManifestReconciliationItem, ...]:
        return self.by_status(MANIFEST_STATUS_DUPLICATE)

    @property
    def needs_reindex(self) -> tuple[ManifestReconciliationItem, ...]:
        return tuple(
            item for item in self.items if item.status in MANIFEST_NEEDS_REINDEX_STATUSES
        )

    def by_status(
        self, status: ManifestReconciliationStatus
    ) -> tuple[ManifestReconciliationItem, ...]:
        return tuple(item for item in self.items if item.status == status)


def manifest_reconciliation_payload(
    reconciliation: ManifestReconciliation,
    *,
    include_private: bool = False,
) -> dict[str, object]:
    summary: dict[str, int] = {
        f"{status}_count": len(reconciliation.by_status(status))
        for status in MANIFEST_RECONCILIATION_STATUSES
    }
    summary["needs_reindex_count"] = len(reconciliation.needs_reindex)
    return {
        "summary": summary,
        "items": [
            asdict(item) if include_private else _public_reconciliation_item(index, item)
            for index, item in enumerate(reconciliation.items)
        ],
    }


def _public_reconciliation_item(
    index: int,
    item: ManifestReconciliationItem,
) -> dict[str, object]:
    return {
        "item_index": index,
        "status": item.status,
        "reason": item.reason,
        "has_document_id": item.document_id is not None,
        "has_manifest_content_sha256": item.manifest_content_sha256 is not None,
        "has_source_content_sha256": item.source_content_sha256 is not None,
    }


def reconcile_entries(
    entries: Sequence[CorpusManifestEntry],
    sources: Iterable[ManifestSource],
) -> ManifestReconciliation:
    """Compare manifest entries to current source identity and content hashes."""
    items = build_reconciliation_items(
        entries=entries,
        sources=sources,
        item_type=ManifestReconciliationItem,
    )
    return ManifestReconciliation(items=tuple(items))
