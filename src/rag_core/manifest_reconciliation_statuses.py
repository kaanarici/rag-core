from __future__ import annotations

from typing import Final, Literal

ManifestReconciliationStatus = Literal[
    "changed",
    "duplicate",
    "missing",
    "orphaned",
    "unchanged",
]
ManifestUnknownStatus = Literal["unknown"]
ManifestUnknownUntilFetchStatus = Literal["unknown_until_fetch"]
LocalManifestStatus = ManifestReconciliationStatus | ManifestUnknownStatus
RemoteManifestStatus = (
    ManifestReconciliationStatus
    | ManifestUnknownStatus
    | ManifestUnknownUntilFetchStatus
)

MANIFEST_STATUS_UNKNOWN: Final[ManifestUnknownStatus] = "unknown"
MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH: Final[ManifestUnknownUntilFetchStatus] = (
    "unknown_until_fetch"
)
MANIFEST_STATUS_CHANGED: Final[ManifestReconciliationStatus] = "changed"
MANIFEST_STATUS_DUPLICATE: Final[ManifestReconciliationStatus] = "duplicate"
MANIFEST_STATUS_MISSING: Final[ManifestReconciliationStatus] = "missing"
MANIFEST_STATUS_ORPHANED: Final[ManifestReconciliationStatus] = "orphaned"
MANIFEST_STATUS_UNCHANGED: Final[ManifestReconciliationStatus] = "unchanged"
MANIFEST_RECONCILIATION_STATUSES: Final[tuple[ManifestReconciliationStatus, ...]] = (
    MANIFEST_STATUS_UNCHANGED,
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_ORPHANED,
    MANIFEST_STATUS_DUPLICATE,
)
MANIFEST_NEEDS_REINDEX_STATUSES: Final[tuple[ManifestReconciliationStatus, ...]] = (
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_CHANGED,
)
LOCAL_MANIFEST_SOURCE_STATUSES: Final[tuple[ManifestReconciliationStatus, ...]] = (
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_DUPLICATE,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_UNCHANGED,
)
