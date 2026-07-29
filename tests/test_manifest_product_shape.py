"""Single-owner invariants for manifest reconciliation statuses and reasons.

These check the contracts that matter -- the canonical literal values, the
status/reason set compositions, and that each status/reason constant has exactly
one owning module -- via runtime values and module-level ownership, so they
survive file merges, renames, and reformatting. (Previously these were asserted
by reading a hand-pinned list of source files and scanning for forbidden literal
substrings, which froze the file layout.)
"""

from __future__ import annotations

from rag_core.manifest.reconciliation.reasons import (
    MANIFEST_REASON_CANONICAL_URL_UNKNOWN_UNTIL_FETCH,
    MANIFEST_REASON_CONTENT_SHA256_CHANGED,
    MANIFEST_REASON_CONTENT_SHA256_MATCH,
    MANIFEST_REASON_DUPLICATE_DOCUMENT_KEY,
    MANIFEST_REASON_ENTRY_WITHOUT_SOURCE,
    MANIFEST_REASON_NOT_CHECKED,
    MANIFEST_REASON_PRESENT_WITHOUT_HASH_CHECK,
    MANIFEST_REASON_SOURCE_NOT_IN_MANIFEST,
    MANIFEST_REASON_SOURCE_READ_FAILED,
)
from rag_core.manifest.reconciliation.statuses import (
    LOCAL_MANIFEST_SOURCE_STATUSES,
    MANIFEST_NEEDS_REINDEX_STATUSES,
    MANIFEST_RECONCILIATION_STATUSES,
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_DUPLICATE,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_ORPHANED,
    MANIFEST_STATUS_UNCHANGED,
    MANIFEST_STATUS_UNKNOWN,
    MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH,
)

from tests.support.source_graph import defining_modules

_MANIFEST_ROOTS = ("src/rag_core/manifest", "src/rag_core/ingest")
_STATUSES_OWNER = "rag_core.manifest.reconciliation.statuses"
_REASONS_OWNER = "rag_core.manifest.reconciliation.reasons"


def test_manifest_reconciliation_statuses_have_single_owner() -> None:
    assert MANIFEST_STATUS_UNKNOWN == "unknown"
    assert MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH == "unknown_until_fetch"
    assert MANIFEST_STATUS_CHANGED == "changed"
    assert MANIFEST_STATUS_DUPLICATE == "duplicate"
    assert MANIFEST_STATUS_MISSING == "missing"
    assert MANIFEST_STATUS_ORPHANED == "orphaned"
    assert MANIFEST_STATUS_UNCHANGED == "unchanged"
    assert MANIFEST_RECONCILIATION_STATUSES == (
        MANIFEST_STATUS_UNCHANGED,
        MANIFEST_STATUS_CHANGED,
        MANIFEST_STATUS_MISSING,
        MANIFEST_STATUS_ORPHANED,
        MANIFEST_STATUS_DUPLICATE,
    )
    assert MANIFEST_NEEDS_REINDEX_STATUSES == (
        MANIFEST_STATUS_MISSING,
        MANIFEST_STATUS_CHANGED,
    )
    assert LOCAL_MANIFEST_SOURCE_STATUSES == (
        MANIFEST_STATUS_CHANGED,
        MANIFEST_STATUS_DUPLICATE,
        MANIFEST_STATUS_MISSING,
        MANIFEST_STATUS_UNCHANGED,
    )

    # Exactly one module binds each status constant, alias, and set, so no
    # consumer can re-derive the literal (or its own status set) under the name.
    for name in (
        "ManifestReconciliationStatus",
        "LocalManifestStatus",
        "RemoteManifestStatus",
        "MANIFEST_STATUS_UNKNOWN",
        "MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH",
        "MANIFEST_STATUS_CHANGED",
        "MANIFEST_STATUS_DUPLICATE",
        "MANIFEST_STATUS_MISSING",
        "MANIFEST_STATUS_ORPHANED",
        "MANIFEST_STATUS_UNCHANGED",
        "MANIFEST_RECONCILIATION_STATUSES",
        "MANIFEST_NEEDS_REINDEX_STATUSES",
        "LOCAL_MANIFEST_SOURCE_STATUSES",
    ):
        assert defining_modules(*_MANIFEST_ROOTS, name=name) == {_STATUSES_OWNER}


def test_manifest_reconciliation_reasons_have_single_owner() -> None:
    assert (
        MANIFEST_REASON_CANONICAL_URL_UNKNOWN_UNTIL_FETCH
        == "canonical_url_unknown_until_fetch"
    )
    assert MANIFEST_REASON_CONTENT_SHA256_CHANGED == "content_sha256_changed"
    assert MANIFEST_REASON_CONTENT_SHA256_MATCH == "content_sha256_match"
    assert MANIFEST_REASON_DUPLICATE_DOCUMENT_KEY == "duplicate_manifest_document_key"
    assert MANIFEST_REASON_ENTRY_WITHOUT_SOURCE == "manifest_entry_without_source"
    assert MANIFEST_REASON_NOT_CHECKED == "manifest_not_checked"
    assert MANIFEST_REASON_PRESENT_WITHOUT_HASH_CHECK == "present_without_hash_check"
    assert MANIFEST_REASON_SOURCE_NOT_IN_MANIFEST == "source_not_in_manifest"
    assert MANIFEST_REASON_SOURCE_READ_FAILED == "source_read_failed"

    for name in (
        "MANIFEST_REASON_CANONICAL_URL_UNKNOWN_UNTIL_FETCH",
        "MANIFEST_REASON_CONTENT_SHA256_CHANGED",
        "MANIFEST_REASON_CONTENT_SHA256_MATCH",
        "MANIFEST_REASON_DUPLICATE_DOCUMENT_KEY",
        "MANIFEST_REASON_ENTRY_WITHOUT_SOURCE",
        "MANIFEST_REASON_NOT_CHECKED",
        "MANIFEST_REASON_PRESENT_WITHOUT_HASH_CHECK",
        "MANIFEST_REASON_SOURCE_NOT_IN_MANIFEST",
        "MANIFEST_REASON_SOURCE_READ_FAILED",
    ):
        assert defining_modules(*_MANIFEST_ROOTS, name=name) == {_REASONS_OWNER}
