from __future__ import annotations

from pathlib import Path

from rag_core.manifest_reconciliation_statuses import (
    LOCAL_MANIFEST_SOURCE_STATUSES,
    MANIFEST_NEEDS_REINDEX_STATUSES,
    MANIFEST_RECONCILIATION_STATUSES,
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_DUPLICATE,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_ORPHANED,
    MANIFEST_STATUS_UNKNOWN,
    MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH,
    MANIFEST_STATUS_UNCHANGED,
)
from rag_core.manifest_reconciliation_reasons import (
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

CANONICAL_LAUNCH_GATES = (
    "uv run ruff check .",
    "uv run mypy src tests examples",
    "uv run pytest -q",
    "./scripts/dx_smoke.sh",
    "./scripts/ci_self_host_smoke.sh",
    "uv build",
    "uv run python scripts/check_dist_artifacts.py",
    "uv run python scripts/wheel_smoke.py",
)


def test_manifest_reconciliation_statuses_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/manifest_reconciliation_statuses.py",
            "src/rag_core/manifest_reconciliation.py",
            "src/rag_core/manifest_reconciliation_matching.py",
            "src/rag_core/local_ingest_manifest.py",
            "src/rag_core/local_ingest_models.py",
            "src/rag_core/local_ingest_records.py",
            "src/rag_core/remote_ingest_manifest.py",
            "src/rag_core/remote_ingest_results.py",
            "src/rag_core/remote_ingest_runner.py",
        )
    }

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
    owner = sources["src/rag_core/manifest_reconciliation_statuses.py"]
    for definition in (
        'MANIFEST_STATUS_UNKNOWN: Final[ManifestUnknownStatus] = "unknown"',
        (
            "MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH: "
            "Final[ManifestUnknownUntilFetchStatus] = ("
        ),
        'MANIFEST_STATUS_CHANGED: Final[ManifestReconciliationStatus] = "changed"',
        'MANIFEST_STATUS_DUPLICATE: Final[ManifestReconciliationStatus] = "duplicate"',
        'MANIFEST_STATUS_MISSING: Final[ManifestReconciliationStatus] = "missing"',
        'MANIFEST_STATUS_ORPHANED: Final[ManifestReconciliationStatus] = "orphaned"',
        'MANIFEST_STATUS_UNCHANGED: Final[ManifestReconciliationStatus] = "unchanged"',
    ):
        assert owner.count(definition) == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/manifest_reconciliation_statuses.py"
    )
    for symbol in (
        "MANIFEST_STATUS_CHANGED",
        "MANIFEST_STATUS_DUPLICATE",
        "MANIFEST_STATUS_MISSING",
        "MANIFEST_STATUS_ORPHANED",
        "MANIFEST_STATUS_UNKNOWN",
        "MANIFEST_STATUS_UNKNOWN_UNTIL_FETCH",
        "MANIFEST_STATUS_UNCHANGED",
        "MANIFEST_RECONCILIATION_STATUSES",
        "MANIFEST_NEEDS_REINDEX_STATUSES",
        "LOCAL_MANIFEST_SOURCE_STATUSES",
        "LocalManifestStatus",
        "RemoteManifestStatus",
    ):
        assert symbol in consumers
    for duplicate in (
        'status="changed"',
        'status="duplicate"',
        'status="missing"',
        'status="orphaned"',
        'status="unchanged"',
        'manifest_status: LocalManifestStatus = "unknown"',
        'manifest_status: RemoteManifestStatus = "unknown"',
        'manifest_status == "unknown"',
        'return "unknown",',
        'return "unknown_until_fetch",',
        'Literal["unknown"',
        'by_status("changed")',
        'by_status("duplicate")',
        'by_status("missing")',
        'by_status("orphaned")',
        'by_status("unchanged")',
        '{"changed", "duplicate", "missing", "unchanged"}',
        '{"changed", "duplicate", "missing", "orphaned", "unchanged"}',
        '{"missing", "changed"}',
    ):
        assert duplicate not in consumers




def test_manifest_reconciliation_reasons_have_single_owner() -> None:
    sources = {
        path: Path(path).read_text(encoding="utf-8")
        for path in (
            "src/rag_core/manifest_reconciliation_reasons.py",
            "src/rag_core/manifest_reconciliation_matching.py",
            "src/rag_core/local_ingest_manifest.py",
            "src/rag_core/local_ingest_models.py",
            "src/rag_core/local_ingest_records.py",
            "src/rag_core/remote_ingest_manifest.py",
            "src/rag_core/remote_ingest_results.py",
        )
    }

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

    owner = sources["src/rag_core/manifest_reconciliation_reasons.py"]
    for symbol in (
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
        assert owner.count(symbol) == 1

    consumers = "\n".join(
        source
        for path, source in sources.items()
        if path != "src/rag_core/manifest_reconciliation_reasons.py"
    )
    for symbol in (
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
        assert symbol in consumers
    for duplicate in (
        '"canonical_url_unknown_until_fetch"',
        '"content_sha256_changed"',
        '"content_sha256_match"',
        '"duplicate_manifest_document_key"',
        '"manifest_entry_without_source"',
        '"manifest_not_checked"',
        '"present_without_hash_check"',
        '"source_not_in_manifest"',
        '"source_read_failed"',
    ):
        assert duplicate not in consumers
