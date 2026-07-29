"""Concrete on-disk manifest persistence.

JSONL one file per ``(namespace, collection)`` at
``<directory>/<namespace>/<collection>.jsonl``. The latest line for each
``document_id`` is the authoritative entry. Writes append when possible;
delete and compaction rewrite the JSONL atomically to keep the current
manifest small and canonical. ``read_entries`` collapses to the latest entry
per document.

Free functions keep the manifest boundary concrete until another persistence
adapter needs the same shape.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rag_core.core_models import CollectionManifestEntry
from rag_core.manifest.entries import ManifestReadError
from rag_core.manifest.entries import entry_to_dict
from rag_core.manifest.jsonl import append_manifest_jsonl_entry
from rag_core.manifest.jsonl import append_manifest_jsonl_entry_if_stale
from rag_core.manifest.jsonl import read_manifest_jsonl_entries
from rag_core.manifest.jsonl import update_manifest_jsonl_entries
from rag_core.manifest.paths import (
    manifest_scope_segments as _manifest_scope_segments,
    manifest_segment as _manifest_segment,
)
from rag_core.manifest.reconciliation import (
    ManifestReconciliation,
    ManifestReconciliationItem,
    ManifestReconciliationStatus,
    ManifestSource,
    manifest_reconciliation_payload,
    reconcile_entries,
)
from rag_core.private_files import reject_symlink_ancestors, reject_symlink_path


@dataclass(frozen=True)
class ManifestSummary:
    document_count: int
    chunk_count: int
    needs_ocr_count: int
    parser_counts: dict[str, int]


@dataclass(frozen=True)
class ManifestCompactionResult:
    before_entry_count: int
    after_entry_count: int
    content_changed: bool = False

    @property
    def removed_entry_count(self) -> int:
        return self.before_entry_count - self.after_entry_count

    @property
    def changed(self) -> bool:
        return self.content_changed or self.removed_entry_count > 0


def manifest_path(directory: Path, namespace: str, collection: str) -> Path:
    """Resolve the JSONL path for a collection.

    Namespace and collection must be single path segments so the manifest always
    stays under ``directory``.
    """
    namespace_segment, collection_segment = _manifest_scope_segments(
        namespace,
        collection,
    )
    path = Path(directory) / namespace_segment / f"{collection_segment}.jsonl"
    _reject_manifest_scope_symlinks(path)
    return path


def validate_manifest_scope(namespace: str, collection: str) -> None:
    """Reject manifest scope values that cannot map to one JSONL file."""
    _manifest_scope_segments(namespace, collection)


def write_entry(directory: Path, entry: CollectionManifestEntry) -> None:
    """Append a manifest entry as one JSONL line under the manifest lock."""
    path = manifest_path(directory, entry.namespace, entry.collection)
    append_manifest_jsonl_entry(path, entry)


def write_entry_if_stale(directory: Path, entry: CollectionManifestEntry) -> bool:
    path = manifest_path(directory, entry.namespace, entry.collection)
    return append_manifest_jsonl_entry_if_stale(path, entry, _entry_is_stale)


def write_entry_if_content_stale(directory: Path, entry: CollectionManifestEntry) -> bool:
    path = manifest_path(directory, entry.namespace, entry.collection)
    return append_manifest_jsonl_entry_if_stale(path, entry, _content_is_stale)


def delete_entry(
    directory: Path,
    *,
    namespace: str,
    collection: str,
    document_id: str,
) -> bool:
    path = manifest_path(directory, namespace, collection)
    if not path.exists():
        return False

    def without_document(
        entries: list[CollectionManifestEntry],
    ) -> list[CollectionManifestEntry]:
        return [entry for entry in entries if entry.document_id != document_id]

    _before_entry_count, _after_entry_count, changed = update_manifest_jsonl_entries(
        path,
        without_document,
    )
    return changed


def _entry_is_stale(
    current: CollectionManifestEntry | None,
    entry: CollectionManifestEntry,
) -> bool:
    if current is None:
        return True
    return entry_to_dict(current) != entry_to_dict(entry)


def _content_is_stale(
    current: CollectionManifestEntry | None,
    entry: CollectionManifestEntry,
) -> bool:
    if current is None:
        return True
    return current.content_sha256 != entry.content_sha256


def read_entries(
    directory: Path,
    namespace: str,
    collection: str | None = None,
) -> list[CollectionManifestEntry]:
    """Return the latest entry per ``document_id`` for the given scope.

    When ``collection`` is given, only that collection's JSONL is read. When
    ``collection`` is None, every collection directly under ``namespace`` is
    scanned and merged. Empty / missing directories return ``[]``.
    """
    namespace_segment = _manifest_segment("namespace", namespace)
    collection_segment = (
        _manifest_segment("collection", collection) if collection is not None else None
    )
    base = Path(directory) / namespace_segment
    _reject_manifest_scope_symlinks(base)
    if not base.exists():
        return []
    paths: list[Path]
    if collection_segment is not None:
        single = Path(directory) / namespace_segment / f"{collection_segment}.jsonl"
        paths = [single] if single.exists() else []
    else:
        paths = sorted(base.glob("*.jsonl"))

    by_doc: dict[tuple[str, str], CollectionManifestEntry] = {}
    for path in paths:
        for entry in read_manifest_jsonl_entries(path):
            by_doc[(entry.collection, entry.document_id)] = entry
    return list(by_doc.values())


def _reject_manifest_scope_symlinks(path: Path) -> None:
    reject_symlink_ancestors(path)
    reject_symlink_path(path)


def compact_manifest(
    directory: Path,
    namespace: str,
    collection: str,
) -> ManifestCompactionResult:
    """Rewrite one JSONL manifest to the latest entry per canonical source."""
    path = manifest_path(directory, namespace, collection)
    if not path.exists():
        return ManifestCompactionResult(before_entry_count=0, after_entry_count=0)

    def latest_by_document(
        entries: list[CollectionManifestEntry],
    ) -> list[CollectionManifestEntry]:
        by_doc: dict[str, CollectionManifestEntry] = {}
        key_owner: dict[str, str] = {}
        for entry in entries:
            doc_key = f"document_id:{entry.document_id}"
            if entry.document_key:
                previous_doc_key = key_owner.get(entry.document_key)
                if previous_doc_key is not None and previous_doc_key != doc_key:
                    by_doc.pop(previous_doc_key, None)
                key_owner[entry.document_key] = doc_key
                by_doc[doc_key] = entry
            else:
                by_doc[doc_key] = entry
        return list(by_doc.values())

    before_entry_count, after_entry_count, content_changed = (
        update_manifest_jsonl_entries(
            path,
            latest_by_document,
        )
    )
    result = ManifestCompactionResult(
        before_entry_count=before_entry_count,
        after_entry_count=after_entry_count,
        content_changed=content_changed,
    )
    return result


def summarize_entries(entries: Sequence[CollectionManifestEntry]) -> ManifestSummary:
    parser_counts = Counter(entry.parser or "unknown" for entry in entries)
    return ManifestSummary(
        document_count=len(entries),
        chunk_count=sum(entry.chunk_count for entry in entries),
        needs_ocr_count=sum(1 for entry in entries if entry.needs_ocr),
        parser_counts=dict(sorted(parser_counts.items())),
    )


__all__ = [
    "ManifestCompactionResult",
    "ManifestReconciliation",
    "ManifestReconciliationItem",
    "ManifestReconciliationStatus",
    "ManifestReadError",
    "ManifestSource",
    "ManifestSummary",
    "compact_manifest",
    "delete_entry",
    "manifest_reconciliation_payload",
    "manifest_path",
    "read_entries",
    "reconcile_entries",
    "summarize_entries",
    "validate_manifest_scope",
    "write_entry",
    "write_entry_if_content_stale",
    "write_entry_if_stale",
]
