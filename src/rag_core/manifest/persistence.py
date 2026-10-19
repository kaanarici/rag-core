"""Concrete on-disk manifest persistence.

JSONL one file per ``(tenant_id, collection)`` at
``<directory>/<tenant_id>/<collection>.jsonl``. The latest line for each
``document_id`` is the authoritative entry. Writes append when possible;
delete and compaction rewrite the JSONL atomically to keep the current
manifest small and canonical. ``read_entries`` collapses to the latest entry
per document.

Free functions keep the manifest boundary concrete until another persistence
adapter needs the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from rag_core.private_files import reject_symlink_ancestors, reject_symlink_path


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


def manifest_path(directory: Path, tenant_id: str, collection: str) -> Path:
    """Resolve the JSONL path for a collection.

    Scope identifiers are encoded as single path segments under ``directory``.
    """
    tenant_id_segment, collection_segment = _manifest_scope_segments(
        tenant_id,
        collection,
    )
    path = Path(directory) / tenant_id_segment / f"{collection_segment}.jsonl"
    _reject_manifest_scope_symlinks(path)
    return path


def validate_manifest_scope(tenant_id: str, collection: str) -> None:
    """Reject manifest scope values that cannot map to one JSONL file."""
    _manifest_scope_segments(tenant_id, collection)


def write_entry(directory: Path, entry: CollectionManifestEntry) -> None:
    """Append a manifest entry as one JSONL line under the manifest lock."""
    path = manifest_path(directory, entry.tenant_id, entry.collection)
    append_manifest_jsonl_entry(path, entry)


def write_entry_if_stale(directory: Path, entry: CollectionManifestEntry) -> bool:
    path = manifest_path(directory, entry.tenant_id, entry.collection)
    return append_manifest_jsonl_entry_if_stale(path, entry, _entry_is_stale)


def delete_entry(
    directory: Path,
    *,
    tenant_id: str,
    collection: str,
    document_id: str,
) -> bool:
    path = manifest_path(directory, tenant_id, collection)
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


def read_entries(
    directory: Path,
    tenant_id: str,
    collection: str | None = None,
) -> list[CollectionManifestEntry]:
    """Return the latest entry per ``document_id`` for the given scope.

    When ``collection`` is given, only that collection's JSONL is read. When
    ``collection`` is None, every collection directly under ``tenant_id`` is
    scanned and merged. Empty / missing directories return ``[]``.
    """
    tenant_id_segment = _manifest_segment("tenant_id", tenant_id)
    collection_segment = (
        _manifest_segment("collection", collection) if collection is not None else None
    )
    base = Path(directory) / tenant_id_segment
    _reject_manifest_scope_symlinks(base)
    if not base.exists():
        return []
    paths: list[Path]
    if collection_segment is not None:
        single = Path(directory) / tenant_id_segment / f"{collection_segment}.jsonl"
        paths = [single] if single.exists() else []
    else:
        paths = sorted(base.glob("*.jsonl"))

    by_doc: dict[tuple[str, str], CollectionManifestEntry] = {}
    for path in paths:
        for entry in read_manifest_jsonl_entries(path):
            if (
                entry.tenant_id != tenant_id
                or (collection is not None and entry.collection != collection)
                or path.name
                != f"{_manifest_segment('collection', entry.collection)}.jsonl"
            ):
                raise ManifestReadError(path, 0, "invalid_entry")
            by_doc[(entry.collection, entry.document_id)] = entry
    return list(by_doc.values())


def _reject_manifest_scope_symlinks(path: Path) -> None:
    reject_symlink_ancestors(path)
    reject_symlink_path(path)


def compact_manifest(
    directory: Path,
    tenant_id: str,
    collection: str,
) -> ManifestCompactionResult:
    """Rewrite one JSONL manifest to the latest entry per canonical source."""
    path = manifest_path(directory, tenant_id, collection)
    if not path.exists():
        return ManifestCompactionResult(before_entry_count=0, after_entry_count=0)

    def latest_by_document(
        entries: list[CollectionManifestEntry],
    ) -> list[CollectionManifestEntry]:
        seen_documents: set[str] = set()
        seen_keys: set[str] = set()
        current: list[CollectionManifestEntry] = []
        for entry in reversed(entries):
            if entry.document_id in seen_documents:
                continue
            seen_documents.add(entry.document_id)
            if entry.document_key:
                if entry.document_key in seen_keys:
                    continue
                seen_keys.add(entry.document_key)
            current.append(entry)
        return list(reversed(current))

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


__all__ = [
    "ManifestCompactionResult",
    "ManifestReadError",
    "compact_manifest",
    "delete_entry",
    "manifest_path",
    "read_entries",
    "validate_manifest_scope",
    "write_entry",
    "write_entry_if_stale",
]


def clear_entries(directory: Path, *, tenant_id: str, collection: str) -> None:
    path = manifest_path(directory, tenant_id, collection)
    if path.exists():
        update_manifest_jsonl_entries(path, lambda _entries: [])
