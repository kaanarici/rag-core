from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, TypeVar

from rag_core.core_models import CollectionManifestEntry
from rag_core.manifest.reconciliation.reasons import (
    MANIFEST_REASON_CONTENT_SHA256_CHANGED,
    MANIFEST_REASON_CONTENT_SHA256_MATCH,
    MANIFEST_REASON_DUPLICATE_DOCUMENT_KEY,
    MANIFEST_REASON_ENTRY_WITHOUT_SOURCE,
    MANIFEST_REASON_PRESENT_WITHOUT_HASH_CHECK,
    MANIFEST_REASON_SOURCE_NOT_IN_MANIFEST,
)
from rag_core.manifest.reconciliation.statuses import (
    MANIFEST_STATUS_CHANGED,
    MANIFEST_STATUS_DUPLICATE,
    MANIFEST_STATUS_MISSING,
    MANIFEST_STATUS_ORPHANED,
    MANIFEST_STATUS_UNCHANGED,
)

_ItemT = TypeVar("_ItemT")
_ItemFactory = Callable[..., _ItemT]


class _ManifestSourceLike(Protocol):
    @property
    def document_key(self) -> str: ...

    @property
    def content_sha256(self) -> str | None: ...


def build_reconciliation_items(
    *,
    entries: Sequence[CollectionManifestEntry],
    sources: Iterable[_ManifestSourceLike],
    item_type: _ItemFactory[_ItemT],
) -> list[_ItemT]:
    ordered_sources = tuple(sources)
    source_by_key = _sources_by_key(ordered_sources)
    duplicate_keys = _duplicate_entry_keys(entries)
    entry_by_key = _unique_entries_by_key(entries, duplicate_keys)
    items: list[_ItemT] = []

    for entry in entries:
        key = _entry_key(entry)
        if key in duplicate_keys:
            source = source_by_key.get(key)
            items.append(
                item_type(
                    status=MANIFEST_STATUS_DUPLICATE,
                    document_key=key,
                    document_id=entry.document_id,
                    manifest_content_sha256=entry.content_sha256,
                    source_content_sha256=(
                        source.content_sha256 if source is not None else None
                    ),
                    reason=MANIFEST_REASON_DUPLICATE_DOCUMENT_KEY,
                )
            )

    for source in ordered_sources:
        if source.document_key in duplicate_keys:
            continue
        matched_entry = entry_by_key.get(source.document_key)
        items.append(
            _source_item(source=source, matched_entry=matched_entry, item_type=item_type)
        )

    for key, entry in entry_by_key.items():
        if key not in source_by_key:
            items.append(
                item_type(
                    status=MANIFEST_STATUS_ORPHANED,
                    document_key=key,
                    document_id=entry.document_id,
                    manifest_content_sha256=entry.content_sha256,
                    reason=MANIFEST_REASON_ENTRY_WITHOUT_SOURCE,
                )
            )
    return items


def _source_item(
    *,
    source: _ManifestSourceLike,
    matched_entry: CollectionManifestEntry | None,
    item_type: _ItemFactory[_ItemT],
) -> _ItemT:
    if matched_entry is None:
        return item_type(
            status=MANIFEST_STATUS_MISSING,
            document_key=source.document_key,
            source_content_sha256=source.content_sha256,
            reason=MANIFEST_REASON_SOURCE_NOT_IN_MANIFEST,
        )
    if _content_changed(source=source, entry=matched_entry):
        return item_type(
            status=MANIFEST_STATUS_CHANGED,
            document_key=source.document_key,
            document_id=matched_entry.document_id,
            manifest_content_sha256=matched_entry.content_sha256,
            source_content_sha256=source.content_sha256,
            reason=MANIFEST_REASON_CONTENT_SHA256_CHANGED,
        )
    return item_type(
        status=MANIFEST_STATUS_UNCHANGED,
        document_key=source.document_key,
        document_id=matched_entry.document_id,
        manifest_content_sha256=matched_entry.content_sha256,
        source_content_sha256=source.content_sha256,
        reason=(
            MANIFEST_REASON_CONTENT_SHA256_MATCH
            if source.content_sha256 is not None
            and matched_entry.content_sha256 is not None
            else MANIFEST_REASON_PRESENT_WITHOUT_HASH_CHECK
        ),
    )


def _content_changed(
    *,
    source: _ManifestSourceLike,
    entry: CollectionManifestEntry,
) -> bool:
    return (
        source.content_sha256 is not None
        and entry.content_sha256 is not None
        and source.content_sha256 != entry.content_sha256
    )


def _sources_by_key(
    sources: Sequence[_ManifestSourceLike],
) -> dict[str, _ManifestSourceLike]:
    source_by_key: dict[str, _ManifestSourceLike] = {}
    for source in sources:
        key = source.document_key
        if not key:
            raise ValueError("source document_key must be non-empty")
        if key in source_by_key:
            raise ValueError(f"source document_key must be unique: {key!r}")
        source_by_key[key] = source
    return source_by_key


def _duplicate_entry_keys(entries: Sequence[CollectionManifestEntry]) -> set[str]:
    counts = Counter(_entry_key(entry) for entry in entries)
    return {key for key, count in counts.items() if count > 1}


def _unique_entries_by_key(
    entries: Sequence[CollectionManifestEntry],
    duplicate_entry_keys: set[str],
) -> dict[str, CollectionManifestEntry]:
    return {
        _entry_key(entry): entry
        for entry in entries
        if _entry_key(entry) not in duplicate_entry_keys
    }


def _entry_key(entry: CollectionManifestEntry) -> str:
    return entry.document_key or entry.document_id
