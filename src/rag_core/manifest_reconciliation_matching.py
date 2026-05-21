from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, TypeVar

from rag_core.core_models import CorpusManifestEntry

_ItemT = TypeVar("_ItemT")
_ItemFactory = Callable[..., _ItemT]


class _ManifestSourceLike(Protocol):
    @property
    def document_key(self) -> str: ...

    @property
    def content_sha256(self) -> str | None: ...


def build_reconciliation_items(
    *,
    entries: Sequence[CorpusManifestEntry],
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
                    status="duplicate",
                    document_key=key,
                    document_id=entry.document_id,
                    manifest_content_sha256=entry.content_sha256,
                    source_content_sha256=(
                        source.content_sha256 if source is not None else None
                    ),
                    reason="duplicate_manifest_document_key",
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
                    status="orphaned",
                    document_key=key,
                    document_id=entry.document_id,
                    manifest_content_sha256=entry.content_sha256,
                    reason="manifest_entry_without_source",
                )
            )
    return items


def _source_item(
    *,
    source: _ManifestSourceLike,
    matched_entry: CorpusManifestEntry | None,
    item_type: _ItemFactory[_ItemT],
) -> _ItemT:
    if matched_entry is None:
        return item_type(
            status="missing",
            document_key=source.document_key,
            source_content_sha256=source.content_sha256,
            reason="source_not_in_manifest",
        )
    if _content_changed(source=source, entry=matched_entry):
        return item_type(
            status="changed",
            document_key=source.document_key,
            document_id=matched_entry.document_id,
            manifest_content_sha256=matched_entry.content_sha256,
            source_content_sha256=source.content_sha256,
            reason="content_sha256_changed",
        )
    return item_type(
        status="unchanged",
        document_key=source.document_key,
        document_id=matched_entry.document_id,
        manifest_content_sha256=matched_entry.content_sha256,
        source_content_sha256=source.content_sha256,
        reason=(
            "content_sha256_match"
            if source.content_sha256 is not None
            and matched_entry.content_sha256 is not None
            else "present_without_hash_check"
        ),
    )


def _content_changed(
    *,
    source: _ManifestSourceLike,
    entry: CorpusManifestEntry,
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


def _duplicate_entry_keys(entries: Sequence[CorpusManifestEntry]) -> set[str]:
    counts = Counter(_entry_key(entry) for entry in entries)
    return {key for key, count in counts.items() if count > 1}


def _unique_entries_by_key(
    entries: Sequence[CorpusManifestEntry],
    duplicate_entry_keys: set[str],
) -> dict[str, CorpusManifestEntry]:
    return {
        _entry_key(entry): entry
        for entry in entries
        if _entry_key(entry) not in duplicate_entry_keys
    }


def _entry_key(entry: CorpusManifestEntry) -> str:
    return entry.document_key or entry.document_id
