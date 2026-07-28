from __future__ import annotations

import hashlib
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from rag_core.ingest.sources.archive_policy import archive_document_key
from rag_core.ingest.sources.archive_policy import archive_member_sha256
from rag_core.ingest.sources.archive_policy import is_supported_archive_member_path
from rag_core.ingest.sources.archive_policy import safe_archive_member_path
from rag_core.ingest.sources.archive_policy import validate_archive_member_size
from rag_core.file_io import detect_mime_type_for_name
from rag_core.ingest.sources.local import reject_local_hardlink_path
from rag_core.ingest.sources.local import reject_local_symlink_path

DEFAULT_MAX_ARCHIVE_ENTRIES = 1_000
DEFAULT_MAX_ARCHIVE_ENTRY_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_TOTAL_BYTES = 250 * 1024 * 1024


@dataclass(frozen=True)
class ArchiveLimits:
    max_entries: int = DEFAULT_MAX_ARCHIVE_ENTRIES
    max_entry_bytes: int = DEFAULT_MAX_ARCHIVE_ENTRY_BYTES
    max_total_bytes: int = DEFAULT_MAX_ARCHIVE_TOTAL_BYTES

    def __post_init__(self) -> None:
        if self.max_entries <= 0:
            raise ValueError("ArchiveLimits.max_entries must be positive")
        if self.max_entry_bytes <= 0:
            raise ValueError("ArchiveLimits.max_entry_bytes must be positive")
        if self.max_total_bytes <= 0:
            raise ValueError("ArchiveLimits.max_total_bytes must be positive")


@dataclass(frozen=True)
class ArchiveSourceItem:
    archive_path: Path
    member_path: str
    document_key: str
    filename: str
    mime_type: str
    content_sha256: str
    byte_count: int
    _limits: ArchiveLimits = field(
        default_factory=ArchiveLimits,
        repr=False,
        compare=False,
        kw_only=True,
    )

    @property
    def path(self) -> str:
        return f"{self.archive_path}!/{self.member_path}"

    @property
    def display_path(self) -> str:
        return f"{self.archive_path.name}!/{self.member_path}"

    @property
    def member_bytes(self) -> bytes:
        return self.read_bytes()

    def read_bytes(self, *, limits: ArchiveLimits | None = None) -> bytes:
        payload = read_zip_member_bytes(
            self.archive_path,
            self.member_path,
            limits=limits or self._limits,
        )
        if (
            len(payload) != self.byte_count
            or hashlib.sha256(payload).hexdigest() != self.content_sha256
        ):
            raise ValueError(
                f"archive member changed after planning: {self.member_path!r}"
            )
        return payload

    def to_payload(self) -> dict[str, object]:
        return {
            "archive_name": self.archive_path.name,
            "member_path": self.member_path,
            "path": self.display_path,
            "document_key": self.document_key,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "content_sha256": self.content_sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class ArchiveSourcePlan:
    archive_path: Path
    items: tuple[ArchiveSourceItem, ...]

    @property
    def item_count(self) -> int:
        return len(self.items)

    def to_payload(self) -> dict[str, object]:
        return {
            "archive_name": self.archive_path.name,
            "item_count": self.item_count,
            "items": [item.to_payload() for item in self.items],
        }


class ZipArchiveReadSession:
    def __init__(
        self,
        archive_path: str | Path,
        *,
        limits: ArchiveLimits | None = None,
    ) -> None:
        self._path = Path(archive_path)
        self._limits = limits or ArchiveLimits()
        self._archive: zipfile.ZipFile | None = None
        self._entries: dict[str, zipfile.ZipInfo] = {}
        self._lock = threading.Lock()

    def __enter__(self) -> ZipArchiveReadSession:
        _reject_archive_file_path(self._path)
        archive: zipfile.ZipFile | None = None
        try:
            archive = zipfile.ZipFile(self._path)
            entries = _validated_archive_member_index(archive, self._limits)
        except zipfile.BadZipFile as exc:
            if archive is not None:
                archive.close()
            raise ValueError(
                f"archive is not a valid ZIP file: {str(self._path)!r}"
            ) from exc
        except Exception:
            if archive is not None:
                archive.close()
            raise
        self._archive = archive
        self._entries = entries
        return self

    def __exit__(self, *_args: object) -> None:
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._entries = {}

    def read_item(self, item: ArchiveSourceItem) -> bytes:
        if item.archive_path != self._path:
            raise ValueError("archive item does not belong to this read session")
        with self._lock:
            archive = self._archive
            if archive is None:
                raise RuntimeError("archive read session is not open")
            entry = self._entries.get(item.member_path)
            if entry is None:
                raise ValueError(f"archive member not found: {item.member_path!r}")
            validate_archive_member_size(
                entry,
                max_entry_bytes=self._limits.max_entry_bytes,
            )
            payload = archive.read(entry)
        if (
            len(payload) != item.byte_count
            or hashlib.sha256(payload).hexdigest() != item.content_sha256
        ):
            raise ValueError(
                f"archive member changed after planning: {item.member_path!r}"
            )
        return payload


class ZipArchiveSourceReader:
    def read(
        self,
        archive_path: str | Path,
        *,
        limits: ArchiveLimits | None = None,
    ) -> ArchiveSourcePlan:
        resolved_limits = limits or ArchiveLimits()
        path = Path(archive_path)
        _reject_archive_file_path(path)
        items: list[ArchiveSourceItem] = []
        try:
            with zipfile.ZipFile(path) as archive:
                entries = _validated_archive_member_index(
                    archive,
                    resolved_limits,
                )
                for member_path, entry in entries.items():
                    if not is_supported_archive_member_path(member_path):
                        continue
                    items.append(
                        ArchiveSourceItem(
                            archive_path=path,
                            member_path=member_path,
                            document_key=archive_document_key(path, member_path),
                            filename=PurePosixPath(member_path).name,
                            mime_type=detect_mime_type_for_name(member_path),
                            content_sha256=archive_member_sha256(archive, entry),
                            byte_count=entry.file_size,
                            _limits=resolved_limits,
                        )
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError(f"archive is not a valid ZIP file: {str(path)!r}") from exc
        return ArchiveSourcePlan(archive_path=path, items=tuple(items))


def read_zip_member_bytes(
    archive_path: str | Path,
    member_path: str,
    *,
    limits: ArchiveLimits | None = None,
) -> bytes:
    resolved_limits = limits or ArchiveLimits()
    path = Path(archive_path)
    safe_member_path = safe_archive_member_path(member_path)
    _reject_archive_file_path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            entry = _validated_archive_member_index(
                archive,
                resolved_limits,
            ).get(safe_member_path)
            if entry is None:
                raise ValueError(f"archive member not found: {safe_member_path!r}")
            return archive.read(entry)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"archive is not a valid ZIP file: {str(path)!r}") from exc


def _archive_member_index(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        safe_path = safe_archive_member_path(entry.filename)
        if safe_path in entries:
            raise ValueError(
                f"archive contains duplicate member path: {safe_path!r}"
            )
        entries[safe_path] = entry
    return entries


def _validated_archive_member_index(
    archive: zipfile.ZipFile,
    limits: ArchiveLimits,
) -> dict[str, zipfile.ZipInfo]:
    entries = _archive_member_index(archive)
    if len(entries) > limits.max_entries:
        raise ValueError(f"archive exceeds max_entries ({limits.max_entries})")

    total_bytes = 0
    for member_path, entry in entries.items():
        if not is_supported_archive_member_path(member_path):
            continue
        validate_archive_member_size(
            entry,
            max_entry_bytes=limits.max_entry_bytes,
        )
        total_bytes += entry.file_size
        if total_bytes > limits.max_total_bytes:
            raise ValueError(
                f"archive exceeds max_total_bytes ({limits.max_total_bytes})"
            )
    return entries


def _reject_archive_file_path(path: Path) -> None:
    reject_local_symlink_path(path)
    if path.exists():
        reject_local_hardlink_path(path)


__all__ = [
    "ArchiveLimits",
    "ArchiveSourceItem",
    "ArchiveSourcePlan",
    "ZipArchiveReadSession",
    "ZipArchiveSourceReader",
    "archive_document_key",
    "is_supported_archive_member_path",
    "read_zip_member_bytes",
    "safe_archive_member_path",
]
