from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from rag_core.archive_member_policy import archive_document_key
from rag_core.archive_member_policy import is_supported_archive_member_path
from rag_core.archive_member_policy import safe_archive_member_path
from rag_core.archive_member_policy import validate_archive_member_size
from rag_core._engine.core_file_io import detect_mime_type_for_name
from rag_core.local_sources import reject_local_hardlink_path
from rag_core.local_sources import reject_local_symlink_path

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
    _member_bytes: bytes | None = field(
        default=None,
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
        if self._member_bytes is None:
            raise ValueError("archive source item does not include member bytes")
        return self._member_bytes

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
        total_bytes = 0
        seen: set[str] = set()
        try:
            with zipfile.ZipFile(path) as archive:
                entries = [entry for entry in archive.infolist() if not entry.is_dir()]
                if len(entries) > resolved_limits.max_entries:
                    raise ValueError(
                        "archive exceeds max_entries "
                        f"({resolved_limits.max_entries})"
                    )
                for entry in entries:
                    member_path = safe_archive_member_path(entry.filename)
                    if member_path in seen:
                        raise ValueError(
                            f"archive contains duplicate member path: {member_path!r}"
                        )
                    seen.add(member_path)
                    if not is_supported_archive_member_path(member_path):
                        continue
                    validate_archive_member_size(
                        entry,
                        max_entry_bytes=resolved_limits.max_entry_bytes,
                    )
                    total_bytes += entry.file_size
                    if total_bytes > resolved_limits.max_total_bytes:
                        raise ValueError(
                            "archive exceeds max_total_bytes "
                            f"({resolved_limits.max_total_bytes})"
                        )
                    member_bytes, content_sha256 = _read_member_bytes_and_sha256(
                        archive,
                        entry,
                    )
                    items.append(
                        ArchiveSourceItem(
                            archive_path=path,
                            member_path=member_path,
                            document_key=archive_document_key(path, member_path),
                            filename=PurePosixPath(member_path).name,
                            mime_type=detect_mime_type_for_name(member_path),
                            content_sha256=content_sha256,
                            byte_count=entry.file_size,
                            _member_bytes=member_bytes,
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
            entry = _unique_archive_member(archive, safe_member_path)
            validate_archive_member_size(
                entry,
                max_entry_bytes=resolved_limits.max_entry_bytes,
            )
            return archive.read(entry)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"archive is not a valid ZIP file: {str(path)!r}") from exc


def _read_member_bytes_and_sha256(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with archive.open(entry) as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            chunks.append(chunk)
            digest.update(chunk)
    return b"".join(chunks), digest.hexdigest()


def _unique_archive_member(
    archive: zipfile.ZipFile,
    member_path: str,
) -> zipfile.ZipInfo:
    matches: list[zipfile.ZipInfo] = []
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        safe_path = safe_archive_member_path(entry.filename)
        if safe_path == member_path:
            matches.append(entry)
    if not matches:
        raise ValueError(f"archive member not found: {member_path!r}")
    if len(matches) > 1:
        raise ValueError(f"archive contains duplicate member path: {member_path!r}")
    return matches[0]


def _reject_archive_file_path(path: Path) -> None:
    reject_local_symlink_path(path)
    if path.exists():
        reject_local_hardlink_path(path)


__all__ = [
    "ArchiveLimits",
    "ArchiveSourceItem",
    "ArchiveSourcePlan",
    "ZipArchiveSourceReader",
    "archive_document_key",
    "is_supported_archive_member_path",
    "read_zip_member_bytes",
    "safe_archive_member_path",
]
