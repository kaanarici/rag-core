from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path, PurePosixPath

from rag_core.documents.converters.format_support import is_local_ingest_extension


def safe_archive_member_path(raw_path: str) -> str:
    if "\x00" in raw_path or "\\" in raw_path:
        raise ValueError("archive member path is unsafe")
    raw = raw_path
    if not raw or raw.startswith("/"):
        raise ValueError("archive member path is unsafe")
    parts = PurePosixPath(raw).parts
    if any(part in {"", ".", ".."} or part != part.strip() for part in parts):
        raise ValueError("archive member path is unsafe")
    normalized = "/".join(parts)
    if normalized != raw:
        raise ValueError("archive member path is unsafe")
    return normalized


def is_supported_archive_member_path(member_path: str) -> bool:
    try:
        safe_member_path = safe_archive_member_path(member_path)
    except ValueError:
        return False
    path = PurePosixPath(safe_member_path)
    if path.name.startswith("~$"):
        return False
    if "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}:
        return False
    return is_local_ingest_extension(path.suffix)


def archive_document_key(archive_path: Path, member_path: str) -> str:
    safe_member = safe_archive_member_path(member_path)
    source_hash = hashlib.sha256(
        canonical_archive_path(archive_path).encode("utf-8")
    ).hexdigest()[:16]
    return f"archive:{safe_member}#source:{source_hash}"


def canonical_archive_path(archive_path: Path) -> str:
    return str(archive_path.expanduser().resolve(strict=False))


def validate_archive_member_size(
    entry: zipfile.ZipInfo,
    *,
    max_entry_bytes: int,
) -> None:
    if entry.file_size < 0:
        raise ValueError("archive member has invalid size")
    if entry.file_size > max_entry_bytes:
        raise ValueError(f"archive member exceeds max_entry_bytes ({max_entry_bytes})")


def archive_member_sha256(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(entry) as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
