from __future__ import annotations

import glob
import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rag_core.config import INGEST_SOURCE_TYPE_FILE
from rag_core.core_models import IngestedDocument
from rag_core.documents.formats import is_local_ingest_extension
from rag_core.file_io import detect_local_mime_type, read_file_bytes, validate_document_size

if TYPE_CHECKING:
    from rag_core.events.types import AuditContext


@dataclass(frozen=True)
class LocalSourceItem:
    path: Path
    document_key: str
    content_sha256: str | None
    source_error: str = ""


@dataclass(frozen=True)
class LocalSourcePlan:
    source: str
    root: Path
    items: tuple[LocalSourceItem, ...]

    @property
    def item_count(self) -> int:
        return len(self.items)


def read_local_source(
    source: str | Path,
    *,
    hash_file: Callable[[Path], str | None] | None = None,
    hash_content: bool = True,
) -> LocalSourcePlan:
    raw = str(source)
    root = local_source_key_root(raw)
    content_hash = hash_file or (
        file_content_sha256 if hash_content else lambda _: None
    )
    return LocalSourcePlan(
        source=raw,
        root=root,
        items=tuple(
            local_file_source_item(path, root=root, hash_file=content_hash)
            for path in expand_supported_local_files(raw)
        ),
    )


def local_file_source_item(
    path: Path,
    *,
    root: Path,
    hash_file: Callable[[Path], str | None] | None = None,
) -> LocalSourceItem:
    reject_local_symlink_path(path)
    reject_local_symlink_path(root)
    reject_local_hardlink_path(path)
    content_hash = hash_file or file_content_sha256
    try:
        content_sha256 = content_hash(path)
        source_error = ""
    except OSError as exc:
        content_sha256 = None
        source_error = source_error_message(exc)
    return LocalSourceItem(
        path=path,
        document_key=document_key(root, path),
        content_sha256=content_sha256,
        source_error=source_error,
    )


def local_source_key_root(raw: str) -> Path:
    target = Path(raw)
    if target.exists():
        return target.parent if target.is_file() else target
    if any(ch in raw for ch in "*?["):
        prefix_parts: list[str] = []
        for part in Path(raw).parts:
            if any(ch in part for ch in "*?["):
                break
            prefix_parts.append(part)
        return Path(*prefix_parts) if prefix_parts else Path(".")
    return target.parent if target.is_file() else target


def expand_supported_local_files(spec: str) -> list[Path]:
    raw = spec.strip()
    if not raw:
        return []
    target = Path(raw)
    if target.exists():
        if target.is_file():
            if path_has_symlink_segment(target):
                return []
            if is_multi_link_regular_file(target):
                return []
            return [target] if is_supported_local_candidate(target) else []
        if target.is_dir():
            if target.is_symlink():
                return []
            return sorted(
                path
                for path in _safe_directory_files(target)
                if is_supported_local_candidate(path)
            )
        return []
    if any(ch in raw for ch in "*?["):
        root = local_source_key_root(raw)
        resolved_root = (
            _resolved_path(root) if root.exists() and root.is_dir() else None
        )
        return _dedupe_paths(
            [
                path
                for match in sorted(glob.glob(raw, recursive=True))
                if not path_has_symlink_segment(path := Path(match))
                and path.is_file()
                and not is_multi_link_regular_file(path)
                and (resolved_root is None or _path_within_root(path, resolved_root))
                and is_supported_local_candidate(path)
            ]
        )
    return []


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(str(path.expanduser())))


def _resolved_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _path_within_root(path: Path, root: Path) -> bool:
    return _resolved_path(path).is_relative_to(root)


def path_has_symlink_segment(path: Path) -> bool:
    if not path.is_absolute() and _logical_cwd_has_symlink_segment():
        return True
    return _path_has_symlink_segment(path)


def _path_has_symlink_segment(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if str(candidate) in {"", "."}:
            continue
        try:
            if _is_macos_system_alias(candidate):
                continue
            if stat.S_ISLNK(candidate.lstat().st_mode):
                return True
        except FileNotFoundError:
            continue
    return False


def _logical_cwd_has_symlink_segment() -> bool:
    raw_pwd = os.environ.get("PWD")
    if not raw_pwd:
        return False
    pwd = Path(raw_pwd)
    if not pwd.is_absolute():
        return False
    try:
        if _resolved_path(pwd) != _resolved_path(Path.cwd()):
            return False
    except OSError:
        return True
    return _path_has_symlink_segment(pwd)


def _is_macos_system_alias(path: Path) -> bool:
    if path in {Path("/var"), Path("/tmp")}:
        try:
            return path.resolve(strict=False).is_relative_to(Path("/private"))
        except OSError:
            return False
    return False


def reject_local_symlink_path(path: Path) -> None:
    if path_has_symlink_segment(path):
        raise ValueError(
            f"local file ingest does not allow symlink paths: {str(path)!r}"
        )


def reject_local_hardlink_path(path: Path) -> None:
    if is_multi_link_regular_file(path):
        raise ValueError(
            f"local file ingest does not allow multi-link file paths: {str(path)!r}"
        )


def is_multi_link_regular_file(path: Path) -> bool:
    path_stat = path.stat(follow_symlinks=False)
    return stat.S_ISREG(path_stat.st_mode) and path_stat.st_nlink > 1


def _safe_directory_files(root: Path) -> list[Path]:
    def read_failed(error: OSError) -> None:
        raise error

    resolved_root = _resolved_path(root)
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=read_failed
    ):
        current = Path(current_root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if _path_within_root(current / name, resolved_root)
        )
        for name in sorted(filenames):
            path = current / name
            path_stat = path.stat(follow_symlinks=False)
            if stat.S_ISLNK(path_stat.st_mode) or path_stat.st_nlink > 1:
                continue
            if not _path_within_root(path, resolved_root):
                continue
            if stat.S_ISREG(path_stat.st_mode):
                files.append(path)
    return files


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        key = _lexical_absolute(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def document_key(root: Path, file_path: Path) -> str:
    lexical_root = _resolved_path(root)
    lexical_file = _resolved_path(file_path)
    try:
        relative = lexical_file.relative_to(lexical_root)
    except ValueError:
        relative = Path(lexical_file.name)
    if not str(relative) or str(relative) == ".":
        relative = Path(lexical_file.name)
    return f"local:{relative.as_posix()}#source:{local_source_id(root)}"


def local_source_id(root: Path) -> str:
    return hashlib.sha256(str(_resolved_path(root)).encode("utf-8")).hexdigest()[:16]


def file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_error_message(exc: OSError) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"file not found: {exc.filename or exc}"
    return str(exc)


def is_supported_local_file(path: Path) -> bool:
    if path.name.startswith("~$"):
        return False
    return is_local_ingest_extension(path.suffix)


def is_ignored_local_file(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix.lower() in {".pyc", ".pyo"}


def is_supported_local_candidate(path: Path) -> bool:
    return not is_ignored_local_file(path) and is_supported_local_file(path)


class LocalIngestBytes(Protocol):
    async def __call__(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        tenant_id: str,
        collection: str,
        document_id: str | None = None,
        document_key: str | None = None,
        path: str | None = None,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        source_type: str | None = None,
        audit_context: "AuditContext | None" = None,
        ingest_id: str | None = None,
    ) -> IngestedDocument: ...


async def ingest_local_file_source(
    path: str | Path,
    *,
    ingest_bytes: LocalIngestBytes,
    tenant_id: str,
    collection: str,
    document_id: str | None = None,
    document_key: str | None = None,
    metadata: dict[str, str] | None = None,
    force_reindex: bool = False,
    audit_context: "AuditContext | None" = None,
    ingest_id: str | None = None,
    pre_read_bytes: bytes | None = None,
    max_document_bytes: int | None = None,
) -> IngestedDocument:
    file_path = Path(path)
    reject_local_symlink_path(file_path)
    reject_local_hardlink_path(file_path)
    resolved_document_key = _resolve_local_document_key(
        path=path,
        file_path=file_path,
        explicit_key=document_key,
    )
    file_bytes = (
        pre_read_bytes
        if pre_read_bytes is not None
        else await read_file_bytes(file_path, max_bytes=max_document_bytes)
    )
    if max_document_bytes is not None:
        validate_document_size(file_bytes, max_bytes=max_document_bytes)
    return await ingest_bytes(
        file_bytes=file_bytes,
        filename=file_path.name,
        mime_type=detect_local_mime_type(file_path),
        tenant_id=tenant_id,
        collection=collection,
        document_id=document_id,
        document_key=resolved_document_key,
        path=str(file_path),
        metadata=metadata,
        force_reindex=force_reindex,
        source_type=INGEST_SOURCE_TYPE_FILE,
        audit_context=audit_context,
        ingest_id=ingest_id,
    )


def _resolve_local_document_key(
    *,
    path: str | Path,
    file_path: Path,
    explicit_key: str | None,
) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    if any(part == ".." for part in Path(path).parts):
        raise ValueError(
            "document_key is required for single-file ingest when path traversal segments prevent inferring a stable collection root"
        )
    return document_key(local_source_key_root(str(path)), file_path)


__all__ = [
    "LocalIngestBytes",
    "LocalSourceItem",
    "LocalSourcePlan",
    "document_key",
    "expand_supported_local_files",
    "file_content_sha256",
    "ingest_local_file_source",
    "is_ignored_local_file",
    "is_supported_local_candidate",
    "is_supported_local_file",
    "local_file_source_item",
    "local_source_key_root",
    "path_has_symlink_segment",
    "read_local_source",
    "reject_local_symlink_path",
    "source_error_message",
]
