from __future__ import annotations

import glob
import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag_core.documents.converters.format_support import is_local_ingest_extension


@dataclass(frozen=True)
class LocalSourceItem:
    path: Path
    document_key: str
    content_sha256: str | None
    source_error: str = ""

    def to_payload(
        self,
        *,
        manifest_status: str | None = None,
        manifest_reason: str | None = None,
        include_private: bool = False,
    ) -> dict[str, object]:
        if include_private:
            payload: dict[str, object] = {
                "path": str(self.path),
                "document_key": self.document_key,
                "content_sha256": self.content_sha256,
                "source_error": self.source_error,
            }
        else:
            payload = {
                "path": "<local-file>",
                "filename": self.path.name,
                "content_sha256_available": self.content_sha256 is not None,
                "source_error": "source read failed" if self.source_error else "",
            }
        if manifest_status is not None:
            payload["manifest_status"] = manifest_status
            payload["manifest_reason"] = manifest_reason or ""
        return payload


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
    hash_file: Callable[[Path], str] | None = None,
) -> LocalSourcePlan:
    raw = str(source)
    root = local_source_key_root(raw)
    content_hash = hash_file or file_content_sha256
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
    hash_file: Callable[[Path], str] | None = None,
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
    try:
        return _resolved_path(path).is_relative_to(root)
    except OSError:
        return False


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
            if candidate.is_symlink():
                return True
        except OSError:
            return True
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
    try:
        path_stat = path.stat(follow_symlinks=False)
    except OSError:
        return True
    return stat.S_ISREG(path_stat.st_mode) and path_stat.st_nlink > 1


def _safe_directory_files(root: Path) -> list[Path]:
    resolved_root = _resolved_path(root)
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if _path_within_root(current / name, resolved_root)
        )
        for name in sorted(filenames):
            path = current / name
            if path.is_symlink():
                continue
            if is_multi_link_regular_file(path):
                continue
            if not _path_within_root(path, resolved_root):
                continue
            if path.is_file():
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
    source_hash = hashlib.sha256(str(lexical_root).encode("utf-8")).hexdigest()[:16]
    return f"local:{relative.as_posix()}#source:{source_hash}"


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


__all__ = [
    "LocalSourceItem",
    "LocalSourcePlan",
    "document_key",
    "expand_supported_local_files",
    "file_content_sha256",
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
