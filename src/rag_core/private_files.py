from __future__ import annotations

import errno
import io
import os
from pathlib import Path
from typing import TextIO


def ensure_private_parent_dirs(path: Path, *, reject_symlinks: bool = False) -> None:
    parent = path.parent
    if reject_symlinks:
        reject_symlink_ancestors(path)
    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError(f"parent path is not a directory: {parent}")
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        return
    for directory in reversed(missing):
        os.chmod(directory, 0o700)


def open_private_append_handle(
    path: Path,
    *,
    reject_symlink: bool = False,
) -> TextIO:
    """Return an open text file handle in append mode with private-file hardening.

    The handle is already fchmod-ed to 0o600 and, when *reject_symlink* is
    True, opened with O_NOFOLLOW (or a pre-open symlink check on platforms
    that lack the flag).  Caller is responsible for closing the handle.
    """
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    fd = _open_private_fd(path, flags, reject_symlink=reject_symlink)
    return io.open(fd, "a", encoding="utf-8", closefd=True)


def append_private_text(path: Path, text: str, *, reject_symlink: bool = False) -> None:
    ensure_private_parent_dirs(path, reject_symlinks=reject_symlink)
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    with os.fdopen(
        _open_private_fd(path, flags, reject_symlink=reject_symlink),
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(text)


def write_private_text_exclusive(
    path: Path,
    text: str,
    *,
    reject_symlink: bool = False,
) -> None:
    ensure_private_parent_dirs(path, reject_symlinks=reject_symlink)
    flags = os.O_CREAT | os.O_WRONLY | os.O_EXCL
    with os.fdopen(
        _open_private_fd(path, flags, reject_symlink=reject_symlink),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text)


def harden_private_file(path: Path) -> None:
    if os.name == "nt":
        return
    os.chmod(path, 0o600)


def reject_hardlinked_private_fd(fd: int, path: Path) -> None:
    if os.name == "nt":
        return
    stat_result = os.fstat(fd)
    if stat_result.st_nlink > 1:
        raise ValueError(f"private file path must not be hard-linked: {path}")


def reject_hardlinked_private_path(path: Path) -> None:
    if os.name == "nt" or not path.exists():
        return
    stat_result = path.stat()
    if stat_result.st_nlink > 1:
        raise ValueError(f"private file path must not be hard-linked: {path}")


def prepare_private_file_for_open(path: Path, *, reject_symlink: bool = False) -> None:
    ensure_private_parent_dirs(path, reject_symlinks=reject_symlink)
    fd = _open_private_fd(path, os.O_CREAT | os.O_RDWR, reject_symlink=reject_symlink)
    os.close(fd)


def reject_symlink_path(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"path must not be a symlink: {path}")


def reject_symlink_ancestors(path: Path) -> None:
    if not path.is_absolute() and _logical_cwd_has_symlink_segment():
        raise ValueError("relative private file path must not be used from a symlinked PWD")
    for ancestor in reversed(path.parent.parents):
        if str(ancestor) in {"", "."}:
            continue
        try:
            if _is_macos_system_alias(ancestor):
                continue
            if ancestor.is_symlink():
                raise ValueError(f"path parent must not be a symlink: {ancestor}")
        except OSError as exc:
            raise ValueError(f"path parent is not inspectable: {ancestor}") from exc
    try:
        if _is_macos_system_alias(path.parent):
            return
        if path.parent.is_symlink():
            raise ValueError(f"path parent must not be a symlink: {path.parent}")
    except OSError as exc:
        raise ValueError(f"path parent is not inspectable: {path.parent}") from exc


def _is_macos_system_alias(path: Path) -> bool:
    if path not in {Path("/var"), Path("/tmp")}:
        return False
    try:
        return path.resolve(strict=False).is_relative_to(Path("/private"))
    except OSError:
        return False


def _logical_cwd_has_symlink_segment() -> bool:
    raw_pwd = os.environ.get("PWD")
    if not raw_pwd:
        return False
    pwd = Path(raw_pwd)
    if not pwd.is_absolute():
        return False
    try:
        if pwd.resolve(strict=False) != Path.cwd().resolve(strict=False):
            return False
    except OSError:
        return True
    return _path_has_symlink_segment(pwd)


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


def _open_private_fd(path: Path, flags: int, *, reject_symlink: bool) -> int:
    if reject_symlink and not hasattr(os, "O_NOFOLLOW") and path.is_symlink():
        raise ValueError(f"path must not be a symlink: {path}")
    if reject_symlink and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        if reject_symlink and exc.errno == errno.ELOOP:
            raise ValueError(f"path must not be a symlink: {path}") from exc
        raise
    try:
        if reject_symlink:
            reject_hardlinked_private_fd(fd, path)
        if os.name != "nt":
            os.fchmod(fd, 0o600)
    except Exception:
        os.close(fd)
        raise
    return fd
