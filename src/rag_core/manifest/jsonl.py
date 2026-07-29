from __future__ import annotations

import json
import os
import tempfile
import importlib
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Sequence

from rag_core.core_models import CollectionManifestEntry
from rag_core.manifest.entries import ManifestReadError, entry_from_json_line, entry_to_dict
from rag_core.private_files import (
    ensure_private_parent_dirs,
    reject_hardlinked_private_fd,
    reject_hardlinked_private_path,
    reject_symlink_ancestors,
    reject_symlink_path,
)

_PROCESS_LOCKS: dict[str, RLock] = {}
_PROCESS_LOCKS_GUARD = RLock()
_PROCESS_INDEXES: dict[str, "_ManifestIndex"] = {}


@dataclass
class _ManifestIndex:
    signature: tuple[int, int]
    by_document_id: dict[str, CollectionManifestEntry]


def append_manifest_jsonl_entry(path: Path, entry: CollectionManifestEntry) -> None:
    _reject_manifest_io_paths(path)
    _ensure_private_manifest_parent(path)
    with _manifest_lock(path, exclusive=True):
        _append_line_unlocked(path, _manifest_entry_json(entry))
        _update_cached_index_after_append(path, entry)


def append_manifest_jsonl_entry_if_stale(
    path: Path,
    entry: CollectionManifestEntry,
    is_stale: Callable[[CollectionManifestEntry | None, CollectionManifestEntry], bool],
) -> bool:
    _reject_manifest_io_paths(path)
    _ensure_private_manifest_parent(path)
    with _manifest_lock(path, exclusive=True):
        by_document_id = _cached_latest_entries_unlocked(path)
        current = by_document_id.get(entry.document_id)
        if not is_stale(current, entry):
            return False
        _append_line_unlocked(path, _manifest_entry_json(entry))
        by_document_id[entry.document_id] = entry
        _refresh_cached_index_signature(path, by_document_id)
    return True


def read_manifest_jsonl_entries(path: Path) -> list[CollectionManifestEntry]:
    _reject_manifest_io_paths(path)
    with _manifest_lock(path, exclusive=False):
        entries = _read_manifest_jsonl_entries_unlocked(path)
        _set_cached_index(path, entries)
        return entries


def update_manifest_jsonl_entries(
    path: Path,
    transform: Callable[[list[CollectionManifestEntry]], Sequence[CollectionManifestEntry]],
) -> tuple[int, int, bool]:
    _reject_manifest_io_paths(path)
    _ensure_private_manifest_parent(path)
    with _manifest_lock(path, exclusive=True):
        before_content = path.read_text(encoding="utf-8") if path.exists() else ""
        before = _read_manifest_jsonl_entries_unlocked(path)
        after = list(transform(before))
        after_content = _manifest_entries_content(after)
        changed = after_content != before_content
        if changed:
            atomic_write_text(path, after_content)
        _set_cached_index(path, after)
    return len(before), len(after), changed


def _read_manifest_jsonl_entries_unlocked(path: Path) -> list[CollectionManifestEntry]:
    _reject_manifest_data_path(path)
    if not path.exists():
        return []
    reject_hardlinked_private_path(path)
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    entries: list[CollectionManifestEntry] = []
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(entry_from_json_line(path, line_number, line))
        except ManifestReadError:
            raise
    return entries


def write_manifest_jsonl_entries(
    path: Path,
    entries: Sequence[CollectionManifestEntry],
) -> None:
    _reject_manifest_io_paths(path)
    _ensure_private_manifest_parent(path)
    with _manifest_lock(path, exclusive=True):
        _write_manifest_jsonl_entries_unlocked(path, entries)
        _set_cached_index(path, entries)


def _write_manifest_jsonl_entries_unlocked(
    path: Path,
    entries: Sequence[CollectionManifestEntry],
) -> None:
    atomic_write_text(path, _manifest_entries_content(entries))


def _manifest_entries_content(entries: Sequence[CollectionManifestEntry]) -> str:
    content = "\n".join(_manifest_entry_json(entry) for entry in entries)
    return f"{content}\n" if content else ""


def _manifest_entry_json(entry: CollectionManifestEntry) -> str:
    payload = entry_to_dict(entry)
    entry_from_json_line(Path("<manifest-write>"), 1, json.dumps(payload))
    return json.dumps(payload, sort_keys=True)


def _append_line_unlocked(path: Path, line: str) -> None:
    _append_text_durable(path, f"{line}\n")


def _append_text_durable(path: Path, content: str) -> None:
    _reject_manifest_data_path(path)
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        reject_hardlinked_private_fd(fd, path)
        os.fchmod(fd, 0o600)
        data = content.encode("utf-8")
        total = 0
        while total < len(data):
            total += os.write(fd, data[total:])
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _latest_entries(
    entries: Sequence[CollectionManifestEntry],
) -> dict[str, CollectionManifestEntry]:
    by_document_id: dict[str, CollectionManifestEntry] = {}
    for entry in entries:
        by_document_id[entry.document_id] = entry
    return by_document_id


def _cached_latest_entries_unlocked(path: Path) -> dict[str, CollectionManifestEntry]:
    signature = _manifest_signature(path)
    cached = _PROCESS_INDEXES.get(str(path))
    if cached is not None and cached.signature == signature:
        return cached.by_document_id
    entries = _read_manifest_jsonl_entries_unlocked(path)
    by_document_id = _latest_entries(entries)
    _PROCESS_INDEXES[str(path)] = _ManifestIndex(
        signature=signature,
        by_document_id=by_document_id,
    )
    return by_document_id


def _update_cached_index_after_append(path: Path, entry: CollectionManifestEntry) -> None:
    cached = _PROCESS_INDEXES.get(str(path))
    if cached is None:
        return
    cached.by_document_id[entry.document_id] = entry
    _refresh_cached_index_signature(path, cached.by_document_id)


def _set_cached_index(
    path: Path,
    entries: Sequence[CollectionManifestEntry],
) -> None:
    _PROCESS_INDEXES[str(path)] = _ManifestIndex(
        signature=_manifest_signature(path),
        by_document_id=_latest_entries(entries),
    )


def _refresh_cached_index_signature(
    path: Path,
    by_document_id: dict[str, CollectionManifestEntry],
) -> None:
    _PROCESS_INDEXES[str(path)] = _ManifestIndex(
        signature=_manifest_signature(path),
        by_document_id=by_document_id,
    )


def _manifest_signature(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


@contextmanager
def _manifest_lock(path: Path, *, exclusive: bool):
    lock_path = path.with_name(f"{path.name}.lock")
    _reject_manifest_lock_path(lock_path)
    _ensure_private_manifest_parent(lock_path)
    process_lock = _process_lock(lock_path)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    try:
        reject_hardlinked_private_fd(fd, lock_path)
        os.fchmod(fd, 0o600)
        with process_lock:
            _lock_fd(fd, exclusive=exclusive)
            try:
                yield
            finally:
                _unlock_fd(fd)
    finally:
        os.close(fd)


def _process_lock(path: Path) -> RLock:
    key = str(path)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _lock_fd(fd: int, *, exclusive: bool) -> None:
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")

        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        getattr(msvcrt, "locking")(fd, getattr(msvcrt, "LK_LOCK"), 1)
        return

    import fcntl

    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(fd, operation)


def _unlock_fd(fd: int) -> None:
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")

        os.lseek(fd, 0, os.SEEK_SET)
        getattr(msvcrt, "locking")(fd, getattr(msvcrt, "LK_UNLCK"), 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def atomic_write_text(path: Path, content: str) -> None:
    _reject_manifest_io_paths(path)
    _ensure_private_manifest_parent(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _ensure_private_manifest_parent(path: Path) -> None:
    ensure_private_parent_dirs(path, reject_symlinks=True)


def _reject_manifest_data_path(path: Path) -> None:
    reject_symlink_ancestors(path)
    reject_symlink_path(path)
    reject_hardlinked_private_path(path)


def _reject_manifest_lock_path(path: Path) -> None:
    reject_symlink_ancestors(path)
    reject_symlink_path(path)
    reject_hardlinked_private_path(path)


def _reject_manifest_io_paths(path: Path) -> None:
    _reject_manifest_data_path(path)
    _reject_manifest_lock_path(path.with_name(f"{path.name}.lock"))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
