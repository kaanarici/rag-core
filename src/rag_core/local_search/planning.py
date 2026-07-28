from __future__ import annotations

import os
from dataclasses import dataclass
from os import stat_result
from pathlib import Path

from rag_core.ingest.sources.local import (
    document_key,
    is_ignored_local_file,
    is_supported_local_candidate,
    reject_local_symlink_path,
)
from rag_core.local_search.models import LocalSearchFileSet, LocalSearchRequest


@dataclass(frozen=True)
class LocalSearchDocumentSpec:
    path: Path
    document_key: str


@dataclass(frozen=True)
class LocalSearchRunSpec:
    root: Path
    query: str
    namespace: str
    collection: str
    limit: int
    documents: list[LocalSearchDocumentSpec]
    skipped_unsupported_count: int
    skipped_empty_count: int
    truncated: bool


def build_local_search_run_spec(request: LocalSearchRequest) -> LocalSearchRunSpec:
    root = _validated_local_search_root(request)
    document_key_root = root.parent if root.is_file() else root
    fileset = discover_local_files(root, max_files=request.max_files)
    if not fileset.files:
        if fileset.skipped_empty_count:
            raise ValueError("only empty supported files found under %s" % root)
        raise ValueError("no supported files found under %s" % root)
    return LocalSearchRunSpec(
        root=root,
        query=request.query,
        namespace=request.namespace,
        collection=request.collection or default_collection(root),
        limit=request.limit,
        documents=[
            LocalSearchDocumentSpec(
                path=file_path, document_key=document_key(document_key_root, file_path)
            )
            for file_path in fileset.files
        ],
        skipped_unsupported_count=fileset.skipped_unsupported_count,
        skipped_empty_count=fileset.skipped_empty_count,
        truncated=fileset.truncated,
    )


def discover_local_files(root: Path, *, max_files: int) -> LocalSearchFileSet:
    escaped_count = 0
    if root.is_file():
        candidates = [root]
    else:
        candidates, escaped_count = _safe_directory_candidates(root)
    supported: list[Path] = []
    seen_physical_files: set[tuple[int, int]] = set()
    skipped_unsupported_count = escaped_count
    skipped_empty_count = 0
    for path in candidates:
        if is_ignored_local_file(path):
            continue
        path_stat = _file_stat(path)
        if path_stat is None:
            skipped_unsupported_count += 1
            continue
        if _is_empty_package_sentinel(path, path_stat):
            continue
        if path_stat.st_size == 0:
            if is_supported_local_candidate(path):
                skipped_empty_count += 1
            else:
                skipped_unsupported_count += 1
            continue
        if is_supported_local_candidate(path):
            physical_key = _physical_file_key(path_stat)
            if physical_key is not None and physical_key in seen_physical_files:
                skipped_unsupported_count += 1
                continue
            if physical_key is not None:
                seen_physical_files.add(physical_key)
            supported.append(path)
        else:
            skipped_unsupported_count += 1
    truncated = len(supported) > max_files
    return LocalSearchFileSet(
        files=supported[:max_files],
        skipped_unsupported_count=skipped_unsupported_count,
        skipped_empty_count=skipped_empty_count,
        truncated=truncated,
    )


def default_collection(root: Path) -> str:
    name = root.stem if root.is_file() else root.name
    return name or "local-collection"


def _is_empty_package_sentinel(path: Path, path_stat: stat_result) -> bool:
    if path.name != "__init__.py":
        return False
    if path_stat.st_size == 0:
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return text.strip() == ""


def _file_stat(path: Path) -> stat_result | None:
    try:
        return path.stat(follow_symlinks=False)
    except OSError:
        return None


def _physical_file_key(path_stat: stat_result) -> tuple[int, int] | None:
    inode = getattr(path_stat, "st_ino", 0)
    device = getattr(path_stat, "st_dev", 0)
    if not isinstance(inode, int) or not isinstance(device, int) or inode == 0:
        return None
    return device, inode


def _validated_local_search_root(request: LocalSearchRequest) -> Path:
    root = request.path
    if not root.exists():
        raise FileNotFoundError(str(root))
    if request.limit <= 0:
        raise ValueError("limit must be positive")
    if request.max_files <= 0:
        raise ValueError("max-files must be positive")
    reject_local_symlink_path(root)
    return root


def _safe_directory_candidates(root: Path) -> tuple[list[Path], int]:
    resolved_root = root.resolve(strict=False)
    candidates: list[Path] = []
    escaped_count = 0
    for current_root, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            if _path_within_root(current / name, resolved_root):
                kept_dirs.append(name)
            else:
                escaped_count += 1
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = current / name
            if path.is_symlink():
                escaped_count += 1
                continue
            if not _path_within_root(path, resolved_root):
                escaped_count += 1
                continue
            if path.is_file():
                candidates.append(path)
    return sorted(candidates), escaped_count


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        return path.resolve(strict=False).is_relative_to(root)
    except OSError:
        return False


__all__ = [
    "LocalSearchDocumentSpec",
    "LocalSearchRunSpec",
    "build_local_search_run_spec",
    "default_collection",
    "discover_local_files",
]
