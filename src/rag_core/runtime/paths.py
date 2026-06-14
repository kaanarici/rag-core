"""Server-local path policy for the optional HTTP runtime."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Sequence
from pathlib import Path

from rag_core.runtime.errors import RuntimeRequestError


def normalize_ingest_roots(roots: Sequence[Path] | None) -> tuple[Path, ...]:
    candidates = tuple(roots) if roots else (Path.cwd(),)
    return tuple(_normalize_root(root) for root in candidates)


def validate_ingest_path(path: str, *, roots: Sequence[Path]) -> Path:
    """Resolve and harden an ingest path against the configured roots.

    The hardening is paranoid by intent: this is an sensitive sidecar and
    the gateway is the *only* layer that should be expanding user-relative
    paths. The runtime requires:

    - The caller-supplied string parses to an absolute path. We never call
      ``expanduser``, so ``~`` or ``$VAR`` never resolve here; if the gateway
      wants a home-relative ingest it must expand and forward an absolute
      path.
    - No path component is a symbolic link (the resolved file, every parent
      up to its drive root, and the target itself). Symlinks could otherwise
      smuggle reads out of the configured ingest roots.
    - The post-resolve target is a real file (not a directory, not a
      socket/fifo, not a dangling path).
    - The target is contained in one of ``roots`` (already pre-resolved by
      :func:`normalize_ingest_roots`).
    """
    raw = Path(path)
    if not raw.is_absolute():
        raise RuntimeRequestError(
            message="path must be absolute",
            details={"field": "path"},
        )
    # Reject the literal ``~``/env-style prefixes up front. They would parse
    # as absolute on Windows but as a single relative component on POSIX, and
    # in both cases the caller is asking the runtime to expand them.
    if path.startswith("~") or path.lstrip().startswith("$"):
        raise RuntimeRequestError(
            message="path must be absolute",
            details={"field": "path"},
        )
    # ``strict=False`` so we get a normalized form even if the file is missing;
    # we then check existence / file-ness explicitly so the error message is
    # the one callers expect.
    resolved = raw.resolve(strict=False)
    _reject_symlink_in_chain(raw, resolved)
    for root in roots:
        if resolved == root or root in resolved.parents:
            if not resolved.exists():
                raise RuntimeRequestError(
                    message="path does not exist",
                    details={"field": "path"},
                )
            if not resolved.is_file():
                raise RuntimeRequestError(
                    message="path must be a regular file",
                    details={"field": "path"},
                )
            return resolved
    raise RuntimeRequestError(
        message="path is outside configured ingest roots",
        details={
            "field": "path",
            "allowed_roots": [str(root) for root in roots],
        },
    )


def read_validated_ingest_file(path: str, *, roots: Sequence[Path]) -> tuple[Path, bytes]:
    resolved = validate_ingest_path(path, roots=roots)
    before = _stat_regular_file(resolved)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        if hasattr(errno, "ELOOP") and exc.errno == errno.ELOOP:
            raise RuntimeRequestError(
                message="path must not traverse a symbolic link",
                details={"field": "path", "resolved": str(resolved)},
            ) from exc
        raise RuntimeRequestError(
            message="path could not be opened",
            details={"field": "path"},
        ) from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeRequestError(
                message="path must be a regular file",
                details={"field": "path"},
            )
        if not _same_opened_file(before, opened):
            raise RuntimeRequestError(
                message="path changed during validation",
                details={"field": "path"},
            )
        post_resolved = resolved.resolve(strict=False)
        _reject_symlink_in_chain(resolved, post_resolved)
        if not _is_within_any_root(post_resolved, roots):
            raise RuntimeRequestError(
                message="path is outside configured ingest roots",
                details={
                    "field": "path",
                    "allowed_roots": [str(root) for root in roots],
                },
            )
        after = _stat_regular_file(resolved)
        if not _same_opened_file(after, opened):
            raise RuntimeRequestError(
                message="path changed during validation",
                details={"field": "path"},
            )
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1
            return resolved, handle.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _normalize_root(path: Path) -> Path:
    # Roots are server-operator-supplied (CLI flag); we do allow expanduser
    # here because the operator typed them at process start, not over the
    # network. Resolve follows symlinks once so containment checks compare
    # canonical paths.
    return path.expanduser().resolve(strict=False)


def _stat_regular_file(path: Path) -> os.stat_result:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeRequestError(
            message="path could not be inspected",
            details={"field": "path"},
        ) from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeRequestError(
            message="path must be a regular file",
            details={"field": "path"},
        )
    if os.name != "nt" and path_stat.st_nlink > 1:
        raise RuntimeRequestError(
            message="path must not be a multi-link file",
            details={"field": "path"},
        )
    return path_stat


def _same_opened_file(left: os.stat_result, right: os.stat_result) -> bool:
    if os.name == "nt":
        return (
            left.st_size == right.st_size
            and left.st_mtime_ns == right.st_mtime_ns
        )
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _is_within_any_root(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _reject_symlink_in_chain(requested: Path, resolved: Path) -> None:
    """Refuse if the requested or resolved path traverses a symlink.

    Walks ``requested`` (the as-typed path) AND ``resolved`` (its canonical
    form) up to the filesystem root and rejects if any component is a
    symbolic link.  Doing both catches the case where the leaf is itself a
    symlink (``is_symlink`` on the original) and the case where a parent
    directory is a link (``is_symlink`` on any ancestor of the realpath).
    """
    for candidate in (requested, resolved):
        try:
            if candidate.is_symlink():
                raise RuntimeRequestError(
                    message=(
                        "path must not traverse a symbolic link; pass the "
                        "fully resolved path (note: /tmp and /var are "
                        "symlinks on macOS)"
                    ),
                    details={"field": "path", "resolved": str(resolved)},
                )
        except OSError:
            # ``is_symlink`` only raises on broken stat in narrow cases; treat
            # those as a refusal rather than letting them surface as 500.
            raise RuntimeRequestError(
                message="path could not be inspected",
                details={"field": "path"},
            ) from None
        for ancestor in candidate.parents:
            try:
                if ancestor.is_symlink():
                    raise RuntimeRequestError(
                        message=(
                            "path must not traverse a symbolic link; pass the "
                            "fully resolved path (note: /tmp and /var are "
                            "symlinks on macOS)"
                        ),
                        details={"field": "path", "resolved": str(resolved)},
                    )
            except OSError:
                raise RuntimeRequestError(
                    message="path could not be inspected",
                    details={"field": "path"},
                ) from None


__all__ = [
    "normalize_ingest_roots",
    "read_validated_ingest_file",
    "validate_ingest_path",
]
