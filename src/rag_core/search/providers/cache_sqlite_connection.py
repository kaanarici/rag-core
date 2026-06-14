"""Per-task SQLite connection pool for cache providers.

Single-``sqlite3.Connection`` reuse across asyncio tasks corrupts state on at
least some platforms (the connection's internal cursor/state machine is not
safe to share across threads even with ``check_same_thread=False`` when calls
overlap). The cache providers must therefore hand each running asyncio task
its own connection.

This module also owns the WAL + busy-timeout pragma block so the embedding
and chunk-context caches share one canonical open path; restricted-tier
deployments that ship NO_CACHE bypass this module entirely.

WAL leaves two sidecar files (``<db>-wal`` and ``<db>-shm``) on disk that
hold pre-commit data. They must inherit the same 0600 perms and hardlink
rejection as the main DB file to avoid leaking corpus-scoped bytes through a
more permissive sidecar.
"""

from __future__ import annotations

import asyncio
import contextvars
import sqlite3
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from rag_core.private_files import (
    harden_private_file,
    prepare_private_file_for_open,
    reject_hardlinked_private_path,
)

# Carries the originating asyncio task across an ``asyncio.to_thread`` hop so
# the pool can hand the same per-task connection to the worker thread that
# the awaiting coroutine would have received. ``to_thread`` copies the
# caller's context, which copies this var as a side effect.
_running_task: contextvars.ContextVar[
    "asyncio.Task[object] | None"
] = contextvars.ContextVar("rag_core_cache_sqlite_running_task", default=None)

SQLITE_WAL_SUFFIXES: Final[tuple[str, ...]] = ("-wal", "-shm")

_DEFAULT_BUSY_TIMEOUT_MS: Final[int] = 5000


def open_sqlite_cache(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection configured for cache provider use.

    - ``check_same_thread=False`` so the per-task pool can hand the connection
      to whichever thread executes the next ``asyncio.to_thread`` call.
    - WAL + ``synchronous=NORMAL`` for concurrent read/write durability without
      the fsync cost of FULL.
    - ``busy_timeout=5000`` so a writer queued behind another writer waits
      instead of immediately raising ``sqlite3.OperationalError``.
    - WAL sidecar files (``-wal`` / ``-shm``) are hardened to 0600 and
      checked for hardlinks immediately after WAL is enabled (the files are
      only created once the journal mode flips).
    """

    db_path = Path(path)
    prepare_private_file_for_open(db_path, reject_symlink=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        harden_private_file(db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(f"PRAGMA busy_timeout={_DEFAULT_BUSY_TIMEOUT_MS}")
        harden_sqlite_sidecar_files(db_path)
    except BaseException:
        connection.close()
        raise
    return connection


def harden_sqlite_sidecar_files(db_path: Path) -> None:
    """Apply 0600 perms + reject hardlinks on the WAL/SHM sidecars.

    Called after each connection open (cheap chmod) so a sidecar that
    materialized on a later WAL checkpoint also picks up the private perms.
    """

    for suffix in SQLITE_WAL_SUFFIXES:
        sidecar = db_path.with_name(db_path.name + suffix)
        if not sidecar.exists():
            continue
        reject_hardlinked_private_path(sidecar)
        harden_private_file(sidecar)


class SqliteCacheConnectionPool:
    """Hand each asyncio task its own ``sqlite3.Connection``.

    The pool is the single owner of every connection for one cache file.
    ``connection()`` leases the connection bound to the calling task and
    keeps it open until the worker thread leaves the lease; ``close`` closes
    inactive connections and marks active leases for close on release.

    The pool is safe to call from either an event loop (uses
    ``asyncio.current_task()`` as the pool key) or from synchronous code
    (falls back to a thread-local connection key). All ``sqlite3`` calls
    should be wrapped in ``asyncio.to_thread`` by the caller so the
    blocking SQL never stalls the event loop.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._active_operations: dict[int, int] = {}
        self._release_pending: set[int] = set()
        self._task_refs: dict[int, weakref.ref[asyncio.Task[object]]] = {}
        self._armed: set[int] = set()
        self._thread_local = threading.local()
        self._closed = False

    @property
    def path(self) -> str:
        return self._path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        key, task = self._current_key_and_task()
        connection = self._acquire_connection(key=key, task=task)
        try:
            yield connection
        finally:
            self._release_operation(key)

    def _acquire_connection(
        self,
        *,
        key: int,
        task: "asyncio.Task[object] | None",
    ) -> sqlite3.Connection:
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    f"SqliteCacheConnectionPool({self._path!r}) is closed"
                )
            existing = self._connections.get(key)
            if existing is not None:
                self._active_operations[key] = self._active_operations.get(key, 0) + 1
                return existing
            connection = open_sqlite_cache(self._path)
            self._connections[key] = connection
            self._active_operations[key] = 1
            self._register_task_release(key, task)
        return connection

    def close(self) -> None:
        with self._lock:
            self._closed = True
            connections: list[sqlite3.Connection] = []
            for key, connection in list(self._connections.items()):
                if self._active_operations.get(key, 0) > 0:
                    self._release_pending.add(key)
                    continue
                connections.append(connection)
                self._connections.pop(key, None)
            self._task_refs.clear()
            self._armed.clear()
        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error:
                # Per-task connection close should be best-effort; the file
                # itself stays consistent thanks to WAL.
                pass

    # ------------------------------------------------------------------
    # internals

    def _current_key_and_task(
        self,
    ) -> tuple[int, "asyncio.Task[object] | None"]:
        """Resolve a stable per-task key for the calling context.

        Lookup order:

        1. ``asyncio.current_task()``. When ``connection`` is called directly
           inside an awaited coroutine.
        2. ``_running_task`` contextvar set by ``bind_running_task_for_pool`` so a call
           inside ``asyncio.to_thread`` still maps back to the originating
           task.
        3. Per-thread fallback for fully synchronous callers.
        """

        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None:
            task = _running_task.get()
        if task is not None:
            return id(task), task
        thread_key = getattr(self._thread_local, "key", None)
        if thread_key is None:
            thread_key = -threading.get_ident()
            self._thread_local.key = thread_key
        return int(thread_key), None

    def _arm_task_release(self, task: "asyncio.Task[object]") -> None:
        """Pre-register the deterministic release for ``task``.

        Called from the awaiting coroutine in the loop thread (via
        ``bind_running_task_for_pool``) so ``Task.add_done_callback`` runs on
        the owning loop. Idempotent: re-arming the same task for the same
        pool replaces nothing. The done_callback list is already populated.
        """

        key = id(task)
        with self._lock:
            if key in self._armed:
                return
            self._armed.add(key)
        try:
            task.add_done_callback(self._make_done_callback(key))
        except (TypeError, ValueError, RuntimeError):
            with self._lock:
                self._armed.discard(key)

    def _register_task_release(
        self,
        key: int,
        task: "asyncio.Task[object] | None",
    ) -> None:
        """Wire defensive per-task release.

        The primary, deterministic release is armed by
        ``bind_running_task_for_pool`` in the loop thread via
        ``_arm_task_release`` so ``Task.add_done_callback`` is registered
        safely on the owning loop.

        This weakref finalizer is a defensive fallback for callers that lease
        a connection without going through ``bind_running_task_for_pool``.
        ``_release`` is idempotent so both paths firing is safe.
        """

        if task is None:
            return
        try:
            self._task_refs[key] = weakref.ref(task, self._make_finalizer(key))
        except TypeError:
            # asyncio.Task is reference-able on CPython; mock task objects in
            # tests may not be. Skip the finalizer rather than fail.
            pass

    def _make_done_callback(self, key: int):
        # Invoked by the event loop when the task completes.
        def _on_done(_task: object) -> None:
            self._release(key)

        return _on_done

    def _make_finalizer(self, key: int):
        # Defensive fallback: fires when the task is garbage-collected.
        def _finalize(_ref: object) -> None:
            self._release(key)

        return _finalize

    def _release(self, key: int) -> None:
        # Idempotent by ``dict.pop`` semantics. If the task finished while a
        # worker thread still owns the connection, defer the close until that
        # operation leaves ``connection()``.
        with self._lock:
            self._armed.discard(key)
            if self._active_operations.get(key, 0) > 0:
                self._release_pending.add(key)
                self._task_refs.pop(key, None)
                return
            connection = self._connections.pop(key, None)
            self._task_refs.pop(key, None)
            self._release_pending.discard(key)
        if connection is None:
            return
        try:
            connection.close()
        except sqlite3.Error:
            pass

    def _release_operation(self, key: int) -> None:
        with self._lock:
            count = self._active_operations.get(key, 0)
            if count > 1:
                self._active_operations[key] = count - 1
                return
            self._active_operations.pop(key, None)
            if key not in self._release_pending:
                return
            self._release_pending.discard(key)
            self._armed.discard(key)
            self._task_refs.pop(key, None)
            connection = self._connections.pop(key, None)
        if connection is None:
            return
        try:
            connection.close()
        except sqlite3.Error:
            pass


def build_cache_path(
    *,
    base_dir: Path,
    processing_fingerprint: str,
    embedder_identity: str,
    corpus_id: str,
    filename: str = "cache.db",
) -> Path:
    """Compute a cache path scoped to ``(fingerprint, embedder, corpus_id)``.

    Two RAGCore processes that disagree on processing pipeline, embedder
    identity, or corpus tier must never share a cache file: a
    a cached vector for one corpus landing into another corpus' cache would
    leak bytes across the tier boundary.

    ``corpus_id`` must be the same scope value that ends up in
    ``EmbedCacheKey.corpus_id`` / ``ChunkContextKey.corpus_id``. Restricted
    tier processes also receive ``cache_disabled=True`` from their bound
    ``CorpusPolicy`` and never reach this helper.
    """

    for label, value in (
        ("base_dir", base_dir),
        ("processing_fingerprint", processing_fingerprint),
        ("embedder_identity", embedder_identity),
        ("corpus_id", corpus_id),
    ):
        if value in (None, "") or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"build_cache_path requires non-empty {label}")
    base = Path(base_dir)
    segment = (
        f"{_safe_segment(processing_fingerprint)}/"
        f"{_safe_segment(embedder_identity)}/"
        f"{_safe_segment(corpus_id)}/{filename}"
    )
    return base / segment


def _safe_segment(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in value.strip()
    )
    if not cleaned:
        raise ValueError(f"cache path segment is empty after sanitization: {value!r}")
    return cleaned


def bind_running_task_for_pool(
    pool: "SqliteCacheConnectionPool | None" = None,
) -> None:
    """Stamp the current asyncio task into the contextvar and arm release.

    Call this from the awaiting coroutine before dispatching the SQL into
    ``asyncio.to_thread``. ``to_thread`` copies the context so the worker
    thread can recover the task via ``_running_task.get()`` even though
    ``asyncio.current_task()`` returns ``None`` outside the event loop.

    When ``pool`` is supplied, this also registers a ``Task.add_done_callback``
    in the loop thread (where it is safe) so the per-task connection is
    released deterministically the moment the task completes. Without
    waiting for CPython to garbage-collect the Task object.
    """

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is None:
        return
    _running_task.set(task)
    if pool is not None:
        pool._arm_task_release(task)


__all__ = [
    "SQLITE_WAL_SUFFIXES",
    "SqliteCacheConnectionPool",
    "bind_running_task_for_pool",
    "build_cache_path",
    "harden_sqlite_sidecar_files",
    "open_sqlite_cache",
]
