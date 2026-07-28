"""SQLite-backed ingest job records for the optional HTTP runtime.

Job ``error`` bodies are intentionally sanitized: only ``error_type`` (the
exception class name) and ``error_code`` (a stable machine label such as
``ingest_failed``) are persisted. Free-form ``str(exc)`` text, which can
include SDK error strings or licensed-source identifiers, must travel via
the event sink, never the public job body.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

JobStatus = Literal["pending", "running", "completed", "failed"]

INGEST_JOB_STATUS_PENDING: Final[JobStatus] = "pending"
INGEST_JOB_STATUS_RUNNING: Final[JobStatus] = "running"
INGEST_JOB_STATUS_COMPLETED: Final[JobStatus] = "completed"
INGEST_JOB_STATUS_FAILED: Final[JobStatus] = "failed"

INGEST_JOB_STATUSES: Final[tuple[JobStatus, ...]] = (
    INGEST_JOB_STATUS_PENDING,
    INGEST_JOB_STATUS_RUNNING,
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
)
INGEST_JOB_TERMINAL_STATUSES: Final[frozenset[JobStatus]] = frozenset(
    (INGEST_JOB_STATUS_COMPLETED, INGEST_JOB_STATUS_FAILED)
)

# Stable labels for the lifespan-startup orphan sweep. ``error_type`` mirrors
# the sanitized shape ``IngestJobStore.update`` uses elsewhere; ``error_code``
# is the machine-readable reason that gateways and pollers match on.
ORPHANED_ERROR_TYPE: Final[str] = "OrphanedByRestart"
ORPHANED_BY_RESTART_REASON: Final[str] = "orphaned_by_restart"
_TERMINAL_JOB_STATUSES: Final[tuple[JobStatus, JobStatus]] = (
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
)


@dataclass(frozen=True)
class IngestJobRecord:
    job_id: str
    status: JobStatus
    path: str
    namespace: str
    collection: str
    result: dict[str, object] | None = None
    # Sanitized: ``{"error_type": <ClassName>, "error_code": <stable-code>}``.
    # Never carries raw ``str(exc)``.
    error: dict[str, str] | None = None


class IngestJobStore:
    def __init__(
        self,
        db_path: Path,
        *,
        max_age_seconds: float | None = None,
        max_terminal_rows: int | None = None,
    ) -> None:
        if max_age_seconds is not None and (
            max_age_seconds <= 0 or not math.isfinite(max_age_seconds)
        ):
            raise ValueError("max_age_seconds must be positive or None")
        if max_terminal_rows is not None and max_terminal_rows <= 0:
            raise ValueError("max_terminal_rows must be positive or None")
        self._db_path = db_path
        self._max_age_seconds = max_age_seconds
        self._max_terminal_rows = max_terminal_rows
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # WAL + busy_timeout so the per-call connection pattern stays safe
        # under bursts of concurrent ingest webhooks. ``check_same_thread`` is
        # left at the default because every call opens its own connection. The
        # connection is always closed on exit; a bare ``with sqlite3.Connection``
        # commits/rolls back but never closes, which leaks file descriptors
        # under the SSE status-poll hot path.
        connection = sqlite3.connect(self._db_path)
        # Enter the try immediately after connect so a failure in row-factory or
        # PRAGMA setup still closes the connection instead of leaking it.
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    path TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            columns = frozenset(
                str(row[1])
                for row in connection.execute("PRAGMA table_info(ingest_jobs)")
            )
            now = time.time()
            if "updated_at" not in columns:
                connection.execute(
                    "ALTER TABLE ingest_jobs ADD COLUMN updated_at REAL NOT NULL "
                    f"DEFAULT {now}"
                )
            self._migrate_legacy_error_rows(connection)
            self._prune_terminal_jobs(connection, now=now)

    def create(self, *, path: str, namespace: str, collection: str) -> IngestJobRecord:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_jobs (
                    job_id, status, path, namespace, collection, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, INGEST_JOB_STATUS_PENDING, path, namespace, collection, now),
            )
        return IngestJobRecord(
            job_id=job_id,
            status=INGEST_JOB_STATUS_PENDING,
            path=path,
            namespace=namespace,
            collection=collection,
        )

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus,
        result: dict[str, object] | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
    ) -> None:
        # ``error_type`` and ``error_code`` together encode the sanitized
        # failure surface. Callers must not pass formatted ``str(exc)`` text.
        if (error_type is None) != (error_code is None):
            raise ValueError(
                "IngestJobStore.update requires error_type and error_code together"
            )
        error_payload: str | None
        if error_type is None or error_code is None:
            error_payload = None
        else:
            error_payload = _error_payload_json(
                error_type=error_type,
                error_code=error_code,
            )
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, result_json = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    json.dumps(result) if result is not None else None,
                    error_payload,
                    now,
                    job_id,
                ),
            )
            if status in _TERMINAL_JOB_STATUSES:
                self._prune_terminal_jobs(connection, now=now)

    def reconcile_orphaned_jobs(
        self,
        *,
        reason: str = ORPHANED_BY_RESTART_REASON,
    ) -> tuple[str, ...]:
        """Flip any ``pending`` or ``running`` rows to ``failed``.

        Called from the HTTP runtime ``lifespan`` startup hook. ``rag-core``
        does **not** resume in-flight ingest jobs across process restarts.
        That orchestration responsibility lives with the gateway. Any row
        that was mid-flight when the previous process died is therefore
        orphaned at startup: we flip it to ``failed`` with a sanitized
        ``error_type=OrphanedByRestart`` and the supplied stable ``error_code``
        so the gateway / poller sees a terminal status and can decide to retry.

        Returns the tuple of job ids that were flipped (empty if none).
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id FROM ingest_jobs
                WHERE status IN (?, ?)
                """,
                (INGEST_JOB_STATUS_PENDING, INGEST_JOB_STATUS_RUNNING),
            ).fetchall()
            orphaned = tuple(str(row["job_id"]) for row in rows)
            if not orphaned:
                return orphaned
            payload = _error_payload_json(
                error_type=ORPHANED_ERROR_TYPE,
                error_code=reason,
            )
            placeholders = ",".join("?" * len(orphaned))
            now = time.time()
            connection.execute(
                f"""
                UPDATE ingest_jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE job_id IN ({placeholders})
                """,
                (INGEST_JOB_STATUS_FAILED, payload, now, *orphaned),
            )
            self._prune_terminal_jobs(connection, now=now)
        return orphaned

    def get(self, job_id: str) -> IngestJobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        result_json = row["result_json"]
        result = json.loads(result_json) if isinstance(result_json, str) else None
        status = parse_job_status(str(row["status"]))
        error_payload = _decode_error_payload(row["error"])
        return IngestJobRecord(
            job_id=str(row["job_id"]),
            status=status,
            path=str(row["path"]),
            namespace=str(row["namespace"]),
            collection=str(row["collection"]),
            result=result if isinstance(result, dict) else None,
            error=error_payload,
        )

    def _migrate_legacy_error_rows(self, connection: sqlite3.Connection) -> None:
        legacy_payload = _error_payload_json(error_type="Unknown", error_code="legacy")
        rows = connection.execute(
            """
            SELECT job_id, error FROM ingest_jobs
            WHERE error IS NOT NULL AND error != ''
            """
        ).fetchall()
        for row in rows:
            raw_error = row["error"]
            if isinstance(raw_error, str) and _is_sanitized_error_payload(raw_error):
                continue
            connection.execute(
                "UPDATE ingest_jobs SET error = ? WHERE job_id = ?",
                (legacy_payload, str(row["job_id"])),
            )

    def _prune_terminal_jobs(
        self,
        connection: sqlite3.Connection,
        *,
        now: float | None = None,
    ) -> None:
        if self._max_age_seconds is None and self._max_terminal_rows is None:
            return
        if self._max_age_seconds is not None:
            connection.execute(
                """
                DELETE FROM ingest_jobs
                WHERE status IN (?, ?) AND updated_at < ?
                """,
                (
                    *_TERMINAL_JOB_STATUSES,
                    (time.time() if now is None else now) - self._max_age_seconds,
                ),
            )
        if self._max_terminal_rows is not None:
            connection.execute(
                """
                DELETE FROM ingest_jobs
                WHERE job_id IN (
                    SELECT job_id FROM ingest_jobs
                    WHERE status IN (?, ?)
                    ORDER BY updated_at DESC, job_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (*_TERMINAL_JOB_STATUSES, self._max_terminal_rows),
            )


def parse_job_status(value: str) -> JobStatus:
    if value not in INGEST_JOB_STATUSES:
        raise ValueError(f"unexpected ingest job status: {value}")
    return value


def is_terminal_job_status(status: JobStatus) -> bool:
    return status in INGEST_JOB_TERMINAL_STATUSES


def ingest_job_status_payload(record: IngestJobRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "job_id": record.job_id,
        "status": record.status,
        "path": record.path,
        "namespace": record.namespace,
        "collection": record.collection,
    }
    if record.result is not None:
        payload["result"] = record.result
    if record.error is not None:
        payload["error"] = record.error
    return payload


def _error_payload_json(*, error_type: str, error_code: str) -> str:
    return json.dumps({"error_type": error_type, "error_code": error_code})


def _decode_error_payload(raw_error: object) -> dict[str, str] | None:
    if not isinstance(raw_error, str) or not raw_error:
        return None
    try:
        decoded = json.loads(raw_error)
    except ValueError as exc:
        raise ValueError("ingest job error payload is not sanitized JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("ingest job error payload is not sanitized JSON")
    if "error_type" not in decoded or "error_code" not in decoded:
        raise ValueError("ingest job error payload is missing sanitized fields")
    return {
        "error_type": str(decoded["error_type"]),
        "error_code": str(decoded["error_code"]),
    }


def _is_sanitized_error_payload(raw_error: str) -> bool:
    try:
        _decode_error_payload(raw_error)
    except ValueError:
        return False
    return True
