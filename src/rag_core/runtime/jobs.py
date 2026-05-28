"""SQLite-backed ingest job records for the optional HTTP runtime."""

from __future__ import annotations

import json
import sqlite3
import uuid
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


@dataclass(frozen=True)
class IngestJobRecord:
    job_id: str
    status: JobStatus
    path: str
    namespace: str
    corpus_id: str
    result: dict[str, object] | None = None
    error: str | None = None


class IngestJobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    path TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    corpus_id TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT
                )
                """
            )

    def create(self, *, path: str, namespace: str, corpus_id: str) -> IngestJobRecord:
        job_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_jobs (job_id, status, path, namespace, corpus_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, INGEST_JOB_STATUS_PENDING, path, namespace, corpus_id),
            )
        return IngestJobRecord(
            job_id=job_id,
            status=INGEST_JOB_STATUS_PENDING,
            path=path,
            namespace=namespace,
            corpus_id=corpus_id,
        )

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus,
        result: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, result_json = ?, error = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    json.dumps(result) if result is not None else None,
                    error,
                    job_id,
                ),
            )

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
        return IngestJobRecord(
            job_id=str(row["job_id"]),
            status=status,
            path=str(row["path"]),
            namespace=str(row["namespace"]),
            corpus_id=str(row["corpus_id"]),
            result=result if isinstance(result, dict) else None,
            error=row["error"],
        )


def parse_job_status(value: str) -> JobStatus:
    if value not in INGEST_JOB_STATUSES:
        raise ValueError(f"unexpected ingest job status: {value}")
    return value
