"""SQLite-backed ingest job records for the optional HTTP runtime."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JobStatus = Literal["pending", "running", "completed", "failed"]


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
                VALUES (?, 'pending', ?, ?, ?)
                """,
                (job_id, path, namespace, corpus_id),
            )
        return IngestJobRecord(
            job_id=job_id,
            status="pending",
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
        status = row["status"]
        if status not in {"pending", "running", "completed", "failed"}:
            raise ValueError(f"unexpected ingest job status: {status}")
        return IngestJobRecord(
            job_id=str(row["job_id"]),
            status=status,
            path=str(row["path"]),
            namespace=str(row["namespace"]),
            corpus_id=str(row["corpus_id"]),
            result=result if isinstance(result, dict) else None,
            error=row["error"],
        )
