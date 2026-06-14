from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from rag_core.runtime.jobs import INGEST_JOB_STATUS_FAILED, IngestJobStore


def test_job_store_migrates_legacy_raw_error_rows_on_init(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    raw_error = "provider failed for /private/source.md token=sk-test-secret"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE ingest_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                path TEXT NOT NULL,
                namespace TEXT NOT NULL,
                corpus_id TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO ingest_jobs (
                job_id, status, path, namespace, corpus_id, result_json, error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                INGEST_JOB_STATUS_FAILED,
                "/private/source.md",
                "acme",
                "help",
                None,
                raw_error,
                time.time(),
            ),
        )

    store = IngestJobStore(db_path)

    with sqlite3.connect(db_path) as connection:
        stored_error = connection.execute(
            "SELECT error FROM ingest_jobs WHERE job_id = ?",
            ("job-1",),
        ).fetchone()[0]

    assert raw_error not in stored_error
    assert "sk-test-secret" not in stored_error
    assert json.loads(stored_error) == {
        "error_type": "Unknown",
        "error_code": "legacy",
    }
    record = store.get("job-1")
    assert record is not None
    assert record.error == {"error_type": "Unknown", "error_code": "legacy"}
