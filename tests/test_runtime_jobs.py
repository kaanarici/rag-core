from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from rag_core.runtime.jobs import (
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
    IngestJobStore,
)


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
                collection TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO ingest_jobs (
                job_id, status, path, namespace, collection, result_json, error, updated_at
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


def _install_tracking_connect(
    monkeypatch: pytest.MonkeyPatch,
    open_connections: set[int],
) -> None:
    # ``sqlite3.Connection.close`` is read-only on instances, so track open
    # connections via a subclass passed as ``factory=`` rather than patching
    # the method.
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            open_connections.add(id(self))

        def close(self) -> None:
            open_connections.discard(id(self))
            super().close()

    def tracking_connect(database: Path) -> sqlite3.Connection:
        return real_connect(database, factory=TrackingConnection)

    monkeypatch.setattr("rag_core.runtime.jobs.sqlite3.connect", tracking_connect)


def test_job_store_closes_connections_after_each_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SSE status poller hits create/update/get on a sub-second cadence.
    # Each call must close its sqlite3 connection so the runtime does not leak
    # file descriptors. Track open connections and assert none stay open
    # across a burst of writes and reads.
    open_connections: set[int] = set()
    _install_tracking_connect(monkeypatch, open_connections)

    db_path = tmp_path / "jobs.sqlite3"
    store = IngestJobStore(db_path)
    assert open_connections == set()  # _init_db connection is closed

    record = store.create(path="/data/doc.md", namespace="acme", collection="help")
    for _ in range(50):
        store.update(record.job_id, status=INGEST_JOB_STATUS_COMPLETED)
        assert store.get(record.job_id) is not None

    assert open_connections == set()


def test_job_store_rolls_back_and_closes_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An exception mid-transaction must still close the connection (no leak)
    # and must not commit partial work.
    open_connections: set[int] = set()
    db_path = tmp_path / "jobs.sqlite3"
    store = IngestJobStore(db_path)
    record = store.create(path="/data/doc.md", namespace="acme", collection="help")

    real_connect = sqlite3.connect
    boom = RuntimeError("boom")

    class FailingConnection(sqlite3.Connection):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            open_connections.add(id(self))

        def execute(self, *args: object, **kwargs: object) -> sqlite3.Cursor:
            cursor = super().execute(*args, **kwargs)  # type: ignore[arg-type]
            if str(args[0]).lstrip().upper().startswith("UPDATE"):
                raise boom
            return cursor

        def close(self) -> None:
            open_connections.discard(id(self))
            super().close()

    def failing_connect(database: Path) -> sqlite3.Connection:
        return real_connect(database, factory=FailingConnection)

    monkeypatch.setattr("rag_core.runtime.jobs.sqlite3.connect", failing_connect)

    with pytest.raises(RuntimeError):
        store.update(record.job_id, status=INGEST_JOB_STATUS_COMPLETED)

    monkeypatch.undo()
    assert open_connections == set()  # connection closed despite the error

    after = store.get(record.job_id)
    assert after is not None
    assert after.status != INGEST_JOB_STATUS_COMPLETED  # rolled back, not committed


def test_job_store_closes_connection_when_pragma_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failure during row-factory / PRAGMA setup (before the first yield) must
    # still close the connection rather than leak it.
    open_connections: set[int] = set()
    real_connect = sqlite3.connect

    class PragmaFailingConnection(sqlite3.Connection):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            open_connections.add(id(self))

        def execute(self, *args: object, **kwargs: object) -> sqlite3.Cursor:
            sql = args[0] if args else ""
            if isinstance(sql, str) and sql.startswith("PRAGMA"):
                raise sqlite3.OperationalError("pragma setup boom")
            return super().execute(*args, **kwargs)  # type: ignore[arg-type]

        def close(self) -> None:
            open_connections.discard(id(self))
            super().close()

    monkeypatch.setattr(
        "rag_core.runtime.jobs.sqlite3.connect",
        lambda database: real_connect(database, factory=PragmaFailingConnection),
    )

    with pytest.raises(sqlite3.OperationalError):
        IngestJobStore(tmp_path / "jobs.db")
    assert open_connections == set()  # closed despite the PRAGMA-setup failure
