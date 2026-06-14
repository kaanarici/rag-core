"""Lifespan-startup orphan ingest-job sweep.

``rag-core`` does not resume in-flight ingest jobs across restarts. Any row
that was ``pending`` or ``running`` when the prior process died is flipped
to ``failed`` on the next startup with a sanitized
``error_type=OrphanedByRestart`` payload so the gateway / poller sees a
terminal status. Retry orchestration lives in the gateway, not here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.core_models import RAGCoreConfig
from rag_core.runtime.jobs import (
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
    INGEST_JOB_STATUS_PENDING,
    INGEST_JOB_STATUS_RUNNING,
    IngestJobStore,
    ORPHANED_BY_RESTART_REASON,
    ORPHANED_ERROR_TYPE,
)

pytestmark = [pytest.mark.integration]


def _set_job_updated_at(db_path: Path, job_id: str, updated_at: float) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE ingest_jobs SET updated_at = ? WHERE job_id = ?",
            (updated_at, job_id),
        )


@pytest.mark.parametrize("max_age_seconds", [0.0, -1.0, float("nan")])
def test_job_store_rejects_invalid_max_age_seconds(
    tmp_path: Path,
    max_age_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="max_age_seconds"):
        IngestJobStore(
            tmp_path / "jobs.sqlite3",
            max_age_seconds=max_age_seconds,
        )


@pytest.mark.parametrize("max_terminal_rows", [0, -1])
def test_job_store_rejects_invalid_max_terminal_rows(
    tmp_path: Path,
    max_terminal_rows: int,
) -> None:
    with pytest.raises(ValueError, match="max_terminal_rows"):
        IngestJobStore(
            tmp_path / "jobs.sqlite3",
            max_terminal_rows=max_terminal_rows,
        )


def test_job_retention_default_keeps_terminal_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    seed = IngestJobStore(db_path)
    completed = seed.create(path="/tmp/a.md", namespace="acme", corpus_id="public")
    seed.update(
        completed.job_id,
        status=INGEST_JOB_STATUS_COMPLETED,
        result={"document_id": "doc-a"},
    )
    _set_job_updated_at(db_path, completed.job_id, 1.0)

    reopened = IngestJobStore(db_path)

    record = reopened.get(completed.job_id)
    assert record is not None
    assert record.status == INGEST_JOB_STATUS_COMPLETED
    assert record.result == {"document_id": "doc-a"}


def test_job_retention_expires_terminal_rows_after_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    seed = IngestJobStore(db_path)
    old = seed.create(path="/tmp/old.md", namespace="acme", corpus_id="public")
    seed.update(old.job_id, status=INGEST_JOB_STATUS_COMPLETED)
    fresh = seed.create(path="/tmp/fresh.md", namespace="acme", corpus_id="public")
    seed.update(fresh.job_id, status=INGEST_JOB_STATUS_COMPLETED)
    _set_job_updated_at(db_path, old.job_id, 1.0)
    _set_job_updated_at(db_path, fresh.job_id, 20.0)

    monkeypatch.setattr("rag_core.runtime.jobs.time.time", lambda: 20.0)
    reopened = IngestJobStore(db_path, max_age_seconds=10)

    assert reopened.get(old.job_id) is None
    assert reopened.get(fresh.job_id) is not None


def test_job_retention_never_prunes_pending_or_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    seed = IngestJobStore(db_path)
    pending = seed.create(path="/tmp/pending.md", namespace="acme", corpus_id="public")
    running = seed.create(path="/tmp/running.md", namespace="acme", corpus_id="public")
    seed.update(running.job_id, status=INGEST_JOB_STATUS_RUNNING)
    completed = seed.create(path="/tmp/done.md", namespace="acme", corpus_id="public")
    seed.update(completed.job_id, status=INGEST_JOB_STATUS_COMPLETED)
    for job in (pending, running, completed):
        _set_job_updated_at(db_path, job.job_id, 1.0)

    monkeypatch.setattr("rag_core.runtime.jobs.time.time", lambda: 20.0)
    reopened = IngestJobStore(db_path, max_age_seconds=10)

    assert reopened.get(pending.job_id) is not None
    assert reopened.get(running.job_id) is not None
    assert reopened.get(completed.job_id) is None


def test_job_retention_trims_oldest_terminal_rows_on_terminal_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
    monkeypatch.setattr("rag_core.runtime.jobs.time.time", lambda: next(timestamps))
    store = IngestJobStore(tmp_path / "jobs.sqlite3", max_terminal_rows=2)

    first = store.create(path="/tmp/a.md", namespace="acme", corpus_id="public")
    store.update(first.job_id, status=INGEST_JOB_STATUS_COMPLETED)
    second = store.create(path="/tmp/b.md", namespace="acme", corpus_id="public")
    store.update(second.job_id, status=INGEST_JOB_STATUS_COMPLETED)
    third = store.create(path="/tmp/c.md", namespace="acme", corpus_id="public")
    store.update(third.job_id, status=INGEST_JOB_STATUS_COMPLETED)

    assert store.get(first.job_id) is None
    assert store.get(second.job_id) is not None
    assert store.get(third.job_id) is not None


def test_reconcile_orphaned_jobs_flips_running_and_pending(tmp_path: Path) -> None:
    store = IngestJobStore(tmp_path / "jobs.sqlite3")
    pending = store.create(path="/tmp/a.md", namespace="acme", corpus_id="public")
    running = store.create(path="/tmp/b.md", namespace="acme", corpus_id="public")
    store.update(running.job_id, status=INGEST_JOB_STATUS_RUNNING)
    completed = store.create(path="/tmp/c.md", namespace="acme", corpus_id="public")
    store.update(
        completed.job_id,
        status=INGEST_JOB_STATUS_COMPLETED,
        result={"document_id": "doc-c"},
    )
    pre_failed = store.create(path="/tmp/d.md", namespace="acme", corpus_id="public")
    store.update(
        pre_failed.job_id,
        status=INGEST_JOB_STATUS_FAILED,
        error_type="ProviderError",
        error_code="ingest_failed",
    )

    orphaned = store.reconcile_orphaned_jobs()

    # Only the pending + running rows are touched. Order isn't guaranteed.
    assert set(orphaned) == {pending.job_id, running.job_id}

    refreshed_pending = store.get(pending.job_id)
    refreshed_running = store.get(running.job_id)
    refreshed_completed = store.get(completed.job_id)
    refreshed_failed = store.get(pre_failed.job_id)
    assert refreshed_pending is not None
    assert refreshed_running is not None
    assert refreshed_completed is not None
    assert refreshed_failed is not None

    for record in (refreshed_pending, refreshed_running):
        assert record.status == INGEST_JOB_STATUS_FAILED
        assert record.error == {
            "error_type": ORPHANED_ERROR_TYPE,
            "error_code": ORPHANED_BY_RESTART_REASON,
        }

    # Terminal rows are preserved as-is.
    assert refreshed_completed.status == INGEST_JOB_STATUS_COMPLETED
    assert refreshed_completed.result == {"document_id": "doc-c"}
    assert refreshed_failed.status == INGEST_JOB_STATUS_FAILED
    assert refreshed_failed.error == {
        "error_type": "ProviderError",
        "error_code": "ingest_failed",
    }


def test_reconcile_orphaned_jobs_is_no_op_when_clean(tmp_path: Path) -> None:
    store = IngestJobStore(tmp_path / "jobs.sqlite3")
    done = store.create(path="/tmp/a.md", namespace="acme", corpus_id="public")
    store.update(
        done.job_id,
        status=INGEST_JOB_STATUS_COMPLETED,
        result={"document_id": "doc-a"},
    )

    orphaned = store.reconcile_orphaned_jobs()

    assert orphaned == ()
    record = store.get(done.job_id)
    assert record is not None
    assert record.status == INGEST_JOB_STATUS_COMPLETED


def test_reconcile_orphaned_jobs_custom_reason_stamped(tmp_path: Path) -> None:
    store = IngestJobStore(tmp_path / "jobs.sqlite3")
    running = store.create(path="/tmp/a.md", namespace="acme", corpus_id="public")
    store.update(running.job_id, status=INGEST_JOB_STATUS_RUNNING)

    store.reconcile_orphaned_jobs(reason="orphaned_by_custom_signal")

    record = store.get(running.job_id)
    assert record is not None
    assert record.error == {
        "error_type": ORPHANED_ERROR_TYPE,
        "error_code": "orphaned_by_custom_signal",
    }


def test_runtime_lifespan_runs_orphan_sweep(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    from rag_core.core import RAGCore
    from rag_core.demo import DemoEmbeddingProvider, DemoSparseEmbedder
    from rag_core.runtime.app import create_app
    from rag_core.search.providers.memory_store import InMemoryVectorStore

    db_path = tmp_path / "jobs.sqlite3"

    # Seed two pre-existing rows that simulate the prior process dying mid-flight.
    seed = IngestJobStore(db_path)
    leftover_running = seed.create(
        path="/tmp/a.md", namespace="acme", corpus_id="public",
    )
    seed.update(leftover_running.job_id, status=INGEST_JOB_STATUS_RUNNING)
    leftover_pending = seed.create(
        path="/tmp/b.md", namespace="acme", corpus_id="public",
    )
    assert leftover_pending.status == INGEST_JOB_STATUS_PENDING

    config = RAGCoreConfig(
        qdrant=QdrantConfig(location=":memory:"),
        embedding=EmbeddingConfig(
            provider="demo",
            model="demo-dense-v1",
            dimensions=4,
        ),
    )
    core = RAGCore(
        config,
        embedding_provider=DemoEmbeddingProvider(dimensions=4),
        sparse_embedder=DemoSparseEmbedder(),
        vector_store=InMemoryVectorStore(),
    )
    app = create_app(
        config=config,
        core_factory=lambda _cfg: core,
        job_db_path=db_path,
        ingest_roots=(tmp_path,),
    )

    # Lifespan startup runs when the TestClient context is entered.
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

    refreshed = IngestJobStore(db_path)
    after_running = refreshed.get(leftover_running.job_id)
    after_pending = refreshed.get(leftover_pending.job_id)
    assert after_running is not None
    assert after_pending is not None
    for record in (after_running, after_pending):
        assert record.status == INGEST_JOB_STATUS_FAILED
        assert record.error == {
            "error_type": ORPHANED_ERROR_TYPE,
            "error_code": ORPHANED_BY_RESTART_REASON,
        }
