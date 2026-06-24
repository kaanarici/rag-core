from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

from rag_core.core_models import Config
from rag_core.runtime import job_events as job_event_module
from rag_core.runtime.app import create_app
from rag_core.runtime.jobs import (
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_RUNNING,
    IngestJobRecord,
    IngestJobStore,
)

pytestmark = [pytest.mark.integration]


def test_runtime_ingest_events_unknown_job_returns_api_error(tmp_path: Path) -> None:
    app = create_app(
        config=Config.local(),
        core_factory=lambda _: cast(Any, object()),
        job_db_path=tmp_path / "jobs.sqlite3",
        ingest_roots=(tmp_path,),
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/ingest/does-not-exist/events",
            headers={"X-Request-Id": "stream-missing"},
        )

    assert response.status_code == 404
    assert response.headers["x-request-id"] == "stream-missing"
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["details"]["job_id"] == "does-not-exist"


def test_runtime_ingest_events_stream_status_changes_once_and_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(job_event_module, "_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(job_event_module, "_HEARTBEAT_INTERVAL_SECONDS", 0.02)

    asyncio.run(_assert_ingest_event_stream(tmp_path))


def test_runtime_ingest_events_emit_admission_record_if_terminal_row_is_pruned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_db_path = tmp_path / "jobs.sqlite3"
    doc = tmp_path / "stream.md"
    doc.write_text("streaming status transitions\n", encoding="utf-8")
    store = IngestJobStore(job_db_path)
    record = store.create(path=str(doc), namespace="acme", collection="help")
    store.update(
        record.job_id,
        status=INGEST_JOB_STATUS_COMPLETED,
        result={"document_id": "stream-doc", "chunk_count": 1},
    )
    admitted = store.get(record.job_id)
    assert admitted is not None

    original_get = IngestJobStore.get
    calls = 0

    def pruned_after_admission(
        self: IngestJobStore,
        job_id: str,
    ) -> IngestJobRecord | None:
        nonlocal calls
        if job_id == record.job_id:
            calls += 1
            if calls == 1:
                return admitted
            return None
        return original_get(self, job_id)

    monkeypatch.setattr(IngestJobStore, "get", pruned_after_admission)

    async def scenario() -> None:
        app = create_app(
            config=Config.local(),
            core_factory=lambda _: cast(Any, object()),
            job_db_path=job_db_path,
            ingest_roots=(tmp_path,),
        )
        stream_messages, disconnect, stream_task = _start_asgi_stream(
            app,
            f"/v1/ingest/{record.job_id}/events",
        )
        start = await _next_asgi_message(stream_messages)
        assert start["type"] == "http.response.start"
        assert start["status"] == 200

        frame = await _next_body_text(stream_messages)
        assert frame.startswith("event: status\ndata: ")
        assert frame.endswith("\n\n")
        completed = _status_payload_from_sse(frame)
        assert completed["status"] == "completed"
        assert completed["result"] == {"document_id": "stream-doc", "chunk_count": 1}
        assert await _next_body_text(stream_messages) == ""

        disconnect.set()
        await asyncio.wait_for(stream_task, timeout=1.0)

    asyncio.run(scenario())


async def _assert_ingest_event_stream(tmp_path: Path) -> None:
    job_db_path = tmp_path / "jobs.sqlite3"
    doc = tmp_path / "stream.md"
    doc.write_text("streaming status transitions\n", encoding="utf-8")
    store = IngestJobStore(job_db_path)
    record = store.create(path=str(doc), namespace="acme", collection="help")
    app = create_app(
        config=Config.local(),
        core_factory=lambda _: cast(Any, object()),
        job_db_path=job_db_path,
        ingest_roots=(tmp_path,),
    )
    stream_messages, disconnect, stream_task = _start_asgi_stream(
        app,
        f"/v1/ingest/{record.job_id}/events",
    )

    start = await _next_asgi_message(stream_messages)
    assert start["type"] == "http.response.start"
    assert start["status"] == 200
    headers = _headers_from_asgi_start(start)
    assert headers["x-request-id"] == "stream-test"
    assert headers["content-type"].startswith("text/event-stream")

    pending = _status_payload_from_sse(await _next_body_text(stream_messages))
    assert pending["status"] == "pending"

    store.update(record.job_id, status=INGEST_JOB_STATUS_RUNNING)
    running = _status_payload_from_sse(await _next_body_text(stream_messages))
    assert running["status"] == "running"
    assert await _next_body_text(stream_messages) == ": heartbeat\n\n"

    store.update(
        record.job_id,
        status=INGEST_JOB_STATUS_COMPLETED,
        result={"document_id": "stream-doc", "chunk_count": 1},
    )
    completed = _status_payload_from_sse(await _next_body_text(stream_messages))
    assert completed["status"] == "completed"
    assert [pending["status"], running["status"], completed["status"]] == [
        "pending",
        "running",
        "completed",
    ]
    assert await _next_body_text(stream_messages) == ""

    disconnect.set()
    await asyncio.wait_for(stream_task, timeout=1.0)


def _start_asgi_stream(
    app: Any,
    path: str,
) -> tuple[asyncio.Queue[dict[str, Any]], asyncio.Event, asyncio.Task[Any]]:
    messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    disconnect = asyncio.Event()

    async def receive() -> dict[str, object]:
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        await messages.put(message)

    task = asyncio.create_task(
        cast(Any, app)(
            _http_scope("GET", path, headers={"x-request-id": "stream-test"}),
            receive,
            send,
        )
    )
    return messages, disconnect, task


def _http_scope(
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
        "state": {},
    }


async def _next_asgi_message(
    messages: asyncio.Queue[dict[str, Any]],
) -> dict[str, Any]:
    return await asyncio.wait_for(messages.get(), timeout=1.0)


async def _next_body_text(messages: asyncio.Queue[dict[str, Any]]) -> str:
    while True:
        message = await _next_asgi_message(messages)
        if message["type"] != "http.response.body":
            continue
        raw_body = message.get("body", b"")
        if isinstance(raw_body, bytes) and raw_body:
            return raw_body.decode("utf-8")
        if not message.get("more_body", False):
            return ""


def _headers_from_asgi_start(message: dict[str, Any]) -> dict[str, str]:
    return {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in message.get("headers", ())
    }


def _status_payload_from_sse(frame: str) -> dict[str, object]:
    assert "event: status\n" in frame
    for line in frame.splitlines():
        if line.startswith("data: "):
            payload = json.loads(line.removeprefix("data: "))
            assert isinstance(payload, dict)
            return dict(payload)
    raise AssertionError(f"SSE frame has no data line: {frame!r}")
