from __future__ import annotations
from typing import NoReturn

import asyncio
from pathlib import Path
from typing import Any

import pytest

from rag_core.core_models import IngestedDocument
from rag_core.events.sinks import EventBuffer
from rag_core.events.types import IngestBatchFailed, IngestBatchProgress
from rag_core.ingest.local import LocalIngestRequest, run_local_ingest_with_core


class FailingLocalIngestCore:
    async def ensure_ready(self) -> None:
        return None

    async def add_file(
        self,
        file_path: Path,
        *,
        tenant_id: str,
        collection: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
        pre_read_bytes: bytes | None = None,
    ) -> IngestedDocument:
        raise RuntimeError("secret local path /tmp/private/doc.md")

    async def close(self) -> None:
        return None

    async def delete_document(self, *, document_id: str, tenant_id: str, collection: str) -> NoReturn:
        raise AssertionError("prune_missing is disabled in this test")


class UnreadyLocalIngestCore(FailingLocalIngestCore):
    async def ensure_ready(self) -> None:
        raise RuntimeError("secret vector store dsn")


class FailingRemoteUrlCore:
    async def ensure_ready(self) -> None:
        return None

    async def add_url(self, url: str, **kwargs: Any) -> IngestedDocument:
        raise RuntimeError(f"secret remote token for {url}")

    async def close(self) -> None:
        return None


class UnreadyRemoteUrlCore(FailingRemoteUrlCore):
    async def ensure_ready(self) -> None:
        raise RuntimeError("secret remote store dsn")


def test_local_ingest_progress_event_uses_error_type_not_exception_text(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[object, IngestBatchProgress]:
        path = tmp_path / "doc.md"
        path.write_text("hello", encoding="utf-8")
        events = EventBuffer()
        result = await run_local_ingest_with_core(
            LocalIngestRequest(path=path, tenant_id="acme", collection="help"),
            core=FailingLocalIngestCore(),
            event_sink=events,
        )
        [progress] = [
            event
            for event in events.events
            if isinstance(event, IngestBatchProgress)
        ]
        return result.records[0], progress

    record, progress = asyncio.run(run())

    assert "secret local path" not in str(record)
    assert "ingest failed with RuntimeError" in str(record)
    assert progress.error == "RuntimeError"
    assert "secret local path" not in str(progress)


def test_local_ingest_failed_event_uses_error_type_not_exception_text(
    tmp_path: Path,
) -> None:
    async def run() -> IngestBatchFailed:
        path = tmp_path / "doc.md"
        path.write_text("hello", encoding="utf-8")
        events = EventBuffer()
        with pytest.raises(RuntimeError):
            await run_local_ingest_with_core(
                LocalIngestRequest(path=path, tenant_id="acme", collection="help"),
                core=UnreadyLocalIngestCore(),
                event_sink=events,
            )
        [failed] = [
            event for event in events.events if isinstance(event, IngestBatchFailed)
        ]
        return failed

    failed = asyncio.run(run())

    assert failed.error == "RuntimeError"
    assert "secret vector store" not in str(failed)
