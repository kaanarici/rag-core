"""Regressions reproduced by the independent full-source review."""

from __future__ import annotations
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import pytest
from rag_core import Config
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider
from rag_core.documents.ocr import CommandOcrProvider, OcrRequest


def test_parser_limits_invalidate_ingest_and_converter_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        async with Engine(
            Config.local(), embedding_provider=DemoEmbeddingProvider()
        ) as core:

            async def ingest():
                return await core.add_bytes(
                    file_bytes=b"name,value\nfirst,1\nSENTINEL_LATER_ROW,2\n",
                    filename="data.csv",
                    mime_type="text/csv",
                    tenant_id="acme",
                    collection="docs",
                )

            monkeypatch.setenv("LOCAL_PARSE_CSV_MAX_ROWS", "2")
            first = await ingest()
            assert first.metadata["truncated"] is True
            monkeypatch.setenv("LOCAL_PARSE_CSV_MAX_ROWS", "100")
            second = await ingest()
            assert second.ingest_state == "reindexed"
            hits = await core.search(
                query="sentinel", tenant_id="acme", collection="docs", rerank=False
            )
            assert any("SENTINEL_LATER_ROW" in h.text for h in hits)
            assert (await ingest()).ingest_state == "unchanged"

    asyncio.run(run())


def test_embedding_identity_changes_invalidate_same_store() -> None:
    class Model(DemoEmbeddingProvider):
        @property
        def model_name(self) -> str:
            return self.name

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    async def run() -> None:
        first = Engine(Config.local(), embedding_provider=Model("model-a"))

        async def ingest(core):
            return await core.add_bytes(
                file_bytes=b"Invoices paid by ACH.",
                filename="bill.txt",
                mime_type="text/plain",
                tenant_id="acme",
                collection="docs",
            )

        try:
            assert (await ingest(first)).ingest_state == "created"
            with pytest.raises(ValueError, match="different embedding model"):
                Engine(
                    Config.local(),
                    embedding_provider=Model("model-b"),
                    vector_store=first._store,
                )
        finally:
            await first.close()

    asyncio.run(run())


def test_command_ocr_yields_and_enforces_own_timeout() -> None:
    async def run() -> None:
        flags = []
        asyncio.get_running_loop().call_soon(lambda: flags.append(True))
        provider = CommandOcrProvider(
            command=[sys.executable, "-c", "import time;time.sleep(3)"],
            timeout_seconds=0.05,
        )
        request = OcrRequest(
            file_bytes=b"ocr", filename="scan.png", mime_type="image/png"
        )
        with pytest.raises(RuntimeError, match="timed out"):
            await asyncio.wait_for(provider.extract_markdown(request), 1)
        assert flags

    asyncio.run(run())


def test_command_ocr_cancellation_reaps_child_and_removes_input(tmp_path: Path) -> None:
    async def run() -> None:
        marker = tmp_path / "child.json"
        script = 'import json,os,sys,time;from pathlib import Path;p=json.load(sys.stdin);Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid(),"input":p["file_path"]}));time.sleep(10)'
        provider = CommandOcrProvider(
            command=[sys.executable, "-c", script, str(marker)]
        )
        task = asyncio.create_task(
            provider.extract_markdown(
                OcrRequest(
                    file_bytes=b"ocr", filename="scan.png", mime_type="image/png"
                )
            )
        )

        async def ready():
            while not marker.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(ready(), 3)
        child = json.loads(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert not Path(child["input"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(child["pid"], 0)

    asyncio.run(run())


def test_command_ocr_healthy_result() -> None:
    async def run() -> None:
        provider = CommandOcrProvider(
            command=[
                sys.executable,
                "-c",
                'import json; print(json.dumps({"markdown":"OCR text"}))',
            ]
        )
        result = await provider.extract_markdown(
            OcrRequest(file_bytes=b"ocr", filename="scan.png", mime_type="image/png")
        )
        assert result.markdown == "OCR text"

    asyncio.run(run())


def test_scoped_context_uses_shared_trace_and_optional_reranker() -> None:
    from rag_core import RAGCore, Document
    from rag_core.events import EventBuffer
    from rag_core.events.types import SearchStageCompleted

    class Reranker:
        calls = 0

        async def rerank(self, query, documents, top_k=10):
            from rag_core.search.request_models import RerankResult

            self.calls += 1
            return [
                RerankResult(index=i, score=1.0, text=text)
                for i, text in enumerate(documents[:top_k])
            ]

    async def run() -> None:
        events = EventBuffer()
        reranker = Reranker()
        async with RAGCore(
            Engine(
                Config.local(),
                embedding_provider=DemoEmbeddingProvider(),
                event_sink=events,
                reranker=reranker,
            ),
            tenant_id="acme",
            collection="docs",
        ) as rag:
            for key in ("a", "b"):
                await rag.ingest(
                    Document(
                        key=key,
                        content=b"Invoices paid by ACH.",
                        content_type="text/plain",
                    )
                )
            result = await rag.context("invoice", limit=2, rerank=True)
            assert len(result.snippets) == 1 and reranker.calls == 1
            packed = [
                e
                for e in events.events
                if isinstance(e, SearchStageCompleted) and e.stage == "context_pack"
            ]
            assert len(packed) == 1 and packed[0].search_id
            assert (
                len(
                    {
                        e.search_id
                        for e in events.events
                        if getattr(e, "search_id", None)
                    }
                )
                == 1
            )

    asyncio.run(run())


def test_manifest_latest_lookup_reuses_index_and_sees_external_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import asdict
    from rag_core.ingest.manifest import latest_manifest_entry
    from rag_core.manifest.persistence import read_entries, manifest_path
    import rag_core.manifest.jsonl as journal

    async def run() -> None:
        folder = tmp_path / "source"
        folder.mkdir()
        for i in range(8):
            (folder / f"{i}.txt").write_text(f"Invoice number {i}.")
        base = Config.local()
        directory = tmp_path / "manifest"
        config = replace(
            base, ingest=replace(base.ingest, manifest_directory=directory)
        )
        async with Engine(config, embedding_provider=DemoEmbeddingProvider()) as core:
            await core.add(folder, tenant_id="acme", collection="docs")
            original = journal._read_manifest_jsonl_entries_unlocked
            decoded = []

            def counted(path):
                rows = original(path)
                decoded.append(len(rows))
                return rows

            monkeypatch.setattr(
                journal, "_read_manifest_jsonl_entries_unlocked", counted
            )
            again = await core.add(folder, tenant_id="acme", collection="docs")
            assert {row.ingest_state for row in again.succeeded} == {"unchanged"}
            assert len(decoded) <= 2 and sum(decoded) <= 16
            entry = read_entries(directory, "acme", "docs")[0]
            altered = replace(entry, metadata={"external_change": True})
            with manifest_path(directory, "acme", "docs").open("a") as file:
                file.write(json.dumps(asdict(altered)) + "\n")
            current = latest_manifest_entry(
                directory,
                tenant_id="acme",
                collection="docs",
                document_id=entry.document_id,
            )
            assert current is not None and current.metadata["external_change"] is True

    asyncio.run(run())
