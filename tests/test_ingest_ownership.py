"""Real local storage and public CLI checks for the single ingest owner."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest
from qdrant_client import models as rest

from rag_core import Config, cli
from rag_core.core import Engine
from rag_core.demo import DemoEmbeddingProvider
from rag_core.search.document_versions import KIND, CHUNK
from rag_core.search.providers.qdrant_store import QdrantVectorStore


@pytest.mark.integration
@pytest.mark.parametrize(
    "old_version",
    ["rag_core_processing_v6", "rag_core_processing_v7", "rag_core_processing_v8"],
)
def test_reingest_repairs_old_chunk_provenance_and_then_skips(
    tmp_path: Path, old_version: str
) -> None:
    async def run() -> None:
        base = Config.local(tmp_path / "index")
        old = replace(
            base,
            ingest=replace(base.ingest, processing_version=old_version),
        )

        async def ingest(core: Engine):
            return await core.add_bytes(
                file_bytes=b"# Billing\n\nInvoices accept card.",
                filename="billing.md",
                mime_type="text/markdown",
                tenant_id="acme",
                collection="docs",
                document_key="bill",
            )

        async with Engine(old, embedding_provider=DemoEmbeddingProvider()) as core:
            await ingest(core)
            store = core._store
            assert isinstance(store, QdrantVectorStore)
            # Reproduce the persisted metadata emitted by the earlier indexer.
            await store._client.set_payload(
                collection_name=store._config.collection_name,
                payload={"chunker_strategy": "prechunked"},
                points=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key=KIND, match=rest.MatchValue(value=CHUNK)
                        )
                    ]
                ),
            )
        async with Engine(base, embedding_provider=DemoEmbeddingProvider()) as core:
            assert (await ingest(core)).ingest_state == "reindexed"
            hits = await core.search(
                query="invoice", tenant_id="acme", collection="docs", rerank=False
            )
            assert hits and {hit.chunker_strategy for hit in hits} == {"markdown"}
            assert (await ingest(core)).ingest_state == "unchanged"

    asyncio.run(run())


def test_cli_dry_run_scans_without_reading_content_or_starting_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rag_core.ingest.sources.local as sources

    for name in ("a.md", "b.txt"):
        (tmp_path / name).write_text("private file contents")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must only discover names")

    monkeypatch.setattr(sources, "file_content_sha256", forbidden)
    monkeypatch.setattr(cli, "Engine", forbidden)
    assert (
        cli.main(
            [
                "add",
                str(tmp_path),
                "--collection",
                "docs",
                "--dry-run",
                "--max-files",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned_count"] == 1 and payload["truncated"] is True
    assert payload["documents"] == [
        {"path": "<local-file>", "filename": "a.md", "source_error": ""}
    ]
    assert str(tmp_path) not in json.dumps(payload)
    assert "private file contents" not in json.dumps(payload)
