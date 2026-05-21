from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from rag_core.core_models import IngestedDocument
from rag_core.local_corpus import LocalIngestRequest, run_local_ingest
from rag_core.sources import document_key as local_document_key


class _MixedStateLocalCore:
    def __init__(self) -> None:
        self.closed = False
        self.ingest_calls: list[dict[str, Any]] = []

    async def ensure_ready(self) -> None:
        return None

    async def ingest_file(
        self,
        path: Path,
        *,
        namespace: str,
        corpus_id: str,
        document_key: str,
        metadata: dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> IngestedDocument:
        self.ingest_calls.append(
            {
                "path": path,
                "namespace": namespace,
                "corpus_id": corpus_id,
                "document_key": document_key,
                "metadata": metadata,
                "force_reindex": force_reindex,
            }
        )
        ingest_state = "unchanged" if path.name == "reference.md" else "created"
        return IngestedDocument(
            document_id=f"doc-{path.stem}",
            namespace=namespace,
            corpus_id=corpus_id,
            chunk_count=2,
            filename=path.name,
            mime_type="text/markdown",
            document_key=document_key,
            content_sha256=f"hash-{path.stem}",
            ingest_state=ingest_state,
            replaced_existing=False,
        )

    async def close(self) -> None:
        self.closed = True


def test_local_ingest_result_distinguishes_succeeded_written_and_skipped(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("guide", encoding="utf-8")
    (docs / "reference.md").write_text("reference", encoding="utf-8")
    core = _MixedStateLocalCore()

    result = asyncio.run(
        run_local_ingest(
            LocalIngestRequest(path=docs, namespace="acme", corpus_id="help"),
            core_factory=lambda: core,
        )
    )

    assert result.succeeded_count == 2
    assert result.written_count == 1
    assert result.skipped_count == 1
    assert core.closed is True
    guide_key = local_document_key(docs, docs / "guide.md")
    reference_key = local_document_key(docs, docs / "reference.md")
    assert [record.document_key for record in result.succeeded] == [
        guide_key,
        reference_key,
    ]
    assert [record.document_key for record in result.written] == [guide_key]
    assert [record.document_key for record in result.skipped] == [reference_key]

    payload = result.to_payload()
    assert payload["succeeded_count"] == 2
    assert payload["written_count"] == 1
    assert payload["skipped_count"] == 1
    succeeded = cast(list[dict[str, object]], payload["succeeded"])
    written = cast(list[dict[str, object]], payload["written"])
    skipped = cast(list[dict[str, object]], payload["skipped"])
    assert [record["ingest_state"] for record in succeeded] == [
        "created",
        "unchanged",
    ]
    assert [record["ingest_state"] for record in written] == ["created"]
    assert [record["ingest_state"] for record in skipped] == ["unchanged"]
