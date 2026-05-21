"""Integration retrieval over the shared 10-doc corpus and real local Qdrant."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from rag_core.demo import build_demo_core

pytestmark = [pytest.mark.integration]

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "integration_corpus"
_CORPUS_PATH = _FIXTURE_DIR / "corpus.jsonl"
_CASES_PATH = _FIXTURE_DIR / "cases.jsonl"


def _load_corpus() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    with _CORPUS_PATH.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            docs.append(
                {
                    "document_id": str(row["document_id"]),
                    "markdown": f"# {row['title']}\n\n{row['body']}",
                }
            )
    return docs


def _load_cases() -> list[dict[str, object]]:
    with _CASES_PATH.open(encoding="utf-8") as handle:
        return [json.loads(raw) for raw in handle]


async def _ingest_integration_corpus(core: object) -> None:
    for doc in _load_corpus():
        await core.ingest_bytes(  # type: ignore[attr-defined]
            file_bytes=doc["markdown"].encode("utf-8"),
            filename=f"{doc['document_id']}.md",
            mime_type="text/markdown",
            namespace="integration",
            corpus_id="docs",
            document_id=doc["document_id"],
            document_key=f"{doc['document_id']}.md",
        )


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: str(case["query"])[:40])
def test_integration_corpus_search_returns_expected_doc(case: dict[str, object]) -> None:
    async def go() -> None:
        async with build_demo_core(collection=f"integration_{uuid.uuid4().hex}") as core:
            await _ingest_integration_corpus(core)
            hits = await core.search(
                query=str(case["query"]),
                namespace="integration",
                corpus_ids=["docs"],
                limit=5,
                rerank=False,
            )
            top_ids = [hit.document_id for hit in hits[:3]]
            expected_raw = case["expected_document_ids"]
            assert isinstance(expected_raw, list)
            expected = [str(doc_id) for doc_id in expected_raw]
            assert any(doc_id in top_ids for doc_id in expected), (
                f"query={case['query']!r} top3={top_ids} expected one of {expected}"
            )

    asyncio.run(go())


def test_integration_corpus_has_ten_topics() -> None:
    assert len(_load_corpus()) == 10
    assert len(_load_cases()) == 10
