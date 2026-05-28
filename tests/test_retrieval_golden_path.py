"""Golden retrieval path: parse/chunk/index into real local Qdrant, then search.

Most search tests intentionally use fake stores to pin pipeline wiring. This file
is the opposite: keep it small, but make it exercise the user-facing retrieval
path over a real local Qdrant collection.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from rag_core.demo import build_demo_core

pytestmark = [pytest.mark.integration]

_INTEGRATION_CORPUS = Path(__file__).parent / "fixtures" / "integration_corpus" / "corpus.jsonl"


def _integration_corpus_docs() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    with _INTEGRATION_CORPUS.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            docs.append(
                {
                    "document_id": str(row["document_id"]),
                    "markdown": f"# {row['title']}\n\n{row['body']}",
                }
            )
    return docs


def test_ingest_then_search_ranks_relevant_document_above_decoys() -> None:
    async def go() -> None:
        async with build_demo_core(collection=f"golden_{uuid.uuid4().hex}") as core:
            docs = {
                "billing": b"Invoices can be paid by ACH or credit card. Billing runs monthly.",
                "shipping": b"International shipping requires customs forms and carrier tracking.",
                "auth": b"Single sign-on authentication uses SAML identity provider metadata.",
            }
            for document_id, body in docs.items():
                await core.ingest_bytes(
                    file_bytes=body,
                    filename=f"{document_id}.txt",
                    mime_type="text/plain",
                    namespace="golden",
                    corpus_id="docs",
                    document_id=document_id,
                    document_key=f"{document_id}.txt",
                )

            hits = await core.search(
                query="How can invoices be paid?",
                namespace="golden",
                corpus_ids=["docs"],
                limit=3,
                rerank=False,
            )

        assert hits[0].document_id == "billing"
        assert "ACH" in hits[0].text or "credit card" in hits[0].text

    asyncio.run(go())


def test_golden_path_uses_full_integration_corpus() -> None:
    async def go() -> None:
        async with build_demo_core(collection=f"golden10_{uuid.uuid4().hex}") as core:
            for doc in _integration_corpus_docs():
                await core.ingest_bytes(
                    file_bytes=doc["markdown"].encode("utf-8"),
                    filename=f"{doc['document_id']}.md",
                    mime_type="text/markdown",
                    namespace="golden",
                    corpus_id="docs",
                    document_id=doc["document_id"],
                    document_key=f"{doc['document_id']}.md",
                )

            hits = await core.search(
                query="webhook hmac signature header",
                namespace="golden",
                corpus_ids=["docs"],
                limit=3,
                rerank=False,
            )

        top_ids = [hit.document_id for hit in hits[:3]]
        assert "webhooks" in top_ids

    asyncio.run(go())
