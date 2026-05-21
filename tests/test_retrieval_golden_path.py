"""Golden retrieval path: parse/chunk/index into real local Qdrant, then search.

Most search tests intentionally use fake stores to pin orchestration. This file
is the opposite: keep it small, but make it exercise the user-facing retrieval
path over a real local Qdrant collection.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from rag_core.demo import build_demo_core

pytestmark = [pytest.mark.integration]


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

        assert [hit.document_id for hit in hits[:3]] == ["billing", "auth", "shipping"]
        assert "ACH" in hits[0].text or "credit card" in hits[0].text

    asyncio.run(go())
