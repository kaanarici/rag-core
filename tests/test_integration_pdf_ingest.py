"""PDF ingest journey: PyMuPDF text extraction through search."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from rag_core.demo import build_demo_core

pytestmark = [pytest.mark.integration]


def test_pdf_text_route_ingest_and_search() -> None:
    pytest.importorskip("fitz")
    import fitz

    async def go() -> None:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Invoice payments support ACH and monthly billing cycles.")
        pdf_bytes = doc.tobytes()
        doc.close()

        async with build_demo_core(store_collection=f"pdf_ingest_{uuid.uuid4().hex}") as core:
            await core.add_bytes(
                file_bytes=pdf_bytes,
                filename="billing.pdf",
                mime_type="application/pdf",
                namespace="pdf-journey",
                collection="docs",
                document_id="billing-pdf",
                document_key="billing.pdf",
            )
            hits = await core.search(
                query="How are invoices paid?",
                namespace="pdf-journey",
                collections=["docs"],
                limit=3,
                rerank=False,
            )

        assert hits
        assert hits[0].document_id == "billing-pdf"
        assert "ACH" in hits[0].text or "billing" in hits[0].text.lower()

    asyncio.run(go())
