"""Wheel-installable first-run demo (no checkout required).

Run after ``pip install rag-core``:

    python -m rag_core.quickstart
"""

from __future__ import annotations

import asyncio

from rag_core.demo import build_demo_core


async def run() -> None:
    core = build_demo_core(collection="rag_core_quickstart")
    async with core:
        ingested = await core.ingest_bytes(
            file_bytes=(
                b"Billing is due monthly and invoices can be paid by card or ACH."
            ),
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            corpus_id="help-center",
        )
        context = await core.retrieve_context(
            query="How can I pay invoices?",
            namespace="acme",
            corpus_ids=["help-center"],
            limit=3,
            rerank=False,
            max_chars=1_200,
        )
        print(f"Indexed document: {ingested.document_id} ({ingested.chunk_count} chunks)")
        print("Context to pass into your model call:")
        print(context.as_text())
        print("\nCitations:")
        for source in context.citations:
            title = (
                source.title
                or source.document_key
                or source.document_id
                or source.result_id
            )
            print(f"- {source.source_id}: {title}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
