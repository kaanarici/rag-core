"""Small local ingest-and-retrieve example using embedded Qdrant."""

from __future__ import annotations

import asyncio

from rag_core.demo import build_demo_core


async def run_demo() -> None:
    core = build_demo_core(store_collection="minimal_app")
    async with core:
        ingested = await core.add_bytes(
            file_bytes=b"Billing is due monthly and invoices can be paid by card or ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            collection="help-center",
        )
        context = await core.context(
            query="How can I pay invoices?",
            namespace="acme",
            collections=["help-center"],
            limit=3,
            rerank=False,
            max_chars=1_200,
        )
        print(f"Indexed document: {ingested.document_id} ({ingested.chunk_count} chunks)")
        print("Prompt-safe context text:")
        print(context.as_prompt_text())
        if context.prompt_citation_summary:
            print("\nCitations:")
            print(context.prompt_citation_summary)


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
