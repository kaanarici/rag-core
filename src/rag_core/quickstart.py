"""Installed-package first-run demo.

Run after installing a local wheel or editable checkout:

    python -m rag_core.quickstart
"""

from __future__ import annotations

import asyncio

from rag_core.demo import (
    _DEMO_CORPUS_ID,
    _DEMO_NAMESPACE,
    _DEMO_QUERY,
    build_demo_core,
    ingest_demo_billing_document,
)


async def run() -> None:
    core = build_demo_core(collection="rag_core_quickstart")
    async with core:
        ingested = await ingest_demo_billing_document(core)
        context = await core.retrieve_context(
            query=_DEMO_QUERY,
            namespace=_DEMO_NAMESPACE,
            corpus_ids=[_DEMO_CORPUS_ID],
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
    asyncio.run(run())


if __name__ == "__main__":
    main()
