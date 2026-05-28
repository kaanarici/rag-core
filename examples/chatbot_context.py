"""Retrieve prompt-safe context text for an application model call."""

from __future__ import annotations

import asyncio

from rag_core import ContextPack, RAGCore
from rag_core.demo import build_demo_core


async def seed_help_center(core: RAGCore) -> None:
    await core.ingest_bytes(
        file_bytes=(
            b"Billing invoices are due monthly. Customers can pay by card, "
            b"ACH, or wire transfer from the billing portal."
        ),
        filename="billing.txt",
        mime_type="text/plain",
        namespace="acme",
        corpus_id="help-center",
    )
    await core.ingest_bytes(
        file_bytes=b"Shipping usually takes 3-5 business days after fulfillment.",
        filename="shipping.txt",
        mime_type="text/plain",
        namespace="acme",
        corpus_id="help-center",
    )


async def build_chatbot_context(core: RAGCore, query: str) -> ContextPack:
    return await core.retrieve_context(
        query=query,
        namespace="acme",
        corpus_ids=["help-center"],
        limit=4,
        rerank=False,
        max_chars=1_200,
    )


async def run_chatbot_context_demo(query: str) -> ContextPack:
    core = build_demo_core(collection="chatbot_context")
    async with core:
        await seed_help_center(core)
        return await build_chatbot_context(core, query)


async def run_demo() -> None:
    context = await run_chatbot_context_demo("How can a customer pay an invoice?")
    print("Prompt-safe context text:")
    print(context.as_prompt_text())
    if context.prompt_citation_summary:
        print("\nCitations:")
        print(context.prompt_citation_summary)


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
