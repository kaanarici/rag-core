"""Retrieve a context pack for an application model call."""

from __future__ import annotations

import asyncio

from rag_core import ModelContextPack, RAGCore
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


async def build_chatbot_context(core: RAGCore, query: str) -> ModelContextPack:
    return await core.retrieve_context(
        query=query,
        namespace="acme",
        corpus_ids=["help-center"],
        limit=4,
        rerank=False,
        max_chars=1_200,
    )


async def run_chatbot_context_demo(query: str) -> ModelContextPack:
    core = build_demo_core(collection="chatbot_context")
    async with core:
        await seed_help_center(core)
        return await build_chatbot_context(core, query)


async def run_demo() -> None:
    context = await run_chatbot_context_demo("How can a customer pay an invoice?")
    print("Context to pass into your model call:")
    print(context.as_text())
    print("\nCitations:")
    for source in context.citations:
        title = source.title or source.document_key or source.document_id or source.result_id
        print(f"- {source.source_id}: {title}")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
