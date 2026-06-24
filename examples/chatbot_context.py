"""Retrieve prompt-safe context text for an application model call."""

from __future__ import annotations

import asyncio

from rag_core import Context, Engine
from rag_core.demo import build_demo_core


async def seed_help_center(core: Engine) -> None:
    await core.add_bytes(
        file_bytes=(
            b"Billing invoices are due monthly. Customers can pay by card, "
            b"ACH, or wire transfer from the billing portal."
        ),
        filename="billing.txt",
        mime_type="text/plain",
        namespace="acme",
        collection="help-center",
    )
    await core.add_bytes(
        file_bytes=b"Shipping usually takes 3-5 business days after fulfillment.",
        filename="shipping.txt",
        mime_type="text/plain",
        namespace="acme",
        collection="help-center",
    )


async def build_chatbot_context(core: Engine, query: str) -> Context:
    return await core.context(
        query=query,
        namespace="acme",
        collections=["help-center"],
        limit=4,
        rerank=False,
        max_chars=1_200,
    )


async def run_chatbot_context_demo(query: str) -> Context:
    core = build_demo_core(store_collection="chatbot_context")
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
