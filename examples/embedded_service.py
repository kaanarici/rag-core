"""Minimal ASGI-style lifecycle: one RAGCore per process.

Uses ``build_demo_core`` so this module always runs without API keys. For semantic
embeddings and real ``RAGCoreConfig``, see ``examples/configured_retrieval.py`` and
``docs/embed.md``.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from rag_core import RAGCore
from rag_core.demo import build_demo_core


@asynccontextmanager
async def rag_core_lifespan() -> AsyncIterator[RAGCore]:
    """Bind one core for the worker lifetime."""
    core = build_demo_core(collection="embedded_service")
    await core.ensure_ready()
    try:
        yield core
    finally:
        await core.close()


async def handle_retrieve(core: RAGCore, *, tenant_id: str, query: str) -> str:
    """Example handler: tenancy comes from auth, not the model."""
    pack = await core.retrieve_context(
        query=query,
        namespace=tenant_id,
        corpus_ids=["help-center"],
        limit=5,
        rerank=False,
        max_chars=2_000,
    )
    return pack.as_prompt_text()


async def demo() -> None:
    async with rag_core_lifespan() as core:
        await core.ingest_bytes(
            file_bytes=b"Invoices can be paid by card or ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            corpus_id="help-center",
        )
        text = await handle_retrieve(core, tenant_id="acme", query="How to pay invoices?")
        print(text)


def main() -> None:
    asyncio.run(demo())


if __name__ == "__main__":
    main()
