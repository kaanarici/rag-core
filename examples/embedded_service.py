"""Minimal ASGI-style lifecycle: one Engine per process.

Uses ``build_demo_core`` so this module always runs without API keys. For semantic
embeddings and real ``Config``, see ``examples/configured_retrieval.py`` and
https://kaanarici.github.io/rag-core/docs/embed.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from rag_core import Engine
from rag_core.demo import build_demo_core


@asynccontextmanager
async def rag_core_lifespan() -> AsyncIterator[Engine]:
    """Bind one core for the worker lifetime."""
    core = build_demo_core(store_collection="embedded_service")
    await core.ensure_ready()
    try:
        yield core
    finally:
        await core.close()


async def handle_retrieve(core: Engine, *, tenant_id: str, query: str) -> str:
    """Example handler: tenancy comes from auth, not the model."""
    pack = await core.context(
        query=query,
        namespace=tenant_id,
        collections=["help-center"],
        limit=5,
        rerank=False,
        max_chars=2_000,
    )
    return pack.as_prompt_text()


async def demo() -> None:
    async with rag_core_lifespan() as core:
        await core.add_bytes(
            file_bytes=b"Invoices can be paid by card or ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            collection="help-center",
        )
        text = await handle_retrieve(core, tenant_id="acme", query="How to pay invoices?")
        print(text)


def main() -> None:
    asyncio.run(demo())


if __name__ == "__main__":
    main()
