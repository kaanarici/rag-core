"""Retrieve application context with OpenAI embeddings and embedded Qdrant.

Install the openai extra, set OPENAI_API_KEY, then run from the repository root:
    uv run python -m examples.configured_retrieval

This example calls a paid external provider. The local checks do not run that
call. Use examples.ask_folder for real local retrieval without API credentials.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

from rag_core import Config
from rag_core.core import Engine
from rag_core.config import EmbeddingConfig, QdrantConfig, VectorStoreConfig

EXAMPLES_DIR = Path(__file__).parent
CORPUS_DIR = EXAMPLES_DIR / "demo_corpus"
_TENANT_ID = "acme"
_COLLECTION = "help-center"


def _configured_config() -> Config:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for configured retrieval. "
            "For no-key smoke use: uv run python -m rag_core.quickstart"
        )
    return Config(
        vector_store=VectorStoreConfig(qdrant=QdrantConfig(
            location=":memory:",
            store_collection=f"rag_core_configured_{uuid.uuid4().hex}",
            dimension_aware_collection=False,
        )),
        embedding=EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key=api_key,
        ),
    )


async def run() -> None:
    config = _configured_config()
    async with Engine(config) as core:
        for path in sorted(CORPUS_DIR.glob("*.md")):
            await core.add_bytes(
                file_bytes=path.read_bytes(),
                filename=path.name,
                mime_type="text/markdown",
                tenant_id=_TENANT_ID,
                collection=_COLLECTION,
                document_id=path.stem,
                document_key=path.name,
            )
        pack = await core.context(
            query="How can invoices be paid?",
            tenant_id=_TENANT_ID,
            collections=[_COLLECTION],
            limit=5,
            rerank=False,
            max_chars=2_000,
        )
        print("Configured retrieval (semantic embeddings)")
        print(pack.as_prompt_text())
        if pack.prompt_citation_summary:
            print("\nCitations:")
            print(pack.prompt_citation_summary)


def main() -> int:
    try:
        asyncio.run(run())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
