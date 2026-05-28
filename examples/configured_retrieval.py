"""Configured retrieval: real embeddings + Qdrant (requires credentials).

Smoke examples use ``build_demo_core`` (deterministic, not semantic). This script
shows the embed path from ``docs/embed.md``.

Requires:
  - ``OPENAI_API_KEY`` (or set ``RAG_CORE_EMBEDDING_PROVIDER`` / key envs your doctor accepts)
  - Network access to the embedding API

Run from repo root::

    uv run python -m examples.configured_retrieval

The command is not executed by default CI because it requires credentials, but
the file is still source-checked and packaged.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig

EXAMPLES_DIR = Path(__file__).parent
CORPUS_DIR = EXAMPLES_DIR / "demo_corpus"
_NAMESPACE = "acme"
_CORPUS_ID = "help-center"


def _configured_config() -> RAGCoreConfig:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for configured retrieval. "
            "For no-key smoke use: uv run python -m rag_core.quickstart"
        )
    return RAGCoreConfig(
        qdrant=QdrantConfig(
            location=":memory:",
            collection=f"rag_core_configured_{uuid.uuid4().hex}",
            dimension_aware_collection=False,
        ),
        embedding=EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            api_key=api_key,
        ),
    )


async def run() -> None:
    config = _configured_config()
    async with RAGCore(config) as core:
        for path in sorted(CORPUS_DIR.glob("*.md")):
            await core.ingest_bytes(
                file_bytes=path.read_bytes(),
                filename=path.name,
                mime_type="text/markdown",
                namespace=_NAMESPACE,
                corpus_id=_CORPUS_ID,
                document_id=path.stem,
                document_key=path.name,
            )
        pack = await core.retrieve_context(
            query="How can invoices be paid?",
            namespace=_NAMESPACE,
            corpus_ids=[_CORPUS_ID],
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
