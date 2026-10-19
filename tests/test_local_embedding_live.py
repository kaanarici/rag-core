from __future__ import annotations

import asyncio
import math
import os

import pytest

from rag_core.demo import DemoEmbeddingProvider
from rag_core.search.providers.embedding import create_embedding_provider

pytestmark = [pytest.mark.live]

QUERY = "How much does it cost to fix my car?"
RIGHT = (
    "Vehicle maintenance expenses include brake service, oil changes, "
    "and tire replacement."
)
WRONG = "A spreadsheet cost column can be fixed by locking cells."


def _skip_if_download_disabled() -> None:
    if os.environ.get("RAG_CORE_RUN_FASTEMBED_LIVE") != "1":
        pytest.skip(
            "set RAG_CORE_RUN_FASTEMBED_LIVE=1 to run the FastEmbed live smoke "
            "(downloads the local embedding model on first use)"
        )


def _dot(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def test_local_dense_embeddings_rank_paraphrase_above_demo_hash() -> None:
    _skip_if_download_disabled()

    async def _run() -> None:
        local = create_embedding_provider(provider="local")
        demo = DemoEmbeddingProvider()
        local_query = await local.embed_query(QUERY)
        local_docs = await local.embed_texts([RIGHT, WRONG])
        demo_query = await demo.embed_query(QUERY)
        demo_docs = await demo.embed_texts([RIGHT, WRONG])

        assert _dot(local_query, local_docs[0]) > _dot(local_query, local_docs[1])
        assert _dot(demo_query, demo_docs[0]) < _dot(demo_query, demo_docs[1])

    asyncio.run(_run())
