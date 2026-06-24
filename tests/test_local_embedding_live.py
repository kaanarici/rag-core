from __future__ import annotations

import asyncio
import math
import os

import pytest

from rag_core import Config, Engine
from rag_core.demo import DemoEmbeddingProvider
from rag_core.search import query_plan_preset
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


def test_rag_core_local_config_ingests_and_searches_with_dense_local_model() -> None:
    _skip_if_download_disabled()

    async def _run() -> None:
        core = Engine(Config.local())
        try:
            await core.ensure_ready()
            await core.add_bytes(
                file_bytes=RIGHT.encode("utf-8"),
                filename="vehicle.txt",
                mime_type="text/plain",
                namespace="acme",
                collection="proof",
            )
            await core.add_bytes(
                file_bytes=WRONG.encode("utf-8"),
                filename="spreadsheet.txt",
                mime_type="text/plain",
                namespace="acme",
                collection="proof",
            )
            hits = await core.search(
                query=QUERY,
                namespace="acme",
                collections=["proof"],
                limit=2,
                rerank=False,
                query_plan=query_plan_preset("dense_only", limit=2),
            )
        finally:
            await core.close()

        assert hits[0].title == "vehicle.txt"
        assert hits[0].score > hits[1].score

    asyncio.run(_run())
