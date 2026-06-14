"""PR retrieval eval: fixture embedding geometry on real local Qdrant."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from statistics import mean

import pytest

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig
from rag_core.evals import EvalResult, load_cases, run_eval
from rag_core.search import search_profile
from tests.support import FixtureEmbeddingProvider

pytestmark = [pytest.mark.eval]

_FIXTURE_DIR = Path(__file__).resolve().parent / "pr_corpus"
_CORPUS_PATH = _FIXTURE_DIR / "corpus.jsonl"
_CASES_PATH = _FIXTURE_DIR / "cases.jsonl"
_EMBEDDINGS_PATH = _FIXTURE_DIR / "embeddings.json"


def test_pr_corpus_eval_holds_honest_floors() -> None:
    results = asyncio.run(_run_pr_eval())
    _assert_pr_floors(results, recall_at_5=0.55, mrr=0.45, ndcg_at_10=0.45)


def test_fixture_embedding_resolves_each_document_id() -> None:
    corpus = _load_corpus(_CORPUS_PATH)
    provider = FixtureEmbeddingProvider(
        fixture_path=_EMBEDDINGS_PATH,
        pending_document_ids=[doc["document_id"] for doc in corpus],
    )

    async def _collect() -> list[list[float]]:
        return [
            (await provider.embed_texts(["chunk"]))[0] for _ in corpus
        ]

    vectors = asyncio.run(_collect())
    assert len(vectors) == len(corpus)
    assert len({tuple(vector) for vector in vectors}) == len(corpus)


async def _run_pr_eval() -> list[EvalResult]:
    corpus = _load_corpus(_CORPUS_PATH)
    embedding = FixtureEmbeddingProvider(
        fixture_path=_EMBEDDINGS_PATH,
        pending_document_ids=[doc["document_id"] for doc in corpus],
    )
    config = RAGCoreConfig(
        qdrant=QdrantConfig(
            location=":memory:",
            collection=f"rag_core_pr_eval_{uuid.uuid4().hex}",
            dimension_aware_collection=False,
        ),
        embedding=EmbeddingConfig(
            provider="fixture",
            model=embedding.model_name,
            dimensions=embedding.dimensions,
        ),
    )
    core = RAGCore(config, embedding_provider=embedding)
    try:
        for doc in corpus:
            await core.ingest_bytes(
                file_bytes=doc["markdown"].encode("utf-8"),
                filename=f"{doc['document_id']}.md",
                mime_type="text/markdown",
                namespace="pr_eval",
                corpus_id="docs",
                document_id=doc["document_id"],
                document_key=f"{doc['document_id']}.md",
            )
        return await run_eval(
            core,
            load_cases(_CASES_PATH),
            query_plan=search_profile("balanced", limit=10),
        )
    finally:
        await core.close()


def _load_corpus(path: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            docs.append(
                {
                    "document_id": str(row["document_id"]),
                    "markdown": f"# {row['title']}\n\n{row['body']}",
                }
            )
    return docs


def _assert_pr_floors(
    results: list[EvalResult],
    *,
    recall_at_5: float,
    mrr: float,
    ndcg_at_10: float,
) -> None:
    assert results
    assert all(result.error_type is None for result in results)
    assert mean(result.recall_at_5 for result in results) >= recall_at_5
    assert mean(result.mrr for result in results) >= mrr
    assert mean(result.ndcg_at_10 for result in results) >= ndcg_at_10
