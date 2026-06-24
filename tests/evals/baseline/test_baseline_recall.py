"""Baseline retrieval-quality regression run through the eval harness.

The fixture in ``corpus.jsonl`` is a small synthetic corpus where each
document covers a distinct topic anchored by a unique key term, so a
deterministic vocabulary-counting fake embedder + sparse embedder give
reproducible retrieval. ``cases.jsonl`` carries the labelled queries.

Floors are set from the current clean run, with headroom. Raise them as
retrieval improves; do not lower them without recording why.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from statistics import mean

import pytest

from rag_core import Engine
from rag_core.evals import EvalResult, load_cases, run_eval
from tests.support import (
    BASELINE_VOCABULARY,
    KeywordEmbeddingProvider,
    KeywordSparseEmbedder,
    make_test_config,
)

BASELINE_RECALL_AT_5_FLOOR: float = 1.0
BASELINE_RECALL_AT_10_FLOOR: float = 1.0
BASELINE_MRR_FLOOR: float = 1.0
BASELINE_NDCG_AT_10_FLOOR: float = 1.0
"""Floors for the baseline corpus.

Raise these floors as retrieval improves; do not lower them without
recording why in the commit body or an ADR."""

_FIXTURE_DIR = Path(__file__).parent
_CORPUS_PATH = _FIXTURE_DIR / "corpus.jsonl"
_CASES_PATH = _FIXTURE_DIR / "cases.jsonl"


@pytest.mark.eval_harness
def test_baseline_recall_at_5_holds() -> None:
    """End-to-end ingest + search over the baseline corpus must hold floors."""

    asyncio.run(_run_baseline_eval())


async def _run_baseline_eval() -> None:
    config = make_test_config(
        qdrant_collection="rag_core_baseline",
        embedding_dimensions=len(BASELINE_VOCABULARY),
        qdrant_dimension_aware_collection=False,
    )
    core = Engine(
        config,
        embedding_provider=KeywordEmbeddingProvider(BASELINE_VOCABULARY),
        sparse_embedder=KeywordSparseEmbedder(BASELINE_VOCABULARY),
    )
    try:
        await core.ensure_ready()
        for doc in _load_corpus(_CORPUS_PATH):
            await core.add_bytes(
                file_bytes=doc["markdown"].encode("utf-8"),
                filename=f"{doc['document_id']}.md",
                mime_type="text/markdown",
                namespace="baseline",
                collection="docs",
                document_id=doc["document_id"],
                document_key=f"{doc['document_id']}.md",
            )

        cases = load_cases(_CASES_PATH)
        results = await run_eval(core, cases)
    finally:
        await core.close()

    _assert_floors(results)


def _load_corpus(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "document_id": str(row["document_id"]),
                    "markdown": f"# {row['title']}\n\n{row['body']}",
                }
            )
    return rows


def _assert_floors(results: list[EvalResult]) -> None:
    assert results, "eval runner returned no results"
    failed_cases = [
        result.case.case_id or result.case.query
        for result in results
        if result.error_type is not None
        or result.recall_at_5 < BASELINE_RECALL_AT_5_FLOOR
        or result.recall_at_10 < BASELINE_RECALL_AT_10_FLOOR
        or result.mrr < BASELINE_MRR_FLOOR
        or result.ndcg_at_10 < BASELINE_NDCG_AT_10_FLOOR
    ]
    assert not failed_cases, f"baseline eval failed cases: {failed_cases}"
    mean_recall_at_5 = mean(r.recall_at_5 for r in results)
    mean_recall_at_10 = mean(r.recall_at_10 for r in results)
    mean_mrr = mean(r.mrr for r in results)
    mean_ndcg_at_10 = mean(r.ndcg_at_10 for r in results)
    assert mean_recall_at_5 >= BASELINE_RECALL_AT_5_FLOOR, (
        f"recall@5 dropped to {mean_recall_at_5:.2%} "
        f"(floor {BASELINE_RECALL_AT_5_FLOOR:.2%})"
    )
    assert mean_recall_at_10 >= BASELINE_RECALL_AT_10_FLOOR, (
        f"recall@10 dropped to {mean_recall_at_10:.2%} "
        f"(floor {BASELINE_RECALL_AT_10_FLOOR:.2%})"
    )
    assert mean_mrr >= BASELINE_MRR_FLOOR, (
        f"MRR dropped to {mean_mrr:.3f} (floor {BASELINE_MRR_FLOOR:.3f})"
    )
    assert mean_ndcg_at_10 >= BASELINE_NDCG_AT_10_FLOOR, (
        f"nDCG@10 dropped to {mean_ndcg_at_10:.3f} "
        f"(floor {BASELINE_NDCG_AT_10_FLOOR:.3f})"
    )
