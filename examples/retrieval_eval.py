"""Retrieval eval over a local help-center corpus."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag_core.demo import build_demo_core
from rag_core.evals import (
    add_quality_gate,
    eval_exit_code,
    eval_report,
    load_cases,
    redact_eval_report,
    run_eval,
)
from rag_core.search import search_profile

EXAMPLES_DIR = Path(__file__).parent
CASES_PATH = EXAMPLES_DIR / "eval_cases.jsonl"
CORPUS_DIR = EXAMPLES_DIR / "demo_corpus"

QUALITY_GATE = {
    "recall_at_5": {"minimum": 1.0},
    "mrr": {"minimum": 1.0},
    "latency_p95_ms": {"maximum": 250.0},
}


async def run_demo() -> dict[str, object]:
    async with build_demo_core(collection="retrieval_eval") as core:
        for path in sorted(CORPUS_DIR.glob("*.md")):
            await core.ingest_bytes(
                file_bytes=path.read_bytes(),
                filename=path.name,
                mime_type="text/markdown",
                namespace="acme",
                corpus_id="help-center",
                document_id=path.name,
                document_key=path.name,
            )
        results = await run_eval(
            core,
            load_cases(CASES_PATH),
            rerank=False,
            query_plan=search_profile("balanced", limit=10),
        )

    report = eval_report(
        results,
        run={
            "mode": "local",
            "vector_store": "embedded_qdrant",
            "embedding_model": "demo-dense-v1",
            "search_profile": "balanced",
            "rerank": False,
        },
    )
    add_quality_gate(report, {"eval": report}, QUALITY_GATE)
    return report


def main() -> None:
    report = asyncio.run(run_demo())
    print(json.dumps(redact_eval_report(report), indent=2, sort_keys=True))
    raise SystemExit(eval_exit_code(report))


if __name__ == "__main__":
    main()
