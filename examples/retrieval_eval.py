"""Wiring-only retrieval eval over a local corpus with demo hash embeddings."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag_core.demo import build_demo_core
from rag_core.evals import (
    eval_report,
    load_cases,
    redact_eval_report,
    run_eval,
)
from rag_core.search import search_profile

EXAMPLES_DIR = Path(__file__).parent
CASES_PATH = EXAMPLES_DIR / "eval_cases.jsonl"
CORPUS_DIR = EXAMPLES_DIR / "demo_corpus"


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
            "purpose": "wiring_only_demo_embeddings",
            "vector_store": "embedded_qdrant",
            "embedding_model": "demo-dense-v1",
            "search_profile": "balanced",
            "rerank": False,
        },
    )
    return report


def main() -> None:
    report = asyncio.run(run_demo())
    print(json.dumps(redact_eval_report(report), indent=2, sort_keys=True))
    failure_count = report.get("failure_count")
    raise SystemExit(1 if isinstance(failure_count, int) and failure_count > 0 else 0)


if __name__ == "__main__":
    main()
