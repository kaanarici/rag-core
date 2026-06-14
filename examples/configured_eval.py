"""Configured eval over a folder with real embeddings."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, LOCAL_EMBEDDING_MODEL, QdrantConfig
from rag_core.evals import (
    eval_report,
    load_cases,
    run_eval,
)
from rag_core.search import search_profile

NAMESPACE, CORPUS_ID = "acme", "help-center"


def _config(provider: str) -> tuple[RAGCoreConfig, str]:
    if provider == "local":
        return RAGCoreConfig.local(), LOCAL_EMBEDDING_MODEL
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --provider openai.")
    return (
        RAGCoreConfig(
            qdrant=QdrantConfig(
                location=":memory:",
                collection="rag_core_configured_eval",
                dimension_aware_collection=False,
            ),
            embedding=EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-small",
                api_key=api_key,
            ),
        ),
        "text-embedding-3-small",
    )


async def run(corpus_dir: Path, cases_jsonl: Path, *, provider: str) -> dict[str, object]:
    config, embedding_model = _config(provider)
    files = sorted(path for path in corpus_dir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"no files found in {corpus_dir}")
    async with RAGCore(config) as core:
        for path in files:
            document_key = path.relative_to(corpus_dir).as_posix()
            mime_type = "text/markdown" if path.suffix.lower() == ".md" else "text/plain"
            await core.ingest_bytes(file_bytes=path.read_bytes(), filename=document_key, mime_type=mime_type, namespace=NAMESPACE, corpus_id=CORPUS_ID, document_id=document_key, document_key=document_key)
        results = await run_eval(core, load_cases(cases_jsonl), rerank=False, query_plan=search_profile("balanced", limit=10))
    report = eval_report(
        results,
        run={"mode": provider, "vector_store": "embedded_qdrant", "embedding_model": embedding_model, "search_profile": "balanced", "rerank": False},
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real-embedding retrieval eval over a local folder.")
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("cases_jsonl", type=Path)
    parser.add_argument("--provider", choices=("local", "openai"), default="local")
    args = parser.parse_args(argv)
    if not args.corpus_dir.is_dir():
        parser.error(f"corpus_dir must be a folder: {args.corpus_dir}")
    if not args.cases_jsonl.is_file():
        parser.error(f"cases_jsonl must be a JSONL file: {args.cases_jsonl}")
    try:
        report = asyncio.run(run(args.corpus_dir, args.cases_jsonl, provider=args.provider))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    failure_count = report.get("failure_count")
    return 1 if isinstance(failure_count, int) and failure_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
