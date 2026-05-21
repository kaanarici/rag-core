"""Minimal Starlette HTTP runtime over ``RAGCore``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.cli_output import search_hit_payload
from rag_core.runtime.jobs import IngestJobStore

if TYPE_CHECKING:
    from rag_core.core import RAGCore
    from rag_core.core_models import RAGCoreConfig


def create_app(
    *,
    config: RAGCoreConfig,
    core_factory: Callable[..., RAGCore],
    job_db_path: Path | None = None,
) -> Any:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    jobs = IngestJobStore(job_db_path or Path(".rag-core/runtime/jobs.sqlite3"))

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    async def runtime_description(_: Request) -> JSONResponse:
        core = core_factory(config)
        try:
            await core.ensure_ready()
            payload = core.describe_runtime()
        finally:
            await core.close()
        return JSONResponse(payload)

    async def ingest(request: Request) -> JSONResponse:
        body = await request.json()
        path = str(body.get("path") or "")
        namespace = str(body.get("namespace") or "")
        corpus_id = str(body.get("corpus_id") or body.get("corpusId") or "")
        if not path or not namespace or not corpus_id:
            return JSONResponse(
                {"error": "path, namespace, and corpus_id are required"},
                status_code=400,
            )
        record = jobs.create(path=path, namespace=namespace, corpus_id=corpus_id)
        asyncio.create_task(
            _run_ingest_job(
                jobs=jobs,
                record=record,
                core_factory=core_factory,
                config=config,
            )
        )
        return JSONResponse({"job_id": record.job_id, "status": record.status}, status_code=202)

    async def ingest_status(request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        record = jobs.get(job_id)
        if record is None:
            return JSONResponse({"error": "job not found"}, status_code=404)
        payload: dict[str, object] = {
            "job_id": record.job_id,
            "status": record.status,
            "path": record.path,
            "namespace": record.namespace,
            "corpus_id": record.corpus_id,
        }
        if record.result is not None:
            payload["result"] = record.result
        if record.error is not None:
            payload["error"] = record.error
        return JSONResponse(payload)

    async def search(request: Request) -> JSONResponse:
        body = await request.json()
        query = str(body.get("query") or "")
        namespace = str(body.get("namespace") or "")
        corpus_ids = body.get("corpus_ids") or body.get("corpusIds") or []
        if not query or not namespace or not isinstance(corpus_ids, list):
            return JSONResponse(
                {"error": "query, namespace, and corpus_ids are required"},
                status_code=400,
            )
        limit = int(body.get("limit") or 10)
        rerank = bool(body.get("rerank") or False)
        core = core_factory(config)
        try:
            await core.ensure_ready()
            hits = await core.search(
                query=query,
                namespace=namespace,
                corpus_ids=[str(value) for value in corpus_ids],
                limit=limit,
                rerank=rerank,
            )
        finally:
            await core.close()
        return JSONResponse([search_hit_payload(hit) for hit in hits])

    async def retrieve_context(request: Request) -> JSONResponse:
        body = await request.json()
        query = str(body.get("query") or "")
        namespace = str(body.get("namespace") or "")
        corpus_ids = body.get("corpus_ids") or body.get("corpusIds") or []
        if not query or not namespace or not isinstance(corpus_ids, list):
            return JSONResponse(
                {"error": "query, namespace, and corpus_ids are required"},
                status_code=400,
            )
        limit = int(body.get("limit") or 10)
        rerank = bool(body.get("rerank") or False)
        core = core_factory(config)
        try:
            await core.ensure_ready()
            pack = await core.retrieve_context(
                query=query,
                namespace=namespace,
                corpus_ids=[str(value) for value in corpus_ids],
                limit=limit,
                rerank=rerank,
            )
        finally:
            await core.close()
        return JSONResponse({"context_text": pack.as_text(), **pack.to_payload()})

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/v1/runtime", runtime_description, methods=["GET"]),
        Route("/v1/ingest", ingest, methods=["POST"]),
        Route("/v1/ingest/{job_id}", ingest_status, methods=["GET"]),
        Route("/v1/search", search, methods=["POST"]),
        Route("/v1/retrieve-context", retrieve_context, methods=["POST"]),
    ]
    return Starlette(routes=routes)


async def _run_ingest_job(
    *,
    jobs: IngestJobStore,
    record: Any,
    core_factory: Callable[..., RAGCore],
    config: RAGCoreConfig,
) -> None:
    jobs.update(record.job_id, status="running")
    core = core_factory(config)
    try:
        await core.ensure_ready()
        document = await core.ingest_file(
            Path(record.path),
            namespace=record.namespace,
            corpus_id=record.corpus_id,
        )
        jobs.update(
            record.job_id,
            status="completed",
            result={
                "document_id": document.document_id,
                "chunk_count": document.chunk_count,
                "ingest_state": document.ingest_state,
            },
        )
    except Exception as exc:
        jobs.update(record.job_id, status="failed", error=type(exc).__name__)
    finally:
        await core.close()
