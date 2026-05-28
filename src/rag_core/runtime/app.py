"""Minimal Starlette HTTP runtime over ``RAGCore``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.cli_output import search_hit_payload
from rag_core.runtime_defaults import DEFAULT_RUNTIME_JOB_DB_PATH
from rag_core.runtime.errors import RuntimeRequestError, api_error, parse_json_object
from rag_core.runtime.health import (
    liveness_payload,
    readiness_payload,
    readiness_status_code,
)
from rag_core.runtime.jobs import (
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
    INGEST_JOB_STATUS_RUNNING,
    IngestJobStore,
)
from rag_core.runtime.paths import normalize_ingest_roots, validate_ingest_path
from rag_core.runtime.requests import (
    DEFAULT_RUNTIME_CONTEXT_LIMIT,
    DEFAULT_RUNTIME_SEARCH_LIMIT,
    parse_ingest_request,
    parse_retrieval_request,
)
from rag_core.search.context_pack import context_pack_response_payload
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from rag_core.core import RAGCore
    from rag_core.core_models import RAGCoreConfig


def create_app(
    *,
    config: RAGCoreConfig,
    core_factory: Callable[..., RAGCore],
    job_db_path: Path | None = None,
    ingest_roots: tuple[Path, ...] | None = None,
) -> Any:
    from starlette.applications import Starlette
    from starlette.background import BackgroundTask
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.routing import Route

    jobs = IngestJobStore(job_db_path or DEFAULT_RUNTIME_JOB_DB_PATH)
    allowed_ingest_roots = normalize_ingest_roots(ingest_roots)
    shared_core: RAGCore | None = None

    async def _get_or_create_core() -> RAGCore:
        nonlocal shared_core
        if shared_core is None:
            shared_core = core_factory(config)
            await shared_core.ensure_ready()
        return shared_core

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(liveness_payload())

    async def health_ready(_: Request) -> JSONResponse:
        checks: dict[str, object] = {"core": {"status": "pending"}}
        try:
            core = await _get_or_create_core()
            checks["core"] = {"status": "ok"}
            store_health = await core.check_health()
            checks["vector_store"] = store_health
            ready = bool(store_health.get("ok", store_health.get("healthy", False)))
            payload = readiness_payload(ready=ready, checks=checks)
            return JSONResponse(payload, status_code=readiness_status_code(ready=ready))
        except Exception as exc:
            checks["core"] = {
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc),
            }
            payload = readiness_payload(ready=False, checks=checks)
            return JSONResponse(payload, status_code=503)

    async def runtime_description(_: Request) -> JSONResponse:
        core = await _get_or_create_core()
        return JSONResponse(core.describe_runtime())

    async def ingest(request: Request) -> JSONResponse:
        body = await parse_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            ingest_request = parse_ingest_request(body)
            ingest_path = validate_ingest_path(
                ingest_request.path,
                roots=allowed_ingest_roots,
            )
        except RuntimeRequestError as exc:
            return _invalid_request(exc)
        record = jobs.create(
            path=str(ingest_path),
            namespace=ingest_request.namespace,
            corpus_id=ingest_request.corpus_id,
        )
        task = BackgroundTask(
            _run_ingest_job,
            jobs=jobs,
            record=record,
            core=_get_or_create_core,
        )
        return JSONResponse(
            {"job_id": record.job_id, "status": record.status},
            status_code=202,
            background=task,
        )

    async def ingest_status(request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        record = jobs.get(job_id)
        if record is None:
            return api_error(
                code="not_found",
                message="Ingest job not found",
                status_code=404,
                details={"job_id": job_id},
            )
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
        body = await parse_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            retrieval_request = parse_retrieval_request(
                body,
                default_limit=DEFAULT_RUNTIME_SEARCH_LIMIT,
            )
        except RuntimeRequestError as exc:
            return _invalid_request(exc)
        core = await _get_or_create_core()
        hits = await core.search(
            query=retrieval_request.query,
            namespace=retrieval_request.namespace,
            corpus_ids=list(retrieval_request.corpus_ids),
            limit=retrieval_request.limit,
            content_types=(
                list(retrieval_request.content_types)
                if retrieval_request.content_types is not None
                else None
            ),
            document_ids=(
                list(retrieval_request.document_ids)
                if retrieval_request.document_ids is not None
                else None
            ),
            rerank=retrieval_request.rerank,
            use_lexical_search=retrieval_request.use_lexical_search,
        )
        return JSONResponse([search_hit_payload(hit) for hit in hits])

    async def retrieve_context(request: Request) -> JSONResponse:
        body = await parse_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            retrieval_request = parse_retrieval_request(
                body,
                default_limit=DEFAULT_RUNTIME_CONTEXT_LIMIT,
                allow_context_budget=True,
            )
        except RuntimeRequestError as exc:
            return _invalid_request(exc)
        core = await _get_or_create_core()
        pack = await core.retrieve_context(
            query=retrieval_request.query,
            namespace=retrieval_request.namespace,
            corpus_ids=list(retrieval_request.corpus_ids),
            limit=retrieval_request.limit,
            content_types=(
                list(retrieval_request.content_types)
                if retrieval_request.content_types is not None
                else None
            ),
            document_ids=(
                list(retrieval_request.document_ids)
                if retrieval_request.document_ids is not None
                else None
            ),
            rerank=retrieval_request.rerank,
            use_lexical_search=retrieval_request.use_lexical_search,
            max_chars=retrieval_request.max_chars,
            max_tokens=retrieval_request.max_tokens,
        )
        return JSONResponse(context_pack_response_payload(pack))

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        yield
        nonlocal shared_core
        if shared_core is not None:
            await shared_core.close()
            shared_core = None

    async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
        return api_error(
            code="internal_error",
            message="Unexpected server error",
            status_code=500,
            details={"error": type(exc).__name__},
        )

    async def http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, HTTPException):
            return await unhandled_exception(_, exc)
        if exc.status_code == 404:
            return api_error(
                code="not_found",
                message="Route not found",
                status_code=404,
            )
        return api_error(
            code="http_error",
            message=str(exc.detail),
            status_code=exc.status_code,
        )

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/health/ready", health_ready, methods=["GET"]),
        Route("/v1/runtime", runtime_description, methods=["GET"]),
        Route("/v1/ingest", ingest, methods=["POST"]),
        Route("/v1/ingest/{job_id}", ingest_status, methods=["GET"]),
        Route("/v1/search", search, methods=["POST"]),
        Route("/v1/retrieve-context", retrieve_context, methods=["POST"]),
    ]
    return Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={
            Exception: unhandled_exception,
            HTTPException: http_exception_handler,
        },
    )


def _invalid_request(exc: RuntimeRequestError) -> JSONResponse:
    return api_error(
        code="invalid_request",
        message=exc.message,
        status_code=400,
        details=exc.details,
    )


async def _run_ingest_job(
    *,
    jobs: IngestJobStore,
    record: Any,
    core: Callable[[], Awaitable[RAGCore]],
) -> None:
    jobs.update(record.job_id, status=INGEST_JOB_STATUS_RUNNING)
    try:
        rag = await core()
        document = await rag.ingest_file(
            Path(record.path),
            namespace=record.namespace,
            corpus_id=record.corpus_id,
        )
        jobs.update(
            record.job_id,
            status=INGEST_JOB_STATUS_COMPLETED,
            result={
                "document_id": document.document_id,
                "chunk_count": document.chunk_count,
                "ingest_state": document.ingest_state,
            },
        )
    except Exception as exc:
        jobs.update(
            record.job_id,
            status=INGEST_JOB_STATUS_FAILED,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
