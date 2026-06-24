"""Minimal Starlette HTTP runtime over ``Engine``."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.cli.output import search_hit_payload
from rag_core.events.event_types import (
    INGEST_COMPLETED_EVENT,
    INGEST_STARTED_EVENT,
)
from rag_core.events.sinks import describe_event_sink_status
from rag_core.events.types import AuditContext
from rag_core.runtime_defaults import (
    DEFAULT_RUNTIME_INGEST_CONCURRENCY,
    DEFAULT_RUNTIME_JOB_DB_PATH,
    DEFAULT_RUNTIME_MAX_BODY_BYTES,
)
from rag_core.runtime.delete_route import handle_delete_document
from rag_core.runtime.errors import (
    RuntimeRequestError,
    api_error,
    parse_json_object,
    redact_runtime_error,
)
from rag_core.runtime.health import (
    liveness_payload,
    readiness_payload,
    readiness_status_code,
)
from rag_core.runtime.job_events import (
    ingest_job_event_stream_headers,
    stream_ingest_job_events,
)
from rag_core.runtime.jobs import (
    INGEST_JOB_STATUS_COMPLETED,
    INGEST_JOB_STATUS_FAILED,
    INGEST_JOB_STATUS_RUNNING,
    IngestJobStore,
    ingest_job_status_payload,
)
from rag_core.runtime.paths import normalize_ingest_roots, read_validated_ingest_file
from rag_core.runtime.requests import (
    DEFAULT_RUNTIME_CONTEXT_LIMIT,
    DEFAULT_RUNTIME_SEARCH_LIMIT,
    parse_ingest_request,
    parse_retrieval_request,
)
from rag_core.search.context_pack import context_pack_response_payload
from starlette.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from rag_core.core import Engine
    from rag_core.core_models import Config


_logger = logging.getLogger(__name__)


# Header surface for caller-supplied audit correlation. The gateway typically
# mints ``X-Request-Id``; when it is absent the runtime mints a UUIDv4 so
# every request has a stable id threaded through audit events / log lines /
# the response itself (returned in the ``X-Request-Id`` response header).
_HEADER_REQUEST_ID = "x-request-id"
_HEADER_ACTOR = "x-actor"
_HEADER_INGEST_ID = "x-ingest-id"

# Per-request state attached to ``request.scope["state"]`` by the request_id
# middleware so downstream handlers/middleware can read the minted id without
# re-parsing headers.
_SCOPE_REQUEST_ID = "rag_core_request_id"


def _request_id_from_scope(request: Any) -> str | None:
    state = getattr(request, "state", None)
    if state is None:
        return None
    return getattr(state, _SCOPE_REQUEST_ID, None)


def _audit_context_from_headers(request: Any) -> AuditContext | None:
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    actor = headers.get(_HEADER_ACTOR)
    ingest_id = headers.get(_HEADER_INGEST_ID)
    request_id = _request_id_from_scope(request) or headers.get(_HEADER_REQUEST_ID)
    if not (actor or request_id or ingest_id):
        return None
    return AuditContext(
        actor=actor or None,
        request_id=request_id or None,
        ingest_id=ingest_id or None,
    )


def create_app(
    *,
    config: Config,
    core_factory: Callable[..., Engine],
    job_db_path: Path | None = None,
    ingest_roots: tuple[Path, ...] | None = None,
    job_retention_seconds: float | None = None,
    max_body_bytes: int = DEFAULT_RUNTIME_MAX_BODY_BYTES,
    ingest_concurrency: int = DEFAULT_RUNTIME_INGEST_CONCURRENCY,
) -> Any:
    from starlette.applications import Starlette
    from starlette.background import BackgroundTask
    from starlette.exceptions import HTTPException
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.routing import Route

    from rag_core.runtime.body_cap import BodyCapMiddleware

    jobs = IngestJobStore(
        job_db_path or DEFAULT_RUNTIME_JOB_DB_PATH,
        max_age_seconds=job_retention_seconds,
    )
    allowed_ingest_roots = normalize_ingest_roots(ingest_roots)
    shared_core: Engine | None = None
    core_init_lock = asyncio.Lock()

    bound_namespace = _bound_namespace_from_config(config)

    # Per-route ingest semaphore. ``asyncio.Semaphore`` lives at module level
    # of the app instance so the cap applies across all concurrent ingest
    # requests in this process. The control-plane routes (search,
    # context retrieval, delete, health) do not consume this fence. That's
    # what uvicorn ``--limit-concurrency`` is for.
    ingest_semaphore = asyncio.Semaphore(ingest_concurrency)

    async def _get_or_create_core() -> Engine:
        # Publish the core only after ``ensure_ready`` succeeds: concurrent
        # callers must not see a half-initialized instance, and a failed
        # startup must not be cached as the process-lifetime core.
        nonlocal shared_core
        if shared_core is not None:
            return shared_core
        async with core_init_lock:
            if shared_core is None:
                core = core_factory(config)
                try:
                    await core.ensure_ready()
                except BaseException:
                    try:
                        await core.close()
                    except Exception:
                        _logger.warning(
                            "failed to close core after ensure_ready failure",
                            exc_info=True,
                        )
                    raise
                shared_core = core
        return shared_core

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(liveness_payload())

    async def health_ready(_: Request) -> JSONResponse:
        checks: dict[str, object] = {"core": {"status": "pending"}}
        event_sink_status: dict[str, object] | None = None
        try:
            core = await _get_or_create_core()
            checks["core"] = {"status": "ok"}
            store_health = await core.check_health()
            checks["vector_store"] = store_health
            ready = bool(store_health.get("ok", store_health.get("healthy", False)))
            describe_status = getattr(core, "describe_event_sink_status", None)
            if callable(describe_status):
                event_sink_status = describe_status()
            else:
                event_sink_status = describe_event_sink_status(None)
            payload = readiness_payload(
                ready=ready,
                checks=checks,
                event_sink_status=event_sink_status,
            )
            return JSONResponse(payload, status_code=readiness_status_code(ready=ready))
        except Exception as exc:
            # Mirror qdrant_health._build_unhealthy_health: only the exception
            # class name and a stable error_code escape; ``str(exc)`` can
            # include SDK error strings or licensed-source identifiers, so it
            # stays in the structured log only.
            redacted = redact_runtime_error(exc, error_code="unhealthy")
            _logger.warning(
                "runtime readiness check failed: error_type=%s error_code=%s",
                redacted.error_type,
                redacted.error_code,
                exc_info=exc,
            )
            checks["core"] = {
                "status": "error",
                "error_type": redacted.error_type,
                "error_code": redacted.error_code,
            }
            payload = readiness_payload(
                ready=False,
                checks=checks,
                event_sink_status=event_sink_status,
            )
            return JSONResponse(payload, status_code=503)

    async def runtime_description(_: Request) -> JSONResponse:
        core = await _get_or_create_core()
        return JSONResponse(core.describe_runtime())

    async def ingest(request: Request) -> JSONResponse:
        body = await parse_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            ingest_request = parse_ingest_request(
                body,
                bound_namespace=bound_namespace,
            )
            ingest_path, ingest_bytes = await asyncio.to_thread(
                read_validated_ingest_file,
                ingest_request.path,
                roots=allowed_ingest_roots,
            )
        except RuntimeRequestError as exc:
            return _invalid_request(exc)
        if ingest_semaphore.locked():
            # Best-effort admission check: ``locked()`` is True once every
            # slot is in use. The semaphore itself is acquired later in the
            # background worker, so a burst that passes this check still
            # queues on the semaphore. The cap strictly bounds *running*
            # ingests, while this 503 sheds load once the cap is visibly
            # saturated at admission time. The caller retries with backoff.
            return api_error(
                code="busy",
                message="ingest is at the per-process concurrency cap",
                status_code=503,
            )
        audit_context = _audit_context_from_headers(request)
        record = jobs.create(
            path=str(ingest_path),
            namespace=ingest_request.namespace,
            collection=ingest_request.collection,
        )
        # Audit log line at the route boundary. The full event-sink ingest
        # events are emitted by ``Engine.add_file`` once the worker
        # starts; this line lets a compliance reader correlate "we accepted
        # an ingest job for this (request_id, actor) at the HTTP edge."
        _logger.info(
            "audit event=%s request_id=%s actor=%s ingest_id=%s job_id=%s "
            "namespace=%s collection=%s",
            INGEST_STARTED_EVENT,
            audit_context.request_id if audit_context else None,
            audit_context.actor if audit_context else None,
            audit_context.ingest_id if audit_context else None,
            record.job_id,
            ingest_request.namespace,
            ingest_request.collection,
        )
        task = BackgroundTask(
            _run_ingest_job,
            jobs=jobs,
            record=record,
            file_bytes=ingest_bytes,
            core=_get_or_create_core,
            audit_context=audit_context,
            semaphore=ingest_semaphore,
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
        return JSONResponse(ingest_job_status_payload(record))

    async def ingest_events(request: Request) -> JSONResponse | StreamingResponse:
        job_id = request.path_params["job_id"]
        record = jobs.get(job_id)
        if record is None:
            return api_error(
                code="not_found",
                message="Ingest job not found",
                status_code=404,
                details={"job_id": job_id},
            )
        return StreamingResponse(
            stream_ingest_job_events(jobs, job_id, initial_record=record),
            media_type="text/event-stream",
            headers=ingest_job_event_stream_headers(),
        )

    async def search(request: Request) -> JSONResponse:
        body = await parse_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            retrieval_request = parse_retrieval_request(
                body,
                default_limit=DEFAULT_RUNTIME_SEARCH_LIMIT,
                bound_namespace=bound_namespace,
            )
        except RuntimeRequestError as exc:
            return _invalid_request(exc)
        core = await _get_or_create_core()
        hits = await core.search(
            query=retrieval_request.query,
            namespace=retrieval_request.namespace,
            collections=list(retrieval_request.collections),
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
            audit_context=_audit_context_from_headers(request),
        )
        return JSONResponse([search_hit_payload(hit) for hit in hits])

    async def delete_document(request: Request) -> JSONResponse:
        core = await _get_or_create_core()
        response = await handle_delete_document(
            request,
            core=core,
            invalid_request_response=_invalid_request,
            bound_namespace=bound_namespace,
        )
        return response  # type: ignore[no-any-return]

    async def retrieve_context(request: Request) -> JSONResponse:
        body = await parse_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            retrieval_request = parse_retrieval_request(
                body,
                default_limit=DEFAULT_RUNTIME_CONTEXT_LIMIT,
                allow_context_budget=True,
                bound_namespace=bound_namespace,
            )
        except RuntimeRequestError as exc:
            return _invalid_request(exc)
        core = await _get_or_create_core()
        pack = await core.context(
            query=retrieval_request.query,
            namespace=retrieval_request.namespace,
            collections=list(retrieval_request.collections),
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
            audit_context=_audit_context_from_headers(request),
        )
        return JSONResponse(
            context_pack_response_payload(
                pack,
                context_order=retrieval_request.context_order,
            )
        )

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        # Startup orphan sweep: rag-core does not resume in-flight ingest jobs
        # across restarts. Any row that was ``pending`` or ``running`` when
        # the prior process died is flipped to ``failed`` with a sanitized
        # ``OrphanedByRestart`` payload so the gateway / poller sees a
        # terminal status. Retry orchestration lives in the gateway, not
        # here. See https://kaanarici.github.io/rag-core/docs/self-host.
        orphaned = jobs.reconcile_orphaned_jobs()
        if orphaned:
            _logger.warning(
                "runtime startup orphan sweep flipped %d ingest job(s) to failed: %s",
                len(orphaned),
                ",".join(orphaned),
            )
        yield
        nonlocal shared_core
        if shared_core is not None:
            await shared_core.close()
            shared_core = None

    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        return _stamp_request_id(
            request,
            api_error(
                code="internal_error",
                message="Unexpected server error",
                status_code=500,
                details={"error": type(exc).__name__},
            ),
        )

    async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, HTTPException):
            return await unhandled_exception(request, exc)
        if exc.status_code == 404:
            return _stamp_request_id(
                request,
                api_error(
                    code="not_found",
                    message="Route not found",
                    status_code=404,
                ),
            )
        return _stamp_request_id(
            request,
            api_error(
                code="http_error",
                message=str(exc.detail),
                status_code=exc.status_code,
            ),
        )

    class _RequestIdMiddleware(BaseHTTPMiddleware):
        """Mint a UUIDv4 ``X-Request-Id`` when the caller didn't send one.

        The id is attached to ``request.state`` so handlers and audit log
        lines see the same value, then echoed back on the response so the
        caller's logs can correlate.
        """

        async def dispatch(self, request: Any, call_next: Any) -> Any:
            incoming = request.headers.get(_HEADER_REQUEST_ID, "").strip()
            request_id = incoming or uuid.uuid4().hex
            setattr(request.state, _SCOPE_REQUEST_ID, request_id)
            response = await call_next(request)
            response.headers[_HEADER_REQUEST_ID] = request_id
            return response

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/health/ready", health_ready, methods=["GET"]),
        Route("/v1/runtime", runtime_description, methods=["GET"]),
        Route("/v1/ingest", ingest, methods=["POST"]),
        Route("/v1/ingest/{job_id}", ingest_status, methods=["GET"]),
        Route("/v1/ingest/{job_id}/events", ingest_events, methods=["GET"]),
        Route("/v1/search", search, methods=["POST"]),
        Route("/v1/search/context", retrieve_context, methods=["POST"]),
        Route("/v1/documents/{document_id}", delete_document, methods=["DELETE"]),
    ]
    return Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[
            # Outermost: stamp request_id so the body-cap middleware below can
            # use it on 413 responses. ``Middleware`` is applied outside-in.
            Middleware(_RequestIdMiddleware),
            Middleware(BodyCapMiddleware, cap_bytes=max_body_bytes),
        ],
        exception_handlers={
            Exception: unhandled_exception,
            HTTPException: http_exception_handler,
        },
    )


def _bound_namespace_from_config(config: Any) -> str | None:
    """Extract the process-bound tenant namespace from CollectionPolicy, if set.

    Returns ``None`` for unbound deployments. The existing library-only /
    dev-server case. A bound CollectionPolicy makes the HTTP boundary refuse
    cross-namespace requests before they reach the engine.
    """
    policy = getattr(config, "collection_policy", None)
    if policy is None:
        return None
    return getattr(policy, "bound_namespace", None)


def _stamp_request_id(request: Any, response: JSONResponse) -> JSONResponse:
    request_id = _request_id_from_scope(request)
    if request_id is not None:
        response.headers[_HEADER_REQUEST_ID] = request_id
    return response


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
    file_bytes: bytes,
    core: Callable[[], Awaitable[Engine]],
    audit_context: AuditContext | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> None:
    if semaphore is not None:
        await semaphore.acquire()
    try:
        jobs.update(record.job_id, status=INGEST_JOB_STATUS_RUNNING)
        try:
            rag = await core()
            document = await rag.add_file(
                Path(record.path),
                namespace=record.namespace,
                collection=record.collection,
                audit_context=audit_context,
                ingest_id=audit_context.ingest_id if audit_context else None,
                pre_read_bytes=file_bytes,
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
            _logger.info(
                "audit event=%s request_id=%s actor=%s ingest_id=%s "
                "job_id=%s namespace=%s collection=%s document_id=%s",
                INGEST_COMPLETED_EVENT,
                audit_context.request_id if audit_context else None,
                audit_context.actor if audit_context else None,
                audit_context.ingest_id if audit_context else None,
                record.job_id,
                record.namespace,
                record.collection,
                document.document_id,
            )
        except Exception as exc:
            # Sanitized public surface: the SQLite row only stores error_type
            # and a stable error_code. Full ``str(exc)`` text, which can
            # carry SDK message strings or licensed-source identifiers, is
            # logged with ``exc_info`` so the structured event/log sink keeps
            # the diagnostic tail without leaking it onto the HTTP job body.
            redacted = redact_runtime_error(exc, error_code="ingest_failed")
            _logger.warning(
                "ingest job failed: job_id=%s error_type=%s error_code=%s",
                record.job_id,
                redacted.error_type,
                redacted.error_code,
                exc_info=exc,
            )
            jobs.update(
                record.job_id,
                status=INGEST_JOB_STATUS_FAILED,
                error_type=redacted.error_type,
                error_code=redacted.error_code,
            )
    finally:
        if semaphore is not None:
            semaphore.release()
