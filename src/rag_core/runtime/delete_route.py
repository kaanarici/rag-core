"""HTTP handler bits for the right-to-forget ``DELETE`` route.

Lives outside ``runtime/app.py`` so the app stays under the architecture-
pressure size threshold and so the response shape has a single owner module
that the caller's gateway can import for typed parsing.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_core.core_models import DeleteDocumentResult
from rag_core.runtime.errors import (
    RuntimeRequestError,
    api_error,
    parse_json_object,
    redact_runtime_error,
)
from rag_core.runtime.requests import parse_delete_document_request


_logger = logging.getLogger(__name__)


def delete_document_payload(result: DeleteDocumentResult) -> dict[str, object]:
    """Serialize the engine result into the HTTP response body.

    Mirrors ``DeleteDocumentResponse`` in ``docs/self-host/openapi.yaml``;
    keep them in lockstep.
    """
    return {
        "document_id": result.document_id,
        "namespace": result.namespace,
        "collection": result.collection,
        "index_deleted": result.index_deleted,
        "vector_store_acked": result.vector_store_acked,
        "sidecar_deleted": result.sidecar_deleted,
        "lexical_sidecar_purged": result.lexical_sidecar_purged,
        "embedding_cache_purged": result.embedding_cache_purged,
        "chunk_context_cache_purged": result.chunk_context_cache_purged,
        "manifest_entry_deleted": result.manifest_entry_deleted,
        "manifest_removed": result.manifest_removed,
    }


async def handle_delete_document(
    request: Any,
    *,
    core: Any,
    invalid_request_response: Any,
    bound_namespace: str | None = None,
) -> "Any":
    """Drive the DELETE route end-to-end against a resolved Engine.

    Imports Starlette types lazily so the optional runtime dep stays
    optional. The annotated return type is ``Any`` because Starlette is an
    optional install; the caller in ``runtime/app.py`` already declares the
    concrete ``JSONResponse`` return.

    ``bound_namespace`` mirrors the ingest/search seams: when the process is
    bound to a single tenant the body field must match exactly, so the
    gateway cannot mint a cross-tenant delete by editing the request body.
    """
    from starlette.responses import JSONResponse

    document_id = request.path_params.get("document_id", "")
    body = await parse_json_object(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        delete_request = parse_delete_document_request(
            document_id=document_id,
            payload=body,
            bound_namespace=bound_namespace,
        )
    except RuntimeRequestError as exc:
        return invalid_request_response(exc)
    try:
        result = await core.delete_document(
            document_id=delete_request.document_id,
            namespace=delete_request.namespace,
            collection=delete_request.collection,
        )
    except ValueError as exc:
        # ``CollectionPolicyViolation`` is a ``ValueError`` subclass; the engine
        # already refuses cross-namespace / unbound-tier deletes at the
        # seam. Surface a sanitized 400. The full error stays in logs.
        redacted = redact_runtime_error(exc, error_code="delete_refused")
        _logger.warning(
            "delete refused: document_id=%s error_type=%s error_code=%s",
            delete_request.document_id,
            redacted.error_type,
            redacted.error_code,
            exc_info=exc,
        )
        return api_error(
            code="invalid_request",
            message="delete refused by collection policy",
            status_code=400,
            details={"error_type": redacted.error_type},
        )
    return JSONResponse(delete_document_payload(result))


__all__ = [
    "delete_document_payload",
    "handle_delete_document",
]
