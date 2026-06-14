"""Optional MCP server adapter for scope-bound rag-core retrieval."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Final, cast

from mcp import types
from mcp.server import Server

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    SEARCH_USER_DOCUMENTS_INPUT_SCHEMA,
    SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
    SearchUserDocumentsRequest,
    SupportsContextPackPromptPayload,
    normalize_static_retrieval_scope,
    parse_search_user_documents_request,
    search_user_documents_tool_result,
    validate_bound_namespace,
    validate_search_user_documents_bounds,
)
from rag_core.contracts.tool_contract_schemas import JsonObject
from rag_core.integrations.protocols import SupportsRetrieveContext
from rag_core.search.context_pack import ContextOrder, validate_context_order

_MCP_TOOL_ERROR_TEXT: Final[str] = "rag-core MCP tool failed"

_logger = logging.getLogger(__name__)


def build_mcp_server(
    core: SupportsRetrieveContext,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    rerank: bool = False,
    limit_cap: int = 10,
    context_order: ContextOrder = "rank",
    server_name: str = "rag-core",
) -> Server[object, object]:
    """Build a stdio-ready MCP server with launch-bound retrieval scope."""
    if not callable(getattr(core, "retrieve_context", None)):
        raise TypeError("core must provide an async retrieve_context method")

    resolved_namespace = validate_bound_namespace(namespace)
    validate_search_user_documents_bounds(limit=limit_cap)
    context_order = validate_context_order(context_order)
    default_limit = min(SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT, limit_cap)
    default_corpus_ids_tuple, _ = normalize_static_retrieval_scope(
        corpus_ids=corpus_ids,
        document_ids=None,
        limit=default_limit,
    )
    default_corpus_ids = list(default_corpus_ids_tuple)

    server: Server[object, object] = Server(server_name)
    input_schema = _mcp_input_schema(default_limit=default_limit)
    output_schema = deepcopy(SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=SEARCH_USER_DOCUMENTS_TOOL_NAME,
                description=(
                    "Searches the operator-configured corpus; you cannot change "
                    "the corpus. Returns bounded prompt-safe snippets with locators."
                ),
                inputSchema=input_schema,
                outputSchema=output_schema,
                annotations=types.ToolAnnotations(
                    title="Search user documents",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, object] | None,
    ) -> dict[str, object] | types.CallToolResult:
        try:
            if name != SEARCH_USER_DOCUMENTS_TOOL_NAME:
                raise ValueError(f"unknown MCP tool: {name}")
            request = _parse_mcp_request(
                arguments or {},
                default_limit=default_limit,
                limit_cap=limit_cap,
            )
            pack = await _retrieve_mcp_context_pack(
                core,
                request=request,
                namespace=resolved_namespace,
                corpus_ids=default_corpus_ids,
                rerank=rerank,
            )
            return search_user_documents_tool_result(
                pack,
                context_order=context_order,
            )
        except Exception as exc:
            _logger.warning(
                "MCP tool call failed: tool=%s error_type=%s",
                name,
                type(exc).__name__,
                exc_info=True,
            )
            return _mcp_tool_error_result()

    return server


async def _retrieve_mcp_context_pack(
    core: SupportsRetrieveContext,
    *,
    request: SearchUserDocumentsRequest,
    namespace: str,
    corpus_ids: list[str],
    rerank: bool,
) -> SupportsContextPackPromptPayload:
    return await core.retrieve_context(
        query=request.query,
        namespace=namespace,
        corpus_ids=list(corpus_ids),
        limit=request.limit,
        content_types=None,
        document_ids=None,
        rerank=rerank,
        use_lexical_search=SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
        query_plan=None,
        max_chars=SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
        max_tokens=None,
        audit_context=None,
    )


def _parse_mcp_request(
    payload: Mapping[str, object],
    *,
    default_limit: int,
    limit_cap: int,
) -> SearchUserDocumentsRequest:
    capped_payload = dict(payload)
    raw_limit = capped_payload.get("limit")
    if isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
        capped_payload["limit"] = min(raw_limit, limit_cap)
    return parse_search_user_documents_request(
        capped_payload,
        default_limit=default_limit,
        default_rerank=SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
        default_use_lexical_search=SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
        default_max_chars=SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
        default_max_tokens=None,
    )


def _mcp_input_schema(*, default_limit: int) -> JsonObject:
    schema = deepcopy(SEARCH_USER_DOCUMENTS_INPUT_SCHEMA)
    properties = cast(dict[str, object], schema["properties"])
    for field in (
        "document_ids",
        "rerank",
        "use_lexical_search",
        "max_chars",
        "max_tokens",
    ):
        properties.pop(field, None)
    limit = cast(dict[str, object], properties["limit"])
    limit.pop("maximum", None)
    limit["default"] = default_limit
    limit["description"] = (
        "Requested maximum result count. Values above the operator cap are capped."
    )
    return schema


def _mcp_tool_error_result() -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=_MCP_TOOL_ERROR_TEXT)],
        isError=True,
    )


__all__ = ["build_mcp_server"]
