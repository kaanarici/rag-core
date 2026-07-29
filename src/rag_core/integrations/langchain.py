"""Optional LangChain/LangGraph adapters for rag-core retrieval."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
    normalize_static_content_types,
    normalize_static_retrieval_scope,
    parse_search_user_documents_request,
    scope_document_ids,
    validate_bound_namespace,
    validate_search_user_documents_bounds,
)
from rag_core import Engine
from rag_core.events.types import AuditContext
from rag_core.integrations.langchain_retriever import (
    search_langchain_documents as _search,
)
from rag_core.integrations.langchain_payloads import (
    context_pack_to_tool_output,
    search_result_to_document_kwargs,
)
from rag_core.integrations.langchain_runtime import (
    LangChainNotInstalledError,
    require_langchain_symbol as _require_symbol,
    run_coro_blocking as _run_coro_blocking,
)
from rag_core.search import QueryPlan
from rag_core.search.context_pack import ContextOrder, validate_context_order
from rag_core.scope import normalize_namespace, resolve_collections_argument


@dataclass(frozen=True)
class LangChainRetrieverConfig:
    """Static retrieval scope for the LangChain adapter."""

    namespace: str
    collections: tuple[str, ...]
    limit: int = SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT
    content_types: tuple[str, ...] | None = None
    document_ids: tuple[str, ...] | None = None
    rerank: bool = SEARCH_USER_DOCUMENTS_DEFAULT_RERANK
    use_lexical_search: bool = SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH
    query_plan: QueryPlan | None = None
    audit_context: AuditContext | None = None


def build_langchain_retriever(
    core: Engine,
    *,
    collection: str | None = None,
    collections: Sequence[str] | None = None,
    namespace: str | None = None,
    limit: int = SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    content_types: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    rerank: bool = SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    use_lexical_search: bool = SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    query_plan: QueryPlan | None = None,
    audit_context: AuditContext | None = None,
) -> Any:
    """Build a ``BaseRetriever`` backed by ``Engine.search``."""

    normalized_namespace = validate_bound_namespace(normalize_namespace(namespace))
    validate_search_user_documents_bounds(
        limit=limit,
        max_chars=SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
        max_tokens=None,
    )
    collections_tuple, document_ids_tuple = normalize_static_retrieval_scope(
        collections=resolve_collections_argument(
            collection=collection,
            collections=collections,
            caller="build_langchain_retriever",
        ),
        document_ids=document_ids,
        limit=limit,
    )
    content_types_tuple = normalize_static_content_types(content_types)
    config = LangChainRetrieverConfig(
        namespace=normalized_namespace,
        collections=collections_tuple,
        limit=limit,
        content_types=content_types_tuple,
        document_ids=document_ids_tuple,
        rerank=rerank,
        use_lexical_search=use_lexical_search,
        query_plan=query_plan,
        audit_context=audit_context,
    )

    BaseRetrieverType = cast(
        type[Any],
        _require_symbol("langchain_core.retrievers", "BaseRetriever"),
    )
    Document = _require_symbol("langchain_core.documents", "Document")

    class _EngineRetriever(BaseRetrieverType):  # type: ignore[valid-type,misc]
        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any,
        ) -> list[Any]:
            del run_manager
            results = _run_coro_blocking(_search(core=core, query=query, config=config))
            return [Document(**search_result_to_document_kwargs(result)) for result in results]

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any,
        ) -> list[Any]:
            del run_manager
            results = await _search(core=core, query=query, config=config)
            return [Document(**search_result_to_document_kwargs(result)) for result in results]

    return _EngineRetriever()


def create_langchain_retriever_tool(
    retriever: Any,
    *,
    name: str,
    description: str,
    document_separator: str = "\n\n",
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> Any:
    """Wrap a retriever with LangChain's ``create_retriever_tool`` helper."""

    create_retriever_tool = _require_symbol(
        "langchain_core.tools.retriever",
        "create_retriever_tool",
    )
    kwargs: dict[str, object] = {
        "retriever": retriever,
        "name": name,
        "description": description,
        "document_separator": document_separator,
    }
    if "response_format" in inspect.signature(create_retriever_tool).parameters:
        kwargs["response_format"] = response_format
    return create_retriever_tool(**kwargs)


def create_langchain_context_tool(
    core: Engine,
    *,
    collection: str | None = None,
    collections: Sequence[str] | None = None,
    namespace: str | None = None,
    name: str = SEARCH_USER_DOCUMENTS_TOOL_NAME,
    description: str = "Search app-owned documents with rag-core and return grounded context with citations.",
    limit: int = SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    content_types: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    rerank: bool = SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    use_lexical_search: bool = SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    max_chars: int | None = SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    max_tokens: int | None = None,
    query_plan: QueryPlan | None = None,
    context_order: ContextOrder = "rank",
    audit_context: AuditContext | None = None,
) -> Any:
    """Build a LangChain tool returning ``(content, artifact)`` from context packs.

    ``collection`` and optional ``namespace`` are bound at construction time. Pass
    ``document_ids`` to bind a static document allowlist; any tool-call
    document ids must stay within it.
    """

    normalized_namespace = validate_bound_namespace(normalize_namespace(namespace))
    validate_search_user_documents_bounds(
        limit=limit,
        max_chars=max_chars,
        max_tokens=max_tokens,
    )
    context_order = validate_context_order(context_order)
    collections_tuple, document_ids_tuple = normalize_static_retrieval_scope(
        collections=resolve_collections_argument(
            collection=collection,
            collections=collections,
            caller="create_langchain_context_tool",
        ),
        document_ids=document_ids,
        limit=limit,
    )
    content_types_tuple = normalize_static_content_types(content_types)
    tool_decorator = _require_symbol("langchain_core.tools", "tool")

    async def _retrieve(
        *,
        query: str,
        requested_limit: int | None,
        document_ids: Sequence[str] | None,
        requested_rerank: bool | None,
        requested_use_lexical_search: bool | None,
        requested_max_chars: int | None,
        requested_max_tokens: int | None,
    ) -> tuple[str, dict[str, object]]:
        request_payload: dict[str, object] = {"query": query}
        if requested_limit is not None:
            request_payload["limit"] = requested_limit
        if document_ids is not None:
            request_payload["document_ids"] = list(document_ids)
        if requested_rerank is not None:
            request_payload["rerank"] = requested_rerank
        if requested_use_lexical_search is not None:
            request_payload["use_lexical_search"] = requested_use_lexical_search
        if requested_max_chars is not None:
            request_payload["max_chars"] = requested_max_chars
        if requested_max_tokens is not None:
            request_payload["max_tokens"] = requested_max_tokens
        request = parse_search_user_documents_request(
            request_payload,
            default_limit=limit,
            default_rerank=rerank,
            default_use_lexical_search=use_lexical_search,
            default_max_chars=max_chars,
            default_max_tokens=max_tokens,
        )
        scoped_document_ids = scope_document_ids(
            requested=request.document_ids,
            configured=document_ids_tuple,
        )
        pack = await core.context(
            query=request.query,
            namespace=normalized_namespace,
            collections=list(collections_tuple),
            limit=request.limit,
            content_types=(
                list(content_types_tuple) if content_types_tuple is not None else None
            ),
            document_ids=scoped_document_ids,
            rerank=request.rerank,
            use_lexical_search=request.use_lexical_search,
            query_plan=query_plan,
            max_chars=request.max_chars,
            max_tokens=request.max_tokens,
            audit_context=audit_context,
        )
        return context_pack_to_tool_output(pack, context_order=context_order)

    @tool_decorator(name, response_format="content_and_artifact")
    async def _context_tool(
        query: str,
        limit: int | None = None,
        document_ids: list[str] | None = None,
        rerank: bool | None = None,
        use_lexical_search: bool | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Retrieve grounded context for a user query."""

        return await _retrieve(
            query=query,
            requested_limit=limit,
            document_ids=document_ids,
            requested_rerank=rerank,
            requested_use_lexical_search=use_lexical_search,
            requested_max_chars=max_chars,
            requested_max_tokens=max_tokens,
        )

    _context_tool.description = description
    return _context_tool

__all__ = [
    "LangChainNotInstalledError",
    "LangChainRetrieverConfig",
    "build_langchain_retriever",
    "context_pack_to_tool_output",
    "create_langchain_context_tool",
    "create_langchain_retriever_tool",
    "search_result_to_document_kwargs",
]
