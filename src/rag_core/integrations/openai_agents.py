"""Optional OpenAI Agents SDK adapter for rag-core retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    search_user_documents_tool_result,
    validate_bound_namespace,
    validate_search_user_documents_bounds,
)
from rag_core.events.types import AuditContext
from rag_core.integrations.openai_agents_runtime import (
    build_tool_request_payload,
    import_agents_function_tool,
)
from rag_core.integrations.protocols import SupportsRetrieveContext
from rag_core.search import QueryPlan
from rag_core.search.context_pack import ContextOrder, validate_context_order


def build_retrieve_context_tool(
    core: SupportsRetrieveContext,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    content_types: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    tool_name: str = SEARCH_USER_DOCUMENTS_TOOL_NAME,
    tool_description: str | None = None,
    default_limit: int = SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    default_rerank: bool = SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    default_use_lexical_search: bool = SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    default_max_chars: int | None = SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    default_max_tokens: int | None = None,
    query_plan: QueryPlan | None = None,
    context_order: ContextOrder = "rank",
    audit_context: AuditContext | None = None,
    return_payload: bool = True,
    timeout: float | None = None,
) -> Any:
    """Build an OpenAI Agents SDK function tool around ``RAGCore.retrieve_context``.

    ``namespace`` and ``corpus_ids`` are bound when the tool is constructed.
    Models can never choose them. Pass ``document_ids`` to bind a static
    document allowlist; any model-supplied document ids must stay within it.
    """

    if not callable(getattr(core, "retrieve_context", None)):
        raise TypeError("core must provide an async retrieve_context method")

    resolved_namespace = validate_bound_namespace(namespace)
    validate_search_user_documents_bounds(
        limit=default_limit,
        max_chars=default_max_chars,
        max_tokens=default_max_tokens,
    )
    context_order = validate_context_order(context_order)

    default_corpus_ids_tuple, default_document_ids = normalize_static_retrieval_scope(
        corpus_ids=corpus_ids,
        document_ids=document_ids,
        limit=default_limit,
    )
    default_corpus_ids = list(default_corpus_ids_tuple)
    default_content_types = normalize_static_content_types(content_types)

    description = tool_description or (
        "Search app-owned documents with rag-core and return grounded context "
        "snippets plus citation metadata."
    )

    function_tool = import_agents_function_tool()

    async def search_user_documents(
        query: str,
        limit: int | None = None,
        document_ids: list[str] | None = None,
        rerank: bool | None = None,
        use_lexical_search: bool | None = None,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> str | dict[str, object]:
        """Search authorized user documents and return grounded retrieval context.

        Args:
            query: Question or search intent.
            limit: Optional maximum context snippets.
            document_ids: Optional narrowing filter inside the bound app scope.
            rerank: Optional rerank override.
            use_lexical_search: Optional configured lexical/exact-match expansion override.
            max_chars: Optional approximate returned character budget.
            max_tokens: Optional approximate returned token budget.
        """

        request = parse_search_user_documents_request(
            build_tool_request_payload(
                query=query,
                limit=limit,
                document_ids=document_ids,
                rerank=rerank,
                use_lexical_search=use_lexical_search,
                max_chars=max_chars,
                max_tokens=max_tokens,
            ),
            default_limit=default_limit,
            default_rerank=default_rerank,
            default_use_lexical_search=default_use_lexical_search,
            default_max_chars=default_max_chars,
            default_max_tokens=default_max_tokens,
        )
        scoped_document_ids = scope_document_ids(
            requested=request.document_ids,
            configured=default_document_ids,
        )

        pack = await core.retrieve_context(
            query=request.query,
            namespace=resolved_namespace,
            corpus_ids=default_corpus_ids,
            limit=request.limit,
            content_types=(
                list(default_content_types)
                if default_content_types is not None
                else None
            ),
            document_ids=scoped_document_ids,
            rerank=request.rerank,
            use_lexical_search=request.use_lexical_search,
            query_plan=query_plan,
            max_chars=request.max_chars,
            max_tokens=request.max_tokens,
            audit_context=audit_context,
        )

        if return_payload:
            return search_user_documents_tool_result(
                pack,
                context_order=context_order,
            )
        return pack.as_prompt_text(context_order=context_order)

    tool_kwargs: dict[str, object] = {
        "name_override": tool_name,
        "description_override": description,
    }
    if timeout is not None:
        tool_kwargs["timeout"] = timeout
    decorate = function_tool(**tool_kwargs)
    return decorate(search_user_documents)


__all__ = ["build_retrieve_context_tool"]
