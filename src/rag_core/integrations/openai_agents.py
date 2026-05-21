"""Optional OpenAI Agents SDK adapter for rag-core retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
    parse_search_user_documents_request,
    search_user_documents_tool_result,
    validate_search_user_documents_bounds,
)
from rag_core.integrations.integration_context_text import context_pack_model_text
from rag_core.integrations.openai_agents_runtime import (
    build_tool_request_payload,
    import_agents_function_tool,
)


class ContextPackLike(Protocol):
    def as_text(self) -> str: ...

    def to_payload(self) -> dict[str, object]: ...


class SupportsRetrieveContext(Protocol):
    async def retrieve_context(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int,
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        max_chars: int | None,
        max_tokens: int | None,
    ) -> ContextPackLike: ...


def build_retrieve_context_tool(
    core: SupportsRetrieveContext,
    *,
    namespace: str,
    corpus_ids: Sequence[str],
    document_ids: Sequence[str] | None = None,
    tool_name: str = SEARCH_USER_DOCUMENTS_TOOL_NAME,
    tool_description: str | None = None,
    default_limit: int = SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    default_rerank: bool = SEARCH_USER_DOCUMENTS_DEFAULT_RERANK,
    default_use_lexical_search: bool = SEARCH_USER_DOCUMENTS_DEFAULT_USE_LEXICAL_SEARCH,
    default_max_chars: int | None = SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    default_max_tokens: int | None = None,
    return_payload: bool = True,
    timeout: float | None = None,
) -> Any:
    """Build an OpenAI Agents SDK function tool around ``RAGCore.retrieve_context``.

    ``namespace`` and ``corpus_ids`` are bound when the tool is constructed, not
    supplied by the model. Pass ``document_ids`` to statically bound document
    scope when the tool should not search every document in those corpora.
    """

    if not callable(getattr(core, "retrieve_context", None)):
        raise TypeError("core must provide an async retrieve_context method")

    resolved_namespace = namespace.strip()
    if not resolved_namespace:
        raise ValueError("namespace must not be empty")
    validate_search_user_documents_bounds(
        limit=default_limit,
        max_chars=default_max_chars,
        max_tokens=default_max_tokens,
    )

    default_corpus_ids = list(corpus_ids)
    if not default_corpus_ids:
        raise ValueError("corpus_ids must include at least one corpus id")
    default_document_ids = tuple(document_ids) if document_ids is not None else None

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
            document_ids: Optional document IDs after app-side authorization.
            rerank: Optional rerank override.
            use_lexical_search: Optional lexical/exact retrieval override.
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
        scoped_document_ids = _scope_document_ids(
            requested=request.document_ids,
            configured=default_document_ids,
        )

        pack = await core.retrieve_context(
            query=request.query,
            namespace=resolved_namespace,
            corpus_ids=default_corpus_ids,
            limit=request.limit,
            document_ids=scoped_document_ids,
            rerank=request.rerank,
            use_lexical_search=request.use_lexical_search,
            max_chars=request.max_chars,
            max_tokens=request.max_tokens,
        )

        if return_payload:
            return search_user_documents_tool_result(pack)
        return context_pack_model_text(pack)

    tool_kwargs: dict[str, object] = {
        "name_override": tool_name,
        "description_override": description,
    }
    if timeout is not None:
        tool_kwargs["timeout"] = timeout
    decorate = function_tool(**tool_kwargs)
    return decorate(search_user_documents)


def _scope_document_ids(
    *,
    requested: tuple[str, ...] | None,
    configured: tuple[str, ...] | None,
) -> list[str] | None:
    if configured is None:
        return list(requested) if requested is not None else None
    if requested is None:
        return list(configured)
    configured_set = set(configured)
    rejected = [document_id for document_id in requested if document_id not in configured_set]
    if rejected:
        raise ValueError("document_ids contain values outside the configured retrieval scope")
    return list(requested)


__all__ = ["build_retrieve_context_tool"]
