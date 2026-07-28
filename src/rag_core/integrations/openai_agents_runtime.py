"""Runtime helpers for the OpenAI Agents integration."""

from __future__ import annotations

import importlib
from typing import Any, Callable, cast


def build_tool_request_payload(
    *,
    query: str,
    limit: int | None,
    document_ids: list[str] | None,
    rerank: bool | None,
    use_lexical_search: bool | None,
    max_chars: int | None,
    max_tokens: int | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"query": query}
    if limit is not None:
        payload["limit"] = limit
    if document_ids is not None:
        payload["document_ids"] = document_ids
    if rerank is not None:
        payload["rerank"] = rerank
    if use_lexical_search is not None:
        payload["use_lexical_search"] = use_lexical_search
    if max_chars is not None:
        payload["max_chars"] = max_chars
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def import_agents_function_tool() -> Callable[..., Any]:
    try:
        module = importlib.import_module("agents")
    except ImportError as exc:
        raise ImportError(
            "openai-agents package is required for rag_core OpenAI Agents integration. "
            "Install it with: pip install 'rag-core[openai-agents]'"
        ) from exc
    function_tool = getattr(module, "function_tool", None)
    if not callable(function_tool):
        raise ImportError(
            "openai-agents package with agents.function_tool is required for "
            "rag_core OpenAI Agents integration."
        )
    return cast(Callable[..., Any], function_tool)
