from __future__ import annotations

import asyncio
import importlib

import pytest

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
)
from rag_core.integrations.openai_agents import build_retrieve_context_tool
from rag_core.search import DenseChannel, Prefetch, QueryPlan


class _Pack:
    def as_text(self) -> str:
        return "app context"

    def as_prompt_text(self) -> str:
        return "safe context"

    def to_payload(self) -> dict[str, object]:
        return {
            "query": "billing",
            "snippets": [{"source": {"source_id": "private", "result_id": "hit"}}],
        }

    def to_prompt_payload(self) -> dict[str, object]:
        return {
            "query": "billing",
            "snippets": [],
            "citations": [],
            "source_previews": [],
            "citation_summary": "",
            "dropped_count": 0,
            "max_snippets": 5,
            "max_chars": 3000,
            "max_tokens": None,
            "token_estimate": 0,
            "char_count": 0,
            "truncated": False,
        }


def _sample_query_plan() -> QueryPlan:
    return QueryPlan(prefetches=(Prefetch(channel=DenseChannel(), limit=5),))


class _Core:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def retrieve_context(
        self,
        *,
        query: str,
        namespace: str,
        corpus_ids: list[str],
        limit: int,
        content_types: list[str] | None,
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        query_plan: QueryPlan | None,
        max_chars: int | None,
        max_tokens: int | None,
    ) -> _Pack:
        self.calls.append(
            {
                "query": query,
                "namespace": namespace,
                "corpus_ids": corpus_ids,
                "limit": limit,
                "content_types": content_types,
                "document_ids": document_ids,
                "rerank": rerank,
                "use_lexical_search": use_lexical_search,
                "query_plan": query_plan,
                "max_chars": max_chars,
                "max_tokens": max_tokens,
            }
        )
        return _Pack()


def _install_fake_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    class _Agents:
        @staticmethod
        def function_tool(**kwargs: object) -> object:
            del kwargs

            def _decorator(func: object) -> object:
                return func

            return _decorator

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *args, **kwargs: _Agents
        if name == "agents"
        else real_import_module(name, *args, **kwargs),
    )


def test_openai_agents_tool_uses_shared_request_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _Core()
    tool_fn = build_retrieve_context_tool(
        core,
        namespace=" acme ",
        corpus_ids=[" help "],
    )

    asyncio.run(tool_fn(query=" billing ", document_ids=[" doc-1 "]))

    assert core.calls == [
        {
            "query": "billing",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "limit": SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
            "content_types": None,
            "document_ids": ["doc-1"],
            "rerank": False,
            "use_lexical_search": True,
            "query_plan": None,
            "max_chars": SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
            "max_tokens": None,
        }
    ]


def test_openai_agents_tool_preserves_explicit_default_budget_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _Core()
    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        default_max_chars=None,
        default_max_tokens=256,
    )

    asyncio.run(tool_fn(query="billing"))

    assert core.calls[0]["max_chars"] is None
    assert core.calls[0]["max_tokens"] == 256


def test_openai_agents_tool_binds_query_plan_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _Core()
    query_plan = _sample_query_plan()
    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        query_plan=query_plan,
    )

    asyncio.run(tool_fn(query="billing"))

    assert core.calls[0]["query_plan"] is query_plan


def test_openai_agents_tool_rejects_blank_namespace_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)

    with pytest.raises(ValueError, match="namespace must not be empty"):
        build_retrieve_context_tool(_Core(), namespace="   ", corpus_ids=["help"])


def test_openai_agents_tool_rejects_blank_bound_scope_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)

    with pytest.raises(ValueError, match="corpus_ids must contain non-empty strings"):
        build_retrieve_context_tool(_Core(), namespace="acme", corpus_ids=[" "])

    with pytest.raises(ValueError, match="document_ids must contain non-empty strings"):
        build_retrieve_context_tool(
            _Core(),
            namespace="acme",
            corpus_ids=["help"],
            document_ids=[" "],
        )


def test_openai_agents_tool_uses_static_document_scope_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _Core()
    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        document_ids=[" doc-1 "],
    )

    asyncio.run(tool_fn(query="billing"))

    assert core.calls[0]["document_ids"] == ["doc-1"]


def test_openai_agents_tool_rejects_document_ids_outside_static_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _Core()
    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        document_ids=["doc-1"],
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(tool_fn(query="billing", document_ids=["doc-2"]))

    assert str(exc_info.value) == (
        "document_ids contain values outside the configured retrieval scope"
    )
    assert core.calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": "   "}, "query must be a non-empty string"),
        (
            {"query": "billing", "document_ids": [""]},
            "document_ids must be an array of non-empty strings",
        ),
    ],
)
def test_openai_agents_tool_rejects_invalid_requests_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _Core()
    tool_fn = build_retrieve_context_tool(core, namespace="acme", corpus_ids=["help"])

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(tool_fn(**kwargs))

    assert str(exc_info.value) == message
    assert core.calls == []
