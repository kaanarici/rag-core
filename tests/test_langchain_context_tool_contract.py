from __future__ import annotations

import asyncio
from typing import Any, Callable, cast

import pytest

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
)
from rag_core.integrations import langchain as langchain_integration


class _Pack:
    def as_text(self) -> str:
        return "context text"

    def to_payload(self) -> dict[str, object]:
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
            "char_count": 12,
            "truncated": False,
        }


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
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        query_plan: object | None,
        max_chars: int | None,
        max_tokens: int | None,
    ) -> _Pack:
        self.calls.append(
            {
                "query": query,
                "namespace": namespace,
                "corpus_ids": corpus_ids,
                "limit": limit,
                "document_ids": document_ids,
                "rerank": rerank,
                "use_lexical_search": use_lexical_search,
                "query_plan": query_plan,
                "max_chars": max_chars,
                "max_tokens": max_tokens,
            }
        )
        return _Pack()


def _install_fake_langchain_tool(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured: dict[str, object],
) -> None:
    def _require_symbol(module_name: str, symbol: str) -> Callable[..., Callable[[Any], Any]]:
        assert module_name == "langchain_core.tools"
        assert symbol == "tool"

        def _tool(name: str, **kwargs: object) -> Callable[[Any], Any]:
            captured["name"] = name
            captured["kwargs"] = kwargs

            def _decorate(func: Any) -> Any:
                setattr(func, "name", name)
                for key, value in kwargs.items():
                    setattr(func, key, value)
                return func

            return _decorate

        return _tool

    monkeypatch.setattr(langchain_integration, "_require_symbol", _require_symbol)


def test_langchain_context_tool_defaults_to_public_search_tool_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()

    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace="acme",
        corpus_ids=["help"],
    )

    content, artifact = asyncio.run(tool_fn(" billing policy "))

    assert content == "context text"
    assert artifact == {
        "ok": True,
        "context_text": "context text",
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
        "char_count": 12,
        "truncated": False,
    }
    assert captured["name"] == SEARCH_USER_DOCUMENTS_TOOL_NAME
    assert captured["kwargs"] == {"response_format": "content_and_artifact"}
    assert core.calls == [
        {
            "query": "billing policy",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "limit": SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
            "document_ids": None,
            "rerank": True,
            "use_lexical_search": True,
            "query_plan": None,
            "max_chars": SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
            "max_tokens": None,
        }
    ]


def test_langchain_context_tool_normalizes_namespace_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace=" acme ",
        corpus_ids=["help"],
    )

    asyncio.run(tool_fn("billing"))

    assert core.calls[0]["namespace"] == "acme"


def test_langchain_context_tool_rejects_blank_queries_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace="acme",
        corpus_ids=["help"],
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(tool_fn("   "))

    assert str(exc_info.value) == "query must be a non-empty string"
    assert core.calls == []


def test_langchain_context_tool_exposes_public_search_request_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace="acme",
        corpus_ids=["help"],
        limit=3,
        rerank=False,
        use_lexical_search=False,
    )

    asyncio.run(
        tool_fn(
            query="billing",
            limit=2,
            document_ids=["doc-1"],
            rerank=True,
            use_lexical_search=True,
            max_chars=1200,
            max_tokens=256,
        )
    )

    assert core.calls == [
        {
            "query": "billing",
            "namespace": "acme",
            "corpus_ids": ["help"],
            "limit": 2,
            "document_ids": ["doc-1"],
            "rerank": True,
            "use_lexical_search": True,
            "query_plan": None,
            "max_chars": 1200,
            "max_tokens": 256,
        }
    ]


def test_langchain_context_tool_rejects_document_ids_outside_static_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
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
