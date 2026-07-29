from __future__ import annotations

import asyncio
from typing import Any, Callable, cast

import pytest

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
)
from rag_core.events import EventBuffer
from rag_core.events.types import AuditContext, SearchCompleted
from rag_core.integrations import langchain as langchain_integration
from rag_core.search import QueryPlan
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


class _Pack:
    def as_text(self) -> str:
        return "app context text"

    def as_prompt_text(self, *, context_order: object = "rank") -> str:
        if context_order == "extrema":
            return "safe context text extrema"
        assert context_order == "rank"
        return "safe context text"

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
            "char_count": 12,
            "truncated": False,
        }


class _Core:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def context(
        self,
        *,
        query: str,
        namespace: str,
        collections: list[str],
        limit: int,
        content_types: list[str] | None,
        document_ids: list[str] | None,
        rerank: bool,
        use_lexical_search: bool,
        query_plan: QueryPlan | None,
        max_chars: int | None,
        max_tokens: int | None,
        audit_context: AuditContext | None,
    ) -> _Pack:
        self.calls.append(
            {
                "query": query,
                "namespace": namespace,
                "collections": collections,
                "limit": limit,
                "content_types": content_types,
                "document_ids": document_ids,
                "rerank": rerank,
                "use_lexical_search": use_lexical_search,
                "query_plan": query_plan,
                "max_chars": max_chars,
                "max_tokens": max_tokens,
                "audit_context": audit_context,
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
        collections=["help"],
    )

    content, artifact = asyncio.run(tool_fn(" billing policy "))

    assert content == "safe context text"
    assert artifact == {
        "ok": True,
        "context_text": "safe context text",
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
            "collections": ["help"],
            "limit": SEARCH_USER_DOCUMENTS_DEFAULT_LIMIT,
            "content_types": None,
            "document_ids": None,
            "rerank": False,
            "use_lexical_search": True,
            "query_plan": None,
            "max_chars": SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
            "max_tokens": None,
            "audit_context": None,
        }
    ]


def test_langchain_context_tool_default_and_explicit_rank_outputs_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    default_core = _Core()
    explicit_core = _Core()
    default_tool = langchain_integration.create_langchain_context_tool(
        cast(Any, default_core),
        namespace="acme",
        collections=["help"],
    )
    explicit_tool = langchain_integration.create_langchain_context_tool(
        cast(Any, explicit_core),
        namespace="acme",
        collections=["help"],
        context_order="rank",
    )

    assert asyncio.run(default_tool("billing")) == asyncio.run(explicit_tool("billing"))
    assert "context_order" not in default_core.calls[0]
    assert "context_order" not in explicit_core.calls[0]


def test_langchain_context_tool_normalizes_namespace_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace=" acme ",
        collections=["help"],
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
        collections=["help"],
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
        collections=["help"],
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
            "collections": ["help"],
            "limit": 2,
            "content_types": None,
            "document_ids": ["doc-1"],
            "rerank": True,
            "use_lexical_search": True,
            "query_plan": None,
            "max_chars": 1200,
            "max_tokens": 256,
            "audit_context": None,
        }
    ]


def test_langchain_context_tool_context_order_is_builder_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace="acme",
        collections=["help"],
        context_order="extrema",
    )

    content, artifact = asyncio.run(tool_fn(query="billing"))

    assert content == "safe context text extrema"
    assert artifact["context_text"] == "safe context text extrema"
    assert "context_order" not in core.calls[0]
    assert "context_order" not in getattr(tool_fn, "__annotations__", {})


def test_langchain_context_tool_binds_app_owned_content_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace="acme",
        collections=["help"],
        content_types=[" document "],
    )

    asyncio.run(tool_fn("billing"))

    assert core.calls[0]["content_types"] == ["document"]


def test_langchain_context_tool_rejects_document_ids_outside_static_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    core = _Core()
    tool_fn = langchain_integration.create_langchain_context_tool(
        cast(Any, core),
        namespace="acme",
        collections=["help"],
        document_ids=["doc-1"],
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(tool_fn(query="billing", document_ids=["doc-2"]))

    assert str(exc_info.value) == (
        "document_ids contain values outside the configured retrieval scope"
    )
    assert core.calls == []


def test_langchain_context_tool_threads_audit_context_to_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_core import Engine

    captured: dict[str, object] = {}
    _install_fake_langchain_tool(monkeypatch, captured=captured)
    buffer = EventBuffer()
    core = Engine(
        make_test_config(
            qdrant_collection="rag_core_langchain_context_audit",
            embedding_dimensions=4,
        ),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=RecordingVectorStore(
            search_results=[make_search_result(id="hit-1", text="ok")]
        ),
        event_sink=buffer,
    )

    async def scenario() -> None:
        try:
            tool_fn = langchain_integration.create_langchain_context_tool(
                core,
                namespace="acme",
                collections=["help"],
                audit_context=AuditContext(actor="agent-user", request_id="req-1"),
            )
            await tool_fn("billing")
        finally:
            await core.close()

    asyncio.run(scenario())

    [completed] = [event for event in buffer.events if isinstance(event, SearchCompleted)]
    assert completed.actor == "agent-user"
    assert completed.request_id == "req-1"
