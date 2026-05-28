import asyncio
import importlib
from typing import Any, Callable

import pytest

from rag_core.integrations.openai_agents import build_retrieve_context_tool
from rag_core.search import DenseChannel, Prefetch, QueryPlan


class _FakePack:
    def __init__(self, *, text: str, payload: dict[str, object]) -> None:
        self._text = text
        self._payload = payload

    def as_text(self) -> str:
        return self._text

    def as_prompt_text(self) -> str:
        return self._text

    def to_payload(self) -> dict[str, object]:
        return self._payload

    def to_prompt_payload(self) -> dict[str, object]:
        return self._payload


def _context_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
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
    payload.update(overrides)
    return payload


class _FakeCore:
    def __init__(self, pack: _FakePack) -> None:
        self.pack = pack
        self.calls: list[dict[str, Any]] = []

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
        query_plan: object | None,
        max_chars: int | None,
        max_tokens: int | None,
    ) -> _FakePack:
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
        return self.pack


def _install_fake_agents(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decorator: Callable[[Any], Any] | None = None,
    capture: dict[str, Any] | None = None,
) -> None:
    """Patch importlib so the integration sees a stub `agents` module.

    Either pass a `decorator` that wraps the underlying tool function, or
    provide `capture` to record the kwargs `function_tool` was called with.
    Falls back to a pass-through decorator if neither is supplied.
    """
    real_import_module = importlib.import_module

    def _function_tool(**kwargs: Any) -> Any:
        if capture is not None:
            capture["kwargs"] = kwargs

        def _identity(func: Any) -> Any:
            if capture is not None:
                capture["func"] = func
            return decorator(func) if decorator is not None else func

        return _identity

    class _FakeAgentsModule:
        function_tool = staticmethod(_function_tool)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *args, **kwargs: _FakeAgentsModule
        if name == "agents"
        else real_import_module(name, *args, **kwargs),
    )


def test_build_retrieve_context_tool_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def _fake_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "agents":
            raise ImportError("No module named 'agents'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    with pytest.raises(ImportError, match="openai-agents package is required"):
        build_retrieve_context_tool(
            _FakeCore(_FakePack(text="", payload={})),
            namespace="acme",
            corpus_ids=["help"],
        )


def test_build_retrieve_context_tool_rejects_core_without_retrieve_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)

    with pytest.raises(TypeError, match="retrieve_context"):
        build_retrieve_context_tool(object(), namespace="acme", corpus_ids=["help"])  # type: ignore[arg-type]


def test_build_retrieve_context_tool_passes_overrides_to_function_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_agents(
        monkeypatch,
        decorator=lambda func: {"wrapped": func.__name__},
        capture=captured,
    )

    tool = build_retrieve_context_tool(
        _FakeCore(_FakePack(text="", payload={})),
        namespace="acme",
        corpus_ids=["help"],
        tool_name="retrieve_docs",
        tool_description="Fetch docs",
    )

    assert tool == {"wrapped": "search_user_documents"}
    assert captured["kwargs"]["name_override"] == "retrieve_docs"
    assert captured["kwargs"]["description_override"] == "Fetch docs"


def test_retrieve_context_tool_binds_scope_and_default_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    payload = _context_payload()
    core = _FakeCore(_FakePack(text="[doc-1] billing context", payload=payload))

    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        default_limit=4,
        default_rerank=False,
        default_use_lexical_search=False,
        default_max_chars=400,
        return_payload=False,
    )

    assert asyncio.run(tool_fn(query="how billing works")) == "[doc-1] billing context"
    call = core.calls[0]
    assert call["namespace"] == "acme"
    assert call["corpus_ids"] == ["help"]
    assert call["limit"] == 4
    assert call["content_types"] is None
    assert call["rerank"] is False
    assert call["use_lexical_search"] is False
    assert call["query_plan"] is None
    assert call["max_chars"] == 400
    assert call["max_tokens"] is None
    assert call["document_ids"] is None


def test_retrieve_context_tool_prefers_prompt_text_over_as_text_when_payload_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)

    class _SplitPack(_FakePack):
        def __init__(self) -> None:
            super().__init__(text="", payload=_context_payload())

        def as_text(self) -> str:
            return "LEAK private/billing.md"

        def as_prompt_text(self) -> str:
            return "safe billing context"

    core = _FakeCore(_SplitPack())
    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        return_payload=False,
    )

    assert asyncio.run(tool_fn(query="billing")) == "safe billing context"



def test_retrieve_context_tool_allows_per_call_overrides_and_returns_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    payload = _context_payload()
    core = _FakeCore(_FakePack(text="[doc-1] billing context", payload=payload))

    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        return_payload=True,
    )

    result = asyncio.run(
        tool_fn(
            query="how billing works",
            limit=2,
            document_ids=["doc-7"],
            rerank=True,
            use_lexical_search=True,
            max_chars=256,
            max_tokens=64,
        )
    )

    assert result == {"ok": True, "context_text": "[doc-1] billing context", **payload}
    call = core.calls[0]
    assert call["limit"] == 2
    assert call["document_ids"] == ["doc-7"]
    assert call["rerank"] is True
    assert call["use_lexical_search"] is True
    assert call["query_plan"] is None
    assert call["max_chars"] == 256
    assert call["max_tokens"] == 64


def test_retrieve_context_tool_binds_app_owned_query_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _FakeCore(_FakePack(text="context", payload=_context_payload()))
    query_plan = QueryPlan(prefetches=(Prefetch(channel=DenseChannel(), limit=5),))
    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        query_plan=query_plan,
    )

    asyncio.run(tool_fn(query="billing"))

    assert core.calls[0]["query_plan"] is query_plan


def test_retrieve_context_tool_binds_app_owned_content_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _FakeCore(_FakePack(text="context", payload=_context_payload()))
    tool_fn = build_retrieve_context_tool(
        core,
        namespace="acme",
        corpus_ids=["help"],
        content_types=[" document "],
    )

    asyncio.run(tool_fn(query="billing"))

    assert core.calls[0]["content_types"] == ["document"]


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"default_limit": 0}, "limit"),
        ({"corpus_ids": []}, "corpus_ids"),
        ({"content_types": [" "]}, "content_types"),
    ],
)
def test_build_retrieve_context_tool_validates_config(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    error_match: str,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _FakeCore(_FakePack(text="", payload={}))
    base: dict[str, Any] = {"namespace": "acme", "corpus_ids": ["help"]}
    base.update(kwargs)

    with pytest.raises(ValueError, match=error_match):
        build_retrieve_context_tool(core, **base)


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"limit": 1000}, "limit"),
        ({"max_chars": 1}, "max_chars"),
        ({"max_tokens": 1}, "max_tokens"),
    ],
)
def test_retrieve_context_tool_rejects_values_outside_contract_bounds(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    error_match: str,
) -> None:
    _install_fake_agents(monkeypatch)
    core = _FakeCore(_FakePack(text="", payload={}))
    tool_fn = build_retrieve_context_tool(core, namespace="acme", corpus_ids=["help"])

    with pytest.raises(ValueError, match=error_match):
        asyncio.run(tool_fn(query="billing", **kwargs))

    assert core.calls == []
