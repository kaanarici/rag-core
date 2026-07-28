from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4

import anyio
import pytest
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.session import ClientSession
from mcp.server.models import InitializationOptions
from mcp.shared.message import SessionMessage

from rag_core.contracts import (
    SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS,
    SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA,
    SEARCH_USER_DOCUMENTS_TOOL_NAME,
)
from rag_core.demo import build_demo_core, ingest_demo_billing_document
from rag_core.integrations.mcp_server import build_mcp_server
from rag_core.events.types import AuditContext
from rag_core.search.context_pack import ContextOrder, build_context_pack
from rag_core.search.context_pack_models import Context
from rag_core.search.query_plan import QueryPlan
from rag_core.search.vector_models import SearchResult
from tests.support import make_search_result

ResultT = TypeVar("ResultT")


class _RunnableMcpServer(Protocol):
    def create_initialization_options(self) -> InitializationOptions: ...

    async def run(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        initialization_options: InitializationOptions,
    ) -> None: ...


class _FakeCore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.search_calls: list[dict[str, object]] = []
        self.retrieve_context_calls: list[dict[str, object]] = []

    async def search(
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
    ) -> list[SearchResult]:
        self.search_calls.append(
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
            }
        )
        return self.results[:limit]

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
        audit_context: AuditContext | None = None,
    ) -> Context:
        self.retrieve_context_calls.append(
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
            }
        )
        return build_context_pack(
            self.results[:limit],
            query=query,
            max_snippets=limit,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )


class _FailingCore:
    def __init__(self, message: str) -> None:
        self.message = message

    async def search(self, **_: Any) -> list[SearchResult]:
        raise RuntimeError(self.message)

    async def context(self, **_: Any) -> Context:
        raise RuntimeError(self.message)


class _ContextOnlyCore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.retrieve_context_calls: list[dict[str, object]] = []
        self.results = results

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
        audit_context: AuditContext | None = None,
    ) -> Context:
        self.retrieve_context_calls.append(
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
        return build_context_pack(
            self.results[:limit],
            query=query,
            max_snippets=limit,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )


def test_mcp_server_lists_read_only_tools_with_scope_bound_schemas() -> None:
    async def scenario() -> None:
        core = _FakeCore([])
        server = build_mcp_server(core, namespace="acme", collections=["help"])

        async def check(session: ClientSession) -> None:
            result = await session.list_tools()
            tools = {tool.name: tool for tool in result.tools}

            assert sorted(tools) == [SEARCH_USER_DOCUMENTS_TOOL_NAME]
            for tool in tools.values():
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is True
                assert tool.annotations.destructiveHint is False
                schema = tool.inputSchema
                assert schema["additionalProperties"] is False
                assert set(schema["properties"]) == {"query", "limit"}
                assert "namespace" not in schema["properties"]
                assert "collections" not in schema["properties"]
                assert "maximum" not in cast(dict[str, object], schema["properties"]["limit"])

        await _with_mcp_session(server, check)

    anyio.run(scenario)


def test_mcp_search_and_retrieve_context_bind_scope_and_return_structured_content() -> None:
    async def scenario() -> None:
        core = _FakeCore(
            [
                make_search_result(
                    id="hit-1",
                    text="Billing invoices can be paid by ACH.",
                    score=0.87,
                    namespace="acme",
                    collection="help",
                    document_id="doc-1",
                    document_key="billing.md",
                    section_title="Payments",
                    section_path="Billing > Payments",
                    chunk_index=3,
                    metadata={"page_number": 7},
                )
            ]
        )
        server = build_mcp_server(
            core,
            namespace=" acme ",
            collections=[" help "],
            rerank=True,
            limit_cap=3,
        )

        async def check(session: ClientSession) -> None:
            search = await session.call_tool(
                "search_user_documents",
                {"query": "pay invoices", "limit": 1},
            )
            assert search.isError is False
            search_payload = _structured(search.structuredContent)
            snippets = cast(list[dict[str, object]], search_payload["snippets"])
            assert search_payload["query"] == "pay invoices"
            assert search_payload["max_snippets"] == 1
            assert "Billing invoices" in cast(str, search_payload["context_text"])
            source = cast(dict[str, object], snippets[0]["source"])
            assert source["section_path"] == "Billing > Payments"
            assert "document_key" not in source
            assert "document_id" not in source
            assert cast(dict[str, object], snippets[0]["locator"])["page_number"] == 7

        await _with_mcp_session(server, check)

        assert core.search_calls == []
        assert len(core.retrieve_context_calls) == 1

        search_call = core.retrieve_context_calls[0]
        assert search_call["namespace"] == "acme"
        assert search_call["collections"] == ["help"]
        assert search_call["limit"] == 1
        assert search_call["rerank"] is True
        assert search_call["document_ids"] is None
        assert search_call["content_types"] is None
        assert search_call["max_chars"] == 3000

    anyio.run(scenario)


def test_mcp_server_accepts_context_only_core() -> None:
    async def scenario() -> None:
        core = _ContextOnlyCore(
            [
                make_search_result(
                    id="hit-1",
                    text="Billing invoices can be paid by ACH.",
                    document_id="doc-1",
                )
            ]
        )
        server = build_mcp_server(core, namespace="acme", collections=["help"])

        async def check(session: ClientSession) -> None:
            result = await session.call_tool(
                SEARCH_USER_DOCUMENTS_TOOL_NAME,
                {"query": "pay invoices"},
            )

            assert result.isError is False
            payload = _structured(result.structuredContent)
            assert "Billing invoices" in cast(str, payload["context_text"])

        await _with_mcp_session(server, check)

        assert len(core.retrieve_context_calls) == 1

    anyio.run(scenario)


def test_mcp_retrieve_context_alias_is_not_a_tool() -> None:
    async def scenario() -> None:
        core = _FakeCore([make_search_result()])
        server = build_mcp_server(core, namespace="acme", collections=["help"])

        async def check(session: ClientSession) -> None:
            result = await session.call_tool("retrieve_context", {"query": "billing"})

            assert result.isError is True
            assert result.structuredContent is None

        await _with_mcp_session(server, check)

        assert core.search_calls == []
        assert core.retrieve_context_calls == []

    anyio.run(scenario)


def test_mcp_context_order_extrema_reorders_context_text_not_snippet_lists() -> None:
    async def scenario() -> None:
        core = _FakeCore(
            [
                make_search_result(id="hit-1", text="first", document_id="doc-1"),
                make_search_result(id="hit-2", text="second", document_id="doc-2"),
                make_search_result(id="hit-3", text="third", document_id="doc-3"),
                make_search_result(id="hit-4", text="fourth", document_id="doc-4"),
            ]
        )
        server = build_mcp_server(
            core,
            namespace="acme",
            collections=["help"],
            limit_cap=4,
            context_order="extrema",
        )

        async def check(session: ClientSession) -> None:
            listed = await session.list_tools()
            for tool in listed.tools:
                assert set(tool.inputSchema["properties"]) == {"query", "limit"}

            result = await session.call_tool(
                SEARCH_USER_DOCUMENTS_TOOL_NAME,
                {"query": "billing", "limit": 4},
            )
            assert result.isError is False
            payload = _structured(result.structuredContent)
            snippets = cast(list[dict[str, object]], payload["snippets"])
            assert [snippet["citation_id"] for snippet in snippets] == [
                "S1",
                "S2",
                "S3",
                "S4",
            ]
            context_text = cast(str, payload["context_text"])
            assert context_text.find("[S1]") < context_text.find("[S3]")
            assert context_text.find("[S3]") < context_text.find("[S4]")
            assert context_text.find("[S4]") < context_text.find("[S2]")

        await _with_mcp_session(server, check)

        assert core.search_calls == []
        assert "context_order" not in core.retrieve_context_calls[0]

    anyio.run(scenario)


def test_mcp_context_order_default_and_explicit_rank_outputs_are_byte_identical() -> None:
    async def scenario() -> None:
        results = [
            make_search_result(id="hit-1", text="first", document_id="doc-1"),
            make_search_result(id="hit-2", text="second", document_id="doc-2"),
            make_search_result(id="hit-3", text="third", document_id="doc-3"),
        ]

        async def payloads(
            context_order: ContextOrder | None,
        ) -> dict[str, dict[str, object]]:
            core = _FakeCore(results)
            if context_order is None:
                server = build_mcp_server(
                    core,
                    namespace="acme",
                    collections=["help"],
                    limit_cap=3,
                )
            else:
                server = build_mcp_server(
                    core,
                    namespace="acme",
                    collections=["help"],
                    limit_cap=3,
                    context_order=context_order,
                )

            async def check(session: ClientSession) -> dict[str, dict[str, object]]:
                output: dict[str, dict[str, object]] = {}
                result = await session.call_tool(
                    SEARCH_USER_DOCUMENTS_TOOL_NAME,
                    {"query": "billing", "limit": 3},
                )
                assert result.isError is False
                output[SEARCH_USER_DOCUMENTS_TOOL_NAME] = _structured(
                    result.structuredContent
                )
                return output

            return await _with_mcp_session(server, check)

        default_payloads = await payloads(None)
        explicit_rank_payloads = await payloads("rank")

        assert json.dumps(default_payloads, sort_keys=True) == json.dumps(
            explicit_rank_payloads,
            sort_keys=True,
        )

    anyio.run(scenario)


@pytest.mark.parametrize("tool_name", [SEARCH_USER_DOCUMENTS_TOOL_NAME])
def test_mcp_tool_failures_are_sanitized(tool_name: str) -> None:
    async def scenario() -> None:
        leak = "/Users/example/private.db token=secret sk-test-secret"
        core = _FailingCore(f"provider exploded at {leak}")
        server = build_mcp_server(core, namespace="acme", collections=["help"])

        async def check(session: ClientSession) -> None:
            result = await session.call_tool(tool_name, {"query": "billing"})

            assert result.isError is True
            assert result.structuredContent is None
            text = getattr(result.content[0], "text", "")
            assert text == "rag-core MCP tool failed"
            assert "provider exploded" not in text
            assert "private.db" not in text
            assert "token=secret" not in text
            assert "sk-test-secret" not in text

        await _with_mcp_session(server, check)

    anyio.run(scenario)


def test_mcp_search_uses_prompt_safe_tool_projection_without_private_ids() -> None:
    async def scenario() -> None:
        core = _FakeCore(
            [
                make_search_result(
                    id="result-private",
                    text="Billing invoices can be paid by ACH.",
                    document_id="internal-doc-id",
                    collection="tenant-corpus",
                    document_key="private/billing.md",
                    content_sha256="content-hash-secret",
                    section_id="section-secret",
                    title=None,
                    section_title="Payments",
                    section_path="Billing > Payments",
                    chunk_index=3,
                )
            ]
        )
        server = build_mcp_server(core, namespace="acme", collections=["help"])

        async def check(session: ClientSession) -> None:
            result = await session.call_tool(
                SEARCH_USER_DOCUMENTS_TOOL_NAME,
                {"query": "pay invoices", "limit": 1},
            )

            assert result.isError is False
            payload = _structured(result.structuredContent)
            assert set(payload) == set(SEARCH_USER_DOCUMENTS_OUTPUT_SCHEMA["properties"])
            assert "hits" not in payload
            snippets = cast(list[dict[str, object]], payload["snippets"])
            assert snippets[0]["citation_id"] == "S1"
            assert "Billing invoices" in cast(str, payload["context_text"])
            encoded = json.dumps(payload, sort_keys=True)
            for forbidden in (
                "internal-doc-id",
                "tenant-corpus",
                "private/billing.md",
                "content-hash-secret",
                "section-secret",
                "result-private",
            ):
                assert forbidden not in encoded

        await _with_mcp_session(server, check)

    anyio.run(scenario)


def test_mcp_search_bounds_result_text_with_context_budget() -> None:
    async def scenario() -> None:
        huge_text = "oversized result text " * 500
        core = _FakeCore([make_search_result(id="huge", text=huge_text)])
        server = build_mcp_server(core, namespace="acme", collections=["help"])

        async def check(session: ClientSession) -> None:
            result = await session.call_tool(
                SEARCH_USER_DOCUMENTS_TOOL_NAME,
                {"query": "billing", "limit": 1},
            )

            assert result.isError is False
            payload = _structured(result.structuredContent)
            snippets = cast(list[dict[str, object]], payload["snippets"])
            assert payload["max_chars"] == SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS
            assert payload["truncated"] is True
            assert snippets[0]["truncated"] is True
            assert len(cast(str, snippets[0]["text"])) < len(huge_text)
            assert len(cast(str, payload["context_text"])) <= SEARCH_USER_DOCUMENTS_DEFAULT_MAX_CHARS

        await _with_mcp_session(server, check)

    anyio.run(scenario)


def test_mcp_limit_cap_truncates_model_requested_limit_before_core_call() -> None:
    async def scenario() -> None:
        core = _FakeCore(
            [
                make_search_result(id="hit-1", text="one", document_id="doc-1"),
                make_search_result(id="hit-2", text="two", document_id="doc-2"),
                make_search_result(id="hit-3", text="three", document_id="doc-3"),
            ]
        )
        server = build_mcp_server(
            core,
            namespace="acme",
            collections=["help"],
            limit_cap=2,
        )

        async def check(session: ClientSession) -> None:
            result = await session.call_tool(
                "search_user_documents",
                {"query": "billing", "limit": 999},
            )

            assert result.isError is False
            payload = _structured(result.structuredContent)
            assert payload["max_snippets"] == 2
            assert len(cast(list[object], payload["snippets"])) == 2

        await _with_mcp_session(server, check)

        assert core.search_calls == []
        assert core.retrieve_context_calls[0]["limit"] == 2

    anyio.run(scenario)


def test_mcp_scope_injection_is_rejected_by_tool_schema_before_core_call() -> None:
    async def scenario() -> None:
        core = _FakeCore([make_search_result()])
        server = build_mcp_server(core, namespace="acme", collections=["help"])

        async def check(session: ClientSession) -> None:
            result = await session.call_tool(
                "search_user_documents",
                {
                    "query": "billing",
                    "namespace": "evil",
                    "collections": ["other"],
                },
            )

            assert result.isError is True
            assert "Input validation error" in getattr(result.content[0], "text", "")

        await _with_mcp_session(server, check)

        assert core.search_calls == []
        assert core.retrieve_context_calls == []

    anyio.run(scenario)


@pytest.mark.integration
def test_mcp_tools_run_against_demo_core_with_memory_qdrant() -> None:
    async def scenario() -> None:
        core = build_demo_core(store_collection=f"mcp_server_{uuid4().hex}")
        try:
            await core.ensure_ready()
            await ingest_demo_billing_document(core)
            server = build_mcp_server(
                core,
                namespace="acme",
                collections=["help-center"],
                limit_cap=2,
            )

            async def check(session: ClientSession) -> None:
                search = await session.call_tool(
                    "search_user_documents",
                    {"query": "pay invoices", "limit": 2},
                )
                assert search.isError is False
                search_payload = _structured(search.structuredContent)
                snippets = cast(list[dict[str, object]], search_payload["snippets"])
                assert snippets
                assert "invoices" in cast(str, snippets[0]["text"]).lower()

            await _with_mcp_session(server, check)
        finally:
            await core.close()

    anyio.run(scenario)


async def _with_mcp_session(
    server: _RunnableMcpServer,
    check: Callable[[ClientSession], Awaitable[ResultT]],
) -> ResultT:
    client_write_raw, server_read_raw = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](100)
    server_write_raw, client_read_raw = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](100)
    server_read = server_read_raw
    server_write = cast(MemoryObjectSendStream[SessionMessage], server_write_raw)
    client_read = client_read_raw
    client_write = cast(MemoryObjectSendStream[SessionMessage], client_write_raw)

    unset = object()
    result: object = unset
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            server.run,
            server_read,
            server_write,
            server.create_initialization_options(),
        )
        async with ClientSession(client_read, client_write) as session:
            await session.initialize()
            result = await check(session)
        task_group.cancel_scope.cancel()
    assert result is not unset
    return cast(ResultT, result)


def _structured(payload: dict[str, Any] | None) -> dict[str, object]:
    assert payload is not None
    return cast(dict[str, object], payload)
