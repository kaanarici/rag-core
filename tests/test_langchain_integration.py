import asyncio
import builtins
import sys
import types
from typing import cast

import pytest

from rag_core import Engine
from rag_core.events import EventBuffer
from rag_core.events.types import AuditContext, SearchCompleted
from rag_core.integrations.langchain import (
    LangChainNotInstalledError,
    LangChainRetrieverConfig,
    build_langchain_retriever,
    context_pack_to_tool_output,
    create_langchain_context_tool,
    create_langchain_retriever_tool,
    search_result_to_document_kwargs,
)
from rag_core.integrations.langchain_retriever import search_langchain_documents
from rag_core.integrations.langchain_runtime import require_langchain_symbol
from tests.support import (
    FakeEmbeddingProvider,
    FakeSparseEmbedder,
    RecordingVectorStore,
    make_search_result,
    make_test_config,
)


def _make_core(collection: str, store: RecordingVectorStore) -> Engine:
    return Engine(
        make_test_config(qdrant_collection=collection, embedding_dimensions=4),
        embedding_provider=FakeEmbeddingProvider(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )


def test_search_result_to_document_kwargs_maps_payload_and_metadata() -> None:
    result = make_search_result(
        id="hit-42",
        text="Billing runs monthly.",
        score=0.87,
        document_id="billing-doc",
        collection="help-center",
        document_path="/private/docs/billing.md",
        metadata={"source": "faq"},
        section_title="Billing",
        section_path="Help > Billing",
        chunk_index=3,
    )

    payload = search_result_to_document_kwargs(result)

    assert payload["id"] == "hit-42"
    assert payload["page_content"] == "Billing runs monthly."
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source"] == "faq"
    assert metadata["rag_core_document_id"] == "billing-doc"
    assert metadata["rag_core_collection"] == "help-center"
    assert "rag_core_document_path" not in metadata
    assert metadata["rag_core_score"] == 0.87
    assert metadata["rag_core_chunk_index"] == 3
    assert "rag_core_document_key" not in metadata


def test_search_result_to_document_kwargs_omits_document_key_from_metadata() -> None:
    result = make_search_result(
        id="hit-1",
        text="secret body",
        document_key="private/billing.md",
        title=None,
    )

    metadata = search_result_to_document_kwargs(result)["metadata"]
    assert isinstance(metadata, dict)
    assert "rag_core_document_key" not in metadata


def test_create_langchain_context_tool_rejects_out_of_contract_limit() -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        create_langchain_context_tool(
            cast(Engine, object()),
            namespace="acme",
            collections=["help"],
            limit=100,
        )


def test_build_langchain_retriever_rejects_out_of_contract_limit() -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        build_langchain_retriever(
            cast(Engine, object()),
            namespace="acme",
            collections=["help"],
            limit=100,
        )


def test_context_pack_to_tool_output_returns_text_and_payload() -> None:
    from rag_core.search.context_pack import build_context_pack

    pack = build_context_pack(
        [make_search_result(id="hit-1", text="Context row", document_id="doc-1", chunk_index=0)],
        query="what happened?",
    )
    content, artifact = context_pack_to_tool_output(pack)
    snippets = cast(list[dict[str, object]], artifact["snippets"])

    assert content == pack.as_prompt_text()
    assert artifact["ok"] is True
    assert artifact["context_text"] == pack.as_prompt_text()
    assert artifact["query"] == "what happened?"
    assert snippets[0]["citation_id"] == "S1"


def test_context_pack_to_tool_output_omits_document_key_from_content() -> None:
    from rag_core.search.context_pack_models import (
        ContextSnippet,
        Context,
        SourceLocator,
        Citation,
    )

    pack = Context(
        query="billing",
        snippets=(
            ContextSnippet(
                citation_id="c1",
                rank=1,
                text="monthly invoice",
                score=0.9,
                source=Citation(
                    source_id="s1",
                    result_id="hit-1",
                    document_key="private/billing.md",
                    title=None,
                ),
                locator=SourceLocator(chunk_index=0),
                token_estimate=1,
                char_count=16,
                truncated=False,
            ),
        ),
        dropped_count=0,
        max_snippets=5,
        max_chars=3000,
        token_estimate=1,
    )

    content, artifact = context_pack_to_tool_output(pack)

    assert "private/billing.md" not in content
    assert "private/billing.md" not in str(artifact["context_text"])
    assert content == pack.as_prompt_text()
    assert artifact["context_text"] == pack.as_prompt_text()


def test_build_langchain_retriever_raises_when_langchain_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _patched_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("langchain_core"):
            raise ImportError("langchain-core not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    with pytest.raises(LangChainNotInstalledError, match="langchain-core"):
        build_langchain_retriever(
            cast(Engine, object()),
            namespace="acme",
            collections=["help"],
        )


def test_langchain_adapters_reject_blank_namespace_at_construction() -> None:
    with pytest.raises(ValueError, match="namespace must not be empty"):
        build_langchain_retriever(
            cast(Engine, object()),
            namespace="   ",
            collections=["help"],
        )

    with pytest.raises(ValueError, match="namespace must not be empty"):
        create_langchain_context_tool(
            cast(Engine, object()),
            namespace="   ",
            collections=["help"],
        )


def test_langchain_retriever_normalizes_bound_scope_before_search() -> None:
    pytest.importorskip("langchain_core")

    async def scenario() -> RecordingVectorStore:
        store = RecordingVectorStore(
            search_results=[make_search_result(id="hit-1", text="ok")]
        )
        core = _make_core("rag_core_langchain_retriever_namespace", store)
        try:
            retriever = build_langchain_retriever(
                core,
                namespace=" acme ",
                collections=[" help "],
                rerank=False,
            )
            await retriever.ainvoke("billing")
        finally:
            await core.close()
        return store

    store = asyncio.run(scenario())
    assert store.search_calls[0].namespace == "acme"
    assert store.search_calls[0].collections == ["help"]


def test_langchain_adapters_reject_blank_bound_scope_values() -> None:
    with pytest.raises(ValueError, match="collections must contain non-empty strings"):
        build_langchain_retriever(
            cast(Engine, object()),
            namespace="acme",
            collections=[" "],
        )

    with pytest.raises(ValueError, match="document_ids must contain non-empty strings"):
        create_langchain_context_tool(
            cast(Engine, object()),
            namespace="acme",
            collections=["help"],
            document_ids=[" "],
        )


def test_require_langchain_symbol_wraps_missing_symbol_attribute_error() -> None:
    module_name = "langchain_core._rag_core_test_missing_symbol"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    try:
        with pytest.raises(LangChainNotInstalledError, match="Missing `MissingSymbol`"):
            require_langchain_symbol(module_name, "MissingSymbol")
    finally:
        sys.modules.pop(module_name, None)


def test_langchain_retriever_and_tool_wire_into_rag_core() -> None:
    pytest.importorskip("langchain_core")

    async def scenario() -> None:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-1",
                    text="Card and ACH are both supported.",
                    document_id="billing-doc",
                    chunk_index=1,
                )
            ]
        )
        core = _make_core("rag_core_langchain_retriever", store)
        try:
            retriever = build_langchain_retriever(
                core,
                namespace="acme",
                collections=["help"],
                limit=3,
                rerank=False,
            )
            docs = await retriever.ainvoke("How can customers pay?")
            assert docs[0].page_content == "Card and ACH are both supported."
            assert docs[0].metadata["rag_core_document_id"] == "billing-doc"

            tool = create_langchain_retriever_tool(
                retriever,
                name="knowledge_lookup",
                description="Look up product knowledge.",
            )
            tool_output = await tool.ainvoke({"query": "payment methods"})
            assert isinstance(tool_output, str)
            assert "Card and ACH" in tool_output
        finally:
            await core.close()

    asyncio.run(scenario())


def test_langchain_retriever_threads_audit_context_to_events() -> None:
    pytest.importorskip("langchain_core")

    async def scenario() -> EventBuffer:
        buffer = EventBuffer()
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_langchain_retriever_audit",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=RecordingVectorStore(
                search_results=[make_search_result(id="hit-1", text="ok")]
            ),
            event_sink=buffer,
        )
        try:
            retriever = build_langchain_retriever(
                core,
                namespace="acme",
                collections=["help"],
                audit_context=AuditContext(actor="agent-user", request_id="req-1"),
            )
            await retriever.ainvoke("billing")
        finally:
            await core.close()
        return buffer

    buffer = asyncio.run(scenario())
    [completed] = [event for event in buffer.events if isinstance(event, SearchCompleted)]
    assert completed.actor == "agent-user"
    assert completed.request_id == "req-1"


def test_langchain_context_tool_returns_grounded_text_with_artifact_mode() -> None:
    pytest.importorskip("langchain_core")

    async def scenario() -> None:
        store = RecordingVectorStore(
            search_results=[
                make_search_result(
                    id="hit-7",
                    text="Invoices are charged at the beginning of each month.",
                    document_id="invoice-doc",
                    chunk_index=2,
                )
            ]
        )
        core = _make_core("rag_core_langchain_context_tool", store)
        try:
            tool = create_langchain_context_tool(
                core,
                namespace="acme",
                collections=["help"],
                name="rag_context",
                description="Fetch grounded context.",
                limit=2,
                rerank=False,
            )
            content = await tool.ainvoke({"query": "When are invoices charged?"})
            assert isinstance(content, str)
            assert "Invoices are charged" in content
            assert getattr(tool, "response_format") == "content_and_artifact"
        finally:
            await core.close()

    asyncio.run(scenario())


def test_search_helper_forwards_scope_options_to_vector_store() -> None:
    async def scenario() -> tuple[RecordingVectorStore, EventBuffer]:
        buffer = EventBuffer()
        store = RecordingVectorStore(
            search_results=[make_search_result(id="hit-1", text="ok", document_id="doc-1", chunk_index=0)]
        )
        core = Engine(
            make_test_config(
                qdrant_collection="rag_core_langchain_search_helper",
                embedding_dimensions=4,
            ),
            embedding_provider=FakeEmbeddingProvider(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
            event_sink=buffer,
        )
        try:
            await search_langchain_documents(
                core=core,
                query="billing",
                config=LangChainRetrieverConfig(
                    namespace="acme",
                    collections=("help",),
                    limit=2,
                    content_types=("document",),
                    document_ids=("doc-1",),
                    rerank=False,
                    use_lexical_search=False,
                    audit_context=AuditContext(actor="agent-user", request_id="req-1"),
                ),
            )
        finally:
            await core.close()
        return store, buffer

    store, buffer = asyncio.run(scenario())
    call = store.search_calls[0]
    assert call.namespace == "acme"
    assert call.collections == ["help"]
    assert call.content_types == ["document"]
    assert call.document_ids == ["doc-1"]
    assert call.limit == 2
    [completed] = [event for event in buffer.events if isinstance(event, SearchCompleted)]
    assert completed.actor == "agent-user"
    assert completed.request_id == "req-1"
