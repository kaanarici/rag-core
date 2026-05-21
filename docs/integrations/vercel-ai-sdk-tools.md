# Vercel AI SDK Tool Contract

This repo is Python-first. The Vercel AI SDK example shows an application endpoint contract: `rag-core` handles retrieval, your endpoint owns auth and scope, and the SDK tool calls that endpoint.

The key boundary: the model does not choose tenant scope. Your app endpoint binds `namespace`, `corpus_ids`, auth, and any allowed document filters before it calls `RAGCore.retrieve_context(...)`.

## Example Flow

Read the examples in this order:

- `examples/minimal_app.py` for the smallest local ingest-plus-retrieve loop
- `examples/chatbot_context.py` for the app-side retrieval helper that returns a `ModelContextPack`
- `examples/search_endpoint.py` for the endpoint helper that validates tool input and binds authorized scope
- `examples/vercel_ai_sdk_search_tool.ts` for the Vercel AI SDK tool that calls that endpoint

That sequence shows a complete path: seed a corpus, retrieve context in Python, expose retrieval behind an app endpoint, then let the SDK tool call that endpoint.

These files are source-checkout examples. They are not installed into the wheel. Installed-package users should import the Python surfaces from `rag_core` and `rag_core.contracts`, then keep the TypeScript tool code in their own Vercel app.

## Contract: `search_user_documents`

Reference Python API:

- `rag_core.contracts.search_user_documents_tool_contract`

The contract has three stable pieces:

- `tool_name`: `search_user_documents`
- `input_schema`: JSON Schema for model-visible tool inputs (`query` plus optional retrieval knobs)
- `output_schema`: JSON Schema for context-pack style tool results (`ok`, `context_text`, snippets, citations, source previews, compact citation summary, truncation metadata, and nested source locators)

Canonical contract details to mirror in app-layer TypeScript validators:

- `query` and `document_ids[*]` must contain at least one non-whitespace character.
- snippet objects may include optional `retrieval_metadata` (for example rerank diagnostics).

Use `search_user_documents_tool_contract()` when you want one JSON-serializable payload for your app layer. Use `parse_search_user_documents_request(...)` to validate and normalize model-visible input before your endpoint calls retrieval.

Installed-package import surfaces:

- `from rag_core import RAGCore, RAGCoreConfig`
- `from rag_core.contracts import parse_search_user_documents_request, search_user_documents_tool_contract, search_user_documents_tool_result`
- `RAGCore.retrieve_context(...)` for the model-ready context pack your endpoint returns through `search_user_documents_tool_result(...)`

## TypeScript Example (App-Owned Endpoint)

Use these files as the starting point:

- `examples/vercel_ai_sdk_search_tool.ts`
- `examples/search_endpoint.py` for the Python endpoint shape

It demonstrates:

- `tool({ inputSchema, execute })` with `jsonSchema(...)`
- `generateText(...)` with tools
- `streamText(...)` with `onStepFinish` and streamed `tool-result` parts
- `execute` calling your endpoint (`/api/search-user-documents`) via `fetch`
- a framework-neutral Python endpoint helper that parses model-visible input, binds app-owned scope, authorizes document filters, calls `retrieve_context(...)`, and returns the canonical tool-result payload

Your app endpoint should accept input-schema JSON and return output-schema JSON. Keep auth and tenancy checks in your app endpoint, not in `rag-core`.

Minimal endpoint shape:

```python
from rag_core.contracts import (
    parse_search_user_documents_request,
    search_user_documents_tool_result,
)

async def search_user_documents_endpoint(input: dict[str, object]) -> dict[str, object]:
    request = parse_search_user_documents_request(input)
    document_ids = authorized_document_ids(request.document_ids)
    # If request.document_ids is omitted, default to the authorized allowlist
    # instead of searching the full corpus.

    pack = await core.retrieve_context(
        query=request.query,
        namespace=current_workspace.namespace,
        corpus_ids=current_workspace.authorized_corpus_ids,
        limit=request.limit,
        document_ids=document_ids,
        rerank=request.rerank,
        use_lexical_search=request.use_lexical_search,
        max_chars=request.max_chars,
        max_tokens=request.max_tokens,
    )
    return search_user_documents_tool_result(pack)
```

Do not forward arbitrary `namespace` or `corpus_ids` supplied by an LLM. If your app needs corpus selection, map user-facing labels to authorized corpus IDs server-side.
