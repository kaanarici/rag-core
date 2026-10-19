# Python API

## Search a folder

```python
from rag_core.easy import index

with index("./documents", persist_dir="./indexes") as docs:
    result = docs.search("How are invoices paid?")
    print(docs.context("How are invoices paid?"))
    docs.refresh()
```

Opening refreshes the folder. Search uses dense embeddings plus BM25 by default.
Pass `hybrid=False` for dense retrieval. `docs.persist_dir` identifies the actual
index directory. Close one embedded index before another process opens it.

## Bind application scope

This complete async example stores an application-owned document and retrieves it:

```python
import asyncio
from rag_core import Config, Document, RAGCore

async def main():
    config = Config.local("./rag-index", hybrid=True)
    async with RAGCore(config, tenant_id="acme", collection="docs") as rag:
        document = await rag.ingest(Document(
            key="billing.md",
            content=b"Invoices can be paid by ACH bank transfer.",
            content_type="text/markdown",
        ))
        evidence = await rag.search("How are invoices paid?")
        print(evidence.evidence[0].text)
        context = await rag.context("How are invoices paid?")
        print(context.as_prompt_text())
        payload = await rag.tool()("How are invoices paid?", limit=5)
        print(payload["context_text"])
        await rag.delete(document.document_id)

asyncio.run(main())
```

`RAGCore` binds tenant, collection, and optional `document_ids`. The app selects
that scope before exposing the tool. Models cannot select another scope. The
application still owns authentication and document-access policy.

Search returns `Evidence` with text, document identity, locators, and equivalent
sources for duplicate chunks. Context packs citations within character or token
limits. `rag.tool()` is a plain async function with `query` and `limit` arguments.
Wrap it with your application's SDK when needed.

`Config.local()` without a directory is ephemeral and dense. A directory enables
persistence. `hybrid=True` enables dense and BM25 search. For remote Qdrant, use
`Config.qdrant(url=..., embedding_provider="local", hybrid=True)`. See the
[remote storage requirements](dependability.md#remote-storage-contract).

## Inject providers

Pass objects directly to `Engine`, then bind it through `RAGCore`:

```python
from rag_core.core import Engine

engine = Engine(config, embedding_provider=my_model, vector_store=my_store,
                reranker=my_reranker)
rag = RAGCore(engine, tenant_id="acme", collection="docs")
```

`config` and the provider objects in this snippet are supplied by the application.
Optional constructor arguments include `sparse_embedder`, `ocr_provider`,
`chunk_contextualizer`, and `event_sink`. Protocols live in
`rag_core.search.provider_protocols`, `rag_core.documents.ocr`, and
`rag_core.documents.contextualizer`. There is no registration step.

`Engine.search` and `Engine.context` accept `mode="dense"`, `"hybrid"`, or
`"lexical"`. Omitting it uses the configured default. `RAGCore.search` and
`RAGCore.context` use that default and accept `rerank=True` when a reranker is
injected. Reranking is off by default on the scoped facade.

## TurboPuffer

Install the `turbopuffer` extra and set `TURBOPUFFER_API_KEY` in the environment:

```python
from rag_core import Config, RAGCore
from rag_core.core import Engine
from rag_core.search.providers.turbopuffer_store import TurboPufferVectorStore

store = TurboPufferVectorStore(namespace="app-docs", dense_dimensions=384,
                               region="gcp-us-central1")
engine = Engine(Config.local(hybrid=True), vector_store=store)
rag = RAGCore(engine, tenant_id="acme", collection="docs")
```

The app owns namespace and region selection. Every query also carries tenant and
collection filters. Nested metadata is preserved as JSON; only flat attributes
are filterable. Native BM25 avoids loading a local sparse model. This adapter
requires its own schema and embedding identity. Reingest into a fresh namespace
when moving from an older schema.

## Preparation, updates, and output

`Engine` owns `add`, `add_file`, `add_bytes`, `add_url`, `add_archive`, `search`,
`context`, and scoped deletion. `prepare_file` and `prepare_bytes` preview parsed
sources and chunks. See [formats](formats.md) for OCR and location metadata.

Changing chunk settings or the default processing recipe reindexes a document
on its next ingest. Applications using custom processing versions own their
recipe identity. Changing embedding models or dimensions requires a fresh index.
See [migration](migration.md) before reusing an older index.

For caller-prepared chunks, use `DocumentIndexer` and
`IndexRequest(prepared_chunks=...)`. `prepare_pre_chunked_texts` constructs chunks
from supplied text. Applications using this lower-level path own ingest lifecycle.

Use `Context.as_prompt_text()` or `to_prompt_payload()` for model-facing output.
Prompt locators omit `source_hash` and private source paths. Full debug payloads
are an explicit application choice. Close the owning engine when the app exits.
