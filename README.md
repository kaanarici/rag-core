# rag-core

`rag-core` is a Python retrieval engine for applications, services, and workers that need embedded RAG.

It covers the retrieval path between documents and model calls: parsing, PDF routing, chunking, indexing, hybrid search, reranking, citations, context assembly, traces, and evals.

It is not a hosted platform, chat framework, UI, queue system, or auth layer. Your app owns those pieces. `rag-core` owns the retrieval layer.

Good starting points:

- `demo` for the shortest zero-config smoke
- `local-search` for a no-key first run on your own folder using embedded Qdrant
- `RAGCore` in your app when you want persistent indexes, app-owned auth and scope, and explicit provider config

## Quick Start

Prerequisites: Python 3.11+ and `uv`.

Run a no-key local search against your own files:

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync
mkdir -p /tmp/rag-core-quickstart
printf "Billing invoices can be paid by card or ACH. Audit logs export as CSV.\\n" > /tmp/rag-core-quickstart/guide.md
uv run rag-core demo --json
uv run rag-core local-search /tmp/rag-core-quickstart "How can invoices be paid?" --json
```

`demo` is the shortest zero-config check. `local-search` indexes a folder into embedded Qdrant with local demo providers, runs the query, and prints the top chunks.

`local-search` returns raw search hits, not a model-ready context pack. Pass `--max-files` when you want a larger local smoke. For no-key context text with citations, run the small library example below. Use `retrieve-context` once you have a persistent index and provider config.

To capture a trace:

```bash
uv run rag-core local-search examples/demo_corpus "corpus lifecycle" \
  --events-jsonl traces.jsonl
uv run rag-core trace-summary traces.jsonl --json
```

`trace-summary` summarizes the retrieval trace: plan shape, stages, timing, rerank and lexical diagnostics, and sanitized errors. It does not prove the context-pack or citation payload shape by itself.

The `examples/` modules below are checkout examples. They are not installed into the wheel. For installed-package code, use the library snippet in [Library Usage](#library-usage).

For model-ready context with citations:

```bash
uv run python -m examples.minimal_app
```

For an endpoint helper that binds scope before retrieval:

```bash
uv run python -m examples.search_endpoint
```

For local, archive, and URL source ingest through one `RAGCore`:

```bash
uv run python -m examples.source_ingest
```

For a no-key retrieval eval with quality gates:

```bash
uv run python -m examples.retrieval_eval
```

## Install

Install from GitHub source:

```bash
uv add "rag-core @ git+https://github.com/kaanarici/rag-core.git"
```

For local development:

```bash
uv sync
uv sync --group dev
```

Optional extras are grouped by integration and provider:

```bash
uv sync --extra rerank --extra voyage --extra zeroentropy
uv sync --extra semantic --extra html --extra anthropic
uv sync --extra turbopuffer --extra opentelemetry
uv sync --extra langchain --extra openai-agents
```

For installed-package consumers, request extras on the package spec:

```bash
uv add "rag-core[langchain,openai-agents] @ git+https://github.com/kaanarici/rag-core.git"
uv add "rag-core[turbopuffer] @ git+https://github.com/kaanarici/rag-core.git"
```

Declared extras: `rerank` for Cohere reranking, `voyage` for Voyage embeddings and reranking, `zeroentropy` for ZeroEntropy embeddings and reranking, `semantic` for semantic/code chunking helpers, `html` for the faster Rust-backed HTML converter path, `anthropic` for chunk contextualization, `turbopuffer` for the TurboPuffer vector store, `opentelemetry` for tracing sinks, `langchain` for LangChain tools, and `openai-agents` for OpenAI Agents tools. Base installs still parse HTML through the fallback converter.

For PDF routing with page-level OCR signals, install PDF Inspector:

```bash
cargo install --git https://github.com/firecrawl/pdf-inspector --bin detect-pdf --bin pdf2md
uv run rag-core doctor --json
```

## Mental Model

`rag-core` keeps the public model small:

- **Sources** are files, bytes, ZIP members, or explicit URLs.
- **Documents** are parsed source items with stable identity and metadata.
- **Corpora** are application-owned partitions, scoped by `namespace` and `corpus_id`.
- **Collections** are physical vector-store indexes.
- **Indexes** are stored chunks, vectors, sparse signals, payloads, and document records.
- **Search profiles** are named retrieval shapes such as `balanced`, `fast`, `lexical`, `coverage`, and `diverse`.
- **Query plans** are lower-level typed retrieval recipes when you need exact control.
- **Context packs** turn search results into model-ready text with citations and source previews.
- **Events, traces, and evals** make ingest and retrieval behavior inspectable.

The normal flow:

```text
sources -> parse -> chunk -> index -> search -> context pack -> your model call
```

## Persistent Search

Use Qdrant for the default persistent path:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant &
export OPENAI_API_KEY=...

uv run rag-core ingest examples \
  --namespace acme \
  --corpus-id help \
  --qdrant-url http://localhost:6333 \
  --embedding-model text-embedding-3-small \
  --embedding-dimensions 1536

uv run rag-core search "corpus lifecycle" \
  --namespace acme \
  --corpus-id help \
  --qdrant-url http://localhost:6333 \
  --embedding-model text-embedding-3-small \
  --embedding-dimensions 1536 \
  --search-profile balanced \
  --json
```

Supported local ingest includes text, code, HTML, CSV/TSV, JSON/JSONL, XML, PDF, DOCX, PPTX, and XLSX. Images require an injected OCR provider. Binary Office files such as `.doc`, `.ppt`, and `.xls` are skipped.

See [docs/parsing/formats.md](docs/parsing/formats.md) for the full format matrix.

## CLI

The CLI uses the same concepts as the library. Common commands:

```bash
uv run rag-core doctor --json
uv run rag-core local-search /path/to/folder "billing policy" --limit 5
uv run rag-core ingest /path/to/folder --namespace acme --corpus-id help --json
uv run rag-core ingest-url https://example.com/docs/guide --namespace acme --corpus-id help --json
uv run rag-core ingest-urls urls.txt --namespace acme --corpus-id help --json
uv run rag-core discover-remote https://example.com/llms.txt --kind llms-txt --json
uv run rag-core search "billing policy" --namespace acme --corpus-id help --search-profile balanced --json
uv run rag-core retrieve-context "billing policy" --namespace acme --corpus-id help --qdrant-url http://localhost:6333
uv run rag-core trace-summary traces.jsonl --json
uv run rag-core eval --cases cases.jsonl --qdrant-url http://localhost:6333 --search-profile balanced --json
```

Useful notes:

- `doctor` reports runtime metadata, processing fingerprints, provider readiness, vector-store diagnostics, and retrieval-profile descriptions without printing secrets.
- `ingest` supports local files, directories, and globs. It writes JSONL manifests and skips unchanged documents by content hash.
- `ingest-archive`, `ingest-url`, and `ingest-urls` add ZIP and explicit remote-source paths. They do not crawl.
- `discover-remote` reads a sitemap or `llms.txt` artifact and can write URLs for later ingest. Output URL files are redacted by default; any query-bearing discovered URL makes the redacted file lossy, so the command refuses it unless you pass `--output-url-file-raw-queries`.
- `search` returns raw search hits.
- `retrieve-context` returns the library context-pack payload with `context_text`.
- `eval` runs retrieval-quality cases and can fail CI with recall, MRR, NDCG, or latency gates.
- Most commands with `--json` write one JSON document to stdout. Batch ingest commands such as `ingest`, `ingest-archive`, and `ingest-urls` stream one JSON object per record. `--events-jsonl`, manifest files, eval case files, and batch ingest stdout are JSONL: one JSON object per line.

## Library Usage

Create one `RAGCore` per app process or worker. Reuse it for requests and jobs. Close it on shutdown.

For a no-key embedded smoke, use the package demo helper. It creates an in-memory Qdrant-backed `RAGCore` with local demo providers, so it works from an installed wheel without API keys:

```python
import asyncio

from rag_core.demo import build_demo_core


async def main() -> None:
    async with build_demo_core(collection="quickstart") as core:
        await core.ingest_bytes(
            file_bytes=b"Billing is due monthly and invoices can be paid by card.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",
            corpus_id="help-center",
        )

        context = await core.retrieve_context(
            query="How can I pay invoices?",
            namespace="acme",
            corpus_ids=["help-center"],
            limit=3,
            rerank=False,
        )

        print(context.as_text())
        print([citation.source_id for citation in context.citations])


asyncio.run(main())
```

For a no-key persistent local smoke, give the same helper a Qdrant storage directory:

```python
async with build_demo_core(
    collection="quickstart",
    qdrant_location="./rag-core-qdrant",
) as core:
    ...
```

For a persistent production-style path, configure Qdrant and your embedding provider explicitly:

```python
import asyncio

from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig


async def main() -> None:
    core = RAGCore(
        RAGCoreConfig(
            qdrant=QdrantConfig(
                url="http://localhost:6333",
                collection="product_docs",
            ),
            embedding=EmbeddingConfig(
                model="text-embedding-3-small",
                dimensions=1536,
            ),
        )
    )

    async with core:
        await core.ingest_file(
            "/path/to/faq.pdf",
            namespace="acme",
            corpus_id="help-center",
        )

        context = await core.retrieve_context(
            query="How does billing work?",
            namespace="acme",
            corpus_ids=["help-center"],
            limit=5,
            max_chars=4_000,
        )

        print(context.as_text())


asyncio.run(main())
```

Use `search(...)` when you want raw ranked `SearchResult` rows. Use `retrieve_context(...)` when you want a model-ready context pack with citations, snippets, source previews, truncation state, and approximate token counts.

The CLI and library share the same concepts. `local-search` is the shortest end-to-end check, while `ingest`, `search`, and `retrieve_context(...)` are the persistent building blocks you will usually keep in an application.

For application tools, expose retrieval as something like `search_user_documents`. Bind tenant scope in your app, and let the model provide only the query and bounded retrieval knobs.

## Providers

Default runtime shape:

- vector store: Qdrant
- dense embeddings: OpenAI
- sparse retrieval: FastEmbed BM25, with optional SPLADE
- reranker: off by default, optional Cohere, Voyage, or ZeroEntropy
- PDF path: PDF Inspector when available, PyMuPDF fallback
- contextualization: off by default, optional Anthropic chunk contextualizer

TurboPuffer is available as an optional first-party vector-store adapter:

```bash
uv sync --extra turbopuffer
uv run rag-core doctor --vector-store turbopuffer --turbopuffer-namespace product_docs --json
```

Provider categories can be configured directly, registered, or injected:

- Config-backed or CLI-backed: dense embeddings, rerankers, vector stores, and embedding caches.
- Programmatic config-backed: lexical search sidecars.
- Injection-only at `RAGCore(...)`: custom sparse embedders, OCR providers, event sinks, chunk contextualizers, and chunk context caches.
- Registry or factory helpers are available for app assembly, but not every registry is selected from `RAGCoreConfig`.

See [docs/providers/custom-providers.md](docs/providers/custom-providers.md) for extension points and [docs/providers/vector-stores.md](docs/providers/vector-stores.md) for vector-store support levels.

`rag-core` does not build full app config from environment variables. Your app should read its own env and pass explicit config into `RAGCoreConfig`. Provider SDKs may still read their own API keys, such as `OPENAI_API_KEY`, `COHERE_API_KEY`, `VOYAGE_API_KEY`, `ZEROENTROPY_API_KEY`, or `TURBOPUFFER_API_KEY`.

## Retrieval Control

For common behavior, use a search profile. For exact control, pass a typed query plan. For reranking, cap the reranker separately from retrieval:

```python
from rag_core.search import RerankBudget, search_profile

results = await core.search(
    query="billing policy",
    namespace="acme",
    corpus_ids=["help"],
    limit=10,
    query_plan=search_profile("balanced", limit=10),
    rerank=True,
    rerank_budget=RerankBudget(
        candidate_count=40,
        max_output=10,
        timeout_seconds=1.5,
    ),
)
```

Use `describe_runtime()["retrieval"]`, `rag-core doctor --json`, or `describe_retrieval_profiles()` from `rag_core.search` when your app needs to display profile and query-plan behavior.

## Observability And Evals

Use `--events-jsonl` or an event sink to inspect what happened during ingest and search. Trace summaries include search plan shape, stage timings, rerank diagnostics, lexical diagnostics, embedding cache totals, and sanitized errors.

Embedded services can use `EventBuffer`, `summarize_search_trace(...)`, and `summarize_embedding_trace(...)` from `rag_core.events`. `OpenTelemetrySink` is available through the `opentelemetry` extra and omits sensitive fields by default. Retrieval evals use JSONL cases with expected chunk IDs and optional quality gates. See [docs/evals/retrieval-quality.md](docs/evals/retrieval-quality.md).

## Integrations

Optional adapters sit above `search(...)` and `retrieve_context(...)`:

- OpenAI Agents SDK: `from rag_core.integrations import build_retrieve_context_tool`
- LangChain and LangGraph: `from rag_core.integrations import create_langchain_context_tool`
- Vercel AI SDK tool contract: [docs/integrations/vercel-ai-sdk-tools.md](docs/integrations/vercel-ai-sdk-tools.md)

More runnable examples:

- [examples/minimal_app.py](examples/minimal_app.py)
- [examples/chatbot_context.py](examples/chatbot_context.py)
- [examples/search_endpoint.py](examples/search_endpoint.py)
- [examples/source_ingest.py](examples/source_ingest.py)
- [examples/corpus_lifecycle.py](examples/corpus_lifecycle.py)
- [examples/retrieval_eval.py](examples/retrieval_eval.py)
- [examples/pdf_ocr_path.py](examples/pdf_ocr_path.py)

## Troubleshooting

- Run `uv run rag-core demo --json` first, then use `doctor` when you need provider or vector-store diagnostics.
- If Qdrant checks fail, use exactly one of `--qdrant-url` or `--qdrant-location`.
- If PDF Inspector is installed outside `PATH`, set `PDF_INSPECTOR_BINARY_PATH` to the directory that contains `detect-pdf` and `pdf2md`.
- If a provider fails at runtime, install the matching extra, set the provider API key, then inspect `doctor` output.
- If `.doc`, `.ppt`, or `.xls` files are skipped, convert them to `.docx`, `.pptx`, or `.xlsx`.
- Run examples as modules from a local checkout, such as `uv run python -m examples.corpus_lifecycle`.

## Validation

The full validation loop is a source-checkout workflow:

```bash
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
uv run python scripts/architecture_pressure.py --json
uv build
uv run python scripts/wheel_smoke.py
```
