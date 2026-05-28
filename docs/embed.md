# Embed rag-core

Use one `RAGCore` instance per worker process. Your application owns authentication,
connectors, chat, and model calls. rag-core owns parse → chunk → index → search →
context packs and retrieval traces.

## Minimal loop

```python
import asyncio
from rag_core import RAGCore, RAGCoreConfig
from rag_core.config import EmbeddingConfig, QdrantConfig

async def main() -> None:
    config = RAGCoreConfig(
        qdrant=QdrantConfig(location="./rag-core-qdrant"),
        embedding=EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
        ),
    )
    async with RAGCore(config) as core:
        await core.ingest_bytes(
            file_bytes=b"Invoices can be paid by card or ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="acme",  # bound from your auth
            corpus_id="help",
        )
        pack = await core.retrieve_context(
            query="How can customers pay?",
            namespace="acme",
            corpus_ids=["help"],
            limit=8,
            rerank=False,  # pass True when a reranker is configured
        )
        print(pack.as_prompt_text())

asyncio.run(main())
```

No API keys yet? Use `from rag_core.demo import build_demo_core` — deterministic
embeddings for wiring checks only, not semantic retrieval. See [quickstart.md](quickstart.md).

## Tenancy

- **`namespace`** — opaque tenant or workspace key bound from your auth layer.
- **`corpus_id`** — logical corpus inside that namespace (product area, KB, project).
- **`document_id` / `document_key`** — stable document identity. Your app can
  supply external IDs; otherwise core derives deterministic keys from
  filename/path/URL inputs and hashes them into document IDs.

Never accept raw `namespace` / `corpus_id` from model tool calls without binding them
to the authenticated session in your HTTP handler.

## Source reconciliation

Use your app or worker queue to discover changed sources, then call core ingest/delete
primitives with stable `document_key` values. When a manifest directory is configured,
local and URL ingest can report unchanged, changed, missing, orphaned, and duplicate
source states. The status-only primitives are available from `rag_core.sources`
(`ManifestSource`, `reconcile_entries`, `ManifestReconciliation`). rag-core does not
own connector cursors, polling schedules, webhooks, or deleted-source policy.

## Retrieval defaults

| Surface | Default search behavior | Rerank |
| --- | --- | --- |
| `RAGCore.search` / `retrieve_context` | capability-aware default query plan; `balanced` when dense+sparse hybrid RRF is supported | off unless `rerank=True` and reranker configured |
| CLI `search` / `retrieve-context` | same default unless `--search-profile` or `--query-plan-preset` is passed | `--rerank` to enable |
| HTTP `POST /v1/search` | same default through the configured runtime providers | `"rerank": true` in JSON body |

`use_lexical_search` is the request flag for configured lexical/exact-match
expansion. Sparse query-plan channels and named search profiles are separate
retrieval primitives; the `lexical` search profile means sparse BM25 retrieval,
not the portable exact-match sidecar.

Default limits are intentionally surface-specific: search entrypoints return 10 hits,
first-run `local-search` returns 5 hits, context entrypoints return 8 snippets, and
prompt-facing `search_user_documents` tools default to 5 snippets to keep prompt
payloads compact.

In Python, `RAGCore.search` and `RAGCore.retrieve_context` accept
`document_ids`, `content_types`, and `metadata_filter` as app-owned narrowing
filters inside the bound namespace/corpus scope.

Pass `rerank=True` only when `RAGCoreConfig.reranker` (or CLI flags) point at a real
reranker provider. See [expectations.md](expectations.md).

## Agent tools

`rag_core.contracts` parses model tool payloads and provides helpers for app-bound
retrieval scope; your endpoint still owns auth and calls `retrieve_context`.
Examples: [search_endpoint.py](../examples/search_endpoint.py),
[vercel_ai_sdk_search_tool.ts](../examples/vercel_ai_sdk_search_tool.ts).

The Vercel AI SDK example targets stable AI SDK v6 tool contracts verified by
`./scripts/verify_vercel_ai_sdk_example.sh` against current `ai@^6.0.0`
TypeScript declarations: tools use `inputSchema`, streamed text arrives as
`text-delta` parts on `fullStream`, and tool results
expose payloads as `output`, not the older `parameters`, `text`, or `result`
names. It uses `toModelOutput` to send the model compact prompt-facing context
text as a `text` tool output while keeping the full typed search payload
available to application code through `toolResults` and stream `tool-result`
parts. The streaming path also handles `tool-error` and `error` stream parts
explicitly. The example model id is illustrative; verify the current Vercel AI
Gateway model list before copying it into an application.

It is not a v7 beta contract.

Optional integration adapters follow the same boundary: bind `namespace`, `corpus_ids`,
and optional `content_types` in application code, use the contract scope helpers for static document allowlists,
and treat a tool-call `document_ids` value as only a narrowing filter inside that bound scope.
Bind advanced retrieval controls such as `query_plan` in application code too; do not expose them as
model-call inputs.

## Worker lifecycle

Long-running applications should construct `RAGCore` once and reuse it. See
[embedded_service.py](../examples/embedded_service.py).

## Evals in your repo

Keep labeled cases in your application repository and run `rag_core.evals` against
your configured embeddings — not the library’s CI fixture corpus. See
[retrieval_eval.py](../examples/retrieval_eval.py). Use `expected_ids` for the
relevant chunk or document ids you expect retrieval to return; legacy
`expected_chunk_ids` case files still load for compatibility.

For a no-key folder check before wiring real providers, `local-eval` indexes a
local file or folder with deterministic demo embeddings, infers `namespace` and
`corpus_id` from the JSONL cases, and emits the same redacted eval report:

```bash
uv run rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl \
  --min-recall-at-5 1 --min-mrr 1 --json
```

For local folder cases, `expected_ids` may use relative paths such as
`billing.md`; `local-eval` resolves them to the indexed local document keys.

## Configured example (credentials required)

```bash
export OPENAI_API_KEY=sk-...
uv run python -m examples.configured_retrieval
```

This command is not executed by default CI because it requires credentials; the
example file is still source-checked and packaged. Smoke without keys:
`python -m rag_core.quickstart` or `examples/embedded_service`.

## Reference

| Topic | Doc |
| --- | --- |
| Hit and context JSON | [expectations.md](expectations.md) |
| Public surface stability | [stability.md](stability.md) |
| Providers and extras | [providers.md](providers.md) |
| Optional HTTP runtime | [self-host.md](self-host.md) |
