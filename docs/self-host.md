# Optional HTTP runtime

Thin HTTP layer over the same `RAGCore` library you embed. Exposes health, runtime description, async ingest jobs, search, and retrieve-context only (no eval HTTP).

**Contract:** [openapi.yaml](self-host/openapi.yaml) · **Retrieval shapes:** [expectations.md](expectations.md)

## Docker Compose (fastest)

```bash
docker compose up -d --build
curl -s http://127.0.0.1:8787/health/ready
```

The default Compose file uses demo embeddings and Qdrant so the runtime can boot
without provider keys. Use the configured runtime path below for OpenAI
embeddings, or treat `.env.example` as a starting point for your own deployment
wrapper.

## Local serve (no API keys)

```bash
uv sync --extra runtime
uv run rag-core serve \
  --host 127.0.0.1 --port 8787 \
  --ingest-root "$PWD" \
  --job-db-path .rag-core/runtime/jobs.sqlite3 \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-model demo-dense-v1 \
  --embedding-dimensions 64
```

```bash
export BASE_URL=http://127.0.0.1:8787
curl -s "$BASE_URL/health/ready"

JOB=$(curl -s -X POST "$BASE_URL/v1/ingest" \
  -H 'Content-Type: application/json' \
  -d '{"path":"examples/demo_corpus/billing.md","namespace":"acme","corpus_id":"help"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')

until curl -s "$BASE_URL/v1/ingest/$JOB" | grep -q '"status": "completed"'; do sleep 0.2; done

curl -s -X POST "$BASE_URL/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"How can invoices be paid?","namespace":"acme","corpus_ids":["help"],"limit":3}'
```

Or `./scripts/self_host_smoke.sh` with the server already running.

## Configured runtime (Qdrant + OpenAI)

```bash
docker compose up -d qdrant
export RAG_CORE_QDRANT_URL=http://127.0.0.1:6333
export OPENAI_API_KEY=sk-...
uv run rag-core serve --host 0.0.0.0 --port 8787 \
  --qdrant-url "$RAG_CORE_QDRANT_URL" \
  --embedding-provider openai --embedding-model text-embedding-3-small \
  --embedding-dimensions 1536
```

`POST /v1/ingest` `path` must exist on the server host.
By default, `serve` accepts ingest paths under the server working directory only. If
you pass `--ingest-root /srv/docs`, the allowlist becomes the explicit roots you
configured.

## Operational Boundary

`serve` is a thin adapter intended to sit behind an application gateway. Put auth,
API-key management, tenancy, rate limits, connector sync, retry scheduling, and
deleted-source policy in your gateway, BFF, or worker system. Bind `namespace`,
`corpus_id`, `corpus_ids`, and the selected ingest root from the authenticated
session before proxying requests.

## HTTP surface

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `GET` | `/health/ready` | Readiness (`503` when not ready) |
| `GET` | `/v1/runtime` | `describe_runtime()` JSON |
| `POST` | `/v1/ingest` | Start job → `202` + `job_id` |
| `GET` | `/v1/ingest/{job_id}` | Job status |
| `POST` | `/v1/search` | Search hits JSON (`SearchResult`) |
| `POST` | `/v1/retrieve-context` | App-facing context pack + prompt-safe `context_text` |

Errors: JSON envelope with `error.code`, `error.message`, `error.details` — see OpenAPI.

Retrieval requests accept optional `content_types` and `document_ids` to narrow the
app-bound namespace/corpus scope. Search and retrieve-context also accept
`use_lexical_search` to disable configured lexical/exact-match expansion for a
request. The HTTP runtime does not accept raw `query_plan` or `search_profile`
fields; it uses the configured runtime providers' capability-aware defaults.
Only `/v1/retrieve-context` accepts `max_chars` and `max_tokens` prompt-context
budgets.

## Configuration

| Environment variable | CLI flag | Notes |
|---------------------|----------|-------|
| `RAG_CORE_VECTOR_STORE` | `--vector-store` | `qdrant` by default; `turbopuffer` requires the optional extra |
| `RAG_CORE_QDRANT_URL` | `--qdrant-url` | Use **one** of URL or location |
| `RAG_CORE_QDRANT_LOCATION` | `--qdrant-location` | `:memory:` for smoke |
| `RAG_CORE_QDRANT_API_KEY` | `--qdrant-api-key` | Cloud Qdrant |
| `RAG_CORE_QDRANT_COLLECTION` | `--qdrant-collection` | default `rag_core_chunks` |
| `RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION` | `--dimension-aware-collection` / `--no-dimension-aware-collection` | defaults on; includes embedding dimensions in automatic collection names |
| `RAG_CORE_TURBOPUFFER_NAMESPACE` | `--turbopuffer-namespace` | Physical TurboPuffer namespace; distinct from request `namespace` tenancy |
| `TURBOPUFFER_API_KEY` | `--turbopuffer-api-key` | TurboPuffer credentials |
| `TURBOPUFFER_REGION` | `--turbopuffer-region` | TurboPuffer region |
| `TURBOPUFFER_BASE_URL` | `--turbopuffer-base-url` | Optional TurboPuffer API base URL |
| `RAG_CORE_TURBOPUFFER_DISTANCE_METRIC` | `--turbopuffer-distance-metric` | default `cosine_distance` |
| `RAG_CORE_TURBOPUFFER_DELETE_CONTINUATION_LIMIT` | `--turbopuffer-delete-continuation-limit` | fail closed after this many partial delete continuations |
| `RAG_CORE_EMBEDDING_PROVIDER` | `--embedding-provider` | `demo` for no-key |
| `RAG_CORE_EMBEDDING_MODEL` | `--embedding-model` | |
| `RAG_CORE_EMBEDDING_DIMENSIONS` | `--embedding-dimensions` | Required for `demo` |
| `RAG_CORE_EMBEDDING_BATCH_SIZE` | `--embedding-batch-size` | document embedding batch size |
| `OPENAI_API_KEY` | — | OpenAI embeddings |
| `RAG_CORE_RERANKER_PROVIDER` | `--reranker-provider` | optional |
| `RAG_CORE_RERANKER_MODEL` | `--reranker-model` | optional |
| `RAG_CORE_PROCESSING_VERSION` | `--processing-version` | reindex trigger |
| `RAG_CORE_RUNTIME_JOB_DB_PATH` | `--job-db-path` | SQLite path for ingest job status persistence |

Serve-only flags: `--host`, `--port`, `--ingest-root`, `--job-db-path`.

Ingest jobs persist to `.rag-core/runtime/jobs.sqlite3` under the server working
directory unless `--job-db-path` or `RAG_CORE_RUNTIME_JOB_DB_PATH` overrides it.
Mount the parent directory for container restarts.

## Auth (app-owned)

`serve` does not ship accounts, API keys, or tenancy. Put auth in your gateway or BFF:

1. Terminate TLS at the gateway.
2. Validate identity before proxying to `serve`.
3. Map the principal to `namespace` / `corpus_id` server-side — do not trust unvalidated client tenancy fields.

Never expose `serve` on the public internet without auth. Health routes are often left open for deployment health checks; lock them down if required.

Example API-key middleware (add in your deployment wrapper — not enabled in the library):

```python
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from rag_core.runtime.errors import api_error

class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/health"):
            return await call_next(request)
        if request.headers.get("x-api-key") != os.environ["RAG_CORE_API_KEY"]:
            return api_error(code="unauthorized", message="Invalid or missing API key", status_code=401)
        return await call_next(request)
```

## Troubleshooting

- **`ModuleNotFoundError: starlette`** — `uv sync --extra runtime`
- **Qdrant errors** — use exactly one of `--qdrant-url` or `--qdrant-location`
- **Embedding auth** — Path A demo flags or `OPENAI_API_KEY` for OpenAI
- **Ingest failed** — `path` must exist on the server and stay under an ingest root; check job status and logs
