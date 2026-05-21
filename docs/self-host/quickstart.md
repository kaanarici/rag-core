# Self-host quickstart

Run **rag-core** as a thin HTTP layer over the same `RAGCore` library your app embeds. v1 exposes health, runtime description, async ingest jobs, search, and retrieve-context only (no eval HTTP).

**Contract references:** [openapi.yaml](openapi.yaml) (machine-readable surface), [auth.md](auth.md) (app-owned auth recipe), [config.md](config.md) (env ↔ CLI matrix).

## Path 0 — Docker Compose (Qdrant + API)

One command brings up persistent Qdrant and `rag-core serve` (demo embeddings, no API keys):

```bash
docker compose up -d --build
curl -s http://127.0.0.1:8787/health/ready
```

Then ingest/search with the same bodies as Path A below (`BASE_URL=http://127.0.0.1:8787`). Switch to OpenAI embeddings via `.env` and [config.md](config.md).

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Optional: Docker for persistent Qdrant (`compose.yaml` in the repo root)

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync --extra runtime
```

## Path A — No API keys (fastest smoke)

Uses demo embeddings and an in-memory vector store. Good for CI and “is `serve` alive?” checks.

```bash
uv run rag-core serve \
  --host 127.0.0.1 \
  --port 8787 \
  --qdrant-location :memory: \
  --embedding-provider demo \
  --embedding-model demo-dense-v1 \
  --embedding-dimensions 64
```

In another terminal:

```bash
export BASE_URL=http://127.0.0.1:8787
curl -s "$BASE_URL/health"
curl -s "$BASE_URL/health/ready"
curl -s "$BASE_URL/v1/runtime" | head -c 400 && echo

# Ingest a file from the repo (async job)
JOB=$(curl -s -X POST "$BASE_URL/v1/ingest" \
  -H 'Content-Type: application/json' \
  -d '{"path":"examples/demo_corpus/billing.md","namespace":"acme","corpus_id":"help"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')

until curl -s "$BASE_URL/v1/ingest/$JOB" | grep -q '"status": "completed"'; do sleep 0.2; done

curl -s -X POST "$BASE_URL/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"How can invoices be paid?","namespace":"acme","corpus_ids":["help"],"limit":3}'

curl -s -X POST "$BASE_URL/v1/retrieve-context" \
  -H 'Content-Type: application/json' \
  -d '{"query":"invoice payment","namespace":"acme","corpus_ids":["help"],"limit":3}'
```

Or run the bundled smoke script (server must already be running):

```bash
./scripts/self_host_smoke.sh
```

## Path B — Docker Qdrant + OpenAI (production-like)

1. Start Qdrant:

```bash
docker compose up -d
cp .env.example .env
# Set OPENAI_API_KEY in .env, then export vars or use direnv
export RAG_CORE_QDRANT_URL=http://127.0.0.1:6333
export OPENAI_API_KEY=sk-...
```

2. Run the server:

```bash
uv run rag-core serve \
  --host 0.0.0.0 \
  --port 8787 \
  --qdrant-url "$RAG_CORE_QDRANT_URL" \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small
```

3. Ingest, search, and retrieve-context use the same HTTP bodies as Path A. Use absolute paths on the machine where `serve` runs for `POST /v1/ingest` `path`.

## HTTP surface

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness (process up; no dependency checks) |
| `GET` | `/health/ready` | Readiness (`RAGCore` + vector store); `503` when not ready |
| `GET` | `/v1/runtime` | `describe_runtime()` JSON |
| `POST` | `/v1/ingest` | Start ingest job → `202` + `job_id` |
| `GET` | `/v1/ingest/{job_id}` | Job status / result |
| `POST` | `/v1/search` | Ragie-shaped hit list |
| `POST` | `/v1/retrieve-context` | Model context pack + `context_text` |

Request and response shapes match [docs/expectations.md](../expectations.md).

Errors use a stable JSON envelope:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "query, namespace, and corpus_ids are required",
    "details": { "missing_fields": ["corpus_ids"] }
  }
}
```

See [openapi.yaml](openapi.yaml) for codes and schemas. Put auth in front of `serve` per [auth.md](auth.md).

## Troubleshooting

- **`ModuleNotFoundError: starlette`** — install the runtime extra: `uv sync --extra runtime`
- **Qdrant connection errors** — use exactly one of `--qdrant-url` or `--qdrant-location`
- **Embedding auth errors** — use Path A flags, or set `OPENAI_API_KEY` for Path B
- **Ingest job stays `pending`** — check server logs; `path` must exist on the server host

## Next steps

- Embed `RAGCore` in your app for auth, tenancy, and chat ([README](../../README.md#library-usage))
- Map hits to observability with `rag_core.events.export.to_retrieval_hits`
- Library evals: [examples/retrieval_eval.py](../../examples/retrieval_eval.py)
