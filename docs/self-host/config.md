# Self-host configuration

`rag-core serve` reads the same flags as the CLI library commands. Environment variables are the stable contract for Compose and process managers.

## Environment ↔ CLI matrix

| Environment variable | CLI flag | Default | Notes |
|---------------------|----------|---------|-------|
| `RAG_CORE_QDRANT_URL` | `--qdrant-url` | — | Use **one** of URL or location |
| `RAG_CORE_QDRANT_LOCATION` | `--qdrant-location` | — | `:memory:` for no-key smoke |
| `RAG_CORE_QDRANT_API_KEY` | `--qdrant-api-key` | — | Cloud Qdrant |
| `RAG_CORE_QDRANT_COLLECTION` | `--qdrant-collection` | `rag_core_chunks` | Base collection name |
| `RAG_CORE_EMBEDDING_PROVIDER` | `--embedding-provider` | `openai` | `demo` for no-key |
| `RAG_CORE_EMBEDDING_MODEL` | `--embedding-model` | `text-embedding-3-large` | |
| `RAG_CORE_EMBEDDING_DIMENSIONS` | `--embedding-dimensions` | — | Required for `demo` |
| `OPENAI_API_KEY` | — | — | When embedding provider is OpenAI |
| `RAG_CORE_RERANKER_PROVIDER` | `--reranker-provider` | `none` | Optional |
| `RAG_CORE_RERANKER_MODEL` | `--reranker-model` | — | Provider-specific |
| `RAG_CORE_PROCESSING_VERSION` | `--processing-version` | `rag_core_processing_v1` | Reindex trigger |

Serve-specific flags (no env mirror in v1): `--host`, `--port`.

## Compose defaults

`docker compose up` starts **Qdrant** and **serve** with demo embeddings pointed at `http://qdrant:6333`. Copy `.env.example` to `.env` when switching to OpenAI embeddings.

## Health endpoints

| Path | Use |
|------|-----|
| `GET /health` | Liveness — process up |
| `GET /health/ready` | Readiness — `RAGCore` + vector store |

Orchestrators should use `/health` for liveness and `/health/ready` for routing traffic.

## Job database

Ingest jobs persist to `.rag-core/runtime/jobs.sqlite3` under the server working directory unless overridden in a custom deployment. Mount a volume if jobs must survive container restarts.

See [quickstart.md](quickstart.md), [openapi.yaml](openapi.yaml), and [auth.md](auth.md).
