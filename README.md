# rag-core

Embeddable retrieval for Python applications: parse documents, chunk, index into a
vector store, run capability-aware search, optionally rerank, and return prompt-safe context with
citations and traces.

**v0 beta** — library-first. CI proves wiring and retrieval regression on fixed
fixtures; see [tests/README.md](tests/README.md) for what passing tests mean.

Your application keeps authentication, chat, connectors, and billing. rag-core keeps
everything from raw bytes through prompt-safe context assembly. Optional
`rag-core serve` exposes the same `RAGCore` over HTTP.

## What you get

- **Ingest** — files, archives, URLs (CLI helpers exist; sync logic stays in your app)
- **Search** — capability-aware query plans, hybrid when supported, metadata filters
- **Context** — `retrieve_context` packs prompt-safe text + citations for prompt assembly
- **Traces** — JSONL events and summaries for debugging ranking stages
- **Evals** — `rag_core.evals` plus `local-eval` for no-key folder checks

## Smoke (no API keys)

Prove the engine runs with deterministic demo embeddings — not semantic search.

Python 3.11+ and [uv](https://docs.astral.sh/uv/), from the repo root:

```bash
uv sync
./scripts/dx_smoke.sh
```

| Command | Purpose |
| --- | --- |
| `./scripts/dx_smoke.sh` | End-to-end smoke (CI uses this) |
| `uv run rag-core demo --json` | Shortest hit JSON |
| `python -m rag_core.quickstart` | Installed package smoke (editable install or built wheel) |

Step-by-step: [docs/quickstart.md](docs/quickstart.md).

**Folder smoke:**

```bash
uv run rag-core local-search examples/demo_corpus "How can invoices be paid?" --json
uv run rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl \
  --min-recall-at-5 1 --min-mrr 1 --json
```

## Configured (your stack)

Use real embeddings and Qdrant (local or remote). Bind `namespace` and `corpus_id`
from your auth.

```bash
docker compose up -d qdrant   # optional; see compose.yaml
export OPENAI_API_KEY=sk-...
uv run rag-core doctor --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536 --json
uv run rag-core ingest ./docs --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
uv run rag-core retrieve-context "billing policy" --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
```

Embed guide: [docs/embed.md](docs/embed.md). Providers: [docs/providers.md](docs/providers.md).

## Self-host HTTP (optional)

```bash
docker compose up -d --build && curl -s http://127.0.0.1:8787/health/ready
```

Thin runtime over `RAGCore` — no auth or connector sync in core. [docs/self-host.md](docs/self-host.md) · [openapi.yaml](docs/self-host/openapi.yaml)

Server-local ingest paths are limited to the working directory by default. When
`--ingest-root` is set, only the configured roots are allowed.

## Install From Checkout

```bash
uv pip install -e .
```

Extras: `semantic`, `html`, `rerank`, `voyage`, `zeroentropy`, `turbopuffer`,
`opentelemetry`, `anthropic`, `langchain`, `openai-agents`, `runtime`.

## Examples (checkout only)

```bash
uv run python -m examples.minimal_app
uv run python -m examples.embedded_service
uv run python -m examples.configured_retrieval   # needs OPENAI_API_KEY
uv run python -m examples.source_ingest
uv run python -m examples.retrieval_eval
```

| Example | Shows |
| --- | --- |
| [minimal_app.py](examples/minimal_app.py) | Context + citations (demo) |
| [embedded_service.py](examples/embedded_service.py) | Worker lifespan (demo) |
| [configured_retrieval.py](examples/configured_retrieval.py) | Real embeddings + Qdrant |
| [source_ingest.py](examples/source_ingest.py) | File, ZIP, URL ingest |
| [search_endpoint.py](examples/search_endpoint.py) | App-owned scope + tool contract |
| [retrieval_eval.py](examples/retrieval_eval.py) | App-owned eval cases |

## CLI cheatsheet

```bash
uv run rag-core doctor --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64 --json
uv run rag-core local-search ./docs "billing policy" --events-jsonl traces.jsonl --json
uv run rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --json
uv run rag-core ingest ./docs --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
uv run rag-core search "billing policy" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536 --json
uv run rag-core retrieve-context "billing policy" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --embedding-dimensions 1536
```

## Docs

| Doc | Topic |
| --- | --- |
| [quickstart.md](docs/quickstart.md) | Smoke vs configured walkthrough |
| [embed.md](docs/embed.md) | Embed in your application |
| [stability.md](docs/stability.md) | Public surface stability |
| [expectations.md](docs/expectations.md) | Hits, context, traces, defaults |
| [providers.md](docs/providers.md) | Vector stores and providers |
| [self-host.md](docs/self-host.md) | Optional HTTP runtime |
| [parsing/formats.md](docs/parsing/formats.md) | Supported formats |

## Validate changes

Fast iteration:

```bash
./scripts/landing_check.sh --quick
```

Full release short form:

```bash
./scripts/landing_check.sh
```

Expanded release gate:

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
./scripts/dx_smoke.sh
./scripts/verify_vercel_ai_sdk_example.sh
./scripts/ci_self_host_smoke.sh
uv build
uv run python scripts/check_dist_artifacts.py
uv run python scripts/wheel_smoke.py
```

## Troubleshooting

- `uv run rag-core doctor` when config looks wrong; add `--json` for scripts.
- Use exactly one of `--qdrant-url` or `--qdrant-location`.
- No-key smoke: `--embedding-provider demo --embedding-dimensions 64`.
- Examples: `uv run python -m examples.minimal_app` from a checkout.
