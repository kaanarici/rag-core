# rag-core

**Own the retrieval layer** — parsing, chunking, hybrid search, reranking, citations, and model-ready context — without renting a black-box RAG API or adopting a full OSS platform.

Your app keeps auth, chat, connectors, and billing. rag-core keeps everything between raw documents and the prompt.

> Product identifiers (`rag_core`, `rag-core`, `RAGCore`) are stable on purpose. To try a different **display name** locally (README/compose only), see [dev/REBRAND.md](dev/REBRAND.md).

## Who this is for

- Teams leaving **managed RAG** (Ragie-class APIs) who still want familiar hit JSON and context packs.
- Teams who rejected **RAGFlow / Haystack / LlamaIndex** as the core dependency but want the same retrieval seriousness.
- Engineers who need **traces and eval hooks**, not a hosted console.

## Who this is not for

- Drop-in hosted ingest + Drive/Notion connectors + SaaS auth (by design).
- “One curl and never think about namespaces again” (you bind tenancy in your app).

## Try it (no API keys)

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync
./scripts/dx_smoke.sh
```

Step-by-step: [docs/quickstart.md](docs/quickstart.md). After `pip install`: `python -m rag_core.quickstart`.

**Your own folder:**

```bash
mkdir -p /tmp/rag-core-quickstart
printf "Invoices can be paid by ACH.\n" > /tmp/rag-core-quickstart/guide.md
uv run rag-core local-search /tmp/rag-core-quickstart "How can invoices be paid?" --json
```

## Self-host (optional)

```bash
docker compose up -d --build && curl -s http://127.0.0.1:8787/health/ready
```

[docs/self-host.md](docs/self-host.md) · [openapi.yaml](docs/self-host/openapi.yaml)

## Embed in your app

One `RAGCore` per worker. Bind `namespace` / `corpus_id` from **your** auth.

```python
import asyncio
from rag_core.demo import build_demo_core

async def main() -> None:
    async with build_demo_core(collection="quickstart") as core:
        await core.ingest_bytes(
            file_bytes=b"Invoices can be paid by card or ACH.",
            filename="billing.txt",
            mime_type="text/plain",
            namespace="tenant:acme",
            corpus_id="help",
        )
        pack = await core.retrieve_context(
            query="How can customers pay?",
            namespace="tenant:acme",
            corpus_ids=["help"],
            limit=5,
        )
        print(pack.as_text())

asyncio.run(main())
```

Persistent local Qdrant: `build_demo_core(..., qdrant_location="./rag-core-qdrant")`. `build_demo_core` works from an installed wheel without API keys. Worker lifespan: [examples/embedded_service.py](examples/embedded_service.py). Contracts: [docs/expectations.md](docs/expectations.md).

## Examples (checkout only)

Not installed into the wheel — import from `rag_core` when using `pip install rag-core`.

```bash
uv run python -m examples.minimal_app
uv run python -m examples.search_endpoint
uv run python -m examples.source_ingest
uv run python -m examples.retrieval_eval
```

| Example | Shows |
|---------|--------|
| [minimal_app.py](examples/minimal_app.py) | Context + citations |
| [embedded_service.py](examples/embedded_service.py) | Worker lifespan |
| [source_ingest.py](examples/source_ingest.py) | File, ZIP, URL ingest |
| [search_endpoint.py](examples/search_endpoint.py) | App-owned scope + tool contract |
| [vercel_ai_sdk_search_tool.ts](examples/vercel_ai_sdk_search_tool.ts) | Vercel AI SDK → your endpoint |
| [retrieval_eval.py](examples/retrieval_eval.py) | `rag_core.evals` quality gate |

**Evals:** keep cases in your app repo (`cases = load_cases(Path("cases.jsonl"))`). See `examples/retrieval_eval.py` and `examples/eval_cases.jsonl`.

**Vercel AI SDK:** `from rag_core.contracts import parse_search_user_documents_request` — model calls your endpoint; your app calls `RAGCore.retrieve_context(...)`.

## Install

```bash
uv add "rag-core @ git+https://github.com/kaanarici/rag-core.git"
```

Extras: `semantic`, `html`, `rerank`, `voyage`, `zeroentropy`, `turbopuffer`, `opentelemetry`, `anthropic`, `langchain`, `openai-agents`, `runtime`. See [docs/providers.md](docs/providers.md).

## CLI cheatsheet

```bash
uv run rag-core demo --json
uv run rag-core doctor --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64 --json
uv run rag-core local-search ./docs "billing policy" --events-jsonl traces.jsonl --json
uv run rag-core ingest ./docs --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333
uv run rag-core search "billing policy" --namespace acme --corpus-id help --json
uv run rag-core retrieve-context "billing policy" --namespace acme --corpus-id help --json
```

Batch ingest commands stream one JSON object per record. `--events-jsonl`, manifest files, and batch ingest stdout are JSONL.

## Docs

| Doc | Topic |
|-----|--------|
| [quickstart.md](docs/quickstart.md) | First-run proof |
| [expectations.md](docs/expectations.md) | Hits, context, traces |
| [providers.md](docs/providers.md) | Vector stores + custom providers |
| [self-host.md](docs/self-host.md) | HTTP API |
| [parsing/formats.md](docs/parsing/formats.md) | Supported formats |
| [DESIGN.md](dev/DESIGN.md) | Architecture principles |

Maintainers: `./scripts/setup_agent_docs.sh` (local agent docs) · [scripts/README.md](scripts/README.md)

## Validate changes

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
./scripts/dx_smoke.sh
./scripts/ci_self_host_smoke.sh
uv build
uv run python scripts/check_dist_artifacts.py
uv run python scripts/wheel_smoke.py
```

## Troubleshooting

- `uv run rag-core demo --json`, then `doctor --json` when config looks wrong.
- Use exactly one of `--qdrant-url` or `--qdrant-location`.
- No-key mode: `--embedding-provider demo --embedding-dimensions 64`.
- Examples: `uv run python -m examples.minimal_app` from a checkout.
