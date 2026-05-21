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

## Maintainers (us)

- **Product docs:** [docs/README.md](docs/README.md)
- **Agent docs (local, gitignored):** `./scripts/setup_agent_docs.sh` once per machine — [docs/templates/README.md](docs/templates/README.md)
- **Scripts:** [scripts/README.md](scripts/README.md)

```bash
uv sync --group dev
./scripts/dx_smoke.sh
```

Full CI parity: [scripts/README.md](scripts/README.md#ci-github-actions-mirrors-this). Config truth: `uv run rag-core doctor --json`.

## Try it in ten minutes (no API keys)

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync
./scripts/dx_smoke.sh
```

That single script runs demo search, folder ingest, trace summary, doctor, context + citations, and a small library eval. Step-by-step output expectations: [docs/quickstart.md](docs/quickstart.md).

**After `pip install`** (no git checkout):

```bash
pip install rag-core
python -m rag_core.quickstart
```

**Your own folder** (still no keys): `local-search` indexes a folder into embedded Qdrant (use `--max-files` to cap batch size).

```bash
mkdir -p /tmp/rag-core-quickstart
printf "Invoices can be paid by ACH.\n" > /tmp/rag-core-quickstart/guide.md
uv run rag-core local-search /tmp/rag-core-quickstart "How can invoices be paid?" --json
```

Trace on the bundled demo corpus (see [docs/quickstart.md](docs/quickstart.md)):

```bash
uv run rag-core local-search examples/demo_corpus "corpus lifecycle" \
  --events-jsonl traces.jsonl --json
```

## Self-host the HTTP API (optional)

```bash
docker compose up -d --build
curl -s http://127.0.0.1:8787/health/ready
```

Full path: [docs/self-host/quickstart.md](docs/self-host/quickstart.md). Contract: [docs/self-host/openapi.yaml](docs/self-host/openapi.yaml). Auth stays in your gateway: [docs/self-host/auth.md](docs/self-host/auth.md).

## Embed in your app

One `RAGCore` per worker. Bind `namespace` / `corpus_id` from **your** auth — not from model text.

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

For a persistent local directory without API keys:

```python
async with build_demo_core(
    collection="quickstart",
    qdrant_location="./rag-core-qdrant",
) as core:
    ...
```

`build_demo_core` works from an installed wheel without API keys.

Production lifecycle, shutdown, and connector replacement: [docs/embedding/production-guide.md](docs/embedding/production-guide.md) and [docs/embedding/connector-pattern.md](docs/embedding/connector-pattern.md). Example worker pattern: [examples/embedded_service.py](examples/embedded_service.py).

## How the pieces fit together

```text
sources → parse → chunk → index → search → context pack → your LLM call
```

| Term | Meaning |
|------|---------|
| **namespace** | Tenant or app partition you control |
| **corpus_id** | Logical collection inside a namespace |
| **search** | Raw ranked hits (Ragie-shaped JSON) |
| **retrieve_context** | Model-ready text + citations |
| **events / traces** | JSONL you can summarize without a vendor UI |

Deeper contract: [docs/expectations.md](docs/expectations.md).

## CLI cheatsheet

```bash
uv run rag-core demo --json
uv run rag-core doctor --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64 --json
uv run rag-core local-search ./docs "billing policy" --events-jsonl traces.jsonl --json
uv run rag-core ingest ./docs --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333
uv run rag-core search "billing policy" --namespace acme --corpus-id help --json
uv run rag-core retrieve-context "billing policy" --namespace acme --corpus-id help --json
```

`local-search` is the fastest end-to-end check. `ingest` + `search` + `retrieve-context` are what you keep in production. Every command has `--help` with copy-paste Examples.

## Install

```bash
uv add "rag-core @ git+https://github.com/kaanarici/rag-core.git"
```

Optional extras (declare on install): `semantic`, `html`, `rerank`, `voyage`, `zeroentropy`, `turbopuffer`, `opentelemetry`, `anthropic`, `langchain`, `openai-agents`, `runtime`. Example: `uv sync --extra runtime --extra rerank`.

## Examples (checkout only)

The `examples/` modules below are checkout examples. They are not installed into the wheel. Use `python -m rag_core.quickstart` for a wheel-only demo.

```bash
uv run python -m examples.minimal_app
uv run python -m examples.search_endpoint
uv run python -m examples.source_ingest
uv run python -m examples.retrieval_eval
```

| Example | Shows |
|---------|--------|
| [examples/minimal_app.py](examples/minimal_app.py) | Context + citations |
| [examples/embedded_service.py](examples/embedded_service.py) | Worker lifespan |
| [examples/source_ingest.py](examples/source_ingest.py) | File, ZIP, URL ingest |
| [examples/retrieval_eval.py](examples/retrieval_eval.py) | Library eval gate |

## CLI output shapes

Commands with `--json` write one JSON document to stdout. Batch ingest commands such as `ingest`, `ingest-archive`, and `ingest-urls` stream one JSON object per record. `--events-jsonl`, manifest files, and batch ingest stdout are JSONL: one JSON object per line.

## Docs map

See [docs/README.md](docs/README.md). Maintainer agent layers: `./scripts/setup_agent_docs.sh` (gitignored; not on remote).

## Validate changes

```bash
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
./scripts/dx_smoke.sh
uv build && uv run python scripts/wheel_smoke.py
./scripts/brand_check.sh
```

## Troubleshooting

- Start with `uv run rag-core demo --json`, then `doctor --json` when config looks wrong.
- Use exactly one of `--qdrant-url` or `--qdrant-location`.
- For no-key mode: `--embedding-provider demo --embedding-dimensions 64`.
- Examples must run as modules from a checkout: `uv run python -m examples.minimal_app`.

Strategy and scope boundaries: [AGENTS.md](AGENTS.md) (if present in your tree) and the strategy doc above.
