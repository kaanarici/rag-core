# rag-core

An embeddable retrieval engine for RAG: ingest documents, search them, and get
back ranked chunks with citations.

## Run locally

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync
uv run python -m examples.ask_folder examples/demo_corpus \
  "How can invoices be paid?"
```

rag-core is not published to PyPI. The commands in this repository run from a
Git checkout.

## Usage

```python
from rag_core import index

with index("./docs") as docs:
    print(docs.context("How can invoices be paid?"))
```

`index()` embeds the folder locally with no API key. `context()` returns
prompt-ready text with citations; `search()` returns structured ranked hits.
The default mode is dense retrieval; lexical and hybrid retrieval remain opt-in.

```bash
uv run python -m examples.ask_folder examples/demo_corpus "How can invoices be paid?"
```

More detail: [Quickstart](https://kaanarici.github.io/rag-core/docs/quickstart). Everything below is
optional.

## Command line

```bash
uv run rag-core search "billing policy" ./docs --trace-jsonl traces.jsonl --json
uv run rag-core search "How can invoices be paid?" examples/demo_corpus --demo --json
uv run rag-core eval examples/demo_corpus examples/eval_cases.jsonl --json
uv run rag-core search "billing policy" --collection help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --json
```

Path search uses real local semantic retrieval by default. `--demo` swaps in
deterministic embeddings for a no-download smoke and is not semantic retrieval.

## Embedded use

For an application that owns lifecycle, scope, or configuration, construct the
engine from provider objects:

```python
import os

from rag_core import Config, Document, Engine, Scope

config = Config.qdrant(
    url="http://127.0.0.1:6333",
    embedding_provider="openai",
    model="text-embedding-3-small",
    embedding_api_key=os.environ["OPENAI_API_KEY"],
)
scope = Scope(tenant_id="acme", collection="help")

async with Engine(config) as engine:
    await engine.ingest(
        Document(
            key="billing.md",
            content=b"Invoices are paid by bank transfer.",
            content_type="text/markdown",
        ),
        scope=scope,
    )
    result = await engine.search("billing policy", scope=scope)
```

## Configured stores

With a running Qdrant and real embeddings:

```bash
docker compose up -d qdrant
export OPENAI_API_KEY=sk-...

rag-core doctor --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --json

rag-core add ./docs --collection help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small

rag-core context "billing policy" --collection help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small
```

Known local and OpenAI models infer their dimensions. Pass
`--embedding-dimensions` only for custom or unknown provider/model pairs.

## Scope

rag-core ingests app-owned documents into an explicit `Scope`; returns ranked
`Evidence`, duplicate diagnostics, equivalent-source provenance, and an honest
answerability state; and emits traces and eval reports. Prompt formatting is an
optional helper. Auth, connectors, product workflows, and model orchestration
stay in the application.

## Migration

| Earlier surface | Common surface |
| --- | --- |
| `Engine.add_bytes(...)` | `Engine.ingest(Document(...), scope=Scope(...))` |
| `Engine.search(query=..., collection=...)` | `Engine.search(query, scope=Scope(...), options=SearchOptions(...))` |
| `SearchResult` | `Evidence` |
| `Engine.context(...)` | `format_evidence(result.evidence)` when prompt text is needed |
| `QueryPlan` for routine search | `SearchOptions(mode="dense" | "lexical" | "hybrid")` |

These advanced methods remain on `Engine` for the CLI, runtime, and integrations.
They are not part of the common path and may change during the beta.

## Examples

```bash
uv run python -m examples.ask_folder examples/demo_corpus "How can invoices be paid?"
uv run python -m examples.minimal_app
uv run python -m examples.embedded_service
uv run python -m examples.configured_retrieval
uv run python -m examples.configured_eval examples/demo_corpus examples/eval_cases.jsonl
uv run python -m examples.source_ingest
uv run python -m examples.retrieval_eval
```

| Example | Shows |
| --- | --- |
| [ask_folder.py](examples/ask_folder.py) | Index a folder, get context |
| [minimal_app.py](examples/minimal_app.py) | Context and citations with demo embeddings |
| [embedded_service.py](examples/embedded_service.py) | Worker lifespan with demo embeddings |
| [configured_retrieval.py](examples/configured_retrieval.py) | Real embeddings with Qdrant |
| [configured_eval.py](examples/configured_eval.py) | Eval with real embeddings |
| [source_ingest.py](examples/source_ingest.py) | File, ZIP, and URL ingest |
| [search_endpoint.py](examples/search_endpoint.py) | App-owned scope and tool contract |
| [retrieval_eval.py](examples/retrieval_eval.py) | Eval cases, wiring only |

## Documentation

| Doc | Topic |
| --- | --- |
| [Quickstart](https://kaanarici.github.io/rag-core/docs/quickstart) | Index a folder and search it |
| [Agent integration](https://kaanarici.github.io/rag-core/docs/agent-integration) | Recipes for coding agents |
| [Eval quality](https://kaanarici.github.io/rag-core/docs/eval-quality) | Measuring retrieval quality |
| [Python API](https://kaanarici.github.io/rag-core/docs/api-python) | Embedding rag-core in an application |
| [Stability](https://kaanarici.github.io/rag-core/docs/stability) | Public surface and stability |
| [Traces](https://kaanarici.github.io/rag-core/docs/traces) | Hits, context, traces, defaults |
| [Providers](https://kaanarici.github.io/rag-core/docs/providers) | Vector stores and providers |
| [Self-host](https://kaanarici.github.io/rag-core/docs/self-host) | Optional HTTP runtime |
| [Formats](https://kaanarici.github.io/rag-core/docs/formats) | Supported file formats |
| [release.md](docs/release.md) | Release readiness checks |

## HTTP server (optional)

```bash
docker compose up -d --build && curl -s http://127.0.0.1:8787/health/ready
```

A thin runtime over `Engine`, meant to sit behind your gateway.
[Self-host docs](https://kaanarici.github.io/rag-core/docs/self-host) ([openapi.yaml](docs/self-host/openapi.yaml)).
Server-local ingest paths are limited to the working directory by default; with
`--ingest-root` set, only the configured roots are allowed.

## Install from Git

```bash
uv venv
uv pip install "rag-core @ git+https://github.com/kaanarici/rag-core.git@main"
```

For development, clone the repository and run `uv sync --group dev`. Optional
extras include `semantic`, `html`, `pdf`, `rerank`, `voyage`,
`zeroentropy`, `turbopuffer`, `opentelemetry`, `anthropic`, `langchain`,
`openai-agents`, `mcp`, and `runtime`.

## Development

Run `pre-commit install` once after cloning for the local commit and pre-push
hooks (see [scripts/README.md](scripts/README.md#hooks)).

Fast iteration, then the full release check:

```bash
./scripts/landing_check.sh --quick
./scripts/landing_check.sh
```

Package proof after pushing:

```bash
./scripts/public_checkout_smoke.sh --package
./scripts/github_install_smoke.sh https://github.com/kaanarici/rag-core.git main
```

Full gate:

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
./scripts/dx_smoke.sh
./scripts/verify_vercel_ai_sdk_example.sh
./scripts/ci_self_host_smoke.sh
./scripts/verify_optional_integrations.sh
uv build
uv run python scripts/check_dist_artifacts.py
uv run python scripts/wheel_smoke.py
```

## Troubleshooting

- `rag-core doctor` when configuration looks wrong; add `--json` for scripts.
- Pass exactly one of `--qdrant-url` or `--qdrant-location`.
- No-key smoke: `--demo` on `rag-core search "<query>" <path>`, or `--embedding-provider demo --embedding-dimensions 64` on configured commands.
- Unknown embedding models: pass `--embedding-dimensions <n>`.
- From a checkout: `uv run python -m examples.ask_folder ./docs "billing policy"`.
