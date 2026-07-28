# rag-core

An embeddable retrieval engine for RAG: ingest documents, search them, and get
back ranked chunks with citations.

## Install

```bash
pip install rag-core   # or: uv add rag-core
pip install "rag-core[openai-agents]"   # for the Agent example below
```

## Usage

```python
from agents import Agent, Runner
from rag_core import Config, Document, RAGCore

async with RAGCore(
    Config.local(),
    tenant_id="acme",
    index="company-docs",
) as rag:
    await rag.ingest(
        Document(
            key="billing.md",
            content=b"Invoices can be paid by card or ACH.",
            content_type="text/markdown",
        )
    )
    agent = Agent(name="support", tools=[rag.tool()])
    result = await Runner.run(agent, "How can invoices be paid?")
```

Tenant and index scope are bound when `RAGCore` is created. The model cannot
change them. Retrieval tools return prompt-safe context with citation labels;
`search()` returns structured evidence with source identity and locators.

`Config.local()` is for development and tests. Production configuration is
covered in [Configured stores](#configured-stores).

For a process configured through environment variables:

```python
from agents import Agent
from rag_core import RAGCore

rag = RAGCore.from_env(index="company-docs")
agent = Agent(name="support", tools=[rag.tool()])
```

At minimum, set `RAG_CORE_TENANT_ID`, one store target, and the credentials for
the selected embedding provider:

```bash
export RAG_CORE_TENANT_ID=acme
export RAG_CORE_QDRANT_URL=https://qdrant.example.com
export RAG_CORE_QDRANT_API_KEY=...
export OPENAI_API_KEY=...
```

The default index and query path is dense-only. Sparse indexes and hybrid
retrieval are explicit advanced configuration.

## Local folder search

```python
from rag_core.easy import index

with index("./docs") as idx:
    print(idx.context("How can invoices be paid?"))
```

From a checkout:

```bash
uv run python -m examples.ask_folder examples/demo_corpus "How can invoices be paid?"
```

More detail: [Quickstart](https://kaanarici.github.io/rag-core/docs/quickstart).

## Embedded use

For an application that already assembles provider objects, wrap the advanced
engine without exposing it to the agent:

```python
from rag_core import RAGCore
from rag_core.core import Engine

engine = Engine(config, embedding_provider=embedding, vector_store=store)
rag = RAGCore(engine, tenant_id="acme", index="company-docs")
```

`Engine` is the advanced integration surface.

## Command line

```bash
uv run rag-core search "billing policy" ./docs --trace-jsonl traces.jsonl --json
uv run rag-core search "How can invoices be paid?" examples/demo_corpus --demo --json
uv run rag-core eval examples/demo_corpus examples/eval_cases.jsonl --json
uv run rag-core search "billing policy" --collection help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --json
```

Path search uses real local semantic retrieval by default. `--demo` swaps in
deterministic embeddings for a no-download smoke and is not semantic retrieval.

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

`RAGCore` ingests, searches, and deletes documents inside an application-owned
tenant and index. It returns stable `Evidence` objects and exposes the same
scope as an agent tool. Dense retrieval is the default. Advanced engine APIs
provide archives, URLs, query plans, traces, and eval hooks without expanding
the common facade. Auth, connectors, product workflows, and model orchestration
stay in the application.

## Examples

```bash
uv run python -m examples.ask_folder examples/demo_corpus "How can invoices be paid?"
uv run python -m examples.minimal_app
uv run python -m examples.embedded_service
uv run python -m examples.configured_retrieval
uv run python -m examples.configured_eval examples/demo_corpus examples/eval_cases.jsonl
uv run python -m examples.openai_agents "How can invoices be paid?"
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
| [openai_agents.py](examples/openai_agents.py) | Production-scoped OpenAI Agents tool |
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

## Install from a checkout

```bash
uv pip install -e .
```

Extras: `semantic`, `html`, `pdf`, `rerank`, `voyage`, `zeroentropy`, `turbopuffer`,
`opentelemetry`, `anthropic`, `langchain`, `openai-agents`, `mcp`, `runtime`.

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
