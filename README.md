# rag-core

An embeddable retrieval engine for RAG: ingest documents, search them, and get
back ranked chunks with citations.

## Install

```bash
pip install rag-core   # or: uv add rag-core
```

## Usage

```python
import rag_core

rag = rag_core.index("./docs")
print(rag.ask("How can invoices be paid?"))
```

`index()` embeds the folder locally (FastEmbed, no API key) and returns a handle.
`ask()` returns prompt-ready context with citations; `search()` returns the raw
ranked hits.

From a checkout:

```bash
uv run python -m examples.ask_folder examples/demo_corpus "How can invoices be paid?"
```

More detail: [Quickstart](https://kaanarici.github.io/rag-core/docs/quickstart). Everything below is
optional.

## Command line

```bash
uv run rag-core local-search ./docs "billing policy" --events-jsonl traces.jsonl --json
uv run rag-core local-search examples/demo_corpus "How can invoices be paid?" --demo --json
uv run rag-core local-eval examples/demo_corpus examples/eval_cases.jsonl --json
uv run rag-core search "billing policy" --namespace acme --corpus-id help --qdrant-url http://127.0.0.1:6333 --embedding-provider openai --embedding-model text-embedding-3-small --json
```

`local-search` uses real local semantic retrieval by default. `--demo` swaps in
deterministic embeddings for a no-download smoke and is not semantic retrieval.

## Embedded use

For an application that owns lifecycle, scope, or configuration, use the async
core directly:

```python
from rag_core import RAGCore, RAGCoreConfig

async with RAGCore(RAGCoreConfig.local()) as core:
    await core.ingest_files("./docs", namespace="acme", corpus_id="help")
    context = await core.retrieve_context(
        query="billing policy",
        namespace="acme",
        corpus_ids=["help"],
    )
print(context.as_prompt_text())
```

These calls return prompt-safe context with citation labels. The local
configuration uses FastEmbed and may download the model on first use.

## Configured stores

With a running Qdrant and real embeddings:

```bash
docker compose up -d qdrant
export OPENAI_API_KEY=sk-...

rag-core doctor --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small --json

rag-core ingest ./docs --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small

rag-core search --context "billing policy" --namespace acme --corpus-id help \
  --qdrant-url http://127.0.0.1:6333 \
  --embedding-provider openai --embedding-model text-embedding-3-small
```

Known local and OpenAI models infer their dimensions. Pass
`--embedding-dimensions` only for custom or unknown provider/model pairs.

## Scope

rag-core ingests files, archives, and URLs into an app-owned scope; searches with
capability-aware query plans; returns `SearchResult` hits or a `ContextPack` of
prompt-safe cited context; and emits JSONL traces and eval reports. Auth,
tenancy, connectors, product workflows, and model orchestration stay in the
application.

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
| [ask_folder.py](examples/ask_folder.py) | Index a folder, ask a question |
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
| [Embed](https://kaanarici.github.io/rag-core/docs/embed) | Embedding rag-core in an application |
| [Stability](https://kaanarici.github.io/rag-core/docs/stability) | Public surface and stability |
| [Expectations](https://kaanarici.github.io/rag-core/docs/expectations) | Hits, context, traces, defaults |
| [Providers](https://kaanarici.github.io/rag-core/docs/providers) | Vector stores and providers |
| [Self-host](https://kaanarici.github.io/rag-core/docs/self-host) | Optional HTTP runtime |
| [Formats](https://kaanarici.github.io/rag-core/docs/formats) | Supported file formats |
| [release.md](docs/release.md) | Release readiness checks |

## HTTP server (optional)

```bash
docker compose up -d --build && curl -s http://127.0.0.1:8787/health/ready
```

A thin runtime over `RAGCore`, meant to sit behind your gateway.
[Self-host docs](https://kaanarici.github.io/rag-core/docs/self-host) ([openapi.yaml](docs/self-host/openapi.yaml)).
Server-local ingest paths are limited to the working directory by default; with
`--ingest-root` set, only the configured roots are allowed.

## Install from a checkout

```bash
uv pip install -e .
```

Extras: `semantic`, `html`, `rerank`, `voyage`, `zeroentropy`, `turbopuffer`,
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
- No-key smoke: `--demo` on `rag-core local-search`, or `--embedding-provider demo --embedding-dimensions 64` on configured commands.
- Unknown embedding models: pass `--embedding-dimensions <n>`.
- From a checkout: `uv run python -m examples.ask_folder ./docs "billing policy"`.
