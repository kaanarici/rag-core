# Documentation (product)

Shipped, user-facing docs for **rag-core** embedders and self-hosters.

## Start here

| Need | Doc |
|------|-----|
| First 10 minutes (no API keys) | [quickstart.md](quickstart.md) |
| Search hits, context packs, traces | [expectations.md](expectations.md) |
| Embed in your application | [embedding/production-guide.md](embedding/production-guide.md) |
| Connector / worker pattern | [embedding/connector-pattern.md](embedding/connector-pattern.md) |
| Self-host HTTP API | [self-host/quickstart.md](self-host/quickstart.md) |
| OpenAPI contract | [self-host/openapi.yaml](self-host/openapi.yaml) |
| Auth at the edge | [self-host/auth.md](self-host/auth.md) |
| Compose / env matrix | [self-host/config.md](self-host/config.md) |
| Vector stores | [providers/vector-stores.md](providers/vector-stores.md) |
| Custom providers | [providers/custom-providers.md](providers/custom-providers.md) |
| Provider wire shapes | [providers/provider-output-shapes.md](providers/provider-output-shapes.md) |
| Library evals | [evals/retrieval-quality.md](evals/retrieval-quality.md) |
| Parsing formats | [parsing/formats.md](parsing/formats.md) |
| Vercel AI SDK | [integrations/vercel-ai-sdk-tools.md](integrations/vercel-ai-sdk-tools.md) |

Checklist: [../roadmap.md](../roadmap.md) · Automation: [../scripts/README.md](../scripts/README.md)

## ADRs

| ADR | Topic |
|-----|--------|
| [0001](adr/0001-vendor-neutral-vector-store.md) | VectorStore protocol |
| [0002](adr/0002-linear-pipeline-no-dsl.md) | Linear retrieval pipeline |
| [0003](adr/0003-no-platform-provider-model-lock-in.md) | Provider injection |
| [0004](adr/0004-library-core-with-optional-self-hostable-runtime.md) | Library + `[runtime]` |

## Maintainers & coding agents (local only)

Planning, routing, and agent instructions are **not** on the remote repo. Generate them locally:

```bash
./scripts/setup_agent_docs.sh
```

Templates: [templates/README.md](templates/README.md). Those files are listed in `.gitignore` — **do not `git add` them**.
