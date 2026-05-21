# Documentation

## Reading layers

| Layer | File | When |
|-------|------|------|
| 1 — Context | [CONTEXT.md](CONTEXT.md) | Every agent session: mission, status, code map |
| 2 — Routing | [plans/ROUTING.md](plans/ROUTING.md) | Before any substantial edit: journey + shape gate |
| 3 — Catalog | This file | Lookup by topic |
| 4 — Deep spec | [plans/one-repo-retrieval-engine-strategy.md](plans/one-repo-retrieval-engine-strategy.md) | Full thesis, packets, history |
| 5 — Checklist | [../roadmap.md](../roadmap.md) | Open vs done items |

**Agents:** [AGENTS.md](AGENTS.md) · **Scripts:** [../scripts/README.md](../scripts/README.md)

## Maintainer loop

```bash
./scripts/dx_smoke.sh
uv run rag-core doctor --json
```

## Topic index

| Need | Doc |
|------|-----|
| First 10 minutes, no keys | [quickstart.md](quickstart.md) |
| Hit JSON, context packs, traces | [expectations.md](expectations.md) |
| Embed `RAGCore` in an app | [embedding/production-guide.md](embedding/production-guide.md) |
| Connector / worker pattern | [embedding/connector-pattern.md](embedding/connector-pattern.md) |
| Self-host HTTP API | [self-host/quickstart.md](self-host/quickstart.md) · [openapi.yaml](self-host/openapi.yaml) · [auth.md](self-host/auth.md) · [config.md](self-host/config.md) |
| Vector stores | [providers/vector-stores.md](providers/vector-stores.md) |
| Custom providers | [providers/custom-providers.md](providers/custom-providers.md) |
| Provider wire shapes | [providers/provider-output-shapes.md](providers/provider-output-shapes.md) |
| Evals (library only) | [evals/retrieval-quality.md](evals/retrieval-quality.md) |
| Parsing formats | [parsing/formats.md](parsing/formats.md) |
| Vercel AI SDK | [integrations/vercel-ai-sdk-tools.md](integrations/vercel-ai-sdk-tools.md) |

## Research (comparison only)

[research/README.md](research/README.md)

## ADRs

| ADR | Topic |
|-----|--------|
| [0001](adr/0001-vendor-neutral-vector-store.md) | VectorStore; Qdrant default, TurboPuffer optional |
| [0002](adr/0002-linear-pipeline-no-dsl.md) | Linear pipeline |
| [0003](adr/0003-no-platform-provider-model-lock-in.md) | Provider injection |
| [0004](adr/0004-library-core-with-optional-self-hostable-runtime.md) | Library + `[runtime]` |

## Archive

[plans/archive/](plans/archive/) — historical; do not execute.

## Hygiene

1. Claims must match code and `tests/`.  
2. One active routing file: `plans/ROUTING.md`.  
3. Smoke steps live in `scripts/`; quickstart explains output.  
4. v0 pre-release wording; no “v1.1” product labels.
