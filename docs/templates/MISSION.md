# Mission

**Local only** — generated from [templates/MISSION.md](../templates/MISSION.md). Do not commit.

## Project Direction

First-party retrieval engine for embedded and self-hosted RAG: app-owned ingest, chunking, hybrid search, rerank hooks, traces, evals, and model-ready context.

## Current Maturity

pre-prod · v0 pre-release · Beta classifier in `pyproject.toml`

## Current Focus

Qdrant default wheel; TurboPuffer optional extra; `./scripts/dx_smoke.sh` as trust gate; optional `serve` behind `[runtime]`.

## Scope Boundaries

No hosted app, billing, connector marketplace, or agent canvas. Runtime optional and thin over `RAGCore`.

## Decision Defaults

```bash
./scripts/dx_smoke.sh
uv run ruff check . && uv run mypy src tests examples && uv run pytest -q
```

Agent routing: `docs/plans/ROUTING.md` (local). Product docs: `README.md` + `docs/quickstart.md`.
