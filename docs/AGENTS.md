# Agent instructions (tracked)

Cursor/Codex may also load a gitignored root `AGENTS.md`; **this file is the canonical copy in git.**

## Read first

1. [CONTEXT.md](CONTEXT.md) — mission, maturity, journey status, code map  
2. [plans/ROUTING.md](plans/ROUTING.md) — shape gate, routing, hard stops  
3. [README.md](README.md) — full doc catalog  
4. [../scripts/README.md](../scripts/README.md) — automation only  
5. [../roadmap.md](../roadmap.md) — open checklist items  

Use skills from `.codex/skills/` when planning or executing substantial work: `decide`, `repo-intake`, `blueprint`, `grind`, `review`, `finish-line`.

## Product lock

Library-first retrieval engine. Optional `[runtime]` for `serve`. Do not expand into a hosted platform.

For journey shapes and packets, see [plans/one-repo-retrieval-engine-strategy.md](plans/one-repo-retrieval-engine-strategy.md). For **what to do next**, [plans/ROUTING.md](plans/ROUTING.md) wins over the long strategy body.

## Commands (mirror CI)

```bash
uv sync --group dev
./scripts/dx_smoke.sh
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
```

## Coding standards

- Python 3.11+; type hints on touched signatures  
- Do not expand `core.py` without splitting  
- Single owner for aggregator `__init__.py` files under `rag_core`  
- Substantial behavior → targeted tests  
- Verify doc claims against code/tests  

## Work slicing

One journey, one shape, one gate. Run `decide` when choosing between architecture options; do not ask the user for low-level forks the repo can answer.
