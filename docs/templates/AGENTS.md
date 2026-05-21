# Agent instructions (local)

**Local only** — generated from [templates/AGENTS.md](../templates/AGENTS.md). Do not commit.

## Read first

1. [CONTEXT.md](CONTEXT.md)  
2. [plans/ROUTING.md](plans/ROUTING.md)  
3. [README.md](README.md) — product doc catalog  
4. [../scripts/README.md](../scripts/README.md)  
5. [../roadmap.md](../roadmap.md)  

Use `.codex/skills/` for substantial work: `decide`, `repo-intake`, `blueprint`, `grind`, `review`, `finish-line`.

## Product lock

Library-first retrieval engine. Optional `[runtime]` for `serve`. No hosted platform sprawl.

## Commands

```bash
uv sync --group dev
./scripts/dx_smoke.sh
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
```

## Work slicing

One journey, one shape, one gate. Use `decide` before architectural forks.
