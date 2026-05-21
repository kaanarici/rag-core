# Scripts

Maintainer and CI entrypoints. Prefer these over inventing one-off shell in chat.

**Agent docs (local, gitignored):** `./scripts/setup_agent_docs.sh` · [docs/templates/README.md](../docs/templates/README.md)

## Daily (you)

| Script | Purpose |
|--------|---------|
| [`dx_smoke.sh`](dx_smoke.sh) | **Journey A** — demo, `local-search`, trace, `doctor`, context, library eval. No API keys. |
| [`self_host_smoke.sh`](self_host_smoke.sh) | **Journey C** — HTTP against `docker compose` (`/health/ready`, ingest job, search). Run after `docker compose up -d --build`. |

```bash
uv sync --group dev
./scripts/dx_smoke.sh
```

## CI (GitHub Actions mirrors this)

| Script | CI step | Purpose |
|--------|---------|---------|
| `dx_smoke.sh` | Journey A (Python 3.12 only) | End-user path regression |
| `ci_self_host_smoke.sh` | Journey C (Python 3.12 only) | Starts `rag-core serve`, then runs HTTP ingest/search/context smoke |
| `architecture_pressure.py` | Architecture pressure | Large files, boundary warnings, mypy ignore inventory (`--json`) |
| `pytest` | Test | Not under `scripts/` |
| `check_dist_artifacts.py` | Check built artifacts | Wheel/sdist required paths after `uv build` |
| `wheel_smoke.py` | Wheel smoke test | Install wheel in fresh venv, run consumer + `rag_core.quickstart` |

```bash
uv run ruff check . && uv run mypy src tests examples && uv run pytest -q
./scripts/dx_smoke.sh
./scripts/ci_self_host_smoke.sh
uv build
uv run python scripts/check_dist_artifacts.py
uv run python scripts/wheel_smoke.py
```

## Packaging / structure

| Script | When |
|--------|------|
| [`check_dist_artifacts.py`](check_dist_artifacts.py) | After changing `MANIFEST.in`, packaged docs, or `pyproject` package data |
| [`wheel_smoke.py`](wheel_smoke.py) | After packaging or public export surface changes |
| [`architecture_pressure.py`](architecture_pressure.py) | Optional local debloat report; read the JSON, but do not treat it as a gate unless a test asserts bounds |

## Local-only (not CI)

| Script | Purpose |
|--------|---------|
| [`brand_check.sh`](brand_check.sh) | README title matches `dev/project_identity.toml` |
| [`local_rebrand.sh`](local_rebrand.sh) | Rewrite display name in markdown/compose only — never `src/` or `pyproject.toml` |

See [dev/REBRAND.md](../dev/REBRAND.md).

## Rules for agents

- Do **not** add scripts without listing them here and wiring CI when the gate matters.
- Do **not** duplicate `dx_smoke` steps in new scripts — extend `dx_smoke.sh` or call it.
- Prefer `uv run rag-core …` for product behavior; scripts are **orchestration** only.
- Before claiming validation strength, read [tests/README.md](../tests/README.md).
