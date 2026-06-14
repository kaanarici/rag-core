# Scripts

Maintainer and CI entrypoints. Prefer these over one-off command sequences.

## Daily (you)

| Script | Purpose |
|--------|---------|
| [`landing_check.sh`](landing_check.sh) | Local validation wrapper: `--quick` for iteration, default for the full release-gate commands below. |
| [`public_checkout_smoke.sh`](public_checkout_smoke.sh) | Clone the public repo into a temp directory and prove quick/package validation without local-only files. |
| [`github_install_smoke.sh`](github_install_smoke.sh) | Install from a Git remote/ref in a fresh venv and run installed quickstart/demo. |
| [`dx_smoke.sh`](dx_smoke.sh) | **Journey A** - demo, `local-search --demo`, trace, `doctor`, context, library eval, `local-eval`. No API keys. |
| [`self_host_smoke.sh`](self_host_smoke.sh) | **Journey C**: HTTP probe for an already-running `rag-core serve` (`/health/ready`, ingest job, search, context retrieval). |
| [`verify_vercel_ai_sdk_example.sh`](verify_vercel_ai_sdk_example.sh) | **Journey B**: install current AI SDK v6 types in a temp project and type-check the copyable Vercel example. |
| [`verify_optional_integrations.sh`](verify_optional_integrations.sh) | Install optional LangChain, OpenAI Agents, and MCP extras, run adapter tests against real packages, and smoke exported helper construction. |
| [`worktree_slices.py`](worktree_slices.py) | Local reviewability report for grouping the dirty tree into landing slices. |

```bash
uv sync --group dev
./scripts/landing_check.sh --quick
./scripts/public_checkout_smoke.sh --quick
./scripts/github_install_smoke.sh https://github.com/kaanarici/rag-core.git main
uv run python scripts/worktree_slices.py --staged
uv run python scripts/worktree_slices.py --staged --fail-on-uncategorized
uv run python scripts/worktree_slices.py --staged --slice validation-tooling
```

### Hooks

Install once after cloning; commit hooks run ruff+mypy on staged files, and the pre-push hook runs the fast pytest tier. The full gate remains `./scripts/landing_check.sh`.

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

## Launch gates and CI mirror

`landing_check.sh` is a local coordination wrapper over existing validation
surfaces. Use `./scripts/landing_check.sh --quick` while iterating; it runs
sync, lint, typecheck, the fast pytest tier, and `dx_smoke`. Run
`./scripts/landing_check.sh` before treating the tree as release-landable.
The scripts are **coordination** only; they do not implement resume-after-crash recovery.
CI runs the full validation surface directly, with pytest split into marker
tiers for clearer failures.

The `release-artifacts` GitHub workflow is a non-publishing artifact build for
manual runs and `v*` tags. It runs the quick gate, the optional-integrations
smoke, builds the sdist/wheel, checks artifact contents, runs the wheel smoke,
and uploads `dist/*`.

```bash
./scripts/landing_check.sh --quick
```

Full local release wrapper:

```bash
./scripts/landing_check.sh
```

Fresh public checkout proof after pushing:

```bash
./scripts/public_checkout_smoke.sh --package
./scripts/github_install_smoke.sh https://github.com/kaanarici/rag-core.git main
```

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

| Script | CI step | Purpose |
|--------|---------|---------|
| `landing_check.sh --quick` | Local only | Runs the iteration gate: sync, lint, typecheck, fast pytest, `dx_smoke` |
| `landing_check.sh` | Local only | Runs the full release-gate command list in order |
| `public_checkout_smoke.sh --package` | Local only | Clones the configured remote/ref, then runs quick gate, build, artifact check, and wheel smoke |
| `github_install_smoke.sh` | Local only | Installs from a Git remote/ref into a fresh venv and runs installed package smoke |
| `dx_smoke.sh` | Journey A (Python 3.12 only) | End-user path regression |
| `verify_vercel_ai_sdk_example.sh` | Journey B (Python 3.12 only) | Copyable Vercel AI SDK v6 example typecheck against current declarations |
| `ci_self_host_smoke.sh` | Journey C (Python 3.12 only) | Starts `rag-core serve`, then runs HTTP ingest/search/context smoke |
| `verify_optional_integrations.sh` | Optional integrations (Python 3.12 only) | Installs `[langchain]`, `[openai-agents]`, and `[mcp]`, runs real-package adapter tests, and constructs exported helper tools |
| `architecture_pressure.py` | Architecture pressure | Large files, boundary warnings, mypy ignore inventory (`--json`) |
| `validate_provider_fixtures.sh` | Local only | Run `provider_contract` tests against checked-in JSON fixtures |
| `pytest` (tiered) | Test | See CI table below |
| `check_dist_artifacts.py` | Check built artifacts | Wheel/sdist required paths after `uv build` |
| `wheel_smoke.py` | Wheel smoke test | Install wheel in fresh venv, run consumer, installed CLI smoke, installed `[runtime]` smoke, integration import checks, and `rag_core.quickstart` |
| `worktree_slices.py` | Local only | Groups current git changes into reviewable landing slices with focused validation suggestions |

| Gate | Command |
| --- | --- |
| Fast | `uv run pytest -q -m "not live and not eval and not eval_harness and not provider_contract and not integration"` |
| Provider replay | `uv run pytest -q -m provider_contract` |
| Integration | `uv run pytest -q -m integration` |
| PR retrieval eval | `uv run pytest -q tests/evals/test_retrieval_eval_pr.py` |

```bash
uv run ruff check . && uv run mypy src tests examples
uv run pytest -q -m "not live and not eval and not eval_harness and not provider_contract and not integration"
uv run pytest -q -m provider_contract
uv run pytest -q -m integration
uv run pytest -q tests/evals/test_retrieval_eval_pr.py
./scripts/dx_smoke.sh
./scripts/verify_vercel_ai_sdk_example.sh
./scripts/ci_self_host_smoke.sh
./scripts/verify_optional_integrations.sh
uv build
uv run python scripts/check_dist_artifacts.py
uv run python scripts/wheel_smoke.py
```

## Packaging / structure

| Script | When |
|--------|------|
| [`landing_check.sh`](landing_check.sh) | Before asking someone else to review or treat the current tree as v0 beta landable |
| [`public_checkout_smoke.sh`](public_checkout_smoke.sh) | After pushing or changing packaging docs, to prove a fresh public checkout works without local-only files |
| [`github_install_smoke.sh`](github_install_smoke.sh) | After pushing, to prove direct Git installation works from the public repo/ref |
| [`check_dist_artifacts.py`](check_dist_artifacts.py) | After changing `MANIFEST.in`, packaged docs, or `pyproject` package data |
| [`wheel_smoke.py`](wheel_smoke.py) | After packaging or public surface changes |
| [`wheel_extra_smoke.py`](wheel_extra_smoke.py) | After changing `pyproject` extras or optional adapter surfaces |
| [`verify_vercel_ai_sdk_example.sh`](verify_vercel_ai_sdk_example.sh) | After changing `examples/vercel_ai_sdk_search_tool.ts` or its documented AI SDK contract |
| [`architecture_pressure.py`](architecture_pressure.py) | Optional local debloat report; read the JSON, but do not treat it as a gate unless a test asserts bounds |
| [`worktree_slices.py`](worktree_slices.py) | When the dirty tree is too large to review as one undifferentiated change set |

Useful variants:

```bash
uv run python scripts/worktree_slices.py --list-slices
uv run python scripts/worktree_slices.py --staged --fail-on-uncategorized
uv run python scripts/worktree_slices.py --staged --slice validation-tooling
uv run python scripts/worktree_slices.py --staged --slice providers-integrations --json
```

## Script Rules

- Do **not** add scripts without listing them here. Wire CI when the script is
  the authoritative gate rather than a local wrapper over existing CI commands.
- Do **not** duplicate `dx_smoke` steps in new scripts; extend `dx_smoke.sh` or call it.
- Prefer `uv run rag-core …` for product behavior; scripts are **coordination** only.
- Before claiming validation strength, read [tests/README.md](../tests/README.md).
