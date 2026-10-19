# Development and release

Requires Python 3.11+ and uv. Run checks locally from the repository root:

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q -m 'not live'
./scripts/dx_smoke.sh
uv build
uv run python scripts/check_dist_artifacts.py
uv run python scripts/wheel_smoke.py --keep-temp
```

`./scripts/landing_check.sh` runs those checks. `--quick` selects the smaller
iteration checks. Neither command publishes, pushes, or starts remote CI.
GitHub Actions workflows are not shipped in this repository.

The wheel smoke creates a fresh consumer environment, installs the built wheel,
and exercises real local embeddings, hybrid folder retrieval, persistence,
CLI reuse, and the scoped application tool. Cached models can be reused.
`--keep-temp` retains the environment and outputs for inspection.

Artifact checks reject stale modules and local agent/planning files. They verify
that packaged source matches the checkout and that public docs and examples are
present in the source archive. Build into an empty `dist/` directory so old wheels
do not get mistaken for the current package.

## What the checks establish

The suite covers contracts, parsing, source locations, storage publication,
recovery, context, and local retrieval. Test doubles establish wiring and work
ownership. They do not establish semantic quality. See the full
[verification boundaries](../tests/README.md).

TurboPuffer checks use its actual SDK with HTTP fixtures. They do not establish
hosted ranking, service availability, or capacity. Credentialed provider checks
are separate, explicit operations. Do not include them in the local gate.

A passing suite does not prove answer correctness or performance on arbitrary
documents. Use your application's documents and expected evidence when
[evaluating retrieval](evaluation.md).

## Preparing an update

1. Read the diff, including untracked source files. Exclude secrets, local state,
   generated indexes, and unrelated work.
2. Update the [migration guide](migration.md) for changed APIs or storage. Keep
   the [README](../README.md) examples runnable against the final code.
3. Run the local gate against the final files. Record which provider and format
   paths were exercised. Do not describe fixture or local checks as live proof.
4. Commit only the verified update. Include `[skip ci]` when remote automation
   must remain off. Never force-push over another contributor's work.
5. After an authorized push, verify the remote commit SHA. No tag, GitHub release,
   or package upload is implied by a source push.

rag-core is an early unpublished library. PyPI publication, version tags, hosted
deployments, and live provider runs require their own explicit request.
