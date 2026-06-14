# Test trust model

Large suite (~2000 tests). Passing everything does **not** mean production retrieval works.

## Validation ladder

| Tier | Marker | What it proves |
| --- | --- | --- |
| Provider replay | `provider_contract` | Recorded SDK shapes (no network) |
| Integration | `integration` | Local process, HTTP runtime, or real adapter paths: Qdrant `:memory:` retrieval on `integration_corpus`, parser/chunker pipeline, ASGI runtime. Some use fake embedders where the claim is wiring, not ranking |
| Plumbing | `plumbing` | Fake embedders / scripted stores, wiring only |
| Retrieval regression | `eval` | `tests/evals/pr_corpus/`: fixed vectors + Qdrant for pipeline/metric regression; `tests/evals/semantic_corpus/`: real local embeddings + Qdrant for a small end-to-end semantic-quality floor |
| Eval harness | `eval_harness` | Keyword metric plumbing (`baseline/`), not retrieval regression |
| Meta | `meta` | Docs, packaging, public-surface checks |
| Live | `live` | Paid APIs; **not** in default CI. Run locally when you have credentials |

Semantic retrieval quality on your data belongs in **your app** via `rag_core.evals` (`examples/retrieval_eval.py`), not in this repo's CI.

## CI (pull request / push)

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`.

Non-pytest checks:

- `./scripts/dx_smoke.sh` on Python 3.12
- `./scripts/verify_vercel_ai_sdk_example.sh` on Python 3.12
- `./scripts/ci_self_host_smoke.sh` on Python 3.12
- `./scripts/verify_optional_integrations.sh` on Python 3.12
- `uv run ruff check .`
- `uv run mypy src tests examples`
- `uv run python scripts/architecture_pressure.py --json`
- `uv build`
- `uv run python scripts/check_dist_artifacts.py`
- `uv run python scripts/wheel_smoke.py`

Pytest tiers:

1. Fast pytest tier (unit, contracts, plumbing)
2. `provider_contract`
3. `integration`
4. `tests/evals/test_retrieval_eval_pr.py` and `tests/evals/test_semantic_quality_local.py`: retrieval regression

No scheduled workflows. No API-key eval in CI.

`.github/workflows/release-artifacts.yml` runs on manual dispatch and `v*` tags.
It uploads checked `dist/*` artifacts only; it does not publish to PyPI.

## Launch gates

Run these before a public release or launch claim:

Fast iteration during local work:

```bash
./scripts/landing_check.sh --quick
```

Full release short form:

```bash
./scripts/landing_check.sh
```

Public checkout/package proof after pushing:

```bash
./scripts/public_checkout_smoke.sh --package
./scripts/github_install_smoke.sh https://github.com/kaanarici/rag-core.git main
```

Expanded release gate:

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

These prove packaging, typing, fixed-fixture retrieval regressions, a small
local semantic-quality regression gate, no-key developer journeys, and the
optional HTTP wrapper. They still do not prove semantic quality on arbitrary
user data or live paid-provider behavior.

CI runs the pytest suite as marker tiers so failures identify the broken claim
faster. The release gate remains the canonical command list above.

## Confidence aids

Use these while iterating locally or debugging a specific claim:

```bash
uv sync --group dev
./scripts/landing_check.sh --quick
./scripts/dx_smoke.sh

uv run ruff check . && uv run mypy src tests examples
uv run pytest -q -m "not live and not eval and not eval_harness and not provider_contract and not integration"
uv run pytest -q -m provider_contract
uv run pytest -q -m integration
uv run pytest -q tests/evals/test_retrieval_eval_pr.py tests/evals/test_semantic_quality_local.py
```

Optional (credentials required):

```bash
uv run pytest -q -m live --maxfail=3
```

## Fixture layout

| Path | Role |
| --- | --- |
| `tests/evals/pr_corpus/` | PR retrieval regression corpus + precomputed embeddings |
| `tests/evals/semantic_corpus/` | Local semantic-quality regression corpus with near-miss distractors |
| `tests/evals/baseline/` | Keyword fake embedder metric harness |
| `tests/fixtures/integration_corpus/` | 10-doc integration search corpus |
| `tests/fixtures/real_documents/` | Small externally authored parser fixtures with source/license notes |
| `tests/fixtures/providers/` | Provider contract JSON replay |

## Rules

- `RecordingVectorStore` and `KeywordEmbeddingProvider` are not product proof.
- `tests/evals/semantic_corpus/` proves end-to-end ranking on one fixed small corpus with local embeddings; it does not prove quality on arbitrary user data.
- TurboPuffer contract uses `tests/support/turbopuffer_fake.py`.
- Do not lower eval floors without noting why in commit or research doc.
- Validate provider fixtures: `./scripts/validate_provider_fixtures.sh`

## Known gaps

- Log-sanitization tests share `tests.support.log_sanitization` helpers; migrate remaining files incrementally.
- Live provider tests are optional and credential-gated (local only).
