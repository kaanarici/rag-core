# Test trust model

Large suite (~2000 tests). Passing everything does **not** mean production retrieval works.

## Validation ladder

| Tier | Marker | What it proves |
| --- | --- | --- |
| Provider replay | `provider_contract` | Recorded SDK shapes (no network) |
| Integration | `integration` | Real Qdrant `:memory:` + parse/chunk/pipeline on `integration_corpus` |
| Plumbing | `plumbing` | Fake embedders / scripted stores — wiring only |
| PR retrieval regression | `eval` | `tests/evals/pr_corpus/` — fixed vectors + Qdrant; pipeline/metric regression only |
| Eval harness | `eval_harness` | Keyword metric plumbing (`baseline/`) — not retrieval regression |
| Meta | `meta` | Docs, packaging, exports |
| Live | `live` | Paid APIs; **not** in default CI — run locally when you have credentials |

Semantic retrieval quality on your data belongs in **your app** via `rag_core.evals` (`examples/retrieval_eval.py`), not in this repo's CI.

## CI (pull request / push)

`.github/workflows/ci.yml` only:

1. Fast pytest tier (unit, contracts, plumbing)
2. `provider_contract`
3. `integration`
4. `tests/evals/test_retrieval_eval_pr.py` — retrieval regression

No scheduled workflows. No API-key eval in CI.

## Local gates (run before you ship)

```bash
uv sync --group dev
./scripts/dx_smoke.sh

uv run ruff check . && uv run mypy src tests examples
uv run pytest -q -m "not live and not eval and not eval_harness and not provider_contract and not integration"
uv run pytest -q -m provider_contract
uv run pytest -q -m integration
uv run pytest -q tests/evals/test_retrieval_eval_pr.py
```

Optional (credentials required):

```bash
uv run pytest -q -m live --maxfail=3
```

## Fixture layout

| Path | Role |
| --- | --- |
| `tests/evals/pr_corpus/` | PR retrieval regression corpus + precomputed embeddings |
| `tests/evals/baseline/` | Keyword fake embedder metric harness |
| `tests/fixtures/integration_corpus/` | 10-doc integration search corpus |
| `tests/fixtures/providers/` | Provider contract JSON replay |

## Rules

- `RecordingVectorStore` and `KeywordEmbeddingProvider` are not product proof.
- TurboPuffer contract uses `tests/support/turbopuffer_fake.py`.
- Do not lower eval floors without noting why in commit or research doc.
- Validate provider fixtures: `./scripts/validate_provider_fixtures.sh`

## Known gaps

- Log-sanitization tests are per-module files; `assert_log_sanitized()` exists for future consolidation.
- Live provider tests are optional and credential-gated (local only).
