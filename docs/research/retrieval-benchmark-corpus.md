# Retrieval benchmark corpus (planned)

**Accessed:** 2026-05-20  
**Status:** Planning doc for Journey Q — fast fixture exists; public corpus TBD.

## Goals

| Tier | Purpose | Target gate |
|------|---------|-------------|
| Q1 fast fixture | Every PR | `examples/retrieval_eval.py` + `pytest -m eval` (subset) |
| Q2 public corpus | Credibility vs managed RAG | Named dataset + license + nightly job |

## Q1 (today)

- **Corpus:** `examples/demo_corpus/` (billing, security, lifecycle markdown)
- **Runner:** `examples/retrieval_eval.py` using `rag_core.evals`
- **Command:** `uv run python -m examples.retrieval_eval`

Acceptance: exit code 0 on main; documented in [quickstart.md](../quickstart.md).

## Q2 (candidate shapes — pick one in Shape Gate)

| Candidate | Corpus | Pros | Cons |
|-----------|--------|------|------|
| BEIR subset | Academic IR | Reproducible | Not product docs |
| Custom SaaS-doc slice | On-topic | Matches escape narrative | License curation work |
| NanoBEIR / small HF set | Fast CI | Cheap | Less “real docs” |

**Default recommendation:** curated **SaaS-help-center slice** (10–30 markdown/PDF files, MIT/Apache only) checked into `tests/evals/fixtures/` with source URLs in this file.

## Metrics (v1 library evals)

- Recall@k on labeled queries
- Citation presence in `retrieve_context`
- Regression triage via JSON report (no hosted dashboard)

## CI tiers (target)

| Tier | When | Command |
|------|------|---------|
| Fast | Every PR | `pytest -q` (excludes heavy `eval`) |
| Eval | Nightly / manual | `pytest -m eval -q` |
| DX | PR (3.12) | `./scripts/dx_smoke.sh` |

## Next implementation slice

1. Name chosen corpus + license paragraph in this file.
2. Add fixture path + one labeled query set.
3. Wire `.github/workflows/eval.yml` nightly trigger.

Parent: [one-repo-retrieval-engine-strategy.md](../plans/one-repo-retrieval-engine-strategy.md).
