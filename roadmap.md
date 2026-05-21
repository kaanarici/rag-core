# rag-core v1 roadmap

One-page checklist for the v1 product shape.

## Phase A — Debloat

- [x] Qdrant-only vector store in v1 wheel
- [x] Remove `rag-core eval` and `trace-summary` CLI
- [x] Slim `rag_core.evals` library + `examples/retrieval_eval.py`
- [x] MISSION, README, ADRs, tests, CI aligned

## Phase B — Contract clarity

- [x] `docs/expectations.md`
- [x] `rag_core.events.export.to_retrieval_hits`
- [x] Export shape tests

## Phase C — CLI for agents

- [x] `--help` Examples on core commands
- [x] README: five canonical agent invocations

## Phase D — Minimal runtime

- [x] `[runtime]` optional extra
- [x] `rag-core serve` (health, runtime, ingest jobs, search, retrieve-context)
- [x] No eval HTTP in v1

## Phase E — v1 gate

- [x] Local gates: ruff, mypy, pytest, build, wheel smoke
- [x] `test_v1_product_gate.py` (CLI surface, Ragie-shaped hits)
- [x] `roadmap.md` beta → stable checklist (Tier 0 freeze documented in plan)

## Post-v1.1

- TurboPuffer first-party adapter
- Heavy eval reporting module if needed
- Runtime eval endpoint only if demanded
