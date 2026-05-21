# rag-core v0 roadmap

One-page checklist for the **v0 pre-release** product shape (public `0.1.x` until we call it stable).

## Phase A — Debloat

- [x] Qdrant-only vector store in the default wheel
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

## Phase E — v0 release gate

- [x] Local gates: ruff, mypy, pytest, build, wheel smoke
- [x] `test_v1_product_gate.py` (CLI surface, Ragie-shaped hits)
- [x] Tier 0 contract freeze documented in strategy plan

## Phase F — One repo / DX (active)

Maintainers: `./scripts/setup_agent_docs.sh` (local) · Product docs: `docs/README.md`

### Journey C — self-host

- [x] Golden path: `compose.yaml` + `docs/self-host/quickstart.md` + `.env.example`
- [x] HTTP journey tests + `scripts/self_host_smoke.sh`; `serve` reuses one `RAGCore` per process
- [x] Shape C3: OpenAPI + stable HTTP errors + rich health (`/health/ready`, `docs/self-host/auth.md`)
- [x] Shape C2: `compose.yaml` Qdrant + `serve` image (`Dockerfile`), `docs/self-host/config.md`
- [x] Ingest job errors include exception message (truncated)
- [ ] Auth middleware shipped in core (recipe only — app-owned by design)

### Journey A — first 10 minutes

- [x] Shape A2: `docs/quickstart.md` guided path (hits, context, trace, eval)
- [x] Shape A1: `scripts/dx_smoke.sh` non-interactive smoke
- [x] Wheel-installed quickstart: `python -m rag_core.quickstart` (+ wheel smoke)
- [x] CI DX smoke job (Python 3.12 in `ci.yml`)

### Journey B — embed

- [x] Shape B2: `docs/embedding/production-guide.md`
- [x] Shape B1: `examples/embedded_service.py` lifecycle pattern
- [x] Connector pattern: `docs/embedding/connector-pattern.md`

### Research + quality

- [x] Research notes (local `docs/research/`, gitignored)
- [ ] CI tiers: nightly `eval` workflow (Q2a)

## Phase G — Shape refresh

Product docs: `docs/README.md` · Agent templates: `docs/templates/`

### Default next slices

- [ ] **Q2a** — named public benchmark corpus + nightly `eval` workflow
- [ ] **B2.5** — production embed proof (lifecycle + tenancy + trace in example)
- [ ] **C3b** — OpenAPI/route contract drift test

### Journey V — TurboPuffer (v0 optional)

Docs: [docs/providers/vector-stores.md](docs/providers/vector-stores.md) (TurboPuffer section)

- [x] **TP1** — base adapter (ANN, filters, upsert, delete, health, doctor, `--extra turbopuffer`)
- [x] **TP2** — hybrid `QueryPlan` (BM25 + dense multi-query RRF; `lexical_query` on `SearchQuery`)
- [x] **TP3** — SparseKNN + sparse upsert schema (honest capability matrix in doctor)

**Maintainer order when something breaks:** `./scripts/dx_smoke.sh` → `doctor --json` → provider docs.

## Later (only when we have external users or claims)

- Q2a public benchmark corpus + nightly eval
- B2.5 production embed proof slice
- C3b OpenAPI drift gate

## Phase F cleanup (done)

- [x] Naming P1: `use_lexical_search`, `DocumentIndexer`, `cli_search`, `local_ingest`
- [x] Local display rebrand tooling (`dev/project_identity.toml`, `scripts/local_rebrand.sh`)
- [x] Human README + `docs/README.md` index
