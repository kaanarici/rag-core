# One repo: retrieval engine strategy

**Status:** Full product spec (reference). **Agents: read [ROUTING.md](ROUTING.md) + [../CONTEXT.md](../CONTEXT.md) before editing.**  
**Audience:** Humans and agents planning rag-core work  
**Historical plans:** [archive/](archive/) (do not execute).

**Product shape (one sentence):** rag-core is the **retrieval plane** you embed in your app; CLI and optional HTTP are proof/ops surfaces, not a second product.

---

## Agent Non-Divergence Contract

**Routing and shape gate:** [ROUTING.md](ROUTING.md) (canonical, short).  
**Mission and code map:** [../CONTEXT.md](../CONTEXT.md).  
**Tracked agent copy:** [../AGENTS.md](../AGENTS.md).

This long file is thesis, journey packets, and history. If another doc conflicts on *what to do next*, ROUTING wins. If conflicts on *mission or scope*, CONTEXT wins.

### Current goal lock

Build `rag-core` into the **one repo retrieval engine** that lets serious teams leave black-box managed RAG without adopting platform sprawl.

That means every substantial slice must improve at least one selected journey shape:

| Journey | Purpose | Best shape | Phase F status |
|---------|---------|------------|----------------|
| A — First 10 minutes | Trust without account/key/platform | A2 guide + A1 smoke + **A3** wheel quickstart | **Done** |
| B — Embed | Replace managed RAG inside an app | B2 guide + B1 app + connector **doc** | **Docs done** — B2.5 proof **next** |
| C — Self-host API | Operable HTTP over same core | C3 contract → C2 compose → C3b drift gate | **C3+C2 done** — C3b/C-ops optional |
| Q — Quality proof | Evidence over anecdotes | **Q2a** named corpus + Q1 fast fixture | **Open** (Q2a); not default maintenance |
| R — Research | Competitor context tracked | R0a docs + dated refresh | **Done** |
| V — Managed vector (v0) | Production without operating Qdrant | **TP1→TP3** TurboPuffer depth | **Done** — optional extra |

Shape gate, default routing, hard stops, and slice rules: **[ROUTING.md](ROUTING.md)**.

---

## Thesis

**rag-core is the single Python repo that owns the retrieval plane** for products that refuse black-box managed RAG APIs (Ragie-style “upload → magic search”) and refuse OSS platform sprawl (RAGFlow/Haystack/LlamaIndex monoliths with graph UIs, agent canvases, and hosted control planes).

Your app still owns:

- Auth, tenancy, billing, admin UI  
- Chat, agents, tool orchestration, model routing  
- Connector credentials and product-specific sync jobs  

**rag-core owns:**

- Parse → chunk → manifest → index → hybrid search → rerank orchestration → citations → model-ready context → traces → library evals  
- Optional thin HTTP runtime (`serve`) over the **same** `RAGCore` contracts  

That is what “one repo to rule them all” means here: **one embeddable engine**, not one hosted SaaS, not one agent builder, not one connector marketplace.

---

## Why leave managed RAG

Managed platforms sell a **closed retrieval plane**:

| You rent | What you lose |
|----------|----------------|
| Document ingest + parsing | No real parser/chunk policy control; upgrades change behavior silently |
| Chunking + indexing | Opaque chunk boundaries; hard to debug bad answers |
| Search + rerank | Black-box fusion; limited query-plan honesty |
| “Scored chunks” API | Vendor owns IDs, metadata schema, and retention |
| Connectors (Drive, Notion, …) | Data leaves your boundary; sync semantics are theirs |
| Ops + scaling | Simple start; migration off is expensive |

**rag-core’s bet:** teams with serious RAG needs will pay engineering once to own retrieval quality, then embed or self-host the same code forever. API shape can stay **Ragie-compatible at the hit layer** (`docs/expectations.md`) without depending on Ragie infrastructure.

---

## Competitive landscape (research summary)

### Managed RAG APIs (escape target)

**Ragie** (primary API benchmark in-repo):

- Hosted ingest (files + connectors), processing pipelines (incl. audio/video modes)  
- `/retrievals` semantic search, reranking, metadata filters, `top_k`, chunk metadata  
- Summary index across documents; webhooks; Python/TS SDKs  
- **rag-core parity target:** `SearchResult` / `scored_chunks` field map, retrieve-context, traces—not hosted ingest/connectors  

**Others in the same category** (not documented in-repo yet; treat as same escape class):

- Vectara, Contextual.ai file-search APIs, cloud “knowledge base” products bundled with models  

### OSS “full stack” RAG (inspiration only — do not become)

**RAGFlow** (~80k stars, Docker microservices):

- DeepDoc PDF/layout/OCR, template chunking with human preview  
- Hybrid search + rerank, grounded citations, GraphRAG, agent canvas, MCP  
- Connectors (Drive, Notion, S3, …), Redis task queue, admin UI  
- **Steal:** deep document understanding ambition, citation UX, hybrid+rerank seriousness  
- **Refuse:** microservice platform, visual agent editor, connector marketplace, Redis/Go admin shell  

**Haystack 2** (pipeline graphs):

- Component graphs, branches, loops, async pipelines, many retrievers/generators  
- **Steal:** retrieval technique vocabulary (HyDE, multi-query, MMR, rerank stages)  
- **Refuse:** graph DSL as core model (`docs/adr/0002-linear-pipeline-no-dsl.md`)  

**LlamaIndex** (data-first framework + LlamaCloud):

- 160+ connectors, LlamaParse, index types, query/chat engines, agents  
- **Steal:** ingest breadth goals, index/query separation, framework integration patterns  
- **Refuse:** becoming a second framework; cloud index as default; agent runtime in core  

**RAG-Anything / similar multimodal stacks:**

- **Steal:** multimodal ingest ideas when they strengthen parsing  
- **Refuse:** scope explosion before core retrieval is undeniable  

Per `AGENTS.md`: external repos are **planning/review inspiration**, not feature parity mandates.

---

## Capability scorecard (honest, 2026-05-21)

| Capability | Managed (Ragie) | OSS platform (RAGFlow) | rag-core today | “One repo” bar |
|------------|-----------------|-------------------------|----------------|----------------|
| No-key first run | N/A (account) | Docker heavy | `demo`, `local-search` | **Strong** |
| Deep PDF/OCR path | Yes | DeepDoc | PDF inspector + OCR routing | **Good**, needs DX story |
| Hybrid + rerank | Yes | Yes | Yes, profile + QueryPlan | **Strong** |
| Citations + context pack | Yes | Yes | `retrieve_context` | **Strong** |
| Inspectability | Low | Medium (UI) | events JSONL + trace summaries | **Strong** differentiator |
| Library embed | No | Partial | `RAGCore` facade | **Strong** |
| Self-host HTTP | Yes (their cloud) | Yes (full app) | `serve` minimal | **Early** (Lane 1 landed plumbing only) |
| Connectors | Yes | Yes | file/URL/archive only | **App-owned** by design |
| Agent/graph orchestration | Some | Canvas | App-owned | **Out of scope** |
| Hosted eval API | Some | Yes | Library evals only | **Correct for v1** |
| Public benchmark corpus | N/A | Some | Open question | **Missing** |
| OpenAPI / SDK ergonomics | Yes | Yes | Partial | **Weak** |
| Production compose/k8s | N/A | Yes | Qdrant compose only | **Weak** |
| TurboPuffer managed vector | N/A | N/A | v0 TP1–TP3 | **Useful** — [research](../research/turbopuffer-landscape.md); not Journey A default |

**Takeaway:** Core **library retrieval** is competitive. **Adoption packaging** (self-host as a product experience, benchmarks, connector patterns, ops docs) is where “one repo” still loses to managed + OSS platforms—**not** because hybrid search is missing.

---

## Shape-selection rule

Do **not** assume the current A/B/C names are the best product shapes. They are working hypotheses. Before implementation, the coordinator must run a **Shape Gate**:

1. Pick one journey to improve.
2. Compare at least **three candidate shapes** for that journey.
3. Choose one shape using the scorecard below.
4. Convert it into exact repo artifacts and acceptance gates.
5. Implement only that shape in the current slice.

### Shape scorecard

Score each candidate from 1–5. Highest total wins unless it violates an anti-goal.

| Criterion | Question |
|-----------|----------|
| Time-to-proof | Can a serious developer see retrieval working quickly? |
| Transferability | Does it teach concepts reused in embedded and self-hosted mode? |
| Inspectability | Does it expose manifests, traces, evals, runtime, or contract JSON? |
| Migration value | Does it reduce dependence on managed RAG APIs? |
| Small core | Does it avoid platform, connector-marketplace, or graph-runtime scope? |
| Agent executability | Can a fresh agent implement and verify it from this plan? |

### Required shape output

Every selected shape must produce this block before code changes:

```markdown
## Selected Shape
Journey: A | B | C | Quality | Research
Shape name:
User promise:
Why this shape won:
Artifacts to create/update:
Commands a developer runs:
Acceptance gates:
Subagents needed:
Out of scope:
```

If an agent cannot fill that block concretely, it must not implement.

### Shape packet template

After the Shape Gate, copy one of these packets into the work notes and execute it literally.

#### Packet A — First 10 minutes

```markdown
## Selected Shape
Journey: A
Shape name: A2 guided doc + A1 smoke script
User promise: In 10 minutes, a developer sees hits, model context, trace evidence, and an eval without keys.
Why this shape won:
- Teaches reusable concepts instead of hiding them behind a magic script.
- Produces a smoke command agents and CI can run.
- Proves inspectability, not just search output.
Artifacts to create/update:
- docs/quickstart.md or README first-run section
- scripts/dx_smoke.sh
- .github/workflows/ci.yml only if adding DX smoke now
Commands a developer runs:
- uv run rag-core demo --json
- uv run rag-core local-search examples/demo_corpus "How can invoices be paid?" --events-jsonl /tmp/rag-core-events.jsonl --json
- uv run python -m examples.retrieval_eval
- ./scripts/dx_smoke.sh
Acceptance gates:
- commands above pass with no external keys
- smoke emits stable success lines
- docs show expected output snippets
Subagents needed: R3 only; add R6 if adding CI
Out of scope: new CLI commands, connectors, hosted evals
```

#### Packet B — Embedded app

```markdown
## Selected Shape
Journey: B
Shape name: B2 production guide + B1 minimal app
User promise: A developer can replace a managed RAG API inside their app without guessing lifecycle, tenancy, or context handoff.
Why this shape won:
- Embedding is the default product path.
- It teaches app-owned auth/tenancy without adding platform code.
- It reuses the same core as CLI and serve.
Artifacts to create/update:
- docs/embedding/production-guide.md
- examples/embedded_service.py or tightened existing example
- docs/embedding/connector-pattern.md if connector replacement is in scope
Commands a developer runs:
- uv run python -m examples.minimal_app
- uv run python -m examples.source_ingest
- uv run pytest tests/test_examples.py -q
Acceptance gates:
- one RAGCore lifecycle per process/worker
- namespace/corpus bound by app code, not model input
- trace/context handoff shown
Subagents needed: R4; add R1 only if contract shape changes
Out of scope: built-in SaaS connectors, auth implementation, agent orchestration
```

#### Packet C — Self-host API

```markdown
## Selected Shape
Journey: C
Shape name: C3 API contract first
User promise: A developer can point services and tools at a stable self-hosted retrieval API.
Why this shape won:
- Lane 1 already made serve runnable.
- Contract credibility beats more Docker before the API is stable.
- Agents/SDKs need OpenAPI and stable errors, not README curls.
Artifacts to create/update:
- docs/self-host/openapi.yaml
- docs/self-host/auth.md
- src/rag_core/runtime/app.py
- tests/test_runtime_http.py
Commands a developer runs:
- uv run pytest tests/test_runtime_http.py -q
- uv run ruff check src/rag_core/runtime tests/test_runtime_http.py
- uv run mypy src/rag_core/runtime tests/test_runtime_http.py
Acceptance gates:
- consistent HTTP error response shape
- readiness/health distinguishes liveness from dependency status
- no eval HTTP
- quickstart links OpenAPI and auth recipe
Subagents needed: R5 and optionally R1
Out of scope: core auth, hosted accounts, connector marketplace, full compose image policy
```

#### Packet Q — Quality proof

```markdown
## Selected Shape
Journey: Q
Shape name: Q2 public benchmark corpus + Q1 fast fixture
User promise: Retrieval changes are measured against a public corpus and a fast local fixture.
Why this shape won:
- Managed-RAG escape needs evidence, not vibes.
- Heavy benchmarks should not slow every PR.
- Fast fixtures catch regressions locally.
Artifacts to create/update:
- docs/research/retrieval-benchmark-corpus.md
- docs/evals/retrieval-quality.md
- .github/workflows/eval.yml if changing schedule/trigger
- tests/evals/... only for the fast fixture
Commands a developer runs:
- uv run pytest -m eval -q
- uv run python -m examples.retrieval_eval
Acceptance gates:
- public corpus named with source/date/license notes
- default CI remains fast
- eval results are reproducible enough for regression triage
Subagents needed: R6 and R7
Out of scope: hosted eval UI, eval HTTP, broad metric framework
```

#### Packet R — Research corpus

```markdown
## Selected Shape
Journey: R
Shape name: R0a tracked research docs
User promise: Future agents do not rediscover competitor research in chat.
Why this shape won:
- Repo currently lacks tracked comparison docs.
- Research must guide shapes without forcing parity.
- Read-only docs reduce implementation drift.
Artifacts to create/update:
- docs/research/managed-rag-landscape.md
- docs/research/oss-rag-landscape.md
- docs/research/retrieval-benchmark-corpus.md
Commands a developer runs:
- none required beyond markdown review
Acceptance gates:
- every claim links to a public source and access date
- each row says mirror, steal, refuse, or defer
- no product implementation changes
Subagents needed: R7 plus targeted web research
Out of scope: code changes, benchmark implementation, vendor parity
```

---

## Candidate journey shapes

These are **candidate shapes**, not mandates. The coordinator should use them as a menu and adjust after reading current code/docs.

### Journey A — First 10 minutes (trust)

Goal: prove this is a real retrieval engine without an account, provider key, hosted API, or framework lock-in.

| Candidate | Shape | Best when | Risks |
|-----------|-------|-----------|-------|
| A1 | **Single command smoke**: one script runs `demo`, `local-search`, trace summary, and library eval | We need shortest possible proof | Can hide concepts if it is just a magic script |
| A2 | **Guided notebook-style doc**: step-by-step commands with expected JSON excerpts | We need teachability | Slower than a script |
| A3 | **Wheel-first example**: install package, run `python -m rag_core.quickstart` or installed sample | We need install confidence | Requires packaging decisions |

**Current best default:** **A2 + A1 + A3.** Guide teaches concepts; smoke automates CI; `python -m rag_core.quickstart` proves wheel install.

**Landed (Phase F):** `docs/quickstart.md`, `scripts/dx_smoke.sh`, CI on 3.12, `rag_core.quickstart`.

**Maintenance only:** keep user-folder-before-demo-corpus ordering in README/tests aligned.

**Acceptance gates:**

- No external keys.
- Finishes in under 2 minutes on a normal laptop.
- Shows one raw hit, one context pack, one trace summary, and one eval result.

### Journey B — Embed in your app (default)

Goal: replace managed RAG inside a real application process.

| Candidate | Shape | Best when | Risks |
|-----------|-------|-----------|-------|
| B1 | **Minimal embedded app**: one Python file with lifecycle, ingest, retrieve-context | We need the smallest app story | Too toy if it skips tenancy/errors |
| B2 | **Production embed guide**: lifecycle, namespace/corpus binding, shutdown, retries, traces | We need serious adoption | Could become too long |
| B3 | **Connector pattern**: external sync job feeds `RAGCore` through manifest contracts | We need replace managed connectors | Easy to drift into connector marketplace |

**Current best default:** **B2 + B1 + B2.5.** Docs landed; next slice must prove tenancy, shutdown, trace handoff, and one realistic failure path.

**B2.5 artifact shape:**

- Tighten `examples/embedded_service.py` or add thin `examples/fastapi_mount.py` (no framework in core)
- Checklist section in production-guide
- `pytest tests/test_examples.py` coverage for lifecycle pattern

**Acceptance gates:**

- App binds `namespace` and `corpus_id`; model never controls tenant scope.
- One `RAGCore` per app/worker lifecycle, closed on shutdown.
- Shows file/bytes/URL/archive ingest choices and when to use each.
- Shows trace capture and context pack handoff.

### Journey C — Self-host retrieval API (optional)

Goal: offer an operable API over the same engine without becoming a hosted platform.

| Candidate | Shape | Best when | Risks |
|-----------|-------|-----------|-------|
| C1 | **Host-run serve + Qdrant compose** | Fastest local success | Still feels half self-hosted |
| C2 | **Full compose: Qdrant + rag-core API** | Matches developer expectation of self-host | Needs image/build story |
| C3 | **API contract first**: OpenAPI + stable error model, runtime still host-run | Best for SDK/agent consumers | Less visually satisfying than `docker compose up` |

**Current best default:** **C3 and C2 done.** Next C work is **ops honesty**, not feature sprawl.

**C3b (contract drift gate):** test that routes/error codes match `docs/self-host/openapi.yaml`.

**C-ops (doc):** ingest `path` is server-local; volume mounts; when not to use compose for multi-tenant prod.

**Anti-shape:** eval HTTP, in-core auth, connector APIs on `serve`.

**Acceptance gates:**

- `serve` has no eval HTTP.
- `GET /health` distinguishes liveness from dependency readiness.
- HTTP errors are consistent enough for agents/SDKs.
- `docs/self-host/quickstart.md` links to OpenAPI and auth recipe.

### Quality journey — prove it is better than anecdotes

Goal: make retrieval quality measurable enough that managed platforms stop being attractive.

| Candidate | Shape | Best when | Risks |
|-----------|-------|-----------|-------|
| Q1 | **Tiny built-in eval fixture** | Fast CI signal | Not credible externally |
| Q2 | **Public benchmark corpus doc + optional nightly job** | Serious comparison | Takes research and curation |
| Q3 | **Provider/profile bakeoff harness** | Useful for tuning | Can balloon into eval platform |

**Current best default:** **Q2 with Q1 as CI smoke.** Keep heavyweight benchmarks optional/nightly; keep the default loop fast.

### Research journey — keep strategy grounded

Goal: stop rediscovering managed/OSS comparisons in chat.

| Candidate | Shape | Best when | Risks |
|-----------|-------|-----------|-------|
| R0a | **Research docs only**: managed, OSS, benchmark corpus | Need shared context | Docs can go stale |
| R0b | **Scorecard script**: machine-readable competitor matrix | Need repeatable comparisons | Overkill now |
| R0c | **Quarterly review protocol** | Long-lived product planning | Process without content |

**Current best default:** **R0a.** Write the comparison corpus first.

**Landed:** `docs/research/managed-rag-landscape.md`, `oss-rag-landscape.md`, `retrieval-benchmark-corpus.md`, **`turbopuffer-landscape.md`**.

---

### Journey V — Managed vector store (v0 pre-release, TurboPuffer)

Goal: offer a **credible managed vector** path without breaking Qdrant-as-default or Journey A no-key story.

| Candidate | Shape | Best when | Risks |
|-----------|-------|-----------|-------|
| V1 | **TP1 restore** — ANN, filters, upsert, delete, health, doctor | Need managed option fast | Looks “done” without hybrid |
| **V2** | **TP1 + TP2 hybrid** — BM25 + dense multi-query → `QueryPlan` fusion | Matches rag-core retrieval story | Live API tests, query cost fan-out |
| V3 | TurboPuffer-only query shortcuts in core | Max perf on TP | Violates vendor-neutral `QueryPlan`; refuse |

**Current best default:** **V2 phased: TP1 then TP2.** See [turbopuffer-landscape.md](../research/turbopuffer-landscape.md).

**Hard stops for Journey V:**

- No TurboPuffer in default wheel without `--extra turbopuffer`
- No claiming parity until `test_vector_store_contract.py` + doctor + capability matrix say so
- No Journey A / compose golden path dependency on TurboPuffer API keys

#### Packet V — TurboPuffer TP1 (restore base adapter)

```markdown
## Selected Shape
Journey: V
Shape name: TP1 restore base adapter
User promise: I can configure rag-core to use TurboPuffer for dense ANN retrieval with honest limits.
Why this shape won:
- ADR-0001 already commits to first-party managed vector.
- Prior adapter pattern exists in repo history; provider-output audit defines the floor.
- TP1 is testable without full hybrid complexity.
Artifacts to create/update:
- src/rag_core/search/providers/turbopuffer_*.py (restore/split modules)
- src/rag_core/config/turbopuffer_config.py + VectorStoreConfig provider allowlist
- registry + core_vector_store_factory + CLI flags + .env.example
- docs/providers/vector-stores.md + doctor surfaces
- tests/test_vector_store_contract.py (TurboPuffer marker)
- optional pyproject extra `turbopuffer`
Commands a developer runs:
- uv sync --extra turbopuffer
- uv run pytest tests/test_vector_store_contract.py -q -m turbopuffer  # when marker exists
- uv run rag-core doctor --vector-store turbopuffer --json  # when wired
Acceptance gates:
- contract tests pass for ANN + filters + delete + health
- unsupported QueryPlan stages fail closed with clear errors
- doctor lists capabilities and does not print secrets
Out of scope: BM25, SparseKNN, multi-query fusion (TP2), eval HTTP
```

#### Packet V2 — TurboPuffer TP2 (hybrid depth)

```markdown
## Selected Shape
Journey: V
Shape name: TP2 hybrid query-plan
User promise: Common search profiles (e.g. balanced) work on TurboPuffer, not only dense ANN.
Artifacts: turbopuffer_query_plan.py, profile mapping tests, live-backed hybrid tests (marker)
Acceptance gates:
- `search_profile=balanced` path produces fused results on TP namespace
- StoreCapabilities documents which stages are TP-backed
Out of scope: MMR/boost unless Qdrant parity proven necessary first
```

#### Packet Q2a — Quality (default continue)

```markdown
## Selected Shape
Journey: Q
Shape name: Q2a one public corpus + nightly
User promise: Retrieval regressions are measured on a named, licensed corpus.
Artifacts:
- docs/research/retrieval-benchmark-corpus.md (corpus chosen)
- tests/evals/fixtures/...
- .github/workflows/eval.yml
Acceptance gates:
- pytest -m eval on fixture in default CI or documented nightly
- corpus has source, date, license in doc
Out of scope: hosted eval UI, eval HTTP
```

---

## Product-shape map

Refined after the archived hardening plan (`plans/archive/2026-05-17-retrieval-core-hardening.md`):

```text
ONE REPO = one retrieval plane, three surfaces (library, CLI, optional HTTP)
├── A — Trust: A2 + A1 + A3          [DONE]
├── B — Embed: B2 + B1 + B2.5        [docs done; proof next]
├── C — Operate: C3 + C2 + C3b       [C3+C2 done]
├── Q — Believe: Q2a + Q1            [BLOCKER for “better than Ragie” claims]
├── R — Ground: R0a + TP research    [DONE]
└── V — Managed vector: TP1→TP2→TP3  [v0; launch now unless overridden]
```

**Funnel:** Trust (A) → Embed (B) → Operate (C) → **Believe (Q)**. Research (R) and Managed vector (V) feed decisions; they are not substitutes for Q.

**DX is glue**, not a journey. Self-host without A+B still does not convert.

---

## What Lane 1 actually was (and was not)

**Lane 1 (landed):** `compose.yaml`, `docs/self-host/quickstart.md`, `.env.example`, `self_host_smoke.sh`, demo provider on `serve`, process-lifetime `RAGCore`, HTTP journey tests.

**That is necessary table stakes** for Journey C. It is **not** sufficient for “rule them all.”

**Phase F landed vs remaining “one repo” gaps:**

| Gap | Status |
|-----|--------|
| `serve` in Compose | **Done** (C2) |
| OpenAPI + stable errors + `/health/ready` | **Done** (C3) |
| Auth middleware recipe | **Done** (doc) |
| Connector pattern doc | **Done** (B) |
| Wheel quickstart | **Done** (A3) |
| Public eval corpus + CI tier | **Open** (Q2a) |
| OpenAPI drift gate | **Open** (C3b) |
| TurboPuffer first-party managed path | **Done** (V / TP1–TP3; optional extra) |

---

## Research debt in this repository

Research docs exist under `docs/research/`; a **public benchmark CI gate (Q2a)** is still open on the roadmap.

| Missing artifact | Purpose |
|------------------|---------|
| `docs/research/managed-rag-landscape.md` | Ragie + peers: API surfaces we mirror vs reject |
| `docs/research/oss-rag-landscape.md` | RAGFlow/Haystack/LlamaIndex: steal/refuse matrix |
| `docs/research/retrieval-benchmark-corpus.md` | Chosen public corpus, metrics, CI gate |
| Track `MISSION.md` in git | Strategy currently gitignored; drifts from README |

**Subagent Lane R0 (research, read-only):** fill the three research docs from public docs only—no code changes. Coordinator merges into this strategy quarterly.

---

## Execution program (agent-ready)

This section is written so a fresh agent can execute without prior chat context.

**Rule:** do not implement a lane until the Shape Gate is complete and written into the work notes or plan file.

### Phase 0 — Intake and shape gate

Run these first:

```bash
git status --short
uv run rag-core demo --json
uv run rag-core local-search examples/demo_corpus "How can invoices be paid?" --json
uv run rag-core doctor --qdrant-location :memory: --embedding-provider demo --embedding-dimensions 64 --json
```

Then fill this table in the agent response before editing files:

| Field | Required content |
|-------|------------------|
| Journey chosen | A, B, C, Q, or R |
| Candidate shapes compared | At least three from the journey section |
| Winner | Candidate ID + name |
| Why it won | 2–4 bullets tied to scorecard |
| User-visible promise | One sentence a developer would care about |
| Exact artifacts | Files to add/update |
| Acceptance gates | Commands/tests/docs checks |
| Out of scope | What the slice will not touch |

If the chosen journey is unclear, default order is:

1. **A** if first-run trust is not scripted and documented.
2. **B** if embedded production docs are thin.
3. **C** if HTTP contract/ops is thin.
4. **Q** if claims are strong but quality proof is weak.
5. **R** if comparison/research docs are missing.

### Phase 1 — Read-only discovery subagents

Use at most six read-only subagents. Launch only the scopes needed by the selected shape.

Copy this exact handoff:

```text
WORKSPACE: /Users/kaanaricioglu/rag-core
MODE: READ-ONLY DISCOVERY
PARENT PLAN: docs/plans/one-repo-retrieval-engine-strategy.md

Goal: help choose or verify the best shape for <Journey ID>.
Do not propose platform sprawl. Do not implement.

SCOPE: <one scope from table below>

Return at most 8 rows:
| id | severity | file/path | claim | evidence | recommended shape impact |

Severity:
- P0 = blocks the selected journey from replacing managed RAG for retrieval
- P1 = weakens DX or proof but does not block
- D = defer; useful later but outside current shape
```

Discovery scopes:

| ID | Use for | Scope |
|----|---------|-------|
| R1 | A/C/contracts | Managed API parity: `docs/expectations.md`, `events/export.py`, runtime search/retrieve vs Ragie-style retrieval |
| R2 | A/B/parser | OSS steal/refuse: parser/chunk/citation story vs RAGFlow/Haystack/LlamaIndex claims |
| R3 | A | First 10 minutes: README, demo, local-search, doctor, eval example friction |
| R4 | B | Embedded app: `RAGCore`, lifecycle examples, LangChain/Vercel contracts, tenant binding |
| R5 | C | Self-host ops: `runtime/**`, `compose.yaml`, quickstart, auth pattern, OpenAPI |
| R6 | Q | Quality proof: evals, benchmark corpus, CI tiers, duplicate/low-signal tests |
| R7 | R | Research docs: managed + OSS comparison sources and missing tracked claims |

### Phase 2 — Coordinator adjudication

The coordinator, not subagents, decides. For every subagent row:

1. Read the cited file/path.
2. Decide `accept`, `reject`, or `defer`.
3. Update the selected shape block if evidence changes the winner.
4. Keep the slice to one journey and one shape.

Use this ledger:

| id | decision | reason | action |
|----|----------|--------|--------|
| R?-? | accept/reject/defer | evidence | file/test/doc change or none |

### Phase 3 — Implementation lanes

Each lane below already has a current best shape. Agents may choose a different shape only by completing the Shape Gate and explaining why.

#### Lane 2 — Journey C credible (current best: C3)

**User-visible promise:** “I can point tools and services at a stable self-hosted retrieval API.”

Artifacts:

- `docs/self-host/openapi.yaml`
- `docs/self-host/auth.md`
- Runtime error model in `src/rag_core/runtime/app.py`
- Rich `/health` or separate `/ready` semantics
- Tests in `tests/test_runtime_http.py`

Acceptance gates:

```bash
uv run pytest tests/test_runtime_http.py -q
uv run ruff check src/rag_core/runtime tests/test_runtime_http.py
uv run mypy src/rag_core/runtime tests/test_runtime_http.py
```

Non-goals: no auth implementation in core, no eval HTTP, no connector marketplace.

#### Lane 3 — Journey A bulletproof (current best: A2 + A1)

**User-visible promise:** “In 10 minutes, I can see raw hits, model context, traces, and an eval without keys.”

Artifacts:

- `docs/quickstart.md` or focused README first-run replacement
- `scripts/dx_smoke.sh`
- Optional CI job or documented future job
- Expected-output snippets that are stable enough for agents

Acceptance gates:

```bash
uv run rag-core demo --json
uv run rag-core local-search examples/demo_corpus "How can invoices be paid?" --events-jsonl /tmp/rag-core-events.jsonl --json
uv run python -m examples.retrieval_eval
./scripts/dx_smoke.sh
```

Non-goals: no new CLI surface unless the smoke reveals a real missing command.

#### Lane 4 — Journey B embed templates (current best: B2 + B1)

**User-visible promise:** “I can replace a managed RAG API inside my app without guessing lifecycle, tenancy, or context handoff.”

Artifacts:

- `docs/embedding/production-guide.md`
- Minimal embed example that is runnable and not framework-specific
- Connector pattern doc: external sync job → `RAGCore` ingest → manifest → retrieve-context
- Tests only if examples execute in CI

Acceptance gates:

```bash
uv run python -m examples.minimal_app
uv run python -m examples.source_ingest
uv run pytest tests/test_examples.py -q
```

Non-goals: do not add built-in Google Drive/Notion/etc. connectors.

#### Lane 5 — Quality proof (current best: Q2 + Q1)

**User-visible promise:** “Retrieval changes are measured against a public corpus and a fast local fixture.”

Artifacts:

- `docs/research/retrieval-benchmark-corpus.md`
- CI eval tier design
- One fast eval fixture retained in default tests
- Nightly/manual eval command documented

Acceptance gates:

```bash
uv run pytest -m eval -q
uv run python -m examples.retrieval_eval
```

Non-goals: no hosted eval dashboard; no eval HTTP.

#### Lane 6 — Parser/chunk narrative (shape must be chosen)

**User-visible promise:** “I understand exactly what rag-core extracts from hard documents and how citations survive retrieval.”

Required Shape Gate: compare RAGFlow-style deep-doc narrative vs compact format matrix vs snapshot fixtures.

Likely artifacts:

- `docs/parsing/formats.md`
- `docs/parsing/pdf-ocr.md`
- Parser snapshot fixtures/tests if useful

#### Lane R0 — Research corpus (current best: R0a)

**User-visible promise:** “Future agents do not rediscover competitor research in chat.”

Artifacts:

- `docs/research/managed-rag-landscape.md`
- `docs/research/oss-rag-landscape.md`
- `docs/research/retrieval-benchmark-corpus.md`

Acceptance gates:

- Each claim links to a public source and has an access date.
- Each competitor row says **steal**, **mirror**, **refuse**, or **defer**.
- No implementation changes.

### Phase 4 — Closure

Every completed slice must end with:

```markdown
## Slice closeout
Journey:
Selected shape:
Files changed:
Validation:
What this improves vs managed RAG:
What remains:
Next best lane:
```

Update `roadmap.md` only for completed artifacts, not intentions.

Lanes 2–4 are the real “one repo” work. Lane 1 was prerequisite.

---

## Phase G — Shape refresh (2026-05-20)

Phase F completed adoption packaging for A, B (docs), C, and R. **The bottleneck moved from trust to belief (Q) and managed-vector credibility (V).**

### Revised default order

```text
1. Q2a  — named public corpus + nightly eval
2. TP1   — TurboPuffer base adapter (if managed vector is a product commitment)
3. B2.5  — embed production proof
4. TP2   — TurboPuffer hybrid QueryPlan depth
5. C3b   — OpenAPI/route contract test
```

### Anti-shapes (Phase G)

- More compose/README polish while Q2a corpus is unnamed
- TP2 hybrid before TP1 contract tests green
- TurboPuffer in no-key smoke or default wheel
- Eval HTTP or in-core auth to “finish” self-host

### Three surfaces rule

| Surface | May grow when | Must not become |
|---------|---------------|-----------------|
| `RAGCore` library | Contract-stable retrieval features | Hosted platform |
| CLI | Agent/automation parity with library | Second product API |
| `serve` HTTP | Thin delegate to `RAGCore` | Ragie clone |

---

## Anti-goals (repeat until true)

- Hosted accounts, billing, admin UI, connector marketplace  
- Graph DSL / agent canvas in core  
- Eval/trace HTTP on `serve` in v1  
- Chasing RAGFlow star count or Haystack pipeline YAML  
- Git archaeology or contributor stat scrubbing (reset remote when ready)  

---

## Paste to a continuing agent

```text
Read docs/plans/one-repo-retrieval-engine-strategy.md.

User wants debloat + escape managed blackbox RAG—not platform sprawl.
Lane 1 self-host plumbing is done; insufficient alone.

Phase 0: complete the Shape Gate before editing. Compare at least three candidate shapes for the chosen journey.
Default priority (sole-maintainer): **`./scripts/dx_smoke.sh`** and Journey **A** maintenance. Then open roadmap items (**Q2a**, **B2.5**, **C3b**) only when named. See [ROUTING.md](ROUTING.md).
Phase 1: optional read-only subagents R1–R7 (max 8 findings each) only for the selected shape.
Pick ONE shape and ONE lane. Do not implement broad platform features.
Implement with proportional tests. Product over git. No co-author trailers.
Do not commit unless asked.
```
