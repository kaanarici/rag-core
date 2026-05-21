# Work routing (agents)

**Use this file before editing.** Full thesis and packets live in [one-repo-retrieval-engine-strategy.md](one-repo-retrieval-engine-strategy.md).

**Repo context:** [../CONTEXT.md](../CONTEXT.md) · **Checklist:** [../../roadmap.md](../../roadmap.md)

---

## Shape gate (required before substantial work)

State in your first response:

```markdown
Selected journey:
Candidate shapes compared:
Winning shape:
User-visible promise:
Files I expect to touch:
Acceptance gates:
Out of scope:
```

If you cannot fill this, run read-only discovery (`repo-intake`, `decide`) — do not “just clean up” or expand scope.

---

## Default routing (sole-maintainer repo)

Run `./scripts/dx_smoke.sh` before adding unrelated scope.

| User says | Do |
|-----------|-----|
| `continue`, `next`, `DX`, `first run`, `cleanup` | `./scripts/dx_smoke.sh`; Journey **A** maintenance; [../quickstart.md](../quickstart.md) |
| `embed`, `app`, `managed RAG`, `Ragie` | Journey **B** — [../embedding/production-guide.md](../embedding/production-guide.md); B2.5 proof only if you need embed evidence |
| `self-host`, `serve`, `API` | Journey **C** — [../self-host/quickstart.md](../self-host/quickstart.md); `./scripts/self_host_smoke.sh` after compose changes |
| `turbopuffer`, `managed vector` | Journey **V** — landed; [../research/turbopuffer-landscape.md](../research/turbopuffer-landscape.md) |
| `quality`, `benchmarks`, `evals` | Journey **Q** — Q2a open; [../research/retrieval-benchmark-corpus.md](../research/retrieval-benchmark-corpus.md) |
| `research`, `competitors`, `OSS` | Journey **R** — [../research/](../research/) |
| `debloat`, `refactor tests` | Tie to a journey gate above; no free-floating work |

**When you need external credibility** (not for day-to-day maintenance): Journey **Q2a** — named public corpus + CI eval tier.

---

## Journey acceptance gates (pick one)

| Journey | Gate |
|---------|------|
| A | `./scripts/dx_smoke.sh` green |
| B | `examples/embedded_service.py` + embed docs accurate vs `RAGCore` lifecycle |
| C | `tests/test_runtime_http.py` + `./scripts/self_host_smoke.sh` |
| V | `tests/test_turbopuffer_*.py` + `doctor --json` with `--vector-store turbopuffer` |
| Q | Named corpus doc + eval workflow (not started) |
| R | Research docs dated; no code required |

---

## Slice size rule

One slice = **one journey + one shape + one acceptance gate**.

**Allowed:** “Journey C / fix `/health/ready` when Qdrant down; `test_runtime_http.py`.”

**Disallowed:** “Improve docs and tests.” “Make it like RAGFlow.” “Run a broad audit.”

---

## Hard stops

Stop if you are about to:

- Add auth, billing, admin UI, connector marketplace, agent canvas, or hosted accounts
- Add a graph DSL or orchestration runtime
- Bypass `RAGCore` in the runtime layer
- Execute anything under [archive/](archive/)
- Spawn broad subagents before journey + shape are chosen

---

## Canonical validation

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src tests examples
uv run pytest -q
uv build && uv run python scripts/check_dist_artifacts.py && uv run python scripts/wheel_smoke.py
```

Packaging-only changes: wheel smoke path above. Daily trust: `./scripts/dx_smoke.sh` only.
