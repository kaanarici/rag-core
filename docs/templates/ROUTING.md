# Work routing (agents)

**Local only** — generated from [templates/ROUTING.md](../templates/ROUTING.md). Do not commit.

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
| `embed`, `app`, `managed RAG`, `Ragie` | Journey **B** — [../embedding/production-guide.md](../embedding/production-guide.md) |
| `self-host`, `serve`, `API` | Journey **C** — [../self-host/quickstart.md](../self-host/quickstart.md); `./scripts/self_host_smoke.sh` after compose changes |
| `turbopuffer`, `managed vector` | Journey **V** — [../providers/vector-stores.md](../providers/vector-stores.md) |
| `quality`, `benchmarks`, `evals` | Journey **Q** — Q2a on [roadmap.md](../../roadmap.md) |
| `debloat`, `refactor tests` | Tie to a journey gate above; no free-floating work |

---

## Journey acceptance gates (pick one)

| Journey | Gate |
|---------|------|
| A | `./scripts/dx_smoke.sh` green |
| B | `examples/embedded_service.py` + embed docs accurate vs `RAGCore` lifecycle |
| C | `tests/test_runtime_http.py` + `./scripts/self_host_smoke.sh` |
| V | `tests/test_turbopuffer_*.py` + `doctor --json` with `--vector-store turbopuffer` |
| Q | Named corpus + eval workflow (roadmap) |

---

## Slice size rule

One slice = **one journey + one shape + one acceptance gate**.

---

## Hard stops

Stop if you are about to:

- Add auth, billing, admin UI, connector marketplace, agent canvas, or hosted accounts
- Add a graph DSL or orchestration runtime
- Bypass `RAGCore` in the runtime layer
- Commit files under `docs/plans/`, `docs/research/`, or root `AGENTS.md` / `MISSION.md`
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

Daily trust: `./scripts/dx_smoke.sh` only.
