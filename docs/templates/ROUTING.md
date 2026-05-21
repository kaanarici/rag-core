# Work routing (agents)

**Local only** — generated from [templates/ROUTING.md](../templates/ROUTING.md). Do not commit.

**Repo context:** [../CONTEXT.md](../CONTEXT.md) · **Validation trust:** [../../tests/README.md](../../tests/README.md) · **Checklist:** local `roadmap.md` if present

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
| `continue`, `next`, `DX`, `first run`, `cleanup` | `./scripts/dx_smoke.sh`; Journey **A** — [../quickstart.md](../quickstart.md) |
| `embed`, `app`, `RAGCore` | Journey **B** — [../embed.md](../embed.md); `examples/embedded_service.py`, `examples/configured_retrieval.py` |
| `self-host`, `serve`, `API` | Journey **C** — [../self-host.md](../self-host.md); `./scripts/ci_self_host_smoke.sh` |
| `turbopuffer`, `managed vector` | Journey **V** — [../providers.md](../providers.md) |
| `regression`, `benchmarks`, `evals` | Journey **Q** — [../../tests/README.md](../../tests/README.md); `pytest -q tests/evals/test_retrieval_eval_pr.py` |
| `debloat`, `refactor tests` | Tie to a journey gate above; no free-floating work |

---

## Journey acceptance gates (pick one)

| Journey | Gate |
|---------|------|
| A | `./scripts/dx_smoke.sh` green |
| B | `examples/embedded_service.py` + [../embed.md](../embed.md) |
| C | `tests/test_runtime_http.py` + `./scripts/ci_self_host_smoke.sh` |
| V | `tests/test_turbopuffer_*.py` + `doctor --json` with `--vector-store turbopuffer` |
| Q | `provider_contract` + `integration` + PR retrieval regression |
| R | Research docs dated; no code required |

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
