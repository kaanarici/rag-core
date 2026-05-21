# Agent doc templates

These files are **tracked** so clones stay small. **Generated copies are gitignored** and must not be pushed.

## One-time setup (each machine)

```bash
./scripts/setup_agent_docs.sh
```

Creates (if missing):

| Template | Local output |
|----------|----------------|
| `AGENTS.md` | `/AGENTS.md` and `docs/AGENTS.md` |
| `CONTEXT.md` | `docs/CONTEXT.md` |
| `MISSION.md` | `/MISSION.md` |
| `ROUTING.md` | `docs/plans/ROUTING.md` |

Edit the local copies freely. Refresh from templates after a deliberate template update:

```bash
./scripts/setup_agent_docs.sh --force
```

## Remote repo policy

**On `origin`:** product docs only (`quickstart`, `expectations`, `providers.md`, `self-host.md`, `openapi.yaml`, `parsing/formats.md`). Maintainer checklist: local `roadmap.md` (gitignored).

**Never push:** `docs/plans/`, `docs/research/`, root `AGENTS.md` / `MISSION.md` / `CONTEXT.md`, or local agent layers (see root `.gitignore`).
