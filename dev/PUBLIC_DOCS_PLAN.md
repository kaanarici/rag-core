# Public docs trim (peer-aligned)

**Goal:** OSS-library doc surface — README + examples + a few reference pages, not an in-repo docs site.

## Target tree (remote)

| File | Role |
|------|------|
| `README.md` | Pitch, install, smoke, embed snippet, examples table, eval note |
| `docs/quickstart.md` | Sole scripted first-run path |
| `docs/expectations.md` | Hit/context/trace contracts |
| `docs/self-host.md` | Compose, serve, auth, config, HTTP table |
| `docs/self-host/openapi.yaml` | Machine contract |
| `docs/providers.md` | Vector stores + custom providers + output-shape audit |
| `docs/parsing/formats.md` | Format support (CLI errors link here) |
| `dev/DESIGN.md` | Compressed architecture principles (was ADRs) |

## Remove from remote

- `docs/README.md`, `docs/embedding/*`, `docs/integrations/*`, `docs/evals/*`, `docs/naming.md`
- `docs/adr/*`, `docs/self-host/{quickstart,auth,config}.md`
- `docs/providers/*` (merged into `providers.md`)
- `roadmap.md` → gitignored maintainer checklist (local only)

## Tests / packaging

- `tests/test_provider_docs.py`, `test_packaging_manifest.py`, `scripts/check_dist_artifacts.py` → new paths
- Eval + Vercel contract phrases → README + examples

## Out of scope

- External docs site (Mintlify/RTD)
- Rewriting OpenAPI or examples
