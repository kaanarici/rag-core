# Public docs trim (peer-aligned)

**Goal:** OSS-library doc surface: README, examples, and the Fumadocs MDX site.

## Target tree (remote)

| File | Role |
|------|------|
| `README.md` | Pitch, install, smoke, embed snippet, examples table, eval note |
| `docs-site/content/docs/quickstart.mdx` | Sole scripted first-run path |
| `docs-site/content/docs/expectations.mdx` | Hit/context/trace contracts |
| `docs-site/content/docs/self-host.mdx` | Compose, serve, auth, config, HTTP table |
| `docs/self-host/openapi.yaml` | Machine contract |
| `docs-site/content/docs/providers.mdx` | Vector stores + custom providers + output-shape audit |
| `docs-site/content/docs/formats.mdx` | Format support |
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

- Alternative docs site stack
- Rewriting OpenAPI or examples
