# Local rebrand (display only)

The **product** stays `rag_core` / `rag-core` / `RAGCore`. That is intentional: embedders import a stable package name.

For a **local experiment** (fork README, compose service labels, doc tone):

1. Edit `dev/project_identity.toml` `[display].name` or copy `dev/project_identity.local.toml.example` → `dev/project_identity.local.toml`.
2. Run `./scripts/local_rebrand.sh your-new-name` to rewrite human-facing markdown and compose labels.
3. Run `./scripts/brand_check.sh` to confirm README title matches identity.

`scripts/local_rebrand.sh` never touches `src/rag_core/`, `tests/`, or `pyproject.toml`.
