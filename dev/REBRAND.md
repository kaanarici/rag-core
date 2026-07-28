# Local display identity

The **product** stays `rag_core` / `rag-core`. That is intentional: embedders import a stable package name. The old `RAGCore` facade name is a deprecated alias for `Engine`.

For a **local experiment**, keep display-name notes in `dev/project_identity.toml` or an untracked `dev/project_identity.local.toml`. Do not rewrite product docs, compose labels, package metadata, imports, or tests from a script.

The old local rebrand scripts were removed because they encouraged product-identity drift. If a fork needs a different display identity, make that fork-specific change explicitly and keep `src/rag_core/`, tests, and `pyproject.toml` on the stable package identity unless the package itself is intentionally renamed.
