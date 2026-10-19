# CLI

Run from the checkout after `uv sync --group dev`:

```bash
uv run rag-core search "How can invoices be paid?" examples/demo_corpus --json
uv run rag-core context "How can invoices be paid?" examples/demo_corpus --json
uv run rag-core eval examples/demo_corpus examples/eval_cases.jsonl --json
```

Path search, context, and eval use a persistent local hybrid index by default.
Use `--persist-dir ./indexes` to choose the storage root. `--mode dense` or
`--no-hybrid` selects a dense folder index. `--max-files` bounds the scan.
Truncated or failed scans do not prune unseen files.

Search JSON contains hits and source metadata. Context JSON contains packed
text and citations. Eval uses real local embeddings unless `--demo` explicitly
selects fake embeddings. A retrieval score is a measurement, not an execution
success condition.

## Ingest into a configured target

```bash
uv run rag-core add ./documents --collection docs --qdrant-location ./app-index --embedding-provider local --hybrid --json
uv run rag-core search "invoice payments" --collection docs --qdrant-location ./app-index --embedding-provider local --hybrid --json
```

`add` accepts a local path, HTTPS URL, or ZIP. It reports created, replaced,
reindexed, and unchanged outcomes. `add --dry-run` discovers files without
reading their contents or loading models.

A folder search against an explicit Qdrant target also adds or updates files.
Automatic removal of missing files requires `--manifest-dir` or
`RAG_CORE_MANIFEST_DIR`. The default local folder workflow supplies its own
persistent manifest. See [formats and source limits](formats.md) before fetching
URLs or ingesting archives.

CLI flags override environment variables. Configuration is shared with
`RAGCore.from_env`. The library does not load `.env` files. The
[environment example](../.env.example) shows common settings.

## Output, tracing, and optional tools

`add`, folder search, and eval return a nonzero exit status for ingest or execution
errors. A partial index is not reported as a complete success. Context preserves
its JSON shape and reports incomplete ingest on stderr.

`--trace-jsonl trace.jsonl`, also named `--events-jsonl`, writes local events.
See [tracing](tracing.md) for the output contract.

`rag-core doctor --json` reports local configuration. `rag-core demo --json`
uses fake embeddings and demonstrates wiring only. Install the `mcp` extra to
run the optional scoped MCP entry point. It uses the same engine.

Use `uv run rag-core --help` and each command's `--help` for current flags.
See [migration](migration.md) for removed commands and integrations.
