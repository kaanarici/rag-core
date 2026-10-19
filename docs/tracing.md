# Traces

```bash
uv run rag-core search "How can invoices be paid?" examples/demo_corpus --trace-jsonl trace.jsonl --json
```

Events report ingest decisions, embedding work, retrieval channels, ranking,
context packing, failures, and correlation IDs. Default prompt and trace
projections omit private paths and document text where appropriate.

Applications can pass an `EventSink` directly. JSONL, logging, in-memory event
buffers, and optional OpenTelemetry sinks remain available. Sink failures are
observable; CLI trace-write failures fail the command instead of reporting a
successful traced run.
