# Evaluate retrieval

Run the supplied small corpus with real local embeddings:

```bash
uv run rag-core eval examples/demo_corpus examples/eval_cases.jsonl --json
```

The JSON report separates execution failures from retrieval scores. Label your
own documents with expected IDs or keys before comparing configurations. The
[sample cases](../examples/eval_cases.jsonl) show the supported fields.

Use `rag_core.evals.load_cases`, `run_eval`, and `eval_report` against the
same engine your app uses. Label expected document IDs or keys before changing
retrieval settings, and include irrelevant questions and near misses.

`examples/retrieval_eval.py` demonstrates metric plumbing with deterministic
embeddings. The local semantic test uses real embeddings on a bounded corpus.
Neither proves quality on arbitrary documents. A correct source in the top five
also does not establish enough context to answer the whole question.

Keep ranking metrics, context support, latency, indexing cost, and generated
answer correctness as separate measurements. Do not replace real-document checks
with mocks or lower quality floors to make a simplification pass.
