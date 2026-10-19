# Examples

Run Python examples from the repository root with `uv run python -m examples.NAME`.
Use `--help` where the example accepts arguments. Install the `pdf` or `office`
extra when your input documents need those parsers.

| Example | Purpose | Embeddings |
| --- | --- | --- |
| `ask_folder` | Search a real folder and print cited context | Real local models |
| `configured_retrieval` | Configure an engine and retrieve app documents | Configured provider |
| `configured_eval` | Evaluate an engine against labeled cases | Configured provider |
| `minimal_app` | Show the scoped Python API and prompt payload | Deterministic demo |
| `chatbot_context` | Pass retrieved context into an application-owned answer step | Deterministic demo |
| `retrieval_eval` | Demonstrate metric/report plumbing | Deterministic demo |
| `pdf_ocr_path` | Inspect PDF extraction and OCR requirements | No retrieval model |

```bash
uv run python -m examples.ask_folder examples/demo_corpus "How can invoices be paid?"
```

The small Markdown corpus is for a reproducible first run. It does not establish
retrieval quality on business documents. Demo embeddings prove orchestration,
not semantic search. See [evaluation](../docs/evaluation.md) before interpreting
scores or choosing a retrieval configuration.
