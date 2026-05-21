# Test trust model

This repo has a large test suite. Do not treat every passing test as equal
evidence that retrieval works in production.

## Validation ladder

| Tier | Examples | What it proves |
| --- | --- | --- |
| Meta | `test_provider_docs.py`, `test_doc_templates.py`, `test_packaging_manifest.py` | Docs, packaging, exports, and repo hygiene stay aligned. Not runtime proof. |
| Unit / fake-boundary | `RecordingVectorStore`, fake embedders, fake rerankers | Orchestration, errors, sanitization, and call shape. Not ranking quality. |
| Adapter contract | `test_vector_store_contract.py`, Qdrant helper tests | Store API behavior and wire translation for covered backends. |
| Local integration | `test_local_smoke.py`, `test_runtime_http.py`, `test_retrieval_golden_path.py` | Real `RAGCore` paths over local in-memory services or subprocesses. |
| Eval | `tests/evals/baseline/` | Regression on a synthetic labelled corpus. Not a product benchmark. |
| Smoke scripts | `scripts/dx_smoke.sh`, `scripts/ci_self_host_smoke.sh`, `scripts/wheel_smoke.py` | User-facing CLI, HTTP, and installed-wheel journeys. |
| Live | `@pytest.mark.live` tests | External providers or services; skipped without credentials. |

## Rules for agents

- Check which fixture a test uses before citing it as proof. `RecordingVectorStore`
  returns scripted hits and is never a vector-store conformance substitute.
- `@pytest.mark.eval_harness` tests prove metric or rerank plumbing. They are not
  retrieval-quality gates.
- `@pytest.mark.eval` is reserved for real eval paths with a corpus and cases.
- Use `@pytest.mark.meta` for string, docs, export, packaging, and repo-shape tests.
- Use `@pytest.mark.integration` when a test starts a subprocess, uses Starlette
  `TestClient`, or exercises real local Qdrant through `RAGCore`.
- Do not claim TurboPuffer parity from Qdrant contract tests. TurboPuffer has
  adapter and optional live tests unless explicitly added to shared contracts.
- CI architecture pressure is a report unless a test or script asserts bounds.

## Known validation gaps

Address these before claiming the project is production-proven:

- Add TurboPuffer to a shared vector-store contract path, or document it as
  adapter-tested only.
- Add provider live conformance scripts or remove wording that implies they
  exist.
- Strengthen evals with a less synthetic corpus, hard negatives, and realistic
  floors. The baseline eval uses local Qdrant, but the labelled corpus is still
  deliberately small and keyword-friendly.
- Keep OpenAPI route drift and self-host smoke in CI when runtime routes change.
