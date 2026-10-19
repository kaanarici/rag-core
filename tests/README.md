# Verification boundaries

Run the [local gate](../docs/release.md) before making package or release claims.
Use the smallest relevant test files while iterating. Live provider checks need
a separate explicit request and credentials.

## What a passing check means

| Label | What to inspect |
| --- | --- |
| `integration` | Real local adapters, parser pipelines, HTTP fixtures, or processes. Some tests inject embeddings and establish wiring only. |
| `plumbing` | Fake models, scripted stores, demo embeddings, and orchestration. |
| `eval` | Fixed-vector ranking, real local semantic retrieval, or parser-quality checks. Parser scores are not ranking scores. |
| `meta` | Documentation, package contents, and repository shape. |
| `provider_contract` | Recorded SDK response shapes, not a live service. |
| `live` | Credentialed external providers, excluded from the local gate. |

`RecordingVectorStore`, `KeywordEmbeddingProvider`, and demo embeddings do not
establish semantic retrieval quality. Read the fixtures and providers before
interpreting a passing test.

## Retained checks

- Native PDF and Office parsing, source locators, OCR routing, archive safety,
  and quality floors.
- Scope filtering, document identity, partial-index repair, conditional
  publication, lost acknowledgements, tombstones, and failed replacements.
- Persistent local Qdrant, model-space checks, write retries, admission,
  cancellation, source byte limits, and shutdown.
- Context packing, prompt privacy, tracing, CLI configuration, and MCP plumbing.
- Real local semantic retrieval with near misses and scope controls, separately
  from deterministic ranking fixtures.
- The real TurboPuffer SDK against stateful HTTP fixtures. These establish wire
  and publication contracts, not hosted behavior.
- An installed-wheel consumer using real local embeddings, Python, CLI,
  persistence, and a scoped tool. Artifact checks compare the packaged code to
  the current checkout and reject retired modules and private planning files.

Neither tests nor source-hit scores establish answer correctness on arbitrary
user data, hosted capacity, a distributed failover SLA, or production latency.
Evaluate the application's [actual documents and expected evidence](../docs/evaluation.md).

Do not lower parser or semantic floors to make cleanup pass. When a failure only
appears after another test file, prove the smallest paired-file run before
changing product behavior. Shared mocks can leak between modules.

The generated PDF form regression checks readable checkboxes without OCR. The
existing private-use glyph test still requires OCR for broken font output.
Actual public-document repros and native model runs are separate evidence.

Licensed real-document fixtures retain their [sources](evals/pr_corpus/SOURCES.md)
and [license terms](evals/pr_corpus/LICENSE.md).
