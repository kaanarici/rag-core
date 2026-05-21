# Provider Output Shapes

Checked against provider docs on 2026-05-20.

This document records the output shapes that `rag-core` expects from first-party provider adapters and checks them against linked provider documentation. It is a contract audit, not a benchmark.

## Internal Contracts

| Surface | `rag-core` shape | Contract check |
| --- | --- | --- |
| Dense embeddings | one finite `list[float]` per input, ordered by input position or provider index | embedding result validation |
| Rerankers | `RerankResult(index, score, text)` with indexes relative to the submitted document list | reranker result validation |
| Sparse embeddings | `SparseVector(indices: list[int], values: list[float])` with matching lengths | sparse vector validation |
| Qdrant search | `ScoredPoint` rows with `id`, `score`, and payload | Qdrant payload validation |
| TurboPuffer search | rows with `id`, attributes, and optional `$dist` | TurboPuffer row validation |
| Command OCR | command JSON with `markdown`, provider/model names, merge mode, processed pages, and metadata | OCR command result validation |

## Provider Matrix

| Provider | Provider fields checked | Adapter fit |
| --- | --- | --- |
| OpenAI embeddings | The embeddings response has `data[]` rows with `index` and `embedding`; `embedding` is a list of floating point numbers. `text-embedding-3-small` defaults to 1536 dimensions and `text-embedding-3-large` defaults to 3072, with a `dimensions` parameter for `text-embedding-3` and later models. Source: [OpenAI embeddings API reference](https://platform.openai.com/docs/api-reference/embeddings). | Fits. `OpenAIEmbeddingProvider` sends `dimensions` only for known models that support it and uses indexed validation. Unknown models must return the configured vector length; the adapter does not send provider-specific dimension overrides for them. |
| Voyage embeddings | Voyage documents 1024 default dimensions and selectable 256, 512, 1024, or 2048 dimensions for `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-4-nano`, `voyage-code-3`, `voyage-3-large`, `voyage-3.5`, and `voyage-3.5-lite`. `voyage-finance-2` and `voyage-law-2` are fixed at 1024 dimensions; `voyage-code-2` is fixed at 1536 dimensions. The Python client returns a list of embedding vectors. Source: [Voyage embeddings](https://docs.voyageai.com/docs/embeddings). | Fits. `VoyageEmbeddingProvider` derives `output_dimension` support from model metadata, sends it only for known flexible-dimension models, and validates ordered vectors. Unknown models must return the configured vector length; the adapter does not send `output_dimension` for them. |
| ZeroEntropy embeddings | `models/embed` returns `results[]` with an `embedding` field, in the same order as the text input. `zembed-1` supports 2560, 1280, 640, 320, 160, 80, or 40 dimensions and `float` or `base64` encodings. The upstream response also includes usage fields such as token and byte totals, which the current `rag-core` provider protocol does not expose. Source: [ZeroEntropy embed](https://docs.zeroentropy.dev/api-reference/models/embed). | Fits for `float` output. The adapter reads `results[]` and validates ordered vectors. It ignores usage metadata and does not support the base64 output path. |
| Cohere rerank | The v2 response has `results[]` rows with `index` and `relevance_score`, sorted by relevance. The upstream response also includes envelope metadata such as response id and billed units, which the current `rag-core` reranker protocol does not expose. Source: [Cohere rerank API](https://docs.cohere.com/v2/reference/rerank). | Fits. `CohereReranker` sends `top_n` and uses indexed result validation. It ignores envelope metadata. |
| Voyage rerank | The API can return `index` and `relevance_score`; the API reference says `document` is included only when requested. Source: [Voyage reranker API](https://docs.voyageai.com/reference/reranker-api-1). | Fits. `VoyageReranker` sends `top_k` and reads `index` plus `relevance_score`. |
| ZeroEntropy rerank | `models/rerank` returns `results[]` rows with `index` and `relevance_score`; `top_n` limits returned rows. The upstream response also includes latency and usage fields, which the current `rag-core` reranker protocol does not expose. Source: [ZeroEntropy rerank](https://docs.zeroentropy.dev/api-reference/models/rerank). | Fits. The adapter sends `top_n` and parses `index` plus `relevance_score`. It ignores latency and usage metadata. |
| FastEmbed sparse | FastEmbed sparse outputs are `SparseEmbedding` objects with `values` and `indices`; corresponding entries represent token weights and vocabulary indexes. Source: [Qdrant FastEmbed SPLADE](https://qdrant.tech/documentation/fastembed/fastembed-splade/). | Fits. `FastEmbedSparseEmbedder` validates provider result cardinality before converting `result.indices` and `result.values` into `SparseVector`. |
| Qdrant Query API | Qdrant query points supports sparse vectors as `values` plus `indices`, dense vectors, prefetch, fusion, and formula boost. The response has `result.points[]` with `id`, `score`, `payload`, and optional vector/order fields. Source: [Qdrant query points](https://api.qdrant.tech/api-reference/search/query-points/). | Fits for rag-core-managed collections, where point ids are UUID strings derived by the adapter. The Qdrant adapter translates dense, sparse, RRF, DBSF, weighted RRF, MMR, and boost plans and materializes `ScoredPoint` rows. |
| TurboPuffer query | `POST /v2/namespaces/:namespace/query` supports ANN, kNN, BM25, SparseKNN, filters, ordering, aggregation, and multi-queries. `rows[]` include `id`; `$dist` is present for ranking functions such as ANN and BM25. Source: [TurboPuffer query](https://turbopuffer.com/docs/query). | Base contract fits. The current adapter intentionally supports dense ANN plus filters, count-backed document lookup, deletes, and health. Missing or malformed document-count aggregation fails closed. Sparse, hybrid, MMR, boost, and multi-query plans remain unsupported until covered by live-backed contract work. |
| Mistral OCR | OCR responses contain `pages[]`; each page has an `index` and `markdown`. Source: [Mistral OCR API](https://docs.mistral.ai/api/endpoint/ocr). | Fits. The command adapter uploads files, calls `/v1/ocr`, collects page markdown, and maps documented one-based page indexes to zero-based internal page indexes. |
| Gemini command OCR | `generateContent` accepts inline media through `inline_data` parts and returns candidate content parts with text. Source: [Gemini generateContent](https://ai.google.dev/api/generate-content). | Fits as a generic multimodal conversion adapter, not a dedicated OCR API. Page selection is reported unsupported and ignored deliberately. |

## Findings

1. ZeroEntropy was the only adapter mismatch found in this pass. Embed responses use `results[]`, not `data[]`. Rerank responses use `index` and `relevance_score`, not `document` and `score`, and accept `top_n`. The adapters and targeted tests now cover those shapes.

2. Embedding adapters are structurally aligned, but they depend on dimension tables staying current. Unknown models now fail closed by omitting provider-specific dimension override fields and validating the returned vector length. Voyage metadata was updated to match current `output_dimension` support for `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, and fixed 1536-dimensional `voyage-code-2`. The other high-risk drift point is ZeroEntropy `zembed-1` dimensions.

3. Qdrant support is broader than the base vector-store contract and is accurately represented by its adapter capabilities. It is the right backend for first-party hybrid, sparse, MMR, and boost experiments.

4. TurboPuffer has documented support for more query forms than the current adapter exposes. That is not a bug because the adapter fails closed, but it is the main future expansion lane: SparseKNN, BM25, and multi-query fusion can become first-party only once contract tests and live-backed verification are added.

5. OCR providers have less strict typed contracts than the search providers. Mistral has a documented OCR response shape and maps cleanly. Gemini is a prompt-backed `generateContent` adapter, so treat it as a convenience conversion path unless a stricter structured-output contract is added.

## Follow-Up Work

- Add live conformance scripts for provider adapters that can run when API keys are present. These should prove shape only: count, dimension, index mapping, finite scores, and secret-safe errors.
- Extend the TurboPuffer adapter only through explicit query-plan contract work. Do not infer feature parity from the provider docs without tests.
- Keep provider dimension metadata next to a source link and update it as a deliberate slice when providers add or deprecate models.
