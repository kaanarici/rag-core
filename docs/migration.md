# Migrating an older checkout

This update narrows rag-core to an embedded Python retrieval library. It removes
old integrations and changes managed index storage. Keep your original documents
and old index until you have checked retrieval against the new one.

## Rebuild the index

Automatic folder indexes use a separate `v2` cache directory. Opening the folder
creates or refreshes that index and preserves the older cache.

For an explicit local directory, remote Qdrant collection, or TurboPuffer
namespace, choose a fresh target and ingest the original documents again. Old
unversioned records remain readable, but managed ingest refuses to mix them with
the new document-head format. TurboPuffer also requires the current adapter
schema and namespace embedding identity.

```python
from rag_core.easy import index

with index("./documents", persist_dir="./indexes-v2") as docs:
    print(docs.context("A question with a known answer in these documents"))
```

Compare source evidence before directing your app to the new index. Keep the old
configuration and index for rollback. Do not copy stored vectors between formats.
Changing the embedding model or dimensions also requires a fresh index.

The default processing recipe is now `rag_core_processing_v9`. A refresh
reprocesses earlier recipes, including v8, to apply corrected PDF form-symbol
classification. Valid checkboxes and box-drawing characters no longer force OCR.
If you set `processing_version` yourself, increment it when changing extraction
or chunking behavior.

## Update application code

Use `rag_core.easy.index` for a local folder. Use `RAGCore` for scoped application
retrieval. Keep advanced ownership in `Engine`. The [Python guide](api.md) shows
both scoped retrieval and custom provider injection.

| Removed path | Replacement |
| --- | --- |
| Hosted HTTP runtime, `serve`, and Docker deployment | Call `Engine` from the application's existing service |
| LangChain and OpenAI Agents wrappers | Wrap the plain async `rag.tool()` function in your SDK |
| Provider registries and vendor factories | Pass model, reranker, OCR, and store objects directly to `Engine` |
| pgvector and generic memory adapters | Embedded or remote Qdrant, optional TurboPuffer, or an injected store |
| Semantic and AST chunkers | Built-in text chunking or caller-prepared chunks |
| Query plans, profiles, and stage graphs | `Engine.search(mode="dense" | "hybrid" | "lexical")` |
| Manifest preview/reconciliation commands | `Engine.add` outcomes and one ingest journal |
| URL-batch workflow | Ordinary Python iteration over `Engine.add_url` |
| `Config.from_cli` | `Config.local`, `Config.qdrant`, explicit `Config`, or `RAGCore.from_env` |
| PDF Inspector adapter | Native PyMuPDF extraction with optional injected OCR |
| Separate lexical index and built-in caches | The selected backend's retrieval and persistent index |

Source files with binary Office extensions `.doc`, `.ppt`, and `.xls` still need
conversion to OOXML or another supported format. Removing wrappers does not add
those formats or image understanding.

## Check output handling

Search summary traces use schema 2. Retired plan, cache, and sidecar fields and
event types are removed. Update consumers that decode those fields.

Folder search, `add`, and eval report ingest/execution failures with a nonzero
exit status. Partial ingest is not a successful complete index. Context JSON
keeps its canonical shape and reports incomplete ingest on stderr. Eval scores
are measurements, not an automatic quality gate.

Remote Qdrant requires server 1.17+ and the consistency configuration described
in [dependability](dependability.md). Embedded storage supports one open process
per index. Applications must coordinate cross-process scope deletion or rebuilds.

No package was published as part of this update. Install from the checkout or
pin a Git commit. [Local checks](release.md) exercise the installed artifact.
