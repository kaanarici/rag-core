# rag-core

A Python library for searching documents your app owns. Ingest a folder, keep a
persistent index, and retrieve evidence with original-file locations and citations.

rag-core handles parsing, chunking, hybrid search, context packing, and updates.
Your app owns access rules, agents, and generated answers. This is an early,
unpublished library. Evaluate it on your own documents before shipping.

## Try it

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kaanarici/rag-core.git
cd rag-core
uv sync --group dev
uv run rag-core search "How can invoices be paid?" examples/demo_corpus --json
```

The result contains matching excerpts and source metadata. The first search
downloads local embedding and BM25 models. Later searches run locally and reuse
the index. Replace `examples/demo_corpus` with your folder.

For a Python app:

```python
from rag_core.easy import index

with index("examples/demo_corpus") as docs:
    print(docs.context("How can invoices be paid?"))
    docs.refresh()
```

Opening an index refreshes the folder. Unchanged documents skip parsing and
embedding. Edits replace stored versions, and removed files leave search.
Failed replacements preserve the last good version. Incomplete scans do not
prune unseen documents.

Use `persist_dir="./indexes"` to choose the storage root. The default is
`~/.cache/rag-core`, respecting `XDG_CACHE_HOME`. Each source folder and search
mode has its own index. Embedded Qdrant supports one open process per index.

## Embed it

Use `RAGCore` to bind an application's tenant, collection, and document scope.
It returns typed evidence, builds cited context, and exposes a plain async tool.
See the [Python API](docs/api.md) for a complete example.

Qdrant, local FastEmbed embeddings, and BM25 are built in. TurboPuffer is an
optional backend. Models, rerankers, OCR, and stores can be passed directly as
Python objects to `Engine`. There is no provider registration step.

For an installation outside this checkout, add the extras your files need:

```bash
uv pip install 'rag-core[pdf,office] @ git+https://github.com/kaanarici/rag-core.git'
```

Other extras are `turbopuffer`, `openai`, `mcp`, `opentelemetry`, and `html`.
The library is not published to PyPI. Pin a commit when installing from Git.

## Know the limits

PDF and Office extraction depends on document layout. Scans and images need an
injected OCR provider. Visual meaning from charts and images is not retrieved.
A source hit does not prove that an answer is correct.

Hosted TurboPuffer behavior and production-scale throughput are not established
by the local tests. Applications own deployment capacity and access control.
`HF_HUB_OFFLINE=1` can prevent BM25 startup even when model files are cached.

## Documentation

| Task | Guide |
| --- | --- |
| Use scoped retrieval, custom providers, or TurboPuffer | [Python API](docs/api.md) |
| Search, ingest, evaluate, or trace from a terminal | [CLI](docs/cli.md) |
| Check file support, OCR, and source locators | [Formats](docs/formats.md) |
| Understand ownership and the code layout | [Architecture](docs/architecture.md) |
| Understand writes, recovery, resource limits, and remote storage | [Dependability](docs/dependability.md) |
| Upgrade an older checkout or index | [Migration](docs/migration.md) |
| Measure retrieval quality | [Evaluation](docs/evaluation.md) |
| Inspect retrieval events | [Tracing](docs/tracing.md) |
| Run or copy a working example | [Examples](examples/README.md) |
| Contribute and check a package locally | [Development and release](docs/release.md) |

Documentation lives in this repository. Verification boundaries are described in
[tests/README.md](tests/README.md). Licensed evaluation sources are listed with
their [corpus](tests/evals/pr_corpus/SOURCES.md).
