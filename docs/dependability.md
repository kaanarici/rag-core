# Dependable embedded retrieval

`Engine` owns work from source admission through publication and retrieval. The
application owns the job queue, backend deployment, access policy, and generated
answers. Embedded Qdrant is the local workflow; remote Qdrant and TurboPuffer are
the storage paths for larger deployments. No additional service is required by
rag-core.

## Resource ownership

```python
from dataclasses import replace
from rag_core import Config, EngineBusyError
from rag_core.core import Engine

config = Config.local(hybrid=True)
config = replace(config, ingest=replace(config.ingest, max_document_bytes=16 * 1024**2))
engine = Engine(config, max_concurrent_ingests=8,
                max_concurrent_searches=32, max_pending_operations=64)
```

These are per-engine limits. Direct ingestion and preparation reserve one ingest
slot. Folder and ZIP calls reserve their requested `max_concurrency` slots before
planning or reading, and retain those slots until the entire batch finishes.
Reservations larger than engine capacity fail before source work. This simple
reservation scheme can leave some slots idle during a small or slow batch.

Queries have separate capacity so a queued import does not block an otherwise
admissible search. Pending operations share a bounded admission budget; callers
receive `EngineBusyError` when it is full. Applications can retry with their own
deadline or put work in their existing job queue. Setting the pending limit to
zero provides immediate rejection when capacity is unavailable.

The default source limit is 64 MiB. Local reads request at most that limit plus
one byte; oversized sources fail before parsing or replacing a published
document. The same engine limit applies to supplied bytes, preparation, URL
bodies and ZIP entries, alongside their existing security/expansion limits.
Caller-allocated bytes, materialized source plans, extracted text, model state,
and injected providers have separate memory costs. Concurrency and source limits
are not a process RSS guarantee.

`close()` stops new admissions, rejects queued operations, drains active work,
and then closes storage once. Concurrent close callers share cleanup. A cancelled
close caller still waits for cleanup before receiving cancellation. Call close
from the application's lifetime owner, outside callbacks belonging to an active
engine operation.

Blocking file, parser, chunker and model workers remain owned until they finish,
including after repeated cancellation. Cancellation then propagates without
running later ingest stages. Python threads cannot be forcibly stopped: an
uncooperative parser can delay cancellation and shutdown. Command OCR has its
separate process timeout and reaping path. Injected providers must own any hidden
background work they start.

Use an async engine and its SDK clients on one event loop for their lifetime.
The local synchronous facade supports successive calls using successive loops;
its idle model/write limiters reset between loops. Concurrent cross-loop/thread
use is unsupported.

## Publication and visibility

Each document stages immutable chunks under a unique revision. Only a verified,
complete version can become its conditional head. Confirmed rejected writers
remove their own staged revision. An ambiguous acknowledgement preserves it,
because the backend may already have published that revision. TurboPuffer SDK
replays are reconciled against the head before interpreting a zero-row outcome.

Queries verify candidate revisions against document heads. Refill excludes entire
unpublished revisions without increasing the requested candidate budget. It makes
at most eight candidate reads and stops early if exclusion makes no progress.
`SearchVisibilityError` reports the read and excluded-version counts; it must not
be treated as an empty successful result. Unknown/crash-orphan revisions can
still exhaust that budget. Scope deletion or a rebuild removes those records;
the library does not guess which potentially active writers are safe to collect.

Collection deletion is fenced within one engine. Applications sharing a remote
scope across engines/processes must coordinate collection deletion and rebuilds.
Per-document conditional publication is the concurrency boundary; it does not
make an entire corpus update atomic.

## Remote storage contract

Remote Qdrant requires conditional writes (server 1.17+), the expected embedding
identity and dimensions, and compatible scope/version payload indexes. Readiness
reconciles missing indexes on both fresh and reopened collections and awaits
their completion. Permission/schema failures leave the adapter unready. Existing
unrelated indexes are preserved.

Managed publication requires positive replication and write-consistency factors
with acknowledgement by every configured replica. Unsupported configurations
fail readiness; rag-core does not silently change replication or weaken strict
mode. Candidate/head/document reads explicitly request `all` consistency, chunk
writes wait for completion, and head writes use strong ordering. This chooses
consistency over availability during replica failures. These settings are a
supported deployment requirement, not proof of an atomic cross-shard transaction
or a tested distributed failover SLA.

TurboPuffer uses native BM25/vector retrieval, atomic conditional head writes,
and its documented default strong reads. Backpressure remains enabled. The app
owns namespace selection, region and backend capacity; use independently routed
engines/stores for physical isolation or partitioning where needed.

Qdrant chunk upserts retry transient failures at most twice using the same IDs
and batch. Server retry hints up to eight seconds are respected; longer hints
fail for the caller to reschedule. Only HTTP 413 splits a payload, with a maximum
split depth of six. Splits share the original batch's transient retry budget.
Conditional publication is deliberately outside this automatic retry helper.
SDK/network request timeouts are separate from the bounded backoff delays.

## Prior art and decisions

| Source | Applied lesson |
| --- | --- |
| [QMD](https://github.com/tobi/qmd) | Keep persistent local retrieval callable through a small interface. A larger backend must not require a second product API. |
| [txtai indexing](https://neuml.github.io/txtai/embeddings/indexing/) | Prefer incremental updates and bounded embedding batches over rebuilding the corpus for every change. |
| [Qdrant bulk upload](https://qdrant.tech/documentation/manage-data/bulk-upload/) and [SDK uploader](https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/uploader/rest_uploader.py) | Batch writes, bound concurrent requests, and respect overload delays while replaying idempotent batches. |
| [Qdrant consistency](https://qdrant.tech/documentation/scaling/consistency-guarantees/) | Write acknowledgement, ordering, and read consistency are separate contracts; validate all three at their owners. |
| [TurboPuffer ingestion](https://turbopuffer.com/docs/ingestion) and [guarantees](https://turbopuffer.com/docs/guarantees) | Keep backend backpressure and atomic conditional writes; throughput tuning must preserve the publication protocol. |
| [AnyIO threads](https://anyio.readthedocs.io/en/stable/threads.html#running-a-function-in-a-worker-thread) and [Python cancellation](https://docs.python.org/3.12/library/asyncio-task.html#shielding-from-cancellation) | Retain ownership of running synchronous work through cancellation. A new runtime dependency is unnecessary for this behavior. |
| [Elasticsearch concurrency](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/optimistic-concurrency-control) and [search visibility](https://www.elastic.co/docs/manage-data/data-store/near-real-time-search) | A conditional update result and its subsequent search visibility are separate facts. |

These are design inputs, not claims that rag-core matches those systems'
operational scale. Local Qdrant tests prove local persistence/recovery; scripted
load tests prove resource bounds; TurboPuffer SDK HTTP fixtures prove wire
contracts. Real local-model evaluations and installed-wheel checks separately
exercise retrieval and packaging. Large-corpus throughput, remote ranking,
replica lag, partitions and failover still require deployment-specific trials.
