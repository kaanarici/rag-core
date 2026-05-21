# Retrieval Core Hardening Risk Register

Status: May 21, 2026 merge-closure triage for the hardening diff against `main`.

## P0 Merge Blockers

All P0 items are fixed in the current diff:

- Model-facing retrieval tool payloads no longer fall back to private `document_key` or `document_id` values when titles are absent.
- Empty `corpus_ids` now behaves as an empty allowlist through search requests and first-party vector stores.
- Explicit CLI flags now override invalid env-backed numeric and boolean config values.
- Local file ingest and URL-file ingest reject symlink and multi-link file paths before reading source content.
- Reindexed existing documents no longer report failure after a successful final-manifest retry, and restored filenames are derived from source document keys without leaking fingerprint suffixes.

## P1 Follow-Up Issues

- Pipeline event ordering can still emit `SearchStarted.limit` before transform-injected plans settle.
- Rerank failure and empty-rerank traces still need unknown score ranges instead of `0.0..0.0`.
- Metadata filter capability checks should be enforced before adapter calls instead of relying on adapter translation failures.
- Architecture pressure validation is narrower than the product surface now covered by public exports, CLI, traces, evals, and adapters.
- `core_ingest.py` remains a hotspot and should be split only after the merge diff is stable.

## Deferred Named Risks

- Existing-document sidecar sync failures can leave lexical sidecar state degraded after vector indexing succeeds. The current behavior raises instead of pretending success, but automatic sidecar repair needs a dedicated design because vector stores do not expose chunk payload reads.
- Single-file local ingest still constructs runtime before some source-read failures are surfaced. This is a developer-experience risk, not a data-integrity blocker.
- Broad public surface cleanup under `rag_core.search` is deferred to avoid renaming churn inside this merge.

## Rejected

- Stale review notes about bbox payload shape, duplicate ZIP members, and missing model payload schemas were rejected after current-tree checks; the current diff already covers those contracts.
