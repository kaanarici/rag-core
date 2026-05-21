# Retrieval Core Hardening Risk Register

Status: Deep D1–D7 review at `db6c1fd` + remediation through current HEAD.

## Merge Verdict: NO BLOCK

| Item | Result |
| --- | --- |
| Baseline reviewed | `db6c1fd` → post-review fixes on `main` |
| Gates | ruff OK · mypy OK · **2080 passed**, 5 skipped |
| P0 open | **0** (after remediation) |

## Remediated this pass

- LangChain `context_pack_to_tool_output` → `context_pack_model_text()` (CI contract tests).
- `preview_manifest` hardlink rejection (parity with ingest).
- CLI `run_with_ready_core` / `doctor --check-store` infrastructure errors sanitized (no traceback).
- PDF single-PUA glyph → page `needs_ocr` (`private_use_count >= 1`).

## P1 Follow-Up (not fixed)

**D2 CLI:** Path redaction in `FileNotFoundError`; clearer Qdrant env+CLI conflict message.

**D5 telemetry:** Sidecar prefetch stale limit; `SearchStarted` vs `SearchPlanned` limit drift on custom transforms; missing `SearchPlanned` on transform failure; `SearchCompleted.requested_sidecar` semantics; rerank event `0.0` score range defaults.

**D6:** Memory store O(n) scan on empty allowlist; metadata capability preflight optional; cross-provider empty-allowlist contract test gap.

**D7/docs:** README/examples still use `as_text()` for demos.

**D3 deferred:** Symlink TOCTOU (concurrent swap).

## Discovery Log (deep pass)

| Scope | Depth | P0 found | Action |
| --- | --- | --- | --- |
| D1 packaging | deep | 0 | — |
| D2 CLI/config | deep | 0 | partial fix (store errors) |
| D3 ingest/security | deep | 0 | preview hardlink fixed |
| D4 parsers | deep | 0 | PUA OCR fixed |
| D5 search pipeline | deep | 0 | telemetry P1 |
| D6 providers | deep | 0 | — |
| D7 integrations | deep | 1 (CI) | langchain fallback fixed |
