# Format Support

`rag-core` exposes two related support surfaces:

- converter support: `RAGCore.parse_bytes`, `prepare_bytes`, and `ingest_bytes` route bytes to a converter from MIME type or filename
- default local ingest support: `rag-core ingest`, `rag-core manifest`, and `local-search` accept file extensions that work without extra per-file OCR wiring

The support matrix below mirrors the converter registry used by the library and CLI.

Parser quality is visible in both parsed-document metadata and trace events. `parse_bytes`, `prepare_bytes`, and `ingest_bytes` attach structured `metadata["quality"]` with verdict, details, character count, meaningful-character ratio, mojibake ratio, text-to-page ratio, and page count when the converter reports a quality score. `parse.completed` events expose the same quality fields, plus OCR page counts, OCR page indices, and extraction ratio when the parser provides them.

Markdown-style headings add `section_path` and `section_title` metadata to prepared chunks. `Slide N` and `Sheet: ... (Rows A-B)` headings also add `slide_number`, `sheet_name`, and `row_range` locators. Converter-emitted figure metadata can add `figure_id` and `figure_caption` to the matching prepared chunk. These locators flow into indexing payloads and context-pack citations when present.

## Support Matrix

| Key | Formats | Support level | Default local ingest | OCR behavior | Extensions | MIME types |
| --- | --- | --- | --- | --- | --- | --- |
| `text` | Text and markdown | `first_party_stable` | yes | none | `.txt`, `.md`, `.markdown`, `.yaml`, `.yml`, `.toml`, `.rst`, `.adoc`, `.tex`, `.ini`, `.cfg`, `.conf`, `.env`, `.properties`, `.log` | `text/plain`, `text/markdown`, `text/x-markdown`, `text/yaml`, `text/x-yaml`, `application/x-yaml`, `application/toml`, `image/svg+xml` |
| `code` | Source code | `first_party_stable` | yes | none | `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.c`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.m`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.kts`, `.scala`, `.d`, `.jl`, `.ex`, `.exs`, `.erl`, `.clj`, `.groovy`, `.dart`, `.hs`, `.ml`, `.fs`, `.nim`, `.cr`, `.zig`, `.lua`, `.pl`, `.r`, `.sh`, `.bash`, `.zsh`, `.ps1`, `.bat`, `.cmd`, `.sql`, `.graphql`, `.gql`, `.proto`, `.tf`, `.tfvars`, `.hcl`, `.gradle`, `.cmake`, `.make`, `.mak`, `.css`, `.scss`, `.sass`, `.less`, `.vue`, `.svelte` | none |
| `html` | HTML | `first_party_stable` | yes | none | `.html`, `.htm` | `text/html`, `application/xhtml+xml` |
| `csv` | CSV and TSV | `first_party_stable` | yes | none | `.csv`, `.tsv` | `text/csv`, `text/tab-separated-values` |
| `json` | JSON, JSONL, and NDJSON | `first_party_stable` | yes | none | `.json`, `.jsonl`, `.ndjson` | `application/json`, `application/jsonl`, `application/ndjson`, `application/x-ndjson` |
| `xml` | XML | `first_party_stable` | yes | none | `.xml` | `application/xml`, `text/xml` |
| `pdf` | PDF | `first_party_beta` | yes | optional_pages | `.pdf` | `application/pdf` |
| `docx` | DOCX | `first_party_beta` | yes | quality_fallback | `.docx` | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` |
| `pptx` | PPTX | `first_party_beta` | yes | quality_fallback | `.pptx` | `application/vnd.openxmlformats-officedocument.presentationml.presentation` |
| `xlsx` | XLSX | `first_party_beta` | yes | none | `.xlsx` | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |
| `image` | Images | `first_party_beta` | no | required | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif` | `image/jpeg`, `image/png`, `image/gif`, `image/webp`, `image/bmp`, `image/tiff` |

## Important Boundaries

Images are recognized by the converter registry, but default local ingest excludes them because image extraction requires an OCR provider. Use the library API with an injected OCR provider when you want image ingestion.

Remote fetch defaults allow the registered non-image converter MIME types above. Image MIME types remain excluded from the default remote fetch allowlist because image parsing requires an injected OCR provider; use explicit fetch limits and an OCR-capable ingest path when fetching images intentionally.

Binary Office extensions `.doc`, `.ppt`, and `.xls`, plus their binary Office MIME types, are unsupported.

PDF support is beta because quality depends on document shape and local runtime tools. When PDF Inspector is available, `rag-core` uses it first and records page-level OCR routing signals. Without it, PyMuPDF is the fallback. PDF chunks prepared from `## Page N` headings carry `page_number` and `page_index` locator metadata through indexing payloads.

HTML support is local extraction, not browser rendering. Fetching, scraping, redirects, and rendered-page extraction belong to separate source-reader and fetcher paths.

ZIP archives are supported as an explicit source path through `ZipArchiveSourceReader`, `RAGCore.ingest_archive(...)`, and `rag-core ingest-archive`. Archive ingest reads only supported member formats, rejects unsafe member paths instead of extracting to disk, and enforces entry-count, per-entry byte, and total-byte limits. It is not part of default folder ingest, so local directory traversal and archive member traversal stay explicit.
