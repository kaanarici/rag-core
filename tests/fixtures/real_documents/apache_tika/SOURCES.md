# Apache Tika Real-Document Fixtures

Small real-world parser fixtures copied from Apache Tika's public test resources.
They are used to prove that `rag-core` parser contracts handle externally authored
PDF, DOCX, PPTX, and XLSX files, not only files generated inside this test suite.

Copied from upstream `main` at commit
`da1801a84c4136850fb1d9bba985ddd0ec275193` on 2026-05-21.

| File | Source |
| --- | --- |
| `testPDF.pdf` | `apache/tika`, `tika-detectors/tika-detector-magika/src/test/resources/test-documents/testPDF.pdf` |
| `testWORD.docx` | `apache/tika`, `tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-microsoft-module/src/test/resources/test-documents/testWORD.docx` |
| `testPPT.pptx` | `apache/tika`, `tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-microsoft-module/src/test/resources/test-documents/testPPT.pptx` |
| `testEXCEL.xlsx` | `apache/tika`, `tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-microsoft-module/src/test/resources/test-documents/testEXCEL.xlsx` |

Upstream repository: <https://github.com/apache/tika>
