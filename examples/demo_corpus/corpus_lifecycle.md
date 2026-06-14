# Corpus Lifecycle

A corpus lifecycle starts when source files are parsed into documents, chunked, embedded, and written into an index.
After the first ingest, unchanged documents can be skipped by fingerprint while changed documents are reindexed.
Search queries then retrieve ranked chunks from the selected namespace and corpus.
