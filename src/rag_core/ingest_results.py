"""Curated public ingest-result types.

Aggregates the small set of result and outcome dataclasses that callers (CLI,
integrations, the caller's gateway) need to consume the outcome of file,
directory, archive, and URL ingestion. Owner modules
(``rag_core.local_ingest_models``, ``rag_core.remote_ingest_results``) remain
the canonical definitions; this module is a stable import seam so consumers do
not depend on internal module layout.

This module is part of the public beta ``rag_core`` surface listed in
``https://kaanarici.github.io/rag-core/docs/stability``.
"""

from __future__ import annotations

from rag_core.local_ingest_models import (
    LocalIngestFailure,
    LocalIngestPlan,
    LocalIngestRequest,
    LocalIngestResult,
    LocalIngestSuccess,
    LocalManifestStatus,
)
from rag_core.remote_ingest_results import (
    RemoteManifestStatus,
    RemoteUrlIngestFailure,
    RemoteUrlIngestResult,
    RemoteUrlIngestSuccess,
)

__all__ = (
    "LocalIngestFailure",
    "LocalIngestPlan",
    "LocalIngestRequest",
    "LocalIngestResult",
    "LocalIngestSuccess",
    "LocalManifestStatus",
    "RemoteManifestStatus",
    "RemoteUrlIngestFailure",
    "RemoteUrlIngestResult",
    "RemoteUrlIngestSuccess",
)
