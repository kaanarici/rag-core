"""Translate CLI flags into the same config decoder used by RAGCore.from_env."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rag_core.config.env_config import build_config_from_env, env_bool
from rag_core.core_models import Config
from rag_core.local import local_config

_ENV_FLAGS = {
    "hybrid": "RAG_CORE_HYBRID",
    "embedding_provider": "RAG_CORE_EMBEDDING_PROVIDER",
    "embedding_model": "RAG_CORE_EMBEDDING_MODEL",
    "embedding_dimensions": "RAG_CORE_EMBEDDING_DIMENSIONS",
    "embedding_batch_size": "RAG_CORE_EMBEDDING_BATCH_SIZE",
    "qdrant_url": "RAG_CORE_QDRANT_URL",
    "qdrant_location": "RAG_CORE_QDRANT_LOCATION",
    "qdrant_collection": "RAG_CORE_QDRANT_COLLECTION",
    "qdrant_api_key": "RAG_CORE_QDRANT_API_KEY",
    "dimension_aware_collection": "RAG_CORE_QDRANT_DIMENSION_AWARE_COLLECTION",
    "processing_version": "RAG_CORE_PROCESSING_VERSION",
    "manifest_dir": "RAG_CORE_MANIFEST_DIR",
}


def config_from_args(args: argparse.Namespace, *, path: Path | None = None) -> Config:
    env = dict(os.environ)
    for flag, name in _ENV_FLAGS.items():
        if (value := getattr(args, flag, None)) is not None:
            env[name] = str(value)
    # An explicit target replaces a target inherited from the environment.
    if (
        getattr(args, "qdrant_url", None) is not None
        and getattr(args, "qdrant_location", None) is None
    ):
        env.pop("RAG_CORE_QDRANT_LOCATION", None)
    elif (
        getattr(args, "qdrant_location", None) is not None
        and getattr(args, "qdrant_url", None) is None
    ):
        env.pop("RAG_CORE_QDRANT_URL", None)
    target = bool(
        (env.get("RAG_CORE_QDRANT_URL") or "").strip()
        or (env.get("RAG_CORE_QDRANT_LOCATION") or "").strip()
    )
    persist_dir = getattr(args, "persist_dir", None)
    if persist_dir is not None and (path is None or target):
        raise ValueError(
            "--persist-dir requires a local path without a configured Qdrant target"
        )
    defaults = None
    if not target:
        if path is not None:
            defaults = local_config(
                path,
                persist_dir=persist_dir,
                hybrid=env_bool(
                    env, "RAG_CORE_HYBRID", getattr(args, "mode", None) != "dense"
                ),
            )
        elif args.command == "doctor":
            defaults = Config.local()
    config = build_config_from_env(environ=env, defaults=defaults)
    if getattr(args, "mode", None) in {"hybrid", "lexical"} and not config.hybrid:
        raise ValueError(
            "hybrid and lexical modes require --hybrid (or omit --no-hybrid for local folders)"
        )
    return config
