from __future__ import annotations

import argparse
from pathlib import Path


def run_serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "rag-core serve requires the runtime extra: uv sync --extra runtime"
        ) from exc

    from rag_core.core import RAGCore
    from rag_core.core_models import RAGCoreConfig
    from rag_core.runtime.app import create_app

    config = RAGCoreConfig.from_cli(args)
    app = create_app(
        config=config,
        core_factory=RAGCore,
        ingest_roots=tuple(Path(path) for path in args.ingest_root),
        job_db_path=Path(args.job_db_path),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
