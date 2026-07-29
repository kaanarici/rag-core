from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from rag_core.runtime_defaults import LOOPBACK_HOSTS


_logger = logging.getLogger(__name__)


def run_serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "rag-core serve requires the runtime extra: uv sync --extra runtime"
        ) from exc

    from rag_core.core import Engine
    from rag_core.core_models import Config
    from rag_core.runtime.app import create_app

    unix_socket = getattr(args, "unix_socket", None)
    bind_non_loopback = getattr(args, "bind_non_loopback", False)
    if unix_socket is None:
        _enforce_loopback_bind(host=args.host, bind_non_loopback=bind_non_loopback)

    config = Config.from_cli(args)
    app = create_app(
        config=config,
        core_factory=Engine,
        ingest_roots=tuple(Path(path) for path in args.ingest_root),
        job_db_path=Path(args.job_db_path),
        job_retention_seconds=getattr(args, "job_retention_seconds", None),
        max_body_bytes=getattr(args, "max_body_bytes", None) or 4 * 1024 * 1024,
        ingest_concurrency=getattr(args, "ingest_concurrency", None) or 8,
    )
    uvicorn_kwargs: dict[str, Any] = {
        "log_level": "info",
        "limit_concurrency": getattr(args, "limit_concurrency", None) or 64,
    }
    if unix_socket is not None:
        uvicorn_kwargs["uds"] = unix_socket
    else:
        uvicorn_kwargs["host"] = args.host
        uvicorn_kwargs["port"] = args.port
    uvicorn.run(app, **uvicorn_kwargs)
    return 0


def _enforce_loopback_bind(*, host: str, bind_non_loopback: bool) -> None:
    """Refuse to bind a non-loopback interface without explicit opt-in.

    The runtime is a sidecar; the gateway in front of it terminates auth and
    is the only legitimate caller. Default-loopback means an operator can't
    accidentally expose the runtime on a container network just by editing
    ``--host``. ``--bind-non-loopback`` is the explicit acknowledgement; we
    also emit a startup warning so the log carries the exposure decision.
    """
    if host in LOOPBACK_HOSTS:
        return
    if not bind_non_loopback:
        raise SystemExit(
            "refusing to bind non-loopback host without --bind-non-loopback "
            f"(got --host={host!r}; allowed loopback hosts: "
            f"{sorted(LOOPBACK_HOSTS)})"
        )
    _logger.warning(
        "rag-core serve is binding non-loopback host=%s; the gateway in "
        "front of this process MUST terminate auth and tenant binding.",
        host,
    )
