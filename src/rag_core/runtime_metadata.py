from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "rag-core"


def describe_runtime_metadata() -> dict[str, object]:
    package_version = _package_version()
    return {
        "package_name": _DISTRIBUTION_NAME,
        "package_version": package_version,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
    }


def _package_version() -> str | None:
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return None


__all__ = ["describe_runtime_metadata"]
