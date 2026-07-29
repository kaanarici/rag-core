from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION_NAME = "rag-core"


def describe_runtime_metadata() -> dict[str, object]:
    package_version = _package_version()
    return {
        "package_name": DISTRIBUTION_NAME,
        "package_version": package_version,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
    }


def _package_version() -> str | None:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return None


def package_version() -> str | None:
    return _package_version()


__all__ = ["DISTRIBUTION_NAME", "describe_runtime_metadata", "package_version"]
