from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Sequence, cast

if TYPE_CHECKING:
    from ._main import CliRuntimeError as CliRuntimeError

Engine: Any = None


def _main_module() -> ModuleType:
    module = import_module("rag_core.cli._main")
    engine_override = globals().get("Engine")
    if engine_override is None:
        from rag_core.core import Engine as default_engine

        setattr(module, "Engine", default_engine)
    else:
        setattr(module, "Engine", engine_override)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    run = cast(Callable[[Sequence[str] | None], int], getattr(_main_module(), "main"))
    return run(argv)


async def async_main(argv: Sequence[str] | None = None) -> int:
    run = cast(
        Callable[[Sequence[str] | None], Awaitable[int]],
        getattr(_main_module(), "async_main"),
    )
    return await run(argv)


def _build_parser() -> argparse.ArgumentParser:
    from rag_core.cli.parser import _build_parser as build_parser

    return build_parser()


def __getattr__(name: str) -> Any:
    if name == "CliRuntimeError":
        value = getattr(import_module("rag_core.cli._main"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rag_core.cli' has no attribute {name!r}")
