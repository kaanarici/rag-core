from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

from rag_core.cli.inputs import cli_safe_error_message
from rag_core.provider_errors import (
    ProviderCliError,
    is_provider_error,
    is_provider_bootstrap_error,
    provider_bootstrap_message,
    provider_runtime_message,
)


class ReadyClosableCore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def close(self) -> None: ...


CoreT = TypeVar("CoreT", bound=ReadyClosableCore)
ResultT = TypeVar("ResultT")


async def run_with_ready_core(
    *,
    core_factory: Callable[[], CoreT],
    action: str,
    run: Callable[[CoreT], Awaitable[ResultT]],
) -> ResultT:
    try:
        core = core_factory()
    except Exception as exc:
        if is_provider_bootstrap_error(exc):
            raise ProviderCliError(
                provider_bootstrap_message(exc, action=action)
            ) from exc
        raise
    try:
        try:
            await core.ensure_ready()
        except Exception as exc:
            if is_provider_bootstrap_error(exc):
                raise ProviderCliError(
                    provider_bootstrap_message(exc, action=action)
                ) from exc
            raise ValueError(cli_safe_error_message(exc, action=action)) from exc
        try:
            return await run(core)
        except Exception as exc:
            if is_provider_error(exc) or is_provider_bootstrap_error(exc):
                raise ProviderCliError(
                    provider_runtime_message(exc, action=action)
                ) from exc
            raise
    finally:
        await core.close()
