from __future__ import annotations

import asyncio

import pytest

from rag_core.cli_core_runtime import run_with_ready_core
from rag_core.cli_provider_errors import (
    is_provider_error,
    is_provider_bootstrap_error,
    provider_bootstrap_message,
    provider_runtime_message,
)


class _OpenAIError(Exception):
    __module__ = "openai"


class _CohereError(Exception):
    __module__ = "cohere"


class _VoyageError(Exception):
    __module__ = "voyageai"


class _MistralError(Exception):
    __module__ = "mistral"


class _GeminiError(Exception):
    __module__ = "google"


class _AnthropicError(Exception):
    __module__ = "anthropic"


@pytest.mark.parametrize(
    ("exc", "provider", "env_var"),
    (
        (_OpenAIError("raw api_key client option OPENAI_API_KEY"), "openai", "OPENAI_API_KEY"),
        (_CohereError("missing key"), "cohere", "COHERE_API_KEY"),
        (_VoyageError("missing key"), "voyage", "VOYAGE_API_KEY"),
        (_MistralError("missing key"), "mistral", "MISTRAL_API_KEY"),
        (_GeminiError("missing key"), "gemini", "GEMINI_API_KEY"),
        (_AnthropicError("missing key"), "anthropic", "ANTHROPIC_API_KEY"),
        (RuntimeError("Voyage api key is missing"), "voyage", "VOYAGE_API_KEY"),
    ),
)
def test_provider_bootstrap_message_reports_provider_without_raw_error(
    exc: Exception,
    provider: str,
    env_var: str,
) -> None:
    assert is_provider_bootstrap_error(exc) is True

    message = provider_bootstrap_message(exc, action="search")

    assert f"provider={provider}" in message
    assert "provider setup failed before search" in message
    assert env_var in message
    if provider != "openai":
        assert "OPENAI_API_KEY" not in message
    assert "raw api_key client option" not in message


def test_cohere_provider_bootstrap_message_lists_supported_env_names() -> None:
    message = provider_bootstrap_message(_CohereError("missing key"), action="search")

    assert "provider=cohere" in message
    assert "COHERE_API_KEY" in message
    assert "CO_API_KEY" in message


def test_provider_bootstrap_error_ignores_unrelated_errors() -> None:
    assert is_provider_bootstrap_error(RuntimeError("network timeout")) is False


def test_provider_runtime_error_is_not_misclassified_as_bootstrap() -> None:
    exc = _OpenAIError("rate limit reached for private prompt text")

    assert is_provider_bootstrap_error(exc) is False
    assert is_provider_error(exc) is True

    message = provider_runtime_message(exc, action="search")

    assert "provider failed during search" in message
    assert "provider=openai" in message
    assert "rate limit reached" not in message
    assert "private prompt text" not in message
    assert "OPENAI_API_KEY" not in message


@pytest.mark.parametrize(
    "exc",
    (
        _MistralError("rate limit for private OCR payload"),
        _GeminiError("rate limit for private OCR payload"),
        _AnthropicError("rate limit for private context payload"),
    ),
)
def test_provider_runtime_error_covers_document_understanding_providers(
    exc: Exception,
) -> None:
    assert is_provider_bootstrap_error(exc) is False
    assert is_provider_error(exc) is True

    message = provider_runtime_message(exc, action="parse")

    assert "provider failed during parse" in message
    assert "rate limit" not in message
    assert "private" not in message


def test_run_with_ready_core_wraps_provider_failures_during_action() -> None:
    class ReadyCore:
        closed = False

        async def ensure_ready(self) -> None:
            return None

        async def close(self) -> None:
            type(self).closed = True

    async def run(core: ReadyCore) -> int:
        raise _OpenAIError("rate limit reached for private prompt text")

    async def scenario() -> None:
        with pytest.raises(ValueError) as exc_info:
            await run_with_ready_core(
                core_factory=ReadyCore,
                action="search",
                run=run,
            )
        message = str(exc_info.value)
        assert "provider failed during search" in message
        assert "rate limit reached" not in message
        assert "private prompt text" not in message
        assert ReadyCore.closed is True

    asyncio.run(scenario())


def test_run_with_ready_core_treats_action_auth_failures_as_runtime() -> None:
    class ReadyCore:
        closed = False

        async def ensure_ready(self) -> None:
            return None

        async def close(self) -> None:
            type(self).closed = True

    async def run(core: ReadyCore) -> int:
        raise _OpenAIError("Unauthorized: OPENAI_API_KEY invalid for private prompt")

    async def scenario() -> None:
        with pytest.raises(ValueError) as exc_info:
            await run_with_ready_core(
                core_factory=ReadyCore,
                action="search",
                run=run,
            )
        message = str(exc_info.value)
        assert "provider failed during search" in message
        assert "provider setup failed before search" not in message
        assert "OPENAI_API_KEY invalid" not in message
        assert "private prompt" not in message
        assert ReadyCore.closed is True

    asyncio.run(scenario())
