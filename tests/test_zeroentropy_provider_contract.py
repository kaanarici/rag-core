"""ZeroEntropy embedding provider replay from recorded fixtures."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_core.search.providers.zeroentropy import ZeroEntropyEmbeddingProvider
from tests.support.provider_fixtures import (
    load_provider_fixture,
    record_mode,
    zeroentropy_embed_response_from_fixture,
)

pytestmark = [pytest.mark.provider_contract]


@pytest.fixture(autouse=True)
def _no_network_replay() -> None:
    assert record_mode() in {"none", "once", "new_episodes"}


def test_zeroentropy_embed_texts_replays_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = load_provider_fixture("zeroentropy", "embed.json")
    response = zeroentropy_embed_response_from_fixture(payload)

    models = MagicMock()
    models.embed = MagicMock(return_value=response)
    client = SimpleNamespace(models=models)
    zeroentropy_module = SimpleNamespace(ZeroEntropy=lambda **_: client)
    monkeypatch.setattr(
        "rag_core.search.providers.zeroentropy._import_zeroentropy",
        lambda: zeroentropy_module,
    )

    provider = ZeroEntropyEmbeddingProvider(model="zembed-1", dimensions=4, api_key="test-key")
    vectors = asyncio.run(provider.embed_texts(["alpha", "beta"]))

    assert vectors == [row["embedding"] for row in payload["results"]]
    assert models.embed.call_args.kwargs["dimensions"] == 4
