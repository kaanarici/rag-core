from rag_core.integrations.integration_context_text import context_pack_model_text


def test_context_pack_model_text_prefers_as_model_text() -> None:
    class _Pack:
        def as_text(self) -> str:
            return "leak"

        def as_model_text(self) -> str:
            return "safe"

    assert context_pack_model_text(_Pack()) == "safe"


def test_context_pack_model_text_falls_back_to_as_text() -> None:
    class _Pack:
        def as_text(self) -> str:
            return "legacy"

    assert context_pack_model_text(_Pack()) == "legacy"
