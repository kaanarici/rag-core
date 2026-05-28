from rag_core.integrations.integration_context_text import context_pack_prompt_text


def test_context_pack_prompt_text_prefers_as_prompt_text() -> None:
    class _Pack:
        def as_text(self) -> str:
            return "leak"

        def as_prompt_text(self) -> str:
            return "safe"

        def to_payload(self) -> dict[str, object]:
            return {}

        def to_prompt_payload(self) -> dict[str, object]:
            return {}

    assert context_pack_prompt_text(_Pack()) == "safe"
