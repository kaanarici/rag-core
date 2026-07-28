"""Spec-driven readiness builders for model providers (embedding + reranker)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from rag_core.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_PROVIDER,
    DEMO_EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_PROVIDER,
)
from rag_core.core_models import Config
from rag_core.provider_api_keys import (
    COHERE_API_KEY_ENVS,
    OPENAI_API_KEY_ENVS,
    VOYAGE_API_KEY_ENVS,
    ZEROENTROPY_API_KEY_ENVS,
    api_key_configured,
)
from rag_core.provider_package_names import (
    COHERE_PACKAGE,
    FASTEMBED_PACKAGE,
    OPENAI_PACKAGE,
    VOYAGE_PACKAGE,
    ZEROENTROPY_PACKAGE,
)

from .cohere import COHERE_PROVIDER
from .diagnostic_support import (
    FIELD_API_KEY_CONFIGURED,
    FIELD_API_KEY_ENV,
    FIELD_CONFIGURED,
    FIELD_PACKAGE_AVAILABLE,
    FIELD_RUNTIME_CONFIG,
    FIELD_MATURITY,
    MATURITY_DEFAULT,
    MATURITY_DISABLED,
    MATURITY_OPTIONAL,
    MATURITY_UTILITY,
    ProviderDiagnosticMaturity,
)
from .embedding_models import get_embedding_model_spec
from .openai_embedding import normalize_optional_provider_config_value
from .reranker_resolution import reranker_api_key_env_names
from .voyage import VOYAGE_PROVIDER
from .zeroentropy import ZEROENTROPY_PROVIDER


@dataclass(frozen=True)
class ModelProviderDiagnosticSpec:
    maturity: ProviderDiagnosticMaturity
    package_name: str | None
    api_key_envs: tuple[str, ...] = ()
    runtime_config: str = ""

    @property
    def api_key_env_payload(self) -> str | list[str]:
        if len(self.api_key_envs) == 1:
            return self.api_key_envs[0]
        return list(self.api_key_envs)


EMBEDDING_PROVIDER_SPECS = {
    DEFAULT_EMBEDDING_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_DEFAULT,
        package_name=OPENAI_PACKAGE,
        api_key_envs=OPENAI_API_KEY_ENVS,
        runtime_config="Config.embedding",
    ),
    DEMO_EMBEDDING_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_UTILITY,
        package_name=None,
        runtime_config="Config.embedding",
    ),
    LOCAL_EMBEDDING_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_UTILITY,
        package_name=FASTEMBED_PACKAGE,
        runtime_config="Config.embedding",
    ),
    COHERE_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_OPTIONAL,
        package_name=COHERE_PACKAGE,
        api_key_envs=COHERE_API_KEY_ENVS,
        runtime_config="Config.embedding",
    ),
    VOYAGE_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_OPTIONAL,
        package_name=VOYAGE_PACKAGE,
        api_key_envs=VOYAGE_API_KEY_ENVS,
        runtime_config="Config.embedding",
    ),
    ZEROENTROPY_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_OPTIONAL,
        package_name=ZEROENTROPY_PACKAGE,
        api_key_envs=ZEROENTROPY_API_KEY_ENVS,
        runtime_config="Config.embedding",
    ),
}
RERANKER_PROVIDER_SPECS = {
    DEFAULT_RERANKER_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_DISABLED,
        package_name=None,
        runtime_config="Config.reranker",
    ),
    COHERE_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_OPTIONAL,
        package_name=COHERE_PACKAGE,
        api_key_envs=reranker_api_key_env_names(COHERE_PROVIDER),
        runtime_config="Config.reranker",
    ),
    VOYAGE_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_OPTIONAL,
        package_name=VOYAGE_PACKAGE,
        api_key_envs=reranker_api_key_env_names(VOYAGE_PROVIDER),
        runtime_config="Config.reranker",
    ),
    ZEROENTROPY_PROVIDER: ModelProviderDiagnosticSpec(
        maturity=MATURITY_OPTIONAL,
        package_name=ZEROENTROPY_PACKAGE,
        api_key_envs=reranker_api_key_env_names(ZEROENTROPY_PROVIDER),
        runtime_config="Config.reranker",
    ),
}


def embedding_provider_diagnostics(
    provider: str,
    *,
    spec: ModelProviderDiagnosticSpec,
    config: Config,
    configured: str,
    embedding_dimensions: int,
) -> dict[str, object]:
    selected = provider == configured
    model_spec = (
        get_embedding_model_spec(provider, config.embedding.model) if selected else None
    )
    payload: dict[str, object] = {
        FIELD_MATURITY: spec.maturity,
        FIELD_CONFIGURED: selected,
        FIELD_PACKAGE_AVAILABLE: package_available(spec),
        FIELD_RUNTIME_CONFIG: spec.runtime_config,
    }
    if spec.api_key_envs:
        payload.update(
            {
                FIELD_API_KEY_ENV: spec.api_key_env_payload,
                FIELD_API_KEY_CONFIGURED: spec_api_key_configured(
                    spec,
                    selected=selected,
                    explicit_key=config.embedding.api_key,
                ),
            }
        )
    if selected:
        payload.update(
            {
                "model": config.embedding.model,
                "dimensions": embedding_dimensions,
                "model_known": model_spec is not None,
                "dimensions_override": config.embedding.dimensions is not None,
                "batch_size": config.embedding.batch_size,
                "base_url_configured": bool(
                    normalize_optional_provider_config_value(config.embedding.base_url)
                ),
            }
        )
    if model_spec is not None:
        payload.update(
            {
                "default_dimensions": model_spec.default_dimensions,
                "max_dimensions": model_spec.max_dimensions,
                "allowed_dimensions": list(model_spec.allowed_dimensions or ()),
                "supports_dimensions_override": model_spec.supports_dimensions_override,
            }
        )
    return payload


def reranker_provider_diagnostics(
    provider: str,
    *,
    spec: ModelProviderDiagnosticSpec,
    config: Config,
    configured: str,
) -> dict[str, object]:
    selected = provider == configured
    payload: dict[str, object] = {
        FIELD_MATURITY: spec.maturity,
        FIELD_CONFIGURED: selected,
        FIELD_PACKAGE_AVAILABLE: package_available(spec),
        FIELD_RUNTIME_CONFIG: spec.runtime_config,
    }
    if spec.api_key_envs:
        payload.update(
            {
                FIELD_API_KEY_ENV: spec.api_key_env_payload,
                FIELD_API_KEY_CONFIGURED: spec_api_key_configured(
                    spec,
                    selected=selected,
                    explicit_key=config.reranker.api_key,
                ),
            }
        )
    if selected:
        payload["model"] = config.reranker.model
    return payload


def spec_api_key_configured(
    spec: ModelProviderDiagnosticSpec,
    *,
    selected: bool,
    explicit_key: str | None,
) -> bool:
    selected_explicit_key = explicit_key if selected else None
    return api_key_configured(
        spec.api_key_envs,
        explicit_key=selected_explicit_key,
    )


def package_available(spec: ModelProviderDiagnosticSpec) -> bool:
    if spec.package_name is None:
        return True
    return importlib.util.find_spec(spec.package_name) is not None


__all__ = [
    "EMBEDDING_PROVIDER_SPECS",
    "RERANKER_PROVIDER_SPECS",
    "ModelProviderDiagnosticSpec",
    "embedding_provider_diagnostics",
    "package_available",
    "reranker_provider_diagnostics",
    "spec_api_key_configured",
]
