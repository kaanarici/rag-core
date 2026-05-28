"""Shared provider-diagnostic payload vocabulary."""

from __future__ import annotations

from typing import Literal, TypeAlias

ProviderDiagnosticSupportLevel: TypeAlias = Literal[
    "default",
    "default_noop",
    "first_party_optional",
    "first_party_utility",
    "injected",
]
ProviderDiagnosticReadinessScope: TypeAlias = Literal["package_and_env"]
ProviderDiagnosticField: TypeAlias = Literal[
    "api_key_configured",
    "api_key_env",
    "configured",
    "package_available",
    "providers",
    "readiness_scope",
    "registered",
    "runtime_config",
    "support_level",
]

FIELD_API_KEY_CONFIGURED: ProviderDiagnosticField = "api_key_configured"
FIELD_API_KEY_ENV: ProviderDiagnosticField = "api_key_env"
FIELD_CONFIGURED: ProviderDiagnosticField = "configured"
FIELD_PACKAGE_AVAILABLE: ProviderDiagnosticField = "package_available"
FIELD_PROVIDERS: ProviderDiagnosticField = "providers"
FIELD_READINESS_SCOPE: ProviderDiagnosticField = "readiness_scope"
FIELD_REGISTERED: ProviderDiagnosticField = "registered"
FIELD_RUNTIME_CONFIG: ProviderDiagnosticField = "runtime_config"
FIELD_SUPPORT_LEVEL: ProviderDiagnosticField = "support_level"
SUPPORT_DEFAULT: ProviderDiagnosticSupportLevel = "default"
SUPPORT_DEFAULT_NOOP: ProviderDiagnosticSupportLevel = "default_noop"
SUPPORT_FIRST_PARTY_OPTIONAL: ProviderDiagnosticSupportLevel = "first_party_optional"
SUPPORT_FIRST_PARTY_UTILITY: ProviderDiagnosticSupportLevel = "first_party_utility"
SUPPORT_INJECTED: ProviderDiagnosticSupportLevel = "injected"

PROVIDER_DIAGNOSTIC_SUPPORT_LEVELS: tuple[ProviderDiagnosticSupportLevel, ...] = (
    SUPPORT_DEFAULT,
    SUPPORT_DEFAULT_NOOP,
    SUPPORT_FIRST_PARTY_OPTIONAL,
    SUPPORT_FIRST_PARTY_UTILITY,
    SUPPORT_INJECTED,
)
READINESS_PACKAGE_AND_ENV: ProviderDiagnosticReadinessScope = "package_and_env"
PROVIDER_DIAGNOSTIC_READINESS_SCOPES: tuple[ProviderDiagnosticReadinessScope, ...] = (
    READINESS_PACKAGE_AND_ENV,
)

__all__ = [
    "FIELD_API_KEY_CONFIGURED",
    "FIELD_API_KEY_ENV",
    "FIELD_CONFIGURED",
    "FIELD_PACKAGE_AVAILABLE",
    "FIELD_PROVIDERS",
    "FIELD_READINESS_SCOPE",
    "FIELD_REGISTERED",
    "FIELD_RUNTIME_CONFIG",
    "FIELD_SUPPORT_LEVEL",
    "PROVIDER_DIAGNOSTIC_READINESS_SCOPES",
    "PROVIDER_DIAGNOSTIC_SUPPORT_LEVELS",
    "ProviderDiagnosticField",
    "ProviderDiagnosticReadinessScope",
    "ProviderDiagnosticSupportLevel",
    "READINESS_PACKAGE_AND_ENV",
    "SUPPORT_DEFAULT",
    "SUPPORT_DEFAULT_NOOP",
    "SUPPORT_FIRST_PARTY_OPTIONAL",
    "SUPPORT_FIRST_PARTY_UTILITY",
    "SUPPORT_INJECTED",
]
