"""Shared provider-diagnostic payload vocabulary."""

from __future__ import annotations

from typing import Literal, TypeAlias

ProviderDiagnosticMaturity: TypeAlias = Literal[
    "default",
    "disabled",
    "optional",
    "utility",
    "injected",
]
ProviderDiagnosticReadinessScope: TypeAlias = Literal["installed_and_configured"]
ProviderDiagnosticField: TypeAlias = Literal[
    "api_key_configured",
    "api_key_env",
    "configured",
    "package_available",
    "providers",
    "readiness_scope",
    "registered",
    "runtime_config",
    "maturity",
]

FIELD_API_KEY_CONFIGURED: ProviderDiagnosticField = "api_key_configured"
FIELD_API_KEY_ENV: ProviderDiagnosticField = "api_key_env"
FIELD_CONFIGURED: ProviderDiagnosticField = "configured"
FIELD_PACKAGE_AVAILABLE: ProviderDiagnosticField = "package_available"
FIELD_PROVIDERS: ProviderDiagnosticField = "providers"
FIELD_READINESS_SCOPE: ProviderDiagnosticField = "readiness_scope"
FIELD_REGISTERED: ProviderDiagnosticField = "registered"
FIELD_RUNTIME_CONFIG: ProviderDiagnosticField = "runtime_config"
FIELD_MATURITY: ProviderDiagnosticField = "maturity"
MATURITY_DEFAULT: ProviderDiagnosticMaturity = "default"
MATURITY_DISABLED: ProviderDiagnosticMaturity = "disabled"
MATURITY_OPTIONAL: ProviderDiagnosticMaturity = "optional"
MATURITY_UTILITY: ProviderDiagnosticMaturity = "utility"
MATURITY_INJECTED: ProviderDiagnosticMaturity = "injected"

PROVIDER_DIAGNOSTIC_MATURITIES: tuple[ProviderDiagnosticMaturity, ...] = (
    MATURITY_DEFAULT,
    MATURITY_DISABLED,
    MATURITY_OPTIONAL,
    MATURITY_UTILITY,
    MATURITY_INJECTED,
)
READINESS_INSTALLED_AND_CONFIGURED: ProviderDiagnosticReadinessScope = (
    "installed_and_configured"
)
PROVIDER_DIAGNOSTIC_READINESS_SCOPES: tuple[ProviderDiagnosticReadinessScope, ...] = (
    READINESS_INSTALLED_AND_CONFIGURED,
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
    "FIELD_MATURITY",
    "MATURITY_DEFAULT",
    "MATURITY_DISABLED",
    "MATURITY_INJECTED",
    "MATURITY_OPTIONAL",
    "MATURITY_UTILITY",
    "PROVIDER_DIAGNOSTIC_MATURITIES",
    "PROVIDER_DIAGNOSTIC_READINESS_SCOPES",
    "ProviderDiagnosticField",
    "ProviderDiagnosticMaturity",
    "ProviderDiagnosticReadinessScope",
    "READINESS_INSTALLED_AND_CONFIGURED",
]
