from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

COMMON_SUBPROCESS_ENV_KEYS = (
    "PATH",
    "HOME",
    "SYSTEMROOT",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)
PYTHON_SUBPROCESS_ENV_KEYS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUTF8",
    "VIRTUAL_ENV",
)
NODE_SUBPROCESS_ENV_KEYS = ("NODE_OPTIONS",)
TRANSPORT_SUBPROCESS_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def allowlisted_subprocess_env(
    *,
    runtime_env_keys: Sequence[str] = (),
    provider_env_keys: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (
        *COMMON_SUBPROCESS_ENV_KEYS,
        *runtime_env_keys,
        *TRANSPORT_SUBPROCESS_ENV_KEYS,
        *provider_env_keys,
    ):
        value = os.environ.get(key)
        if value and value.strip():
            env[key] = value
    for key, value in (extra_env or {}).items():
        if value and value.strip():
            env[key] = value
    return env


__all__ = [
    "COMMON_SUBPROCESS_ENV_KEYS",
    "NODE_SUBPROCESS_ENV_KEYS",
    "PYTHON_SUBPROCESS_ENV_KEYS",
    "TRANSPORT_SUBPROCESS_ENV_KEYS",
    "allowlisted_subprocess_env",
]
