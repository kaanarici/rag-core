from __future__ import annotations

import importlib
import logging

from rag_core.documents.exception_names import root_exception_type

from .base import BaseConverter
from .registry_specs import CONVERTER_SPECS, REQUIRED_CONVERTER_KEYS, ConverterSpec

logger = logging.getLogger(__name__)

_converters: dict[str, BaseConverter] | None = None


class _ConverterInitError(RuntimeError):
    def __init__(self, *, converter_key: str, error_type: str) -> None:
        self.converter_key = converter_key
        self.error_type = error_type
        super().__init__(
            "Failed to initialize %s converter (error_type=%s)"
            % (converter_key, error_type)
        )


def _converter_init_error_type(exc: Exception) -> str:
    if isinstance(exc, _ConverterInitError):
        return exc.error_type
    return root_exception_type(exc)


def _build_converter(spec: ConverterSpec) -> BaseConverter:
    error_type: str
    try:
        module = importlib.import_module(spec.module_name, package=__package__)
        converter_cls = getattr(module, spec.class_name)
        if not isinstance(converter_cls, type) or not issubclass(
            converter_cls, BaseConverter
        ):
            raise TypeError("%s is not a BaseConverter" % spec.class_name)
        return converter_cls()
    except Exception as exc:
        error_type = _converter_init_error_type(exc)
    raise _ConverterInitError(converter_key=spec.key, error_type=error_type)


def get_registered_converters() -> dict[str, BaseConverter]:
    global _converters
    if _converters is not None:
        return _converters

    converters: dict[str, BaseConverter] = {}
    for spec in CONVERTER_SPECS:
        try:
            converters[spec.key] = _build_converter(spec)
        except Exception as exc:
            if spec.required:
                raise
            logger.warning(
                "Skipping unavailable %s converter after %s",
                spec.key,
                _converter_init_error_type(exc),
            )

    missing_required = [key for key in REQUIRED_CONVERTER_KEYS if key not in converters]
    if missing_required:
        raise RuntimeError(
            f"Required converters unavailable after registry load: {', '.join(sorted(missing_required))}"
        )

    _converters = converters
    return converters
