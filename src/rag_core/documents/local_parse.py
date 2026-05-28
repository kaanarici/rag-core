"""Local document parsing entrypoint for the converter-based parse path."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from rag_core.documents.converters.registry_maps import is_registered_image_document
from rag_core.documents.exception_names import root_exception_type
from rag_core.documents.page_indices import normalize_page_indices

logger = logging.getLogger(__name__)


def _is_pdf_document(*, filename: str, mime_type: str) -> bool:
    mt = (mime_type or "").strip().lower()
    if mt == "application/pdf":
        return True
    return filename.strip().lower().endswith(".pdf")


def _normalize_ocr_page_indices(raw_indices: Any) -> List[int]:
    return normalize_page_indices(raw_indices, sort=True)


def _allows_empty_ocr_only_output(
    *,
    filename: str,
    mime_type: str,
    metadata: Dict[str, Any],
) -> bool:
    if not bool(metadata.get("needs_ocr")):
        return False
    if _is_pdf_document(filename=filename, mime_type=mime_type):
        if bool(metadata.get("ocr_processed_entire_document")) or bool(
            metadata.get("is_encrypted")
        ):
            return True
        return bool(_normalize_ocr_page_indices(metadata.get("ocr_page_indices")))
    if is_registered_image_document(filename=filename, mime_type=mime_type):
        return metadata.get("parser") == "ocr_required"
    if metadata.get("parser") in {"local:python-docx", "local:python-pptx"}:
        return True
    return False


class LocalParseError(RuntimeError):
    """Raised when local parsing fails."""


def _quality_to_metadata(quality: Any) -> Dict[str, Any]:
    from rag_core.core_prepare_metadata import quality_score_to_metadata

    return quality_score_to_metadata(quality)


async def parse_file_bytes(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> Tuple[str, Dict[str, Any]]:
    """Parse file using the converter system.

    Returns parsed markdown and converter metadata for preparation and indexing.
    The metadata dict contains "parser", "needs_ocr", and optionally
    "ocr_page_indices" for partial PDF OCR.
    """
    error_type: str
    try:
        from .converters import convert_file

        result = await convert_file(file_bytes, filename, mime_type)
        metadata: Dict[str, Any] = dict(result.metadata) if result.metadata else {}
        metadata.setdefault("parser", "local:converter")
        metadata.setdefault("needs_ocr", result.needs_ocr)

        if result.ocr_page_indices is not None:
            metadata["ocr_page_indices"] = result.ocr_page_indices

        normalized_ocr_page_indices = _normalize_ocr_page_indices(metadata.get("ocr_page_indices"))
        if "ocr_page_indices" in metadata:
            metadata["ocr_page_indices"] = normalized_ocr_page_indices

        content = result.content or ""
        if not content.strip() and not _allows_empty_ocr_only_output(
            filename=filename,
            mime_type=mime_type,
            metadata=metadata,
        ):
            raise LocalParseError(
                "Converter returned empty output for %s (parser=%s)"
                % (filename, metadata.get("parser", "unknown"))
            )

        if result.quality:
            metadata["quality"] = _quality_to_metadata(result.quality)

        return content, metadata

    except LocalParseError:
        raise
    except Exception as exc:
        error_type = root_exception_type(exc)
        logger.error("Converter system failed with %s", error_type)
        reason = _safe_parse_failure_reason(exc)
        if reason:
            raise LocalParseError(
                "Converter parse failed for %s: %s (error_type=%s)"
                % (filename, reason, error_type)
            ) from None
    raise LocalParseError(
        "Converter parse failed for %s (error_type=%s)" % (filename, error_type)
    ) from None


def _safe_parse_failure_reason(exc: Exception) -> str:
    reason = str(exc).strip()
    if not reason:
        return ""
    if reason.startswith("Unsupported format"):
        return reason
    if _is_structured_parse_failure(reason):
        return reason
    return ""


def _is_structured_parse_failure(reason: str) -> bool:
    prefixes = (
        "DOCX parse failed (",
        "PPTX parse failed (",
        "XLSX parse failed (",
        "PDF parse failed (",
    )
    return reason.startswith(prefixes) and reason.endswith(")")
