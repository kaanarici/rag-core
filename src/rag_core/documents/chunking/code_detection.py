from __future__ import annotations

from typing import Optional

_CODE_MIMES = {
    "text/x-python",
    "text/javascript",
    "text/typescript",
    "application/javascript",
    "text/x-csrc",
    "text/x-c++src",
    "text/x-csharp",
    "text/x-java",
    "text/x-c",
    "text/x-go",
    "text/x-rust",
    "text/x-kotlin",
    "text/x-scala",
    "text/x-ruby",
    "application/x-httpd-php",
    "text/x-swift",
    "application/x-terraform",
}

_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".hpp",
    ".h",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".cc",
    ".sql",
    ".tf",
    ".tfvars",
}

_MIME_TO_LANGUAGE = {
    "text/x-python": "python",
    "text/javascript": "javascript",
    "application/javascript": "javascript",
    "text/typescript": "javascript",
    "text/x-csrc": "c",
    "text/x-c++src": "cpp",
    "text/x-csharp": "csharp",
    "text/x-java": "java",
    "text/x-go": "go",
    "text/x-rust": "rust",
    "text/x-kotlin": "kotlin",
    "text/x-scala": "scala",
    "text/x-ruby": "ruby",
    "application/x-httpd-php": "php",
    "text/x-swift": "swift",
    "application/x-terraform": "terraform",
}

_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".tf": "terraform",
    ".tfvars": "terraform",
}


def is_code_content(
    *,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
    text: Optional[str] = None,
    allow_text_x_prefix: bool = False,
) -> bool:
    """Return whether content should be treated as code for chunking/indexing."""
    normalized_mime = (mime_type or "").strip().lower()
    if normalized_mime:
        if normalized_mime in _CODE_MIMES:
            return True
        if allow_text_x_prefix and normalized_mime.startswith("text/x-"):
            return True

    if filename_extension(filename) in _CODE_EXTENSIONS:
        return True

    if text is None:
        return False

    return _looks_like_code(text)


def detect_code_language(
    *,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[str]:
    if mime_type and mime_type in _MIME_TO_LANGUAGE:
        return _MIME_TO_LANGUAGE[mime_type]
    ext = filename_extension(filename)
    if ext:
        return _EXT_TO_LANGUAGE.get(ext)
    return None


def filename_extension(filename: Optional[str]) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _looks_like_code(text: str) -> bool:
    lines = text[:5000].split("\n")
    code_indicators = sum(
        1
        for line in lines
        if line.strip().startswith(
            (
                "def ",
                "class ",
                "function ",
                "import ",
                "from ",
                "const ",
                "let ",
                "var ",
                "fn ",
                "interface ",
                "module ",
                "resource ",
            )
        )
    )
    return code_indicators > len(lines) * 0.1


__all__ = ["detect_code_language", "filename_extension", "is_code_content"]
