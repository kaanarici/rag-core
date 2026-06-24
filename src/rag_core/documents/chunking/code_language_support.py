from __future__ import annotations

import re
from typing import Sequence

LANGUAGE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^\s*(class\s+|def\s+|async\s+def\s+)", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"^\s*(export\s+)?(async\s+)?function\s+", re.MULTILINE),
        re.compile(r"^\s*(export\s+)?class\s+", re.MULTILINE),
        re.compile(
            r"^\s*(export\s+)?(const|let|var)\s+[\w$]+\s*=\s*(async\s*)?\([^\)]*\)\s*=>",
            re.MULTILINE,
        ),
    ],
    "go": [re.compile(r"^\s*func\s+", re.MULTILINE)],
    "rust": [re.compile(r"^\s*(pub\s+)?(fn|impl|struct|enum)\s+", re.MULTILINE)],
    "java": [
        re.compile(
            r"^\s*(public|private|protected)?\s*(static\s+)?class\s+",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(public|private|protected)?\s*(static\s+)?[\w<>\[\]]+\s+\w+\s*\([^\)]*\)\s*\{",
            re.MULTILINE,
        ),
    ],
    "c": [
        re.compile(r"^\s*(typedef\s+)?(struct|enum|union)\s+\w+", re.MULTILINE),
        re.compile(
            r"^\s*(static\s+|inline\s+|extern\s+)?[\w\*\s]+\s+\w+\s*\([^\)]*\)\s*\{",
            re.MULTILINE,
        ),
    ],
    "cpp": [
        re.compile(r"^\s*(class|struct|namespace|template)\s+\w+", re.MULTILINE),
        re.compile(
            r"^\s*[\w:\<\>\~\*&\s]+\s+\w+\s*\([^\)]*\)\s*(const\s*)?\{",
            re.MULTILINE,
        ),
    ],
    "csharp": [
        re.compile(
            r"^\s*(public|private|protected|internal)?\s*(class|struct|interface|enum)\s+",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(public|private|protected|internal)\s+[\w<>\[\],\?]+\s+\w+\s*\([^\)]*\)\s*\{",
            re.MULTILINE,
        ),
    ],
    "ruby": [
        re.compile(r"^\s*(class|module|def)\s+", re.MULTILINE),
    ],
    "php": [
        re.compile(r"^\s*(class|trait|interface)\s+", re.MULTILINE),
        re.compile(
            r"^\s*(public|private|protected)?\s*function\s+\w+\(",
            re.MULTILINE,
        ),
    ],
    "swift": [
        re.compile(
            r"^\s*(class|struct|enum|protocol|extension|func)\s+",
            re.MULTILINE,
        ),
    ],
    "kotlin": [
        re.compile(
            r"^\s*(class|data\s+class|sealed\s+class|object|interface|fun)\s+",
            re.MULTILINE,
        ),
    ],
    "scala": [
        re.compile(r"^\s*(class|object|trait|def)\s+", re.MULTILINE),
    ],
    "terraform": [
        re.compile(
            r'^\s*(resource|module|variable|output|provider|data|locals|terraform)\s+"',
            re.MULTILINE,
        ),
    ],
}

FALLBACK_PATTERNS = [
    re.compile(r"^\s*(class\s+|def\s+|async\s+def\s+)", re.MULTILINE),
    re.compile(r"^\s*(export\s+)?(async\s+)?function\s+", re.MULTILINE),
    re.compile(r"^\s*func\s+", re.MULTILINE),
    re.compile(r"^\s*(pub\s+)?(fn|impl|struct|enum)\s+", re.MULTILINE),
    re.compile(r"^\s*(class|module|interface|trait|resource)\s+", re.MULTILINE),
]

TREE_SITTER_LANGUAGE_CANDIDATES: dict[str, Sequence[str]] = {
    "python": ("python",),
    "javascript": ("javascript", "typescript", "tsx"),
    "java": ("java",),
    "go": ("go",),
    "rust": ("rust",),
    "c": ("c",),
    "cpp": ("cpp", "c++"),
    "csharp": ("c_sharp", "csharp"),
    "ruby": ("ruby",),
    "php": ("php",),
    "swift": ("swift",),
    "kotlin": ("kotlin",),
    "scala": ("scala",),
    "terraform": ("hcl", "terraform"),
}

MAGIKA_TO_INTERNAL_LANGUAGE = {
    "c++": "cpp",
    "c#": "csharp",
    "js": "javascript",
    "ts": "javascript",
}
