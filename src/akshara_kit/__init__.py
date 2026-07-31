"""akshara-kit — hybrid intelligent framework for Sinhala document ingestion.

The public entry point is :func:`route`, which detects a document's format,
extracts its text, normalises any legacy Sinhala encoding to Unicode, and
returns an :class:`~akshara_kit.contracts.extraction.ExtractionResult`::

    from akshara_kit import route

    result = route("document.pdf")
    print(result.text)

``route`` is resolved lazily so that importing the package does not require
the optional extraction extras to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from akshara_kit.contracts.extraction import (
    AdapterAttempt,
    ExtractionResult,
    FontDetectionMethod,
    QualityScore,
    SourceFormat,
)
from akshara_kit.eye.errors import (
    AdapterUnavailableError,
    AksharaKitError,
    ExtractionFailedError,
    OcrUnavailableError,
    UnmappedLegacyFontError,
    UnsupportedFormatError,
)

if TYPE_CHECKING:
    from akshara_kit.router.format_router import detect_format, route

__version__ = "0.1.0"

__all__ = [
    "AdapterAttempt",
    "AdapterUnavailableError",
    "AksharaKitError",
    "ExtractionFailedError",
    "ExtractionResult",
    "FontDetectionMethod",
    "OcrUnavailableError",
    "QualityScore",
    "SourceFormat",
    "UnmappedLegacyFontError",
    "UnsupportedFormatError",
    "__version__",
    "detect_format",
    "route",
]

_LAZY = frozenset({"route", "detect_format"})


def __getattr__(name: str) -> Any:
    """Resolve :func:`route` / :func:`detect_format` on first access."""
    if name in _LAZY:
        from akshara_kit.router import format_router

        return getattr(format_router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
