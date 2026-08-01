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

from akshara_kit.contracts.chunking import (
    BoundaryKind,
    ChunkConfig,
    ChunkedDocument,
    SegmentKind,
    SemanticChunk,
)
from akshara_kit.contracts.extraction import (
    AdapterAttempt,
    ExtractionResult,
    FontDetectionMethod,
    MultimodalConfig,
    MultimodalProvider,
    QualityScore,
    SourceFormat,
)
from akshara_kit.eye.errors import (
    AdapterUnavailableError,
    AksharaKitError,
    ExtractionFailedError,
    MultimodalBudgetExceededError,
    MultimodalUnavailableError,
    OcrUnavailableError,
    PrologUnavailableError,
    UnmappedLegacyFontError,
    UnsupportedFormatError,
)

if TYPE_CHECKING:
    from akshara_kit.brain.coordinator import chunk, chunk_text
    from akshara_kit.preprocess import clean
    from akshara_kit.router.format_router import detect_format, route

__version__ = "0.1.0"

__all__ = [
    "AdapterAttempt",
    "AdapterUnavailableError",
    "AksharaKitError",
    "BoundaryKind",
    "ChunkConfig",
    "ChunkedDocument",
    "ExtractionFailedError",
    "ExtractionResult",
    "FontDetectionMethod",
    "MultimodalBudgetExceededError",
    "MultimodalConfig",
    "MultimodalProvider",
    "MultimodalUnavailableError",
    "OcrUnavailableError",
    "PrologUnavailableError",
    "QualityScore",
    "SegmentKind",
    "SemanticChunk",
    "SourceFormat",
    "UnmappedLegacyFontError",
    "UnsupportedFormatError",
    "__version__",
    "chunk",
    "chunk_text",
    "clean",
    "detect_format",
    "route",
]

#: Resolved on first access so importing the package needs none of the optional
#: extras — the Brain in particular must not drag in SWI-Prolog or torch just
#: because someone imported ``akshara_kit``.
_LAZY_ROUTER = frozenset({"route", "detect_format"})
_LAZY_BRAIN = frozenset({"chunk", "chunk_text"})


def __getattr__(name: str) -> Any:
    """Resolve the Eye, Brain and preprocessing entry points on first access."""
    if name in _LAZY_ROUTER:
        from akshara_kit.router import format_router

        return getattr(format_router, name)
    if name in _LAZY_BRAIN:
        from akshara_kit.brain import coordinator

        return getattr(coordinator, name)
    if name == "clean":
        from akshara_kit.preprocess import clean

        return clean
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
