"""Named exceptions for the Eye module.

These live in one shared module rather than beside their primary raiser so the
layering stays acyclic: ``encoding_normaliser`` needs
:class:`UnmappedLegacyFontError` but must never import the router.
``format_router`` re-exports :class:`UnsupportedFormatError`, so callers still
find it where the spec says it lives.
"""

from __future__ import annotations


class AksharaKitError(Exception):
    """Base class for every error this library raises deliberately."""


class UnsupportedFormatError(AksharaKitError):
    """The file is not a PDF, DOCX or XLSX, by extension or by structure."""


class UnmappedLegacyFontError(AksharaKitError):
    """A legacy font was detected that no conversion table covers.

    Deliberately **not raised** during extraction: unmapped fonts are a
    warning-level condition, and their text passes through unchanged rather
    than being silently corrupted. Carried as a payload so callers can report
    on coverage. See ``ExtractionResult.unmapped_legacy_fonts``.
    """

    def __init__(self, font_name: str) -> None:
        self.font_name = font_name
        super().__init__(
            f"No conversion mapping is available for legacy font {font_name!r}; "
            "its text was passed through unchanged."
        )


class AdapterUnavailableError(AksharaKitError):
    """An extraction backend's optional dependency is not installed."""


class ExtractionFailedError(AksharaKitError):
    """Every adapter available for this document failed."""


class OcrUnavailableError(AksharaKitError):
    """The Tesseract binary or the Sinhala language pack is missing."""


class MultimodalUnavailableError(AksharaKitError):
    """The requested vision-language provider cannot be used.

    Either no API key is configured for it, or its SDK is not installed.
    """


class MultimodalBudgetExceededError(AksharaKitError):
    """The document needs more pages transcribed than the caller allowed.

    Raised rather than silently truncating to the cap: a partial transcription
    presented as a whole one is worse than a clear refusal, and the caller is
    the only one who can decide whether the extra pages are worth paying for.
    """

    def __init__(self, needed: int, allowed: int) -> None:
        self.needed = needed
        self.allowed = allowed
        super().__init__(
            f"{needed} page(s) need multimodal transcription but the limit is "
            f"{allowed}. Raise MultimodalConfig.max_pages to proceed."
        )
