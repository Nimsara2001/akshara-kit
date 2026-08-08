"""PDF extraction via pdfminer.six.

The slowest of the four backends. Retained because the interim report's
benchmark compares all four, and a race that silently drops a backend cannot
report on it.
"""

from __future__ import annotations

import time

from akshara_kit.contracts.extraction import ExtractionResult, SourceFormat
from akshara_kit.eye.errors import AdapterUnavailableError, ExtractionFailedError
from akshara_kit.eye.quality_probe import score

__all__ = ["BACKEND_ID", "extract"]

BACKEND_ID = "pdfminer"

_PAGE_SEPARATOR = "\n\n"

#: pdfminer emits a form feed between pages, which is how we recover page
#: boundaries from its single-string output.
_FORM_FEED = "\f"


def _extract_pages(file_path: str) -> list[str]:
    """Text of each page, in document order. See ``pymupdf_adapter``."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "pdfminer.six is not installed; install the 'pdf' extra"
        ) from exc

    try:
        text = extract_text(file_path)
    except Exception as exc:
        raise ExtractionFailedError(
            f"{BACKEND_ID} failed to extract {file_path}: {exc}"
        ) from exc

    pages = text.split(_FORM_FEED)
    # A trailing form feed leaves an empty final page.
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def extract(file_path: str) -> ExtractionResult:
    """Extract raw text from a PDF. Returns un-normalised text."""
    started = time.perf_counter()
    text = _PAGE_SEPARATOR.join(_extract_pages(file_path))
    return ExtractionResult(
        text=text,
        backend_id=BACKEND_ID,
        source_format=SourceFormat.PDF,
        latency_seconds=time.perf_counter() - started,
        quality=score(text),
    )
