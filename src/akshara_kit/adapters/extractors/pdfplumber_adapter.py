"""PDF extraction via pdfplumber.

Layout-aware, and slower than pypdf for it. Unlike the flat text stream,
pdfplumber does expose a font name per character; :func:`iter_char_fonts` is
kept as a documented fallback should PyMuPDF ever be unavailable, but the
coordinator prefers PyMuPDF spans, which are already grouped into runs.
"""

from __future__ import annotations

import time
from typing import Iterator

from akshara_kit.contracts.extraction import ExtractionResult, SourceFormat
from akshara_kit.eye.errors import AdapterUnavailableError, ExtractionFailedError
from akshara_kit.eye.quality_probe import score

__all__ = ["BACKEND_ID", "extract", "iter_char_fonts"]

BACKEND_ID = "pdfplumber"

_PAGE_SEPARATOR = "\n\n"


def _open(file_path: str):
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "pdfplumber is not installed; install the 'pdf' extra"
        ) from exc
    return pdfplumber.open(file_path)


def _extract_pages(file_path: str) -> list[str]:
    """Text of each page, in document order. See ``pymupdf_adapter``."""
    try:
        with _open(file_path) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]
    except AdapterUnavailableError:
        raise
    except Exception as exc:
        raise ExtractionFailedError(
            f"{BACKEND_ID} failed to extract {file_path}: {exc}"
        ) from exc


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


def iter_char_fonts(file_path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(character, font_name)`` pairs.

    Font names here retain their PDF subset prefix (``SCPIOQ+FMAbhaya``), so
    callers must pass them through ``normalise_font_name``.
    """
    with _open(file_path) as pdf:
        for page in pdf.pages:
            for char in page.chars:
                yield char.get("text", ""), char.get("fontname", "")
