"""PDF extraction via pypdf.

The cheapest of the four text-stream backends, and therefore the head of the
cost ordering the coordinator uses to break ties. Returns flat text with no
font association.
"""

from __future__ import annotations

import time

from akshara_kit.contracts.extraction import ExtractionResult, SourceFormat
from akshara_kit.eye.errors import AdapterUnavailableError, ExtractionFailedError
from akshara_kit.eye.quality_probe import score

__all__ = ["BACKEND_ID", "extract"]

BACKEND_ID = "pypdf"

_PAGE_SEPARATOR = "\n\n"


def _extract_pages(file_path: str) -> list[str]:
    """Text of each page, in document order. See ``pymupdf_adapter``."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "pypdf is not installed; install the 'pdf' extra"
        ) from exc

    try:
        reader = PdfReader(file_path)
        return [page.extract_text() or "" for page in reader.pages]
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
