"""PDF extraction via PyMuPDF.

This adapter carries extra weight beyond the race. PyMuPDF is the only one of
the four PDF backends that associates a font with each run of text, so
:func:`iter_spans` is the vehicle for span-gated legacy conversion — the
coordinator uses it even when another adapter wins on raw text. Section 6.2
anticipates exactly this.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

from akshara_kit.contracts.extraction import ExtractionResult, SourceFormat
from akshara_kit.eye.errors import AdapterUnavailableError, ExtractionFailedError
from akshara_kit.eye.quality_probe import score

if TYPE_CHECKING:
    from akshara_kit.eye.font_detection import SpanFont

__all__ = ["BACKEND_ID", "extract", "iter_spans", "page_count"]

BACKEND_ID = "pymupdf"

_PAGE_SEPARATOR = "\n\n"


def _pymupdf():
    """Import PyMuPDF under either of its module names.

    ``fitz`` is the legacy alias; ``pymupdf`` is canonical from 1.24 onward.
    """
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        try:
            import fitz

            return fitz
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise AdapterUnavailableError(
                "PyMuPDF is not installed; install the 'pdf' extra"
            ) from exc


def _extract_pages(file_path: str) -> list[str]:
    """Text of each page, in document order.

    Private because Section 12 fixes the public adapter surface at
    ``extract``. The coordinator needs page granularity to merge per-page OCR
    output back in the right order, which a single joined string cannot supply.
    """
    pymupdf = _pymupdf()
    try:
        with pymupdf.open(file_path) as document:
            return [page.get_text() for page in document]
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


def page_count(file_path: str) -> int:
    """Number of pages, without extracting anything."""
    pymupdf = _pymupdf()
    try:
        with pymupdf.open(file_path) as document:
            return document.page_count
    except Exception as exc:
        raise ExtractionFailedError(
            f"{BACKEND_ID} could not open {file_path}: {exc}"
        ) from exc


def iter_spans(
    file_path: str, first_page: int | None = None, last_page: int | None = None
) -> Iterator[SpanFont]:
    """Yield every text span with its font, plus layout separator spans.

    The separators matter: joining bare span texts would lose all line and
    block structure, so the span path would silently produce worse layout than
    the flat ``get_text()`` it replaces. Synthetic spans restore it.

    ``first_page``/``last_page`` are inclusive 0-based bounds.
    """
    from akshara_kit.eye.font_detection import SpanFont, separator

    pymupdf = _pymupdf()
    try:
        document = pymupdf.open(file_path)
    except Exception as exc:
        raise ExtractionFailedError(
            f"{BACKEND_ID} could not open {file_path}: {exc}"
        ) from exc

    start = 0 if first_page is None else first_page
    stop = document.page_count - 1 if last_page is None else last_page
    try:
        for index in range(start, stop + 1):
            page = document[index]
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text"):
                            yield SpanFont(
                                text=span["text"],
                                font=span.get("font", ""),
                                location=f"p{index}",
                            )
                    yield separator("\n")
                yield separator("\n")
            if index < stop:
                yield separator("\f")
    finally:
        document.close()
