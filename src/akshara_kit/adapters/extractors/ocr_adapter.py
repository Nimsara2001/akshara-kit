"""OCR extraction via Tesseract (realises Section 8's OCR leg).

Rasterisation defaults to PyMuPDF rather than pdf2image. Section 8 names
pdf2image, but pdf2image exists only to turn PDF pages into images, and
PyMuPDF — already a hard requirement of the ``pdf`` extra — does that natively
with no poppler install. That matters twice over: poppler's absence is exactly
why the prior prototype's OCR never once ran, and PyMuPDF rasterises a single
page without re-parsing the document, which per-page OCR routing needs.

Set ``AKSHARA_RASTERISER=pdf2image`` to use the poppler path instead; it reads
``AKSHARA_POPPLER_PATH`` for a non-PATH install.
"""

from __future__ import annotations

import io
import os
import time
from typing import TYPE_CHECKING

from akshara_kit.contracts.extraction import ExtractionResult, SourceFormat
from akshara_kit.eye import capabilities
from akshara_kit.eye.errors import (
    AdapterUnavailableError,
    ExtractionFailedError,
    OcrUnavailableError,
)
from akshara_kit.eye.quality_probe import score

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["BACKEND_ID", "DEFAULT_DPI", "DEFAULT_LANG", "extract", "extract_page"]

BACKEND_ID = "tesseract"

#: Sinhala. The prototype defaulted to "eng", which meant its Sinhala OCR path
#: was never actually exercised even on the runs that reached Tesseract.
DEFAULT_LANG = "sin"

DEFAULT_DPI = 300

_PAGE_SEPARATOR = "\n\n"


def _require_ocr(lang: str) -> None:
    """Fail early, with a message that says what to install."""
    capabilities.resolve_tesseract_cmd()
    if not capabilities.tesseract_available():
        raise OcrUnavailableError(capabilities.describe_ocr_availability())
    if lang == DEFAULT_LANG and not capabilities.sinhala_ocr_available():
        raise OcrUnavailableError(capabilities.describe_ocr_availability())


def rasterise_page(file_path: str, page_number: int, dpi: int = DEFAULT_DPI) -> Image:
    """Render one page to a PIL image."""
    if os.environ.get("AKSHARA_RASTERISER") == "pdf2image":
        return _rasterise_with_pdf2image(file_path, page_number, dpi)
    return _rasterise_with_pymupdf(file_path, page_number, dpi)


def _rasterise_with_pymupdf(file_path: str, page_number: int, dpi: int) -> Image:
    """Default path: PyMuPDF pixmap -> PNG bytes -> PIL. No poppler needed."""
    try:
        from PIL import Image as PILImage
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "pillow is not installed; install the 'ocr' extra"
        ) from exc

    from akshara_kit.adapters.extractors.pymupdf_adapter import _pymupdf

    pymupdf = _pymupdf()
    try:
        with pymupdf.open(file_path) as document:
            pixmap = document[page_number].get_pixmap(dpi=dpi)
            payload = pixmap.tobytes("png")
    except Exception as exc:
        raise ExtractionFailedError(
            f"Could not rasterise page {page_number} of {file_path}: {exc}"
        ) from exc
    return PILImage.open(io.BytesIO(payload))


def _rasterise_with_pdf2image(file_path: str, page_number: int, dpi: int) -> Image:
    """Optional poppler path, kept for parity with Section 8's letter."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "pdf2image is not installed; install the 'ocr' extra"
        ) from exc

    if not capabilities.poppler_available():
        raise OcrUnavailableError(
            "AKSHARA_RASTERISER=pdf2image was requested but poppler (pdftoppm) "
            "is not available; set AKSHARA_POPPLER_PATH or unset the variable "
            "to use the default PyMuPDF rasteriser"
        )

    images = convert_from_path(
        file_path,
        dpi=dpi,
        first_page=page_number + 1,
        last_page=page_number + 1,
        poppler_path=os.environ.get("AKSHARA_POPPLER_PATH"),
    )
    if not images:
        raise ExtractionFailedError(f"pdf2image returned no image for page {page_number}")
    return images[0]


def extract_page(
    file_path: str,
    page_number: int,
    *,
    lang: str = DEFAULT_LANG,
    dpi: int = DEFAULT_DPI,
) -> str:
    """OCR a single page. Output is Unicode Sinhala, never legacy glyphs."""
    _require_ocr(lang)
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "pytesseract is not installed; install the 'ocr' extra"
        ) from exc

    image = rasterise_page(file_path, page_number, dpi)
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except Exception as exc:
        raise ExtractionFailedError(
            f"Tesseract failed on page {page_number} of {file_path}: {exc}"
        ) from exc
    finally:
        image.close()


def _extract_pages(
    file_path: str, *, lang: str = DEFAULT_LANG, dpi: int = DEFAULT_DPI
) -> list[str]:
    """OCR every page, in document order."""
    from akshara_kit.adapters.extractors.pymupdf_adapter import page_count

    return [
        extract_page(file_path, index, lang=lang, dpi=dpi)
        for index in range(page_count(file_path))
    ]


def extract(file_path: str) -> ExtractionResult:
    """OCR an entire PDF.

    The coordinator normally OCRs only the pages that need it; this whole-file
    form exists for callers who already know the document is a scan.
    """
    started = time.perf_counter()
    text = _PAGE_SEPARATOR.join(_extract_pages(file_path))
    return ExtractionResult(
        text=text,
        backend_id=BACKEND_ID,
        source_format=SourceFormat.PDF,
        latency_seconds=time.perf_counter() - started,
        quality=score(text),
        ocr_used=True,
    )
