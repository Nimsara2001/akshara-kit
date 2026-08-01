"""Decide, per page, whether OCR is needed (realises Section 8).

Deliberately per-page. A document that mixes scanned and born-digital pages is
common, and an all-or-nothing document-level decision either wastes OCR on
pages that do not need it or drops the text of pages that do.

A note on the PyMuPDF API. Section 8 suggests ``page.get_images()``, but that
returns ``(xref, smask, width, height, ...)`` — the image's *pixel* dimensions
with no page geometry. Coverage computed from those is wrong: a 4000x3000
source image placed in a 50pt logo box would read as enormous.
``page.get_image_info()`` returns a page-space ``bbox`` per image and is the
correct call.

Two triggers, not one
---------------------
Section 8 routes a page to OCR when its text stream is *empty*. That misses the
worse failure: a page whose text stream is full but wrong. Sinhala PDFs are
routinely produced with a broken ``ToUnicode`` cmap, and every text-stream
backend then returns the same confident nonsense — "පොලී" as "පපොලී", "යටතේ" as
"යටපේ" — at a perfectly healthy Sinhala ratio. Rasterising and OCRing such a
page recovers the real text, so a garbled text layer is a second, independent
reason to OCR.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from akshara_kit.eye.quality_probe import MAX_ORPHAN_VOWEL_RATE, orphan_vowel_rate

__all__ = [
    "MIN_CHARS",
    "MIN_IMAGE_COVERAGE",
    "PageLike",
    "image_area_fraction",
    "is_malformed",
    "needs_ocr",
]

#: A page with less text than this is a candidate for OCR.
MIN_CHARS = 20

#: ...but only if an image covers at least this fraction of it. Real
#: born-digital pages measure a few percent (a masthead banner on a 612x792
#: page came to 5.1%); a rasterised page measures ~99%. The margin is wide.
MIN_IMAGE_COVERAGE = 0.5


@runtime_checkable
class PageLike(Protocol):
    """The slice of ``pymupdf.Page`` this module needs.

    Structural typing so the geometry logic can be tested with a stub, without
    PyMuPDF, a real PDF, or Tesseract.
    """

    @property
    def rect(self) -> Any: ...

    def get_image_info(self) -> list[dict]: ...


def _rect_area(rect: Any) -> float:
    """Area of a rectangle given as a PyMuPDF ``Rect`` or a 4-tuple."""
    if isinstance(rect, (tuple, list)):
        x0, y0, x1, y1 = rect
    else:
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _intersection_area(bbox: Any, page_rect: Any) -> float:
    """Area of ``bbox`` clipped to the page — images can overhang the edge."""
    if isinstance(bbox, (tuple, list)):
        bx0, by0, bx1, by1 = bbox
    else:
        bx0, by0, bx1, by1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    if isinstance(page_rect, (tuple, list)):
        px0, py0, px1, py1 = page_rect
    else:
        px0, py0, px1, py1 = page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1

    return _rect_area(
        (max(bx0, px0), max(by0, py0), min(bx1, px1), min(by1, py1))
    )


def image_area_fraction(page: PageLike) -> float:
    """Largest single image's share of the page area, in ``[0, 1]``.

    The largest image, not the union: a page carrying twelve small photographs
    is an illustrated page, not a scan, and a union would misclassify it.
    """
    page_area = _rect_area(page.rect)
    if page_area <= 0:
        return 0.0

    try:
        images = page.get_image_info()
    except Exception:  # noqa: BLE001 - a malformed page is simply not a scan
        return 0.0

    return max(
        (
            _intersection_area(info["bbox"], page.rect) / page_area
            for info in images
            if "bbox" in info
        ),
        default=0.0,
    )


def is_malformed(
    extracted_text: str, max_orphan_rate: float = MAX_ORPHAN_VOWEL_RATE
) -> bool:
    """True if the text stream is Sinhala, but orthographically impossible.

    Text with too little Sinhala to judge — including an unconverted legacy
    glyph stream, which carries no Sinhala at all — is never malformed by this
    measure. Only text that is plainly Sinhala *and* plainly broken qualifies.
    """
    return orphan_vowel_rate(extracted_text) > max_orphan_rate


def needs_ocr(
    page: PageLike,
    extracted_text: str,
    min_chars: int = MIN_CHARS,
    min_image_coverage: float = MIN_IMAGE_COVERAGE,
    *,
    repair_malformed: bool = True,
) -> bool:
    """True if this page should be routed through OCR (Section 8).

    Either of two independent conditions routes a page to OCR:

    1. The text stream yielded almost nothing *and* the page is mostly covered
       by a single image — a scan. Text alone is not enough, since a genuinely
       blank page has nothing for OCR to find.
    2. The text stream is full but garbled, whatever the page looks like. A
       broken cmap is invisible to the geometry test, so this arm carries no
       image requirement.

    Set ``repair_malformed=False`` to restore condition 1 alone.
    """
    if repair_malformed and is_malformed(extracted_text):
        return True
    if len(extracted_text.strip()) >= min_chars:
        return False
    return image_area_fraction(page) >= min_image_coverage
