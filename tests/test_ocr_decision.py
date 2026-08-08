"""Per-page OCR routing.

The geometry logic is tested against a stub page, so these tests run on any
machine — no PyMuPDF, no PDF, and no Tesseract required. Two tests at the end
confirm the stub matches real PyMuPDF behaviour.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import pymupdf

from akshara_kit.eye.ocr_decision import image_area_fraction, is_malformed, needs_ocr
from samples import GARBLED_CMAP_SINHALA, LEGACY_SINHALA, WELL_FORMED_PARAGRAPH

PAGE = (0.0, 0.0, 612.0, 792.0)


@dataclass
class FakePage:
    """Minimal structural stand-in for ``pymupdf.Page``."""

    rect: tuple[float, float, float, float] = PAGE
    images: list[dict] = field(default_factory=list)

    def get_image_info(self) -> list[dict]:
        return self.images


def image(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"bbox": (x0, y0, x1, y1)}


# --- coverage geometry ----------------------------------------------------


def test_no_images_is_zero_coverage() -> None:
    assert image_area_fraction(FakePage()) == 0.0


def test_full_page_image_is_full_coverage() -> None:
    assert image_area_fraction(FakePage(images=[image(*PAGE)])) == 1.0


def test_masthead_banner_is_low_coverage() -> None:
    """Measured from a real born-digital page: ~5% of a 612x792 page."""
    page = FakePage(images=[image(82.0, 68.0, 538.0, 124.0)])
    assert image_area_fraction(page) < 0.1


def test_many_small_images_do_not_add_up_to_a_scan() -> None:
    """The largest image, not the union — a photo spread is not a scan."""
    tiles = [image(x, y, x + 100, y + 100) for x in (0, 150, 300) for y in (0, 150, 300)]
    assert image_area_fraction(FakePage(images=tiles)) < 0.05


def test_image_overhanging_the_page_is_clipped() -> None:
    """Coverage is bounded by the page, so an oversized image cannot exceed 1."""
    page = FakePage(images=[image(-500.0, -500.0, 1500.0, 1500.0)])
    assert image_area_fraction(page) == 1.0


def test_zero_area_page_does_not_divide_by_zero() -> None:
    assert image_area_fraction(FakePage(rect=(0, 0, 0, 0))) == 0.0


def test_malformed_image_info_is_ignored() -> None:
    page = FakePage(images=[{"width": 4000, "height": 3000}])  # no bbox
    assert image_area_fraction(page) == 0.0


# --- the decision ---------------------------------------------------------


def test_page_with_text_never_needs_ocr() -> None:
    page = FakePage(images=[image(*PAGE)])
    assert not needs_ocr(page, "x" * 500)


def test_empty_page_covered_by_an_image_needs_ocr() -> None:
    assert needs_ocr(FakePage(images=[image(*PAGE)]), "")


def test_blank_page_with_no_image_does_not_need_ocr() -> None:
    """An genuinely empty page has nothing for OCR to find."""
    assert not needs_ocr(FakePage(), "")


def test_whitespace_only_text_counts_as_empty() -> None:
    assert needs_ocr(FakePage(images=[image(*PAGE)]), "   \n\t  ")


def test_page_with_a_caption_and_a_big_image_needs_ocr() -> None:
    """Below the character floor, so the image is what matters."""
    assert needs_ocr(FakePage(images=[image(*PAGE)]), "Fig. 1")


def test_thresholds_are_tunable() -> None:
    page = FakePage(images=[image(0, 0, 612, 400)])  # ~50% coverage
    assert needs_ocr(page, "", min_image_coverage=0.4)
    assert not needs_ocr(page, "", min_image_coverage=0.9)


# --- the second trigger: a garbled text layer -----------------------------


def test_garbled_text_layer_is_malformed() -> None:
    assert is_malformed(GARBLED_CMAP_SINHALA)


def test_correct_text_layer_is_not_malformed() -> None:
    assert not is_malformed(WELL_FORMED_PARAGRAPH)


def test_unconverted_legacy_text_is_not_malformed() -> None:
    """Legacy bytes are not Sinhala yet, so they cannot be malformed Sinhala.

    Without this, every legacy page would be OCR'd before conversion ever got
    a chance to fix it.
    """
    assert not is_malformed(LEGACY_SINHALA)


def test_full_page_of_garbled_text_needs_ocr_despite_having_no_image() -> None:
    """A broken cmap is invisible to the geometry test, so it carries no
    image requirement — this is the arm Section 8 as written does not have."""
    assert needs_ocr(FakePage(), GARBLED_CMAP_SINHALA)


def test_full_page_of_correct_text_does_not_need_ocr() -> None:
    assert not needs_ocr(FakePage(), WELL_FORMED_PARAGRAPH)


def test_malformed_repair_can_be_switched_off() -> None:
    assert not needs_ocr(FakePage(), GARBLED_CMAP_SINHALA, repair_malformed=False)


def test_switching_off_repair_leaves_the_scan_trigger_intact() -> None:
    page = FakePage(images=[image(*PAGE)])
    assert needs_ocr(page, "", repair_malformed=False)


# --- against real PyMuPDF pages -------------------------------------------


def test_real_scanned_page_needs_ocr(scanned_pdf: pathlib.Path) -> None:
    with pymupdf.open(scanned_pdf) as doc:
        page = doc[0]
        assert needs_ocr(page, page.get_text())


def test_real_born_digital_page_does_not_need_ocr(legacy_pdf: pathlib.Path) -> None:
    """A page with a real text layer and no full-page image is left alone."""
    with pymupdf.open(legacy_pdf) as doc:
        page = doc[1]
        assert not needs_ocr(page, page.get_text())


def test_document_decides_per_page_not_per_document(unicode_pdf: pathlib.Path) -> None:
    """The point of Section 8: one document, two different answers.

    Page 0 is an image-only cover, so the scan arm fires; page 2 carries a
    clean text layer, so nothing fires.
    """
    with pymupdf.open(unicode_pdf) as doc:
        cover = needs_ocr(doc[0], doc[0].get_text())
        clean = needs_ocr(doc[2], doc[2].get_text())
    assert cover
    assert not clean


def test_real_garbled_pages_are_caught_only_by_the_malformed_arm(
    mixed_pdf: pathlib.Path,
) -> None:
    """Both pages carry text and no image, so only condition 2 can see them.

    This is the case Section 8's original single condition misses entirely.
    """
    with pymupdf.open(mixed_pdf) as doc:
        with_repair = [needs_ocr(p, p.get_text()) for p in doc]
        without = [needs_ocr(p, p.get_text(), repair_malformed=False) for p in doc]
    assert with_repair == [True, True]
    assert without == [False, False]
