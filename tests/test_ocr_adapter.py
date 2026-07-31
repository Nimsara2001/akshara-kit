"""The OCR adapter.

Rasterisation and capability reporting are tested unconditionally. The tests
that actually invoke Tesseract are marked ``ocr`` and skip with an actionable
message when the Sinhala language pack is absent.
"""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit.adapters.extractors import ocr_adapter
from akshara_kit.eye import capabilities
from akshara_kit.eye.errors import OcrUnavailableError


# --- capability reporting -------------------------------------------------


def test_describe_ocr_availability_is_actionable() -> None:
    """Whatever the machine's state, the message must say what to do next."""
    message = capabilities.describe_ocr_availability()
    assert message
    if not capabilities.tesseract_available():
        assert "AKSHARA_TESSERACT_CMD" in message
    elif not capabilities.sinhala_ocr_available():
        assert "sin.traineddata" in message


def test_default_language_is_sinhala() -> None:
    """The prototype defaulted to 'eng', so its Sinhala OCR never ran."""
    assert ocr_adapter.DEFAULT_LANG == "sin"


# --- rasterisation (no Tesseract required) --------------------------------


def test_rasterise_page_produces_an_image(scanned_pdf: pathlib.Path) -> None:
    """The PyMuPDF path works with no poppler installed."""
    image = ocr_adapter.rasterise_page(str(scanned_pdf), 0, dpi=72)
    try:
        assert image.width > 0 and image.height > 0
    finally:
        image.close()


def test_rasterise_respects_dpi(scanned_pdf: pathlib.Path) -> None:
    low = ocr_adapter.rasterise_page(str(scanned_pdf), 0, dpi=72)
    high = ocr_adapter.rasterise_page(str(scanned_pdf), 0, dpi=144)
    try:
        assert high.width > low.width
    finally:
        low.close()
        high.close()


def test_rasterise_selects_the_right_page(mixed_pdf: pathlib.Path) -> None:
    for index in (0, 1):
        image = ocr_adapter.rasterise_page(str(mixed_pdf), index, dpi=72)
        image.close()


def test_pdf2image_path_reports_missing_poppler(
    scanned_pdf: pathlib.Path, monkeypatch
) -> None:
    monkeypatch.setenv("AKSHARA_RASTERISER", "pdf2image")
    monkeypatch.setattr(capabilities, "poppler_available", lambda: False)
    with pytest.raises(OcrUnavailableError, match="poppler"):
        ocr_adapter.rasterise_page(str(scanned_pdf), 0)


def test_missing_language_pack_raises_a_named_error(monkeypatch) -> None:
    """A missing dependency must not surface as a raw library exception."""
    monkeypatch.setattr(capabilities, "sinhala_ocr_available", lambda: False)
    monkeypatch.setattr(capabilities, "tesseract_available", lambda: True)
    with pytest.raises(OcrUnavailableError):
        ocr_adapter._require_ocr("sin")


# --- real OCR -------------------------------------------------------------


@pytest.mark.ocr
def test_ocr_reads_sinhala_from_a_scanned_page(scanned_pdf: pathlib.Path) -> None:
    from akshara_kit.eye.quality_probe import sinhala_ratio

    text = ocr_adapter.extract_page(str(scanned_pdf), 0)
    assert text.strip(), "OCR returned nothing"
    assert sinhala_ratio(text) > 0.3, f"expected Sinhala, got {text[:120]!r}"


@pytest.mark.ocr
def test_extract_whole_document(scanned_pdf: pathlib.Path) -> None:
    result = ocr_adapter.extract(str(scanned_pdf))
    assert result.ocr_used
    assert result.backend_id == ocr_adapter.BACKEND_ID
    assert result.text.strip()
