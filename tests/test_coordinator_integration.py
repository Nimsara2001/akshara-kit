"""End-to-end ``route()`` over every fixture.

The acceptance suite for Section 13's definition of done.
"""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit import route
from akshara_kit.contracts.extraction import (
    ExtractionResult,
    FontDetectionMethod,
    SourceFormat,
)
from akshara_kit.eye import capabilities
from samples import ASCII_CONTROL, UNICODE_SINHALA, UNICODE_SINHALA_2

pytestmark = pytest.mark.integration

EXPECTED_FORMATS = [
    ("sample_unicode.pdf", SourceFormat.PDF),
    ("sample_legacy_font.pdf", SourceFormat.PDF),
    ("sample_scanned.pdf", SourceFormat.PDF),
    ("sample_mixed.pdf", SourceFormat.PDF),
    ("sample.docx", SourceFormat.DOCX),
    ("sample.xlsx", SourceFormat.XLSX),
]


@pytest.mark.parametrize(("name", "expected"), EXPECTED_FORMATS)
def test_route_returns_a_valid_result(
    fixtures_dir: pathlib.Path, name: str, expected: SourceFormat
) -> None:
    result = route(str(fixtures_dir / name))
    assert isinstance(result, ExtractionResult)
    assert result.source_format is expected
    assert result.backend_id
    assert result.latency_seconds >= 0
    assert result.quality is not None
    assert 0.0 <= result.quality.sinhala_ratio <= 1.0
    # A result must always round-trip through the Brain-facing contract.
    assert ExtractionResult.model_validate(result.model_dump()) == result


@pytest.mark.parametrize(
    "name", ["sample_unicode.pdf", "sample.docx", "sample.xlsx", "sample_mixed.pdf"]
)
def test_documents_with_text_yield_plausible_sinhala(
    fixtures_dir: pathlib.Path, name: str
) -> None:
    result = route(str(fixtures_dir / name))
    assert result.quality.sinhala_ratio > 0.3, (
        f"{name} should be mostly Sinhala, got {result.quality.sinhala_ratio:.3f}"
    )


# --- the two regressions this design exists to prevent --------------------


@pytest.mark.parametrize(
    "name", ["sample_legacy_font.pdf", "sample.docx", "sample.xlsx"]
)
def test_latin_text_is_never_corrupted_by_conversion(
    fixtures_dir: pathlib.Path, name: str
) -> None:
    """The prototype turned this exact URL into 'අඅඅගැාමචමඉගටදඩගකන'."""
    result = route(str(fixtures_dir / name))
    assert result.font_detection_method is FontDetectionMethod.FONT_NAME
    assert ASCII_CONTROL in result.text


@pytest.mark.parametrize("name", ["sample.docx", "sample.xlsx"])
def test_existing_unicode_is_never_corrupted_by_conversion(
    fixtures_dir: pathlib.Path, name: str
) -> None:
    """Blanket conversion mangles correct Sinhala; span gating must not."""
    result = route(str(fixtures_dir / name))
    assert UNICODE_SINHALA in result.text


def test_legacy_pdf_conversion_lifts_the_sinhala_ratio(
    legacy_pdf: pathlib.Path,
) -> None:
    """Raw legacy text scores 0.0; after conversion it should be mostly Sinhala."""
    from akshara_kit.adapters.extractors import pymupdf_adapter
    from akshara_kit.eye.quality_probe import sinhala_ratio

    raw = pymupdf_adapter.extract(str(legacy_pdf)).text
    assert sinhala_ratio(raw) == 0.0

    result = route(str(legacy_pdf))
    assert result.quality.sinhala_ratio > 0.5
    assert UNICODE_SINHALA in result.text


# --- OCR routing ----------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["sample_unicode.pdf", "sample.docx", "sample.xlsx"]
)
def test_born_digital_documents_never_use_ocr(
    fixtures_dir: pathlib.Path, name: str
) -> None:
    result = route(str(fixtures_dir / name))
    assert not result.ocr_used
    assert result.pages_ocred == []


@pytest.mark.ocr
def test_scanned_pdf_uses_ocr(scanned_pdf: pathlib.Path) -> None:
    result = route(str(scanned_pdf))
    assert result.ocr_used
    assert result.pages_ocred == [0]
    assert result.text.strip()


@pytest.mark.ocr
def test_mixed_pdf_ocrs_only_the_scanned_page(mixed_pdf: pathlib.Path) -> None:
    """Per-page routing, and the merge keeps original page order."""
    result = route(str(mixed_pdf))
    assert result.pages_ocred == [1]
    assert result.text.index(UNICODE_SINHALA_2) < len(result.text) // 2


def test_scanned_pdf_degrades_gracefully_without_ocr(
    scanned_pdf: pathlib.Path,
) -> None:
    """Missing OCR must not fail the document — it returns what it could get."""
    if capabilities.sinhala_ocr_available():
        pytest.skip("OCR is available; this covers the degraded path")
    result = route(str(scanned_pdf))
    assert not result.ocr_used
    assert result.text == ""


# --- audit trail ----------------------------------------------------------


def test_pdf_results_carry_the_full_attempt_history(unicode_pdf: pathlib.Path) -> None:
    """Report Section 4.5's auditable extraction history."""
    result = route(str(unicode_pdf))
    assert len(result.attempts) == 4
    assert all(a.quality is not None for a in result.attempts if a.succeeded)


def test_unsupported_format_is_rejected(tmp_path: pathlib.Path) -> None:
    from akshara_kit import UnsupportedFormatError

    plain = tmp_path / "notes.txt"
    plain.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        route(str(plain))


# --- stubs ----------------------------------------------------------------


def test_multimodal_fallback_raises_clearly() -> None:
    from akshara_kit.multimodal import fallback

    with pytest.raises(NotImplementedError, match="multimodal"):
        fallback.extract("anything.pdf")


def test_layout_analyser_raises_clearly() -> None:
    from akshara_kit.layout import analyser

    with pytest.raises(NotImplementedError, match="layout analysis"):
        analyser.analyse("anything.pdf")
