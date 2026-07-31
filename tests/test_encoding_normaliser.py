"""Legacy normalisation — above all, the things it must NOT do.

The prior prototype converted whole documents unconditionally. These tests pin
down the three ways that corrupts text, so the behaviour cannot regress.
"""

from __future__ import annotations

import pytest

from akshara_kit.contracts.extraction import FontDetectionMethod
from akshara_kit.eye.encoding_normaliser import (
    available_mappings,
    convert,
    normalise,
    normalise_legacy,
    normalise_spans,
)
from akshara_kit.eye.font_detection import SpanFont, separator
from samples import ASCII_CONTROL, LEGACY_SINHALA, UNICODE_SINHALA


def span(text: str, font: str) -> SpanFont:
    return SpanFont(text=text, font=font)


# --- the conversion table itself ------------------------------------------


def test_available_mappings_reports_what_pandukabhaya_ships() -> None:
    mappings = available_mappings()
    assert "fm_abhaya" in mappings


def test_convert_maps_legacy_bytes_to_sinhala() -> None:
    assert convert(LEGACY_SINHALA) == UNICODE_SINHALA


def test_convert_is_a_blunt_instrument() -> None:
    """Documents the hazard that gates every other function in this module.

    ``convert`` has no detection. Applied to ASCII it produces garbage — which
    is exactly what the prototype did to every document it touched.
    """
    assert convert(ASCII_CONTROL) != ASCII_CONTROL


def test_unknown_mapping_name_raises_a_clear_error() -> None:
    with pytest.raises(ValueError, match="no mapping named"):
        convert("x", mapping="dl_manel_does_not_exist_yet")


# --- span mode: the non-corruption regressions ----------------------------


def test_ascii_survives_beside_converted_legacy_text() -> None:
    """The `www.edupub.gov.lk` regression, stated directly."""
    outcome = normalise_spans(
        [
            span(LEGACY_SINHALA, "FMAbhaya"),
            separator(" "),
            span(ASCII_CONTROL, "Helvetica"),
        ]
    )
    assert UNICODE_SINHALA in outcome.text
    assert ASCII_CONTROL in outcome.text, "ASCII was corrupted by conversion"
    assert outcome.method is FontDetectionMethod.FONT_NAME
    assert outcome.converted_fonts == ["FMAbhaya"]


def test_existing_unicode_survives_beside_converted_legacy_text() -> None:
    """Blanket conversion mangles correct Sinhala; span mode must not."""
    outcome = normalise_spans(
        [
            span(LEGACY_SINHALA, "FMAbhaya"),
            separator(" "),
            span(UNICODE_SINHALA, "Iskoola Pota"),
        ]
    )
    assert outcome.text == f"{UNICODE_SINHALA} {UNICODE_SINHALA}"


def test_unknown_font_text_passes_through_byte_identical() -> None:
    """Section 6.4: never guess a mapping — pass through unchanged instead."""
    text = "ffjµ ks¾foaY rys;j"
    outcome = normalise_spans([span(text, "SomeFontNobodyHasSeen")])
    assert outcome.text == text
    assert outcome.method is FontDetectionMethod.NONE
    assert outcome.converted_fonts == []


def test_unmappable_legacy_font_is_reported_but_not_converted() -> None:
    """K-Plain converts *almost* correctly, which is why it must not be tried."""
    text = "ffjµ ks¾foaY rys;j"
    outcome = normalise_spans([span(text, "K-Plain")])
    assert outcome.text == text, "unmappable legacy text must not be converted"
    assert outcome.unmapped_fonts == ["K-Plain"]
    assert outcome.converted_fonts == []


def test_unmapped_font_is_logged_by_name(caplog) -> None:
    with caplog.at_level("WARNING"):
        normalise_spans([span("x", "Tharmini-Plain")])
    assert "Tharmini-Plain" in caplog.text


def test_separators_are_never_converted() -> None:
    outcome = normalise_spans([separator("\n\t"), span(LEGACY_SINHALA, "FMAbhaya")])
    assert outcome.text.startswith("\n\t")


def test_empty_span_list_is_handled() -> None:
    outcome = normalise_spans([])
    assert outcome.text == ""
    assert outcome.method is FontDetectionMethod.NONE


# --- document mode --------------------------------------------------------


def test_document_mode_converts_on_a_known_font_name() -> None:
    outcome = normalise(LEGACY_SINHALA, {"FMAbhaya"})
    assert outcome.text == UNICODE_SINHALA
    assert outcome.method is FontDetectionMethod.FONT_NAME


def test_document_mode_leaves_unicode_alone() -> None:
    outcome = normalise(UNICODE_SINHALA, {"IskoolaPota"})
    assert outcome.text == UNICODE_SINHALA
    assert outcome.method is FontDetectionMethod.NONE


def test_ratio_fallback_fires_on_a_legacy_stream_with_no_font_signal() -> None:
    outcome = normalise(LEGACY_SINHALA, set())
    assert outcome.method is FontDetectionMethod.CHARACTER_RATIO
    assert outcome.text == UNICODE_SINHALA


def test_ratio_fallback_does_not_destroy_english() -> None:
    """Section 6.4's heuristic as literally written would mangle this.

    A low Sinhala ratio alone is not evidence of legacy glyphs — plain English
    scores 0.0 too. The extra gates are what stop this text being converted.
    """
    english = "From the government, I received a letter about the new syllabus."
    outcome = normalise(english, set())
    assert outcome.text == english
    assert outcome.method is FontDetectionMethod.NONE


def test_ratio_fallback_is_suppressed_by_a_unicode_font_signal() -> None:
    outcome = normalise("....", {"IskoolaPota"})
    assert outcome.text == "...."
    assert outcome.method is FontDetectionMethod.NONE


def test_ratio_fallback_can_be_disabled() -> None:
    outcome = normalise(LEGACY_SINHALA, set(), allow_ratio_fallback=False)
    assert outcome.text == LEGACY_SINHALA
    assert outcome.method is FontDetectionMethod.NONE


def test_unmappable_font_blocks_document_mode_conversion() -> None:
    outcome = normalise(LEGACY_SINHALA, {"Tharmini-Plain"})
    assert outcome.text == LEGACY_SINHALA
    assert outcome.unmapped_fonts == ["Tharmini-Plain"]


# --- spec-shaped shim -----------------------------------------------------


def test_normalise_legacy_returns_the_spec_three_tuple() -> None:
    text, method, fonts = normalise_legacy(LEGACY_SINHALA, {"FMAbhaya"})
    assert text == UNICODE_SINHALA
    assert method is FontDetectionMethod.FONT_NAME
    assert fonts == ["FMAbhaya"]
