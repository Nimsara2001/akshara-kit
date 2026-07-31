"""Font classification.

``VERIFIED_SAMPLES`` is the provenance record for ``LEGACY_FONT_MAPPINGS``:
every font in the mappable tier appears here with real span text taken from
Sinhala school textbooks, and the conversion output was manually confirmed to
be correct Sinhala. Nothing is in that tier on inference alone.
"""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit.eye.encoding_normaliser import convert
from akshara_kit.eye.font_detection import (
    KNOWN_LEGACY_FONT_NAMES,
    LEGACY_FONT_MAPPINGS,
    FontClass,
    classify_font,
    detect_pdf_fonts,
    is_legacy,
    is_mappable,
    normalise_font_name,
)

#: font -> (real legacy span text, expected Sinhala after conversion)
VERIFIED_SAMPLES: dict[str, tuple[str, str]] = {
    "FMAbhaya": ("ish¨ u fm<fmd;a", "සියලු ම පෙළපොත්"),
    "FMAbhayax": ("wOHdmk m%ldYk", "අධ්‍යාපන ප්‍රකාශන"),
    "FMAbabldBold": ("Y%S ,xld cd;sl .Sh", "ශ්‍රී ලංකා ජාතික ගීය"),
    "FMBindumathix": ("jHdmdr úoHd mSGh", "ව්‍යාපාර විද්‍යා පීඨය"),
    "FMEmaneex": ("i;r lka uka;%Kh", "සතර කන් මන්ත්‍රණය"),
    "FMSamanthax": ("nqoaO O¾uh", "බුද්ධ ධර්මය"),
    "FMSamanthaBoldx": ("iod iqrlsuq", "සදා සුරකිමු"),
    "FMEdwerdBanceBold": ("th Tfí;a rfÜ;a hym;", "එය ඔබේත් රටේත් යහපත"),
    "FMPrabhathbox": ("kS;s úfrdaë ls%hdjls", "නීති විරෝධී ක්‍රියාවකි"),
}


# --- name normalisation ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FMAbhaya", "FMAbhaya"),
        ("CZMRSJ+FMAbhayax", "FMAbhayax"),  # PDF subset prefix
        ("GRAJJE+FMAbhaya", "FMAbhaya"),
        ("FM Abhaya", "FMAbhaya"),  # spaced spelling from the spec
        ("  FMAbhaya  ", "FMAbhaya"),
        ("Iskoola Pota Regular", "IskoolaPotaRegular"),
        ("", ""),
    ],
)
def test_normalise_font_name(raw: str, expected: str) -> None:
    assert normalise_font_name(raw) == expected


def test_subset_prefix_must_be_exactly_six_capitals() -> None:
    """A real name that happens to contain '+' must not be mangled."""
    assert normalise_font_name("ABC+Foo") == "ABC+Foo"
    assert normalise_font_name("ABCDEFG+Foo") == "ABCDEFG+Foo"


# --- classification -------------------------------------------------------

CLASSIFICATION_TABLE: list[tuple[str, FontClass]] = [
    # Mappable legacy — the workhorses of the real corpus.
    ("FMAbhaya", FontClass.LEGACY_MAPPABLE),
    ("FMAbhayax", FontClass.LEGACY_MAPPABLE),
    ("FMAbhaya-Bold", FontClass.LEGACY_MAPPABLE),
    ("FMAbabldBold", FontClass.LEGACY_MAPPABLE),
    ("FMBindumathix", FontClass.LEGACY_MAPPABLE),
    ("FMEmaneex", FontClass.LEGACY_MAPPABLE),
    ("FMSamanthax", FontClass.LEGACY_MAPPABLE),
    ("FMSamanthaBoldx", FontClass.LEGACY_MAPPABLE),
    ("FMMalithix", FontClass.LEGACY_MAPPABLE),
    ("FMPrabhathbox", FontClass.LEGACY_MAPPABLE),
    ("FMEdwerdBanceBold", FontClass.LEGACY_MAPPABLE),
    ("CZMRSJ+FMAbhayax", FontClass.LEGACY_MAPPABLE),  # prefixed
    ("FM Abhaya", FontClass.LEGACY_MAPPABLE),  # spaced
    # Legacy but NOT convertible — must be detected and passed through.
    ("Chamodi", FontClass.LEGACY_UNMAPPABLE),
    ("sandaru-n", FontClass.LEGACY_UNMAPPABLE),
    ("K-Plain", FontClass.LEGACY_UNMAPPABLE),
    ("Tharmini-Plain", FontClass.LEGACY_UNMAPPABLE),
    ("SHREE-TAM7-0800", FontClass.LEGACY_UNMAPPABLE),
    ("SHREE-TAM7-1316", FontClass.LEGACY_UNMAPPABLE),
    ("TAM-Tamil155", FontClass.LEGACY_UNMAPPABLE),
    # Already Unicode — the never-convert guard.
    ("IskoolaPota", FontClass.UNICODE_SINHALA),
    ("IskoolaPota-Bold", FontClass.UNICODE_SINHALA),
    ("Iskoola Pota Regular", FontClass.UNICODE_SINHALA),
    ("NirmalaUI", FontClass.UNICODE_SINHALA),
    ("NirmalaUI-Bold", FontClass.UNICODE_SINHALA),
    ("Nirmala UI", FontClass.UNICODE_SINHALA),
    ("Latha", FontClass.UNICODE_SINHALA),
    ("Latha,Bold", FontClass.UNICODE_SINHALA),
    # Latin and symbol fonts.
    ("TimesNewRomanPSMT", FontClass.NON_SINHALA),
    ("Times-Roman", FontClass.NON_SINHALA),
    ("Helvetica", FontClass.NON_SINHALA),
    ("ArialMT", FontClass.NON_SINHALA),
    ("MinionPro-Regular", FontClass.NON_SINHALA),
    ("Wingdings-Regular", FontClass.NON_SINHALA),
    # Unrecognised.
    ("SomeFontNobodyHasSeen", FontClass.UNKNOWN),
    ("", FontClass.UNKNOWN),
]


@pytest.mark.parametrize(("font", "expected"), CLASSIFICATION_TABLE)
def test_classify_font(font: str, expected: FontClass) -> None:
    assert classify_font(font) is expected


@pytest.mark.parametrize(
    "font", ["IskoolaPota", "NirmalaUI", "Latha", "TimesNewRomanPSMT", "Helvetica"]
)
def test_non_legacy_fonts_are_never_mappable(font: str) -> None:
    """The guard that stops correct text being destroyed."""
    assert is_mappable(font) == (False, None)
    assert not is_legacy(font)


@pytest.mark.parametrize("font", sorted(LEGACY_FONT_MAPPINGS))
def test_mappable_fonts_resolve_to_a_mapping(font: str) -> None:
    mappable, mapping = is_mappable(font)
    assert mappable
    assert mapping == "fm_abhaya"


@pytest.mark.parametrize(
    "font", ["Chamodi", "sandaru-n", "K-Plain", "Tharmini-Plain", "SHREE-TAM7-0800"]
)
def test_unmappable_legacy_is_detected_but_not_converted(font: str) -> None:
    assert is_legacy(font), "must still be reported as legacy"
    assert is_mappable(font) == (False, None), "must not be converted"


def test_fm_family_rule_is_off_by_default() -> None:
    """An unseen FM font is UNKNOWN, not silently assumed convertible."""
    assert classify_font("FMSomethingBrandNew") is FontClass.UNKNOWN


def test_known_legacy_font_names_alias_matches_spec_naming() -> None:
    """Section 6.1's constant name still resolves and stays in step."""
    assert KNOWN_LEGACY_FONT_NAMES == frozenset(LEGACY_FONT_MAPPINGS)
    assert "FMAbhaya" in KNOWN_LEGACY_FONT_NAMES


# --- provenance -----------------------------------------------------------


@pytest.mark.parametrize(("font", "sample"), sorted(VERIFIED_SAMPLES.items()))
def test_verified_sample_converts_to_expected_sinhala(
    font: str, sample: tuple[str, str]
) -> None:
    """Every mappable font's entry is backed by a real, checked conversion."""
    legacy, expected = sample
    _, mapping = is_mappable(font)
    assert convert(legacy, mapping) == expected


def test_every_verified_font_is_in_the_mappable_tier() -> None:
    assert set(VERIFIED_SAMPLES) <= set(LEGACY_FONT_MAPPINGS)


# --- PDF font detection ---------------------------------------------------


def test_detect_pdf_fonts_finds_the_legacy_font(legacy_pdf: pathlib.Path) -> None:
    fonts = detect_pdf_fonts(str(legacy_pdf))
    assert "FMAbhaya" in fonts
    assert "Helvetica" in fonts


def test_detect_pdf_fonts_finds_no_legacy_font_in_unicode_pdf(
    unicode_pdf: pathlib.Path,
) -> None:
    fonts = detect_pdf_fonts(str(unicode_pdf))
    assert fonts, "should still report the fonts it found"
    assert not any(is_legacy(font) for font in fonts), (
        f"a born-digital Unicode PDF needs no conversion, got {fonts}"
    )
