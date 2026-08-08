"""Guard tests: the committed fixtures still have the properties tests rely on.

These are cheap structural assertions, not behaviour tests. They exist so a
stale or accidentally re-saved fixture fails here with a clear message rather
than causing a confusing failure somewhere deep in the extraction tests.

Regenerate with ``uv run python tests/fixtures/generate_fixtures.py``.
"""

from __future__ import annotations

import pathlib

import pymupdf
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

ASCII_CONTROL = "www.edupub.gov.lk"

ALL_FIXTURES = [
    "sample_unicode.pdf",
    "sample_legacy_font.pdf",
    "sample_scanned.pdf",
    "sample_mixed.pdf",
    "sample.docx",
    "sample.xlsx",
]


def _spans(page: pymupdf.Page) -> list[tuple[str, str]]:
    return [
        (span["font"], span["text"])
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]


def _largest_image_coverage(page: pymupdf.Page) -> float:
    page_area = abs(page.rect)
    if not page_area:
        return 0.0
    return max(
        (
            abs(pymupdf.Rect(info["bbox"]) & page.rect) / page_area
            for info in page.get_image_info()
        ),
        default=0.0,
    )


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_fixture_exists_and_is_small(name: str) -> None:
    path = FIXTURES / name
    assert path.is_file(), f"missing fixture {name}; run generate_fixtures.py"
    assert path.stat().st_size > 0
    # Fixtures are committed, so they must stay small.
    assert path.stat().st_size < 200_000, f"{name} has grown to {path.stat().st_size} bytes"


@pytest.mark.parametrize(
    ("name", "magic"),
    [
        ("sample_unicode.pdf", b"%PDF"),
        ("sample_legacy_font.pdf", b"%PDF"),
        ("sample_scanned.pdf", b"%PDF"),
        ("sample_mixed.pdf", b"%PDF"),
        ("sample.docx", b"PK"),
        ("sample.xlsx", b"PK"),
    ],
)
def test_fixture_magic_bytes(name: str, magic: bytes) -> None:
    assert (FIXTURES / name).read_bytes().startswith(magic)


def test_legacy_fixture_reports_font_name_and_raw_bytes() -> None:
    """The whole legacy design depends on PyMuPDF surfacing both of these."""
    with pymupdf.open(FIXTURES / "sample_legacy_font.pdf") as doc:
        spans = _spans(doc[0])

    fonts = {font for font, _ in spans}
    assert "FMAbhaya" in fonts, f"expected a FMAbhaya span, got {fonts}"
    assert "Helvetica" in fonts, f"expected a Helvetica span, got {fonts}"

    legacy_text = next(text for font, text in spans if font == "FMAbhaya")
    assert "wOHdmk" in legacy_text, "legacy bytes did not survive into the span"
    assert not any("඀" <= ch <= "෿" for ch in legacy_text), (
        "legacy span should contain no Sinhala Unicode before conversion"
    )

    ascii_text = next(text for font, text in spans if font == "Helvetica")
    assert ascii_text == ASCII_CONTROL


def test_unicode_fixture_is_born_digital_sinhala() -> None:
    with pymupdf.open(FIXTURES / "sample_unicode.pdf") as doc:
        page = doc[0]
        text = page.get_text()
        assert not page.get_image_info(), "unicode fixture should have no images"

    sinhala = sum("඀" <= ch <= "෿" for ch in text)
    assert sinhala > 30, f"expected substantial Sinhala Unicode, found {sinhala} chars"
    assert ASCII_CONTROL in text


def test_scanned_fixture_has_no_text_layer() -> None:
    with pymupdf.open(FIXTURES / "sample_scanned.pdf") as doc:
        page = doc[0]
        assert not page.get_text().strip()
        assert _largest_image_coverage(page) > 0.95


def test_mixed_fixture_is_per_page_heterogeneous() -> None:
    with pymupdf.open(FIXTURES / "sample_mixed.pdf") as doc:
        assert len(doc) == 2
        assert doc[0].get_text().strip(), "page 0 should be born-digital"
        assert _largest_image_coverage(doc[0]) == 0.0
        assert not doc[1].get_text().strip(), "page 1 should be image-only"
        assert _largest_image_coverage(doc[1]) > 0.95
