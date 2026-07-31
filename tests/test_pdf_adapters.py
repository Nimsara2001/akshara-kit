"""The four PDF backends and the coordinator that races them."""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit.adapters.extractors import (
    pdfminer_adapter,
    pdfplumber_adapter,
    pymupdf_adapter,
    pypdf_adapter,
)
from akshara_kit.contracts.extraction import SourceFormat
from akshara_kit.eye import pdf_coordinator
from akshara_kit.eye.errors import ExtractionFailedError
from samples import ASCII_CONTROL, UNICODE_SINHALA, UNICODE_SINHALA_2

ADAPTERS = [
    pypdf_adapter,
    pymupdf_adapter,
    pdfplumber_adapter,
    pdfminer_adapter,
]

ADAPTER_IDS = [module.BACKEND_ID for module in ADAPTERS]


@pytest.fixture
def corrupt_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    """A file that starts like a PDF but is structurally broken."""
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not really a PDF at all\n")
    return path


# --- each adapter ---------------------------------------------------------


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
def test_adapter_extracts_text(adapter, unicode_pdf: pathlib.Path) -> None:
    result = adapter.extract(str(unicode_pdf))
    assert result.text.strip(), f"{adapter.BACKEND_ID} returned nothing"
    assert result.backend_id == adapter.BACKEND_ID
    assert result.source_format is SourceFormat.PDF
    assert result.latency_seconds >= 0
    assert result.quality is not None


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
def test_adapter_finds_the_sinhala_text(adapter, unicode_pdf: pathlib.Path) -> None:
    # UNICODE_SINHALA_2 contains no zero-width joiner; see the test below for
    # why that matters when comparing backends.
    assert UNICODE_SINHALA_2 in adapter.extract(str(unicode_pdf)).text


def test_pdfminer_splits_lines_at_the_zero_width_joiner(
    unicode_pdf: pathlib.Path,
) -> None:
    """A real backend difference, pinned so it is a known quantity.

    pdfminer treats U+200D as a line break opportunity, so a conjunct like
    ``ප්‍රකාශන`` arrives split across lines. The characters all survive — the
    Sinhala ratio is unaffected — but consumers that match on whole phrases
    will see a difference between pdfminer and the other three backends.
    """
    text = pdfminer_adapter.extract(str(unicode_pdf)).text
    assert UNICODE_SINHALA not in text, "quirk has gone away; simplify this test"
    assert all(part in text for part in UNICODE_SINHALA.split("‍"))

    for adapter in (pypdf_adapter, pymupdf_adapter, pdfplumber_adapter):
        assert UNICODE_SINHALA in adapter.extract(str(unicode_pdf)).text


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
def test_adapter_raises_rather_than_returning_an_error_string(
    adapter, corrupt_pdf: pathlib.Path
) -> None:
    """The prototype wrote "[ERROR] ..." into `text`, which the coordinator
    then scored and could return as if it were a real extraction."""
    with pytest.raises(ExtractionFailedError):
        adapter.extract(str(corrupt_pdf))


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
def test_adapter_reports_pages_separately(adapter, mixed_pdf: pathlib.Path) -> None:
    """Per-page access is what lets OCR output merge back in the right order."""
    pages = adapter._extract_pages(str(mixed_pdf))
    assert len(pages) == 2, f"{adapter.BACKEND_ID} lost the page structure"
    assert pages[0].strip(), "page 0 is born-digital"
    assert not pages[1].strip(), "page 1 is image-only"


# --- PyMuPDF's span path --------------------------------------------------


def test_iter_spans_pairs_text_with_its_font(legacy_pdf: pathlib.Path) -> None:
    spans = [s for s in pymupdf_adapter.iter_spans(str(legacy_pdf)) if not s.is_separator]
    by_font = {span.font: span.text for span in spans}
    assert "FMAbhaya" in by_font
    assert by_font["Helvetica"] == ASCII_CONTROL


def test_iter_spans_preserves_layout(unicode_pdf: pathlib.Path) -> None:
    """Joined spans must not be worse than the flat text they replace."""
    joined = "".join(s.text for s in pymupdf_adapter.iter_spans(str(unicode_pdf)))
    flat = pymupdf_adapter.extract(str(unicode_pdf)).text
    assert joined.split() == flat.split()
    assert joined.count("\n") >= flat.count("\n") - 1


def test_page_count(mixed_pdf: pathlib.Path) -> None:
    assert pymupdf_adapter.page_count(str(mixed_pdf)) == 2


# --- the coordinator ------------------------------------------------------


def test_race_records_every_attempt(unicode_pdf: pathlib.Path) -> None:
    result = pdf_coordinator.extract(str(unicode_pdf))
    assert {a.backend_id for a in result.attempts} == set(ADAPTER_IDS)
    assert all(a.succeeded for a in result.attempts)


def test_race_is_deterministic(unicode_pdf: pathlib.Path) -> None:
    """Ranking is by fixed registry order, never by completion order."""
    winners = {pdf_coordinator.extract(str(unicode_pdf)).backend_id for _ in range(3)}
    assert len(winners) == 1


def test_sequential_and_parallel_agree(unicode_pdf: pathlib.Path) -> None:
    parallel = pdf_coordinator.extract(str(unicode_pdf), parallel=True)
    sequential = pdf_coordinator.extract(str(unicode_pdf), parallel=False)
    assert parallel.backend_id == sequential.backend_id
    assert parallel.text == sequential.text


def test_unicode_pdf_needs_no_conversion(unicode_pdf: pathlib.Path) -> None:
    result = pdf_coordinator.extract(str(unicode_pdf))
    assert result.detected_legacy_fonts == []
    assert result.quality.sinhala_ratio > 0.5


def test_legacy_pdf_is_converted_without_collateral_damage(
    legacy_pdf: pathlib.Path,
) -> None:
    """The thesis of this design, in one test.

    The legacy span becomes Sinhala; the Helvetica span beside it is untouched.
    """
    result = pdf_coordinator.extract(str(legacy_pdf))

    assert result.backend_id == pdf_coordinator.SPAN_BACKEND_ID
    assert result.detected_legacy_fonts == ["FMAbhaya"]
    assert result.quality.sinhala_ratio > 0.5, "conversion should lift the ratio"
    assert UNICODE_SINHALA in result.text
    assert ASCII_CONTROL in result.text, "the Latin span was corrupted"


def test_legacy_pdf_records_the_raw_race_winner(legacy_pdf: pathlib.Path) -> None:
    """The span override must not hide which adapter won on raw text."""
    result = pdf_coordinator.extract(str(legacy_pdf))
    assert result.metadata["raw_race_winner"] in ADAPTER_IDS
    assert set(result.metadata["candidates"]) >= set(ADAPTER_IDS)


def test_every_adapter_failing_raises(corrupt_pdf: pathlib.Path) -> None:
    with pytest.raises(ExtractionFailedError, match="Every PDF adapter failed"):
        pdf_coordinator.extract(str(corrupt_pdf))


def test_a_short_high_ratio_candidate_cannot_win() -> None:
    """A backend returning one Sinhala character scores 1.0; it must still lose."""
    from akshara_kit.eye.pdf_coordinator import _Candidate, _select
    from akshara_kit.eye.quality_probe import score

    tiny = _Candidate("tiny", "ක", score("ක"), cost=1)
    full = _Candidate("full", UNICODE_SINHALA * 4, score(UNICODE_SINHALA * 4), cost=4)
    assert _select({"tiny": tiny, "full": full}).backend_id == "full"


def test_cheapest_adapter_wins_an_exact_tie() -> None:
    from akshara_kit.eye.pdf_coordinator import _Candidate, _select
    from akshara_kit.eye.quality_probe import score

    text = UNICODE_SINHALA * 4
    cheap = _Candidate("cheap", text, score(text), cost=1)
    dear = _Candidate("dear", text, score(text), cost=9)
    assert _select({"dear": dear, "cheap": cheap}).backend_id == "cheap"
