"""DOCX extraction: document order, tables, and per-run font handling."""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit.adapters.extractors import docx_adapter
from akshara_kit.contracts.extraction import FontDetectionMethod, SourceFormat
from akshara_kit.eye.errors import ExtractionFailedError
from samples import ASCII_CONTROL, UNICODE_SINHALA


@pytest.fixture(scope="module")
def result(sample_docx: pathlib.Path):
    return docx_adapter.extract(str(sample_docx))


def test_returns_a_valid_result(result) -> None:
    assert result.source_format is SourceFormat.DOCX
    assert result.backend_id == docx_adapter.BACKEND_ID
    assert result.text.strip()
    assert result.quality is not None


def test_table_content_is_extracted(result) -> None:
    """Section 6.3: do not skip tables."""
    assert result.text.count(UNICODE_SINHALA) >= 4, (
        "expected paragraph runs and table cells, got only some"
    )


def test_document_order_is_preserved(result) -> None:
    """python-docx exposes paragraphs and tables as separate collections;
    reading them in turn would put the table after all the prose."""
    before = result.text.index("Between the paragraph and the table.")
    after = result.text.index("After the table.")
    table_cell = result.text.index("\t")
    assert before < table_cell < after


def test_legacy_run_is_converted(result) -> None:
    assert result.font_detection_method is FontDetectionMethod.FONT_NAME
    assert result.detected_legacy_fonts == ["FMAbhaya", "FMBindumathix"]
    assert result.unmapped_legacy_fonts == []


def test_unicode_run_survives_in_the_same_paragraph(result) -> None:
    """The per-run requirement, stated as a behaviour.

    The first paragraph mixes an FMAbhaya run, an Iskoola Pota run and an
    unstyled run. Per-paragraph detection would either miss the legacy text or
    destroy the Unicode text; per-run detection gets both right.
    """
    first_paragraph = result.text.split("\n")[0]
    assert first_paragraph.count(UNICODE_SINHALA) == 2
    assert ASCII_CONTROL in first_paragraph


def test_run_count_covers_paragraphs_and_table_cells(result) -> None:
    assert result.metadata["runs"] == 9


def test_missing_file_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ExtractionFailedError):
        docx_adapter.extract(str(tmp_path / "nope.docx"))


def test_non_docx_raises(unicode_pdf: pathlib.Path) -> None:
    with pytest.raises(ExtractionFailedError):
        docx_adapter.extract(str(unicode_pdf))
