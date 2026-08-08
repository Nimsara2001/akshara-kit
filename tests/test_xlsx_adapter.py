"""XLSX extraction: multiple sheets, per-cell fonts, and empty cells."""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit.adapters.extractors import xlsx_adapter
from akshara_kit.contracts.extraction import FontDetectionMethod, SourceFormat
from akshara_kit.eye.errors import ExtractionFailedError
from samples import ASCII_CONTROL, UNICODE_SINHALA, UNICODE_SINHALA_2


@pytest.fixture(scope="module")
def result(sample_xlsx: pathlib.Path):
    return xlsx_adapter.extract(str(sample_xlsx))


def test_returns_a_valid_result(result) -> None:
    assert result.source_format is SourceFormat.XLSX
    assert result.backend_id == xlsx_adapter.BACKEND_ID
    assert result.quality is not None


def test_both_sheets_are_extracted_in_order(result) -> None:
    assert result.metadata["sheets"] == ["Legacy", "Unicode"]
    assert result.text.index("=== Legacy ===") < result.text.index("=== Unicode ===")


def test_content_from_every_sheet_is_present(result) -> None:
    assert UNICODE_SINHALA in result.text  # Legacy sheet, after conversion
    assert UNICODE_SINHALA_2 in result.text  # Unicode sheet, untouched
    assert "42" in result.text


def test_legacy_cells_are_converted(result) -> None:
    assert result.font_detection_method is FontDetectionMethod.FONT_NAME
    assert result.detected_legacy_fonts == ["FMAbhaya", "FMBindumathix"]


def test_default_font_cell_is_not_corrupted(result) -> None:
    """B1 carries the ASCII control string and inherits the workbook font."""
    assert ASCII_CONTROL in result.text


def test_cells_in_a_row_stay_on_one_line(result) -> None:
    row = next(line for line in result.text.splitlines() if ASCII_CONTROL in line)
    assert row.startswith(UNICODE_SINHALA)
    assert "\t" in row


def test_empty_rows_are_skipped(result) -> None:
    """The Unicode sheet leaves A2 empty between two populated rows."""
    assert "\n\n\n" not in result.text.split("=== Unicode ===")[1]


def test_missing_file_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ExtractionFailedError):
        xlsx_adapter.extract(str(tmp_path / "nope.xlsx"))


def test_non_xlsx_raises(unicode_pdf: pathlib.Path) -> None:
    with pytest.raises(ExtractionFailedError):
        xlsx_adapter.extract(str(unicode_pdf))
