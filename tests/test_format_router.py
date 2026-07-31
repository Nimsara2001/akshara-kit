"""Format detection: correct classification, and structure over extension."""

from __future__ import annotations

import pathlib
import zipfile

import pytest

from akshara_kit.contracts.extraction import SourceFormat
from akshara_kit.router.format_router import UnsupportedFormatError, detect_format


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("sample_unicode.pdf", SourceFormat.PDF),
        ("sample_legacy_font.pdf", SourceFormat.PDF),
        ("sample_scanned.pdf", SourceFormat.PDF),
        ("sample.docx", SourceFormat.DOCX),
        ("sample.xlsx", SourceFormat.XLSX),
    ],
)
def test_detects_each_format(
    fixtures_dir: pathlib.Path, fixture_name: str, expected: SourceFormat
) -> None:
    assert detect_format(str(fixtures_dir / fixture_name)) is expected


def test_mislabeled_docx_named_pdf_is_detected_as_docx(
    sample_docx: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A zip wearing a .pdf extension must be classified by its structure."""
    liar = tmp_path / "actually_a_docx.pdf"
    liar.write_bytes(sample_docx.read_bytes())
    assert detect_format(str(liar)) is SourceFormat.DOCX


def test_mislabeled_xlsx_named_docx_is_detected_as_xlsx(
    sample_xlsx: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """DOCX and XLSX share a container; only the members distinguish them."""
    liar = tmp_path / "actually_a_xlsx.docx"
    liar.write_bytes(sample_xlsx.read_bytes())
    assert detect_format(str(liar)) is SourceFormat.XLSX


def test_mislabeled_pdf_named_docx_is_detected_as_pdf(
    unicode_pdf: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    liar = tmp_path / "actually_a_pdf.docx"
    liar.write_bytes(unicode_pdf.read_bytes())
    assert detect_format(str(liar)) is SourceFormat.PDF


def test_mislabeled_extension_logs_a_warning(
    sample_docx: pathlib.Path, tmp_path: pathlib.Path, caplog
) -> None:
    liar = tmp_path / "actually_a_docx.pdf"
    liar.write_bytes(sample_docx.read_bytes())
    with caplog.at_level("WARNING"):
        detect_format(str(liar))
    assert "trusting the structure" in caplog.text


def test_pdf_with_leading_junk_is_still_a_pdf(
    unicode_pdf: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Real PDFs sometimes carry preamble bytes; a strict prefix check fails."""
    padded = tmp_path / "padded.pdf"
    padded.write_bytes(b"\n\n   " + unicode_pdf.read_bytes())
    assert detect_format(str(padded)) is SourceFormat.PDF


def test_plain_text_is_rejected(tmp_path: pathlib.Path) -> None:
    plain = tmp_path / "notes.txt"
    plain.write_text("just some text", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError, match="neither a PDF nor an OOXML"):
        detect_format(str(plain))


def test_text_file_named_pdf_is_rejected(tmp_path: pathlib.Path) -> None:
    liar = tmp_path / "not_really.pdf"
    liar.write_text("this is not a PDF", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        detect_format(str(liar))


def test_unrelated_zip_is_rejected(tmp_path: pathlib.Path) -> None:
    """A zip without an OOXML marker member is not a document we handle."""
    archive = tmp_path / "photos.docx"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("holiday.jpg", b"\xff\xd8\xff\xe0")
    with pytest.raises(UnsupportedFormatError):
        detect_format(str(archive))


def test_missing_file_is_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(UnsupportedFormatError, match="No such file"):
        detect_format(str(tmp_path / "nope.pdf"))
