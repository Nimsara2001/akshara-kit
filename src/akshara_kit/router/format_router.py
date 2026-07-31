"""Format detection and dispatch (realises report Section 5).

Detection is extension-first but structure-authoritative: the extension only
decides which structural check to try first. When the two disagree, structure
wins and a warning is logged — a ``.pdf`` file that is really a zip is a zip.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from akshara_kit.contracts.extraction import ExtractionResult, SourceFormat
from akshara_kit.eye.errors import UnsupportedFormatError

__all__ = ["UnsupportedFormatError", "detect_format", "route"]

logger = logging.getLogger(__name__)

#: PDF header. Read a window rather than requiring offset 0 — real-world PDFs
#: sometimes carry leading bytes, and a strict prefix check rejects them.
_PDF_MAGIC = b"%PDF"
_PDF_SEARCH_WINDOW = 1024

#: Marker members that identify the two OOXML containers. DOCX and XLSX are
#: the same zip format, so only the member list tells them apart.
_ZIP_MARKERS: dict[str, SourceFormat] = {
    "word/document.xml": SourceFormat.DOCX,
    "xl/workbook.xml": SourceFormat.XLSX,
}

_EXTENSION_HINTS: dict[str, SourceFormat] = {
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
    ".xlsx": SourceFormat.XLSX,
}


def detect_format(file_path: str) -> SourceFormat:
    """Determine a file's true format from its structure.

    Implements Section 5.2. The extension is used only as a hint for the
    warning message; the returned format always reflects the file's actual
    contents.

    :raises UnsupportedFormatError: if the file matches no known format.
    """
    path = Path(file_path)
    if not path.is_file():
        raise UnsupportedFormatError(f"No such file: {file_path}")

    detected = _detect_by_structure(path)
    if detected is None:
        raise UnsupportedFormatError(
            f"{path.name} is neither a PDF nor an OOXML container "
            f"(leading bytes: {path.open('rb').read(8)!r})"
        )

    hinted = _EXTENSION_HINTS.get(path.suffix.lower())
    if hinted is not None and hinted is not detected:
        logger.warning(
            "%s has extension %s but is structurally %s; trusting the structure",
            path.name,
            path.suffix,
            detected.value,
        )
    return detected


def _detect_by_structure(path: Path) -> SourceFormat | None:
    """Return the format implied by the file's bytes, or ``None``."""
    if _is_pdf(path):
        return SourceFormat.PDF
    return _zip_format(path)


def _is_pdf(path: Path) -> bool:
    """True if the PDF header appears near the start of the file."""
    with path.open("rb") as handle:
        head = handle.read(_PDF_SEARCH_WINDOW)
    offset = head.find(_PDF_MAGIC)
    if offset < 0:
        return False
    if offset > 0:
        logger.warning("%s: %%PDF header found at offset %d, not 0", path.name, offset)
    return True


def _zip_format(path: Path) -> SourceFormat | None:
    """Classify an OOXML container by its marker members."""
    if not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
    except zipfile.BadZipFile:
        return None

    matched = [fmt for marker, fmt in _ZIP_MARKERS.items() if marker in members]
    if not matched:
        return None
    if len(matched) > 1:
        logger.warning(
            "%s contains markers for %s; using the first",
            path.name,
            ", ".join(fmt.value for fmt in matched),
        )
    return matched[0]


def route(file_path: str) -> ExtractionResult:
    """Extract a document's text, whatever its format.

    The documented public entry point of the Eye module (Section 5). Format
    dispatch itself lives in :mod:`akshara_kit.eye.coordinator`; this function
    is the name callers use.

    :raises UnsupportedFormatError: if the file is not a PDF, DOCX or XLSX.
    :raises ExtractionFailedError: if every applicable backend failed.
    """
    from akshara_kit.eye import coordinator

    return coordinator.extract(file_path)
