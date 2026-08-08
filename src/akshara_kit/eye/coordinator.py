"""The Eye Coordinator: one entry point for every supported format.

Detects the container, dispatches to the coordinator that owns it, and returns
a single :class:`~akshara_kit.contracts.extraction.ExtractionResult`. PDFs go
through a race over four text-stream backends; DOCX and XLSX have one
deterministic path each, so their adapters are their own coordinators.

:func:`akshara_kit.route` is the documented public name and delegates here.
"""

from __future__ import annotations

from typing import Callable

from akshara_kit.contracts.extraction import (
    ExtractionResult,
    MultimodalConfig,
    SourceFormat,
)

__all__ = ["SUPPORTED_FORMATS", "extract"]

SUPPORTED_FORMATS: frozenset[SourceFormat] = frozenset(SourceFormat)


def _coordinators() -> dict[SourceFormat, Callable[[str], ExtractionResult]]:
    """Map each format to the callable that handles it.

    Imported lazily so a caller who only reads spreadsheets never needs the
    PDF extras installed.
    """
    from akshara_kit.adapters.extractors import docx_adapter, xlsx_adapter
    from akshara_kit.eye import pdf_coordinator

    return {
        SourceFormat.PDF: pdf_coordinator.extract,
        SourceFormat.DOCX: docx_adapter.extract,
        SourceFormat.XLSX: xlsx_adapter.extract,
    }


def extract(
    file_path: str, *, multimodal: MultimodalConfig | None = None
) -> ExtractionResult:
    """Extract and normalise a document, whatever its format.

    ``multimodal`` reaches the PDF path only. DOCX and XLSX carry their text
    directly — there is no rendered page to photograph and nothing a vision
    model could recover that the container did not already hand over — so
    passing it for those formats is silently a no-op rather than an error.

    :raises UnsupportedFormatError: if the file is not a PDF, DOCX or XLSX.
    :raises ExtractionFailedError: if every applicable backend failed.
    """
    from akshara_kit.router.format_router import detect_format

    source_format = detect_format(file_path)
    if source_format is SourceFormat.PDF:
        from akshara_kit.eye import pdf_coordinator

        return pdf_coordinator.extract(file_path, multimodal=multimodal)
    return _coordinators()[source_format](file_path)
