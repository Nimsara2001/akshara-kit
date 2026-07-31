"""Shared fixtures and capability-based skipping."""

from __future__ import annotations

import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config, items) -> None:
    """Skip tests whose system dependencies are absent, with a usable reason."""
    from akshara_kit.eye import capabilities

    if not capabilities.sinhala_ocr_available():
        skip_ocr = pytest.mark.skip(reason=capabilities.describe_ocr_availability())
        for item in items:
            if "ocr" in item.keywords:
                item.add_marker(skip_ocr)

    if not capabilities.poppler_available():
        skip_poppler = pytest.mark.skip(
            reason="poppler (pdftoppm) is not on PATH; set AKSHARA_POPPLER_PATH"
        )
        for item in items:
            if "poppler" in item.keywords:
                item.add_marker(skip_poppler)


@pytest.fixture(scope="session")
def fixtures_dir() -> pathlib.Path:
    return FIXTURES


@pytest.fixture(scope="session")
def unicode_pdf() -> pathlib.Path:
    return FIXTURES / "sample_unicode.pdf"


@pytest.fixture(scope="session")
def legacy_pdf() -> pathlib.Path:
    return FIXTURES / "sample_legacy_font.pdf"


@pytest.fixture(scope="session")
def scanned_pdf() -> pathlib.Path:
    return FIXTURES / "sample_scanned.pdf"


@pytest.fixture(scope="session")
def mixed_pdf() -> pathlib.Path:
    return FIXTURES / "sample_mixed.pdf"


@pytest.fixture(scope="session")
def sample_docx() -> pathlib.Path:
    return FIXTURES / "sample.docx"


@pytest.fixture(scope="session")
def sample_xlsx() -> pathlib.Path:
    return FIXTURES / "sample.xlsx"
