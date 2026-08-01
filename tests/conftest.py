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

    # `vlm` tests spend real money against a real API. Skipped unless a key is
    # configured, so a clean clone never bills anyone by surprise.
    from akshara_kit.multimodal.fallback import PROVIDERS

    if not any(capabilities.multimodal_available(p) for p in PROVIDERS):
        skip_vlm = pytest.mark.skip(
            reason="no multimodal API key configured; set e.g. AKSHARA_GEMINI_API_KEY"
        )
        for item in items:
            if "vlm" in item.keywords:
                item.add_marker(skip_vlm)


@pytest.fixture(scope="session")
def fixtures_dir() -> pathlib.Path:
    return FIXTURES


@pytest.fixture(scope="session")
def routed(fixtures_dir: pathlib.Path):
    """``route()`` a fixture once per session, then serve the cached result.

    Extraction is expensive — four PDF backends per document, plus OCR for any
    page whose text layer is missing or garbled — and the integration tests
    read the same few results many times over. ``route`` is deterministic and
    the results are never mutated, so caching changes nothing but the clock.
    """
    from akshara_kit import route

    cache: dict[str, object] = {}

    def _routed(name: str):
        if name not in cache:
            cache[name] = route(str(fixtures_dir / name))
        return cache[name]

    return _routed


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
