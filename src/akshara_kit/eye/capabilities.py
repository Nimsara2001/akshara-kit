"""Probes for optional system-level dependencies.

The prior prototype's OCR path never once executed: Tesseract was on the
machine but not wired into ``pytesseract``, poppler was absent, and neither
condition produced a diagnosable message. Everything here exists so that
failure mode is impossible — the probes are cached, side-effect-light and
never raise.

Environment overrides, for installs that are not on ``PATH``:

``AKSHARA_TESSERACT_CMD``
    Full path to ``tesseract.exe`` / ``tesseract``.
``AKSHARA_POPPLER_PATH``
    Directory containing ``pdftoppm``, for the optional pdf2image rasteriser.
"""

from __future__ import annotations

import functools
import os
import shutil

#: Standard Windows install locations, checked when PATH lookup fails.
_WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

#: The Tesseract language code for Sinhala.
SINHALA_LANG = "sin"


@functools.lru_cache(maxsize=1)
def resolve_tesseract_cmd() -> str | None:
    """Locate the Tesseract binary and point ``pytesseract`` at it.

    Checks ``AKSHARA_TESSERACT_CMD``, then ``PATH``, then the standard Windows
    install directories. Returns the resolved path, or ``None`` if Tesseract
    cannot be found. Safe to call when ``pytesseract`` is not installed.
    """
    candidates = [
        os.environ.get("AKSHARA_TESSERACT_CMD"),
        shutil.which("tesseract"),
        *_WINDOWS_TESSERACT_PATHS,
    ]
    command = next((c for c in candidates if c and os.path.isfile(c)), None)
    if command is None:
        return None

    try:
        import pytesseract
    except ImportError:
        return command
    pytesseract.pytesseract.tesseract_cmd = command
    return command


@functools.lru_cache(maxsize=1)
def tesseract_available() -> bool:
    """True if the Tesseract binary can be located."""
    return resolve_tesseract_cmd() is not None


@functools.lru_cache(maxsize=1)
def tesseract_languages() -> frozenset[str]:
    """Language packs Tesseract can see; empty if it cannot be queried."""
    if not tesseract_available():
        return frozenset()
    try:
        import pytesseract

        return frozenset(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001 - a probe must never raise
        return frozenset()


@functools.lru_cache(maxsize=1)
def sinhala_ocr_available() -> bool:
    """True if Tesseract is present *and* has the Sinhala language pack."""
    return SINHALA_LANG in tesseract_languages()


@functools.lru_cache(maxsize=1)
def poppler_available() -> bool:
    """True if poppler's ``pdftoppm`` is reachable.

    Only relevant to the optional ``pdf2image`` rasteriser; the default OCR
    path rasterises with PyMuPDF and needs no poppler.
    """
    if shutil.which("pdftoppm"):
        return True
    poppler_dir = os.environ.get("AKSHARA_POPPLER_PATH")
    return bool(poppler_dir and shutil.which("pdftoppm", path=poppler_dir))


def describe_ocr_availability() -> str:
    """A one-line, actionable explanation of the current OCR capability."""
    if not tesseract_available():
        return (
            "Tesseract binary not found. Install Tesseract OCR, or set "
            "AKSHARA_TESSERACT_CMD to its full path."
        )
    if not sinhala_ocr_available():
        found = ", ".join(sorted(tesseract_languages())) or "none"
        return (
            f"Tesseract found, but the Sinhala '{SINHALA_LANG}' language pack is "
            f"not installed (available: {found}). Download sin.traineddata into "
            "the tessdata directory."
        )
    return f"Tesseract with the '{SINHALA_LANG}' language pack is available."
