"""The Eye module: file-type detection, extraction, font handling and OCR."""

from akshara_kit.eye.errors import (
    AdapterUnavailableError,
    AksharaKitError,
    ExtractionFailedError,
    OcrUnavailableError,
    UnmappedLegacyFontError,
    UnsupportedFormatError,
)

__all__ = [
    "AdapterUnavailableError",
    "AksharaKitError",
    "ExtractionFailedError",
    "OcrUnavailableError",
    "UnmappedLegacyFontError",
    "UnsupportedFormatError",
]
