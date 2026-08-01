"""Data contracts for the Eye module (realises report Table 5.1).

These models are the seam between the Eye module and the future Brain module
(semantic chunking). The field names in :class:`ExtractionResult`,
:class:`QualityScore`, :class:`SourceFormat` and :class:`FontDetectionMethod`
are frozen — downstream code depends on them, so extend rather than rename.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceFormat(str, Enum):
    """The document container the text was extracted from."""

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"


class FontDetectionMethod(str, Enum):
    """How the legacy-encoding decision was reached.

    ``FONT_NAME`` means an embedded font name matched a known legacy family.
    ``CHARACTER_RATIO`` means no font signal was available and the Sinhala
    Unicode ratio heuristic fired instead. ``NONE`` means no conversion.
    """

    FONT_NAME = "font_name"
    CHARACTER_RATIO = "character_ratio"
    NONE = "none"


class QualityScore(BaseModel):
    """Implements Algorithm 2 (Sinhala-Aware Quality Probe).

    ``sinhala_ratio`` is the dominant signal: a well-formed Sinhala document is
    overwhelmingly composed of U+0D80–U+0DFF characters once legacy glyph
    streams have been normalised away.

    ``orphan_vowel_rate`` is the second signal, and it catches what the first
    cannot. A PDF with a broken ``ToUnicode`` cmap yields text that is entirely
    Sinhala code points — a high ``sinhala_ratio`` — while being the *wrong*
    Sinhala code points. Orthographic well-formedness separates the two.
    """

    raw_length: int
    sinhala_ratio: float
    region_coverage: float | None = None
    sample_tokens: list[str] = Field(default_factory=list)

    #: Share of Sinhala consonants carrying a dependent vowel sign that has
    #: nothing to attach to. ~0 for correct text, percent-level for a garbled
    #: text layer. Additive beyond the frozen core contract.
    orphan_vowel_rate: float = 0.0


class AdapterAttempt(BaseModel):
    """One entry in the coordinator's auditable extraction history.

    Recorded for every adapter in the race, successful or not. Failures are
    captured here rather than being folded into ``ExtractionResult.text`` —
    an error string in the text field is indistinguishable from real output.
    """

    backend_id: str
    succeeded: bool
    latency_seconds: float
    quality: QualityScore | None = None
    error_type: str | None = None
    error_message: str | None = None


class ExtractionResult(BaseModel):
    """Canonical output of the Eye module for a single document."""

    text: str
    backend_id: str
    source_format: SourceFormat
    latency_seconds: float
    quality: QualityScore | None = None
    font_detection_method: FontDetectionMethod = FontDetectionMethod.NONE
    detected_legacy_fonts: list[str] = Field(default_factory=list)
    ocr_used: bool = False
    metadata: dict = Field(default_factory=dict)

    # Additive fields beyond the frozen core contract.
    #
    # ``unmapped_legacy_fonts`` makes "log a clear warning naming the unmapped
    # font" assertable: a log line cannot be tested at a library boundary, a
    # list can.
    unmapped_legacy_fonts: list[str] = Field(default_factory=list)
    attempts: list[AdapterAttempt] = Field(default_factory=list)
    pages_ocred: list[int] = Field(default_factory=list)
