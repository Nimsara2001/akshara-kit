"""Legacy-to-Unicode normalisation (realises Section 6.4).

Known limitation
----------------
As of this implementation, ``pandukabhaya`` supports conversion for the FM
Abhaya legacy font only, though it is built with an extensible JSON-mapping
design that is expected to support additional fonts (e.g. DL Manel, FM
Bindumathi) in future releases. ``KNOWN_LEGACY_FONT_NAMES`` and the conversion
routing logic here are intentionally structured so that adding a new font
mapping requires only a new entry in that set plus a corresponding pandukabhaya
mapping table — no change to the detection or routing control flow.

Two modes
---------
:func:`normalise_spans` is the high-fidelity path: it converts each run of text
in isolation, gated on that run's own font, so Latin and already-Unicode text
sitting beside legacy text survives untouched. Use it wherever a font signal
exists — PDF via PyMuPDF spans, DOCX runs, XLSX cells.

:func:`normalise` is the document-level fallback for text with no font
association at all. It is lossy by construction: it converts everything or
nothing. The prior prototype used only this mode, unconditionally, which is how
``www.edupub.gov.lk`` became ``අඅඅගැාමචමඉගටදඩගකන``.
"""

from __future__ import annotations

import functools
import importlib.resources
import logging
import re
from typing import TYPE_CHECKING, Iterable

from pydantic import BaseModel, Field

from akshara_kit.contracts.extraction import FontDetectionMethod
from akshara_kit.eye.font_detection import (
    FontClass,
    classify_font,
    is_mappable,
    normalise_font_name,
)

if TYPE_CHECKING:
    from akshara_kit.eye.font_detection import SpanFont

__all__ = [
    "DEFAULT_MAPPING",
    "NormalisationOutcome",
    "available_mappings",
    "convert",
    "normalise",
    "normalise_legacy",
    "normalise_spans",
]

logger = logging.getLogger(__name__)

DEFAULT_MAPPING = "fm_abhaya"

#: Section 6.4's threshold: below this Sinhala ratio, a document expected to be
#: Sinhala is suspected of carrying legacy glyphs.
DEFAULT_RATIO_THRESHOLD = 0.15

#: Extra gate on the ratio heuristic. Legacy Sinhala streams are dense in
#: high-bit Latin-1 characters (¾ ï ¨ ½ and friends); ordinary English prose is
#: not. Without this, the heuristic fires on any English document and destroys
#: it — "From the government" becomes "ත්‍රදප එයැ ටදඩැරබපැබඑ".
DEFAULT_HIGH_BIT_FLOOR = 0.02

_HIGH_BIT_LATIN = re.compile(r"[ -ÿ]")
_SINHALA = re.compile(r"[඀-෿]")


class NormalisationOutcome(BaseModel):
    """Result of a normalisation pass.

    Returned instead of Section 6.4's literal 3-tuple because that section also
    requires unmapped fonts to be surfaced by name, and a 4-tuple is unreadable.
    :func:`normalise_legacy` provides the literal 3-tuple shim.
    """

    text: str
    method: FontDetectionMethod = FontDetectionMethod.NONE
    converted_fonts: list[str] = Field(default_factory=list)
    unmapped_fonts: list[str] = Field(default_factory=list)


@functools.lru_cache(maxsize=8)
def _converter(mapping: str = DEFAULT_MAPPING):
    """Build and cache a pandukabhaya converter.

    Cached because construction compiles a regex of ~1150 alternatives; the
    prior prototype rebuilt one per document.
    """
    try:
        from pandukabhaya import Converter
    except ImportError as exc:  # pragma: no cover - pandukabhaya is a core dep
        raise RuntimeError(
            "pandukabhaya is required for legacy Sinhala conversion"
        ) from exc

    try:
        return Converter(mapping)
    except FileNotFoundError as exc:
        raise ValueError(
            f"pandukabhaya has no mapping named {mapping!r}; "
            f"available: {', '.join(sorted(available_mappings())) or 'none'}"
        ) from exc


@functools.lru_cache(maxsize=1)
def available_mappings() -> frozenset[str]:
    """Conversion tables pandukabhaya ships.

    Discovered by globbing the package's ``mappings`` directory: pandukabhaya
    exposes no enumeration API and raises only at construction time.
    """
    try:
        directory = importlib.resources.files("pandukabhaya") / "mappings"
        return frozenset(
            path.name.removesuffix(".json")
            for path in directory.iterdir()
            if path.name.endswith(".json")
        )
    except Exception:  # noqa: BLE001 - discovery must never break extraction
        return frozenset({DEFAULT_MAPPING})


def convert(text: str, mapping: str = DEFAULT_MAPPING) -> str:
    """Convert legacy glyph bytes to Sinhala Unicode. No detection, no gating.

    Applying this to text that is not in the given legacy font corrupts it.
    Callers should almost always use :func:`normalise_spans` instead.
    """
    if not text:
        return text
    return _converter(mapping).convert(text)


def normalise_spans(spans: Iterable[SpanFont]) -> NormalisationOutcome:
    """Convert per-span, gated on each span's own font (Section 6.4).

    The high-fidelity path. A Helvetica URL beside an FM-Abhaya heading keeps
    its bytes; an Iskoola Pota run beside the same heading keeps its Unicode.
    """
    pieces: list[str] = []
    converted: dict[str, None] = {}
    unmapped: dict[str, None] = {}

    for span in spans:
        if span.is_separator:
            pieces.append(span.text)
            continue

        name = normalise_font_name(span.font)
        font_class = classify_font(span.font)

        if font_class is FontClass.LEGACY_MAPPABLE:
            _, mapping = is_mappable(span.font)
            pieces.append(convert(span.text, mapping or DEFAULT_MAPPING))
            converted[name] = None
        else:
            if font_class is FontClass.LEGACY_UNMAPPABLE:
                unmapped[name] = None
            pieces.append(span.text)

    _warn_unmapped(unmapped)
    return NormalisationOutcome(
        text="".join(pieces),
        method=(
            FontDetectionMethod.FONT_NAME if converted else FontDetectionMethod.NONE
        ),
        converted_fonts=sorted(converted),
        unmapped_fonts=sorted(unmapped),
    )


def normalise(
    text: str,
    detected_fonts: set[str],
    *,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    allow_ratio_fallback: bool = True,
) -> NormalisationOutcome:
    """Convert whole text, gated on the document's font set (Section 6.4).

    The fallback for backends that expose no per-span font information. Lossy:
    conversion is all-or-nothing across the whole string. Prefer
    :func:`normalise_spans` whenever a font signal is available.
    """
    classified = _classify_all(detected_fonts)
    mappable = classified[FontClass.LEGACY_MAPPABLE]
    unmapped = classified[FontClass.LEGACY_UNMAPPABLE]
    _warn_unmapped(dict.fromkeys(unmapped))

    if mappable:
        _, mapping = is_mappable(mappable[0])
        return NormalisationOutcome(
            text=convert(text, mapping or DEFAULT_MAPPING),
            method=FontDetectionMethod.FONT_NAME,
            converted_fonts=mappable,
            unmapped_fonts=unmapped,
        )

    if allow_ratio_fallback and _ratio_heuristic_fires(text, classified, ratio_threshold):
        logger.info(
            "No legacy font name detected, but the text looks like a legacy "
            "glyph stream; attempting conversion via the character-ratio heuristic"
        )
        return NormalisationOutcome(
            text=convert(text, DEFAULT_MAPPING),
            method=FontDetectionMethod.CHARACTER_RATIO,
            unmapped_fonts=unmapped,
        )

    return NormalisationOutcome(text=text, unmapped_fonts=unmapped)


def _classify_all(fonts: Iterable[str]) -> dict[FontClass, list[str]]:
    """Group font names by classification, preserving a stable order."""
    grouped: dict[FontClass, list[str]] = {member: [] for member in FontClass}
    for font in sorted({normalise_font_name(f) for f in fonts if f}):
        grouped[classify_font(font)].append(font)
    return grouped


def _ratio_heuristic_fires(
    text: str, classified: dict[FontClass, list[str]], threshold: float
) -> bool:
    """Decide whether the character-ratio fallback should convert.

    Three conditions, not one. Section 6.4 states only the first; on its own it
    fires on every English document and destroys it.
    """
    if not text.strip():
        return False
    # 1. The text does not already look like Sinhala Unicode.
    if len(_SINHALA.findall(text)) / max(len(text), 1) >= threshold:
        return False
    # 2. No font told us this text is already fine.
    if classified[FontClass.UNICODE_SINHALA] or classified[FontClass.NON_SINHALA]:
        return False
    # 3. No font told us conversion is known *not* to work here. Guessing with
    #    fm_abhaya on a font we have positively identified as unmappable is
    #    strictly worse than leaving the text alone.
    if classified[FontClass.LEGACY_UNMAPPABLE]:
        return False
    # 4. It carries the high-bit Latin-1 density typical of legacy glyph streams.
    high_bit = len(_HIGH_BIT_LATIN.findall(text)) / max(len(text), 1)
    return high_bit >= DEFAULT_HIGH_BIT_FLOOR


def _warn_unmapped(unmapped: dict[str, None]) -> None:
    """Log each unmapped legacy font by name (Section 6.4)."""
    for font in unmapped:
        logger.warning(
            "Legacy font %r has no conversion mapping; its text was passed "
            "through unchanged rather than corrupted",
            font,
        )


def normalise_legacy(
    text: str, detected_fonts: set[str]
) -> tuple[str, FontDetectionMethod, list[str]]:
    """Section 6.4's literal 3-tuple signature, for spec conformance.

    Prefer :func:`normalise`, which also reports unmapped fonts.
    """
    outcome = normalise(text, detected_fonts)
    return outcome.text, outcome.method, outcome.converted_fonts
