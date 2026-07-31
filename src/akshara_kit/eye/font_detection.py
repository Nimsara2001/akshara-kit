"""Legacy font detection (realises Section 6.1–6.3).

The central idea is that *detected as legacy* and *convertible to Unicode* are
different sets, and conflating them corrupts text. Four data structures encode
the distinction:

``LEGACY_FONT_MAPPINGS``
    Legacy fonts we can convert, and the conversion table to use. Every entry
    was verified by converting real span text from real documents and reading
    the output — see ``tests/test_font_detection.py::VERIFIED_SAMPLES``.
``KNOWN_UNMAPPABLE_LEGACY_FONTS``
    Legacy fonts we can recognise but *cannot* convert. Their text passes
    through unchanged with a warning. Converting them produces garbage.
``UNICODE_SINHALA_FONTS``
    Fonts whose text is already correct Unicode. An explicit never-convert
    guard: blanket conversion mangles these.
``FM_FAMILY_PATTERN``
    An opt-in family rule, disabled by default. See ``FM_FAMILY_RULE_ENABLED``.

On Section 6.4's "do not guess": the prohibition is against inventing a
*mapping* for a font whose encoding is unknown. Everything in
``LEGACY_FONT_MAPPINGS`` is evidence, not inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "FM_FAMILY_RULE_ENABLED",
    "KNOWN_LEGACY_FONT_NAMES",
    "KNOWN_UNMAPPABLE_LEGACY_FONTS",
    "LEGACY_FONT_MAPPINGS",
    "UNICODE_SINHALA_FONTS",
    "FontClass",
    "SpanFont",
    "classify_font",
    "detect_pdf_fonts",
    "is_legacy",
    "is_mappable",
    "normalise_font_name",
    "separator",
]


class FontClass(str, Enum):
    """What may safely be done with text drawn in a given font."""

    LEGACY_MAPPABLE = "legacy_mappable"
    LEGACY_UNMAPPABLE = "legacy_unmappable"
    UNICODE_SINHALA = "unicode_sinhala"
    NON_SINHALA = "non_sinhala"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SpanFont:
    """A run of text together with the font it was drawn in.

    The unit of conversion. PDF spans, DOCX runs and XLSX cells all reduce to
    this, which is why one normalisation routine serves all three formats.

    ``synthetic`` marks layout whitespace this library inserted itself, as
    distinct from a real run whose font simply could not be resolved (``font``
    empty). Both pass through unconverted, but only the latter is document
    content, and conflating them corrupts run counts and font reports.
    """

    text: str
    font: str
    location: str = ""
    synthetic: bool = False

    @property
    def is_separator(self) -> bool:
        """True only for whitespace this library inserted."""
        return self.synthetic


def separator(text: str) -> SpanFont:
    """A synthetic span carrying layout whitespace, never converted."""
    return SpanFont(text=text, font="", synthetic=True)


# --- Name normalisation ---------------------------------------------------

#: PDF subset prefixes, e.g. ``CZMRSJ+FMAbhayax``. PyMuPDF's span dictionaries
#: already strip these, but pdfplumber and pdfminer do not.
_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")

#: Trailing style tokens, stripped only for the *family* fallback lookup —
#: never from the reported name. Stripping eagerly would turn the real font
#: ``FMAbabldBold`` into a non-existent ``FMAbabld``.
_STYLE_SUFFIX = re.compile(
    r"(?:[-,\s]?(?:Regular|Bold|Italic|Oblique|Light|BoldItalic))+$", re.IGNORECASE
)


def normalise_font_name(raw: str) -> str:
    """Canonicalise a font name for lookup.

    Strips any PDF subset prefix and internal spaces, so ``"FM Abhaya"``,
    ``"FMAbhaya"`` and ``"CZMRSJ+FMAbhaya"`` all resolve to the same key.
    """
    return _SUBSET_PREFIX.sub("", raw.strip()).replace(" ", "")


def _family_name(normalised: str) -> str:
    """Drop a trailing style token, e.g. ``IskoolaPotaRegular`` -> ``IskoolaPota``."""
    return _STYLE_SUFFIX.sub("", normalised) or normalised


# --- Tier 1: legacy fonts we can convert ----------------------------------

#: Verified against real documents: each font's span text was converted through
#: pandukabhaya's ``fm_abhaya`` table and the Sinhala output confirmed correct.
#: They are all cuts of the same legacy 8-bit Sinhala codepage.
LEGACY_FONT_MAPPINGS: dict[str, str] = {
    "FMAbhaya": "fm_abhaya",
    "FMAbhayax": "fm_abhaya",
    "FMAbhaya-Bold": "fm_abhaya",
    "FMAbhaya-Regular": "fm_abhaya",
    "FMAbabldBold": "fm_abhaya",
    "FMBindumathix": "fm_abhaya",
    "FMEdwerdBanceBold": "fm_abhaya",
    "FMEmaneex": "fm_abhaya",
    "FMMalithix": "fm_abhaya",
    "FMPrabhathbox": "fm_abhaya",
    "FMSamantha": "fm_abhaya",
    "FMSamanthax": "fm_abhaya",
    "FMSamanthaBoldx": "fm_abhaya",
}

#: Section 6.1 compatibility alias. Kept so the spec's name still resolves.
KNOWN_LEGACY_FONT_NAMES: frozenset[str] = frozenset(LEGACY_FONT_MAPPINGS)

# --- Tier 2: opt-in family rule -------------------------------------------

#: Every ``FM*`` font encountered so far converts correctly through
#: ``fm_abhaya``. Disabled by default: the shipped behaviour is the verified
#: allow-list above. Enable to cover an unseen ``FM*`` font without a release.
FM_FAMILY_RULE_ENABLED = False
FM_FAMILY_PATTERN = re.compile(r"^FM[A-Za-z]", re.ASCII)
FM_FAMILY_MAPPING = "fm_abhaya"

# --- Tier 3: legacy fonts we must NOT convert -----------------------------

#: Recognised as legacy, but no conversion table covers them. Their text is
#: passed through unchanged and reported in ``unmapped_legacy_fonts``.
KNOWN_UNMAPPABLE_LEGACY_FONTS: frozenset[str] = frozenset(
    {
        # Sinhala legacy fonts on a different codepage.
        "Chamodi",
        "sandaru-n",
        # K-Plain converts *almost* correctly through fm_abhaya
        # ("ffjµ ks¾foaY rys;j" -> "වෛi නිර්දේශ රහිතව", leaking a stray "i").
        # A plausible-looking partial is more dangerous than obvious garbage,
        # so it stays here. Do not promote it without new evidence.
        "K-Plain",
        # Tamil legacy fonts. fm_abhaya renders these as noise.
        "Tharmini-Plain",
        "TAM-Tamil155",
    }
)

UNMAPPABLE_LEGACY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^SHREE-TAM", re.IGNORECASE),
    re.compile(r"^TAM-Tamil", re.IGNORECASE),
)

# --- Guard list: already-Unicode fonts ------------------------------------

#: Never convert. Verified hazard: blanket conversion turns the Iskoola Pota
#: text "හැඳින්වීම ......" into "හැඳින්වීම ගගගගගග".
UNICODE_SINHALA_FONTS: frozenset[str] = frozenset(
    {
        "IskoolaPota",
        "NirmalaUI",
        "Latha",
        "NotoSansSinhala",
        "NotoSerifSinhala",
        "AbhayaLibre",
        "YaldeviColombo",
        "UNAbhaya",
    }
)

#: Latin and symbol font families. Behaviourally identical to UNKNOWN; named
#: separately so reports can distinguish "Latin text" from "unrecognised".
_NON_SINHALA_PREFIXES: tuple[str, ...] = (
    "Times",
    "Arial",
    "Helvetica",
    "Calibri",
    "Courier",
    "MinionPro",
    "BookAntiqua",
    "TrebuchetMS",
    "Cambria",
    "Aptos",
    "Verdana",
    "Georgia",
    "Symbol",
    "Webdings",
    "Wingdings",
    "AdobeHebrew",
    "MSMincho",
)


def classify_font(font_name: str) -> FontClass:
    """Classify a font name. Order is the safety argument.

    The Unicode guard and the unmappable deny-list are both consulted *before*
    the mappable allow-list, so a font can always be excluded by adding one
    entry, and never by accident.
    """
    normalised = normalise_font_name(font_name)
    if not normalised:
        return FontClass.UNKNOWN
    family = _family_name(normalised)

    if normalised in UNICODE_SINHALA_FONTS or family in UNICODE_SINHALA_FONTS:
        return FontClass.UNICODE_SINHALA

    if normalised in KNOWN_UNMAPPABLE_LEGACY_FONTS or family in KNOWN_UNMAPPABLE_LEGACY_FONTS:
        return FontClass.LEGACY_UNMAPPABLE
    if any(pattern.match(normalised) for pattern in UNMAPPABLE_LEGACY_PATTERNS):
        return FontClass.LEGACY_UNMAPPABLE

    if normalised in LEGACY_FONT_MAPPINGS or family in LEGACY_FONT_MAPPINGS:
        return FontClass.LEGACY_MAPPABLE
    if FM_FAMILY_RULE_ENABLED and FM_FAMILY_PATTERN.match(normalised):
        return FontClass.LEGACY_MAPPABLE

    if family.startswith(_NON_SINHALA_PREFIXES):
        return FontClass.NON_SINHALA
    return FontClass.UNKNOWN


def is_legacy(font_name: str) -> bool:
    """True if the font is a legacy Sinhala or Tamil font, convertible or not."""
    return classify_font(font_name) in {
        FontClass.LEGACY_MAPPABLE,
        FontClass.LEGACY_UNMAPPABLE,
    }


def is_mappable(font_name: str) -> tuple[bool, str | None]:
    """Return ``(mappable, mapping_name)`` for a font."""
    if classify_font(font_name) is not FontClass.LEGACY_MAPPABLE:
        return False, None
    normalised = normalise_font_name(font_name)
    mapping = LEGACY_FONT_MAPPINGS.get(normalised) or LEGACY_FONT_MAPPINGS.get(
        _family_name(normalised)
    )
    return True, mapping or FM_FAMILY_MAPPING


def detect_pdf_fonts(file_path: str) -> set[str]:
    """Every font name used by any text span in a PDF (Section 6.2).

    PyMuPDF is used specifically here because it is the only one of the four
    PDF backends that exposes an embedded font's name per text span, even when
    another adapter wins the extraction race.
    """
    from akshara_kit.adapters.extractors import pymupdf_adapter

    return {
        span.font
        for span in pymupdf_adapter.iter_spans(file_path)
        if not span.is_separator
    }
