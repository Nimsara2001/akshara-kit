"""Unicode hygiene for extracted Sinhala text.

The delicate part is the zero-width characters. Both appear in extracted text in
quantity — measured on ``output/sample_unicode.pdf.txt``: 5,754 ZWNJ and 5,039
ZWJ — and they must be treated in opposite ways.

**ZWJ (U+200D) is often meaningful and must not be stripped wholesale.** Sinhala
forms two productive conjuncts with it:

- යංසය: ``්`` + ZWJ + ``ය`` — as in ශ්‍ය, ද්‍ය, ක්‍ය
- රකාරාංශය: ``්`` + ZWJ + ``ර`` — as in ක්‍ර, ප්‍ර, ශ්‍ර

Delete the ZWJ from ``ශ්‍ය`` and the word changes: the ligature becomes two
separate letters. So ZWJ is kept exactly where it forms one of those two
sequences, and stripped everywhere else, where it is decorative or an artefact.

**ZWNJ (U+200C) is always removed.** It has no orthographic role in Sinhala —
its job is to *prevent* ligature formation, which Sinhala text does not need —
and in this corpus it is OCR emission, appearing after a hal-kirima at word ends
(``තෝරාගන්`` + ZWNJ). Left in place it splits tokens and perturbs embeddings for
words that are otherwise identical.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "HAL_KIRIMA",
    "ZWJ",
    "ZWNJ",
    "normalise_unicode",
    "strip_zero_width",
]

#: U+200D ZERO WIDTH JOINER — meaningful in යංසය and රකාරාංශය only.
ZWJ = "‍"
#: U+200C ZERO WIDTH NON-JOINER — no role in Sinhala orthography.
ZWNJ = "‌"
#: U+0DCA SINHALA SIGN AL-LAKUNA, the hal kirima.
HAL_KIRIMA = "්"

#: The two conjuncts whose ZWJ is load-bearing. ``ය`` = U+0DBA, ``ර`` = U+0DBB.
_MEANINGFUL_ZWJ = re.compile(f"{HAL_KIRIMA}{ZWJ}([යර])")

#: Placeholder used to hide meaningful ZWJ while the rest are removed. Chosen
#: from a Private Use Area so it cannot occur in real text.
_SENTINEL = ""
_SENTINEL_PATTERN = re.compile(f"{_SENTINEL}([යර])")

#: Control and format characters that carry no textual meaning here. Form feed
#: (U+000C) marks a page break in some extractor output; tab is preserved
#: because the DOCX and XLSX adapters use it as a cell separator.
_JUNK_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f­﻿]")


def normalise_unicode(text: str) -> str:
    """Apply NFC composition.

    Sinhala graphemes can be encoded more than one way; extracted text is not
    NFC by default (``NFC(t) != t`` on every fixture). Two byte sequences for
    the same word would otherwise tokenise and embed differently.
    """
    return unicodedata.normalize("NFC", text)


def strip_zero_width(text: str) -> tuple[str, int, int]:
    """Remove decorative zero-width characters, keeping the meaningful ones.

    Returns ``(text, zwj_removed, zwnj_removed)`` so the pipeline can report
    what it changed rather than assert it.
    """
    zwnj_removed = text.count(ZWNJ)
    zwj_before = text.count(ZWJ)

    # Hide the ZWJ that forms a conjunct, drop every remaining zero-width
    # character, then restore. Doing it in this order means the "remove"
    # step needs no lookbehind and cannot accidentally match across a
    # boundary it should have respected.
    protected = _MEANINGFUL_ZWJ.sub(f"{HAL_KIRIMA}{_SENTINEL}\\1", text)
    stripped = protected.replace(ZWJ, "").replace(ZWNJ, "")
    restored = _SENTINEL_PATTERN.sub(f"{ZWJ}\\1", stripped)

    return restored, zwj_before - restored.count(ZWJ), zwnj_removed


def strip_control_characters(text: str) -> tuple[str, int]:
    """Drop control and format characters, preserving tab and newline."""
    cleaned = _JUNK_CONTROLS.sub("", text)
    return cleaned, len(text) - len(cleaned)
