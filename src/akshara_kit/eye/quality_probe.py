"""Implements Algorithm 2 (Sinhala-Aware Quality Probe).

Stateless. The Sinhala-character ratio is the dominant signal, on the premise
that a well-formed Sinhala document consists overwhelmingly of Sinhala-range
characters *once legacy glyph streams have been normalised away*.

That caveat is load-bearing. Legacy FM-font text is stored as Latin-1 bytes and
scores a ratio of exactly 0.0, so every adapter ties at zero on a legacy
document. The PDF coordinator therefore normalises **before** scoring; see
``pdf_coordinator``. Report Algorithm 1 has this the other way round, which
makes its argmax degenerate for the legacy corpus.
"""

from __future__ import annotations

import re

from akshara_kit.contracts.extraction import QualityScore

__all__ = [
    "MAX_ORPHAN_VOWEL_RATE",
    "MIN_CONSONANTS_TO_JUDGE",
    "SINHALA_RANGE",
    "compare",
    "is_well_formed",
    "orphan_vowel_rate",
    "sample_tokens",
    "sinhala_ratio",
    "score",
]

#: The Sinhala Unicode block.
SINHALA_RANGE = re.compile(r"[඀-෿]")

# --- orthographic well-formedness -----------------------------------------
#
# A second signal is needed because ``sinhala_ratio`` cannot see the difference
# between correct Sinhala and garbled Sinhala. A PDF whose embedded ToUnicode
# cmap is wrong extracts as pure Sinhala code points — ratio 0.67 — while
# reading as nonsense: "පොලී" comes out "පපොලී", "යටතේ" comes out "යටපේ".
#
# The invariant used here is the strongest one Sinhala offers: a *dependent*
# vowel sign is by definition attached to a consonant, and Unicode always
# stores the consonant first. A vowel sign preceded by a space, a digit or
# punctuation therefore has nothing to depend on and cannot be legitimate.

_CONSONANTS = "ක-ෆ"
_VOWEL_SIGNS = "ා-ෟෲෳ"
_HAL_KIRIMA = "්"
_ZWJ = "‍"

#: A dependent vowel sign with no base to attach to.
_ORPHAN_VOWEL = re.compile(
    f"(?<![{_CONSONANTS}{_HAL_KIRIMA}{_ZWJ}{_VOWEL_SIGNS}])[{_VOWEL_SIGNS}]"
)

_CONSONANT = re.compile(f"[{_CONSONANTS}]")

#: Above this share of consonants, a text layer is judged malformed. Measured
#: across the fixture corpus the two populations are an order of magnitude
#: apart — correct extractions land at 0.0000–0.0023, garbled text layers at
#: 0.0297–0.0916 — so anything in between separates them.
MAX_ORPHAN_VOWEL_RATE = 0.01

#: Too little Sinhala to judge. Below this the rate is dominated by noise, and
#: a single stray sign in a caption would condemn the page.
MIN_CONSONANTS_TO_JUDGE = 20

#: Rates within this distance rank as equal, mirroring ``_RATIO_TOLERANCE``.
_ORPHAN_TOLERANCE = 0.005

#: Candidates shorter than this cannot win a race. Without a floor, a backend
#: that returns a single Sinhala character scores a perfect 1.0 and beats a
#: complete extraction.
MIN_VIABLE_CHARS = 32

#: Ratios within this distance are treated as equal — the difference between
#: 0.71 and 0.72 is whitespace handling, not quality.
_RATIO_TOLERANCE = 0.01

#: Lengths within this relative distance are treated as equal.
_LENGTH_TOLERANCE = 0.02

_TOKEN_PREFIX_CHARS = 2000
_TOKEN_LIMIT = 50


def sinhala_ratio(text: str) -> float:
    """Fraction of characters that fall in the Sinhala Unicode block."""
    return len(SINHALA_RANGE.findall(text)) / max(len(text), 1)


def sample_tokens(
    text: str, limit: int = _TOKEN_LIMIT, prefix_chars: int = _TOKEN_PREFIX_CHARS
) -> list[str]:
    """Tokenise a prefix of the text — Algorithm 2's third indicator.

    Uses ``sinlib`` when the ``sinhala`` extra is installed and falls back to a
    whitespace split otherwise. Purely diagnostic metadata, so this never
    raises: a tokeniser failure must not lose an otherwise good extraction.
    """
    prefix = text[:prefix_chars]
    if not prefix.strip():
        return []
    try:
        from sinlib import Tokenizer

        tokens = Tokenizer().tokenize(prefix)
    except Exception:  # noqa: BLE001 - decorative metadata, never fatal
        tokens = prefix.split()
    return [str(token) for token in tokens[:limit]]


def orphan_vowel_rate(text: str) -> float:
    """Share of consonants' worth of dependent vowel signs left unattached.

    Zero for correct Sinhala, and for text with no Sinhala in it at all —
    including an unconverted legacy glyph stream, which is Latin-1 bytes and
    carries no Sinhala vowel signs to orphan. That is deliberate: this measures
    *malformed Sinhala*, not *absent Sinhala*, which ``sinhala_ratio`` covers.
    """
    consonants = len(_CONSONANT.findall(text))
    if consonants < MIN_CONSONANTS_TO_JUDGE:
        return 0.0
    return len(_ORPHAN_VOWEL.findall(text)) / consonants


def score(text: str) -> QualityScore:
    """Implements Algorithm 2 (Sinhala-Aware Quality Probe)."""
    return QualityScore(
        raw_length=len(text),
        sinhala_ratio=sinhala_ratio(text),
        sample_tokens=sample_tokens(text),
        orphan_vowel_rate=orphan_vowel_rate(text),
    )


def is_well_formed(
    quality: QualityScore, max_rate: float = MAX_ORPHAN_VOWEL_RATE
) -> bool:
    """True if the text's Sinhala is orthographically plausible."""
    return quality.orphan_vowel_rate <= max_rate


def is_viable(quality: QualityScore, min_chars: int = MIN_VIABLE_CHARS) -> bool:
    """True if a candidate is long enough to be worth ranking."""
    return quality.raw_length >= min_chars


def compare(left: QualityScore, right: QualityScore) -> int:
    """Total order over candidates: ``-1``/``0``/``1`` for worse/equal/better.

    Section 7 asks only for "the best-scoring" output, which is underspecified.
    The order is: Sinhala ratio (bucketed), then orthographic well-formedness
    (bucketed, lower is better), then raw length (bucketed), then equal.
    Bucketing keeps the choice stable across runs and machines — a raw float
    argmax over 4226 versus 4661 characters is a coin flip wearing a decision's
    clothes. Callers break remaining ties by adapter cost.

    Well-formedness sits above length so that a backend which reads a broken
    ToUnicode cmap correctly beats one that returns more, wronger text. It sits
    below ratio because an empty extraction is trivially well-formed.
    """
    ratio_gap = left.sinhala_ratio - right.sinhala_ratio
    if abs(ratio_gap) > _RATIO_TOLERANCE:
        return 1 if ratio_gap > 0 else -1

    orphan_gap = left.orphan_vowel_rate - right.orphan_vowel_rate
    if abs(orphan_gap) > _ORPHAN_TOLERANCE:
        return -1 if orphan_gap > 0 else 1

    longest = max(left.raw_length, right.raw_length, 1)
    length_gap = (left.raw_length - right.raw_length) / longest
    if abs(length_gap) > _LENGTH_TOLERANCE:
        return 1 if length_gap > 0 else -1

    return 0
