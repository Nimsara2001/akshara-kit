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

__all__ = ["SINHALA_RANGE", "compare", "sample_tokens", "sinhala_ratio", "score"]

#: The Sinhala Unicode block.
SINHALA_RANGE = re.compile(r"[඀-෿]")

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


def score(text: str) -> QualityScore:
    """Implements Algorithm 2 (Sinhala-Aware Quality Probe)."""
    return QualityScore(
        raw_length=len(text),
        sinhala_ratio=sinhala_ratio(text),
        sample_tokens=sample_tokens(text),
    )


def is_viable(quality: QualityScore, min_chars: int = MIN_VIABLE_CHARS) -> bool:
    """True if a candidate is long enough to be worth ranking."""
    return quality.raw_length >= min_chars


def compare(left: QualityScore, right: QualityScore) -> int:
    """Total order over candidates: ``-1``/``0``/``1`` for worse/equal/better.

    Section 7 asks only for "the best-scoring" output, which is underspecified.
    The order is: Sinhala ratio (bucketed), then raw length (bucketed), then
    equal. Bucketing keeps the choice stable across runs and machines — a raw
    float argmax over 4226 versus 4661 characters is a coin flip wearing a
    decision's clothes. Callers break remaining ties by adapter cost.
    """
    ratio_gap = left.sinhala_ratio - right.sinhala_ratio
    if abs(ratio_gap) > _RATIO_TOLERANCE:
        return 1 if ratio_gap > 0 else -1

    longest = max(left.raw_length, right.raw_length, 1)
    length_gap = (left.raw_length - right.raw_length) / longest
    if abs(length_gap) > _LENGTH_TOLERANCE:
        return 1 if length_gap > 0 else -1

    return 0
