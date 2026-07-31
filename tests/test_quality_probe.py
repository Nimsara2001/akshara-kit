"""Quality probe (Algorithm 2) and the candidate ordering it feeds."""

from __future__ import annotations

from akshara_kit.contracts.extraction import QualityScore
from akshara_kit.eye.quality_probe import (
    MIN_VIABLE_CHARS,
    compare,
    is_viable,
    score,
    sinhala_ratio,
)
from samples import ASCII_CONTROL, LEGACY_SINHALA, UNICODE_SINHALA


def q(ratio: float, length: int) -> QualityScore:
    return QualityScore(raw_length=length, sinhala_ratio=ratio)


# --- the probe ------------------------------------------------------------


def test_empty_text_scores_zero_without_dividing_by_zero() -> None:
    result = score("")
    assert result.raw_length == 0
    assert result.sinhala_ratio == 0.0


def test_pure_sinhala_scores_near_one() -> None:
    assert sinhala_ratio("අධ්යාපන") == 1.0


def test_ascii_scores_zero() -> None:
    assert sinhala_ratio(ASCII_CONTROL) == 0.0


def test_legacy_glyph_stream_scores_zero() -> None:
    """The finding that forces normalise-before-score.

    Legacy FM text is Latin-1 bytes, so it contains no Sinhala code points at
    all. Every adapter ties at zero on a legacy document, which makes an argmax
    over raw text meaningless.
    """
    assert sinhala_ratio(LEGACY_SINHALA) == 0.0


def test_conversion_lifts_the_ratio_dramatically() -> None:
    assert sinhala_ratio(UNICODE_SINHALA) > 0.5


def test_score_populates_sample_tokens() -> None:
    result = score(UNICODE_SINHALA)
    assert result.sample_tokens, "Algorithm 2's third indicator should be present"


def test_sample_tokens_never_raise_on_odd_input() -> None:
    assert score("   ").sample_tokens == []


# --- viability floor ------------------------------------------------------


def test_a_single_character_is_not_viable() -> None:
    """Without a floor, one Sinhala character scores 1.0 and wins the race."""
    assert not is_viable(score("ක"))


def test_a_full_extraction_is_viable() -> None:
    assert is_viable(q(0.7, MIN_VIABLE_CHARS))


# --- ordering -------------------------------------------------------------


def test_higher_sinhala_ratio_wins() -> None:
    assert compare(q(0.80, 1000), q(0.10, 1000)) > 0
    assert compare(q(0.10, 1000), q(0.80, 1000)) < 0


def test_near_identical_ratios_are_treated_as_equal() -> None:
    """0.71 vs 0.715 is whitespace handling, not extraction quality."""
    assert compare(q(0.710, 1000), q(0.715, 1000)) == 0


def test_length_breaks_a_ratio_tie() -> None:
    assert compare(q(0.70, 4661), q(0.70, 4226)) > 0


def test_near_identical_lengths_are_treated_as_equal() -> None:
    assert compare(q(0.70, 1000), q(0.70, 1010)) == 0


def test_ordering_is_antisymmetric() -> None:
    left, right = q(0.80, 5000), q(0.20, 1000)
    assert compare(left, right) == -compare(right, left)


def test_identical_scores_compare_equal() -> None:
    assert compare(q(0.7, 1000), q(0.7, 1000)) == 0
