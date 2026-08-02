"""Quality probe (Algorithm 2) and the candidate ordering it feeds."""

from __future__ import annotations

from akshara_kit.contracts.extraction import QualityScore
from akshara_kit.eye.quality_probe import (
    MIN_VIABLE_CHARS,
    compare,
    is_viable,
    is_well_formed,
    orphan_vowel_rate,
    score,
    sinhala_ratio,
)
from samples import (
    ASCII_CONTROL,
    GARBLED_CMAP_SINHALA,
    LEGACY_SINHALA,
    UNICODE_SINHALA,
    WELL_FORMED_PARAGRAPH,
)


def q(ratio: float, length: int, orphan: float = 0.0) -> QualityScore:
    return QualityScore(
        raw_length=length, sinhala_ratio=ratio, orphan_vowel_rate=orphan
    )


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


def test_include_sample_tokens_can_be_switched_off() -> None:
    """The Brain's per-chunk opt-out — everything else still populates."""
    result = score(UNICODE_SINHALA, include_sample_tokens=False)
    assert result.sample_tokens == []
    assert result.raw_length == len(UNICODE_SINHALA)
    assert result.sinhala_ratio > 0.5


def test_sample_tokens_default_to_included() -> None:
    """The Eye's existing behaviour must not change under the new default."""
    assert score(UNICODE_SINHALA).sample_tokens == score(
        UNICODE_SINHALA, include_sample_tokens=True
    ).sample_tokens


# --- orthographic well-formedness -----------------------------------------
#
# The signal sinhala_ratio cannot provide. A PDF with a broken ToUnicode cmap
# extracts as pure Sinhala code points at a healthy ratio, but the wrong ones.


def test_correct_sinhala_has_no_orphan_vowels() -> None:
    assert orphan_vowel_rate(WELL_FORMED_PARAGRAPH) == 0.0


def test_garbled_cmap_text_has_orphan_vowels() -> None:
    """Real text-layer output from a PDF whose cmap is wrong."""
    assert orphan_vowel_rate(GARBLED_CMAP_SINHALA) > 0.01


def test_garbled_text_still_scores_a_high_sinhala_ratio() -> None:
    """Why the second signal is needed at all: the first cannot see this."""
    assert sinhala_ratio(GARBLED_CMAP_SINHALA) > 0.5


def test_unconverted_legacy_stream_is_not_judged_malformed() -> None:
    """Legacy bytes carry no Sinhala, so there is nothing to orphan.

    This measures malformed Sinhala, not absent Sinhala — otherwise every
    pre-conversion legacy page would be sent to OCR needlessly.
    """
    assert orphan_vowel_rate(LEGACY_SINHALA) == 0.0


def test_ascii_is_not_judged_malformed() -> None:
    assert orphan_vowel_rate(ASCII_CONTROL) == 0.0


def test_short_text_is_not_judged() -> None:
    """Below the consonant floor one stray sign would condemn a caption."""
    assert orphan_vowel_rate("ො ක") == 0.0


def test_is_well_formed_separates_the_two_populations() -> None:
    assert is_well_formed(score(WELL_FORMED_PARAGRAPH))
    assert not is_well_formed(score(GARBLED_CMAP_SINHALA))


def test_score_reports_the_rate() -> None:
    assert score(GARBLED_CMAP_SINHALA).orphan_vowel_rate > 0.01


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


def test_well_formed_beats_garbled_at_equal_ratio() -> None:
    """The ordering that lets a correct backend beat a corrupt one."""
    assert compare(q(0.70, 1000, orphan=0.000), q(0.70, 1000, orphan=0.03)) > 0
    assert compare(q(0.70, 1000, orphan=0.03), q(0.70, 1000, orphan=0.000)) < 0


def test_well_formedness_outranks_length() -> None:
    """More, wronger text must not beat less, correcter text."""
    assert compare(q(0.70, 1000, orphan=0.0), q(0.70, 9000, orphan=0.05)) > 0


def test_near_identical_orphan_rates_are_treated_as_equal() -> None:
    assert compare(q(0.70, 1000, orphan=0.001), q(0.70, 1000, orphan=0.003)) == 0


def test_length_breaks_a_ratio_tie() -> None:
    assert compare(q(0.70, 4661), q(0.70, 4226)) > 0


def test_near_identical_lengths_are_treated_as_equal() -> None:
    assert compare(q(0.70, 1000), q(0.70, 1010)) == 0


def test_ordering_is_antisymmetric() -> None:
    left, right = q(0.80, 5000), q(0.20, 1000)
    assert compare(left, right) == -compare(right, left)


def test_identical_scores_compare_equal() -> None:
    assert compare(q(0.7, 1000), q(0.7, 1000)) == 0
