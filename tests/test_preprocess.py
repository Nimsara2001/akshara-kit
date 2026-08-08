"""Preprocessing: Unicode hygiene and PDF layout repair.

Every expectation here is measured against the real corpus in ``output/``, not
invented — the counts in the module docstring of ``layout_noise.py`` come from
the same files these tests read.
"""

from __future__ import annotations

import unicodedata

from akshara_kit.preprocess import PreprocessConfig, clean
from akshara_kit.preprocess.layout_noise import dewrap, drop_noise_lines, is_noise_line
from akshara_kit.preprocess.unicode_rules import (
    HAL_KIRIMA,
    ZWJ,
    ZWNJ,
    normalise_unicode,
    strip_zero_width,
)

#: Words whose ZWJ forms a conjunct — යංසය and රකාරාංශය. Deleting it changes
#: the word, so these must survive byte-identical.
CONJUNCTS = ["ශ්‍ය", "ක්‍ර", "ප්‍රථම", "විද්‍යාල", "ශ්‍රී", "ද්‍ය"]


# --- zero-width policy ----------------------------------------------------


def test_meaningful_zwj_survives_byte_identical() -> None:
    """The whole reason this is not a blanket strip."""
    for word in CONJUNCTS:
        cleaned, removed, _ = strip_zero_width(word)
        assert cleaned == word, f"{word!r} lost its conjunct"
        assert removed == 0


def test_zwnj_is_always_removed() -> None:
    """ZWNJ has no orthographic role in Sinhala; here it is OCR emission."""
    cleaned, _, removed = strip_zero_width(f"තෝරාගන්{ZWNJ} පමණක්{ZWNJ}")
    assert ZWNJ not in cleaned
    assert removed == 2
    assert cleaned == "තෝරාගන් පමණක්"


def test_decorative_zwj_is_removed() -> None:
    """A ZWJ not forming a conjunct carries no meaning."""
    cleaned, removed, _ = strip_zero_width(f"ලෙස{ZWJ}ම")
    assert cleaned == "ලෙසම"
    assert removed == 1


def test_conjunct_survives_beside_spurious_zero_width() -> None:
    """The two policies must not interfere with each other in one string."""
    text = f"ශ්{ZWJ}ය සහ ලෙස{ZWJ}ම{ZWNJ}"
    cleaned, _, _ = strip_zero_width(text)
    assert cleaned == "ශ්‍ය සහ ලෙසම"


def test_zwj_policy_is_idempotent() -> None:
    once, _, _ = strip_zero_width("ශ්‍ය ලෙසම")
    twice, _, _ = strip_zero_width(once)
    assert once == twice


def test_hal_kirima_is_preserved() -> None:
    """Stripping the hal kirima would change every consonant it marks."""
    cleaned, _, _ = strip_zero_width("පමණක්")
    assert HAL_KIRIMA in cleaned


# --- unicode normalisation ------------------------------------------------


def test_nfc_normalisation_is_applied() -> None:
    text = "අධ්‍යාපන"
    assert normalise_unicode(text) == unicodedata.normalize("NFC", text)


def test_nfc_is_idempotent() -> None:
    once = normalise_unicode("ශ්‍රී ලංකා")
    assert normalise_unicode(once) == once


# --- layout noise ---------------------------------------------------------


def test_bare_page_numbers_are_noise() -> None:
    for line in ["94", "  12  ", "iv", "XVII"]:
        assert is_noise_line(line), line


def test_dot_leader_toc_lines_are_noise() -> None:
    assert is_noise_line("හැඳින්වීම ................................ 4")


def test_real_content_is_not_noise() -> None:
    for line in ["මම බත් කමි.", "රු. 100 (දේශීය)", ""]:
        assert not is_noise_line(line), line


def test_page_numbers_are_dropped() -> None:
    text = "මම බත් කමි.\n94\nඔහු ගියේ ය.\n95"
    cleaned, dropped = drop_noise_lines(text)
    assert dropped == 2
    assert "94" not in cleaned


def test_running_header_is_dropped_by_frequency() -> None:
    """Detected by repetition, not a hard-coded pattern."""
    text = "\n".join(["වියරණ විවරණ", "පළමු වගන්තිය", "වියරණ විවරණ", "දෙවන වගන්තිය", "වියරණ විවරණ"])
    cleaned, dropped = drop_noise_lines(text)
    assert "වියරණ විවරණ" not in cleaned
    assert dropped == 3


# --- de-wrapping ----------------------------------------------------------


def test_wrapped_sentence_is_rejoined() -> None:
    """The typesetter's line break is not a sentence break."""
    text = "අනුමත උපාධි පිරිනමන ආයතනයන්හි තෝරාගත් පූර්ණ කාලීන\nපාඨමාලා පමණක් හැදෑරීමට අවස්ථාව හිමි වේ."
    joined, joins = dewrap(text)
    assert joins == 1
    assert "\n" not in joined


def test_sentence_end_is_not_joined() -> None:
    """A line already ending in a finite verb starts a new line."""
    joined, joins = dewrap("මම බත් කමි.\nඔහු පාසල් ගියේ ය.")
    assert joins == 0
    assert joined.count("\n") == 1


def test_table_rows_are_never_joined() -> None:
    """A tab means cell structure; joining would merge two records."""
    text = "සීගිරිය\tමාතලේ\nඇල්ල\tබදුල්ල"
    joined, joins = dewrap(text)
    assert joins == 0
    assert joined.count("\n") == 1


# --- the pipeline ---------------------------------------------------------


def test_pipeline_reports_what_each_stage_changed() -> None:
    """Stage counts are what make preprocessing reportable rather than assumed."""
    result = clean(f"මම බත් කමි.{ZWNJ}\n94\nඔහු ගියේ\nය.")
    assert result.stages["zwnj_removed"] == 1
    assert result.stages["noise_lines_dropped"] == 1
    assert result.original_length > len(result.text)


def test_stages_can_be_switched_off_for_ablation() -> None:
    text = f"මම කමි.{ZWNJ}\n94"
    off = clean(text, config=PreprocessConfig(strip_zero_width=False, drop_noise_lines=False))
    assert ZWNJ in off.text
    assert "94" in off.text


def test_tabs_survive_whitespace_collapsing() -> None:
    """The segmenter depends on tabs to find table rows."""
    assert "\t" in clean("සීගිරිය\tමාතලේ").text


def test_empty_input_is_handled() -> None:
    assert clean("").text == ""
    assert clean("   \n\n  ").text == ""
