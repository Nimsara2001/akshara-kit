"""The Sinhala rule base and its Prolog adapter.

The expectation table below is taken from *වියරණ විවරණ* §4 (උක්ත ආඛ්‍යාත
සම්බන්ධතා), and each row cites the page it comes from. That provenance is the
point: a linguist can check the table against the grammar without reading any
Python, which is the whole argument for a symbolic layer.

Tests marked ``prolog`` need SWI-Prolog. The parity tests do not — they exercise
``rule_tables``, the Python mirror that preprocessing uses for de-wrapping, and
assert it agrees with Prolog wherever both can run.
"""

from __future__ import annotations

import pytest

from akshara_kit.brain import rule_tables
from akshara_kit.contracts.chunking import BoundaryKind

#: (word, sentence-level, clause-level, grammar page)
GRAMMAR_CASES: list[tuple[str, bool, bool, str]] = [
    # --- Tier 1: finite predicate (ආඛ්‍යාතය) ends the sentence, p.94-96 ---
    ("කරමි", True, True, "p.94 උත්තම පුරුෂ ඒක වචන"),
    ("කළෙමි", True, True, "p.94"),
    ("යමු", True, True, "p.94 උත්තම පුරුෂ බහු වචන"),
    ("ගියෙමු", True, True, "p.94"),
    ("කරයි", True, True, "p.94 ප්‍රථම පුරුෂ ඒක වචන"),
    ("වැටෙයි", True, True, "p.110"),
    ("යති", True, True, "p.95 ප්‍රථම පුරුෂ බහු වචන"),
    ("කරති", True, True, "p.95"),
    ("කළහ", True, True, "p.99 අතීත බහු වචන"),
    ("වූහ", True, True, "p.99"),
    ("සේක", True, True, "p.99 ගෞරවාර්ථ"),
    # කෘදන්ත ආඛ්‍යාතය: the particle ය stands as its own token, p.95
    ("ය", True, True, "p.95 කෘදන්ත ආඛ්‍යාතය"),
    # Obligation particles close a sentence, p.107
    ("වටී", True, True, "p.107"),
    ("මැනවි", True, True, "p.107"),
    # Orthographic terminators
    ("ගියේ.", True, True, "punctuation"),
    ("කොයිද?", True, True, "p.109 ප්‍රශ්නවාචී"),
    # --- Tier 2: non-finite forms end a CLAUSE, not the sentence ---
    ("කරලා", False, True, "absolutive ලා"),
    ("බලමින්", False, True, "continuous මින්"),
    ("ගොස්", False, True, "absolutive of motion"),
    ("පැමිණියොත්", False, True, "p.107 conditional"),
    ("උගන්වද්දී", False, True, "p.107 temporal"),
    ("යතත්", False, True, "p.107 concessive"),
    ("නම්", False, True, "conditional particle"),
    # --- Tier 3: discourse connectives start a new thought ---
    ("නමුත්", False, True, "adversative"),
    ("එබැවින්", False, True, "causal"),
    # --- Tier 4: joining particles must NEVER split ---
    ("සහ", False, False, "p.112 සහාර්ථය"),
    ("සමඟ", False, False, "p.112 සහාර්ථය"),
    ("හා", False, False, "p.112 සහාර්ථය"),
    ("කැටුව", False, False, "p.112 සහාර්ථය"),
    ("ද", False, False, "p.110 සමුච්චයාර්ථය"),
    ("හෝ", False, False, "p.112 විකල්පාර්ථය"),
    ("නොහොත්", False, False, "p.112 විකල්පාර්ථය"),
    # --- The quotative යි closes an embedded clause, p.103-104 ---
    ("එතියි", False, False, "p.103 අන්තර් වාක්‍යය"),
    ("දෙසතියි", False, False, "p.104"),
    ("කීහයි", False, False, "p.104"),
    ("ගනිමියි", False, False, "p.104 ප්‍රකෘති කථනය"),
    # --- Words that merely ARE an ending must not be read as verbs ---
    ("මු", False, False, "pronoun-length guard"),
    ("හ", False, False, "length guard"),
    ("ති", False, False, "length guard"),
]


# --- the Python mirror ----------------------------------------------------


@pytest.mark.parametrize(("word", "sentence", "clause", "cite"), GRAMMAR_CASES)
def test_rule_tables_match_the_grammar(
    word: str, sentence: bool, clause: bool, cite: str
) -> None:
    assert rule_tables.is_sentence_end(word) is sentence, f"{word} — {cite}"
    assert rule_tables.is_clause_end(word) is clause, f"{word} — {cite}"


def test_clause_level_is_a_superset_of_sentence_level() -> None:
    """A sentence boundary is a fortiori a clause boundary."""
    for word, sentence, _, _ in GRAMMAR_CASES:
        if sentence:
            assert rule_tables.is_clause_end(word), word


def test_never_split_overrides_every_other_rule() -> None:
    """ද ends in no verb ending, but සමඟ would otherwise look splittable."""
    for word in rule_tables.NEVER_SPLIT_WORDS:
        assert not rule_tables.is_sentence_end(word)
        assert not rule_tables.is_clause_end(word)


def test_quotative_is_distinguished_from_the_finite_third_person() -> None:
    """The one genuinely hard rule, p.103-104.

    ``කරයි`` is a verb and ends a sentence; ``එතියි`` is that same verb plus the
    quotative and must stay attached to its main clause.
    """
    assert rule_tables.is_quotative("එතියි")
    assert rule_tables.is_quotative("එති’යි")
    assert not rule_tables.is_quotative("කරයි")
    assert rule_tables.is_sentence_end("කරයි")
    assert not rule_tables.is_sentence_end("එතියි")


# --- the Prolog engine ----------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from akshara_kit.brain.rule_engine import SymbolicRuleEngine

    with SymbolicRuleEngine() as e:
        yield e


@pytest.mark.prolog
@pytest.mark.parametrize(("word", "sentence", "clause", "cite"), GRAMMAR_CASES)
def test_prolog_matches_the_grammar(
    engine, word: str, sentence: bool, clause: bool, cite: str
) -> None:
    assert engine.is_split_point(word, BoundaryKind.SENTENCE) is sentence, f"{word} — {cite}"
    assert engine.is_split_point(word, BoundaryKind.CLAUSE) is clause, f"{word} — {cite}"


@pytest.mark.prolog
def test_prolog_and_python_tables_agree(engine) -> None:
    """The parity check that keeps the two implementations honest.

    ``rule_tables`` is used by preprocessing, which must run without Prolog. If
    the two ever diverge, de-wrapping and chunking would disagree about where a
    sentence ends — this test makes that a failure rather than a silent quirk.
    """
    for word, _, _, _ in GRAMMAR_CASES:
        assert engine.is_split_point(word, BoundaryKind.SENTENCE) == (
            rule_tables.is_sentence_end(word)
        ), word
        assert engine.is_split_point(word, BoundaryKind.CLAUSE) == (
            rule_tables.is_clause_end(word)
        ), word


@pytest.mark.prolog
def test_utf8_atoms_actually_load(engine) -> None:
    """Guards the encoding directive in the .pl file.

    Without ``:- encoding(utf8).`` the rule base still consults cleanly but no
    Sinhala atom ever matches, silently degrading to punctuation-only splitting.
    A word with no punctuation that must still split is the canary.
    """
    assert engine.is_split_point("කරමි") is True


@pytest.mark.prolog
def test_micro_chunks_implements_algorithm_3(engine) -> None:
    """Accumulate, flush on a boundary, flush the residue at the end."""
    chunks = engine.micro_chunks("මම බත් කමි. ඔහු පාසල් ගියේ ය. ඉතිරි වචන")
    assert chunks == ["මම බත් කමි.", "ඔහු පාසල් ගියේ ය.", "ඉතිරි වචන"]


@pytest.mark.prolog
def test_conjunction_does_not_break_a_compound_subject(engine) -> None:
    """p.110: "මල්ලී ද තෝ ද මම ද එහි යමු." is one sentence, not four."""
    assert engine.micro_chunks("මල්ලී ද තෝ ද මම ද එහි යමු.") == [
        "මල්ලී ද තෝ ද මම ද එහි යමු."
    ]


@pytest.mark.prolog
def test_embedded_clause_stays_with_its_main_clause(engine) -> None:
    """p.103: the quotative must not sever අන්තර් වාක්‍යය from ප්‍රධාන වාක්‍යය."""
    assert engine.micro_chunks("සතුරන් ගමට එතියි ඔවුහු බිය වූහ.") == [
        "සතුරන් ගමට එතියි ඔවුහු බිය වූහ."
    ]


@pytest.mark.prolog
def test_no_word_is_lost(engine) -> None:
    text = "මම බත් කමි. ඔහු පාසල් ගියේ ය. අවසන් වචන කිහිපයක්"
    assert " ".join(engine.micro_chunks(text)).split() == text.split()


@pytest.mark.prolog
def test_repeated_words_are_cached(engine) -> None:
    """Sinhala repeats function words heavily; one query each, not one per token."""
    engine.is_split_point("කරමි")
    before = len(engine._cache)
    for _ in range(50):
        engine.is_split_point("කරමි")
    assert len(engine._cache) == before
