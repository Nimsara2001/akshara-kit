"""Python mirror of ``rules/sinhala_rules.pl``.

The Prolog knowledge base is the authority — it is what the report specifies and
what a linguist audits. This module exists for two jobs that cannot wait on a
SWI-Prolog process:

1. **De-wrapping during preprocessing.** Rejoining PDF line wraps needs a
   sentence-end test, and preprocessing must run on a machine with no Prolog
   installed.
2. **Parity testing.** ``test_rule_engine.py`` asserts that Prolog and this table
   agree on every fixture word, so a divergence surfaces as a test failure rather
   than as quietly different chunking.

Keeping two implementations in step is a real cost, accepted deliberately: the
alternative is either a hard SWI-Prolog dependency for basic text cleanup, or no
way to detect the Prolog rules drifting from what the library assumes.

Every table here corresponds to a fact group in the ``.pl`` file, under the same
name and citing the same grammar pages.
"""

from __future__ import annotations

__all__ = [
    "CLAUSE_ENDINGS",
    "CLAUSE_WORDS",
    "DISCOURSE_CONNECTIVES",
    "FINITE_ENDINGS",
    "INDECLINABLES",
    "NEVER_SPLIT_WORDS",
    "PRONOUNS",
    "PUNCTUATION",
    "SENTENCE_WORDS",
    "is_clause_end",
    "is_never_split",
    "is_quotative",
    "is_sentence_end",
]

#: Finite predicate (ආඛ්‍යාත) endings — වියරණ විවරණ pp. 94-96.
FINITE_ENDINGS: frozenset[str] = frozenset(
    {"මි", "මු", "හි", "හු", "යි", "ති", "හ"}
)

#: Words that close a sentence on their own: the කෘදන්ත particle ය (p.95), the
#: honorific සේක (p.99), and the obligation particles (p.107).
SENTENCE_WORDS: frozenset[str] = frozenset(
    {"ය", "සේක", "වටී", "යුතු", "යෙහෙකි", "මැනවි", "හොබී"}
)

PUNCTUATION: frozenset[str] = frozenset({".", "?", "!", "|", "||"})

#: Non-finite endings: absolutive, continuous, conditional, temporal,
#: concessive — p.107. These close a clause; the sentence runs on.
CLAUSE_ENDINGS: frozenset[str] = frozenset(
    {"ලා", "මින්", "ොත්", "තොත්", "ද්දී", "ද්දි", "තත්"}
)

CLAUSE_WORDS: frozenset[str] = frozenset({"නම්", "විට", "පසු", "ගොස්"})

DISCOURSE_CONNECTIVES: frozenset[str] = frozenset(
    {
        "නමුත්",
        "එහෙත්",
        "එබැවින්",
        "එමනිසා",
        "නිසා",
        "එනිසා",
        "එසේහෙයින්",
        "එහෙයින්",
    }
)

#: Particles binding within one clause — pp. 110-112. Splitting here orphans the
#: verb from its subject, so these override every other rule.
NEVER_SPLIT_WORDS: frozenset[str] = frozenset(
    {"සහ", "සමඟ", "සමග", "හා", "කැටුව", "ද", "හෝ", "නොහොත්", "නො", "නොව"}
)

#: Pronouns (සර්ව නාම) from the උක්ත/අනුක්ත table on p.91, plus the නිත්‍ය බහු
#: වචන of p.92. Enumerated because several end in a finite verb ending and would
#: otherwise be read as predicates — ඔහු, ඔවුහු and මොවුහු all end in හු, the 2pl
#: ending. "ඔහු පාසල් ගියේ ය." splitting after ඔහු is the most damaging false
#: positive available, since ඔහු is among the commonest words in the language.
#:
#: Structurally justified rather than patched: Sinhala is SOV and the sentence
#: ends with its ආඛ්‍යාතය (p.89), so a pronoun — an උක්ත or අනුක්ත by definition,
#: never a predicate — cannot end a sentence.
PRONOUNS: frozenset[str] = frozenset(
    {
        "මම", "මා", "අපි", "අප",
        "තෝ", "තා", "තී", "ඔබ", "නුඹ", "නුඹලා",
        "තොපි", "තෙපි", "තොප", "තෙප",
        "යුෂ්මතා", "යුෂ්මතී", "යුෂ්මත්හු", "යුෂ්මතුන්",
        "හේ", "ඕ", "ඈ", "හෙතෙම", "තෙමේ", "තොමෝ", "මෑ", "මෝ",
        "ඔහු", "ඇය", "මැය", "ඔවුහු", "මොවුහු", "තුමූ",
        "ඔවුන්", "මොවුන්", "මෙවුන්",
        "ඇතැමෙක්", "කෙනෙක්", "අයෙක්", "ඇතැම්හු", "ඇතැම්මු",
        "කවුරු", "ඇතැමෙකු", "කෙනෙකු", "අයෙකු", "ඇතැමුන්", "කවුරුන්",
    }
)

#: Deictic locatives. -හි is the locative case suffix as well as the 2sg verb
#: ending, and these are its highest-frequency victims: "මල්ලී ද තෝ ද මම ද එහි
#: යමු." (p.110) would otherwise split one word before its actual verb.
#:
#: This does not solve the -හි ambiguity in general — an open-class locative
#: noun (සමයෙහි, ලෝකයෙහි) is indistinguishable from a 2sg verb on surface form,
#: and separating them needs part-of-speech information. Closing the frequent
#: cases is what a word list can honestly do; the rest stays a known limitation.
INDECLINABLES: frozenset[str] = frozenset({"එහි", "මෙහි", "කොහි"})

_QUOTATIVE_MARKS = ("'යි", "’යි")


def is_quotative(word: str) -> bool:
    """True if ``යි`` here closes an embedded clause rather than a verb.

    ``යි`` is both the 3sg finite ending (කරයි) and the quotative that ends an
    අන්තර් වාක්‍යය (p.103: "සතුරන් ගමට එති’යි ඔවුහු බිය වූහ"). The grammar's
    discriminator is that the quotative attaches to an *already finite* form, so
    strip it and ask whether what remains is itself finite.
    """
    if word.endswith(_QUOTATIVE_MARKS):
        return True
    if not word.endswith("යි"):
        return False
    stem = word[: -len("යි")]
    return bool(stem) and any(stem.endswith(end) for end in FINITE_ENDINGS)


def is_never_split(word: str) -> bool:
    """True for words whose splitting is known to destroy meaning."""
    return (
        word in NEVER_SPLIT_WORDS
        or word in PRONOUNS
        or word in INDECLINABLES
        or is_quotative(word)
    )


def _ends_with_longer(word: str, endings: frozenset[str]) -> bool:
    """True if ``word`` carries one of ``endings`` and is longer than it.

    The length guard stops a word that merely *is* an ending — the pronoun මු,
    say — being read as a verb inflected with it.
    """
    return any(word.endswith(e) and len(word) > len(e) for e in endings)


def is_sentence_end(word: str) -> bool:
    """True if this word ends a sentence (Tier 1)."""
    if is_never_split(word):
        return False
    if any(word.endswith(p) for p in PUNCTUATION):
        return True
    if word in SENTENCE_WORDS:
        return True
    return _ends_with_longer(word, FINITE_ENDINGS)


def is_clause_end(word: str) -> bool:
    """True if this word ends a clause (Tier 2), sentence ends included.

    A sentence boundary is a fortiori a clause boundary, so the clause pass is a
    superset of the sentence pass.
    """
    if is_never_split(word):
        return False
    if is_sentence_end(word):
        return True
    if word in CLAUSE_WORDS or word in DISCOURSE_CONNECTIVES:
        return True
    return _ends_with_longer(word, CLAUSE_ENDINGS)
