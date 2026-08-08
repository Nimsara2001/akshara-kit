"""Symbolic Rule Engine Adapter (realises report Section 6.5.2).

Bridges the Python host to the SWI-Prolog knowledge base over the Machine Query
Interface, and implements Algorithm 3 — walk the word stream once, flushing the
accumulator whenever the rule base reports a boundary.

Three departures from the prototype in ``fyp_testing/the_brain``, each fixing a
concrete problem:

**One socket round trip per word becomes one per distinct word.** The prototype
queried Prolog for every token; a 50,000-word document meant 50,000 round trips
over a TCP socket. Sinhala prose repeats function words heavily, so caching on
the word collapses that to the size of the vocabulary — typically a few thousand
— and the answer is deterministic, so caching cannot change behaviour.

**``__del__`` teardown becomes a context manager.** Interpreter shutdown ordering
makes ``__del__`` an unreliable place to close a socket; ``__del__`` is kept only
as a backstop for callers who ignore the context manager.

**The rule file is located through ``importlib.resources``.** The prototype's
relative ``consult("sinhala_rules.pl")`` only resolves when the process happens
to be running from inside the source directory, so it breaks in an installed
wheel.
"""

from __future__ import annotations

import logging
from typing import Iterator

from akshara_kit.contracts.chunking import BoundaryKind
from akshara_kit.eye.errors import AdapterUnavailableError, PrologUnavailableError

__all__ = ["RULE_FILE", "SymbolicRuleEngine"]

logger = logging.getLogger(__name__)

RULE_FILE = "sinhala_rules.pl"

_LEVELS = {BoundaryKind.SENTENCE: "sentence", BoundaryKind.CLAUSE: "clause"}


class SymbolicRuleEngine:
    """Answers, per word, whether it terminates a micro-chunk.

    Use as a context manager so the Prolog process is closed deterministically::

        with SymbolicRuleEngine() as engine:
            chunks = engine.micro_chunks(text)
    """

    def __init__(self) -> None:
        self._mqi = None
        self._thread = None
        self._cache: dict[tuple[str, str], bool] = {}
        self._start()

    # --- lifecycle --------------------------------------------------------

    def _start(self) -> None:
        """Boot SWI-Prolog and consult the rule base once."""
        from akshara_kit.eye import capabilities

        if not capabilities.swipl_available():
            raise PrologUnavailableError(capabilities.describe_swipl_availability())

        try:
            from swiplserver import PrologMQI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise AdapterUnavailableError(
                "swiplserver is not installed; install the 'brain' extra"
            ) from exc

        self._mqi = PrologMQI(prolog_path_args=None)
        self._thread = self._mqi.create_thread()
        self._thread.query(f'consult("{_rule_file_path()}")')

    def close(self) -> None:
        """Shut the Prolog process down. Safe to call more than once."""
        if self._mqi is not None:
            try:
                self._mqi.stop()
            except Exception:  # noqa: BLE001 - teardown must never raise
                logger.debug("SWI-Prolog shutdown was not clean", exc_info=True)
            finally:
                self._mqi = None
                self._thread = None

    def __enter__(self) -> SymbolicRuleEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - backstop only
        self.close()

    # --- queries ----------------------------------------------------------

    def is_split_point(
        self, word: str, level: BoundaryKind = BoundaryKind.SENTENCE
    ) -> bool:
        """True if ``word`` terminates a micro-chunk at this level.

        Cached: the rule base is a pure function of the word, so the same answer
        can never become wrong within a session.
        """
        if not word:
            return False

        level_atom = _LEVELS.get(level, "sentence")
        key = (word, level_atom)
        if key in self._cache:
            return self._cache[key]

        answer = self._query(word, level_atom)
        self._cache[key] = answer
        return answer

    def _query(self, word: str, level_atom: str) -> bool:
        """Ask Prolog. Never raises on a linguistic answer, only on transport."""
        escaped = word.replace("\\", "\\\\").replace("'", "\\'")
        result = self._thread.query(
            f"check_split('{escaped}', {level_atom}, Result)"
        )
        # check_split/3 is total, so a bare False here means the transport or
        # the consult failed — worth distinguishing from an honest 'false'.
        if result is False:
            logger.warning("rule base returned no solution for %r", word)
            return False
        return result[0]["Result"] == "true"

    # --- Algorithm 3 ------------------------------------------------------

    def micro_chunks(
        self, text: str, level: BoundaryKind = BoundaryKind.SENTENCE
    ) -> list[str]:
        """Segment text into micro-chunks (Algorithm 3).

        Whitespace-tokenise, accumulate, and flush whenever the rule base reports
        a boundary. Any residue is flushed at the end, so no word is dropped.
        """
        return list(self.iter_micro_chunks(text, level))

    def iter_micro_chunks(
        self, text: str, level: BoundaryKind = BoundaryKind.SENTENCE
    ) -> Iterator[str]:
        """Algorithm 3, streamed rather than materialised."""
        current: list[str] = []
        for word in text.split():
            current.append(word)
            if self.is_split_point(word, level):
                yield " ".join(current)
                current = []
        if current:
            yield " ".join(current)


def _rule_file_path() -> str:
    """Absolute path to the rule base, wheel-safe.

    ``importlib.resources`` handles the installed case; the path is handed to
    Prolog as a string with forward slashes because backslashes are escapes in
    Prolog quoted atoms.
    """
    from importlib import resources

    path = resources.files("akshara_kit.brain") / "rules" / RULE_FILE
    return str(path).replace("\\", "/")
