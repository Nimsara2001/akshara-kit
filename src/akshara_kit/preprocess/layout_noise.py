"""Remove the scars PDF layout leaves on extracted text.

The Eye returns a faithful transcription of the page, which is not the same as
readable prose. Measured on ``output/sample_unicode.pdf.txt`` (334,543 chars):
8,401 lines for what is really a few hundred paragraphs, 2,570 of them blank,
202 bare page numbers, and sentences broken mid-clause wherever the typesetter
wrapped a line.

Feeding that to a chunker produces chunks that are page furniture rather than
content — a chunk containing only "94" is not worth embedding.
"""

from __future__ import annotations

import re
from collections import Counter

from akshara_kit.brain.rule_tables import is_sentence_end

__all__ = [
    "dewrap",
    "drop_noise_lines",
    "is_noise_line",
]

#: A line that is only a page number, in Arabic or Roman numerals. Roman is
#: needed because front matter is numbered i, ii, iii ... — the grammar PDF's
#: first eleven pages are exactly this.
_PAGE_NUMBER = re.compile(r"^\s*[\divxlcdmIVXLCDM]{1,7}\s*$")

#: Table-of-contents dot leaders: "හැඳින්වීම ......... 4".
_DOT_LEADER_MIN = 8

#: A running header or footer repeats across pages. Detected by frequency rather
#: than by pattern, because the text varies per document and a hard-coded regex
#: would only ever fit the corpus it was written against.
_MIN_REPEATS_FOR_FURNITURE = 3
_MAX_FURNITURE_WORDS = 12


def is_noise_line(line: str) -> bool:
    """True for a line that is page furniture rather than content."""
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER.match(stripped):
        return True
    return stripped.count(".") >= _DOT_LEADER_MIN


def _repeated_furniture(lines: list[str]) -> set[str]:
    """Short lines that recur often enough to be a header or footer."""
    counts = Counter(
        stripped
        for line in lines
        if (stripped := line.strip())
        and len(stripped.split()) <= _MAX_FURNITURE_WORDS
    )
    return {
        text
        for text, count in counts.items()
        if count >= _MIN_REPEATS_FOR_FURNITURE and not is_sentence_end(text.split()[-1])
    }


def drop_noise_lines(text: str) -> tuple[str, int]:
    """Remove page numbers, dot leaders and running headers or footers.

    Returns ``(text, lines_removed)``.
    """
    lines = text.split("\n")
    furniture = _repeated_furniture(lines)

    kept = [
        line
        for line in lines
        if not is_noise_line(line) and line.strip() not in furniture
    ]
    return "\n".join(kept), len(lines) - len(kept)


def dewrap(text: str) -> tuple[str, int]:
    """Rejoin lines the typesetter wrapped mid-sentence.

    A line is joined to the next unless it already ends a sentence, is blank, or
    looks tabular. The sentence test is the Sinhala rule base's, so wrapping and
    chunking cannot disagree about where a sentence ends — a boundary the
    chunker would honour is a boundary this leaves alone.

    Returns ``(text, joins_made)``.
    """
    lines = text.split("\n")
    out: list[str] = []
    joins = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue

        # A tab means cell structure; joining would merge table rows.
        if "\t" in line or not out or not out[-1].strip() or "\t" in out[-1]:
            out.append(stripped)
            continue

        previous = out[-1]
        last_word = previous.split()[-1] if previous.split() else ""
        if is_sentence_end(last_word):
            out.append(stripped)
        else:
            out[-1] = f"{previous} {stripped}"
            joins += 1

    return "\n".join(out), joins
