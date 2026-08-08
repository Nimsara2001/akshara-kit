"""Cut text into atomic segments the chunker may never merge across.

Prose can be resegmented freely; tabular data cannot. The XLSX adapter emits one
row per line with ``\\t`` between cells, and the DOCX adapter does the same for
table cells. Run Algorithm 3 over that and two things go wrong, both visible in
``sample.xlsx``:

- Cells from unrelated rows land in one chunk. "සීගිරිය මාතලේ රු. 100 පැරණි
  බලකොටුව … ශ්‍රී දළදා මාළිගාව මහනුවර …" reads like a sentence and describes two
  different places. It is worse than useless in a retrieval index, because it
  will match queries about either and answer about neither.
- Rows split *internally* on incidental punctuation. The price cell ``රු. 100``
  contains a full stop, so the punctuation rule fires mid-row.

So tabular rows are atomic: never split, never merged with a neighbour. One row
is one record and becomes one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from akshara_kit.contracts.chunking import SegmentKind
from akshara_kit.contracts.extraction import SourceFormat

__all__ = ["Segment", "segment"]

#: The sheet banner the XLSX adapter injects, e.g. ``=== සංචාරක ස්ථාන ===``.
_SHEET_HEADING = re.compile(r"^\s*={3}\s*.+?\s*={3}\s*$")

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of text with a kind that decides how it may be chunked."""

    text: str
    kind: SegmentKind

    @property
    def is_atomic(self) -> bool:
        """True if this segment must survive into a chunk unaltered."""
        return self.kind is not SegmentKind.PROSE


def segment(
    text: str,
    source_format: SourceFormat | None = None,
    *,
    respect_table_rows: bool = True,
) -> list[Segment]:
    """Split text into segments, protecting tabular structure.

    ``respect_table_rows=False`` treats everything as prose, reproducing the
    naive behaviour — kept so the guard's effect can be measured rather than
    asserted.
    """
    if not text.strip():
        return []

    if not respect_table_rows:
        return [Segment(text=text.strip(), kind=SegmentKind.PROSE)]

    if source_format is SourceFormat.XLSX:
        return _segment_spreadsheet(text)
    return _segment_prose(text)


def _segment_spreadsheet(text: str) -> list[Segment]:
    """Every line is a record: a sheet heading or one row."""
    segments: list[Segment] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        kind = (
            SegmentKind.SHEET_HEADING
            if _SHEET_HEADING.match(stripped)
            else SegmentKind.TABLE_ROW
        )
        segments.append(Segment(text=stripped, kind=kind))
    return segments


def _segment_prose(text: str) -> list[Segment]:
    """Paragraphs, with any tab-bearing line lifted out as a table row.

    A DOCX mixes prose and tables in one stream, so the tab is the only signal
    that a line is tabular. Checking per line rather than per document means a
    table inside an otherwise prose document is still protected.
    """
    segments: list[Segment] = []
    for block in _PARAGRAPH_BREAK.split(text):
        if not block.strip():
            continue
        segments.extend(_split_block(block))
    return segments


def _split_block(block: str) -> list[Segment]:
    """Separate tab-bearing lines from the prose around them."""
    segments: list[Segment] = []
    prose: list[str] = []

    for line in block.split("\n"):
        if not line.strip():
            continue
        if "\t" in line:
            if prose:
                segments.append(Segment(" ".join(prose), SegmentKind.PROSE))
                prose = []
            segments.append(Segment(line.strip(), SegmentKind.TABLE_ROW))
        else:
            prose.append(line.strip())

    if prose:
        segments.append(Segment(" ".join(prose), SegmentKind.PROSE))
    return segments
