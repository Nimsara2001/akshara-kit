"""Data contracts for the Brain module (realises report Table 5.1).

Where :mod:`akshara_kit.contracts.extraction` is the Eye's output, this module is
the Brain's. :class:`SemanticChunk` is the unit the API layer consumes;
:class:`ChunkedDocument` is the container it arrives in.

On the output surface
---------------------
A caller who wants plain strings should not have to learn pydantic, and a caller
who wants provenance should not have to reconstruct it. So
:class:`ChunkedDocument` behaves like a sequence — ``len``, indexing, slicing,
iteration — while still carrying the full metadata for the callers that need it.
``to_json`` and ``to_jsonl`` cover the two shapes downstream tooling actually
asks for.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Iterator, overload

from pydantic import BaseModel, Field

from akshara_kit.contracts.extraction import QualityScore, SourceFormat

__all__ = [
    "BoundaryKind",
    "ChunkConfig",
    "ChunkedDocument",
    "SegmentKind",
    "SemanticChunk",
]


class BoundaryKind(str, Enum):
    """Why a chunk ends where it does.

    Recorded per chunk so a linguist can audit the chunker's decisions without
    re-running it — a boundary the rule base found reads differently from one the
    length guardrail forced.
    """

    #: A finite verb (ආඛ්‍යාතය) or sentence-final particle closed the sentence.
    SENTENCE = "sentence"
    #: A non-finite form closed a clause but not the sentence.
    CLAUSE = "clause"
    PUNCTUATION = "punctuation"
    #: A discourse connective (නමුත්, එබැවින්) started a new thought.
    DISCOURSE = "discourse"
    #: ``max_words`` forced the split; not a linguistic decision.
    GUARDRAIL = "guardrail"
    #: Topic coherence fell below the threshold.
    COHERENCE = "coherence"
    #: A table row ended. Structural, never merged across.
    TABLE_ROW = "table_row"
    #: An atomic segment ended (paragraph, sheet).
    SEGMENT = "segment"


class SegmentKind(str, Enum):
    """What kind of material a chunk holds.

    The distinction is load-bearing for tabular sources: prose may be merged and
    resplit freely, a spreadsheet row may not. Blending cells from two rows
    produces a chunk that reads like a sentence and means nothing.
    """

    PROSE = "prose"
    TABLE_ROW = "table_row"
    SHEET_HEADING = "sheet_heading"


class ChunkConfig(BaseModel):
    """Tuning for the hybrid bounded agglomerative merge (Algorithm 4)."""

    #: Below this cosine similarity two adjacent micro-chunks are deemed to be
    #: about different things and are not merged. 0.6, the pre-calibration
    #: default, sat *above* the fine-tuned checkpoint's own reported mean for
    #: genuinely adjacent sentence pairs (0.5313 vs. 0.1620 for paragraph-
    #: boundary hard negatives — report Section 6.5.3), so it rejected most
    #: true continuations by construction. Scoring every real adjacent
    #: micro-chunk pair in the fixture corpus with the checkpoint gives a
    #: median of ~0.39 and a 25th percentile of ~0.20; 0.2 merges the
    #: three-quarters of pairs the model does not consider actively
    #: dissimilar, while still splitting on the clearest topic changes.
    #: Raise it to bias toward smaller, more tightly-on-topic chunks; lower it
    #: (toward 0.0) to bias toward hitting ``max_words`` more often at the
    #: cost of merging more loosely related material.
    similarity_threshold: float = Field(default=0.2, ge=-1.0, le=1.0)
    #: Upper bound on a final chunk, in words. Checked *before* similarity, so a
    #: long chunk is split even when its halves are perfectly coherent.
    #: Sinhala prose in the fixture corpus averages ~6.7 characters per word, so
    #: 300 words lands a fully-merged chunk in the ~1,600-2,000-character range
    #: (~400-512 embedding tokens) that downstream retrieval targets; a lower
    #: value is fine when the source material genuinely runs shorter.
    max_words: int = Field(default=300, ge=1)
    #: Which rule-base pass supplies micro-chunk boundaries. ``SENTENCE`` yields
    #: fewer, more self-contained units; ``CLAUSE`` yields finer ones.
    level: BoundaryKind = BoundaryKind.SENTENCE
    #: Keep table rows atomic. Turn off only to reproduce the naive behaviour for
    #: comparison — it will blend cell values across rows.
    respect_table_rows: bool = True


class SemanticChunk(BaseModel):
    """One final chunk — the Brain's unit of output (Table 5.1)."""

    text: str
    #: Stable across runs for identical input: derived from the source document
    #: and the chunk's position, so a re-ingest does not churn vector-store ids.
    chunk_id: str
    index: int
    word_count: int

    source_document: str | None = None
    source_format: SourceFormat | None = None
    segment_kind: SegmentKind = SegmentKind.PROSE

    #: Per-chunk quality indicators, reusing the Eye's probe so a chunk is scored
    #: the same way the document was.
    quality: QualityScore | None = None
    #: The boundaries that fell inside this chunk, in order.
    boundaries: list[BoundaryKind] = Field(default_factory=list)
    #: How many micro-chunks were merged to make this one. ``1`` means the rule
    #: base alone decided it.
    merged_from: int = 1

    # Table 5.1 also specifies a page range and layout-region bounding boxes.
    # Neither is derivable yet: ExtractionResult.text is pages joined with a
    # separator and carries no offset map, and the layout analyser is a stub.
    # Declared so the contract is honest about the gap rather than silent.
    page_range: tuple[int, int] | None = None
    bounding_boxes: list[dict] = Field(default_factory=list)


class ChunkedDocument(BaseModel):
    """Every chunk of one document, plus how it was produced.

    Supports ``len(doc)``, ``doc[3]``, ``doc[2:5]``, iteration, ``doc.texts`` for
    plain strings, and JSON/JSONL export.
    """

    chunks: list[SemanticChunk] = Field(default_factory=list)
    source_document: str | None = None
    source_format: SourceFormat | None = None
    config: ChunkConfig = Field(default_factory=ChunkConfig)
    #: Timings and counts — micro-chunks produced, segments seen, scorer used.
    metadata: dict = Field(default_factory=dict)

    # --- sequence protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator[SemanticChunk]:  # type: ignore[override]
        return iter(self.chunks)

    @overload
    def __getitem__(self, key: int) -> SemanticChunk: ...

    @overload
    def __getitem__(self, key: slice) -> list[SemanticChunk]: ...

    def __getitem__(self, key: int | slice) -> SemanticChunk | list[SemanticChunk]:
        """Index for one chunk, slice for a range."""
        return self.chunks[key]

    def __bool__(self) -> bool:
        # Without this, pydantic's BaseModel truthiness would ignore __len__ and
        # an empty document would read as truthy.
        return bool(self.chunks)

    # --- output shapes ----------------------------------------------------

    @property
    def texts(self) -> list[str]:
        """Just the strings, for callers that want nothing else."""
        return [chunk.text for chunk in self.chunks]

    def range(self, start: int, end: int) -> list[SemanticChunk]:
        """Chunks ``start`` up to but not including ``end``.

        The same as ``doc[start:end]``; named for callers who prefer the explicit
        form, and for keyword use across an API boundary.
        """
        return self.chunks[start:end]

    def to_dicts(self) -> list[dict]:
        """Each chunk as a plain dict, metadata included."""
        return [chunk.model_dump(mode="json") for chunk in self.chunks]

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialise the whole document, chunks and provenance together.

        Round-trips through :meth:`model_validate_json`. Writes to ``path`` when
        given, and returns the JSON either way.
        """
        payload = self.model_dump_json(indent=indent)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_jsonl(self, path: str | Path | None = None) -> str:
        """One JSON object per line — the shape vector-store loaders expect.

        Document-level metadata does not survive this format, so each line
        carries the chunk's own provenance and nothing more.
        """
        payload = "\n".join(
            json.dumps(chunk, ensure_ascii=False) for chunk in self.to_dicts()
        )
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload
