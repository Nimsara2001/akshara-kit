"""Segmentation, the bounded agglomerative merge, and the output surface.

The coherence scorer is stubbed throughout, so none of this needs torch, a model
download, or a network — the coordinator depends on the ``CoherenceScorer``
Protocol precisely so the merge logic can be tested in isolation from LaBSE.

Tests that segment prose need the Prolog rule base and carry the ``prolog``
marker; the segmenter and output-surface tests need neither.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from akshara_kit.brain.segmenter import segment
from akshara_kit.contracts.chunking import (
    ChunkConfig,
    ChunkedDocument,
    SegmentKind,
    SemanticChunk,
)
from akshara_kit.contracts.extraction import SourceFormat


class ConstantScorer:
    """Returns the same similarity for every pair.

    ``1.0`` merges everything the guardrail permits, ``0.0`` merges nothing —
    which makes each branch of Algorithm 4 individually observable.
    """

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def score(self, left: str, right: str) -> float:
        self.calls += 1
        return self.value


# --- segmentation ---------------------------------------------------------


def test_spreadsheet_rows_are_atomic_segments() -> None:
    text = "=== සංචාරක ===\nසීගිරිය\tමාතලේ\nඇල්ල\tබදුල්ල"
    segments = segment(text, SourceFormat.XLSX)
    assert [s.kind for s in segments] == [
        SegmentKind.SHEET_HEADING,
        SegmentKind.TABLE_ROW,
        SegmentKind.TABLE_ROW,
    ]
    assert all(s.is_atomic for s in segments)


def test_prose_is_split_on_blank_lines() -> None:
    segments = segment("පළමු ඡේදය.\n\nදෙවන ඡේදය.", SourceFormat.PDF)
    assert len(segments) == 2
    assert all(s.kind is SegmentKind.PROSE for s in segments)


def test_a_table_inside_prose_is_still_protected() -> None:
    """A DOCX mixes both in one stream; the tab is the only signal."""
    segments = segment("හැඳින්වීම මෙසේ ය.\nසීගිරිය\tමාතලේ\nඅවසානය.", SourceFormat.DOCX)
    kinds = [s.kind for s in segments]
    assert SegmentKind.TABLE_ROW in kinds
    assert kinds.count(SegmentKind.PROSE) == 2


def test_the_guard_can_be_switched_off_for_comparison() -> None:
    segments = segment(
        "සීගිරිය\tමාතලේ\nඇල්ල\tබදුල්ල",
        SourceFormat.XLSX,
        respect_table_rows=False,
    )
    assert len(segments) == 1
    assert segments[0].kind is SegmentKind.PROSE


def test_empty_input_yields_no_segments() -> None:
    assert segment("", SourceFormat.PDF) == []
    assert segment("   \n\n ", SourceFormat.XLSX) == []


# --- tabular safety, against the real fixture -----------------------------


@pytest.mark.prolog
def test_spreadsheet_rows_are_never_blended(sample_xlsx: pathlib.Path) -> None:
    """The regression this guard exists for.

    With a scorer that wants to merge everything and a guardrail high enough to
    allow it, rows must still come out one per chunk. Blending them produces a
    chunk that reads like a sentence and describes two unrelated records.
    """
    from akshara_kit import route
    from akshara_kit.brain import chunk

    doc = chunk(
        route(str(sample_xlsx)),
        scorer=ConstantScorer(1.0),
        config=ChunkConfig(max_words=500),
    )

    rows = [c for c in doc if c.segment_kind is SegmentKind.TABLE_ROW]
    assert rows, "the fixture should contain table rows"

    # Every row of this fixture has four tab-separated cells. A chunk holding
    # more than that has merged two records.
    for chunk_ in rows:
        assert chunk_.text.count("\t") == 3, f"cells blended: {chunk_.text[:60]!r}"


@pytest.mark.prolog
def test_sheet_heading_stands_alone(sample_xlsx: pathlib.Path) -> None:
    from akshara_kit import route
    from akshara_kit.brain import chunk

    doc = chunk(route(str(sample_xlsx)), scorer=ConstantScorer(1.0))
    headings = [c for c in doc if c.segment_kind is SegmentKind.SHEET_HEADING]
    assert len(headings) == 1
    assert "\t" not in headings[0].text


@pytest.mark.prolog
def test_disabling_the_guard_does_blend_rows(sample_xlsx: pathlib.Path) -> None:
    """Proves the guard is what protects the rows, not luck in the data.

    Two things go wrong at once without it, and both are visible here: cells
    from unrelated records land in one chunk, and the tab structure is destroyed
    outright, because Algorithm 3 tokenises on whitespace and rejoins on spaces.
    """
    from akshara_kit import route
    from akshara_kit.brain import chunk

    doc = chunk(
        route(str(sample_xlsx)),
        scorer=ConstantScorer(1.0),
        config=ChunkConfig(max_words=500, respect_table_rows=False),
    )

    # Cell structure is gone — nothing downstream can recover the columns.
    assert all("\t" not in c.text for c in doc)

    # And two different tourist sites are now described by one chunk.
    blended = [c for c in doc if "සීගිරිය" in c.text and "ඇල්ල" in c.text]
    assert blended, "expected unrelated rows to be blended without the guard"


# --- Algorithm 4 ----------------------------------------------------------


@pytest.mark.prolog
def test_low_coherence_never_merges() -> None:
    from akshara_kit.brain import chunk_text

    doc = chunk_text(
        "මම බත් කමි. ඔහු පාසල් ගියේ ය.",
        source_format=SourceFormat.PDF,
        scorer=ConstantScorer(0.0),
    )
    assert len(doc) == 2


@pytest.mark.prolog
def test_high_coherence_merges_within_the_guardrail() -> None:
    from akshara_kit.brain import chunk_text

    doc = chunk_text(
        "මම බත් කමි. ඔහු පාසල් ගියේ ය.",
        source_format=SourceFormat.PDF,
        scorer=ConstantScorer(1.0),
        config=ChunkConfig(max_words=50),
    )
    assert len(doc) == 1
    assert doc[0].merged_from == 2


@pytest.mark.prolog
def test_guardrail_is_checked_before_coherence() -> None:
    """Section 6.5.4: the length question is asked first.

    With a scorer that always wants to merge, a tight bound must still split —
    otherwise a self-similar passage would grow without limit.
    """
    from akshara_kit.brain import chunk_text

    scorer = ConstantScorer(1.0)
    doc = chunk_text(
        "මම බත් කමි. ඔහු පාසල් ගියේ ය.",
        source_format=SourceFormat.PDF,
        scorer=scorer,
        config=ChunkConfig(max_words=4),
    )
    assert len(doc) == 2
    # The guardrail short-circuits, so coherence was never consulted.
    assert scorer.calls == 0


@pytest.mark.prolog
def test_residual_accumulator_is_flushed() -> None:
    """Algorithm 4's final step: no text may be dropped."""
    from akshara_kit.brain import chunk_text

    text = "මම බත් කමි. ඔහු පාසල් ගියේ ය. ඉතිරි වචන කිහිපයක්"
    doc = chunk_text(text, source_format=SourceFormat.PDF, scorer=ConstantScorer(0.0))
    assert " ".join(doc.texts).split() == text.split()


@pytest.mark.prolog
def test_single_micro_chunk_returns_early() -> None:
    from akshara_kit.brain import chunk_text

    doc = chunk_text("මම බත් කමි.", source_format=SourceFormat.PDF, scorer=ConstantScorer(0.0))
    assert len(doc) == 1


def test_empty_text_yields_no_chunks() -> None:
    from akshara_kit.brain import chunk_text

    doc = chunk_text("", source_format=SourceFormat.PDF, scorer=ConstantScorer(1.0))
    assert len(doc) == 0
    assert not doc


# --- the output surface ---------------------------------------------------


@pytest.fixture
def document() -> ChunkedDocument:
    chunks = [
        SemanticChunk(text=f"කොටස {i}", chunk_id=f"c{i}", index=i, word_count=2)
        for i in range(6)
    ]
    return ChunkedDocument(chunks=chunks, source_document="sample.pdf")


def test_length_and_iteration(document: ChunkedDocument) -> None:
    assert len(document) == 6
    assert [c.index for c in document] == list(range(6))


def test_index_returns_one_chunk(document: ChunkedDocument) -> None:
    assert document[2].chunk_id == "c2"


def test_slice_returns_a_range(document: ChunkedDocument) -> None:
    assert [c.chunk_id for c in document[2:5]] == ["c2", "c3", "c4"]


def test_range_matches_slicing(document: ChunkedDocument) -> None:
    assert document.range(1, 3) == document[1:3]


def test_texts_are_plain_strings(document: ChunkedDocument) -> None:
    assert document.texts == [f"කොටස {i}" for i in range(6)]
    assert all(isinstance(t, str) for t in document.texts)


def test_empty_document_is_falsy() -> None:
    assert not ChunkedDocument()


def test_json_round_trips(document: ChunkedDocument, tmp_path: pathlib.Path) -> None:
    path = tmp_path / "chunks.json"
    document.to_json(path)
    assert ChunkedDocument.model_validate_json(path.read_text(encoding="utf-8")) == document


def test_jsonl_is_one_object_per_line(document: ChunkedDocument, tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    document.to_jsonl(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(document)
    assert json.loads(lines[0])["chunk_id"] == "c0"


def test_to_dicts_is_json_safe(document: ChunkedDocument) -> None:
    json.dumps(document.to_dicts())


@pytest.mark.prolog
def test_chunk_ids_are_stable_across_runs(sample_xlsx: pathlib.Path) -> None:
    """A re-ingest must not churn every id in a vector store."""
    from akshara_kit import route
    from akshara_kit.brain import chunk

    result = route(str(sample_xlsx))
    first = chunk(result, scorer=ConstantScorer(0.0), source_document="sample.xlsx")
    second = chunk(result, scorer=ConstantScorer(0.0), source_document="sample.xlsx")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


@pytest.mark.prolog
def test_chunks_carry_provenance_and_quality(sample_xlsx: pathlib.Path) -> None:
    from akshara_kit import route
    from akshara_kit.brain import chunk

    doc = chunk(
        route(str(sample_xlsx)), scorer=ConstantScorer(0.0), source_document="sample.xlsx"
    )
    assert doc.source_format is SourceFormat.XLSX
    assert all(c.source_document == "sample.xlsx" for c in doc)
    assert all(c.quality is not None for c in doc)
    assert doc.metadata["segments"] > 0
