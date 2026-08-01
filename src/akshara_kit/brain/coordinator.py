"""Hybrid Chunking Coordinator (realises report Section 6.5.4).

Owns one rule engine and one coherence scorer, and drives the three-stage
algorithm: symbolic micro-chunking, neural coherence scoring, and the bounded
agglomerative merge of Algorithm 4.

The merge state machine has exactly three branches, and the **order matters**:
the length guardrail is asked *before* the coherence question, so a chunk that
would exceed ``max_words`` is split even when its halves are perfectly coherent.
Asking coherence first would let a highly self-similar passage grow without
bound.

Atomic segments (table rows, sheet headings) bypass both stages entirely — see
:mod:`akshara_kit.brain.segmenter` for why blending spreadsheet rows produces
chunks that read like prose and mean nothing.
"""

from __future__ import annotations

import hashlib
import time

from akshara_kit.brain.encoder import CoherenceScorer
from akshara_kit.brain.rule_engine import SymbolicRuleEngine
from akshara_kit.brain.segmenter import Segment, segment
from akshara_kit.contracts.chunking import (
    BoundaryKind,
    ChunkConfig,
    ChunkedDocument,
    SegmentKind,
    SemanticChunk,
)
from akshara_kit.contracts.extraction import ExtractionResult, SourceFormat
from akshara_kit.eye.quality_probe import score as quality_score

__all__ = ["HybridChunker", "chunk", "chunk_text"]


class HybridChunker:
    """Drives Algorithm 4 over one document.

    Use as a context manager so the Prolog process is closed deterministically::

        with HybridChunker(scorer=my_scorer) as chunker:
            doc = chunker.chunk_text(text)
    """

    def __init__(
        self,
        *,
        config: ChunkConfig | None = None,
        scorer: CoherenceScorer | None = None,
        engine: SymbolicRuleEngine | None = None,
    ):
        self.config = config or ChunkConfig()
        self._scorer = scorer
        self._engine = engine
        self._owns_engine = engine is None

    # --- lifecycle --------------------------------------------------------

    @property
    def engine(self) -> SymbolicRuleEngine:
        """The rule engine, started on first use."""
        if self._engine is None:
            self._engine = SymbolicRuleEngine()
        return self._engine

    @property
    def scorer(self) -> CoherenceScorer:
        """The coherence scorer, defaulting to LaBSE if none was supplied."""
        if self._scorer is None:
            from akshara_kit.brain.encoder import LabseScorer

            self._scorer = LabseScorer()
        return self._scorer

    def close(self) -> None:
        """Release the Prolog process, if this chunker started it."""
        if self._owns_engine and self._engine is not None:
            self._engine.close()
            self._engine = None

    def __enter__(self) -> HybridChunker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- public API -------------------------------------------------------

    def chunk(self, result: ExtractionResult, *, source_document: str | None = None):
        """Chunk an :class:`ExtractionResult`, honouring its source format."""
        return self.chunk_text(
            result.text,
            source_format=result.source_format,
            source_document=source_document,
        )

    def chunk_text(
        self,
        text: str,
        *,
        source_format: SourceFormat | None = None,
        source_document: str | None = None,
    ) -> ChunkedDocument:
        """Chunk raw text. Prefer :meth:`chunk`, which knows the format."""
        started = time.perf_counter()
        segments = segment(
            text,
            source_format,
            respect_table_rows=self.config.respect_table_rows,
        )

        pieces: list[tuple[str, SegmentKind, int, list[BoundaryKind]]] = []
        micro_total = 0
        for seg in segments:
            if seg.is_atomic:
                pieces.append((seg.text, seg.kind, 1, [BoundaryKind.TABLE_ROW]))
                continue
            micro = self.engine.micro_chunks(seg.text, self.config.level)
            micro_total += len(micro)
            pieces.extend(self._merge(micro))

        chunks = [
            self._build_chunk(i, text_, kind, merged, bounds, source_document, source_format)
            for i, (text_, kind, merged, bounds) in enumerate(pieces)
        ]

        return ChunkedDocument(
            chunks=chunks,
            source_document=source_document,
            source_format=source_format,
            config=self.config,
            metadata={
                "segments": len(segments),
                "micro_chunks": micro_total,
                "latency_seconds": time.perf_counter() - started,
                "scorer": type(self._scorer).__name__ if self._scorer else None,
            },
        )

    # --- Algorithm 4 ------------------------------------------------------

    def _merge(
        self, micro_chunks: list[str]
    ) -> list[tuple[str, SegmentKind, int, list[BoundaryKind]]]:
        """Bounded agglomerative merge over one prose segment.

        Note ``max_words`` bounds *merging*, not the micro-chunks themselves: the
        guardrail is tested against ``combined``, so a single micro-chunk that is
        already longer than the bound is emitted intact. This is Algorithm 4 as
        specified, and it is the right behaviour — the alternative is cutting a
        sentence at an arbitrary word, which is precisely what the rule base
        exists to avoid. It does mean a final chunk can exceed ``max_words`` when
        the source sentence does.
        """
        if not micro_chunks:
            return []
        if len(micro_chunks) == 1:
            return [(micro_chunks[0], SegmentKind.PROSE, 1, [self.config.level])]

        final: list[tuple[str, SegmentKind, int, list[BoundaryKind]]] = []
        current = micro_chunks[0]
        merged_from = 1
        boundaries: list[BoundaryKind] = []

        for nxt in micro_chunks[1:]:
            combined = f"{current} {nxt}"

            # Guardrail first: an over-long chunk is split regardless of how
            # coherent it is.
            if len(combined.split()) > self.config.max_words:
                final.append(
                    (current, SegmentKind.PROSE, merged_from, [*boundaries, BoundaryKind.GUARDRAIL])
                )
                current, merged_from, boundaries = nxt, 1, []
                continue

            if self.scorer.score(current, nxt) >= self.config.similarity_threshold:
                current = combined
                merged_from += 1
                boundaries.append(self.config.level)
            else:
                final.append(
                    (current, SegmentKind.PROSE, merged_from, [*boundaries, BoundaryKind.COHERENCE])
                )
                current, merged_from, boundaries = nxt, 1, []

        if current:
            final.append((current, SegmentKind.PROSE, merged_from, boundaries))
        return final

    # --- assembly ---------------------------------------------------------

    def _build_chunk(
        self,
        index: int,
        text: str,
        kind: SegmentKind,
        merged_from: int,
        boundaries: list[BoundaryKind],
        source_document: str | None,
        source_format: SourceFormat | None,
    ) -> SemanticChunk:
        return SemanticChunk(
            text=text,
            chunk_id=_chunk_id(source_document, index, text),
            index=index,
            word_count=len(text.split()),
            source_document=source_document,
            source_format=source_format,
            segment_kind=kind,
            quality=quality_score(text),
            boundaries=boundaries,
            merged_from=merged_from,
        )


def _chunk_id(source_document: str | None, index: int, text: str) -> str:
    """A stable id: same input, same id, so re-ingest does not churn a vector store."""
    digest = hashlib.sha256(
        f"{source_document or ''}|{index}|{text}".encode("utf-8")
    ).hexdigest()
    return f"chunk-{index:05d}-{digest[:12]}"


# --- module-level convenience ---------------------------------------------


def chunk(
    result: ExtractionResult,
    *,
    config: ChunkConfig | None = None,
    scorer: CoherenceScorer | None = None,
    source_document: str | None = None,
) -> ChunkedDocument:
    """Chunk an :class:`ExtractionResult`. The Brain's main entry point."""
    with HybridChunker(config=config, scorer=scorer) as chunker:
        return chunker.chunk(result, source_document=source_document)


def chunk_text(
    text: str,
    *,
    source_format: SourceFormat | None = None,
    config: ChunkConfig | None = None,
    scorer: CoherenceScorer | None = None,
    source_document: str | None = None,
) -> ChunkedDocument:
    """Chunk raw text when there is no :class:`ExtractionResult` to hand."""
    with HybridChunker(config=config, scorer=scorer) as chunker:
        return chunker.chunk_text(
            text, source_format=source_format, source_document=source_document
        )
