"""The Brain: neuro-symbolic semantic chunking (report Sections 4.6 and 6.5).

Consumes the Eye's :class:`~akshara_kit.contracts.extraction.ExtractionResult`
and produces :class:`~akshara_kit.contracts.chunking.SemanticChunk` through three
stages: a Prolog rule base finds linguistically defensible boundaries, LaBSE
supplies a topic-coherence signal, and a bounded agglomerative merge combines
them under a length guardrail.

    from akshara_kit import route, chunk

    doc = chunk(route("textbook.pdf"))
    doc.texts          # plain strings
    doc[2:5]           # a range of chunks
    doc.to_json("chunks.json")
"""

from akshara_kit.brain.coordinator import HybridChunker, chunk, chunk_text
from akshara_kit.brain.encoder import CoherenceScorer, LabseScorer
from akshara_kit.brain.rule_engine import SymbolicRuleEngine
from akshara_kit.brain.segmenter import Segment, segment

__all__ = [
    "CoherenceScorer",
    "HybridChunker",
    "LabseScorer",
    "Segment",
    "SymbolicRuleEngine",
    "chunk",
    "chunk_text",
    "segment",
]
