"""Multimodal vision-language fallback — STUB.

The highest-cost tool in the planned cost-aware escalation ladder (interim
report Sections 4.5 and 6.7). Out of scope for the current build; the
signature deliberately matches the adapter contract so it can be registered
alongside the local extractors without a shim once implemented.
"""

from __future__ import annotations

from akshara_kit.contracts.extraction import ExtractionResult

_MESSAGE = (
    "The multimodal vision-language fallback (e.g. Gemini) is not implemented. "
    "It is future work: the highest-cost tool in the cost-aware escalation "
    "ladder described in interim report Sections 4.5 and 6.7. The current "
    "build uses the deterministic router and the local extractor set only."
)


def extract(file_path: str) -> ExtractionResult:
    """Raise :class:`NotImplementedError`. See the module docstring."""
    raise NotImplementedError(_MESSAGE)
