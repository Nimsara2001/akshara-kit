"""U-Net visual layout analysis — STUB.

The medium-cost tool in the planned escalation ladder (interim report Section
4.5). It would populate ``LayoutRegion`` records (region type, bounding box,
reading-order index, page index) and in turn the ``region_coverage`` field of
:class:`~akshara_kit.contracts.extraction.QualityScore`, which the current
build leaves as ``None``.
"""

from __future__ import annotations

_MESSAGE = (
    "U-Net visual layout analysis is not implemented. It is future work: the "
    "medium-cost tool in the escalation ladder described in interim report "
    "Section 4.5, and the intended source of QualityScore.region_coverage."
)


def analyse(file_path: str) -> list[dict]:
    """Raise :class:`NotImplementedError`. See the module docstring."""
    raise NotImplementedError(_MESSAGE)
