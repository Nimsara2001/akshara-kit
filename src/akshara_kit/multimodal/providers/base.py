"""The contract every multimodal provider satisfies.

Structural typing rather than inheritance, matching
:class:`akshara_kit.eye.ocr_decision.PageLike`: a test can substitute a plain
object with a ``transcribe`` method and never touch a provider SDK, which is
what keeps the multimodal test suite hermetic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Transcriber", "png_to_base64"]


@runtime_checkable
class Transcriber(Protocol):
    """Turns one rendered page image into text.

    ``BACKEND_ID`` names the provider for ``ExtractionResult.backend_id``.
    ``DEFAULT_MODEL`` is the model used when the caller names none; it lives on
    the provider module rather than in a central registry so that adding a
    provider touches exactly one file.
    """

    BACKEND_ID: str
    DEFAULT_MODEL: str

    def transcribe(self, png: bytes, *, model: str | None = None) -> str:
        """Return the page's text, or ``""`` if it carries none.

        :raises AdapterUnavailableError: if the provider's SDK is not installed.
        :raises MultimodalUnavailableError: if no API key is configured.
        :raises ExtractionFailedError: if the provider call failed.
        """
        ...


def png_to_base64(png: bytes) -> str:
    """Base64-encode PNG bytes for an inline image payload."""
    import base64

    return base64.standard_b64encode(png).decode("ascii")
