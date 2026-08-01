"""Gemini vision transcription via the google-genai SDK.

Uses ``client.models.generate_content`` with an inline image part. The image is
passed as raw bytes with an explicit mime type rather than a file upload: pages
are transient here — each one is transcribed once and never referenced again —
so an upload would add a round trip and leave server-side state to clean up.
"""

from __future__ import annotations

from akshara_kit.eye.errors import (
    AdapterUnavailableError,
    ExtractionFailedError,
    MultimodalUnavailableError,
)
from akshara_kit.multimodal.prompts import TRANSCRIPTION_PROMPT

__all__ = ["BACKEND_ID", "DEFAULT_MODEL", "transcribe"]

BACKEND_ID = "gemini"

#: Generally available, strong on multimodal work, and cheaper than the wider
#: Flash line — the best cost/accuracy point of the three defaults for
#: page-at-a-time transcription. Override with ``MultimodalConfig.model``.
DEFAULT_MODEL = "gemini-3.6-flash"

_PROVIDER = "gemini"


def _client_and_types():
    """Build a Gemini client, or explain what is missing."""
    from akshara_kit.eye import capabilities

    api_key = capabilities.resolve_api_key(_PROVIDER)
    if api_key is None:
        raise MultimodalUnavailableError(
            capabilities.describe_multimodal_availability(_PROVIDER)
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "google-genai is not installed; install the 'multimodal' extra"
        ) from exc

    return genai.Client(api_key=api_key), types


def transcribe(png: bytes, *, model: str | None = None) -> str:
    """Transcribe one rendered page with Gemini."""
    client, types = _client_and_types()
    try:
        response = client.models.generate_content(
            model=model or DEFAULT_MODEL,
            contents=[
                types.Part.from_bytes(data=png, mime_type="image/png"),
                TRANSCRIPTION_PROMPT,
            ],
        )
    except Exception as exc:  # noqa: BLE001 - never leak a raw SDK exception
        raise ExtractionFailedError(f"Gemini transcription failed: {exc}") from exc

    # `.text` is None when the response carried no text part at all — a page
    # with no legible text is an empty transcription, not a failure.
    return response.text or ""
