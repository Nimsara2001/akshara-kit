"""Claude vision transcription via the Anthropic Messages API.

Three details here are load-bearing and easy to get wrong:

**Thinking stays on.** Claude Opus 5 thinks by default, and it is left that way
deliberately. With ``thinking={"type": "disabled"}`` the model can leak
``<thinking>`` tags into its visible response — harmless in a chat UI, but here
the visible response *is* the extracted text, so a leaked tag lands in the
corpus. ``effort: "low"`` keeps the cost of thinking down instead.

**Refusals are not errors.** A declined request returns HTTP 200 with
``stop_reason == "refusal"`` and empty or partial content. Reading
``content[0]`` first would raise ``IndexError`` and report a transcription
failure as a crash, so the stop reason is checked before the content.

**The image goes first.** Anthropic's guidance is to place the image block
before the text block; the prompt reads as being about the image that precedes
it.
"""

from __future__ import annotations

from akshara_kit.eye.errors import (
    AdapterUnavailableError,
    ExtractionFailedError,
    MultimodalUnavailableError,
)
from akshara_kit.multimodal.prompts import TRANSCRIPTION_PROMPT
from akshara_kit.multimodal.providers.base import png_to_base64

__all__ = ["BACKEND_ID", "DEFAULT_MODEL", "transcribe"]

BACKEND_ID = "claude"

#: The current Opus tier. This path only runs on pages that defeated both the
#: text layer and OCR, so capability is worth more here than the per-page
#: saving of a smaller model. Override with ``MultimodalConfig.model``.
DEFAULT_MODEL = "claude-opus-5"

#: Enough for a dense page; low enough to stay under the SDK's HTTP timeout on
#: a non-streaming request.
MAX_TOKENS = 16000

_PROVIDER = "claude"


def _client():
    """Build an Anthropic client, or explain what is missing."""
    from akshara_kit.eye import capabilities

    api_key = capabilities.resolve_api_key(_PROVIDER)
    if api_key is None:
        raise MultimodalUnavailableError(
            capabilities.describe_multimodal_availability(_PROVIDER)
        )

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "anthropic is not installed; install the 'multimodal' extra"
        ) from exc

    return anthropic.Anthropic(api_key=api_key)


def transcribe(png: bytes, *, model: str | None = None) -> str:
    """Transcribe one rendered page with Claude."""
    client = _client()
    try:
        response = client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": png_to_base64(png),
                            },
                        },
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - never leak a raw SDK exception
        raise ExtractionFailedError(f"Claude transcription failed: {exc}") from exc

    return _text_of(response)


def _text_of(response) -> str:
    """Join the response's text blocks, refusing to misread a refusal."""
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise ExtractionFailedError(
            "Claude declined to transcribe this page"
            + (f" (category: {category})" if category else "")
        )

    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
