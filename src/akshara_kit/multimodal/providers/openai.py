"""OpenAI vision transcription via the Responses API.

Note this is ``responses.create``, not ``chat.completions``: the Responses API
is the current surface, and its image content block is ``input_image`` carrying
a data URI rather than a nested ``image_url`` object.

The ``detail`` parameter is deliberately not sent — GPT-5 models ignore it, and
passing an ignored knob invites someone to later "fix" its value in the belief
that it does something.
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

BACKEND_ID = "openai"

#: The flagship alias. Like the Claude default, capability is preferred here
#: because this path only sees pages nothing cheaper could read. Callers wanting
#: the cheaper tiers can pass ``gpt-5.6-terra`` or ``gpt-5.6-luna``.
DEFAULT_MODEL = "gpt-5.6"

_PROVIDER = "openai"


def _client():
    """Build an OpenAI client, or explain what is missing."""
    from akshara_kit.eye import capabilities

    api_key = capabilities.resolve_api_key(_PROVIDER)
    if api_key is None:
        raise MultimodalUnavailableError(
            capabilities.describe_multimodal_availability(_PROVIDER)
        )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise AdapterUnavailableError(
            "openai is not installed; install the 'multimodal' extra"
        ) from exc

    return OpenAI(api_key=api_key)


def transcribe(png: bytes, *, model: str | None = None) -> str:
    """Transcribe one rendered page with OpenAI."""
    client = _client()
    data_uri = f"data:image/png;base64,{png_to_base64(png)}"
    try:
        response = client.responses.create(
            model=model or DEFAULT_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": data_uri},
                        {"type": "input_text", "text": TRANSCRIPTION_PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - never leak a raw SDK exception
        raise ExtractionFailedError(f"OpenAI transcription failed: {exc}") from exc

    return _text_of(response)


def _text_of(response) -> str:
    """Prefer the SDK's flattened text, falling back to walking the blocks."""
    text = getattr(response, "output_text", None)
    if text is not None:
        return text

    parts: list[str] = []  # pragma: no cover - only on SDKs without output_text
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) == "output_text":
                parts.append(block.text)
    return "".join(parts)
