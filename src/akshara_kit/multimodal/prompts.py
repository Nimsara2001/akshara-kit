"""The transcription prompt shared by every multimodal provider.

One prompt, not three. The providers differ in SDK shape, not in what they are
being asked to do, and a per-provider prompt would make output differences
impossible to attribute — the evaluation chapter needs to compare providers on
the same instruction.

The instructions exist to counter three specific vision-model habits that would
corrupt an extraction:

1. **Commentary.** "Here is the text from the image:" is not in the document.
   Everything returned is treated as page content, so a preamble becomes a
   corrupt line in the middle of a chunked corpus.
2. **Helpful correction.** A model that silently fixes a typo, expands an
   abbreviation, or translates a heading has produced a *better* document and a
   *wrong* transcription. Ingestion needs what the page says.
3. **Legacy-glyph transliteration.** Legacy Sinhala fonts render correct Sinhala
   on screen; the model sees the rendered glyphs and should report the Sinhala
   it sees, not the Latin-1 bytes an extractor would have found underneath.
"""

from __future__ import annotations

__all__ = ["TRANSCRIPTION_PROMPT"]

TRANSCRIPTION_PROMPT = """\
Transcribe all text visible in this page image, exactly as it appears.

This page is from a Sinhala-language document. It may contain Sinhala, English,
Tamil, numerals, or a mixture.

Rules:
- Output only the transcribed text. No preamble, no explanation, no commentary,
  no markdown fences, and no notes about what you did or could not read.
- Reproduce the text verbatim. Do not correct spelling, expand abbreviations,
  translate, summarise, modernise, or otherwise improve on the original.
- Write Sinhala in Unicode (U+0D80-U+0DFF), whatever font the page uses.
- Preserve the reading order, line breaks, and paragraph breaks you see.
- Transcribe table cells row by row, separating cells with a tab.
- Describe nothing. If a region is an image, a diagram, or a photograph with no
  text, skip it silently rather than writing a description of it.
- If part of the page is genuinely illegible, omit it. Do not guess, and do not
  write a placeholder marking the gap.
- If the page contains no text at all, return nothing.
"""
