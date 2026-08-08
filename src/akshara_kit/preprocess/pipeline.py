"""The preprocessing pipeline: extracted text in, chunkable prose out.

Ordered stages, each individually switchable through :class:`PreprocessConfig`
so the evaluation chapter can ablate them rather than assert their value. Every
stage reports how much it changed, and those counts land on
:class:`CleanedText.stages`.

Order matters. Unicode hygiene runs first so that later stages compare like with
like; noise removal runs before de-wrapping so a page number caught between two
wrapped lines does not get welded into the sentence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from akshara_kit.preprocess.layout_noise import dewrap, drop_noise_lines
from akshara_kit.preprocess.unicode_rules import (
    normalise_unicode,
    strip_control_characters,
    strip_zero_width,
)

__all__ = ["CleanedText", "PreprocessConfig", "clean"]


class PreprocessConfig(BaseModel):
    """Which cleanup stages to run."""

    normalise_unicode: bool = True
    strip_zero_width: bool = True
    drop_noise_lines: bool = True
    dewrap_lines: bool = True
    collapse_whitespace: bool = True


class CleanedText(BaseModel):
    """Cleaned text plus an account of what was changed.

    ``stages`` is what makes preprocessing reportable: it records per stage how
    many characters, lines or joins that stage was responsible for, so its
    contribution can be measured instead of assumed.
    """

    text: str
    original_length: int
    stages: dict[str, int] = Field(default_factory=dict)

    @property
    def removed_characters(self) -> int:
        return self.original_length - len(self.text)


def clean(text: str, *, config: PreprocessConfig | None = None) -> CleanedText:
    """Turn raw extracted text into something worth chunking."""
    config = config or PreprocessConfig()
    stages: dict[str, int] = {}
    original_length = len(text)

    if config.normalise_unicode:
        before = text
        text = normalise_unicode(text)
        stages["nfc_changed"] = int(text != before)

    if config.strip_zero_width:
        text, controls = strip_control_characters(text)
        text, zwj, zwnj = strip_zero_width(text)
        stages["controls_removed"] = controls
        stages["zwj_removed"] = zwj
        stages["zwnj_removed"] = zwnj

    if config.drop_noise_lines:
        text, dropped = drop_noise_lines(text)
        stages["noise_lines_dropped"] = dropped

    if config.dewrap_lines:
        text, joins = dewrap(text)
        stages["lines_joined"] = joins

    if config.collapse_whitespace:
        text, collapsed = _collapse_whitespace(text)
        stages["whitespace_collapsed"] = collapsed

    return CleanedText(text=text, original_length=original_length, stages=stages)


def _collapse_whitespace(text: str) -> tuple[str, int]:
    """Squeeze space runs and blank-line runs.

    Tabs survive: the DOCX and XLSX adapters use them as cell separators, and
    losing them would erase the row structure the segmenter depends on.
    """
    import re

    before = len(text)
    text = re.sub(r"[^\S\t\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), before - len(text.strip())
