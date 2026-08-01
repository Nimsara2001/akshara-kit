"""Preprocessing: extracted text in, chunkable prose out.

The Eye transcribes a page faithfully, which is not the same as producing
readable prose — hard line wraps, page numbers, running headers and stray
zero-width characters all survive extraction. This package removes them before
the Brain sees the text.
"""

from akshara_kit.preprocess.pipeline import CleanedText, PreprocessConfig, clean

__all__ = ["CleanedText", "PreprocessConfig", "clean"]
