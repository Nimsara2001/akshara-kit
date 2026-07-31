# akshara-kit

A hybrid intelligent framework for ingesting Sinhala-language documents into clean,
Unicode-normalised text suitable for downstream semantic chunking and Retrieval-Augmented
Generation.

This repository currently implements the **Eye module**: file-type detection, text
extraction, legacy-font detection, legacy-to-Unicode conversion and OCR routing. The Brain
module (semantic chunking) consumes this module's output and is out of scope here.

```python
from akshara_kit import route

result = route("textbook.pdf")

print(result.text)                    # Unicode Sinhala
print(result.backend_id)              # which extractor won
print(result.quality.sinhala_ratio)   # 0.0-1.0
print(result.detected_legacy_fonts)   # e.g. ["FMAbhaya"]
print(result.ocr_used)                # was any page OCR'd?
```

`route()` accepts PDF, DOCX and XLSX. The format is decided by the file's structure, not its
extension — a `.pdf` file that is really a zip is handled as a zip.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra all --group dev
```

Extras can be installed selectively if you only handle some formats: `pdf`, `docx`, `xlsx`,
`ocr`, `sinhala`. Importing the package never requires extras you are not using.

### System dependency: Tesseract (OCR only)

OCR is only reached for pages that have no text layer. If you never ingest scanned PDFs, you
can skip this entirely — everything else works without it.

1. Install **Tesseract OCR** (`https://github.com/UB-Mannheim/tesseract/wiki` on Windows,
   `apt install tesseract-ocr` on Debian/Ubuntu, `brew install tesseract` on macOS).
2. Install the **Sinhala language pack**. Download
   [`sin.traineddata`](https://github.com/tesseract-ocr/tessdata/raw/main/sin.traineddata)
   into your `tessdata` directory — on Windows that is normally
   `C:\Program Files\Tesseract-OCR\tessdata\`.
3. Verify:

```bash
tesseract --list-langs
```

`sin` must appear in the output. If Tesseract is installed somewhere unusual, point the
library at it with `AKSHARA_TESSERACT_CMD=/full/path/to/tesseract`.

**poppler is not required.** Pages are rasterised with PyMuPDF, which is already a
dependency of the `pdf` extra. If you specifically want the `pdf2image` path instead, set
`AKSHARA_RASTERISER=pdf2image` (and `AKSHARA_POPPLER_PATH` if poppler is not on `PATH`).

When OCR is unavailable, documents still extract — pages with no text layer simply come back
empty, and a warning explains exactly what to install.

## How it works

| Format | Path |
|---|---|
| PDF | All four text-stream backends (pypdf, PyMuPDF, pdfplumber, pdfminer.six) run in parallel; each output is scored and the best is kept. Pages with no text layer are OCR'd individually and merged back in original page order. |
| DOCX | python-docx, walking the document body so tables stay in reading order. Fonts resolved per run. |
| XLSX | openpyxl, every non-empty cell of every sheet. Fonts resolved per cell. |

### Legacy Sinhala conversion

Many Sinhala documents predate Unicode and store text as glyph indices in fonts such as FM
Abhaya. Converting these is only safe when applied to *exactly* the text drawn in a legacy
font — applied to a whole document it destroys ASCII and any already-correct Unicode.

akshara-kit therefore converts **per run**: PDF spans, DOCX runs and XLSX cells are each
converted based on their own font. A URL in Helvetica sitting beside an FM Abhaya heading
keeps its characters; an Iskoola Pota paragraph beside the same heading keeps its Unicode.

Fonts fall into three groups, and the distinction matters:

- **Convertible** — the FM family. Every entry was verified by converting real text from
  real documents and checking the output; see `tests/test_font_detection.py`.
- **Legacy but not convertible** — `Chamodi`, `sandaru-n`, `K-Plain`, the Tamil
  `SHREE-TAM*` families. These are detected and reported in
  `ExtractionResult.unmapped_legacy_fonts`, and their text passes through **unchanged**
  rather than being silently corrupted.
- **Already Unicode** — Iskoola Pota, Nirmala UI, Latha and friends. Never converted.

### Known limitation

As of this implementation, [`pandukabhaya`](https://github.com/akuruAI/Pandukabhaya)
supports conversion for the FM Abhaya legacy font only, though it is built with an
extensible JSON-mapping design that is expected to support additional fonts (e.g. DL Manel,
FM Bindumathi) in future releases. `KNOWN_LEGACY_FONT_NAMES` and the conversion routing
logic in `eye/encoding_normaliser.py` are intentionally structured so that adding a new font
mapping requires only a new entry in that set plus a corresponding pandukabhaya mapping
table — no change to the detection or routing control flow.

## Development

```bash
uv run pytest --cov=akshara_kit --cov-report=term-missing
```

Tests are hermetic: fixtures are committed, and no network or document corpus is needed.
Tests requiring Tesseract are marked `ocr` and skip with an actionable message when the
Sinhala pack is absent. Regenerate the fixtures with:

```bash
uv run python tests/fixtures/generate_fixtures.py
```

### Not yet implemented

`multimodal/fallback.py` (vision-language model fallback) and `layout/analyser.py` (U-Net
layout analysis) are stubs that raise `NotImplementedError`. Both are future work, and their
signatures already match the adapter contract so they can be registered alongside the local
extractors when built.
