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

print(result.text)                        # Unicode Sinhala
print(result.backend_id)                  # which extractor won
print(result.quality.sinhala_ratio)       # 0.0-1.0, how much Sinhala
print(result.quality.orphan_vowel_rate)   # ~0.0 when the Sinhala is well-formed
print(result.detected_legacy_fonts)       # e.g. ["FMAbhaya"]
print(result.ocr_used)                    # was any page OCR'd?
print(result.pages_ocred)                 # which pages, in document order
```

`route()` accepts PDF, DOCX and XLSX. The format is decided by the file's structure, not its
extension — a `.pdf` file that is really a zip is handled as a zip.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra all --group dev
```

Extras can be installed selectively if you only handle some formats: `pdf`, `docx`, `xlsx`,
`ocr`, `sinhala`, `multimodal`. Importing the package never requires extras you are not
using.

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
| PDF | All four text-stream backends (pypdf, PyMuPDF, pdfplumber, pdfminer.six) run in parallel; each output is scored and the best is kept. Pages with no text layer — or a garbled one — are OCR'd individually and merged back in original page order. |
| DOCX | python-docx, walking the document body so tables stay in reading order. Fonts resolved per run. |
| XLSX | openpyxl, every non-empty cell of every sheet. Fonts resolved per cell. |

### Broken text layers

A surprising number of Sinhala PDFs carry a broken `ToUnicode` cmap: the page
*renders* correctly, but the text underneath maps to the wrong code points. Every
text-stream extractor then returns the same confident nonsense — `පොලී` comes out
`පපොලී`, `යටතේ` comes out `යටපේ` — and because it is still entirely Sinhala
characters, a character-range score cannot tell that anything is wrong.

akshara-kit measures orthographic well-formedness alongside the Sinhala ratio. A
dependent vowel sign must attach to a consonant, so vowel signs left stranded after
a space are counted; correct text scores ~0.000, a garbled text layer scores in the
percent range. Pages that fail this check are rasterised and OCR'd, which recovers
the text the page actually displays, and merged back in document order.

This costs roughly 1.5s per affected page. To turn it off and take the text layer
as-is:

```python
from akshara_kit.eye import pdf_coordinator

result = pdf_coordinator.extract("textbook.pdf", repair_malformed=False)
```

OCR must be available for the repair to happen. When it is not, the document still
extracts and a warning names the affected pages.

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

## Multimodal fallback (opt-in)

Some pages defeat everything local: the text layer is a broken cmap *and* the
rendering is too degraded or decorative for Tesseract. A vision-language model reads
what the page displays, so akshara-kit can send those pages to Gemini, OpenAI or
Claude — but only if you ask it to.

**An API key in your environment is not permission.** Keys get set for all kinds of
unrelated reasons, and this library will not start uploading your documents because
it found one. Nothing leaves your machine unless you pass a `MultimodalConfig`:

```python
from akshara_kit import route, MultimodalConfig

result = route("textbook.pdf", multimodal=MultimodalConfig(provider="gemini"))

print(result.pages_multimodal)                  # which pages were sent
print(result.multimodal_provider)               # and to whom
print(result.metadata["multimodal_model"])      # and which model read them
```

`provider` is required — there is no default and no priority order. With two keys
configured, picking one for you would be a guess about where your documents should
go.

### It is a last resort, and only pays for what OCR could not fix

A page is sent only when it was already a repair candidate **and** is still unusable
after OCR ran — empty, or still failing the well-formedness check. A page Tesseract
fixed never reaches a paid API. On the bundled `sample_mixed.pdf`, OCR repairs both
pages, so enabling the fallback costs nothing at all.

### Models

Each provider has a default you can override per config or per call:

| Provider | Default model | Alternatives |
|---|---|---|
| `gemini` | `gemini-3.6-flash` | `gemini-3.5-flash-lite`, `gemini-2.5-pro` |
| `openai` | `gpt-5.6` | `gpt-5.6-terra`, `gpt-5.6-luna` |
| `claude` | `claude-opus-5` | `claude-sonnet-5`, `claude-haiku-4-5` |

```python
MultimodalConfig(provider="openai", model="gpt-5.6-terra")
```

Model strings are passed straight through, never checked against a list — a model
released after this library still works without waiting for a release of it.

### Keys, budget, and calling it directly

```bash
uv sync --extra multimodal
export AKSHARA_GEMINI_API_KEY=...   # or GEMINI_API_KEY / GOOGLE_API_KEY
```

`AKSHARA_OPENAI_API_KEY` / `OPENAI_API_KEY` and `AKSHARA_ANTHROPIC_API_KEY` /
`ANTHROPIC_API_KEY` work the same way.

`max_pages` defaults to 20 and is checked **before** the first request, so an
over-budget document costs nothing — it raises `MultimodalBudgetExceededError`
rather than silently transcribing the first 20 pages and presenting the result as
complete.

To bypass the escalation logic entirely and transcribe directly:

```python
from akshara_kit.multimodal import fallback

text = fallback.extract_page("scan.pdf", 3, config=MultimodalConfig(provider="claude"))
result = fallback.extract("scan.pdf", config=MultimodalConfig(provider="gemini", max_pages=40))
```

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
Sinhala pack is absent.

The suite is slow when OCR is available, because the fixtures include real documents
with broken text layers and repairing them means rasterising and re-reading a few
hundred pages. Integration tests share one cached extraction per fixture; skip the
OCR legs entirely with `-m "not ocr"` for a fast inner loop.

Tests that call a real vision API are marked `vlm` and skip unless a key is configured,
so a clean clone never bills anyone by surprise. Every other multimodal test stubs its
provider and needs no key, no SDK and no network.

### Not yet implemented

`layout/analyser.py` (U-Net layout analysis) is a stub that raises `NotImplementedError`.
It is future work, and its signature already matches the adapter contract so it can be
registered alongside the local extractors when built.
