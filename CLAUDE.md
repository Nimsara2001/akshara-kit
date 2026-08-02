# akshara-kit — Eye Module & Multi-Format Orchestrator

## Implementation Instructions for Claude Code

This file is a build specification. Implement it **phase by phase**, in the order given in
Section 8. Do not skip ahead to a later phase until the current phase's tests pass. After
each phase, run the test suite and report results before proceeding.

---

## 1. Project Context

`akshara-kit` is a Python library for ingesting Sinhala-language documents (PDF, DOCX,
XLSX) into clean, Unicode-normalised text suitable for downstream semantic chunking and
Retrieval-Augmented Generation. This spec covers only the **Eye module**: file-type
detection, text extraction, legacy-font detection, legacy-to-Unicode conversion, and
OCR routing. A separate **Brain module** (semantic chunking) consumes this module's
output later and is **out of scope** for this build — only its input contract matters
here (Section 4).

### Non-goals for this phase (do not implement)

- ~~Multimodal vision-language API fallback~~ — **now implemented**, see Section 14.
  It was deferred as a stub; the broken-cmap fixtures showed pages that defeat both
  the text layer and Tesseract, which is exactly the case a vision model exists for.
- U-Net visual layout analysis — stub only.
- Agentic orchestration controller (cost-aware ReAct-style escalation) — the current
  scope uses a simpler deterministic router (Section 5), not a learned/agentic policy.
- ~~Any Brain-module code (chunking, Prolog rules, sentence embeddings)~~ —
  **now implemented**, see Section 15. The Eye's output had no consumer, and the
  interim report (§4.6, §6.5) specifies the Brain in more detail than this file
  specifies parts of the Eye. Preprocessing came with it: extracted text carries
  PDF layout scars that make chunking meaningless without cleanup.

---

## 2. Environment & Tooling

- Python **3.11+**
- Use **uv** for environment and dependency management, not pip/venv directly.
- Package layout: `src/` layout, not flat.

Initialize with:

```bash
uv init akshara-kit --lib
cd akshara-kit
```

`pyproject.toml` requirements:

```toml
[project]
name = "akshara-kit"
version = "0.1.0"
description = "Hybrid intelligent framework for Sinhala document ingestion"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
]

[project.optional-dependencies]
pdf = ["pypdf", "pdfplumber", "pymupdf", "pdfminer.six"]
ocr = ["pytesseract", "pdf2image"]
docx = ["python-docx"]
xlsx = ["openpyxl"]
sinhala = ["sinlib"]
all = ["akshara-kit[pdf,ocr,docx,xlsx,sinhala]"]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-cov"]
```

Install everything for development:

```bash
uv sync --extra all --group dev
```

System dependency note: `pytesseract` requires the Tesseract OCR binary installed at
the OS level, **with the Sinhala `sin` language pack** — `sin.traineddata` in the
`tessdata` directory. Tesseract without that pack is the common failure mode: the
binary resolves, so the library looks installed, but every Sinhala page silently
returns nothing. `eye/capabilities.py` probes for the pack specifically and
`describe_ocr_availability()` says which of the two is missing.

Rasterisation goes through PyMuPDF, which is already a `pdf` dependency, so
**poppler is not required**. The `pdf2image`/poppler path is retained behind
`AKSHARA_RASTERISER=pdf2image` only. Add a note in `README.md` about installing
these; do not attempt to pip-install them.

---

## 3. Package Layout

Create this exact structure:

```
src/akshara_kit/
├── __init__.py
├── contracts/
│   ├── __init__.py
│   └── extraction.py          # data contracts, Section 4
├── router/
│   ├── __init__.py
│   └── format_router.py       # Section 5
├── adapters/
│   ├── __init__.py
│   └── extractors/
│       ├── __init__.py
│       ├── pypdf_adapter.py
│       ├── pdfplumber_adapter.py
│       ├── pymupdf_adapter.py
│       ├── pdfminer_adapter.py
│       ├── ocr_adapter.py         # pytesseract; rasterises via PyMuPDF
│       ├── docx_adapter.py
│       └── xlsx_adapter.py
├── eye/
│   ├── __init__.py
│   ├── quality_probe.py       # Section 7
│   ├── font_detection.py      # Section 6
│   ├── encoding_normaliser.py # Section 6.4 — pandukabhaya wrapper
│   ├── ocr_decision.py        # Section 8
│   ├── pdf_coordinator.py     # multi-adapter race for PDF only
│   ├── coordinator.py         # top-level Eye Coordinator, dispatches by format
│   ├── capabilities.py        # cached probes for Tesseract / poppler
│   └── errors.py              # the named exception hierarchy
├── multimodal/                # Section 14
│   ├── __init__.py
│   ├── fallback.py            # public surface + provider dispatch
│   ├── prompts.py             # the shared transcription prompt
│   └── providers/
│       ├── __init__.py
│       ├── base.py            # Transcriber Protocol
│       ├── gemini.py
│       ├── openai.py
│       └── claude.py
└── layout/
    ├── __init__.py
    └── analyser.py             # STUB ONLY — raise NotImplementedError

tests/
├── fixtures/                  # sample PDF/DOCX/XLSX files go here — see Section 9
├── conftest.py                # shared fixtures; capability-based skipping
├── samples.py                 # the shared sample strings tests assert on
├── test_format_router.py
├── test_pdf_adapters.py
├── test_docx_adapter.py
├── test_xlsx_adapter.py
├── test_font_detection.py
├── test_encoding_normaliser.py
├── test_quality_probe.py
├── test_ocr_decision.py
├── test_ocr_adapter.py
├── test_multimodal.py         # Section 14; providers stubbed, no network
├── test_preprocess.py         # Section 15.1
├── test_rule_engine.py        # Section 15.2; grammar-derived expectations
├── test_chunk_coordinator.py  # Section 15.3-15.5; scorer stubbed, no torch
├── test_fixtures_current.py   # guards the fixtures against silent drift
└── test_coordinator_integration.py
```

Tests that need Tesseract carry the `ocr` marker and skip with an actionable
message when the Sinhala pack is absent; `conftest.py` applies this. Because
extraction is expensive — four backends per PDF, plus OCR for any page whose text
layer is missing or garbled — the integration tests take their results from the
session-scoped `routed` fixture rather than calling `route()` per test.

---

## 4. Data Contracts

Define these in `contracts/extraction.py` using `pydantic.BaseModel` (not plain
dataclasses — we want field validation since this is a public library boundary).

```python
from enum import Enum
from pydantic import BaseModel


class SourceFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"


class FontDetectionMethod(str, Enum):
    FONT_NAME = "font_name"
    CHARACTER_RATIO = "character_ratio"
    NONE = "none"


class QualityScore(BaseModel):
    raw_length: int
    sinhala_ratio: float
    region_coverage: float | None = None
    orphan_vowel_rate: float = 0.0   # Section 7.1


class ExtractionResult(BaseModel):
    text: str
    backend_id: str
    source_format: SourceFormat
    latency_seconds: float
    quality: QualityScore | None = None
    font_detection_method: FontDetectionMethod = FontDetectionMethod.NONE
    detected_legacy_fonts: list[str] = []
    ocr_used: bool = False
    metadata: dict = {}
    pages_multimodal: list[int] = []        # Section 14
    multimodal_provider: str | None = None  # Section 14


class MultimodalProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


class MultimodalConfig(BaseModel):
    """Presence of this object IS the opt-in to send page images off-machine."""
    provider: MultimodalProvider   # required — no default, no implicit ordering
    model: str | None = None       # None -> the provider module's DEFAULT_MODEL
    max_pages: int = 20
    dpi: int = 200
```

Do not change field names once other modules depend on them — this contract is the
seam between the Eye module and the future Brain module.

---

## 5. Format Router

File: `router/format_router.py`

Responsibilities:

1. Accept a file path.
2. Determine `SourceFormat` using **extension first, then structural validation**
   (never trust extension alone):
   - `.pdf` → verify file starts with bytes `%PDF`
   - `.docx` → verify `zipfile.is_zipfile(path)` is True AND the zip contains
     `word/document.xml`
   - `.xlsx` → verify `zipfile.is_zipfile(path)` is True AND the zip contains
     `xl/workbook.xml`
   - If extension and structural check disagree, trust the structural check and
     log a warning.
   - If neither matches a known format, raise a clear `UnsupportedFormatError`
     (define this exception in the same module).
3. Dispatch to the appropriate coordinator:
   - PDF → `eye/pdf_coordinator.py`
   - DOCX → `adapters/extractors/docx_adapter.py` directly (no race needed)
   - XLSX → `adapters/extractors/xlsx_adapter.py` directly (no race needed)
4. Return a single `ExtractionResult`.

Public function signature:

```python
def route(file_path: str) -> ExtractionResult:
    ...
```

---

## 6. Font Detection & Legacy Encoding Normalisation

### 6.1 Known legacy font names

Create a constant list/set in `eye/font_detection.py`:

```python
KNOWN_LEGACY_FONT_NAMES = {
    "FMAbhaya", "FM Abhaya", "FMAbhaya-Regular",
    # extend this list as more legacy fonts are identified in real documents;
    # do NOT guess mappings for fonts not in this list — flag as UNKNOWN instead
}
```

### 6.2 PDF font detection

Use **PyMuPDF** (`fitz`) specifically for this, even though other adapters may win
the extraction race — PyMuPDF is the only one of the four PDF backends that exposes
the embedded font's PostScript name per text span.

```python
import fitz

def detect_pdf_fonts(file_path: str) -> set[str]:
    doc = fitz.open(file_path)
    fonts_found = set()
    for page in doc:
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    fonts_found.add(span.get("font", ""))
    return fonts_found
```

### 6.3 DOCX and XLSX font detection

- DOCX: iterate every `run` in every paragraph **and every table cell's paragraphs**
  (do not skip tables). Check `run.font.name` per run — detection must be per-run,
  not per-paragraph, since a single paragraph can mix legacy and Unicode runs.
- XLSX: iterate every non-empty cell in every worksheet, check `cell.font.name`.

### 6.4 Legacy-to-Unicode normalisation

File: `eye/encoding_normaliser.py`

```python
def normalise(text: str, detected_fonts: set[str]) -> tuple[str, FontDetectionMethod, list[str]]:
    """
    Returns (normalised_text, detection_method_used, legacy_fonts_actually_converted).
    """
```

Logic:

1. If `detected_fonts` intersects `KNOWN_LEGACY_FONT_NAMES` → route matched text
   through `pandukabhaya` (FM Abhaya mapping only, as of this writing — see Section
   11 limitations). Method = `FONT_NAME`.
2. If no known legacy font detected but the **Sinhala Unicode character ratio**
   (U+0D80–U+0DFF range) is unexpectedly low for a document expected to be Sinhala
   (below a configurable threshold, default 0.15), still attempt `pandukabhaya`
   conversion as a fallback heuristic. Method = `CHARACTER_RATIO`.
3. Otherwise, pass text through unchanged. Method = `NONE`.
4. Always return which legacy fonts were detected/converted so this surfaces in
   `ExtractionResult.detected_legacy_fonts` for evaluation reporting later.

Do **not** attempt to guess conversions for fonts outside `KNOWN_LEGACY_FONT_NAMES`
that pandukabhaya cannot map — pass such text through unchanged and log a clear
warning naming the unmapped font, rather than silently producing corrupted text.

---

## 7. Quality Probe

File: `eye/quality_probe.py`

```python
import re

SINHALA_RANGE = re.compile(r'[\u0D80-\u0DFF]')

def score(text: str) -> QualityScore:
    raw_length = len(text)
    sinhala_chars = len(SINHALA_RANGE.findall(text))
    sinhala_ratio = sinhala_chars / max(raw_length, 1)
    return QualityScore(raw_length=raw_length, sinhala_ratio=sinhala_ratio)
```

Used by the PDF coordinator to pick the best-scoring adapter output. DOCX/XLSX paths
still compute this score (store it on the result) but do not use it to choose between
adapters, since there is only one deterministic path for those formats.

### 7.1 Orthographic well-formedness (second signal)

`sinhala_ratio` alone cannot detect a broken `ToUnicode` cmap: the extracted text
is entirely Sinhala code points, just the *wrong* ones, so it scores well while
reading as nonsense. Add a second measure, `orphan_vowel_rate`, on `QualityScore`.

It relies on the strongest invariant Sinhala offers: a **dependent** vowel sign is
by definition attached to a consonant, and Unicode always stores the consonant
first. A vowel sign preceded by a space, digit or punctuation therefore has nothing
to depend on and cannot be legitimate. The rate is orphaned signs \u00F7 consonants.

Two guards keep it honest:

- Text with fewer than 20 Sinhala consonants is never judged \u2014 below that the rate
  is noise and one stray sign in a caption would condemn a page.
- An **unconverted legacy glyph stream scores 0.0**, because it is Latin-1 bytes
  and carries no Sinhala vowel signs to orphan. This measures *malformed* Sinhala,
  not *absent* Sinhala. Without this property every legacy page would be sent to
  OCR before conversion ever got a chance to fix it.

Measured across the fixture corpus the two populations are an order of magnitude
apart \u2014 correct extractions land at 0.0000\u20130.0023, garbled text layers at
0.0297\u20130.0916 \u2014 so the threshold sits at 0.01.

Ordering in `compare()` is: Sinhala ratio, then well-formedness (lower is better),
then raw length. Well-formedness sits above length so more, wronger text cannot
beat less, correcter text; it sits below ratio because an empty extraction is
trivially well-formed.

---

## 8. OCR Decision (PDF only)

File: `eye/ocr_decision.py`

Per-page, not per-document:

```python
def needs_ocr(page, extracted_text: str, min_chars: int = 20) -> bool:
    """
    Returns True if this page should be routed through OCR instead of / in
    addition to text-stream extraction.
    """
```

Logic: a page needs OCR if **either** of two independent conditions holds.

1. **Blank scan.** Its extracted text length is below `min_chars` AND
   `page.get_image_info()` (PyMuPDF) shows an image covering a large fraction
   of the page area. Note this is `get_image_info()`, not the `get_images()`
   named in earlier drafts: `get_images()` returns the image's *pixel*
   dimensions with no page geometry, so a 4000×3000 source placed in a 50pt
   logo box reads as enormous. Only `get_image_info()` gives a page-space bbox.

2. **Garbled text layer.** The page's text is present but orthographically
   impossible Sinhala — see the well-formedness probe in Section 7. Sinhala
   PDFs are routinely produced with a broken `ToUnicode` cmap, and every
   text-stream backend then returns the same confident nonsense at a perfectly
   healthy Sinhala ratio (`පොලී` → `පපොලී`, `යටතේ` → `යටපේ`). This arm carries
   **no** image requirement, because a broken cmap is invisible to the
   geometry test. Rasterising and OCRing the page recovers the real text.

Condition 2 is a deliberate extension beyond this spec's original letter, added
because the fixture corpus showed 160 of 180 pages of `sample_unicode.pdf`
affected. It is controlled by `repair_malformed=True` on
`pdf_coordinator.extract`; set it to `False` to restore condition 1 alone.

Build this to operate per-page so a mixed document (some scanned pages, some
born-digital) is handled correctly — do not make this an all-or-nothing
decision at the document level.

When OCR is triggered for a page, use the `ocr_adapter.py` (pytesseract, with
the Sinhala `sin` language pack) for that page only, and merge its output with
the text-stream results from other pages in original page order. The pages
merged in must be the **normalised** per-page texts whenever legacy conversion
applied — merging raw pages instead silently discards the conversion for any
document needing both conversion and OCR.

---

## 9. Test Fixtures & Testing Requirements

Create `tests/fixtures/` with:

- `sample_unicode.pdf` — a small PDF with born-digital Sinhala Unicode text
- `sample_legacy_font.pdf` — a small PDF using a known legacy font (FM Abhaya) if
  available; if you cannot source one, create a synthetic test using mocked
  `fitz` span data instead, and note this clearly in the test file
- `sample_scanned.pdf` — a rasterised/image-only page to exercise the OCR path
- `sample_mixed.pdf` — one document combining a legacy-font region and a
  Unicode region, so per-page and per-span routing are exercised together
- `sample.docx` — a Word doc with at least one paragraph and one table, mixing
  a legacy-font run and a Unicode run in the same paragraph
- `sample.xlsx` — a workbook with at least two sheets and some legacy-font cells

Fixtures are drawn from real documents, so their properties are facts to be
measured, not assumed. `test_fixtures_current.py` pins what each one actually
contains; when a fixture is regenerated, re-measure before updating any test
that asserts a page index or an OCR decision.

Write tests for:

1. **Format Router**: correct detection for all three formats, correct rejection
   of an unsupported format, correct handling of a mislabeled extension (e.g. a
   `.pdf`-named file that is actually a zip).
2. **Each PDF adapter**: extracts non-empty text from `sample_unicode.pdf`.
3. **Font detection**: correctly identifies the legacy font in
   `sample_legacy_font.pdf` / the mocked span data, and correctly identifies
   `sample_unicode.pdf` as needing no conversion.
4. **Encoding normaliser**: converts known legacy text correctly, and — critically —
   passes unknown-font text through unchanged rather than corrupting it.
5. **DOCX adapter**: correctly extracts run-level text including tables, and
   correctly separates the legacy-font run from the Unicode run within the same
   paragraph.
6. **XLSX adapter**: correctly extracts cell values across multiple sheets.
7. **OCR decision**: correctly flags `sample_scanned.pdf`'s page as needing OCR
   and correctly does *not* flag a born-digital page.
8. **Integration test**: full `route()` call against each of the six fixture
   files, asserting a valid `ExtractionResult` with the expected `source_format`,
   plausible `quality.sinhala_ratio`, and correct `ocr_used` flag.
9. **Well-formedness (Section 7.1)**: garbled text scores above the threshold,
   correct text scores ~0, and an unconverted legacy stream scores 0 rather
   than being mistaken for corruption.
10. **Malformed-page repair (Section 8, condition 2)**: a page carrying garbled
    text and no image is routed to OCR, and is not routed when
    `repair_malformed=False`.

Run with `uv run pytest --cov=akshara_kit` after each phase.

---

## 10. Implementation Phases (build in this order)

**Phase 1 — Scaffolding**
Create the package layout (Section 3), `pyproject.toml`, contracts (Section 4).
No extraction logic yet. Verify: package imports cleanly, `pytest` collects
zero tests without error.

**Phase 2 — Format Router**
Implement Section 5 with structural validation. Verify: `test_format_router.py`
passes for all three formats plus the mislabeled-extension case.

**Phase 3 — XLSX path (simplest, do this before PDF/DOCX)**
Implement `xlsx_adapter.py`, wire through the router with no font/OCR logic yet
(just raw extraction returning a valid `ExtractionResult` with
`font_detection_method=NONE`). Verify: integration test passes for
`sample.xlsx`.

**Phase 4 — DOCX path**
Implement `docx_adapter.py`, same as Phase 3 but for DOCX, including table
iteration. Verify: integration test passes for `sample.docx`.

**Phase 5 — Font detection & normalisation (shared)**
Implement Section 6 in full: `font_detection.py` and `encoding_normaliser.py`.
Wire into both the XLSX and DOCX paths first (simpler containers), verify
correct per-run/per-cell detection, then wire into a stub PDF path.

**Phase 6 — PDF adapters (all four) + PDF Coordinator**
Implement all four PDF adapters and `pdf_coordinator.py`'s race-and-select
logic using the quality probe (Section 7). Verify against `sample_unicode.pdf`
and `sample_legacy_font.pdf`.

**Phase 7 — OCR decision & OCR adapter**
Implement Section 8. Verify against `sample_scanned.pdf`, including the
mixed-page case if you can construct one.

**Phase 8 — Full integration + stubs**
Wire everything into `eye/coordinator.py` as the single public entry point.
Add the `multimodal/fallback.py` and `layout/analyser.py` stubs (raise
`NotImplementedError` with a message pointing to future work). Run the full
test suite and report coverage.

Do not proceed to a phase until the previous phase's tests are green. If a
fixture file cannot be sourced or created, use a mocked/synthetic substitute
and note the limitation in the test file's docstring rather than skipping the
test entirely.

---

## 11. Known Limitations to Document in Code

Add a module-level docstring note in `encoding_normaliser.py`:

> As of this implementation, `pandukabhaya` supports conversion for the FM Abhaya
> legacy font only, though it is built with an extensible JSON-mapping design that
> is expected to support additional fonts (e.g. DL Manel, FM Bindumathi) in future
> releases. `KNOWN_LEGACY_FONT_NAMES` and the conversion routing logic here are
> intentionally structured so that adding a new font mapping requires only a new
> entry in that set plus a corresponding pandukabhaya mapping table — no change to
> the detection or routing control flow.

---

## 12. Coding Conventions

- Full type hints on every public function.
- Docstrings on every public function/class citing which report section it
  realises (e.g. `"""Implements Algorithm 2 (Sinhala-Aware Quality Probe)."""`)
  so the code stays traceable back to the thesis document.
- Every adapter module exposes exactly one public function:
  `extract(file_path: str) -> ExtractionResult`. No adapter should import
  another adapter — cross-format logic belongs in the router/coordinator layer
  only.
- Raise specific, named exceptions (`UnsupportedFormatError`,
  `UnmappedLegacyFontError` as a warning-level log, not a raised error — see
  Section 6.4) rather than letting library internals leak (e.g. don't let a
  raw `fitz` exception propagate uncaught).
- Keep functions under ~40 lines; if a function is doing "detect + convert +
  log", split it into three.

---

## 13. Definition of Done for This Spec

- `uv sync --extra all --group dev` succeeds from a clean clone.
- `uv run pytest --cov=akshara_kit` passes with all tests green.
- `route()` produces a valid `ExtractionResult` for all six fixture files.
- Calling the layout-analyser stub raises `NotImplementedError` with a clear
  message rather than failing silently or crashing ungracefully.
- `route(path)` with no `multimodal` argument never contacts an external
  service, whatever API keys happen to be set in the environment.
- README.md documents: install instructions (including the Tesseract/poppler
  system dependencies), a minimal usage example calling `route()`, the
  Section 11 limitation note, and the Section 14 opt-in and cost model.

---

## 14. Multimodal Vision-Language Fallback

Files: `multimodal/fallback.py`, `multimodal/prompts.py`, `multimodal/providers/*`

The highest-cost rung of the escalation ladder and the only one that leaves the
machine. It exists for the failure the local stack cannot reach: a page whose text
layer is a broken `ToUnicode` cmap *and* whose rendering also defeats Tesseract,
or a scan on a machine with no OCR installed.

### 14.1 Consent is the design constraint

**An API key in the environment is not permission.** Keys get exported for all sorts
of unrelated reasons, and a document-ingestion library that began uploading a user's
documents because it found one would be doing something the user never asked for.

Nothing contacts a provider unless the caller passes a `MultimodalConfig`. That one
object carries the opt-in, the provider and the budget together so none can be
forgotten separately, and `provider` has no default — with two keys configured,
"use the first" is a guess about where someone's documents should be sent, and this
library does not make that guess.

This property is asserted directly:
`test_multimodal.py::test_route_without_config_never_contacts_a_provider` booby-traps
every route into a provider rather than checking a return value.

### 14.2 Last resort, not an alternative

`_run_multimodal` in `pdf_coordinator.py` runs **after** the OCR stage, over the same
page list. A page escalates only if it was a repair candidate *and* still reads as
unusable afterwards — empty, or malformed by Section 7.1. That one test covers all
three ways OCR falls short (unavailable, raised, or returned text as garbled as the
text layer), and guarantees a page OCR fixed never reaches a paid API.

### 14.3 Providers and models

One module per provider, each owning its own `DEFAULT_MODEL` so adding a provider
touches one file. Modules never import each other; dispatch lives in `fallback.py`.

| Provider | `DEFAULT_MODEL` | SDK | Notes |
|---|---|---|---|
| `gemini` | `gemini-3.6-flash` | `google-genai` | Inline image part |
| `openai` | `gpt-5.6` | `openai` | **Responses API**, `input_image` with a data URI |
| `claude` | `claude-opus-5` | `anthropic` | Image block before text; `effort: "low"` |

Model resolution is per-call override → `config.model` → `DEFAULT_MODEL`, done once in
`fallback.resolve_model` so the model that actually ran is recordable on the result.
Model strings are **never** validated against an allow-list: a model released after
this library must not require a release of this library.

Two Claude-specific rules, both load-bearing:

- **Do not disable thinking.** With `thinking={"type": "disabled"}`, Opus 5 can leak
  `<thinking>` tags into its visible response. Here the visible response *is* the
  extracted text, so a leaked tag lands in the corpus. Use `effort: "low"` instead.
- **Check `stop_reason == "refusal"` before reading `content`.** A refusal is HTTP 200
  with empty or partial content; indexing `content[0]` first turns a declined page
  into an `IndexError`.

### 14.4 Budget

`max_pages` (default 20) is checked before the first request, so an over-budget
document costs nothing. Exceeding it raises `MultimodalBudgetExceededError` rather
than trimming to fit — a partial transcription presented as a whole one is worse than
a clear refusal, and only the caller can decide whether the extra pages are worth
paying for.

### 14.5 Environment variables

Per provider, `AKSHARA_*_API_KEY` first, then the provider's conventional name
(`GEMINI_API_KEY` / `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Reading
a provider-native variable is safe **only** because finding a key never causes
anything — see 14.1.

Unlike the Tesseract probes, key lookup is deliberately **not** cached: a key can be
exported between calls in a notebook or a test, and a cached `None` would be
impossible to clear.
---

## 15. The Brain: Preprocessing & Neuro-Symbolic Chunking

Files: `preprocess/*`, `brain/*`, `contracts/chunking.py`

Realises interim report §4.6 and §6.5. Three stages: a Prolog rule base finds
linguistically defensible boundaries (Algorithm 3), LaBSE supplies topic
coherence (§6.5.3), and a bounded agglomerative merge combines them (Algorithm 4).

### 15.1 Preprocessing comes first

The Eye transcribes a page faithfully, which is not the same as producing prose.
Measured on `output/sample_unicode.pdf.txt`: 8,401 lines for a few hundred
paragraphs, 2,570 of them blank, 202 bare page numbers, 5,754 ZWNJ, 5,039 ZWJ,
and text that is not NFC.

The zero-width policy is the delicate part and is **not** a blanket strip:

- **Keep** ZWJ in `් + ZWJ + ය|ර` — යංසය (ශ්‍ය) and රකාරාංශය (ක්‍ර). Deleting it
  changes the word.
- **Strip** every other ZWJ and **all** ZWNJ. ZWNJ has no role in Sinhala
  orthography; here it is OCR emission after a hal kirima at word ends.

De-wrapping uses the same sentence-end test the chunker uses, so wrapping and
chunking cannot disagree about where a sentence ends.

### 15.2 The rule base is grounded in a named grammar

Every fact in `rules/sinhala_rules.pl` cites *වියරණ විවරණ* (හෙ. ව. බිහේෂ්
ඉන්දික සම්පත්, කැලණිය විශ්වවිද්‍යාලය, 2013), §4 උක්ත ආඛ්‍යාත සම්බන්ධතා,
pp. 89-112. Test expectations cite the same pages, so a linguist can audit the
rules against the source without reading Python.

The organising fact (p.89): Sinhala is **SOV**, so the *final verb* of a sentence
is its ආඛ්‍යාතය. A finite verb ending is therefore a sentence boundary — a far
stronger signal than punctuation, which Sinhala prose uses sparingly.

Four tiers, kept separate because conflating them was the prototype's main flaw:

| tier | predicate | examples |
|---|---|---|
| 1 | `sentence_terminator` | මි, මු, හි, හු, යි, ති, හ; the particle ය; වටී, මැනවි; `.` `?` `!` |
| 2 | `clause_boundary` | ලා, මින්, ගොස්, ොත්, ද්දී, තත් — clause ends, **sentence continues** |
| 3 | `discourse_connective` | නමුත්, එබැවින්, නිසා |
| 4 | `never_split` | සහ, සමඟ, හා, ද, හෝ (pp.110-112) — these bind *within* a clause |

Two rules do the heavy lifting:

- **Quotative `යි` disambiguation.** `යි` is both the 3sg ending (කරයි) and the
  quotative closing an අන්තර් වාක්‍යය (p.103, "සතුරන් ගමට එති’යි ඔවුහු බිය වූහ").
  The grammar's discriminator: the quotative attaches to an already-finite form,
  so strip it and ask whether the stem is itself finite.
- **Pronouns are never sentence ends.** ඔහු, ඔවුහු and මොවුහු end in හු, the 2pl
  verb ending; without the p.91 pronoun list, "ඔහු පාසල් ගියේ ය." splits after
  ඔහු — the most damaging false positive available. Justified structurally: a
  pronoun is an උක්ත or අනුක්ත, never a predicate.

**Known limitation.** `-හි` and `-හු` are ambiguous between verb inflection and
noun case marking (locative සමයෙහි, plural nominative සිත්තරහු). The closed lists
above cover the high-frequency cases; open-class nouns still over-split, and
separating them needs part-of-speech information this rule base does not have.

`:- encoding(utf8).` at the top of the `.pl` file is load-bearing: without it the
file consults cleanly but no Sinhala atom matches, silently degrading the whole
rule base to punctuation-only splitting.

### 15.3 The Prolog adapter

`swiplserver` over the Machine Query Interface, per report Table 4.1 — not
pyswip. Three fixes over the prototype: per-word queries are cached on the word
(deterministic, so caching cannot change behaviour) rather than one socket round
trip per token; lifecycle is a context manager rather than `__del__`; and the
`.pl` is located with `importlib.resources` so it works from an installed wheel.

`rule_tables.py` mirrors the rule base in Python, because preprocessing must run
without SWI-Prolog. A parity test asserts the two agree on every fixture word, so
drift is a test failure rather than quietly different chunking.

### 15.4 Tabular data must not be blended

The XLSX adapter emits one row per line with `\t` between cells. Whitespace-
tokenising that merges cells from unrelated rows into one chunk that reads like a
sentence and means nothing, and splits rows internally on incidental punctuation
(the price cell `රු. 100` contains a full stop).

So `segmenter.py` cuts text into atomic segments first. `TABLE_ROW` and
`SHEET_HEADING` bypass both micro-chunking and the merge: one row is one record
and becomes one chunk. `respect_table_rows=False` restores the naive behaviour
for comparison.

### 15.5 The merge and the output

Algorithm 4 verbatim: **the length guardrail is checked before the coherence
question**, so a self-similar passage cannot grow without bound. Note `max_words`
bounds *merging*, not micro-chunks — a single sentence longer than the bound is
emitted intact, because the alternative is cutting at an arbitrary word.

The coordinator depends on a `CoherenceScorer` Protocol, not on LaBSE, so the
merge is testable with a stub — no torch, no download, no network. This is also
the seam §6.5.3 asks for: swapping in a fine-tuned encoder is a constructor
argument.

`ChunkedDocument` supports `len`, `doc[3]`, `doc[2:5]`, iteration, `.texts` for
plain strings, and `to_json` / `to_jsonl`.

**A fine-tuned checkpoint is integrated, not just a ready seam.** `LabseScorer`
resolves its model in the same order `capabilities.py` already uses for
Tesseract and SWI-Prolog: constructor argument → `AKSHARA_LABSE_MODEL` → the
base hub model (`setu4993/LaBSE`). Fine-tuned on Sinhala Wikipedia (adjacent
sentences as positives, paragraph-boundary sentences as hard negatives) and
evaluated on a held-out split:

| | AUC | mean sim, coherent | mean sim, hard-negative |
|---|---|---|---|
| Base LaBSE | 0.7269 | 0.3871 | 0.2746 |
| Fine-tuned | 0.8832 | 0.5313 | 0.1620 |

Bootstrap 95% CI on the AUC improvement: `[+0.1419, +0.1694]` — excludes 0. The
checkpoint (1.8 GB, gitignored) lives at `models/labse-sinhala-finetuned/`;
`test_encoder.py`'s `finetuned_labse_dir` fixture skips cleanly wherever it
isn't present, and asserts the property the model exists to deliver — a
coherent pair must score above an incoherent one — so a checkpoint swap that
silently regresses this fails a test rather than only showing up as worse
chunking downstream.

### 15.6 Definition of done for the Brain

- `route()` output chunks without the caller touching preprocessing by hand.
- No chunk mixes cells from two spreadsheet rows.
- Prolog and `rule_tables` agree on every word in the grammar-derived table.
- The rule base loads its Sinhala atoms — asserted directly, since the failure is
  otherwise silent.
