# akshara-kit

<p align="center">
  <img src="https://raw.githubusercontent.com/Nimsara2001/akshara-kit/main/assets/akshara-kit-logo.png" alt="akshara-kit logo" width="200">
</p>

A hybrid intelligent framework for ingesting Sinhala-language documents into clean,
Unicode-normalised text suitable for downstream semantic chunking and Retrieval-Augmented
Generation.

Two modules. The **Eye** handles file-type detection, text extraction, legacy-font
detection, legacy-to-Unicode conversion, OCR routing and a vision-model fallback. The
**Brain** turns that text into semantic chunks, combining a Sinhala grammar rule base in
Prolog with a multilingual sentence encoder.

```python
from akshara_kit import route, chunk

result = route("textbook.pdf")     # extract
doc = chunk(result)                # chunk

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
`ocr`, `sinhala`, `multimodal`, `brain`. Importing the package never requires extras you
are not using. Note `brain` is **not** included in `all`, because `sentence-transformers`
pulls in torch — install it explicitly when you need the neural coherence scorer.

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

## Chunking (the Brain)

Extraction gives you a document. Chunking gives you the units a retrieval index
actually stores.

```python
from akshara_kit import route, chunk

doc = chunk(route("textbook.pdf"))

doc.texts                      # ['...', '...']      plain strings
doc[0]                         # SemanticChunk       one chunk
doc[10:20]                     # list[SemanticChunk] a range
len(doc)                       # how many
doc.to_json("chunks.json")     # everything, with provenance
doc.to_jsonl("chunks.jsonl")   # one object per line, for vector-store loaders
```

Each chunk carries its text, a stable id, word count, source document and format,
per-chunk quality scores, and the boundaries that fell inside it — so you can see
*why* a chunk ends where it does.

### How the boundaries are decided

Three stages. A Prolog rule base finds linguistically defensible boundaries,
LaBSE scores how strongly two adjacent pieces discuss the same topic, and a
bounded agglomerative merge combines them under a word limit.

The rule base is the interesting part, because Sinhala is **SOV**: the final verb
of a sentence is its ආඛ්‍යාතය (predicate), so a finite verb ending marks a
sentence end far more reliably than punctuation, which Sinhala prose uses
sparingly. The rules are grounded in a named grammar — *වියරණ විවරණ* (හෙ. ව.
බිහේෂ් ඉන්දික සම්පත්, කැලණිය විශ්වවිද්‍යාලය, 2013) — and every rule and test
cites the page it comes from, so a linguist can audit them without reading Python.

They also encode what must **not** be split. `සහ`, `සමඟ`, `හා` and `ද` join
phrases inside a single clause, so splitting at them orphans the verb from its
subject: `පියා දරුවන් සමඟ වැඩ කරයි` is one thought, not two. The quotative `යි`
gets the same protection — in `සතුරන් ගමට එතියි ඔවුහු බිය වූහ` it closes an
embedded clause that belongs with its main clause.

Choose sentence-level or clause-level boundaries:

```python
from akshara_kit import chunk, ChunkConfig, BoundaryKind

chunk(result, config=ChunkConfig(level=BoundaryKind.CLAUSE, max_words=20))
```

### Spreadsheets are not prose

A row is a record. Whitespace-tokenising a spreadsheet merges cells from
unrelated rows into a chunk that reads like a sentence and describes two
different things — worse than useless in a retrieval index, because it matches
queries about either and answers about neither.

So table rows are atomic: never split, never merged with a neighbour. One row is
one chunk, cell structure intact. This happens automatically — `chunk()` reads
`source_format` from the extraction result.

### Preprocessing

`chunk()` runs on whatever text you give it, but extracted text carries PDF
layout scars — hard line wraps mid-sentence, page numbers, running headers, and
stray zero-width characters. Clean it first:

```python
from akshara_kit import clean

cleaned = clean(result.text)
print(cleaned.stages)   # what each stage changed, for reporting
```

Zero-width handling is deliberately not a blanket strip. `ශ්‍ය` and `ක්‍ර` need
their joiner — deleting it changes the word — so ZWJ is kept exactly where it
forms one of those two conjuncts and removed everywhere else. ZWNJ is always
removed; it has no role in Sinhala orthography and in extracted text is OCR
emission.

### System dependency: SWI-Prolog

The rule base runs on a real Prolog engine, reached through the Machine Query
Interface.

1. Install **SWI-Prolog 9+** from <https://www.swi-prolog.org/download>.
2. Install the extra: `uv sync --extra brain`
3. Verify: `swipl --version`

If SWI-Prolog is somewhere unusual, point the library at it with
`AKSHARA_SWIPL_CMD=/full/path/to/swipl`.

The `brain` extra also installs `sentence-transformers`, which pulls in torch.
It is deliberately **not** part of `all` — the rule base and the merge run
without it against any object with a `score(a, b) -> float` method, so you can
supply your own scorer, or the default fine-tuned encoder, or a checkpoint of
your own:

```python
from akshara_kit.brain import LabseScorer

chunk(result)  # no scorer given -> LabseScorer() -> DEFAULT_MODEL
```

`chunk()` uses `LabseScorer()` — a Sinhala fine-tune of LaBSE — by default, so
**no setup is required to get the better model**: `pip install akshara-kit[brain]`
and call `chunk()`. `sentence-transformers` fetches the checkpoint from the
Hugging Face Hub on first call and caches it locally; every call after that is
local, no network. The fine-tune was trained on Sinhala Wikipedia to separate
topically-coherent sentence pairs from paragraph-boundary hard negatives, and
measurably outperforms the base multilingual model on this task:

| | AUC | mean sim, coherent | mean sim, hard-negative |
|---|---|---|---|
| Base LaBSE | 0.7269 | 0.3871 | 0.2746 |
| Fine-tuned (default) | 0.8832 | 0.5313 | 0.1620 |

To use the base multilingual model instead, or your own checkpoint, override
with `model_name` or `AKSHARA_LABSE_MODEL` — same resolution order as
`AKSHARA_SWIPL_CMD` above (explicit argument, then the environment variable,
then `DEFAULT_MODEL`):

```python
chunk(result, scorer=LabseScorer("setu4993/LaBSE"))        # base model
chunk(result, scorer=LabseScorer("path/to/your-finetune")) # your own checkpoint
```

```bash
export AKSHARA_LABSE_MODEL=setu4993/LaBSE
```

`scripts/chunk_fixtures.py` prefers a local checkpoint for local development:
if `models/labse-sinhala-finetuned/` exists in the project root it's used in
place of whatever `AKSHARA_LABSE_MODEL` resolves to, and the script prints
which one ran. That directory is gitignored — a checkpoint is a local
artifact, not something to commit.

Nothing downloads until `chunk()` actually runs — there is no install-time or
import-time fetch, matching this library's consent-first stance on network
access (see [Multimodal fallback](#multimodal-fallback-opt-in) below). The
download is a one-time cost: `sentence-transformers` caches the checkpoint
under `~/.cache/huggingface/hub`, so every call after the first is local.

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
