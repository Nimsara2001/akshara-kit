"""Run route() -> preprocess.clean() -> Brain chunk_text() over every
tests/fixtures/* file and dump each document's final chunks to a JSON file
under output/, exactly as an installed consumer of the library would call it.

Preprocessing runs before chunking (Section 15.1): the Eye transcribes a page
faithfully, which is not the same as prose, so line wraps, page numbers and
stray zero-width characters are cleaned up here rather than left for the rule
base to trip on.

Chunking needs SWI-Prolog for the symbolic rule engine and, for the coherence
merge, a sentence encoder (LaBSE by default, which needs sentence-transformers
+ torch). Both are optional extras; if either is missing, this script reports
why for the affected fixture and moves on rather than crashing the whole run.

If a fine-tuned checkpoint is present at ``models/labse-sinhala-finetuned``
(see the project README), this script uses that local copy in place of
downloading the same fine-tune from the Hub — that is project-local
convenience, not a library default; see ``brain/encoder.py``'s
``AKSHARA_LABSE_MODEL`` resolution for how to point the library itself at a
checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

from akshara_kit import route
from akshara_kit.brain.coordinator import HybridChunker
from akshara_kit.brain.encoder import LabseScorer
from akshara_kit.eye.errors import AksharaKitError
from akshara_kit.preprocess import clean

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "labse-sinhala-finetuned"

FIXTURE_FILES = [
    "sample_unicode.pdf",
    "sample_legacy_font.pdf",
    "sample_mixed.pdf",
    "sample_scanned.pdf",
    "sample.docx",
    "sample.xlsx",
]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    counts = {"ok": 0, "fail": 0, "skip": 0}

    if MODEL_DIR.is_dir():
        scorer = LabseScorer(model_name=str(MODEL_DIR))
        print(f"[scorer] fine-tuned LaBSE: {MODEL_DIR}")
    else:
        scorer = None  # HybridChunker falls back to LabseScorer()'s own default
        print(f"[scorer] {MODEL_DIR} not found; downloading the fine-tune from the Hub")

    # One chunker for the whole run: the Prolog process and the sentence
    # encoder are both expensive to start, and the rule base is a pure
    # function of the word, so nothing about reusing them across fixtures
    # changes the answer.
    with HybridChunker(scorer=scorer) as chunker:
        for name in FIXTURE_FILES:
            src = FIXTURES_DIR / name
            if not src.is_file():
                print(f"[skip] {name}: fixture not found")
                counts["skip"] += 1
                continue

            # ``src.stem`` alone collides for sample.docx / sample.xlsx (both
            # stem to "sample"), so the second would silently clobber the
            # first's output. Keep the original suffix in the stem.
            out_path = OUTPUT_DIR / f"{src.name}.chunks.json"
            try:
                result = route(str(src))
                cleaned = clean(result.text)
                doc = chunker.chunk_text(
                    cleaned.text,
                    source_format=result.source_format,
                    source_document=name,
                )
            except AksharaKitError as exc:
                out_path.write_text(
                    json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[fail] {name}: {exc}")
                counts["fail"] += 1
                continue

            doc.to_json(out_path)
            print(
                f"[ok]   {name} -> {out_path.name} "
                f"(chunks={len(doc)}, segments={doc.metadata.get('segments')}, "
                f"micro_chunks={doc.metadata.get('micro_chunks')})"
            )
            counts["ok"] += 1

    print(
        f"\n{counts['ok']} ok, {counts['fail']} failed, {counts['skip']} skipped "
        f"-> {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
