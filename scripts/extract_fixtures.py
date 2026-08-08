"""Run akshara_kit.route() over every tests/fixtures/* file and dump each
extracted result to its own .txt file under output/, exactly as an installed
consumer of the library would call it.
"""

from __future__ import annotations

from pathlib import Path

from akshara_kit import route
from akshara_kit.eye.errors import AksharaKitError

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

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

    for name in FIXTURE_FILES:
        src = FIXTURES_DIR / name
        if not src.is_file():
            print(f"[skip] {name}: fixture not found")
            continue


        out_path = OUTPUT_DIR / f"{src.name}.txt"
        try:
            result = route(str(src))
        except AksharaKitError as exc:
            out_path.write_text(f"EXTRACTION FAILED: {exc}\n", encoding="utf-8")
            print(f"[fail] {name}: {exc}")
            continue

        out_path.write_text(result.text, encoding="utf-8")
        ratio = result.quality.sinhala_ratio if result.quality else float("nan")
        print(
            f"[ok]   {name} -> {out_path.name} "
            f"(backend={result.backend_id}, chars={len(result.text)}, "
            f"sinhala_ratio={ratio:.2f}, ocr_used={result.ocr_used})"
        )


if __name__ == "__main__":
    main()
