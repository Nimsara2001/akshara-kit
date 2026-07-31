"""Implements Algorithm 1 (Eye Module Flow) for PDFs.

Runs every text-stream adapter, scores each with the quality probe and keeps
the best — a race, per Section 10 Phase 6. The report's Section 4.5 describes a
cost-ordered escalation ladder instead; the race is built here because it is
what the spec asks for and because a ladder can never report on the adapters it
skips, which the evaluation chapter needs. ``_sweep`` and the reserved
``_escalate`` share everything downstream, so adding the ladder later is a
small change rather than a rewrite.

One deviation from Algorithm 1 as written. The report normalises the *winner's*
text after the argmax. That does not work: legacy FM-font text is stored as
Latin-1 bytes, so every candidate scores a Sinhala ratio of exactly 0.0 and the
argmax is a four-way tie at zero. Normalisation therefore happens **before**
scoring.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Callable

from akshara_kit.adapters.extractors import (
    pdfminer_adapter,
    pdfplumber_adapter,
    pymupdf_adapter,
    pypdf_adapter,
)
from akshara_kit.contracts.extraction import (
    AdapterAttempt,
    ExtractionResult,
    FontDetectionMethod,
    QualityScore,
    SourceFormat,
)
from akshara_kit.eye.errors import ExtractionFailedError
from akshara_kit.eye.quality_probe import compare, is_viable, score

__all__ = ["PDF_ADAPTERS", "extract"]

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0

_PAGE_SEPARATOR = "\n\n"

#: Span-normalised output is reported under this backend id, so a reader can
#: tell it apart from a plain adapter win.
SPAN_BACKEND_ID = "pymupdf+spans"


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """A registered extraction backend.

    ``cost`` is the relative expense of running it, and doubles as the
    tie-break order: when quality is indistinguishable, the cheapest wins.
    """

    backend_id: str
    extract: Callable[[str], ExtractionResult]
    cost: int


#: Declared in cost order. The reserved escalation ladder walks this same
#: registry in the same order, which is why the ordering is meaningful here
#: even though the race runs everything.
PDF_ADAPTERS: tuple[AdapterSpec, ...] = (
    AdapterSpec("pypdf", pypdf_adapter.extract, cost=1),
    AdapterSpec("pymupdf", pymupdf_adapter.extract, cost=2),
    AdapterSpec("pdfplumber", pdfplumber_adapter.extract, cost=3),
    AdapterSpec("pdfminer", pdfminer_adapter.extract, cost=4),
)


def extract(
    file_path: str,
    *,
    parallel: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    use_ocr: bool = True,
) -> ExtractionResult:
    """Extract a PDF's text, normalised to Sinhala Unicode.

    Implements Algorithm 1 with normalise-before-score. Raises
    :class:`ExtractionFailedError` if every adapter failed.
    """
    started = time.perf_counter()
    results, attempts = _sweep(
        file_path, parallel=parallel, timeout_seconds=timeout_seconds
    )
    if not results:
        raise ExtractionFailedError(
            f"Every PDF adapter failed on {file_path}: "
            + ", ".join(f"{a.backend_id} ({a.error_type})" for a in attempts)
        )

    candidates = _build_candidates(file_path, results)
    for attempt in attempts:
        if attempt.backend_id in candidates:
            attempt.quality = candidates[attempt.backend_id].quality

    winner = _select(candidates)
    winner = _apply_ocr(file_path, winner, enabled=use_ocr)

    return ExtractionResult(
        text=winner.text,
        backend_id=winner.backend_id,
        source_format=SourceFormat.PDF,
        latency_seconds=time.perf_counter() - started,
        quality=winner.quality,
        font_detection_method=winner.method,
        detected_legacy_fonts=winner.converted_fonts,
        unmapped_legacy_fonts=winner.unmapped_fonts,
        ocr_used=bool(winner.pages_ocred),
        pages_ocred=winner.pages_ocred,
        attempts=attempts,
        metadata=winner.metadata,
    )


# --- the sweep ------------------------------------------------------------


def _sweep(
    file_path: str, *, parallel: bool, timeout_seconds: float
) -> tuple[dict[str, ExtractionResult], list[AdapterAttempt]]:
    """Run every adapter, recording failures instead of swallowing them.

    Threads rather than processes: the PDF libraries are C-extension bound and
    release the GIL, and there is nothing to pickle. This is only safe because
    adapters take a *path* and open their own handle — never a shared document
    object.
    """
    if not parallel:
        return _sweep_sequential(file_path)

    results: dict[str, ExtractionResult] = {}
    attempts: dict[str, AdapterAttempt] = {}
    started = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PDF_ADAPTERS)) as pool:
        futures = {
            pool.submit(spec.extract, file_path): spec for spec in PDF_ADAPTERS
        }
        for future, spec in futures.items():
            elapsed = time.perf_counter() - started
            try:
                result = future.result(timeout=max(timeout_seconds - elapsed, 0.1))
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                attempts[spec.backend_id] = _failure(spec, exc, elapsed)
                continue
            results[spec.backend_id] = result
            attempts[spec.backend_id] = AdapterAttempt(
                backend_id=spec.backend_id,
                succeeded=True,
                latency_seconds=result.latency_seconds,
                quality=result.quality,
            )

    # Rank in the fixed registry order, never in completion order, so the
    # outcome is identical run to run despite the parallelism.
    ordered = [attempts[s.backend_id] for s in PDF_ADAPTERS if s.backend_id in attempts]
    return results, ordered


def _sweep_sequential(
    file_path: str,
) -> tuple[dict[str, ExtractionResult], list[AdapterAttempt]]:
    """The same sweep without threads, for debugging and deterministic timing."""
    results: dict[str, ExtractionResult] = {}
    attempts: list[AdapterAttempt] = []
    for spec in PDF_ADAPTERS:
        started = time.perf_counter()
        try:
            result = spec.extract(file_path)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            attempts.append(_failure(spec, exc, time.perf_counter() - started))
            continue
        results[spec.backend_id] = result
        attempts.append(
            AdapterAttempt(
                backend_id=spec.backend_id,
                succeeded=True,
                latency_seconds=result.latency_seconds,
                quality=result.quality,
            )
        )
    return results, attempts


def _failure(spec: AdapterSpec, exc: BaseException, elapsed: float) -> AdapterAttempt:
    """Record an adapter failure. Never folded into the returned text."""
    logger.warning("adapter %s failed: %s", spec.backend_id, exc)
    return AdapterAttempt(
        backend_id=spec.backend_id,
        succeeded=False,
        latency_seconds=elapsed,
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],
    )


# --- candidates and selection --------------------------------------------


@dataclass
class _Candidate:
    """A scored, already-normalised candidate output."""

    backend_id: str
    text: str
    quality: QualityScore
    cost: int
    method: FontDetectionMethod = FontDetectionMethod.NONE
    converted_fonts: list[str] = None  # type: ignore[assignment]
    unmapped_fonts: list[str] = None  # type: ignore[assignment]
    pages: list[str] = None  # type: ignore[assignment]
    pages_ocred: list[int] = None  # type: ignore[assignment]
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.converted_fonts = self.converted_fonts or []
        self.unmapped_fonts = self.unmapped_fonts or []
        self.pages = self.pages or []
        self.pages_ocred = self.pages_ocred or []
        self.metadata = self.metadata or {}


def _build_candidates(
    file_path: str, results: dict[str, ExtractionResult]
) -> dict[str, _Candidate]:
    """Score every adapter's output, plus the span-normalised alternative.

    Font detection runs once for the whole document. When a mappable legacy
    font is present, the span-normalised text is added as an extra candidate —
    and it wins by construction, because the raw candidates all score zero.
    """
    candidates = {
        spec.backend_id: _Candidate(
            backend_id=spec.backend_id,
            text=results[spec.backend_id].text,
            quality=results[spec.backend_id].quality or score(results[spec.backend_id].text),
            cost=spec.cost,
        )
        for spec in PDF_ADAPTERS
        if spec.backend_id in results
    }

    span_candidate = _build_span_candidate(file_path)
    if span_candidate is not None:
        candidates[SPAN_BACKEND_ID] = span_candidate
    return candidates


def _build_span_candidate(file_path: str) -> _Candidate | None:
    """Span-normalised text, or ``None`` if this document needs no conversion.

    Per-span conversion is the only safe kind: it converts exactly the runs
    drawn in a legacy font and leaves Latin and existing Unicode text alone.
    """
    from akshara_kit.eye.encoding_normaliser import normalise_spans
    from akshara_kit.eye.font_detection import FontClass, classify_font

    try:
        spans = list(pymupdf_adapter.iter_spans(file_path))
    except Exception as exc:  # noqa: BLE001 - fall back to the plain race
        logger.warning("span extraction unavailable for %s: %s", file_path, exc)
        return None

    if not any(
        classify_font(span.font) is FontClass.LEGACY_MAPPABLE
        for span in spans
        if not span.is_separator
    ):
        return None

    outcome = normalise_spans(spans)
    return _Candidate(
        backend_id=SPAN_BACKEND_ID,
        text=outcome.text,
        quality=score(outcome.text),
        cost=PDF_ADAPTERS[1].cost,
        method=outcome.method,
        converted_fonts=outcome.converted_fonts,
        unmapped_fonts=outcome.unmapped_fonts,
    )


def _select(candidates: dict[str, _Candidate]) -> _Candidate:
    """Pick the best candidate: quality first, then cost.

    Candidates below the viability floor are excluded unless nothing clears it,
    so a backend returning three characters cannot win on a perfect ratio.
    """
    pool = [c for c in candidates.values() if is_viable(c.quality)] or list(
        candidates.values()
    )

    best = pool[0]
    for candidate in pool[1:]:
        verdict = compare(candidate.quality, best.quality)
        if verdict > 0 or (verdict == 0 and candidate.cost < best.cost):
            best = candidate

    best.metadata = dict(best.metadata)
    best.metadata["raw_race_winner"] = _raw_winner(candidates)
    best.metadata["candidates"] = {
        backend_id: {
            "sinhala_ratio": round(c.quality.sinhala_ratio, 4),
            "raw_length": c.quality.raw_length,
        }
        for backend_id, c in candidates.items()
    }
    return best


def _raw_winner(candidates: dict[str, _Candidate]) -> str | None:
    """Which plain adapter would have won, ignoring the span candidate."""
    raw = [c for backend_id, c in candidates.items() if backend_id != SPAN_BACKEND_ID]
    if not raw:
        return None
    best = raw[0]
    for candidate in raw[1:]:
        verdict = compare(candidate.quality, best.quality)
        if verdict > 0 or (verdict == 0 and candidate.cost < best.cost):
            best = candidate
    return best.backend_id


# --- OCR ------------------------------------------------------------------


def _apply_ocr(file_path: str, winner: _Candidate, *, enabled: bool) -> _Candidate:
    """Replace pages that have no usable text stream with OCR output.

    Merged back at the original page index, so a document mixing scanned and
    born-digital pages keeps its reading order. OCR output is Unicode Sinhala
    already, so it is never passed through legacy conversion.
    """
    if not enabled:
        return winner

    from akshara_kit.eye import capabilities

    pages = _page_texts(file_path)
    if pages is None:
        return winner

    needed = _pages_needing_ocr(file_path, pages)
    if not needed:
        return winner

    if not capabilities.sinhala_ocr_available():
        logger.warning(
            "%d page(s) of %s have no text layer but OCR is unavailable: %s",
            len(needed),
            file_path,
            capabilities.describe_ocr_availability(),
        )
        return winner

    from akshara_kit.adapters.extractors import ocr_adapter

    ocred: list[int] = []
    for index in needed:
        try:
            pages[index] = ocr_adapter.extract_page(file_path, index)
        except Exception as exc:  # noqa: BLE001 - a failed page is not a failed doc
            logger.warning("OCR failed on page %d of %s: %s", index, file_path, exc)
            continue
        ocred.append(index)

    if not ocred:
        return winner

    winner.text = _PAGE_SEPARATOR.join(pages)
    winner.quality = score(winner.text)
    winner.pages_ocred = ocred
    return winner


def _page_texts(file_path: str) -> list[str] | None:
    """Per-page text from PyMuPDF, or ``None`` if it is unavailable."""
    try:
        return pymupdf_adapter._extract_pages(file_path)
    except Exception as exc:  # noqa: BLE001 - OCR routing is best-effort
        logger.warning("per-page text unavailable for %s: %s", file_path, exc)
        return None


def _pages_needing_ocr(file_path: str, pages: list[str]) -> list[int]:
    """Indices of pages whose text stream is empty and which look scanned."""
    from akshara_kit.eye.ocr_decision import needs_ocr

    pymupdf = pymupdf_adapter._pymupdf()
    try:
        with pymupdf.open(file_path) as document:
            return [
                index
                for index, page in enumerate(document)
                if index < len(pages) and needs_ocr(page, pages[index])
            ]
    except Exception as exc:  # noqa: BLE001 - OCR routing is best-effort
        logger.warning("OCR decision unavailable for %s: %s", file_path, exc)
        return []
