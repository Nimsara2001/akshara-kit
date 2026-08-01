"""Multimodal vision-language fallback (realises report Sections 4.5 and 6.7).

The highest-cost rung of the escalation ladder, and the only one that leaves the
machine. A vision-language model reads what a page *displays*, which makes it
the right last resort for the two failures nothing local can fix: a page whose
text layer is a broken ``ToUnicode`` cmap *and* whose rendering also defeats
Tesseract, or a scan on a machine with no OCR installed at all.

Consent
-------
Nothing here runs unless the caller passes a
:class:`~akshara_kit.contracts.extraction.MultimodalConfig`. An API key present
in the environment is *not* consent — keys are exported for all sorts of
unrelated reasons, and a document-ingestion library that started uploading a
user's documents because it happened to find one would be doing something the
user never asked for. The config object carries the opt-in, the provider and the
page budget together, so none of the three can be forgotten separately.

The provider is a required field rather than an inferred one. With two keys
configured, "use the first" is a guess about where the user's documents should
be sent, and that is not a guess this library makes.
"""

from __future__ import annotations

import io
import logging
import time
from types import ModuleType

from akshara_kit.contracts.extraction import (
    ExtractionResult,
    MultimodalConfig,
    MultimodalProvider,
    SourceFormat,
)
from akshara_kit.eye.errors import (
    ExtractionFailedError,
    MultimodalBudgetExceededError,
    MultimodalUnavailableError,
)
from akshara_kit.eye.quality_probe import score

__all__ = [
    "PROVIDERS",
    "extract",
    "extract_page",
    "render_page_png",
    "resolve_model",
    "transcribe_pages",
    "transcriber_for",
]

logger = logging.getLogger(__name__)

PROVIDERS: tuple[str, ...] = tuple(member.value for member in MultimodalProvider)

_PAGE_SEPARATOR = "\n\n"


def transcriber_for(provider: str | MultimodalProvider) -> ModuleType:
    """Import the module implementing one provider.

    Imported on demand so a caller who only uses Gemini never needs the other
    two SDKs installed.
    """
    name = provider.value if isinstance(provider, MultimodalProvider) else str(provider)
    if name not in PROVIDERS:
        raise MultimodalUnavailableError(
            f"Unknown multimodal provider {name!r}; "
            f"choose one of: {', '.join(PROVIDERS)}"
        )

    from importlib import import_module

    return import_module(f"akshara_kit.multimodal.providers.{name}")


def resolve_model(config: MultimodalConfig, override: str | None = None) -> str:
    """Decide which model to call: per-call override, then config, then default.

    Resolved here rather than inside each provider so the model that actually
    ran can be recorded on the result whichever provider served it — the
    evaluation chapter needs every number traceable to an exact model.
    """
    if override:
        return override
    if config.model:
        return config.model
    return transcriber_for(config.provider).DEFAULT_MODEL


def render_page_png(file_path: str, page_number: int, dpi: int) -> bytes:
    """Render one page to PNG bytes.

    Delegates to the OCR adapter's rasteriser rather than opening the document
    again: that path is already poppler-free, already honours
    ``AKSHARA_RASTERISER``, and is already covered by the OCR tests.
    """
    from akshara_kit.adapters.extractors import ocr_adapter

    image = ocr_adapter.rasterise_page(file_path, page_number, dpi)
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        image.close()


def extract_page(
    file_path: str,
    page_number: int,
    *,
    config: MultimodalConfig,
    model: str | None = None,
) -> str:
    """Transcribe a single page. Returns Unicode Sinhala, never legacy glyphs.

    The direct entry point for a caller who already knows which page needs a
    vision model, and the unit the coordinator escalates with.
    """
    provider = transcriber_for(config.provider)
    png = render_page_png(file_path, page_number, config.dpi)
    return provider.transcribe(png, model=resolve_model(config, model))


def extract(
    file_path: str,
    *,
    config: MultimodalConfig,
    model: str | None = None,
) -> ExtractionResult:
    """Transcribe an entire PDF with a vision-language model.

    For a caller who already knows the whole document needs one. The coordinator
    uses :func:`extract_page` instead, escalating only the pages that earned it.

    Note this takes a required keyword-only ``config``, deviating from Section
    12's one-argument adapter contract. That is deliberate: the contract exists
    so adapters are interchangeable, and an adapter that silently uploads the
    document is not interchangeable with one that does not.

    :raises MultimodalBudgetExceededError: if the document is longer than
        ``config.max_pages``.
    """
    from akshara_kit.adapters.extractors.pymupdf_adapter import page_count

    started = time.perf_counter()
    total = page_count(file_path)
    if total > config.max_pages:
        raise MultimodalBudgetExceededError(total, config.max_pages)

    chosen = resolve_model(config, model)
    pages = [
        extract_page(file_path, index, config=config, model=chosen)
        for index in range(total)
    ]
    text = _PAGE_SEPARATOR.join(pages)

    return ExtractionResult(
        text=text,
        backend_id=config.provider.value,
        source_format=SourceFormat.PDF,
        latency_seconds=time.perf_counter() - started,
        quality=score(text),
        pages_multimodal=list(range(total)),
        multimodal_provider=config.provider.value,
        metadata={"multimodal_model": chosen, "pages": total},
    )


def transcribe_pages(
    file_path: str,
    page_numbers: list[int],
    *,
    config: MultimodalConfig,
) -> tuple[dict[int, str], str]:
    """Transcribe selected pages, enforcing the budget before spending anything.

    Returns ``(page index -> text, model used)``. The budget is checked before
    the first request, so an over-budget document costs nothing. A page that
    fails is logged and skipped rather than discarding the pages that
    succeeded — the same best-effort contract the OCR leg follows.
    """
    if len(page_numbers) > config.max_pages:
        raise MultimodalBudgetExceededError(len(page_numbers), config.max_pages)

    chosen = resolve_model(config)
    transcribed: dict[int, str] = {}
    for index in page_numbers:
        try:
            transcribed[index] = extract_page(
                file_path, index, config=config, model=chosen
            )
        except ExtractionFailedError as exc:
            logger.warning(
                "multimodal transcription failed on page %d of %s: %s",
                index,
                file_path,
                exc,
            )
    return transcribed, chosen
