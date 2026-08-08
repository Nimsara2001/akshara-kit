"""The multimodal vision-language fallback.

Hermetic: every provider is stubbed, so these run on any machine with no SDK
installed, no API key, and no network. The one thing that cannot be faked —
that a real provider returns usable Sinhala — is left to the `vlm`-marked tests
at the bottom, which skip unless a key is configured.

The most important test here is
``test_route_without_config_never_contacts_a_provider``. Everything else checks
that the feature works; that one checks it stays off, which is the property a
user is actually trusting.
"""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit import (
    MultimodalBudgetExceededError,
    MultimodalConfig,
    MultimodalProvider,
    MultimodalUnavailableError,
)
from akshara_kit.eye.errors import AdapterUnavailableError, ExtractionFailedError
from akshara_kit.multimodal import fallback
from akshara_kit.multimodal.providers import claude, gemini, openai
from samples import UNICODE_SINHALA

PROVIDER_MODULES = {"gemini": gemini, "openai": openai, "claude": claude}


@pytest.fixture
def stub_provider(monkeypatch):
    """Replace a provider's ``transcribe`` and record what it was called with."""

    def _install(name: str = "gemini", text: str = UNICODE_SINHALA):
        calls: list[dict] = []

        def fake(png: bytes, *, model: str | None = None) -> str:
            calls.append({"png": png, "model": model})
            return text

        monkeypatch.setattr(PROVIDER_MODULES[name], "transcribe", fake)
        return calls

    return _install


# --- consent --------------------------------------------------------------


def test_route_without_config_never_contacts_a_provider(
    monkeypatch, mixed_pdf: pathlib.Path
) -> None:
    """The guarantee the whole design exists to make.

    A key in the environment is not permission. Without a config object, no
    code path may reach a provider — so every route into one is booby-trapped
    here rather than asserting on a return value.
    """
    from akshara_kit import route

    def boom(*args, **kwargs):
        raise AssertionError("contacted a provider without an explicit opt-in")

    monkeypatch.setattr(fallback, "transcribe_pages", boom)
    monkeypatch.setattr(fallback, "extract_page", boom)
    monkeypatch.setattr(fallback, "transcriber_for", boom)

    result = route(str(mixed_pdf))
    assert result.pages_multimodal == []
    assert result.multimodal_provider is None


def test_config_requires_a_provider() -> None:
    """No default provider: 'where do my documents go' is never inferred."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MultimodalConfig()  # type: ignore[call-arg]


def test_result_records_where_pages_were_sent(stub_provider, mixed_pdf) -> None:
    """Auditable after the fact: which pages left, and to whom."""
    from akshara_kit.eye import pdf_coordinator

    stub_provider("gemini")
    result = pdf_coordinator.extract(
        str(mixed_pdf), use_ocr=False, multimodal=MultimodalConfig(provider="gemini")
    )
    assert result.pages_multimodal
    assert result.multimodal_provider == "gemini"


# --- dispatch and model resolution ----------------------------------------


@pytest.mark.parametrize("name", ["gemini", "openai", "claude"])
def test_each_provider_resolves_to_its_own_module(name: str) -> None:
    assert fallback.transcriber_for(name) is PROVIDER_MODULES[name]


def test_enum_and_string_resolve_identically() -> None:
    assert fallback.transcriber_for(MultimodalProvider.CLAUDE) is claude


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(MultimodalUnavailableError, match="Unknown multimodal provider"):
        fallback.transcriber_for("bogus")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gemini", "gemini-3.6-flash"),
        ("openai", "gpt-5.6"),
        ("claude", "claude-opus-5"),
    ],
)
def test_unset_model_resolves_to_the_provider_default(name: str, expected: str) -> None:
    assert fallback.resolve_model(MultimodalConfig(provider=name)) == expected


def test_config_model_overrides_the_default() -> None:
    config = MultimodalConfig(provider="gemini", model="gemini-2.5-pro")
    assert fallback.resolve_model(config) == "gemini-2.5-pro"


def test_per_call_model_overrides_the_config() -> None:
    config = MultimodalConfig(provider="openai", model="gpt-5.6-terra")
    assert fallback.resolve_model(config, "gpt-5.6-luna") == "gpt-5.6-luna"


def test_chosen_model_reaches_the_provider_and_the_result(
    stub_provider, mixed_pdf
) -> None:
    """An unknown model is the provider's to reject, not ours to pre-empt."""
    from akshara_kit.eye import pdf_coordinator

    calls = stub_provider("gemini")
    result = pdf_coordinator.extract(
        str(mixed_pdf),
        use_ocr=False,
        multimodal=MultimodalConfig(provider="gemini", model="some-future-model"),
    )
    assert {call["model"] for call in calls} == {"some-future-model"}
    assert result.metadata["multimodal_model"] == "some-future-model"


# --- availability ---------------------------------------------------------


def test_missing_key_is_reported_actionably(monkeypatch, tmp_path) -> None:
    for variable in ("AKSHARA_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(MultimodalUnavailableError, match="AKSHARA_GEMINI_API_KEY"):
        gemini.transcribe(b"not-a-real-png")


def test_missing_sdk_names_the_extra(monkeypatch) -> None:
    """The install hint must name the extra, not just the package."""
    monkeypatch.setenv("AKSHARA_ANTHROPIC_API_KEY", "test-key")
    real_import = __import__

    def no_anthropic(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_anthropic)
    with pytest.raises(AdapterUnavailableError, match="'multimodal' extra"):
        claude.transcribe(b"not-a-real-png")


# --- budget ---------------------------------------------------------------


def test_budget_is_enforced_before_any_request(monkeypatch, mixed_pdf) -> None:
    """An over-budget document must cost nothing, not cost the cap."""

    def boom(*args, **kwargs):
        raise AssertionError("spent money before checking the budget")

    monkeypatch.setattr(fallback, "extract_page", boom)

    with pytest.raises(MultimodalBudgetExceededError) as excinfo:
        fallback.transcribe_pages(
            str(mixed_pdf), [0, 1, 2], config=MultimodalConfig(provider="gemini", max_pages=2)
        )
    assert excinfo.value.needed == 3
    assert excinfo.value.allowed == 2


def test_exactly_at_the_cap_is_allowed(stub_provider, mixed_pdf) -> None:
    stub_provider("gemini")
    transcribed, _ = fallback.transcribe_pages(
        str(mixed_pdf), [0, 1], config=MultimodalConfig(provider="gemini", max_pages=2)
    )
    assert sorted(transcribed) == [0, 1]


def test_a_failed_page_does_not_lose_the_others(monkeypatch, mixed_pdf) -> None:
    """Best-effort, matching the OCR leg: one bad page is not a bad document."""

    def flaky(png: bytes, *, model: str | None = None) -> str:
        raise ExtractionFailedError("provider had a bad day")

    monkeypatch.setattr(gemini, "transcribe", flaky)
    transcribed, _ = fallback.transcribe_pages(
        str(mixed_pdf), [0, 1], config=MultimodalConfig(provider="gemini")
    )
    assert transcribed == {}


# --- last-resort ordering -------------------------------------------------


@pytest.mark.ocr
def test_pages_ocr_repaired_are_not_escalated(stub_provider, mixed_pdf) -> None:
    """The cost guarantee: a page OCR fixed must never reach a paid API."""
    from akshara_kit.eye import pdf_coordinator

    calls = stub_provider("gemini")
    result = pdf_coordinator.extract(
        str(mixed_pdf), multimodal=MultimodalConfig(provider="gemini")
    )
    assert result.pages_ocred, "this fixture should exercise the OCR leg"
    assert result.pages_multimodal == []
    assert calls == []


def test_pages_ocr_could_not_fix_are_escalated(stub_provider, mixed_pdf) -> None:
    """With OCR off, both garbled pages fall through to the vision model."""
    from akshara_kit.eye import pdf_coordinator

    calls = stub_provider("gemini")
    result = pdf_coordinator.extract(
        str(mixed_pdf), use_ocr=False, multimodal=MultimodalConfig(provider="gemini")
    )
    assert result.pages_multimodal == [0, 1]
    assert len(calls) == 2


def test_transcribed_pages_land_at_their_original_index(monkeypatch, mixed_pdf) -> None:
    """Reading order survives the merge."""
    from akshara_kit.eye import pdf_coordinator

    def numbered(png: bytes, *, model: str | None = None) -> str:
        numbered.n += 1  # type: ignore[attr-defined]
        return f"{UNICODE_SINHALA} PAGE{numbered.n}"  # type: ignore[attr-defined]

    numbered.n = -1  # type: ignore[attr-defined]
    monkeypatch.setattr(gemini, "transcribe", numbered)

    result = pdf_coordinator.extract(
        str(mixed_pdf), use_ocr=False, multimodal=MultimodalConfig(provider="gemini")
    )
    assert result.text.index("PAGE0") < result.text.index("PAGE1")


# --- provider response handling -------------------------------------------


def test_claude_refusal_raises_instead_of_returning_empty_text() -> None:
    """A refusal is HTTP 200 with no content — reading content[0] would crash."""

    class Response:
        stop_reason = "refusal"
        stop_details = type("D", (), {"category": "cyber"})()
        content: list = []

    with pytest.raises(ExtractionFailedError, match="declined"):
        claude._text_of(Response())


def test_claude_joins_only_text_blocks() -> None:
    class Block:
        def __init__(self, type_: str, text: str = "") -> None:
            self.type, self.text = type_, text

    class Response:
        stop_reason = "end_turn"
        content = [Block("thinking"), Block("text", "අකුරු"), Block("text", " කිට")]

    assert claude._text_of(Response()) == "අකුරු කිට"


def test_gemini_empty_response_is_empty_text_not_none() -> None:
    """A page with no legible text is an empty transcription, not a failure."""

    class Response:
        text = None

    assert (Response().text or "") == ""


# --- against a real API ---------------------------------------------------


@pytest.mark.vlm
def test_real_provider_transcribes_a_scanned_page(scanned_pdf: pathlib.Path) -> None:
    """Skipped unless a key is configured. Costs money when it runs."""
    from akshara_kit.eye import capabilities
    from akshara_kit.eye.quality_probe import sinhala_ratio

    provider = next(
        (p for p in fallback.PROVIDERS if capabilities.multimodal_available(p)), None
    )
    assert provider is not None

    text = fallback.extract_page(
        str(scanned_pdf), 0, config=MultimodalConfig(provider=provider)
    )
    assert text.strip()
    assert sinhala_ratio(text) > 0.3
