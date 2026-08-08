"""The Neural Encoding Component, and the fine-tuned checkpoint specifically.

Two different things are under test here. The env-var resolution tests are pure
logic — no model loaded, no marker needed, run everywhere. The tests against the
fine-tuned checkpoint itself are marked ``labse`` (needs sentence-transformers)
and additionally depend on the ``finetuned_labse_dir`` fixture, which skips when
the 1.8 GB checkpoint isn't present — true on any machine but this one, since
it's deliberately not committed (see .gitignore).
"""

from __future__ import annotations

import pathlib

import pytest

from akshara_kit.brain.encoder import DEFAULT_MODEL, LabseScorer

# --- model resolution: pure logic, no model loaded, no marker ------------


def test_no_override_resolves_to_the_default_model(monkeypatch) -> None:
    monkeypatch.delenv("AKSHARA_LABSE_MODEL", raising=False)
    assert LabseScorer().model_name == DEFAULT_MODEL


def test_env_var_overrides_the_default(monkeypatch) -> None:
    monkeypatch.setenv("AKSHARA_LABSE_MODEL", "models/labse-sinhala-finetuned")
    assert LabseScorer().model_name == "models/labse-sinhala-finetuned"


def test_explicit_argument_wins_over_the_env_var(monkeypatch) -> None:
    monkeypatch.setenv("AKSHARA_LABSE_MODEL", "from-env")
    assert LabseScorer(model_name="from-argument").model_name == "from-argument"


# --- the fine-tuned checkpoint itself -------------------------------------


@pytest.mark.labse
def test_finetuned_checkpoint_loads_and_embeds(finetuned_labse_dir: pathlib.Path) -> None:
    scorer = LabseScorer(model_name=str(finetuned_labse_dir))
    vector = scorer.embed("මම බත් කමි.")
    assert vector.shape == (768,)


@pytest.mark.labse
def test_coherent_pair_scores_higher_than_incoherent_pair(
    finetuned_labse_dir: pathlib.Path,
) -> None:
    """The property this checkpoint exists to deliver.

    Mirrors the training notebook's own held-out evaluation (adjacent sentences
    as positives, topically unrelated sentences as hard negatives) at unit-test
    scale, so a checkpoint swap that silently regresses this ordering fails a
    test rather than only showing up as worse chunking downstream.
    """
    scorer = LabseScorer(model_name=str(finetuned_labse_dir))

    # Same topic: heavy rain, then its consequence.
    coherent = scorer.score(
        "අද වැස්ස තද ලෙස පවතී.",
        "මාර්ග කිහිපයක ජලය එකතු වී ඇත.",
    )
    # Unrelated fact spliced in — the shape of a paragraph-boundary hard negative.
    incoherent = scorer.score(
        "අද වැස්ස තද ලෙස පවතී.",
        "ශ්‍රී ලංකාවේ අගනුවර කොළඹ වේ.",
    )

    assert coherent > incoherent


@pytest.mark.labse
def test_score_is_symmetric(finetuned_labse_dir: pathlib.Path) -> None:
    """Cosine similarity of two independently-embedded vectors is order-free."""
    scorer = LabseScorer(model_name=str(finetuned_labse_dir))
    a = "අද වැස්ස තද ලෙස පවතී."
    b = "මාර්ග කිහිපයක ජලය එකතු වී ඇත."
    assert scorer.score(a, b) == pytest.approx(scorer.score(b, a))
