"""Tests for the pure (CPU-only) evaluation metric functions."""

from __future__ import annotations

from clinical_scribe.config import Config
from clinical_scribe.eval import (
    ModelScores,
    check_success_gate,
    completeness_score,
    extract_clinical_terms,
    grounding_score,
    is_structurally_valid,
    structure_validity_rate,
)


def test_extract_clinical_terms_keeps_numbers_and_content_words() -> None:
    terms = extract_clinical_terms("Patient on Lisinopril 10 mg daily, BP 120/80.")
    assert "lisinopril" in terms
    assert "10" in terms  # dose number retained
    assert "120" in terms and "80" in terms
    # stopword-ish / short tokens dropped
    assert "on" not in terms
    assert "mg" not in terms  # length < 4 and not a number


def test_is_structurally_valid_basic() -> None:
    assert is_structurally_valid("Three-day history of productive cough.")
    assert is_structurally_valid("N/A")
    assert not is_structurally_valid("")
    assert not is_structurally_valid("   ")


def test_is_structurally_valid_rejects_artifacts() -> None:
    assert not is_structurally_valid("Cough <|im_end|>")
    assert not is_structurally_valid("<think>reasoning</think> cough")


def test_is_structurally_valid_rejects_degenerate_repetition() -> None:
    assert not is_structurally_valid("cough " * 20)


def test_structure_validity_rate() -> None:
    preds = ["Good note.", "", "N/A", "bad <|im_start|>"]
    # 2 of 4 valid (Good note, N/A)
    assert structure_validity_rate(preds) == 0.5


def test_grounding_score_detects_hallucination() -> None:
    source = "Patient reports cough for three days."
    grounded = grounding_score("Cough for three days.", source)
    hallucinated = grounding_score("Patient has diabetes and hypertension.", source)
    assert grounded > hallucinated
    assert grounding_score("", source) == 1.0  # nothing asserted


def test_completeness_score_detects_omission() -> None:
    reference = "Productive cough, fever, and fatigue for three days."
    complete = completeness_score("Productive cough, fever, fatigue three days.", reference)
    partial = completeness_score("Cough.", reference)
    assert complete > partial


def test_check_success_gate_pass_and_fail() -> None:
    config = Config()
    base = ModelScores("base", n=10, rouge_l=0.30, bertscore_f=0.80, hallucination_rate=0.20)
    better = ModelScores(
        "ft",
        n=10,
        rouge_l=0.45,
        bertscore_f=0.85,
        structure_validity=0.99,
        hallucination_rate=0.15,
    )
    gate = check_success_gate(base, better, config)
    assert gate["passed"] is True

    worse = ModelScores(
        "ft",
        n=10,
        rouge_l=0.25,  # worse than base
        bertscore_f=0.85,
        structure_validity=0.99,
        hallucination_rate=0.15,
    )
    assert check_success_gate(base, worse, config)["passed"] is False
