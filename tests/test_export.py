"""Tests for the pure (CPU-only) export helpers: the model card builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_scribe.config import Config
from clinical_scribe.export import _find_quantize_binary, build_model_card, run_export


def test_model_card_has_frontmatter_and_disclaimer() -> None:
    config = Config()
    card = build_model_card(config, repo_id="acme/clinical-scribe", is_adapter=False)
    # YAML frontmatter for the Hub.
    assert card.startswith("---\n")
    assert f"license: {config.export.model_license}" in card
    assert f"base_model: {config.model.base_model_id}" in card
    # The clinician-review disclaimer must be present verbatim.
    assert config.prompts.disclaimer in card
    # Safety framing.
    assert "not a diagnostic tool" in card.lower()
    assert "Out of scope" in card


def test_model_card_adapter_vs_merged_differ() -> None:
    config = Config()
    adapter = build_model_card(config, repo_id="acme/cs-lora", is_adapter=True)
    merged = build_model_card(config, repo_id="acme/cs", is_adapter=False)
    assert "library_name: peft" in adapter
    assert "library_name: transformers" in merged
    assert "PeftModel.from_pretrained" in adapter
    assert "PeftModel.from_pretrained" not in merged


def test_model_card_embeds_eval_report() -> None:
    config = Config()
    report = "## Results\n\nROUGE-L: 0.42 (base 0.30)."
    card = build_model_card(config, repo_id="acme/cs", is_adapter=False, eval_report=report)
    assert "ROUGE-L: 0.42" in card


def test_model_card_cites_datasets() -> None:
    card = build_model_card(Config(), repo_id="acme/cs", is_adapter=False)
    assert "MTS-Dialog" in card
    assert "CC BY 4.0" in card
    assert "ACI-Bench" in card


def test_find_quantize_binary_missing(tmp_path: Path) -> None:
    assert _find_quantize_binary(tmp_path) is None


def test_find_quantize_binary_in_build_bin(tmp_path: Path) -> None:
    binary = tmp_path / "build" / "bin" / "llama-quantize"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    assert _find_quantize_binary(tmp_path) == binary


def test_run_export_missing_adapter_raises() -> None:
    with pytest.raises(FileNotFoundError):
        run_export(Config(), adapter_path="does/not/exist")
