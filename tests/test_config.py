"""Tests for config schema, defaults, extends-merging, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from clinical_scribe.config import Config, load_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_valid(base_config: Config) -> None:
    assert base_config.model.base_model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert base_config.model.enable_thinking is False
    assert base_config.data.max_prompt_tokens < base_config.model.max_seq_length


def test_prompt_max_tokens_must_be_under_seq_length() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"model": {"max_seq_length": 512}, "data": {"max_prompt_tokens": 512}}
        )


def test_render_system_includes_disclaimer_and_na(base_config: Config) -> None:
    rendered = base_config.prompts.render_system(base_config.data.na_placeholder)
    assert "clinician review" in rendered.lower()
    assert base_config.data.na_placeholder in rendered


@pytest.mark.parametrize("name", ["base.yaml", "phase1_t4.yaml", "phase1_smoke.yaml"])
def test_shipped_configs_load(name: str) -> None:
    config = load_config(CONFIGS_DIR / name)
    assert isinstance(config, Config)


def test_extends_merging_overrides_parent() -> None:
    t4 = load_config(CONFIGS_DIR / "phase1_t4.yaml")
    # phase1_t4 overrides dtype/attn from base's "auto".
    assert t4.model.dtype == "fp16"
    assert t4.model.attn_implementation == "sdpa"


def test_smoke_config_sets_max_steps() -> None:
    smoke = load_config(CONFIGS_DIR / "phase1_smoke.yaml")
    assert smoke.train.max_steps == 2
    assert smoke.model.max_seq_length == 512


def test_overrides_applied_last() -> None:
    config = load_config(CONFIGS_DIR / "phase1_t4.yaml", overrides={"general": {"seed": 7}})
    assert config.general.seed == 7


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(CONFIGS_DIR / "does_not_exist.yaml")
