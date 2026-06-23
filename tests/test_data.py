"""Tests for the MTS-Dialog data pipeline (offline; uses a local CSV fixture)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_scribe.config import Config
from clinical_scribe.data import (
    SchemaError,
    build_example,
    clean_text,
    humanize_section,
    materialize_split,
    read_csv_rows,
    validate_schema,
    write_jsonl,
)


def test_humanize_known_and_unknown() -> None:
    assert humanize_section("GENHX") == "History of Present Illness"
    assert humanize_section("CC") == "Chief Complaint"
    # Unknown code falls back to a title-cased form.
    assert humanize_section("WEIRD_CODE") == "Weird Code"


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("  hi \r\n there  \r\n") == "hi\nthere"
    assert clean_text(None) == ""


def test_validate_schema_detects_missing_columns(base_config: Config) -> None:
    with pytest.raises(SchemaError):
        validate_schema([{"ID": "0", "dialogue": "x"}], base_config, source="bad")


def test_build_example_structure(base_config: Config) -> None:
    row = {
        "ID": "0",
        "section_header": "GENHX",
        "section_text": "Three-day cough.",
        "dialogue": "Doctor: hi\nPatient: cough",
    }
    ex = build_example(row, base_config)
    assert ex is not None
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["section_name"] == "History of Present Illness"
    assert ex["messages"][2]["content"] == "Three-day cough."
    # System prompt carries the disclaimer; user prompt carries the dialogue.
    assert "clinician review" in ex["messages"][0]["content"].lower()
    assert "Doctor: hi" in ex["messages"][1]["content"]


def test_build_example_drops_empty_target(base_config: Config) -> None:
    row = {"ID": "9", "section_header": "ALLERGY", "section_text": "  ", "dialogue": "x"}
    assert build_example(row, base_config) is None


def test_materialize_split_counts_and_stats(base_config: Config, mts_csv: Path) -> None:
    rows = read_csv_rows(mts_csv)
    validate_schema(rows, base_config, source="fixture")
    examples, stats = materialize_split(rows, base_config, "train")
    # 3 rows in, 1 dropped (empty target) -> 2 usable.
    assert stats.num_rows == 2
    assert stats.num_dropped == 1
    assert stats.prompt_token_max >= stats.prompt_token_min >= 0


def test_csv_handles_multiline_dialogue(mts_csv: Path) -> None:
    rows = read_csv_rows(mts_csv)
    assert "\n" in rows[0]["dialogue"]  # quoted multi-line field preserved


def test_write_jsonl_roundtrip(tmp_path: Path, base_config: Config, mts_csv: Path) -> None:
    rows = read_csv_rows(mts_csv)
    examples, _ = materialize_split(rows, base_config, "train")
    out = tmp_path / "train.jsonl"
    write_jsonl(examples, out)
    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(loaded) == len(examples)
    assert loaded[0]["messages"][0]["role"] == "system"
