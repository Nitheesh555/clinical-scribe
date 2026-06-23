"""Shared pytest fixtures."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from clinical_scribe.config import Config


@pytest.fixture
def base_config() -> Config:
    """A default, valid Config (all pydantic defaults)."""
    return Config()


@pytest.fixture
def mts_csv(tmp_path: Path) -> Path:
    """Write a tiny MTS-Dialog-shaped CSV (incl. a multi-line quoted dialogue)."""
    path = tmp_path / "mini.csv"
    rows = [
        {
            "ID": "0",
            "section_header": "GENHX",
            "section_text": "Patient reports a 3-day history of productive cough.",
            "dialogue": "Doctor: What brings you in?\nPatient: I've had a cough for three days.",
        },
        {
            "ID": "1",
            "section_header": "MEDICATIONS",
            "section_text": "Lisinopril 10 mg daily.",
            "dialogue": "Doctor: Any medications?\nPatient: Just lisinopril.",
        },
        {
            # Unusable: empty target -> should be dropped by build_example.
            "ID": "2",
            "section_header": "ALLERGY",
            "section_text": "",
            "dialogue": "Doctor: Any allergies?\nPatient: No.",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ID", "section_header", "section_text", "dialogue"])
        writer.writeheader()
        writer.writerows(rows)
    return path
