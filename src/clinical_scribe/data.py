"""MTS-Dialog data pipeline: download -> schema-validate -> clean -> chat JSONL.

Phase 1 trains on MTS-Dialog at the *section* level: given a dialogue and a
target section, produce that section's note text. We use the dataset's OFFICIAL
splits and never mix test rows into train/val.

MTS-Dialog (CC BY 4.0): Ben Abacha et al., "An Empirical Study of Clinical Note
Generation from Doctor-Patient Encounters", EACL 2023.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

# MTS-Dialog normalized section-header codes -> human-readable names used in the
# user prompt. Unknown codes fall back to a title-cased form of the code.
SECTION_LABELS: dict[str, str] = {
    "GENHX": "History of Present Illness",
    "CC": "Chief Complaint",
    "PASTMEDICALHX": "Past Medical History",
    "PASTSURGICAL": "Past Surgical History",
    "FAM/SOCHX": "Family and Social History",
    "ALLERGY": "Allergies",
    "MEDICATIONS": "Medications",
    "ROS": "Review of Systems",
    "ASSESSMENT": "Assessment",
    "PLAN": "Plan",
    "DIAGNOSIS": "Diagnosis",
    "DISPOSITION": "Disposition",
    "EXAM": "Physical Examination",
    "EDCOURSE": "Emergency Department Course",
    "IMMUNIZATIONS": "Immunizations",
    "LABS": "Laboratory Results",
    "IMAGING": "Imaging",
    "PROCEDURES": "Procedures",
    "GYNHX": "Gynecologic History",
    "OTHER_HISTORY": "Other History",
}


def humanize_section(code: str) -> str:
    """Map an MTS-Dialog section code to a readable name (title-cased fallback)."""
    key = code.strip().upper()
    if key in SECTION_LABELS:
        return SECTION_LABELS[key]
    return code.strip().replace("_", " ").replace("/", " / ").title()


@dataclass
class SplitStats:
    """Summary statistics for one materialized split."""

    name: str
    num_rows: int
    num_dropped: int
    num_over_max: int
    prompt_token_min: int
    prompt_token_max: int
    prompt_token_mean: float
    prompt_token_p95: float


class SchemaError(ValueError):
    """Raised when a CSV does not match the expected MTS-Dialog schema."""


def _local_csv_path(config: Config, rel: str) -> Path:
    """Local cache path for a raw CSV given its repo-relative name."""
    return Path(config.data.raw_dir) / Path(rel).name


def ensure_csv(config: Config, rel: str) -> Path:
    """Return a local path to ``rel``, downloading from the repo if absent."""
    dest = _local_csv_path(config, rel)
    if dest.exists():
        logger.info("Using cached CSV: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{config.data.mts_dialog_repo.rstrip('/')}/{rel}"
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted GitHub raw URL)
    return dest


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV (handles quoted multi-line dialogue fields) into dict rows."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
    logger.info("Read %d rows from %s", len(rows), path.name)
    return rows


def validate_schema(rows: list[dict[str, str]], config: Config, source: str) -> None:
    """Ensure required columns exist; raise :class:`SchemaError` otherwise."""
    required = {
        config.data.id_column,
        config.data.section_header_column,
        config.data.section_text_column,
        config.data.dialogue_column,
    }
    if not rows:
        raise SchemaError(f"{source}: no rows found.")
    present = set(rows[0].keys())
    missing = required - present
    if missing:
        raise SchemaError(
            f"{source}: missing expected columns {sorted(missing)}. "
            f"Found columns: {sorted(present)}. The MTS-Dialog schema may have "
            "changed — update DataConfig column names."
        )
    logger.info("%s: schema OK (%d columns)", source, len(present))


def clean_text(text: str | None) -> str:
    """Normalize whitespace and strip; return '' for None."""
    if text is None:
        return ""
    # Normalize CRLF, trim each line both sides, strip outer whitespace.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in normalized.split("\n")]
    return "\n".join(lines).strip()


def build_example(row: dict[str, str], config: Config) -> dict[str, Any] | None:
    """Convert one raw row into a chat-formatted training example.

    Returns None if the row is unusable (empty dialogue or empty target),
    so callers can count drops.
    """
    d = config.data
    dialogue = clean_text(row.get(d.dialogue_column))
    target = clean_text(row.get(d.section_text_column))
    section_code = clean_text(row.get(d.section_header_column))

    if not dialogue or not target:
        return None

    section_name = humanize_section(section_code)
    system = config.prompts.render_system(d.na_placeholder)
    user = config.prompts.user_template.format(dialogue=dialogue, section=section_name)

    return {
        "id": row.get(d.id_column, ""),
        "section_code": section_code,
        "section_name": section_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
    }


def _count_prompt_tokens(examples: list[dict[str, Any]], config: Config) -> list[int]:
    """Token counts of the rendered system+user prompt per example.

    Uses the model tokenizer + chat template when transformers is available;
    otherwise falls back to a whitespace word count and logs that the estimate
    is approximate.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.model.base_model_id)
        counts: list[int] = []
        for ex in examples:
            prompt_msgs = ex["messages"][:2]  # system + user (exclude target)
            ids = tok.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
            counts.append(len(ids))
        return counts
    except Exception as exc:  # noqa: BLE001 — fall back gracefully off-GPU/offline
        logger.warning("Tokenizer unavailable (%s); using approximate word-count token stats.", exc)
        return [
            len((ex["messages"][0]["content"] + " " + ex["messages"][1]["content"]).split())
            for ex in examples
        ]


def _percentile(values: list[int], pct: float) -> float:
    """Return the ``pct`` percentile (0-100) of ``values`` via linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def materialize_split(
    rows: list[dict[str, str]], config: Config, name: str
) -> tuple[list[dict[str, Any]], SplitStats]:
    """Build examples for one split, compute token stats, and apply guardrails."""
    raw_examples = [build_example(r, config) for r in rows]
    examples = [e for e in raw_examples if e is not None]
    num_dropped = len(raw_examples) - len(examples)

    token_counts = _count_prompt_tokens(examples, config)
    over_max = [c for c in token_counts if c > config.data.max_prompt_tokens]

    if config.data.drop_over_max and over_max:
        kept = [
            (e, c)
            for e, c in zip(examples, token_counts, strict=True)
            if c <= config.data.max_prompt_tokens
        ]
        examples = [e for e, _ in kept]
        token_counts = [c for _, c in kept]

    if over_max and config.data.warn_truncation:
        logger.warning(
            "%s: %d/%d prompts exceed max_prompt_tokens=%d (max seen=%d). " "%s",
            name,
            len(over_max),
            len(raw_examples),
            config.data.max_prompt_tokens,
            max(token_counts) if token_counts else 0,
            "Dropped." if config.data.drop_over_max else "Will truncate at train time.",
        )

    stats = SplitStats(
        name=name,
        num_rows=len(examples),
        num_dropped=num_dropped,
        num_over_max=len(over_max),
        prompt_token_min=min(token_counts) if token_counts else 0,
        prompt_token_max=max(token_counts) if token_counts else 0,
        prompt_token_mean=round(statistics.mean(token_counts), 1) if token_counts else 0.0,
        prompt_token_p95=round(_percentile(token_counts, 95), 1),
    )
    logger.info(
        "%s: %d examples (%d dropped) | prompt tokens min/mean/p95/max = %d/%.1f/%.1f/%d",
        name,
        stats.num_rows,
        stats.num_dropped,
        stats.prompt_token_min,
        stats.prompt_token_mean,
        stats.prompt_token_p95,
        stats.prompt_token_max,
    )
    return examples, stats


def write_jsonl(examples: list[dict[str, Any]], path: Path) -> None:
    """Write examples to ``path`` as one JSON object per line (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples -> %s", len(examples), path)


def prepare_dataset(config: Config) -> dict[str, SplitStats]:
    """Run the full MTS-Dialog pipeline for all official splits.

    Writes ``{processed_dir}/{split}.jsonl`` and returns per-split stats. The
    official train/val/test splits are kept strictly separate (no leakage).
    """
    processed = Path(config.data.processed_dir)
    split_files = {
        "train": config.data.train_csv,
        "val": config.data.val_csv,
        "test1": config.data.test_csv_1,
        "test2": config.data.test_csv_2,
    }

    all_stats: dict[str, SplitStats] = {}
    for split_name, rel in split_files.items():
        csv_path = ensure_csv(config, rel)
        rows = read_csv_rows(csv_path)
        validate_schema(rows, config, source=f"{split_name} ({rel})")
        examples, stats = materialize_split(rows, config, split_name)
        write_jsonl(examples, processed / f"{split_name}.jsonl")
        all_stats[split_name] = stats

    logger.info("Dataset preparation complete: %s", {k: v.num_rows for k, v in all_stats.items()})
    return all_stats
