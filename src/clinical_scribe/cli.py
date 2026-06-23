"""Console entrypoints. Logic lives in the modules; these only parse args.

Each ``*_main`` is registered as a console script in pyproject and is also
callable from the thin wrappers in ``scripts/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .utils import log_versions, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=str, help="Path to a YAML config file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...).")
    return parser


def prepare_data_main(argv: list[str] | None = None) -> int:
    """Build JSONL datasets from MTS-Dialog per the config."""
    parser = _common_parser("Prepare MTS-Dialog JSONL datasets.")
    parser.add_argument(
        "--stats-out",
        type=str,
        default=None,
        help="Optional path to write per-split stats as JSON.",
    )
    args = parser.parse_args(argv)
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    config = load_config(args.config)
    set_seed(config.general.seed)
    log_versions()

    # Imported here so config/lint/CI don't require the data deps at import time.
    from .data import prepare_dataset

    stats = prepare_dataset(config)
    stats_dict = {name: asdict(s) for name, s in stats.items()}

    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")
        logger.info("Wrote stats -> %s", args.stats_out)

    print(json.dumps(stats_dict, indent=2))  # noqa: T201 — CLI summary to stdout
    return 0


def train_main(argv: list[str] | None = None) -> int:
    """Train QLoRA SFT (implemented in the train step)."""
    parser = _common_parser("Train Qwen3 QLoRA on the prepared dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    args = parser.parse_args(argv)
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    config = load_config(args.config)
    set_seed(config.general.seed)
    log_versions()

    from .train import run_training

    run_training(config, resume=args.resume)
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    """Evaluate base vs fine-tuned (implemented in the eval step)."""
    parser = _common_parser("Evaluate the model and write eval_report.md.")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter.")
    args = parser.parse_args(argv)
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    config = load_config(args.config)
    set_seed(config.general.seed)

    from .eval import run_evaluation

    run_evaluation(config, adapter_path=args.adapter)
    return 0


def export_main(argv: list[str] | None = None) -> int:
    """Merge, push to Hub, and export GGUF (implemented in the export step)."""
    parser = _common_parser("Export the trained model.")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter.")
    args = parser.parse_args(argv)
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    config = load_config(args.config)

    from .export import run_export

    run_export(config, adapter_path=args.adapter)
    return 0
